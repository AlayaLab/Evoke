

from .sp_runtime import is_sp_enabled


_SP_PIN_SENTINEL = -0x5350


_PINNED = {}


def _zero3_params(module):


    params = [p for p in module.parameters() if hasattr(p, "ds_id")]
    return sorted(set(params), key=lambda p: p.ds_id)


def pin_module_params(module):


    if not is_sp_enabled():
        return False
    if id(module) in _PINNED:
        return True
    params = _zero3_params(module)
    if not params:
        return False
    from .sp_runtime import sp_diag
    sp_diag(f"pin start (n={len(params)})")

    params[0].all_gather(param_list=params)
    sp_diag("pin all_gather done")

    for p in params:
        p.ds_active_sub_modules.add(_SP_PIN_SENTINEL)
    _PINNED[id(module)] = params
    return True


def _release_param_list(params):


    if not params:
        return
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    for p in params:
        p.ds_active_sub_modules.discard(_SP_PIN_SENTINEL)
    still_avail = [p for p in params if p.ds_status == ZeroParamStatus.AVAILABLE]
    if still_avail:

        still_avail[0].partition(param_list=still_avail, has_been_updated=False)


def unpin_module_params(module):

    _release_param_list(_PINNED.pop(id(module), None))


def unpin_all():


    for mid in list(_PINNED.keys()):
        _release_param_list(_PINNED.pop(mid, None))


def any_pinned():
    return len(_PINNED) > 0


"""
Root cause recap (measured on two separate runs):
under 2D (sp<world) the critic ZeRO-3 parameter all-gather (forward) and grad reduce-scatter (backward)
land on the **WORLD** group by default, whose rank coverage overlaps the SP-subgroup all_to_all -> circular-wait deadlock. Approach (B) pin only suppresses
the per-parameter all-gather of the forward; the WORLD reduce-scatter of the backward still deadlocks.

Approach (A) mpu, the real fix (verified against the deepspeed 0.14.5 source, see the subagent report):
- pass an mpu to the critic engine so that ZeRO-3's **data-parallel group = the DP-stride-G group**
  ({r: r%G==j}, e.g. 8 GPUs with G=2 -> {0,2,4,6}/{1,3,5,7}). That group **excludes the SP-peer ranks**:
    - the forward parameter all-gather lands on the DP group -> rank0's all-gather only waits for {0,2,4,6}, not for rank1 which is
      blocked in the SP all_to_all -> no cycle -> the forward deadlock is solved (pin becomes redundant, already a no-op).
    - the backward grad reduce-scatter lands on the DP group (stage3.py reduce_scatter_coalesced uses
      self.dp_process_group) -> likewise it does not interleave with the SP all_to_all -> the backward deadlock is solved.
  The sharding degree drops from world to world/G (parameters cost G times more: dual-expert 56G/(world/G)).
- Key correctness point (§14 derivation): the DP-stride-G reduce-scatter only sums inside the DP group -> the partial grad of the SP-peer
  rank (which handles the other half of the frame shards of the same clip) lands in **another** DP group -> the parameter replicas of the two
  DP groups diverge. After the reduce-scatter and before optimizer.step, the **already-reduced grad shards** must be all-reduced (SUM) inside the SP group:
  the geometry lines up -- SP peers (2k,2k+1) both have dp_rank k inside their own DP group -> they hold the **same shard index** ->
  grad_partitions_flat_buffer is element-wise aligned -> a single SUM all-reduce merges the partials of the two frame shards.
  The normalization works out exactly: reduce_scatter already divided by dp_world_size (=world/G), so after the SP-SUM it is (1/(world/G))*sum_clip full_grad
  = the correct batch-mean (effective batch = world/G clips) -> **no xG loss-scale needed** (route-B's xG is void).
  Precondition: the mpu does not implement get_sequence_parallel_world_size() -> deepspeed's self.sequence_parallel_size==1
  -> the reduce_scatter path adds no hidden SP scaling (the one at stage3.py is the all-reduce path, not taken when reduce_scatter=true).
- get_model_parallel_group() = the SP group: used only for grad-**norm** deduplication (stage3.py/1762), it does not reduce the gradients
  themselves (verified in subagent Q4) -> no double correction; after the SP-SUM the two replica shards are identical -> the norm is counted once, correctly.
- global groups.mpu pollution: engine.py `groups.mpu = self.mpu` is **global** and is only set, never reset ->
  groups.mpu must be **saved/restored** around the critic prepare, otherwise a evoke-critic prepared later (mpu=None) would wrongly read
  the critic's stride-G group. During training stage3 uses the captured self.dp_process_group (not the global) -> restoring is safe.
"""


class CriticMPU:


    def __init__(self, dp_group, dp_ranks, sp_group, sp_ranks):
        import torch.distributed as dist
        self._dp_group = dp_group
        self._dp_ranks = list(dp_ranks)
        self._sp_group = sp_group
        self._sp_ranks = list(sp_ranks)
        self._dp_world_size = len(self._dp_ranks)
        self._sp_world_size = len(self._sp_ranks)
        self._dp_rank = dist.get_rank(group=dp_group)
        self._mp_rank = dist.get_rank(group=sp_group)


    def get_data_parallel_group(self):
        return self._dp_group

    def get_data_parallel_world_size(self):
        return self._dp_world_size

    def get_data_parallel_rank(self):
        return self._dp_rank


    def get_model_parallel_group(self):
        return self._sp_group

    def get_model_parallel_world_size(self):
        return self._sp_world_size

    def get_model_parallel_rank(self):
        return self._mp_rank


def build_critic_mpu():

    from .sp_runtime import is_2d_sp, get_dp_group, get_dp_ranks, get_sp_group, get_sp_size
    import torch.distributed as dist
    if not is_2d_sp():
        return None
    dp_group = get_dp_group()
    dp_ranks = get_dp_ranks()
    sp_group = get_sp_group()
    assert dp_group is not None and sp_group is not None, "[mpu] 2D SP requires sp_runtime to have built the DP+SP groups"

    g = get_sp_size()
    start = (dist.get_rank() // g) * g
    sp_ranks = list(range(start, start + g))
    return CriticMPU(dp_group, dp_ranks, sp_group, sp_ranks)


def _iter_grad_shard_buffers(engine):


    opt = getattr(engine, "optimizer", None)
    if opt is None:
        return
    flat = getattr(opt, "grad_partitions_flat_buffer", None)
    if flat is not None and flat.numel() > 0:
        yield flat
        return

    fp32_flat = getattr(opt, "fp32_partitioned_groups_flat", None)
    if fp32_flat is not None:
        for g in fp32_flat:
            if getattr(g, "grad", None) is not None:
                yield g.grad


def sp_allreduce_grad_shards(engine):


    from .sp_runtime import is_2d_sp, get_sp_group, sp_diag
    if not is_2d_sp():
        return
    import torch.distributed as dist
    sp_group = get_sp_group()
    if sp_group is None:
        return
    n = 0
    for buf in _iter_grad_shard_buffers(engine):
        sp_diag(f"sp-sum all_reduce start (numel={buf.numel()})")
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=sp_group)
        sp_diag("sp-sum all_reduce done")
        n += 1
    if n == 0:

        print("[mpu][WARN] sp_allreduce_grad_shards: no grad shard buffer found (SP-SUM did NOT take effect!) "
              "-- check the deepspeed version / whether the critic is ZeRO-3 / whether optimizer offload is enabled.", flush=True)


def wrap_critic_engine_step(engine):


    from .sp_runtime import is_2d_sp
    if not is_2d_sp():
        return False
    if getattr(engine, "_sp_step_wrapped", False):
        return True
    _orig_step = engine.step

    def _sp_step(*args, **kwargs):
        from .sp_runtime import sp_diag
        sp_diag("engine.step ENTER (pre sp-sum)")
        sp_allreduce_grad_shards(engine)
        sp_diag("engine.step orig-step begin")
        r = _orig_step(*args, **kwargs)
        sp_diag("engine.step DONE")
        return r

    engine.step = _sp_step
    engine._sp_step_wrapped = True
    return True
