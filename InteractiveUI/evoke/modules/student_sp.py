

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

import torch
import torch.distributed as dist
from torch.autograd import Function


_inited: bool = False
_cp_enabled: bool = False
_G: int = 1
_G_u: int = 1
_G_p: int = 1
_sp_rank: int = 0
_u_rank: int = 0
_p_rank: int = 0
_u_group: Optional[dist.ProcessGroup] = None
_diag: bool = False
_skip_first_chunk: bool = False
_seq_counter: int = 0
_trace: bool = os.environ.get("SF_STUSP_TRACE", "0") == "1"
_profile_collective = ContextVar("student_sp_profile_collective", default=None)
_fused_qkv = ContextVar("student_sp_fused_qkv", default=os.environ.get("EVOKE_SP_FUSED_QKV", "0") == "1")
_hoist_timestep_cast = ContextVar("student_sp_hoist_timestep_cast", default=False)
_timestep_cast_observations = ContextVar("student_sp_timestep_cast_observations", default=None)


def prepare_timestep_projection(value: torch.Tensor) -> torch.Tensor:


    enabled = _hoist_timestep_cast.get() and not torch.is_grad_enabled()
    result = value.float() if enabled else value
    observations = _timestep_cast_observations.get()
    if observations is not None:
        observations.append({'enabled': enabled, 'shape': list(value.shape),
                             'inputDtype': str(value.dtype), 'outputDtype': str(result.dtype),
                             'inputStride': list(value.stride()), 'outputStride': list(result.stride()),
                             'converted': result is not value})
    return result


def _run_collective(op, tensor, call):

    profiler = _profile_collective.get()
    if profiler is None:
        return call()
    return profiler(op, tensor, call)


class UlyssesCtx:


    __slots__ = ("group", "size", "rank")

    def __init__(self, group, size: int, rank: int):
        self.group = group
        self.size = int(size)
        self.rank = int(rank)


class ShardPlan:


    __slots__ = ("ctx", "S_real", "hist_global", "noise_global", "lh", "ln", "L", "_vidx")

    def __init__(self, ctx: UlyssesCtx, S_real: int, hist_global: int):
        g = ctx.size
        self.ctx = ctx
        self.S_real = int(S_real)
        self.hist_global = int(hist_global)
        self.noise_global = int(S_real) - int(hist_global)
        assert self.noise_global >= g, (
        f"[STU-SP] noise token count {self.noise_global} < G_u={g}: the dual-region split requires at least 1 noise token per rank")
        assert self.hist_global >= 0
        self.lh = (self.hist_global + g - 1) // g
        self.ln = (self.noise_global + g - 1) // g
        self.L = self.lh + self.ln
        self._vidx = None

    @property
    def hist_local(self) -> int:

        return self.lh

    @property
    def has_pad(self) -> bool:
        return (self.lh * self.ctx.size != self.hist_global) or (self.ln * self.ctx.size != self.noise_global)

    def valid_index(self, device) -> torch.Tensor:


        if self._vidx is None:
            g, lh, ln, L = self.ctx.size, self.lh, self.ln, self.L
            keep = []
            for u in range(g):
                for j in range(lh):
                    if u * lh + j < self.hist_global:
                        keep.append(u * L + j)
                for j in range(ln):
                    if u * ln + j < self.noise_global:
                        keep.append(u * L + lh + j)
            self._vidx = torch.tensor(keep, dtype=torch.long)
        return self._vidx.to(device, non_blocking=True)


def init_student_sp(
    sp_world_size: int,
    chunk_parallel: bool,
    ulysses_size: int = 1,
    diag: bool = False,
    skip_first_chunk: bool = False,
):


    global _inited, _cp_enabled, _G, _G_u, _G_p, _sp_rank, _u_rank, _p_rank
    global _u_group, _diag, _skip_first_chunk, _seq_counter

    import datetime

    _cp_enabled = bool(chunk_parallel)
    _G_u = int(ulysses_size)
    _diag = bool(diag)
    _skip_first_chunk = bool(skip_first_chunk)
    _seq_counter = 0
    _u_group = None

    if not _cp_enabled and _G_u <= 1:

        _G, _G_p, _G_u = 1, 1, 1
        _sp_rank = _u_rank = _p_rank = 0
        _inited = True
        return

    assert dist.is_available() and dist.is_initialized(), "[STU-SP] requires an already initialized torch.distributed"
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    _G = int(sp_world_size)
    assert _G >= 1 and world_size % _G == 0, f"[STU-SP] world_size={world_size} is not divisible by G={_G}"
    assert _G % _G_u == 0, f"[STU-SP] G={_G} is not divisible by G_u={_G_u}"
    _G_p = _G // _G_u
    _sp_rank = rank % _G
    _u_rank = _sp_rank % _G_u
    _p_rank = _sp_rank // _G_u

    if _G_u > 1:
        _timeout = datetime.timedelta(milliseconds=int(os.environ.get("NCCL_TIMEOUT", "1800000")))
        if _G_u == world_size:

            _u_group = dist.group.WORLD
        else:

            for base in range(0, world_size, _G):
                for p in range(_G_p):
                    ranks = [base + p * _G_u + u for u in range(_G_u)]
                    grp = dist.new_group(ranks, timeout=_timeout)
                    if rank in ranks:
                        _u_group = grp
            assert _u_group is not None, f"[STU-SP] rank={rank} did not land in any U-subgroup"

    _inited = True
    if rank % _G == 0:
        print(
            f"[STU-SP] init: world={world_size} G={_G} -> G_p={_G_p} x G_u={_G_u}; "
            f"chunk_parallel={_cp_enabled} ulysses={_G_u > 1} diag={_diag} "
            f"skip_first_chunk={_skip_first_chunk} (rank {rank}: p_rank={_p_rank} u_rank={_u_rank})",
            flush=True,
        )


def is_cp_enabled() -> bool:
    return _cp_enabled


def is_ulysses_enabled() -> bool:
    return _G_u > 1


def is_any_enabled() -> bool:
    return _cp_enabled or _G_u > 1


def get_G() -> int:
    return _G


def get_G_p() -> int:
    return _G_p


def get_G_u() -> int:
    return _G_u


def get_p_rank() -> int:
    return _p_rank


def get_u_rank() -> int:
    return _u_rank


def get_u_group():
    return _u_group


def is_diag() -> bool:
    return _diag


def loss_scale() -> int:


    return _G if is_any_enabled() else 1


def redundant_grad_div() -> int:


    return _G_u if is_ulysses_enabled() else 1


def make_ulysses_ctx() -> Optional[UlyssesCtx]:

    if _G_u <= 1 or _u_group is None:
        return None
    return UlyssesCtx(_u_group, _G_u, _u_rank)


def cp_owns(k: int) -> bool:


    if not _cp_enabled:
        return True
    if _skip_first_chunk and k == 0:
        return False
    return (int(k) % _G_p) == _p_rank


def set_skip_first_chunk(v: bool) -> None:


    global _skip_first_chunk
    _skip_first_chunk = bool(v)


def get_skip_first_chunk() -> bool:
    return _skip_first_chunk


def _diag_device():

    return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")


def bump_seq(n: int = 1, op: str = "", shape=None):


    global _seq_counter
    _seq_counter += int(n)
    if _trace:
        import threading
        try:
            r = dist.get_rank()
        except Exception:
            r = -1
        print(f"[STU-SP-TRACE r{r} u{_u_rank} #{_seq_counter} {op} {tuple(shape) if shape is not None else ''} "
              f"thr={threading.current_thread().name}", flush=True)


def get_seq() -> int:
    return _seq_counter


def check_seq_in_group(tag: str = ""):

    if _u_group is None:
        return
    t = torch.tensor([_seq_counter, -_seq_counter], dtype=torch.long, device=_diag_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=_u_group)
    hi, lo = int(t[0].item()), -int(t[1].item())
    assert hi == lo == _seq_counter, (
        f"[STU-SP SEQ{(' ' + tag) if tag else ''}] collective sequence numbers disagree inside the U-subgroup: "
        f"local={_seq_counter} min={lo} max={hi} => forward graphs are not isomorphic within the group (recompute timing / branch misalignment), will deadlock or compute wrong gradients"
    )


_selfcheck: bool = os.environ.get("SF_STUSP_SELFCHECK", "0") == "1"
_selfcheck_done: bool = False


def selfcheck_pending() -> bool:

    return _selfcheck and not _selfcheck_done


def selfcheck_report(got: torch.Tensor, ref: torch.Tensor, plan, tag: str = ""):


    global _selfcheck_done
    _selfcheck_done = True
    d = (got.detach().float() - ref.detach().float()).abs()
    scale = ref.detach().float().abs().max().clamp_min(1e-12)
    b = plan.hist_global
    print(f"[STU-SP SELFCHECK {tag}] single forward, sharded vs unsharded: max|delta|={d.max().item():.3e} "
          f"rel={d.max().item() / scale.item():.3e} (history region {d[:, :b].max().item():.3e} / "
          f"noise region {d[:, b:].max().item():.3e}); dtype={ref.dtype} "
          f"lh={plan.lh} ln={plan.ln} L={plan.L} pad={plan.has_pad} "
          f"=> {'bf16 noise magnitude, implementation correct' if d.max().item() / scale.item() < 3e-2 else '!! magnitude too large, implementation has a bug'}",
          flush=True)


def selfcheck_grad(run_fn, full_inputs, shard_inputs, plan, params, tag: str = "",
                   depths=(1, 5, 10, 20), n_layers: int = 40):


    global _selfcheck_done
    _selfcheck_done = True

    def _one(upto):


        with torch.enable_grad():
            ref_out = run_fn(*(t.detach() for t in full_inputs), None, upto)


            _g = torch.Generator(device="cpu").manual_seed(20260725)
            cot = torch.randn(ref_out.shape, generator=_g, dtype=torch.float32).to(ref_out.device)
            assert_same_in_group(int(cot.sum().mul(1e6).item()), "cotangent fingerprint", plan.ctx.group)
            g_ref = torch.autograd.grad((ref_out.float() * cot.float()).sum(), params,
                                        allow_unused=True, retain_graph=False)
            fwd_ref = ref_out.detach().float()
            sh_out = run_fn(*(t.detach() for t in shard_inputs), plan, upto)
            g_sh = torch.autograd.grad((sh_out.float() * cot.float()).sum(), params,
                                       allow_unused=True, retain_graph=False)
            fwd_rel = ((sh_out.detach().float() - fwd_ref).norm()
                       / fwd_ref.norm().clamp_min(1e-30)).item()
        n_ok = n_bad = n_none = n_tiny = 0
        worst = (0.0, "", 0.0, 0.0)
        dot = nr = ns = 0.0
        for (name, _), a, b in zip(params_named(params), g_ref, g_sh):
            if a is None or b is None:
                n_none += 1
                continue
            bs = b.detach().float().clone()
            dist.all_reduce(bs, group=plan.ctx.group)
            af = a.detach().float()
            na, dn = af.norm().item(), (bs - af).norm().item()
            rel = dn / max(na, 1e-30)
            dot += float((af * bs).sum()); nr += float((af * af).sum()); ns += float((bs * bs).sum())

            if na < 1e-6:
                n_tiny += 1
            elif rel <= 3e-2:
                n_ok += 1
            else:
                n_bad += 1
            if rel > worst[0] and na >= 1e-6:
                worst = (rel, name, na, dn)
        cos = dot / max((nr ** 0.5) * (ns ** 0.5), 1e-30)
        print(f"[STU-SP SELFCHECK-GRAD {tag} depth={upto:>3}] fwd_rel={fwd_rel:.3e} | "
              f"ok={n_ok} bad={n_bad} tiny(|ref|<1e-6)={n_tiny} none={n_none} cos={cos:.8f} "
              f"worst_rel={worst[0]:.3e}@{worst[1]} (|ref|={worst[2]:.3e} |delta|={worst[3]:.3e})", flush=True)
        return cos, n_bad


    for d in depths:
        if d <= n_layers:
            _one(d)
    cos, n_bad = _one(n_layers)
    verdict = "gradients equivalent (within bf16 precision)" if (n_bad == 0 and cos > 0.999) else "!! not equivalent (read the depth sweep to characterize it)"
    print(f"[STU-SP SELFCHECK-GRAD {tag}] verdict: {verdict}", flush=True)


_PARAM_NAMES = []


def set_param_names(named):
    global _PARAM_NAMES
    _PARAM_NAMES = list(named)


def params_named(params):
    if len(_PARAM_NAMES) == len(params):
        return _PARAM_NAMES
    return [(f"p{i}", p) for i, p in enumerate(params)]


def check_ipg_bucket_order(engine, tag: str = "") -> int:


    import zlib

    opt = getattr(engine, "optimizer", None)
    bucket = getattr(opt, "params_in_ipg_bucket", None)
    if bucket is None:
        print("WARN: optimizer has no params_in_ipg_bucket (not ZeRO-1/2?), skipping the bucket-order check", flush=True)
        return -1

    seq = [int(e[2]) for e in bucket]
    h = zlib.crc32((",".join(map(str, seq))).encode()) & 0x7FFFFFFF
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([h, -h, len(seq), -len(seq)], dtype=torch.long, device=_diag_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        hi, lo, nhi, nlo = int(t[0]), -int(t[1]), int(t[2]), -int(t[3])
        assert hi == lo == h and nhi == nlo == len(seq), (
            f"[STU-SP §8-(6){(' ' + tag) if tag else ''}] param order inside the IPG bucket **differs across ranks**: "
            f"local hash={h} (min={lo} max={hi}), local len={len(seq)} (min={nlo} max={nhi}) => "
            f"average_tensor will add the gradients of different params together (silent scrambling). for the fix see "
            f"the 'gradient-order anchor' in this function's docstring.")
    return h


def assert_same_in_group(value: int, name: str, group=None):

    grp = group if group is not None else _u_group
    if grp is None:
        return
    v = int(value)
    t = torch.tensor([v, -v], dtype=torch.long, device=_diag_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=grp)
    hi, lo = int(t[0].item()), -int(t[1].item())
    assert hi == lo == v, f"[STU-SP] {name} inconsistent within the group: local={v} min={lo} max={hi}"


def _seq_to_head(x: torch.Tensor, group, gu: int) -> torch.Tensor:

    B, s, H, D = x.shape
    hl = H // gu
    t = x.view(B, s, gu, hl, D).permute(2, 0, 1, 3, 4).contiguous()
    o = torch.empty_like(t)
    _run_collective("a2a_s2h", t, lambda: dist.all_to_all_single(o, t, group=group))
    bump_seq(op="a2a_s2h", shape=x.shape)
    return o.permute(1, 0, 2, 3, 4).reshape(B, gu * s, hl, D)


def _head_to_seq(y: torch.Tensor, group, gu: int) -> torch.Tensor:

    B, S, hl, D = y.shape
    s = S // gu
    t = y.view(B, gu, s, hl, D).permute(1, 0, 2, 3, 4).contiguous()
    o = torch.empty_like(t)
    _run_collective("a2a_h2s", t, lambda: dist.all_to_all_single(o, t, group=group))
    bump_seq(op="a2a_h2s", shape=y.shape)
    return o.permute(1, 2, 0, 3, 4).reshape(B, s, gu * hl, D)


class _A2ASeqToHead(Function):
    @staticmethod
    def forward(ctx, x, group, gu):
        ctx.group, ctx.gu = group, gu
        return _seq_to_head(x, group, gu)

    @staticmethod
    def backward(ctx, grad):
        return _head_to_seq(grad.contiguous(), ctx.group, ctx.gu), None, None


class _A2AHeadToSeq(Function):
    @staticmethod
    def forward(ctx, x, group, gu):
        ctx.group, ctx.gu = group, gu
        return _head_to_seq(x, group, gu)

    @staticmethod
    def backward(ctx, grad):
        return _seq_to_head(grad.contiguous(), ctx.group, ctx.gu), None, None


def a2a_seq_to_head(x: torch.Tensor, ctx: UlyssesCtx) -> torch.Tensor:
    return _A2ASeqToHead.apply(x, ctx.group, ctx.size)


def a2a_qkv_to_head(query, key, value, ctx: UlyssesCtx):


    if not _fused_qkv.get() or torch.is_grad_enabled():
        return tuple(a2a_seq_to_head(x.contiguous(), ctx) for x in (query, key, value))
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Fused Ulysses Q/K/V must have identical shapes")
    B, s, H, D = query.shape
    gu = ctx.size
    if H % gu:
        raise ValueError("Ulysses head count must be divisible by group size")
    hl = H // gu


    packed = torch.stack([x.reshape(B, s, gu, hl, D).permute(2, 0, 1, 3, 4)
                          for x in (query, key, value)], dim=1)
    received = torch.empty_like(packed)
    _run_collective("a2a_qkv", packed,
                    lambda: dist.all_to_all_single(received, packed, group=ctx.group))
    bump_seq(op="a2a_qkv", shape=packed.shape)
    return tuple(received[:, i].permute(1, 0, 2, 3, 4).reshape(B, gu * s, hl, D).contiguous()
                 for i in range(3))


def a2a_head_to_seq(x: torch.Tensor, ctx: UlyssesCtx) -> torch.Tensor:
    return _A2AHeadToSeq.apply(x, ctx.group, ctx.size)


def _all_gather_tokens(x_local: torch.Tensor, group, gu: int) -> torch.Tensor:

    xt = x_local.transpose(0, 1).contiguous()
    out = torch.empty((gu * xt.shape[0],) + tuple(xt.shape[1:]), dtype=xt.dtype, device=xt.device)
    _run_collective("allgather", xt, lambda: dist.all_gather_into_tensor(out, xt, group=group))
    bump_seq(op="allgather", shape=x_local.shape)
    return out.transpose(0, 1).contiguous()


def _pad_tokens(x: torch.Tensor, n: int) -> torch.Tensor:


    cur = x.shape[1]
    if cur == n:
        return x
    assert cur < n, f"[STU-SP] _pad_tokens: current length {cur} > target {n}"
    return torch.cat([x, x.new_zeros((x.shape[0], n - cur) + tuple(x.shape[2:]))], dim=1)


def _split_local(full: torch.Tensor, plan: ShardPlan) -> torch.Tensor:

    b, u, lh, ln = plan.hist_global, plan.ctx.rank, plan.lh, plan.ln
    h = _pad_tokens(full[:, :b], plan.lh * plan.ctx.size)[:, u * lh : (u + 1) * lh]
    n = _pad_tokens(full[:, b:], plan.ln * plan.ctx.size)[:, u * ln : (u + 1) * ln]
    return torch.cat([h, n], dim=1).contiguous()


def _merge_global(gathered: torch.Tensor, plan: ShardPlan) -> torch.Tensor:

    g, lh, ln, L = plan.ctx.size, plan.lh, plan.ln, plan.L
    hs = [gathered[:, u * L : u * L + lh] for u in range(g)]
    ns = [gathered[:, u * L + lh : (u + 1) * L] for u in range(g)]
    return torch.cat([torch.cat(hs, dim=1)[:, : plan.hist_global],
                      torch.cat(ns, dim=1)[:, : plan.noise_global]], dim=1)


class _ScatterTokens(Function):


    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        return _split_local(x, plan)

    @staticmethod
    def backward(ctx, grad):
        p = ctx.plan
        full = _merge_global(_all_gather_tokens(grad.contiguous(), p.ctx.group, p.ctx.size), p)
        return (full / p.ctx.size) if p.ctx.size > 1 else full, None


class _ScaleGrad(Function):


    @staticmethod
    def forward(ctx, x, k):
        ctx.k = float(k)
        return x

    @staticmethod
    def backward(ctx, grad):
        return (grad * ctx.k) if ctx.k != 1.0 else grad, None


def scale_grad(x: torch.Tensor, k: float) -> torch.Tensor:
    return _ScaleGrad.apply(x, k) if k != 1.0 else x


class _GatherTokens(Function):


    @staticmethod
    def forward(ctx, x, plan):
        ctx.plan = plan
        return _merge_global(_all_gather_tokens(x, plan.ctx.group, plan.ctx.size), plan)

    @staticmethod
    def backward(ctx, grad):
        return _split_local(grad.contiguous(), ctx.plan), None


def scatter_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    return _ScatterTokens.apply(x, plan)


def gather_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:
    return _GatherTokens.apply(x, plan)


def drop_pad_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:


    return x.index_select(1, plan.valid_index(x.device))


def restore_pad_tokens(x: torch.Tensor, plan: ShardPlan) -> torch.Tensor:

    idx = plan.valid_index(x.device)
    out = x.new_zeros((x.shape[0], plan.L * plan.ctx.size) + tuple(x.shape[2:]))
    return out.index_copy(1, idx, x)


def phase_barrier():


    if _u_group is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def install_redundant_grad_hooks(model, log: bool = True) -> int:


    raise AssertionError(
        "[STU-SP] install_redundant_grad_hooks is deprecated: /G_u is now done in _ScatterTokens.backward and in the tail's "
        "_ScaleGrad pair (acting only on the sharded path, so text_embedder and GEO-REG are naturally unharmed). see this function's docstring.")
