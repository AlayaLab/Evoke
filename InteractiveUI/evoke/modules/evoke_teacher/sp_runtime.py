

import os
import contextlib
import torch
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional


_sp_group: Optional[dist.ProcessGroup] = None
_sp_size: int = 1
_sp_rank: int = 0
_world_size: int = 1


_dp_group: Optional[dist.ProcessGroup] = None
_dp_ranks: Optional[list] = None


def init_sequence_parallel(sp_size: int):


    global _sp_group, _sp_size, _sp_rank, _world_size, _dp_group, _dp_ranks
    import os, datetime
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, f"world_size={world_size} not divisible by sp_size={sp_size}"

    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _world_size = world_size
    _dp_group = None
    _dp_ranks = None

    _nccl_timeout_ms = int(os.environ.get('NCCL_TIMEOUT', '1800000'))
    _sp_timeout = datetime.timedelta(milliseconds=_nccl_timeout_ms)

    if sp_size == world_size:


        _sp_group = dist.group.WORLD
        print(f"[SP] init: world_size={world_size}, sp_size={sp_size}, "
              f"sp_rank={_sp_rank}, dp_size=1 (reusing WORLD group), "
              f"timeout={_nccl_timeout_ms/1000:.0f}s (set via WORLD)")
    else:

        for i in range(0, world_size, sp_size):
            ranks = list(range(i, i + sp_size))
            group = dist.new_group(ranks, timeout=_sp_timeout)
            if rank in ranks:
                _sp_group = group

        for j in range(sp_size):
            dp_ranks = list(range(j, world_size, sp_size))
            dp_group = dist.new_group(dp_ranks, timeout=_sp_timeout)
            if rank in dp_ranks:
                _dp_group = dp_group
                _dp_ranks = dp_ranks
        print(f"[SP] init: world_size={world_size}, sp_size={sp_size}, "
              f"sp_rank={_sp_rank}, dp_size={world_size // sp_size}, "
              f"dp_group(stride-{sp_size})={_dp_ranks}, "
              f"timeout={_nccl_timeout_ms/1000:.0f}s")


def sp_diag(msg: str):


    if _sp_size <= 1 or os.environ.get("SP_DIAG", "0") != "1":
        return
    try:
        r = dist.get_rank()
    except Exception:
        r = -1
    print(f"[SP-DIAG r{r}] {msg}", flush=True)


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return _sp_group

def get_sp_size() -> int:
    return _sp_size

def get_sp_rank() -> int:
    return _sp_rank

def is_sp_enabled() -> bool:
    return _sp_size > 1

def get_world_size() -> int:
    return _world_size

def is_2d_sp() -> bool:


    return 1 < _sp_size < _world_size

def get_dp_group() -> Optional[dist.ProcessGroup]:

    return _dp_group

def get_dp_ranks() -> Optional[list]:
    return _dp_ranks


class _ScatterFrames(Function):


    @staticmethod
    def forward(ctx, x, per_frame_tokens, group, sp_size, sp_rank):
        ctx.per_frame_tokens = per_frame_tokens
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.total_seq_len = x.shape[1]

        B, S, D = x.shape
        num_frames = S // per_frame_tokens
        frames_per_rank = (num_frames + sp_size - 1) // sp_size
        ctx.frames_per_rank = frames_per_rank
        ctx.num_frames = num_frames


        padded_frames = frames_per_rank * sp_size
        if padded_frames > num_frames:
            pad_tokens = (padded_frames - num_frames) * per_frame_tokens
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_tokens))

        start = sp_rank * frames_per_rank * per_frame_tokens
        end = start + frames_per_rank * per_frame_tokens
        return x[:, start:end].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        gathered = [torch.zeros_like(grad_output) for _ in range(ctx.sp_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        grad_full = torch.cat(gathered, dim=1)
        grad_full = grad_full[:, :ctx.total_seq_len]
        return grad_full, None, None, None, None


class _GatherFrames(Function):


    @staticmethod
    def forward(ctx, x, total_seq_len, group, sp_size, sp_rank):
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.group = group
        ctx.local_len = x.shape[1]
        ctx.total_seq_len = total_seq_len

        gathered = [torch.zeros_like(x) for _ in range(sp_size)]
        dist.all_gather(gathered, x.contiguous(), group=group)
        result = torch.cat(gathered, dim=1)
        return result[:, :total_seq_len]

    @staticmethod
    def backward(ctx, grad_output):
        padded_len = ctx.local_len * ctx.sp_size
        if grad_output.shape[1] < padded_len:
            grad_output = torch.nn.functional.pad(
                grad_output, (0, 0, 0, padded_len - grad_output.shape[1]))
        start = ctx.sp_rank * ctx.local_len
        end = start + ctx.local_len
        return grad_output[:, start:end].contiguous(), None, None, None, None


class _AllReduceSum(Function):


    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


def scatter_frames(x: torch.Tensor, per_frame_tokens: int) -> torch.Tensor:

    if not is_sp_enabled():
        return x
    return _ScatterFrames.apply(x, per_frame_tokens, _sp_group, _sp_size, _sp_rank)


def gather_frames(x: torch.Tensor, total_seq_len: int) -> torch.Tensor:

    if not is_sp_enabled():
        return x
    return _GatherFrames.apply(x, total_seq_len, _sp_group, _sp_size, _sp_rank)


def allreduce_sum(x: torch.Tensor) -> torch.Tensor:

    if not is_sp_enabled():
        return x
    return _AllReduceSum.apply(x, _sp_group)


def allgather_frames_no_grad(x: torch.Tensor) -> torch.Tensor:

    if not is_sp_enabled():
        return x
    gathered = [torch.zeros_like(x) for _ in range(_sp_size)]
    dist.all_gather(gathered, x.contiguous(), group=_sp_group)
    return torch.cat(gathered, dim=1)


def broadcast_tensor(x: torch.Tensor, src_rank: int = 0) -> torch.Tensor:

    if not is_sp_enabled():
        return x

    global_rank = dist.get_rank()
    sp_group_start = (global_rank // _sp_size) * _sp_size
    src_global = sp_group_start + src_rank
    dist.broadcast(x, src=src_global, group=_sp_group)
    return x


def get_sp_frame_info(num_frames_total: int):


    if not is_sp_enabled():
        return num_frames_total, 0, num_frames_total, num_frames_total

    frames_per_rank = (num_frames_total + _sp_size - 1) // _sp_size
    f_start = _sp_rank * frames_per_rank
    f_end = min(f_start + frames_per_rank, num_frames_total)
    f_local = f_end - f_start
    return frames_per_rank, f_start, f_end, f_local


def get_sp_frame_info_for_rank(num_frames_total: int, rank: int, sp_size: int):

    frames_per_rank = (num_frames_total + sp_size - 1) // sp_size
    f_start = rank * frames_per_rank
    f_end = min(f_start + frames_per_rank, num_frames_total)
    return frames_per_rank, f_start, f_end


def get_ghost_info_for_rank(num_frames_total: int, rank: int, sp_size: int, chunk_size: int):

    fpr = (num_frames_total + sp_size - 1) // sp_size
    f_start = rank * fpr
    f_end_real = min(f_start + fpr, num_frames_total)
    aligned_start = (f_start // chunk_size) * chunk_size
    ghost_f_start = max(0, aligned_start - chunk_size)
    aligned_end = ((f_end_real + chunk_size - 1) // chunk_size) * chunk_size
    ghost_f_end = min(num_frames_total, aligned_end + chunk_size)
    ghost_before = f_start - ghost_f_start
    ghost_after = ghost_f_end - f_end_real
    return ghost_before, ghost_after


class _HaloExchange(Function):


    @staticmethod
    def forward(ctx, x, num_halo_frames, per_frame_tokens, group, sp_size, sp_rank):
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.num_halo_frames = num_halo_frames
        ctx.per_frame_tokens = per_frame_tokens
        ctx.x_shape = x.shape

        B, S, D = x.shape
        halo_tokens = num_halo_frames * per_frame_tokens


        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size
        prev_rank = sp_group_start + sp_rank - 1
        next_rank = sp_group_start + sp_rank + 1


        send_buf = x[:, -halo_tokens:].contiguous() if sp_rank < sp_size - 1 else None

        recv_buf = torch.zeros(B, halo_tokens, D, device=x.device, dtype=x.dtype) if sp_rank > 0 else None


        recv_bufs = {sp_rank - 1: recv_buf} if sp_rank > 0 else {}
        send_bufs = {sp_rank + 1: send_buf} if sp_rank < sp_size - 1 else {}
        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, x.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        return recv_buf if recv_buf is not None else torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_halo):

        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        B, S, D = ctx.x_shape
        halo_tokens = ctx.num_halo_frames * ctx.per_frame_tokens

        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size
        prev_rank = sp_group_start + sp_rank - 1
        next_rank = sp_group_start + sp_rank + 1

        grad_x = torch.zeros(B, S, D, device=grad_halo.device, dtype=grad_halo.dtype)


        send_buf = grad_halo.contiguous() if sp_rank > 0 and grad_halo.shape[1] > 0 else None

        recv_buf = torch.zeros(B, halo_tokens, D, device=grad_halo.device, dtype=grad_halo.dtype) if sp_rank < sp_size - 1 else None


        recv_bufs = {sp_rank + 1: recv_buf} if sp_rank < sp_size - 1 else {}
        send_bufs = {sp_rank - 1: send_buf} if (sp_rank > 0 and send_buf is not None) else {}
        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, grad_halo.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        if recv_buf is not None:
            grad_x[:, -halo_tokens:] += recv_buf

        return grad_x, None, None, None, None, None


def halo_exchange(x: torch.Tensor, num_halo_frames: int, per_frame_tokens: int) -> Optional[torch.Tensor]:


    if not is_sp_enabled():
        return None
    return _HaloExchange.apply(x, num_halo_frames, per_frame_tokens, _sp_group, _sp_size, _sp_rank)


_SP_SCORE_OWNER = [0]


_SP_DECOUPLE_ACTIVE = [False]


_SP_IN_OWNER_BLOCK = [False]


@contextlib.contextmanager
def sp_score_owner(owner_local_rank: int):


    _prev = _SP_SCORE_OWNER[0]
    _prev_blk = _SP_IN_OWNER_BLOCK[0]
    _SP_SCORE_OWNER[0] = int(owner_local_rank)
    _SP_IN_OWNER_BLOCK[0] = True
    try:
        yield
    finally:
        _SP_SCORE_OWNER[0] = _prev
        _SP_IN_OWNER_BLOCK[0] = _prev_blk


@contextlib.contextmanager
def sp_decouple_scope():


    _prev = _SP_DECOUPLE_ACTIVE[0]
    _SP_DECOUPLE_ACTIVE[0] = True
    try:
        yield
    finally:
        _SP_DECOUPLE_ACTIVE[0] = _prev


def sp_decouple_active() -> bool:
    return bool(_SP_DECOUPLE_ACTIVE[0])


def sp_in_owner_block() -> bool:
    return bool(_SP_IN_OWNER_BLOCK[0])


def sp_current_score_owner() -> int:
    return int(_SP_SCORE_OWNER[0])


def sync_tensor_in_sp_group(x: torch.Tensor) -> torch.Tensor:


    if not is_sp_enabled():
        return x
    if _SP_DECOUPLE_ACTIVE[0] and not _SP_IN_OWNER_BLOCK[0]:
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    dist.broadcast(x, src=sp_group_start + _SP_SCORE_OWNER[0], group=_sp_group)
    return x


def broadcast_from_owner(x: torch.Tensor, owner_local_rank: int) -> torch.Tensor:


    if not is_sp_enabled():
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    dist.broadcast(x, src=sp_group_start + int(owner_local_rank), group=_sp_group)
    return x


def broadcast_varshape_from_owner(x, owner_local_rank: int, ref_device, ref_dtype=None):


    if not is_sp_enabled():
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    src = sp_group_start + int(owner_local_rank)
    is_src = (dist.get_rank() == src)

    meta = torch.zeros(10, dtype=torch.long, device=ref_device)
    if is_src:
        if x is None:
            meta[0] = 1
        else:
            meta[1] = x.dim()
            assert x.dim() <= 8, f"[THROUGHPUT-B] broadcast_varshape ndim>8 not supported: {x.dim()}"
            for _i, _s in enumerate(x.shape):
                meta[2 + _i] = int(_s)
    dist.broadcast(meta, src=src, group=_sp_group)
    if int(meta[0].item()) == 1:
        return None
    _ndim = int(meta[1].item())
    _shape = [int(meta[2 + _i].item()) for _i in range(_ndim)]
    _dt = ref_dtype
    if is_src:
        if _dt is None:
            _dt = x.dtype
        buf = x.to(device=ref_device, dtype=_dt).contiguous()
    else:
        if _dt is None:
            _dt = torch.bfloat16
        buf = torch.empty(_shape, dtype=_dt, device=ref_device)
    dist.broadcast(buf, src=src, group=_sp_group)
    return buf


def broadcast_object_from_owner(obj, owner_local_rank: int):


    if not is_sp_enabled():
        return obj
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    src = sp_group_start + int(owner_local_rank)
    holder = [obj]
    dist.broadcast_object_list(holder, src=src, group=_sp_group)
    return holder[0]


class _BroadcastFromRank0(Function):

    @staticmethod
    def forward(ctx, x, group, sp_size, sp_rank):
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        out = x.clone()
        sp_group_start = (dist.get_rank() // sp_size) * sp_size
        dist.broadcast(out, src=sp_group_start, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous()
        sp_group_start = (dist.get_rank() // ctx.sp_size) * ctx.sp_size
        dist.reduce(grad, dst=sp_group_start, op=dist.ReduceOp.SUM, group=ctx.group)
        if ctx.sp_rank != 0:
            return torch.zeros_like(grad), None, None, None
        return grad, None, None, None


def broadcast_with_grad(x: torch.Tensor, src_rank: int = 0) -> torch.Tensor:

    if not is_sp_enabled():
        return x
    return _BroadcastFromRank0.apply(x, _sp_group, _sp_size, _sp_rank)


def _pad_p2p_ops_for_sync(ops, sp_rank, sp_size, sp_group_start, device, group):


    dummy_send = torch.zeros(1, device=device, dtype=torch.uint8)
    dummy_recv = torch.zeros(1, device=device, dtype=torch.uint8)
    next_rank = sp_group_start + (sp_rank + 1) % sp_size
    prev_rank = sp_group_start + (sp_rank - 1 + sp_size) % sp_size
    ops.append(dist.P2POp(dist.isend, dummy_send, next_rank, group=group))
    ops.append(dist.P2POp(dist.irecv, dummy_recv, prev_rank, group=group))


def _build_uniform_p2p_ops(recv_bufs: dict, send_bufs: dict,
                           sp_rank: int, sp_size: int, sp_group_start: int,
                           device, group):


    ops = []
    for r in range(sp_size):
        if r == sp_rank:
            continue
        target = sp_group_start + r

        if r in recv_bufs and recv_bufs[r] is not None:
            recv_t = recv_bufs[r]
        else:
            recv_t = torch.empty(1, dtype=torch.uint8, device=device)
        ops.append(dist.P2POp(dist.irecv, recv_t, target, group=group))

        if r in send_bufs and send_bufs[r] is not None:
            send_t = send_bufs[r].contiguous()
        else:
            send_t = torch.zeros(1, dtype=torch.uint8, device=device)
        ops.append(dist.P2POp(dist.isend, send_t, target, group=group))
    return ops


class _ExchangeFrameTokensGrad(Function):


    @staticmethod
    def forward(ctx, x, send_map, recv_map, pf, group, sp_size, sp_rank):


        B, S, D = x.shape
        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size


        send_bufs = {}
        for dest_rank, indices in send_map:
            if indices:
                tokens = torch.cat([x[:, idx * pf:(idx + 1) * pf] for idx in indices], dim=1)
                send_bufs[dest_rank] = tokens.contiguous()


        recv_bufs = {}
        recv_order = [r for r, _ in recv_map]
        for src_rank, nf in recv_map:
            recv_bufs[src_rank] = torch.zeros(B, nf * pf, D, device=x.device, dtype=x.dtype)


        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, x.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


        parts = [recv_bufs[r] for r in recv_order if r in recv_bufs]
        received = torch.cat(parts, dim=1) if parts else torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)


        ctx.send_map = send_map
        ctx.recv_map = recv_map
        ctx.recv_order = recv_order
        ctx.pf = pf
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.x_shape = (B, S, D)
        return received

    @staticmethod
    def backward(ctx, grad_received):
        B, S, D = ctx.x_shape
        pf = ctx.pf
        global_rank = dist.get_rank()
        sp_group_start = (global_rank // ctx.sp_size) * ctx.sp_size
        recv_map_dict = dict(ctx.recv_map)


        grad_to_send = {}
        offset = 0
        for src_rank in ctx.recv_order:
            nf = recv_map_dict[src_rank]
            n_tokens = nf * pf
            grad_to_send[src_rank] = grad_received[:, offset:offset + n_tokens].contiguous()
            offset += n_tokens


        grad_to_recv = {}
        for dest_rank, indices in ctx.send_map:
            n_tokens = len(indices) * pf
            grad_to_recv[dest_rank] = torch.zeros(B, n_tokens, D,
                device=grad_received.device, dtype=grad_received.dtype)


        ops = _build_uniform_p2p_ops(grad_to_recv, grad_to_send,
                                     ctx.sp_rank, ctx.sp_size, sp_group_start,
                                     grad_received.device, ctx.group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


        grad_x = torch.zeros(B, S, D, device=grad_received.device, dtype=grad_received.dtype)
        for dest_rank, indices in ctx.send_map:
            grad_buf = grad_to_recv[dest_rank]
            buf_offset = 0
            for idx in indices:
                grad_x[:, idx * pf:(idx + 1) * pf] += grad_buf[:, buf_offset:buf_offset + pf]
                buf_offset += pf

        return grad_x, None, None, None, None, None, None


def exchange_frame_tokens(
    requests: dict,
    x: torch.Tensor,
    per_frame_tokens: int,
    sp_frame_offset: int,
    num_local_frames: int,
    frames_per_rank: int,
) -> dict:


    if not is_sp_enabled():
        return {}

    sp_size = _sp_size
    sp_rank = _sp_rank
    group = _sp_group
    global_rank = dist.get_rank()
    sp_group_start = (global_rank // sp_size) * sp_size
    B, S, D = x.shape
    pf = per_frame_tokens


    send_counts = torch.zeros(sp_size, dtype=torch.long, device=x.device)
    for src_rank, frame_list in requests.items():
        send_counts[src_rank] = len(frame_list)

    recv_counts = torch.zeros(sp_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts, group=group)


    send_idx_bufs = {}
    for r in range(sp_size):
        if send_counts[r] > 0:
            send_idx_bufs[r] = torch.tensor(requests[r], dtype=torch.long, device=x.device)

    recv_idx_bufs = {}
    for r in range(sp_size):
        cnt = recv_counts[r].item()
        if cnt > 0:
            recv_idx_bufs[r] = torch.zeros(cnt, dtype=torch.long, device=x.device)


    ops = _build_uniform_p2p_ops(recv_idx_bufs, send_idx_bufs,
                                 sp_rank, sp_size, sp_group_start, x.device, group)
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


    send_map = []
    for r in range(sp_size):
        if r == sp_rank or r not in recv_idx_bufs:
            continue
        local_indices = [(gfi.item() - sp_frame_offset) for gfi in recv_idx_bufs[r]]
        send_map.append((r, local_indices))


    recv_map = []
    for r in range(sp_size):
        if r == sp_rank or send_counts[r] == 0:
            continue
        recv_map.append((r, send_counts[r].item()))


    received = _ExchangeFrameTokensGrad.apply(
        x, send_map, recv_map, pf, group, sp_size, sp_rank)


    token_cache = {}
    offset = 0
    for src_rank, nf in recv_map:
        for gfi in requests[src_rank]:
            token_cache[gfi] = received[:, offset:offset + pf]
            offset += pf


    return token_cache, received
