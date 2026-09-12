import triton
import triton.language as tl


@triton.jit
def causal_pack(x, cache, output, next_cache, N: tl.constexpr,
                C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                CT: tl.constexpr, OT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                PT: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr, X4: tl.constexpr,
                K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, K4: tl.constexpr,
                BLOCK: tl.constexpr, NT: tl.constexpr = 0):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = index % OW - PW
    h = (index // OW) % OH - PH
    t = (index // (OW * OH)) % OT - PT
    c = (index // (OW * OH * OT)) % C
    b = index // (OW * OH * OT * C)
    inside = (index < N) & (w >= 0) & (w < W) & (h >= 0) & (h < H) & (t >= 0) & (t < CT + T)
    old = tl.load(cache + b*K0 + c*K1 + t*K2 + h*K3 + w*K4,
                  mask=inside & (t < CT), other=0)
    new = tl.load(x + b*X0 + c*X1 + (t-CT)*X2 + h*X3 + w*X4,
                  mask=inside & (t >= CT), other=0)
    value = tl.where(t < CT, old, new)
    tl.store(output + index, value, mask=index < N)
    if NT > 0:
        next_t = t - (CT + T - NT)
        next_index = (((b * C + c) * NT + next_t) * H + h) * W + w
        tl.store(next_cache + next_index, value,
                 mask=inside & (next_t >= 0) & (next_t < NT))


@triton.jit
def causal_pack_joint_planes(x, cache, output, next_cache,
                C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                CT: tl.constexpr, OT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                PT: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr, X4: tl.constexpr,
                K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, K4: tl.constexpr,
                BLOCK: tl.constexpr):


    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    plane = tl.program_id(1)
    t = plane % OT - PT
    bc = plane // OT
    b = bc // C
    c = bc % C
    h = index // OW - PH
    w = index % OW - PW
    inside = (index < OH * OW) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
    if t >= 0 and t < CT:
        value = tl.load(cache + b*K0 + c*K1 + t*K2 + h*K3 + w*K4, mask=inside, other=0)
    elif t >= CT and t < CT + T:
        value = tl.load(x + b*X0 + c*X1 + (t-CT)*X2 + h*X3 + w*X4, mask=inside, other=0)
    else:
        value = tl.full((BLOCK,), 0, tl.float32)
    tl.store(output + plane * OH * OW + index, value, mask=index < OH * OW)
    next_t = t - (CT + T - 2)
    if next_t >= 0 and next_t < 2:
        tl.store(next_cache + (bc * 2 + next_t) * H * W + h * W + w, value, mask=inside)


def assemble(x, cache, padding, block_size=256, next_cache_t=0, joint_planes=False, num_warps=4):
    import torch
    if type(block_size) is not int or block_size not in (256, 512, 1024, 2048):
        raise ValueError('Invalid causal assembly block_size')
    pw0, pw1, ph0, ph1, pt0, pt1 = padding
    b, c, t, h, w = x.shape
    ct = cache.shape[2]
    shape = (b, c, ct + t + pt0 + pt1, h + ph0 + ph1, w + pw0 + pw1)
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    if next_cache_t not in (0, 2):
        raise ValueError('Only the audited steady two-frame cache is supported')
    next_cache = (torch.empty((b, c, next_cache_t, h, w), device=x.device, dtype=x.dtype)
                  if next_cache_t else output)
    if next_cache_t and joint_planes:
        causal_pack_joint_planes[(triton.cdiv(shape[3] * shape[4], block_size), b*c*shape[2])](
            x, cache, output, next_cache, c, t, h, w, ct, *shape[2:], pt0, ph0, pw0,
            *x.stride(), *cache.stride(), BLOCK=block_size, num_warps=num_warps)
    else:
        causal_pack[(triton.cdiv(output.numel(), block_size),)](
            x, cache, output, next_cache, output.numel(), c, t, h, w, ct, *shape[2:], pt0, ph0, pw0,
            *x.stride(), *cache.stride(), BLOCK=block_size, NT=next_cache_t, num_warps=num_warps,
        )
    return (output, next_cache) if next_cache_t else output
