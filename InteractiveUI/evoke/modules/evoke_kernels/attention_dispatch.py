import torch


import os as _os
flash_attn_func = None
flash_attn_varlen_func = None
_backend_name = None

try:
    _major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
except Exception:
    _major = 0

_force_fa2 = _os.environ.get("EVOKE_FORCE_FA2", "0") == "1"


if _major >= 9 and not _force_fa2:
    try:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        _backend_name = "FA3 (flash_attn_interface pip)"
    except ImportError:
        pass


if flash_attn_func is None:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        _backend_name = "FA2 (flash_attn pip standalone)"
    except ImportError:
        print("[attention_dispatch] standalone flash_attn not installed, falling back to kernels-community...")
        try:
            from kernels import get_kernel
            if _major >= 9:
                flash_attn3 = get_kernel("kernels-community/flash-attn3")
                flash_attn_func = flash_attn3.flash_attn_func
                flash_attn_varlen_func = flash_attn3.flash_attn_varlen_func
                _backend_name = "FA3 (kernels-community)"
            else:
                flash_attn2 = get_kernel("kernels-community/flash-attn2")
                flash_attn_func = flash_attn2.flash_attn_func
                flash_attn_varlen_func = flash_attn2.flash_attn_varlen_func
                _backend_name = "FA2 (kernels-community)"
        except Exception as _kernels_err:
            print(f"[attention_dispatch] kernels-community fallback failed ({type(_kernels_err).__name__}). attn_varlen_func=None.")

if _backend_name:
    print(f"Flash Attn backend: {_backend_name}  (sm_{_major}0, force_fa2={_force_fa2})")


try:
    from sageattention import sageattn, sageattn_varlen

    print("Sage Attn is installed!")
except ImportError:
    print("Sage Attn is not installed!")
    sageattn_varlen = None
    sageattn = None

try:
    from xformers.ops import memory_efficient_attention as xformers_attn_func

    print("Xformers is installed!")
except ImportError:
    print("Xformers is not installed!")
    xformers_attn_func = None


def create_navit_attention_masks(
    batch_size: int,
    original_context_length_list: list,
    history_context_length: int,
    encoder_hidden_states_seq_len: int,
    device: torch.device,
    restrict_self_attn: bool = False,
    guidance_cross_attn: bool = False,
    warp_len_list: list = None,
):


    _wl = warp_len_list if warp_len_list is not None else [0] * len(original_context_length_list)
    assert len(_wl) == len(original_context_length_list), (
        f"warp_len_list len {len(_wl)} != original_context_length_list len {len(original_context_length_list)}"
    )

    _self_kv = [length + history_context_length + w for length, w in zip(original_context_length_list, _wl)]


    if restrict_self_attn:
        cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
        max_seqlen_q = max(original_context_length_list)

        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + kvlen)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(_self_kv)
    else:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + kvlen)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(_self_kv)
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
    navit_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv


    navit_history_hidden_attention_mask = None
    if restrict_self_attn:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
        navit_history_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv


    if guidance_cross_attn:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_max_seqlen_q = max(original_context_length_list)
    else:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for kvlen in _self_kv:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + kvlen)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_cu_seqlens_q[0] = 0
        cross_max_seqlen_q = max(_self_kv)

    cu_seqlens_kv = [0]
    for _ in range(batch_size):
        for length in original_context_length_list:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + encoder_hidden_states_seq_len)
    cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
    max_seqlen_kv = encoder_hidden_states_seq_len
    navit_encoder_attention_mask = cross_cu_seqlens_q, cu_seqlens_kv, cross_max_seqlen_q, max_seqlen_kv

    return navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask


@torch.compiler.disable
def _flash_attn_wrapper(q, k, v):

    out = flash_attn_func(q, k, v)
    return out[0] if isinstance(out, tuple) else out


@torch.compiler.disable
def _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    return out[0] if isinstance(out, tuple) else out


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is None:
        if flash_attn_func is not None:
            x = _flash_attn_wrapper(q, k, v)
            return x

        if sageattn is not None:
            x = sageattn(q, k, v, tensor_layout="NHD")
            return x

        if xformers_attn_func is not None:
            x = xformers_attn_func(q, k, v)
            return x

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return x

    B, L, H, C = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
    if flash_attn_varlen_func is not None:
        x = _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        raise NotImplementedError("No Attn Installed!")

    x = x.unflatten(0, (B, L))

    return x
