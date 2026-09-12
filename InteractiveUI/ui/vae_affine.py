from __future__ import annotations

from contextlib import contextmanager
import math
import struct
from types import MethodType

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _affine_kernel(
        normalized, denominator, gamma, bias, output, count,
        CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
        SCALE_BITS: tl.constexpr, BIAS_BITS: tl.constexpr,
        TENSOR_BIAS: tl.constexpr, FUSE_DIVISION: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < count
        channel = (offset // SPATIAL) % CHANNELS
        value = tl.load(normalized + offset, mask=valid, other=0).to(tl.float32)
        if FUSE_DIVISION:


            denominator_offset = (offset // (CHANNELS * SPATIAL)) * SPATIAL + offset % SPATIAL
            divisor = tl.load(denominator + denominator_offset, mask=valid, other=1).to(tl.float32)
            value = tl.inline_asm_elementwise(
                "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                args=[value, divisor], dtype=tl.float32, is_pure=True, pack=1,
            )
        gain = tl.load(gamma + channel, mask=valid, other=0).to(tl.float32)
        scale = tl.full((), SCALE_BITS, tl.uint32).to(tl.float32, bitcast=True)
        if TENSOR_BIAS:
            shift = tl.load(bias + channel, mask=valid, other=0).to(tl.float32)
        else:
            shift = tl.full((), BIAS_BITS, tl.uint32).to(tl.float32, bitcast=True)


        scaled = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[value, scale], dtype=tl.float32, is_pure=True, pack=1,
        )
        weighted = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[scaled, gain], dtype=tl.float32, is_pure=True, pack=1,
        )
        result = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[weighted, shift], dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(output + offset, result, mask=valid)


def _scalar_fp32_bits(value):


    if type(value) not in (int, float):
        return None
    if type(value) is int and abs(value) > 2**24:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    try:
        encoded = struct.pack('<f', value)
    except (OverflowError, struct.error):
        return None
    rounded = struct.unpack('<f', encoded)[0]
    if not math.isfinite(rounded):
        return None
    return struct.unpack('<I', encoded)[0]


def _layout(module, x):

    if (x.dtype != torch.float32 or x.layout != torch.strided or x.ndim not in (4, 5)
            or not x.is_contiguous() or x.numel() == 0 or x.numel() >= 2**31 - 1024
            or not module.channel_first):
        return None
    channels = x.shape[1]
    expected = (channels,) + (1,) * (x.ndim - 2)
    gamma = module.gamma
    if (gamma.dtype != torch.float32 or gamma.device != x.device or gamma.layout != torch.strided
            or tuple(gamma.shape) != expected or not gamma.is_contiguous()):
        return None
    scale_bits = _scalar_fp32_bits(module.scale)
    if scale_bits is None:
        return None
    bias = module.bias
    if isinstance(bias, torch.Tensor):
        if (bias.dtype != torch.float32 or bias.device != x.device or bias.layout != torch.strided
                or tuple(bias.shape) != expected or not bias.is_contiguous()):
            return None
        bias_bits = 0
    else:
        bias_bits = _scalar_fp32_bits(bias)
        if bias_bits is None:
            return None
    return channels, math.prod(x.shape[2:]), scale_bits, bias_bits


def _run_affine(normalized, module, layout, denominator=None):
    channels, spatial, scale_bits, bias_bits = layout
    output = torch.empty_like(normalized)
    tensor_bias = isinstance(module.bias, torch.Tensor)

    bias = module.bias if tensor_bias else module.gamma
    fuse_division = denominator is not None
    denominator_ptr = denominator if fuse_division else module.gamma
    with torch.cuda.device(normalized.device):
        _affine_kernel[(triton.cdiv(normalized.numel(), 1024),)](
            normalized, denominator_ptr, module.gamma, bias, output, normalized.numel(),
            CHANNELS=channels, SPATIAL=spatial, SCALE_BITS=scale_bits, BIAS_BITS=bias_bits,
            TENSOR_BIAS=tensor_bias, FUSE_DIVISION=fuse_division,
            BLOCK=1024, num_warps=4, enable_fp_fusion=False,
        )
    return output


@contextmanager
def affine_mode(vae, enabled=False, verify=False, fuse_division=False):


    stats = {'enabled': bool(enabled), 'verify': bool(verify), 'fuseDivision': bool(fuse_division),
             'patchedModules': 0, 'fusedCalls': 0, 'fusedDivisionCalls': 0,
             'fallbackCalls': 0, 'verifiedCalls': 0}
    if not enabled:
        yield stats
        return
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    decoder = getattr(vae, 'decoder', None)
    if not isinstance(decoder, torch.nn.Module):
        raise ValueError('affine_mode requires a VAE with a decoder module')
    patched = []
    try:
        for name, module in decoder.named_modules():
            if type(module) is not WanRMS_norm:
                continue
            original = module.forward


            if getattr(original, '__func__', None) is not WanRMS_norm.forward:
                continue
            had_instance_forward = 'forward' in module.__dict__
            instance_forward = module.__dict__.get('forward')

            def forward(self, x, _original=original, _name=name):
                layout = _layout(self, x)
                if (triton is None or x.device.type != 'cuda' or self.training
                        or torch.is_grad_enabled() or layout is None):
                    stats['fallbackCalls'] += 1
                    return _original(x)
                if fuse_division:


                    denominator = x.norm(2.0, 1, keepdim=True).clamp_min(1e-12)
                    if not denominator.is_contiguous():
                        stats['fallbackCalls'] += 1
                        return _original(x)
                    result = _run_affine(x, self, layout, denominator=denominator)
                    stats['fusedDivisionCalls'] += 1
                else:


                    normalized = F.normalize(x, dim=1).to(x.dtype)
                    if not normalized.is_contiguous():
                        stats['fallbackCalls'] += 1
                        return _original(x)
                    result = _run_affine(normalized, self, layout)
                stats['fusedCalls'] += 1
                if verify:


                    reference = _original(x)
                    if not torch.equal(result.view(torch.int32), reference.contiguous().view(torch.int32)):
                        raise RuntimeError(f'Wan affine FP32 bitwise verification failed in decoder.{_name}')
                    stats['verifiedCalls'] += 1
                return result

            module.forward = MethodType(forward, module)
            patched.append((module, had_instance_forward, instance_forward))
        stats['patchedModules'] = len(patched)
        yield stats
    finally:
        for module, had_instance_forward, instance_forward in reversed(patched):
            if had_instance_forward:
                module.forward = instance_forward
            else:
                delattr(module, 'forward')
