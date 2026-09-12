from contextlib import contextmanager
from types import MethodType

_KERNEL_SIGNATURES = set()
_MISSING = object()


def canonical_contiguous(shape, stride):
    expected = 1
    for size, actual in zip(reversed(shape), reversed(stride)):
        if actual != expected:
            return False
        expected *= size
    return True


def effective_padding(module, cache):
    padding = list(module._padding)
    uses_cache = cache is not None and padding[4] > 0
    if uses_cache:
        padding[4] -= cache.shape[2]
    return tuple(padding), uses_cache


def _original_input(module, x, cache):
    import torch
    import torch.nn.functional as F
    padding, uses_cache = effective_padding(module, cache)
    combined = torch.cat([cache.to(x.device), x], dim=2) if uses_cache else x
    return F.pad(combined, padding)


def fused_eligibility(x, cache, padding, uses_cache):
    import torch
    import math
    if not uses_cache:
        return 'no-causal-cache'
    if x.ndim != 5 or cache.ndim != 5:
        return 'rank'
    if x.dtype != torch.float32 or cache.dtype != torch.float32:
        return 'dtype'
    if x.device.type != 'cuda' or cache.device != x.device:
        return 'device'
    if any(value < 0 for value in padding):
        return 'negative-padding'
    if any(size <= 0 for size in x.shape) or any(size <= 0 for size in cache.shape):
        return 'empty-shape'
    if tuple(x.shape[i] for i in (0, 1, 3, 4)) != tuple(cache.shape[i] for i in (0, 1, 3, 4)):
        return 'cache-shape'


    if not canonical_contiguous(cache.shape, cache.stride()):
        return 'cache-layout'
    if cache.is_contiguous(memory_format=torch.channels_last_3d) and x.is_contiguous(memory_format=torch.channels_last_3d):
        return 'channels-last-ambiguous'
    if any(stride < 0 for stride in (*x.stride(), *cache.stride())):
        return 'negative-stride'
    out_shape = (x.shape[0], x.shape[1], x.shape[2] + cache.shape[2] + padding[4] + padding[5],
                 x.shape[3] + padding[2] + padding[3], x.shape[4] + padding[0] + padding[1])
    if (math.prod(out_shape) >= 2**31
            or any(sum((size-1)*stride for size, stride in zip(value.shape, value.stride())) >= 2**31
                   for value in (x, cache))):
        return 'index-range'
    return None


def _bump(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


def _timed(stats, label, x, enabled, operation):
    import torch
    if not enabled or x.device.type != 'cuda':
        return operation()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = operation()
    end.record()
    stats['events'].append((label, begin, end))
    return result


def _forward(module, x, cache, original, options, stats, name):
    import torch
    _bump(stats, 'convCalls')
    padding, uses_cache = effective_padding(module, cache)
    skip = options['skipZeroPad'] and x.dtype == torch.float32 and not any(padding)


    if skip and uses_cache:
        skip = (cache.device == x.device and cache.dtype == x.dtype
                and canonical_contiguous(cache.shape, cache.stride())
                and not (cache.is_contiguous(memory_format=torch.channels_last_3d)
                         and x.is_contiguous(memory_format=torch.channels_last_3d)))
    elif skip:
        skip = canonical_contiguous(x.shape, x.stride())
    reason = fused_eligibility(x, cache, padding, uses_cache) if options['fuseCausalInput'] else 'disabled'
    fused = options['fuseCausalInput'] and reason is None and not skip
    if not skip and not fused:
        _bump(stats, 'originalCalls')
        _bump(stats['fallbackReasons'], reason if any(padding) else 'zero-pad-layout-or-disabled')
        if not options['profile']:
            return original(x, cache_x=cache)
        prepared = _timed(stats, 'originalAssembly', x, True, lambda: _original_input(module, x, cache))
        return torch.nn.Conv3d.forward(module, prepared)

    def candidate():
        if skip:
            return torch.cat([cache, x], dim=2) if uses_cache else x
        from .vae_memory_triton import assemble
        signature = (tuple(x.shape), tuple(x.stride()), tuple(cache.shape), tuple(cache.stride()),
                     padding, options['blockSize'])
        if signature not in _KERNEL_SIGNATURES:
            _bump(stats, 'kernelNewSignatures')
        result = assemble(x, cache, padding, block_size=options['blockSize'])
        _KERNEL_SIGNATURES.add(signature)
        return result

    prepared = _timed(stats, 'candidateAssembly', x, options['profile'], candidate)
    _bump(stats, 'skipZeroPadCalls' if skip else 'fusedCausalInputCalls')
    _bump(stats, 'assembledOutputBytes', prepared.numel() * prepared.element_size())
    avoided = prepared.numel() * prepared.element_size() if skip else (x.numel() + cache.numel()) * x.element_size()
    _bump(stats, 'avoidedIntermediatePayloadBytes', avoided)
    if options['verify']:
        reference = _timed(stats, 'verifyOriginalAssembly', x, options['profile'], lambda: _original_input(module, x, cache))
        layout_exact = (reference.shape == prepared.shape and reference.stride() == prepared.stride()
                        and reference.dtype == prepared.dtype and reference.device == prepared.device)

        value_exact = torch.equal(reference.view(torch.int32), prepared.view(torch.int32)) if prepared.dtype == torch.float32 else torch.equal(reference, prepared)
        _bump(stats, 'verifiedCalls')
        if not layout_exact or not value_exact:
            raise AssertionError(f'Causal input audit failed at {name}: layout={layout_exact}, bits={value_exact}; '
                                 f'old={tuple(reference.shape)}/{reference.stride()}, '
                                 f'new={tuple(prepared.shape)}/{prepared.stride()}')
    return torch.nn.Conv3d.forward(module, prepared)


@contextmanager
def memory_mode(vae, options=None):
    options = dict(options or {})
    unknown = set(options) - {'skipZeroPad', 'fuseCausalInput', 'verify', 'profile', 'blockSize'}
    if unknown:
        raise ValueError(f'Unknown VAE memory options: {sorted(unknown)}')
    block_size = options.get('blockSize', 256)
    if type(block_size) is not int or block_size not in (256, 512, 1024):
        raise ValueError('VAE memory blockSize must be an integer in (256, 512, 1024)')
    flags = ('skipZeroPad', 'fuseCausalInput', 'verify', 'profile')
    options = {name: bool(options.get(name, False)) for name in flags}
    options['blockSize'] = block_size
    stats = {'options': options, 'events': [], 'fallbackReasons': {}}
    if not any(options[name] for name in flags):
        yield stats
        return
    if getattr(vae, '_evoke_memory_mode_active', False):
        raise RuntimeError('VAE memory_mode is already active; serialize decoder access')
    vae._evoke_memory_mode_active = True
    originals = []
    try:
        for name, module in vae.decoder.named_modules():
            if type(module).__name__ != 'WanCausalConv3d':
                continue
            previous = module.__dict__.get('forward', _MISSING)
            original = module.forward
            originals.append((module, previous))
            def patched(this, x, cache_x=None, _original=original, _name=name):
                return _forward(this, x, cache_x, _original, options, stats, _name)
            patched._evoke_memory_original = original
            module.forward = MethodType(patched, module)
        stats['patchedModules'] = len(originals)
        yield stats
    finally:
        for module, previous in reversed(originals):
            if previous is _MISSING:
                del module.forward
            else:
                module.forward = previous
        del vae._evoke_memory_mode_active


def resolve_memory_timing(stats):

    result = {key: value for key, value in stats.items() if key != 'events'}
    for label, begin, end in stats['events']:
        _bump(result, label + 'CudaSeconds', begin.elapsed_time(end) / 1000)
        _bump(result, label + 'TimedCalls')
    return result
