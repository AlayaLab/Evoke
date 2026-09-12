import ast
import inspect
import textwrap
from contextlib import contextmanager
from types import MethodType

_TEMPLATES = {}
_MISSING = object()
_PATTERN = ast.dump(ast.parse('x[:, :, -CACHE_T:, :, :].clone()', mode='eval').body)
_CAT_BODY = ast.dump(ast.parse('cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)').body[0])
_CAT_CONDITIONS = {
    ast.dump(ast.parse(expression, mode='eval').body) for expression in (
        'cache_x.shape[2] < 2 and feat_cache[idx] is not None',
        'cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep"')
}


def _bump(stats, key, n=1):
    stats[key] = stats.get(key, 0) + n


def _timed(stats, name, value, profile, operation):
    import torch
    if not profile or value.device.type != 'cuda':
        return operation()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = operation()
    end.record()
    stats['events'].append((name, begin, end))
    return result


def _tail(module, x, cache, idx):
    import torch
    stats, verify, profile, pending = module._evoke_cache_context
    previous = cache[idx]
    tail = x[:, :, -2:, :, :]
    _bump(stats, 'tailCalls')
    eligible = (tail.shape[2] == 1 and isinstance(previous, torch.Tensor)
                and x.dtype == previous.dtype == torch.float32 and x.device == previous.device
                and previous.ndim == 5 and previous.shape[2] > 0
                and previous.is_contiguous()
                and not previous.is_contiguous(memory_format=torch.channels_last_3d))
    if not eligible:
        _bump(stats, 'retainedCloneCalls')
        _bump(stats, 'retainedClonePayloadBytes', tail.numel() * tail.element_size())
        return _timed(stats, 'retainedTailClone', tail, profile, tail.clone)
    _bump(stats, 'removedCloneCalls')
    _bump(stats, 'removedClonePayloadBytes', tail.numel() * tail.element_size())
    if verify:
        def reference_write():
            original_tail = tail.clone()
            return torch.cat([previous[:, :, -1, :, :].unsqueeze(2).to(original_tail.device), original_tail], dim=2)
        expected = _timed(stats, 'verifyOriginalWrite', tail, profile, reference_write)
        pending[-1].append((cache, idx, expected))


    return tail


def _template(function, expected_count):
    if function in _TEMPLATES:
        return _TEMPLATES[function]
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    def matches(node):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'cache_x'
                and ast.dump(node.value) == _PATTERN)
    guarded = 0
    for parent in ast.walk(tree):
        for _, block in ast.iter_fields(parent):
            if not isinstance(block, list):
                continue
            for i, node in enumerate(block):
                if not matches(node):
                    continue
                following = block[i+1] if i+1 < len(block) else None
                if (not isinstance(following, ast.If) or ast.dump(following.test) not in _CAT_CONDITIONS
                        or len(following.body) != 1 or ast.dump(following.body[0]) != _CAT_BODY
                        or following.orelse):
                    raise ValueError('Clone is not followed by the audited conditional cache cat')
                guarded += 1
    if guarded != expected_count:
        raise ValueError(f'Stock guarded cache pattern count {guarded} != {expected_count}')
    class Replace(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            if matches(node):
                node.value = ast.copy_location(ast.parse('_evoke_tail(self, x, feat_cache, idx)', mode='eval').body, node.value)
                self.count += 1
            return node
    replace = Replace()
    tree = replace.visit(tree)
    if replace.count != expected_count or function.__globals__.get('CACHE_T') != 2:
        raise ValueError(f'Stock cache pattern mismatch: {replace.count} != {expected_count}, CACHE_T != 2?')
    namespace = dict(function.__globals__)
    namespace['_evoke_tail'] = _tail
    ast.fix_missing_locations(tree)
    exec(compile(tree, f'<evoke-cache:{function.__qualname__}>', 'exec'), namespace)
    result = namespace[function.__name__]
    _TEMPLATES[function] = result
    return result


@contextmanager
def cache_mode(vae, enabled=False, verify=False, profile=False):


    import torch
    stats = {'enabled': bool(enabled), 'verify': bool(verify), 'profile': bool(profile),
             'patchedModules': 0, 'events': [], 'sourceFallbacks': [],
             'profileNote': 'retainedTailClone excludes final cat; verifyOriginalWrite is extra audit work.'}
    if not enabled:
        yield stats
        return
    if getattr(vae, '_evoke_cache_mode_active', False):
        raise RuntimeError('cache_mode is already active; serialize decoder access')
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d, WanResidualBlock, WanResample
    counts = {WanDecoder3d: 2, WanResidualBlock: 2, WanResample: 1}
    vae._evoke_cache_mode_active = True
    patched = []
    try:
        for name, module in vae.decoder.named_modules():
            if type(module) not in counts:
                continue
            original = module.forward
            function = getattr(original, '__func__', None)
            if function is not type(module).forward:
                stats['sourceFallbacks'].append({'module': name, 'reason': 'custom-forward'})
                continue
            try:
                compiled = _template(function, counts[type(module)])
            except (OSError, IOError, TypeError, ValueError, SyntaxError) as error:
                stats['sourceFallbacks'].append({'module': name, 'reason': str(error)})
                continue
            previous = module.__dict__.get('forward', _MISSING)
            previous_context = module.__dict__.get('_evoke_cache_context', _MISSING)
            module._evoke_cache_context = (stats, bool(verify), bool(profile), [])
            def forward(self, *args, _original=original, _compiled=compiled, _name=name, **kwargs):
                if self.training or torch.is_grad_enabled():
                    _bump(stats, 'executionFallbackCalls')
                    return _original(*args, **kwargs)
                pending = self._evoke_cache_context[3]
                pending.append([])
                try:
                    output = _compiled(self, *args, **kwargs)
                    for cache, idx, expected in pending[-1]:
                        actual = cache[idx]
                        same_layout = (actual.shape, actual.stride(), actual.dtype, actual.device) == (
                            expected.shape, expected.stride(), expected.dtype, expected.device)
                        same_bits = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                        if not same_layout or not same_bits:
                            raise AssertionError(f'T1 cache write mismatch at {_name}, slot {idx}: layout={same_layout}, bits={same_bits}')
                        _bump(stats, 'verifiedWrites')
                    return output
                finally:
                    pending.pop()
            module.forward = MethodType(forward, module)
            patched.append((module, previous, previous_context))
        stats['patchedModules'] = len(patched)
        yield stats
    finally:
        for module, previous, previous_context in reversed(patched):
            if previous is _MISSING:
                del module.forward
            else:
                module.forward = previous
            if previous_context is _MISSING:
                del module._evoke_cache_context
            else:
                module._evoke_cache_context = previous_context
        del vae._evoke_cache_mode_active


def resolve_cache_timing(stats):

    result = {key: value for key, value in stats.items() if key != 'events'}
    for name, begin, end in stats['events']:
        _bump(result, name + 'CudaSeconds', begin.elapsed_time(end) / 1000)
        _bump(result, name + 'TimedCalls')
    return result
