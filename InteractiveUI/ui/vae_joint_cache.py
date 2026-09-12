import ast
import inspect
import textwrap
from contextlib import contextmanager
from types import MethodType

from .vae_memory import canonical_contiguous, effective_padding, fused_eligibility, _original_input

_MISSING = object()
_TEMPLATES = {}
_PREFIX = ast.parse('''
cache_x = x[:, :, -CACHE_T:, :, :].clone()
if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
''').body
_WRITE = ast.parse('feat_cache[idx] = cache_x').body[0]


def _bump(stats, key, n=1):
    stats[key] = stats.get(key, 0) + n


def _reference_cache(x, old):
    import torch
    tail = x[:, :, -2:, :, :].clone()
    if tail.shape[2] < 2 and old is not None:
        tail = torch.cat([old[:, :, -1, :, :].unsqueeze(2).to(tail.device), tail], dim=2)
    return tail


def _joint(module, conv, x, cache, idx):
    import torch
    stats, verify, block_size, planes, warps = module._evoke_joint_context
    old = cache[idx]
    padding, uses_cache = effective_padding(conv, old)
    reason = fused_eligibility(x, old, padding, uses_cache)
    if reason is None and x.shape[2] >= 2 and not canonical_contiguous(x.shape, x.stride()):
        reason = 'tail-clone-layout'
    if reason is None and (type(conv).__name__ != 'WanCausalConv3d'
                           or any(conv.padding) or conv.padding_mode != 'zeros'):
        reason = 'conv-contract'
    underlying = getattr(conv.forward, '_evoke_memory_original', conv.forward)
    if reason is None and (getattr(underlying, '__func__', None) is not type(conv).forward
                           or conv._forward_hooks or conv._forward_pre_hooks):
        reason = 'custom-conv-or-hooks'
    if reason is not None:
        _bump(stats['fallbackReasons'], reason)
        next_cache = _reference_cache(x, old)
        output = conv(x, old)
    else:
        from .vae_memory_triton import assemble
        prepared, next_cache = assemble(x, old, padding, block_size=block_size, next_cache_t=2,
                                       joint_planes=planes, num_warps=warps)
        _bump(stats, 'fusedCalls')
        _bump(stats, 'cacheOutputPayloadBytes', next_cache.numel() * next_cache.element_size())
        _bump(stats, 'avoidedTailClonePayloadBytes', x[:, :, -2:].numel() * x.element_size())
        if verify:
            expected_input = _original_input(conv, x, old)
            expected_cache = _reference_cache(x, old)
            for label, expected, actual in (('input', expected_input, prepared), ('cache', expected_cache, next_cache)):
                layout = (expected.shape, expected.stride(), expected.dtype, expected.device) == (
                    actual.shape, actual.stride(), actual.dtype, actual.device)
                bits = torch.equal(expected.view(torch.int32), actual.view(torch.int32))
                if not layout or not bits:
                    raise AssertionError(f'Joint conv/cache {label} mismatch at slot {idx}: layout={layout}, bits={bits}')
            _bump(stats, 'verifiedWrites')
        output = torch.nn.Conv3d.forward(conv, prepared)


    cache[idx] = next_cache
    return output


def _template(function, conv_names):
    key = (function, tuple(conv_names))
    if key in _TEMPLATES:
        return _TEMPLATES[key]
    if function.__globals__.get('CACHE_T') != 2:
        raise ValueError('Unsupported CACHE_T')
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    seen = []
    class Replace(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field, block in ast.iter_fields(node):
                if not isinstance(block, list):
                    continue
                out = []; i = 0
                while i < len(block):
                    if (i + 3 < len(block) and isinstance(block[i], ast.Assign)
                            and ast.dump(block[i]) == ast.dump(_PREFIX[0])
                            and ast.dump(block[i+1]) == ast.dump(_PREFIX[1])
                            and ast.dump(block[i+3]) == ast.dump(_WRITE)):
                        name = next((name for name in conv_names if ast.dump(block[i+2]) == ast.dump(
                            ast.parse(f'x = self.{name}(x, feat_cache[idx])').body[0])), None)
                        if name is None:
                            raise ValueError('Unsupported cache consumer')
                        seen.append(name)
                        out.append(ast.copy_location(ast.parse(
                            f'x = _evoke_joint(self, self.{name}, x, feat_cache, idx)').body[0], block[i]))
                        i += 4
                    else:
                        out.append(block[i]); i += 1
                setattr(node, field, out)
            return node
    tree = Replace().visit(tree)
    if sorted(seen) != sorted(conv_names):
        raise ValueError(f'Unsupported stock cache transactions: {seen} != {conv_names}')
    namespace = dict(function.__globals__); namespace['_evoke_joint'] = _joint
    ast.fix_missing_locations(tree)
    exec(compile(tree, f'<evoke-joint:{function.__qualname__}>', 'exec'), namespace)
    _TEMPLATES[key] = namespace[function.__name__]
    return _TEMPLATES[key]


@contextmanager
def joint_cache_mode(vae, enabled=False, verify=False, block_size=1024, planes=False, warps=4):
    import torch
    stats = {'enabled': bool(enabled), 'events': [], 'fallbackReasons': {}, 'sourceFallbacks': []}
    if not enabled:
        yield stats
        return
    if type(block_size) is not int or block_size not in (256, 512, 1024, 2048):
        raise ValueError('Invalid joint cache block size')
    if type(warps) is not int or warps not in (4, 8):
        raise ValueError('Invalid joint cache warps')
    if torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks:
        stats['sourceFallbacks'].append({'module': '*', 'reason': 'global-hooks'})
        yield stats
        return
    root = vae.decoder
    if getattr(root, '_evoke_joint_active', False):
        raise RuntimeError('Joint cache mode already active; serialize this VAE')
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d, WanEncoder3d, WanResidualBlock
    specs = {WanDecoder3d: ('conv_in', 'conv_out'), WanEncoder3d: ('conv_in', 'conv_out'),
             WanResidualBlock: ('conv1', 'conv2')}
    root._evoke_joint_active = True
    originals = []
    try:
        for name, module in root.named_modules():
            if type(module) not in specs:
                continue
            original = module.forward
            if getattr(original, '__func__', None) is not type(module).forward:
                stats['sourceFallbacks'].append({'module': name, 'reason': 'custom-forward'})
                continue
            try:
                compiled = _template(type(module).forward, specs[type(module)])
            except (ValueError, TypeError, SyntaxError, OSError) as error:
                stats['sourceFallbacks'].append({'module': name, 'reason': str(error)})
                continue
            previous = module.__dict__.get('forward', _MISSING)
            context = module.__dict__.get('_evoke_joint_context', _MISSING)
            module._evoke_joint_context = (stats, bool(verify), block_size, bool(planes), warps)
            def forward(self, *args, _compiled=compiled, _original=original, **kwargs):
                if self.training or torch.is_grad_enabled():
                    _bump(stats, 'executionFallbackCalls')
                    return _original(*args, **kwargs)
                return _compiled(self, *args, **kwargs)
            module.forward = MethodType(forward, module)
            originals.append((module, previous, context))
        stats['patchedModules'] = len(originals)
        yield stats
    finally:
        for module, previous, context in reversed(originals):
            if previous is _MISSING:
                del module.forward
            else:
                module.forward = previous
            if context is _MISSING:
                del module._evoke_joint_context
            else:
                module._evoke_joint_context = context
        del root._evoke_joint_active
