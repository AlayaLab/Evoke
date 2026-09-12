import time


def _spec(cache):
    import torch
    return tuple((tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
                 if isinstance(value, torch.Tensor) else value for value in cache)


def _clone_cache(cache):
    import torch
    return [value.clone() if isinstance(value, torch.Tensor) else value for value in cache]


def _ready(cache, count):
    import torch
    return (0 < count <= len(cache)
            and all(isinstance(value, torch.Tensor) for value in cache[:count])
            and all(value is None for value in cache[count:]))


def _step(vae, x, cache, first_chunk=False):
    index = [0]
    output = vae.decoder(x, feat_cache=cache, feat_idx=index, first_chunk=first_chunk)
    return output, index[0]


def _build(vae, x, source_cache, count, signature, stats):
    import torch
    start = time.perf_counter()
    allocated_before = torch.cuda.memory_allocated(x.device)
    reserved_before = torch.cuda.memory_reserved(x.device)
    current = torch.cuda.current_stream(x.device)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        static_input = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
        static_input.copy_(x)
        buffers = _clone_cache(source_cache)

        def body():
            updated = list(buffers)
            output, used = _step(vae, static_input, updated)
            if used != count or _spec(updated) != _spec(buffers):
                raise AssertionError('Decoder cache shape/slot count changed during graph capture')


            for target, value in zip(buffers[:count], updated[:count]):
                target.copy_(value)
            return output

        for _ in range(3):
            body()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
        static_output = body()
    stream.synchronize()


    resident = {'signature': signature, 'input': static_input, 'output': static_output,
                'buffers': buffers, 'count': count, 'graph': graph, 'captureStream': stream}
    vae._evoke_decoder_graph = resident
    stats.update(graphInitialized=True,
                 graphInitializationHostSeconds=time.perf_counter() - start,
                 graphAllocatedDeltaBytes=torch.cuda.memory_allocated(x.device) - allocated_before,
                 graphReservedDeltaBytes=torch.cuda.memory_reserved(x.device) - reserved_before,
                 cacheBufferBytes=sum(value.numel() * value.element_size() for value in buffers[:count]))
    return resident


def decode_slice(vae, x, first_chunk, verify=False, enabled=False):


    import torch
    from .vae_latency import _graph_signature

    stats = {'graphRequested': bool(enabled), 'graphUsed': False,
             'graphInitialized': False, 'verified': False}
    source = vae._feat_map
    count = getattr(vae, '_evoke_decoder_graph_active_count', 0)
    stable = getattr(vae, '_evoke_decoder_graph_stable', False)
    reason = None
    if not enabled:
        reason = 'disabled'
    elif first_chunk:
        reason = 'first-chunk'
    elif x.device.type != 'cuda' or x.shape[2] != 1:
        reason = 'unsupported-input'
    elif any(module.training for module in vae.decoder.modules()):
        reason = 'training'
    elif not _ready(source, count):
        reason = 'cold-cache'
    elif not stable:
        reason = 'cache-shape-transition'
    if reason is not None:
        before = _spec(source)
        output, count = _step(vae, x, source, first_chunk=first_chunk)
        vae._conv_idx = [count]
        vae._evoke_decoder_graph_active_count = count
        vae._evoke_decoder_graph_stable = before == _spec(source) and _ready(source, count)
        stats.update(eagerReason=reason, activeCacheSlots=count,
                     cacheShapeStable=vae._evoke_decoder_graph_stable)
        return output, stats

    signature = (_graph_signature(vae, x), _spec(source), count)
    resident = getattr(vae, '_evoke_decoder_graph', None)
    if resident is not None and resident['signature'] != signature:

        vae._evoke_decoder_graph = None
        resident = None
        stats['graphInvalidated'] = True
    audit = verify or resident is None
    if audit:
        rng_cuda = torch.cuda.get_rng_state(x.device)
        rng_cpu = torch.get_rng_state()
        expected_cache = _clone_cache(source)
        expected, expected_count = _step(vae, x, expected_cache)
    if resident is None:
        resident = _build(vae, x, source, count, signature, stats)

    copied = 0
    for destination, value in zip(resident['buffers'][:count], source[:count]):
        if destination is not value:
            destination.copy_(value)
            copied += 1
    resident['input'].copy_(x)
    resident['graph'].replay()

    output = resident['output'].clone()
    vae._feat_map = resident['buffers']
    vae._conv_idx = [count]
    stats.update(graphUsed=True, activeCacheSlots=count, cacheReloadedSlots=copied,
                 outputCloneBytes=output.numel() * output.element_size())
    if audit:
        output_exact = torch.equal(expected, output)
        slots_exact = [torch.equal(a, b) for a, b in zip(expected_cache[:count], vae._feat_map[:count])]
        state_exact = (expected_count == count and _spec(expected_cache) == _spec(vae._feat_map)
                       and all(slots_exact))
        rng_exact = (torch.equal(rng_cuda, torch.cuda.get_rng_state(x.device))
                     and torch.equal(rng_cpu, torch.get_rng_state()))
        stats.update(verified=True, audit={'outputExact': output_exact, 'cacheExact': state_exact,
                     'cacheSlotsExact': slots_exact, 'rngStateExact': rng_exact})
        if not output_exact or not state_exact or not rng_exact:
            raise AssertionError(f'Decoder graph exact-output audit failed: {stats["audit"]}')
    return output, stats
