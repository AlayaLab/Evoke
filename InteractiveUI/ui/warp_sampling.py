from __future__ import annotations

import time


def _inputs(valid_sources, count, device):
    import torch
    if type(count) is not int or count < 0:
        raise ValueError('Sample count must be a nonnegative integer')
    sources = tuple(valid_sources)
    target = torch.device(device)
    if target.type == 'cuda' and target.index is None:
        target = torch.device('cuda', torch.cuda.current_device())
    for source in sources:
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError('Each valid source must have shape N,3')
    return sources, target


def reference_points(valid_sources, count=2000, device='cuda', generator=None):

    import torch
    sources, target = _inputs(valid_sources, count, device)
    output = torch.full((len(sources), count, 3), 1e6, device=target)
    for row, source in enumerate(sources):
        if source.shape[0] > 0:
            sampled = source[torch.randint(0, source.shape[0], (count,),
                                          device=target, generator=generator)]
            output[row] = sampled
    return output


def _unsupported(sources, count, target):
    import torch
    if target.type != 'cuda':
        return 'CUDA required'
    if torch.get_default_dtype() != torch.float32:
        return 'default output dtype must be FP32'
    if count == 0:
        return 'zero count uses original path'
    if count % 4:
        return 'count must be a multiple of four for aligned randint output rows'
    if any(source.device != target or source.dtype != torch.float32
           or not source.is_contiguous() or (source.shape[0] > 0 and source.stride() != (3, 1))
           for source in sources):
        return 'all sources must be same-device contiguous FP32 N,3'
    return None


def sample_points(valid_sources, count=2000, device='cuda', generator=None, *, fallback=True):


    import torch
    sources, target = _inputs(valid_sources, count, device)
    reason = _unsupported(sources, count, target)
    if reason:
        if not fallback:
            raise ValueError(f'Fused warp sampling unsupported: {reason}')
        return reference_points(sources, count, target, generator)
    if not sources:
        return torch.empty((0, count, 3), dtype=torch.float32, device=target)

    from .warp_sampling_triton import gather
    with torch.cuda.device(target):
        indices = torch.empty((len(sources), count), dtype=torch.int64, device=target)
        output = torch.empty((len(sources), count, 3), dtype=torch.float32, device=target)


        table = torch.tensor([[source.data_ptr() for source in sources],
                              [source.shape[0] for source in sources]],
                             dtype=torch.int64, device=target)
        stream = torch.cuda.current_stream(target)
        for row, source in enumerate(sources):
            source.record_stream(stream)
            if source.shape[0] > 0:
                torch.randint(0, source.shape[0], (count,), device=target,
                              generator=generator, out=indices[row])
        gather(table[0], table[1], indices, output, count)
        return output


def audit_benchmark(valid_sources, count=2000, device='cuda', generator=None, *, repeats=2):


    import torch
    sources, target = _inputs(valid_sources, count, device)
    reason = _unsupported(sources, count, target)
    if reason:
        raise ValueError(f'CUDA audit requires fused path: {reason}')
    if type(repeats) is not int or not 1 <= repeats <= 8:
        raise ValueError('Benchmark repeats must be 1–8')
    if not sources:
        raise ValueError('Benchmark needs at least one source')

    def snapshot():
        return (torch.get_rng_state().clone(), torch.cuda.get_rng_state(target).clone(),
                generator.get_state().clone() if generator is not None else None)

    def restore(state):
        torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1], target)
        if generator is not None:
            generator.set_state(state[2])

    def states_equal(left, right):
        return all((a is None and b is None) or
                   (a is not None and b is not None and torch.equal(a, b))
                   for a, b in zip(left, right))

    def bits_equal(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype and
                left.stride() == right.stride() and
                torch.equal(left.view(torch.int32), right.view(torch.int32)))

    original = snapshot()
    reference = None
    functions = {'original': reference_points, 'fused': sample_points}
    samples = []
    try:
        torch.cuda.synchronize(target)

        for name in ('original', 'fused'):
            restore(original)
            result = functions[name](sources, count, target, generator)
            torch.cuda.synchronize(target)
            resulting_state = snapshot()
            if reference is None:
                reference, expected_state = result, resulting_state
            elif not bits_equal(reference, result) or not states_equal(expected_state, resulting_state):
                raise AssertionError('Fused sampling warmup changed P_all bits/layout or RNG state')
        for _ in range(repeats):
            for name in ('original', 'fused', 'fused', 'original'):
                restore(original)
                torch.cuda.synchronize(target)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(target))
                started = time.perf_counter()
                result = functions[name](sources, count, target, generator)
                end.record(torch.cuda.current_stream(target)); end.synchronize()
                wall = time.perf_counter() - started
                exact = bits_equal(reference, result)
                rng_exact = states_equal(expected_state, snapshot())
                if not exact or not rng_exact:
                    raise AssertionError('Fused sampling ABBA changed P_all bits/layout or RNG state')
                samples.append({'variant': name, 'wallSeconds': wall,
                                'cudaSeconds': begin.elapsed_time(end) / 1000,
                                'bitsExact': exact, 'rngExact': rng_exact})
        return {'sourceCount': len(sources), 'nonemptySources': sum(s.shape[0] > 0 for s in sources),
                'count': count, 'generator': 'explicit' if generator is not None else 'global',
                'order': 'ABBA', 'warmupExact': True, 'samples': samples,
                'includes': 'randint, allocations, pointer metadata transfer and gather/copy',
                'excludes': 'geometry preparation, comparisons, RNG reset, render/encoder'}
    finally:

        try:
            torch.cuda.synchronize(target)
        finally:
            restore(original)


def synthetic_sources(valid_sources, source_count=4800):


    if type(source_count) is not int or not 1 <= source_count <= 20000:
        raise ValueError('source_count must be 1–20000')
    sources = tuple(valid_sources)
    if not sources:
        raise ValueError('At least one existing source buffer is required')
    return tuple(sources[i % len(sources)] for i in range(source_count))
