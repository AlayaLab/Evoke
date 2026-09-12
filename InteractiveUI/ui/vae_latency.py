import time
import json


def _graph_signature(vae, video, optimizations=None):
    import torch
    tensors = tuple((name, id(tensor), tensor.data_ptr(), tensor._version,
                     tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                    for name, tensor in list(vae.named_parameters()) + list(vae.named_buffers()))
    signature = (tuple(video.shape), tuple(video.stride()), str(video.dtype), str(video.device),
            bool(vae.training), bool(getattr(vae, 'use_tiling', False)),
            bool(getattr(vae, 'use_slicing', False)),
            tuple((name, getattr(vae, name, None)) for name in (
                'tile_sample_min_height', 'tile_sample_min_width',
                'tile_sample_stride_height', 'tile_sample_stride_width')),
            json.dumps(dict(vae.config), sort_keys=True, default=str),
            torch.is_autocast_enabled(), str(torch.get_autocast_gpu_dtype()),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(), tensors)
    if optimizations is None:
        return signature
    from .vae_encoder_runtime import compute_signature
    return signature, compute_signature(optimizations)


def _eager_encode(vae, video, optimizations=None, stats=None, verify_operators=False):
    from .vae_encoder_runtime import COMPUTE_DEFAULTS, normalize_options, encoder_mode
    values = normalize_options(optimizations)
    active = any(values[key] for key in ('skipZeroPad', 'fuseCausalInput', 'fuseAffine', 'fuseJointCache'))
    if not active:
        return vae.encode(video).latent_dist.parameters
    with encoder_mode(vae, values, verify=verify_operators) as scope:
        parameters = vae.encode(video).latent_dist.parameters
    if stats is not None:
        stats['operatorScope'] = scope
    return parameters


def _graph_encode(vae, video, stats, optimizations=None):


    import torch
    from .vae_encoder_runtime import normalize_options, compute_signature, encoder_mode
    values = normalize_options(optimizations)


    allow_initialization = optimizations is None or values['allowGraphInitialization']
    signature = _graph_signature(vae, video, values)
    key = compute_signature(values)
    pool = getattr(vae, '_evoke_encoder_graphs', None)
    if pool is None:
        pool = {}
        vae._evoke_encoder_graphs = pool
    resident = pool.get(key)
    if resident is not None and resident['signature'] != signature:
        del pool[key]
        resident = None
        stats['graphInvalidated'] = True
    stats['graphInitialized'] = False
    if resident is None and (not allow_initialization or video.device.type != 'cuda'):
        stats.update(graphDeferred=True, graphUsed=False,
                     graphDeferReason='not-cuda' if video.device.type != 'cuda' else 'outside-safe-first-window')
        return _eager_encode(vae, video, values, stats)
    if resident is None:
        stats['graphInitialized'] = True
        baseline_key = compute_signature({**values, 'fuseJointCache': False}) if values['fuseJointCache'] else compute_signature()


        if len(pool) >= 2:
            victim = next((item for item in pool if item != baseline_key), next(iter(pool)))
            del pool[victim]
        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated(video.device)
        reserved_before = torch.cuda.memory_reserved(video.device)
        current = torch.cuda.current_stream(video.device)
        capture_stream = torch.cuda.Stream(device=video.device)
        capture_stream.wait_stream(current)


        with encoder_mode(vae, values, verify=False) as scope:
            with torch.cuda.stream(capture_stream):
                static_input = torch.empty_strided(video.shape, video.stride(), dtype=video.dtype, device=video.device)
                static_input.copy_(video)
                for _ in range(3):
                    vae.encode(static_input).latent_dist.parameters
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode='thread_local'):
                static_output = vae.encode(static_input).latent_dist.parameters
            capture_stream.synchronize()
        stats['graphBuildOperatorScope'] = scope
        stats['graphBuildEncoderCalls'] = 4
        stats['graphBuildScopeMeaning'] = 'cumulative over three warmups plus one capture; not one encode'
        resident = {'signature': signature, 'input': static_input, 'output': static_output,
                    'graph': graph, 'captureStream': capture_stream}
        pool[key] = resident
        stats['graphInitializationHostSeconds'] = time.perf_counter() - start
        stats['graphAllocatedDeltaBytes'] = torch.cuda.memory_allocated(video.device) - allocated_before
        stats['graphReservedDeltaBytes'] = torch.cuda.memory_reserved(video.device) - reserved_before
    resident['input'].copy_(video)
    resident['graph'].replay()

    output = resident['output'].clone()

    vae.clear_cache()
    stats['graphUsed'] = True
    return output


def _cache_state(vae):
    result = {}
    for name in ('_feat_map', '_enc_feat_map'):
        cache = getattr(vae, name, None)
        result[name] = None if cache is None else [
            None if value is None else type(value).__name__ for value in cache
        ]
    for name in ('_conv_idx', '_enc_conv_idx'):
        value = getattr(vae, name, None)
        result[name] = None if value is None else list(value)
    return result


def verify_moments_sampling(old_first, old_full, new_first, new_full):


    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    if (old_first.shape, old_first.dtype, old_first.device) != (
        new_first.shape, new_first.dtype, new_first.device
    ):
        raise AssertionError('Discarded posterior sample signature changed')
    records = []
    devices = [torch.device('cpu')]
    if old_full.device.type == 'cuda':
        devices.append(old_full.device)
    for device in devices:
        for use_list in (False, True):
            count = old_full.shape[0] if use_list else 1
            old_generators = [torch.Generator(device=device).manual_seed(90403 + i) for i in range(count)]
            new_generators = [torch.Generator(device=device).manual_seed(90403 + i) for i in range(count)]
            old_generator = old_generators if use_list else old_generators[0]
            new_generator = new_generators if use_list else new_generators[0]
            DiagonalGaussianDistribution(old_first).sample(generator=old_generator)
            old_sample = DiagonalGaussianDistribution(old_full).sample(generator=old_generator)
            DiagonalGaussianDistribution(new_first).sample(generator=new_generator)
            new_sample = DiagonalGaussianDistribution(new_full).sample(generator=new_generator)
            sample_equal = (_bits_equal(old_sample, new_sample)
                            if old_sample.dtype == torch.float32 else torch.equal(old_sample, new_sample))
            rng_equal = all(torch.equal(a.get_state(), b.get_state()) for a, b in zip(old_generators, new_generators))
            records.append({'generatorDevice': str(device), 'generatorList': use_list,
                            'usedSampleExact': sample_equal, 'rngStateExact': rng_equal})
            if not sample_equal or not rng_equal:
                raise AssertionError(f'Warp posterior sampling audit failed: {records[-1]}')
    return records


def _bits_equal(old, new):
    import torch
    return (old.dtype == new.dtype and old.shape == new.shape
            and torch.equal(old.contiguous().view(torch.int32), new.contiguous().view(torch.int32)))


def _tensor_spec(value):
    return {'shape': list(value.shape), 'stride': list(value.stride()),
            'dtype': str(value.dtype), 'device': str(value.device)}


def _rng_state(device):
    import torch
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _rng_equal(left, right):
    import torch
    return all((left[key] is None and right[key] is None)
               or (left[key] is not None and right[key] is not None and torch.equal(left[key], right[key]))
               for key in ('cpu', 'cuda'))


def _check_full(vae, reference, candidate, reference_cache, reference_rng, device, label):
    import torch
    current_cache = _cache_state(vae)
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise AssertionError('VAE encoder optimization audit requires FP32 posterior parameters')
    checks = {'fullParametersExact': _bits_equal(reference, candidate),
              'fullLayoutExact': _tensor_spec(reference) == _tensor_spec(candidate),
              'cacheStateExact': reference_cache == current_cache,
              'cacheCleared': all(value is None for name in ('_feat_map', '_enc_feat_map')
                                  for value in (current_cache[name] or [])),
              'encoderGlobalRngExact': _rng_equal(reference_rng, _rng_state(device))}
    if not all(checks.values()):
        raise AssertionError(f'Warp encoder {label} strict audit failed: {checks}')
    return {**checks, 'fullLayout': _tensor_spec(candidate), 'cacheState': current_cache}


def _benchmark_encoder(vae, frames, reference, cache, rng, values):

    import statistics
    import torch
    from .vae_encoder_runtime import COMPUTE_DEFAULTS

    baseline = ({**values, 'fuseJointCache': False} if values['fuseJointCache']
                else {**values, **COMPUTE_DEFAULTS})
    modes = ('oldGraph', 'candidateGraph', 'candidateEager')
    records = []
    def execute(mode):
        detail = {}
        if mode == 'oldGraph':
            output = _graph_encode(vae, frames, detail, baseline)
        elif mode == 'candidateGraph':
            output = _graph_encode(vae, frames, detail, values)
        else:
            output = _eager_encode(vae, frames, values, detail)
        return output, detail
    initialization = {}
    for mode in modes:
        for index in range(values['benchmarkWarmup']):
            output, detail = execute(mode)
            _check_full(vae, reference, output, cache, rng, frames.device, f'benchmark-warmup-{mode}')
            if index == 0:
                initialization[mode] = detail
    for repeat in range(values['benchmarkRepeats']):
        for order_name, order in (('forward', modes), ('reverse', tuple(reversed(modes)))):
            for mode in order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(frames.device))
                started = time.perf_counter()
                output, detail = execute(mode)
                end.record(torch.cuda.current_stream(frames.device))
                end.synchronize()
                host_ms = (time.perf_counter() - started) * 1000
                _check_full(vae, reference, output, cache, rng, frames.device, f'benchmark-{mode}')
                if detail.get('graphInitialized'):
                    raise AssertionError('Cold encoder graph entered timed benchmark samples')
                records.append({'mode': mode, 'repeat': repeat, 'order': order_name,
                                'cudaMilliseconds': begin.elapsed_time(end), 'hostMilliseconds': host_ms,
                                'strictExact': True, 'graphUsed': detail.get('graphUsed', False)})
    summary = {}
    for mode in modes:
        rows = [row for row in records if row['mode'] == mode]
        summary[mode] = {'samples': len(rows),
                         'cudaMedianMilliseconds': statistics.median(row['cudaMilliseconds'] for row in rows),
                         'hostMedianMilliseconds': statistics.median(row['hostMilliseconds'] for row in rows)}
    return {'executed': True, 'warmupPerMode': values['benchmarkWarmup'],
            'baselineOptions': {key: baseline[key] for key in COMPUTE_DEFAULTS},
            'initialization': initialization, 'samples': records, 'summary': summary,
            'note': 'Same video; original eager reference; every output/layout/cache/RNG audited. Cold build excluded. Diagnostic only.'}


def encode_warp_moments(vae, video, reuse_first=False, verify=False, graph=False, optimizations=None):


    from .vae_encoder_runtime import encoder_access, normalize_options
    values = normalize_options(optimizations)
    with encoder_access(vae):
        return _encode_warp_moments_locked(vae, video, reuse_first, verify, graph, values,
                                           legacy_graph_initialization=optimizations is None)


def _encode_warp_moments_locked(vae, video, reuse_first, verify, graph, values,
                                legacy_graph_initialization=False):
    import torch
    from .vae_encoder_runtime import COMPUTE_DEFAULTS

    if graph and not reuse_first:
        raise ValueError('Encoder CUDA graph requires reuse_first=True')
    stats = {'reuseFirst': bool(reuse_first), 'verified': bool(verify),
             'graphRequested': bool(graph), 'graphUsed': False, 'events': {},
             'encoderOptimizations': {key: values[key] for key in COMPUTE_DEFAULTS}}

    def encode(frames, label, use_graph=False, candidate=False):
        cuda = frames.device.type == 'cuda'
        begin = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        host_begin = time.perf_counter()
        if cuda:
            begin.record(torch.cuda.current_stream(frames.device))
        if use_graph:
            parameters = _graph_encode(vae, frames, stats,
                                       None if legacy_graph_initialization else values)
        elif candidate:
            parameters = _eager_encode(vae, frames, values, stats, verify_operators=verify)
        else:


            parameters = vae.encode(frames).latent_dist.parameters
        if cuda:
            end.record(torch.cuda.current_stream(frames.device))
            stats['events'][label] = (begin, end)
        stats[label + 'HostLaunchSeconds'] = time.perf_counter() - host_begin
        return parameters

    if not verify:
        first = None if reuse_first else encode(video[:, :, :1], 'firstEncode', candidate=True)
        full = encode(video[:, :, -33:], 'fullEncode', use_graph=graph, candidate=True)
        if reuse_first:
            first = full[:, :, :1]
        if values['benchmark']:
            stats['benchmark'] = {'executed': False, 'reason': 'requires-verification'}
        return first, full, stats

    rng_before = _rng_state(video.device)
    old_first = encode(video[:, :, :1], 'firstEncode')
    old_full = encode(video[:, :, -33:], 'fullEncode')
    old_cache = _cache_state(vae)
    if not _rng_equal(rng_before, _rng_state(video.device)):
        raise AssertionError('Original deterministic VAE encoder changed global RNG during audit')
    new_first = None if reuse_first else encode(video[:, :, :1], 'auditFirstEncode', candidate=True)
    new_full = encode(video[:, :, -33:], 'auditFullEncode', use_graph=graph, candidate=True)
    if reuse_first:
        new_first = new_full[:, :, :1]
    stats['audit'] = _check_full(vae, old_full, new_full, old_cache, rng_before, video.device, 'candidate')
    stats['audit'].update({
        'discardedFirstParametersExact': _bits_equal(old_first, new_first),
        'discardedFirstLayoutExact': _tensor_spec(old_first) == _tensor_spec(new_first),
        'discardedFirstMaxAbs': float((old_first.float() - new_first.float()).abs().max().item()),
    })


    stats['audit']['sampling'] = verify_moments_sampling(old_first, old_full, new_first, new_full)
    if values['benchmark']:
        if values['allowGraphInitialization'] and video.device.type == 'cuda':
            stats['benchmark'] = _benchmark_encoder(vae, video[:, :, -33:], old_full,
                                                     old_cache, rng_before, values)
        else:
            stats['benchmark'] = {'executed': False, 'reason': 'outside-safe-first-window-or-not-cuda'}
    if not _rng_equal(rng_before, _rng_state(video.device)):
        raise AssertionError('Deterministic warp encoder audit changed global CPU/CUDA RNG')
    return new_first, new_full, stats


def resolve_encode_timing(stats):

    result = {key: value for key, value in stats.items() if key != 'events'}
    for label, (begin, end) in stats['events'].items():
        result[label + 'CudaSeconds'] = begin.elapsed_time(end) / 1000
    return result
