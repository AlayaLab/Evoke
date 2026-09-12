from contextlib import ExitStack
import time
import torch


def clone_cache(cache):
    return [v.clone() if isinstance(v, torch.Tensor) else v for v in cache]


def tensor_exact(a, b):
    if (a.shape, a.stride(), a.dtype, a.device) != (b.shape, b.stride(), b.dtype, b.device):
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def cache_exact(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor) or not tensor_exact(x, y):
                return False
        elif isinstance(y, torch.Tensor) or x != y:
            return False
    return True


def step(vae, x, cache, first_chunk):
    index = [0]
    output = vae.decoder(x, feat_cache=cache, feat_idx=index, first_chunk=first_chunk)
    return output, index[0]


class DecoderProfile:

    def __init__(self, vae):
        self.vae = vae
        self.handles = []
        self.events = []

    def __enter__(self):
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm, WanAttentionBlock
        classes = ((WanCausalConv3d, 'causalConvWithPrepare'),
                   (WanRMS_norm, 'normalizeWithAffine'),
                   (WanAttentionBlock, 'spatialAttention'),
                   (torch.nn.SiLU, 'silu'), (torch.nn.Conv2d, 'conv2d'))
        for name, module in self.vae.decoder.named_modules():
            kind = next((kind for cls, kind in classes if isinstance(module, cls)), None)
            if kind is None:
                continue
            pending = []
            def before(module, inputs, pending=pending):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                pending.append((start, end))
            def after(module, inputs, output, name=name, kind=kind, pending=pending):
                start, end = pending.pop()
                end.record()
                self.events.append((name, kind, start, end))
            self.handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def result(self):
        rows = [{'name': n, 'kind': k, 'cudaSeconds': a.elapsed_time(b)/1000}
                for n, k, a, b in self.events]
        return {'modules': rows,
                'note': 'Nested CUDA stream ranges include dispatch gaps; attention contains norms/convs.'}


def _run(vae, x, source, first_chunk, options, verify_layers=False, profile=False):
    options = options or {}
    if options.get('fuseCacheWrites') and options.get('fuseJointCache'):
        raise ValueError('Select one cache optimization')
    records = {}
    with ExitStack() as stack:
        if profile or options.get('skipZeroPad') or options.get('fuseCausalInput'):
            from .vae_memory import memory_mode
            records['memory'] = stack.enter_context(memory_mode(vae, {
                'skipZeroPad': bool(options.get('skipZeroPad')),
                'fuseCausalInput': bool(options.get('fuseCausalInput')),
                'blockSize': int(options.get('memoryBlockSize',256)),
                'verify': verify_layers, 'profile': profile,
            }))
        if options.get('fuseAffine'):
            from .vae_affine import affine_mode
            records['affine'] = stack.enter_context(affine_mode(vae, enabled=True, verify=verify_layers,
                fuse_division=bool(options.get('fuseNormDivision'))))
        if options.get('fuseCacheWrites'):
            from .vae_cache import cache_mode
            records['cacheWrites'] = stack.enter_context(cache_mode(
                vae, enabled=True, verify=verify_layers, profile=profile))
        if options.get('fuseJointCache'):
            from .vae_joint_cache import joint_cache_mode
            records['jointCache'] = stack.enter_context(joint_cache_mode(
                vae, enabled=True, verify=verify_layers,
                block_size=options.get('jointCacheBlockSize', options.get('memoryBlockSize', 256)),
                planes=options.get('jointCachePlanes', False), warps=options.get('jointCacheWarps', 4)))
        measurement = stack.enter_context(DecoderProfile(vae)) if profile else None
        output, count = step(vae, x, source, first_chunk)
    if profile:
        torch.cuda.current_stream(x.device).synchronize()
        records['profile'] = measurement.result()


    if 'memory' in records and profile:
        from .vae_memory import resolve_memory_timing
        records['memory'] = resolve_memory_timing(records['memory'])
    if 'cacheWrites' in records and profile:
        from .vae_cache import resolve_cache_timing
        records['cacheWrites'] = resolve_cache_timing(records['cacheWrites'])
    return output, count, records


def decode_slice(vae, x, first_chunk, options, verify=False, profile=False, benchmark=False):
    source = vae._feat_map
    stats = {'enabled': {key: bool(options.get(key)) for key in ('skipZeroPad','fuseCausalInput','fuseAffine','fuseNormDivision','fuseCacheWrites','fuseJointCache')},
             'verified': False, 'profiled': bool(profile), 'benchmark': None}
    rng_cpu = torch.get_rng_state() if verify or benchmark else None
    rng_cuda = torch.cuda.get_rng_state(x.device) if verify or benchmark else None
    if verify or benchmark:
        reference_cache = clone_cache(source)
        expected, expected_count = step(vae, x, reference_cache, first_chunk)
        reference_rng_cpu = torch.get_rng_state()
        reference_rng_cuda = torch.cuda.get_rng_state(x.device)
        if not torch.equal(rng_cpu, reference_rng_cpu) or not torch.equal(rng_cuda, reference_rng_cuda):
            raise AssertionError('VAE reference decoder unexpectedly consumed RNG')
    if benchmark:
        stats['benchmark'] = benchmark_slice(vae, x, source, first_chunk, expected, reference_cache, expected_count,
                                             current_options=options)
    output, count, detail = _run(vae, x, source, first_chunk, options, verify_layers=verify, profile=profile)
    vae._feat_map = source
    vae._conv_idx = [count]
    stats.update(detail)
    if verify or benchmark:
        checks = {'outputBitwiseExact': tensor_exact(expected, output),
                  'cacheBitwiseExact': cache_exact(reference_cache, source),
                  'cacheCountExact': count == expected_count,
                  'cpuRngExact': torch.equal(rng_cpu, torch.get_rng_state()),
                  'cudaRngExact': torch.equal(rng_cuda, torch.cuda.get_rng_state(x.device))}
        stats.update(verified=True, audit=checks)
        if not all(checks.values()):
            stats['maxAbs'] = (expected-output).abs().max().item()
            raise AssertionError(f'Decoder optimization audit failed: {stats}')
    return output, stats


def benchmark_slice(vae, x, source, first_chunk, expected, expected_cache, expected_count, current_options=None):

    variants = [('baseline', {}), ('zeroPad', {'skipZeroPad':True}),
                ('causalInput', {'fuseCausalInput':True}),
                ('memory', {'skipZeroPad':True,'fuseCausalInput':True}),
                ('affine', {'fuseAffine':True}),
                ('all', {'skipZeroPad':True,'fuseCausalInput':True,'fuseAffine':True}),
                ('all512', {'skipZeroPad':True,'fuseCausalInput':True,'fuseAffine':True,'memoryBlockSize':512}),
                ('all1024', {'skipZeroPad':True,'fuseCausalInput':True,'fuseAffine':True,'memoryBlockSize':1024}),
                ('allDivision', {'skipZeroPad':True,'fuseCausalInput':True,'fuseAffine':True,'fuseNormDivision':True}),
                ('allDivision1024', {'skipZeroPad':True,'fuseCausalInput':True,'fuseAffine':True,'fuseNormDivision':True,'memoryBlockSize':1024})]
    cache_benchmark = bool((current_options or {}).get('fuseCacheWrites') or (current_options or {}).get('fuseJointCache'))
    if cache_benchmark:
        production = {key: (current_options or {}).get(key, 256 if key == 'memoryBlockSize' else False) for key in
                      ('skipZeroPad', 'fuseCausalInput', 'fuseAffine', 'fuseNormDivision', 'memoryBlockSize')}
        cache_option = 'fuseJointCache' if (current_options or {}).get('fuseJointCache') else 'fuseCacheWrites'
        variants = [('production', production), ('jointCache' if cache_option == 'fuseJointCache' else 'cacheWrites',
                                                  {**production, cache_option: True})]
        if cache_option == 'fuseJointCache':
            variants = [('production', production)] + [
                (f'joint-{planes}-{block}-{warps}', {**production, 'fuseJointCache': True,
                    'jointCachePlanes': planes, 'jointCacheBlockSize': block, 'jointCacheWarps': warps})
                for planes, block, warps in ((False, 512, 4), (False, 1024, 8),
                    (True, 512, 4), (True, 1024, 4), (True, 2048, 4), (True, 1024, 8))]
    rows = []
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(x.device)
    warmup_count = len(variants) * (2 if cache_benchmark else 1)
    order = (list(range(len(variants))) * 2 +
             (list(range(len(variants))) + list(reversed(range(len(variants))))) * 3) if cache_benchmark else (
             list(range(len(variants))) + list(range(len(variants))) + list(reversed(range(len(variants)))))
    for position, index in enumerate(order):
        name, options = variants[index]
        cache = clone_cache(source)
        torch.cuda.current_stream(x.device).synchronize()
        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        host_begin = time.perf_counter(); begin.record()
        output, count, _ = _run(vae, x, cache, first_chunk, options)
        end.record(); end.synchronize()
        elapsed = time.perf_counter()-host_begin
        checks = (tensor_exact(output,expected), cache_exact(cache,expected_cache), count==expected_count,
                  torch.equal(rng_cpu,torch.get_rng_state()),torch.equal(rng_cuda,torch.cuda.get_rng_state(x.device)))
        if not all(checks) and name in ('baseline', 'production'):
            raise AssertionError(f'Decoder microbenchmark exact check failed for {name}: {checks}; maxAbs={(output-expected).abs().max().item()}')
        rows.append({'variant':name,'warmup':position<warmup_count,'cudaSeconds':begin.elapsed_time(end)/1000,
                     'hostSeconds':elapsed,'bitwiseExact':all(checks),
                     'checks':dict(zip(('output','cache','count','cpuRng','cudaRng'),checks)),
                     'maxAbs':0.0 if all(checks) else (output-expected).abs().max().item()})
    return {'samples':rows,
            'backend': {'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
                        'cudnnBenchmark': torch.backends.cudnn.benchmark,
                        'cudnnDeterministic': torch.backends.cudnn.deterministic,
                        'cudnnAllowTf32': torch.backends.cudnn.allow_tf32,
                        'matmulAllowTf32': torch.backends.cuda.matmul.allow_tf32},
            'note':'Same input/cache, comparison and cache snapshot excluded, module patch dispatch included.'}


def probe_group(vae, x):

    source = vae._feat_map; old_index = vae._conv_idx
    cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state(x.device)
    try:
        before=clone_cache(source); begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();outputs=[]
        for i in range(x.shape[2]):
            output,count=step(vae,x[:,:,i:i+1],before,False);outputs.append(output)
        end.record();end.synchronize();one=begin.elapsed_time(end)/1000
        expected=torch.cat(outputs,dim=2)
        after=clone_cache(source);begin.record();actual,count2=step(vae,x,after,False);end.record();end.synchronize();group=begin.elapsed_time(end)/1000
        return {'latentCount':x.shape[2],'usedForGeneration':False,
                'outputBitwiseExact':tensor_exact(expected,actual),'cacheBitwiseExact':cache_exact(before,after),
                'cacheCountExact':count==count2,'maxAbs':(expected-actual).abs().max().item(),
                'singleCudaSeconds':one,'groupCudaSeconds':group,
                'rngExact':torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(x.device))}
    finally:
        vae._feat_map=source;vae._conv_idx=old_index
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,x.device)
