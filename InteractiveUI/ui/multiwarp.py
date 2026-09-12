from concurrent.futures import ThreadPoolExecutor
import time
import json
import torch
from evoke.modules.geometric_state.da3_cloud import _render_backward_multisrc_zbuf as render_zbuf, frame_signature


class MultiWarp:
    def __init__(self, devices):
        self.devices = tuple(torch.device('cuda', int(d)) for d in devices)
        self.executors = [ThreadPoolExecutor(1, thread_name_prefix=f'warp-{d.index}') for d in self.devices]
        self.streams = [torch.cuda.Stream(device=d) for d in self.devices]
        self.replicas = [{} for _ in self.devices]
        self.world_caches = [{} for _ in self.devices]
        self.geometry = {}; self.last_stats = {}; self.audit_path = None; self.calls = 0
        self.optimizations = {}
        self.sampling_benchmark_path = None
        self.covis_benchmark_path = None
        self.benchmark_optimizations = False
        self.peer_access = {f'{a.index}->{b.index}':torch.cuda.can_device_access_peer(a,b) for a in self.devices for b in self.devices if a!=b}

    def clear(self):

        self.geometry.clear(); self.calls = 0; self.last_stats = {}
        for replica in self.replicas + self.world_caches: replica.clear()

    def __call__(self, store, ids, poses, K, height, width, **kwargs):
        ids = sorted(ids); source = poses.device
        if not ids:return render_zbuf(store, ids, poses, K, height, width, **kwargs)
        started = time.perf_counter()
        render_kwargs = dict(kwargs,
            _reuse_target_inverse=bool(self.optimizations.get('reuseTargetInverse', False)),
            _device_fusion_gate=bool(self.optimizations.get('deviceFusionGate', False)),
            _static_splat=bool(self.optimizations.get('staticSplat', False)),
            _reuse_source_index=bool(self.optimizations.get('reuseSourceIndex', False)),
            _fused_source_sampling=bool(self.optimizations.get('fusedSourceSampling', False)),
            _reuse_full_source_rows=bool(self.optimizations.get('reuseFullSourceRows', False)),
            _fused_covis=bool(self.optimizations.get('fusedCovis', False)))
        self.calls += 1
        benchmark = None
        if self.calls in {8,32,95}:
            if self.optimizations.get('benchmarkReuseSourceIndex', False):
                benchmark = 'sourceIndex'
            elif self.benchmark_optimizations:
                benchmark = 'all'
        audit = self.audit_path is not None and self.calls in {1,8,32,55,95}
        rng_before = torch.cuda.get_rng_state(source) if audit else None
        begin = torch.cuda.Event(enable_timing=True); prepared_event = torch.cuda.Event(enable_timing=True)
        begin.record()
        prepared = render_zbuf(store, ids, poses, K, height, width, **dict(render_kwargs,_prepare_only=True,_geometry_cache=self.geometry,_scale_batch=True))
        prepared_event.record()
        tasks = []
        for i, executor in enumerate(self.executors):
            lo = len(poses)*i//len(self.devices); hi = len(poses)*(i+1)//len(self.devices)
            if hi > lo:
                shard_prepared={**prepared,'covis':prepared['covis'][lo:hi],'orders':prepared['orders'][lo:hi]}
                tasks.append(executor.submit(self._shard, i, store, ids, shard_prepared, poses[lo:hi], K,
                                             height, width, render_kwargs, prepared_event, source, benchmark))

        results = []; error = None
        for task in tasks:
            try: results.append(task.result())
            except BaseException as exc: error = exc
        if error is not None: raise error
        gather_start = time.perf_counter()
        video = torch.cat([r[0] for r in results], dim=2)
        mask = torch.cat([r[1] for r in results], dim=2)
        done = torch.cuda.Event(enable_timing=True); done.record(); done.synchronize()
        self.last_stats = {'devices':[d.index for d in self.devices], 'sourceFrames':len(ids),
            'optimizations':dict(self.optimizations),
            'prepareCudaSeconds':begin.elapsed_time(prepared_event)/1000,
            'sampleCudaSeconds':begin.elapsed_time(prepared['sampleDone'])/1000,
            'selectionCudaSeconds':prepared['sampleDone'].elapsed_time(prepared['selectionDone'])/1000,
            'scaleCudaSeconds':prepared['selectionDone'].elapsed_time(prepared_event)/1000,
            'scaleSourceCount':prepared['scaleSourceCount'],
            'gatherHostSeconds':time.perf_counter()-gather_start,
            'totalHostSeconds':time.perf_counter()-started, 'peerAccess':self.peer_access, 'shards':[r[2] for r in results]}
        if benchmark:
            self.last_stats['optimizationBenchmark'] = {
                'call':self.calls, 'suite':benchmark, 'diagnosticOnly':True, 'includesCopy':False,
                'shards':[r[2]['optimizationBenchmark'] for r in results]}
        if self.optimizations.get('benchmarkSourceSampling', False) and self.calls == 1:
            from .warp_sampling import audit_benchmark, synthetic_sources
            sources=[self.geometry[g][3] for g in ids]
            reports=[]
            for source_count in (len(sources), 1200, 4800):
                for explicit_generator in (False, True):
                    generator=torch.Generator(device=source).manual_seed(20260912) if explicit_generator else None
                    reports.append(audit_benchmark(synthetic_sources(sources, source_count),
                                                   count=2000,device=source,generator=generator))
            ragged=[torch.empty((0,3),dtype=torch.float32,device=source),
                    sources[0][:1],sources[0][:7],sources[0][:31],*sources[:3]]
            for explicit_generator in (False, True):
                generator=torch.Generator(device=source).manual_seed(4312) if explicit_generator else None
                report=audit_benchmark(ragged,count=2000,device=source,generator=generator)
                report['scenario']='empty-and-ragged-sources';reports.append(report)
            if self.sampling_benchmark_path is None:raise RuntimeError('Sampling benchmark output path missing')
            self.sampling_benchmark_path.write_text(json.dumps(reports,indent=2))
        if self.optimizations.get('benchmarkFusedCovis',False) and self.calls==1:


            import importlib
            from . import warp_covis, warp_covis_triton
            importlib.reload(warp_covis_triton);importlib.reload(warp_covis)
            from .warp_covis import audit_benchmark as audit_covis, boundary_cases
            if self.covis_benchmark_path is None:raise RuntimeError('Covis benchmark path missing')
            reports=[];name='preparing';before=torch.cuda.get_rng_state(source)
            try:
                base=prepared['points'];calibration=torch.as_tensor(K,device=source,dtype=torch.float32)
                rotation=torch.linalg.inv(poses[0])[:3,:3];translation=poses[0,:3,3]
                for count in (len(base),1200,4800):
                    name=f'repeated-real-points-{count}'
                    points=base.repeat((count+len(base)-1)//len(base),1,1)[:count].contiguous()
                    camera=torch.einsum('cmj,kj->cmk',points-translation,rotation)
                    reports.append({'name':name,**audit_covis(camera,points,
                        calibration[0,0],calibration[1,1],calibration[0,2],calibration[1,2],int(width),int(height))})
                    del points,camera
                for case in boundary_cases(device=source):
                    name=case.pop('name')
                    try:reports.append({'name':name,**audit_covis(**case)})
                    except AssertionError as error:
                        reports.append({'name':name,'error':str(error),
                                        'diagnostics':getattr(error,'diagnostics',None)})
                if any('error' in report for report in reports):
                    raise AssertionError('Covis boundary audit failed; see per-case diagnostics')
                if not torch.equal(before,torch.cuda.get_rng_state(source)):
                    raise AssertionError('Covis benchmark changed caller RNG')
                self.covis_benchmark_path.write_text(json.dumps({'reports':reports,'callerRngExact':True},indent=2))
            except BaseException as error:
                self.covis_benchmark_path.write_text(json.dumps({'reports':reports,'failedCase':name,
                    'error':type(error).__name__+': '+str(error)},indent=2))
                raise
        if audit:
            from .fixtures.zbuf_original import _render_backward_multisrc_zbuf as original
            rng_after = torch.cuda.get_rng_state(source)
            torch.cuda.set_rng_state(rng_before, source)
            try:
                expected_video, expected_mask = original(store, ids, poses, K, height, width, **kwargs)
                rng_matches = torch.equal(torch.cuda.get_rng_state(source), rng_after)
                record = {'call':self.calls, 'sourceFrames':len(ids), 'rngEqual':rng_matches,
                    'videoEqual':torch.equal(video, expected_video),'maskEqual':torch.equal(mask, expected_mask),
                    'videoMaxError':float((video-expected_video).abs().max()),
                    'maskMismatchPixels':int((mask!=expected_mask).sum()),'performance':self.last_stats}
                with self.audit_path.open('a') as handle:handle.write(json.dumps(record)+'\n')
                if not all(record[k] for k in ('rngEqual','videoEqual','maskEqual')):
                    raise RuntimeError(f'Multiwarp numerical regression: {record}')
            finally:
                torch.cuda.set_rng_state(rng_after,source)
        return video, mask

    @torch.no_grad()
    def _shard(self, i, store, ids, prepared, poses, K, height, width, kwargs, ready, output_device, benchmark=False):
        device = self.devices[i]; stream = self.streams[i]; replica = self.replicas[i]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            stream.wait_event(ready)
            start = time.perf_counter(); b = torch.cuda.Event(enable_timing=True); c = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            b.record(); local = {}; changed = 0; copied_bytes = 0
            for g in set(replica)-set(ids): del replica[g]
            for g in ids:
                signature = frame_signature(store[g]); entry = replica.get(g)
                if entry is None or entry[0] != signature:
                    frame = tuple(x.to(device, non_blocking=True) for x in store[g])
                    entry = (signature, store[g], frame); replica[g] = entry; changed += 1
                    if device != output_device: copied_bytes += sum(x.numel()*x.element_size() for x in frame)
                local[g] = entry[2]
            covis = prepared['covis'].to(device, non_blocking=True)
            targets = poses.to(device, non_blocking=True)
            c.record()
            options = dict(kwargs, device=device, _prepared={'points':None,'scales':prepared['scales'],'covis':covis,'orders':prepared['orders']},_world_cache=self.world_caches[i])
            video, mask = render_zbuf(local, ids, targets, K, height, width, **options)
            e.record()
            if benchmark:
                reference_video, reference_mask = video, mask
            copied = torch.cuda.Event(enable_timing=True)
            video = video.to(output_device, non_blocking=True); mask = mask.to(output_device, non_blocking=True)
            copied.record();stream.synchronize()
            stats = {'gpu':device.index,'frames':len(poses),'changedSources':changed,
                'geometryCopyBytes':copied_bytes,'copyCudaSeconds':b.elapsed_time(c)/1000,
                'renderCudaSeconds':c.elapsed_time(e)/1000,'returnCopyCudaSeconds':e.elapsed_time(copied)/1000,'hostSeconds':time.perf_counter()-start}
            if benchmark:
                stats['optimizationBenchmark'] = self._benchmark_shard(
                    local, ids, targets, K, height, width, options, stream,
                    reference_video, reference_mask, reuse_source_index_only=benchmark=='sourceIndex')
            return video, mask, stats

    @torch.no_grad()
    def _benchmark_shard(self, store, ids, poses, K, height, width, options, stream,
                         reference_video, reference_mask, *, reuse_source_index_only=False):


        if reuse_source_index_only:
            variants = [('staticInverse', True, False, True, False),
                        ('staticInverseIndex', True, False, True, True)]
        else:
            variants = [('baseline', False, False, False, False), ('inverse', True, False, False, False),
                        ('fusion', False, True, False, False), ('both', True, True, False, False),
                        ('static', False, False, True, False), ('staticInverse', True, False, True, False),
                        ('all', True, True, True, False)]


        run_options = dict(options, _world_cache=dict(options['_world_cache']))
        device = poses.device
        rng_before = torch.cuda.get_rng_state(device)
        started = time.perf_counter()
        measurements = []

        def run_variant(index, phase, order_index):
            name, inverse, fusion, static, source_index = variants[index]
            run_options.update(_reuse_target_inverse=inverse, _device_fusion_gate=fusion,
                               _static_splat=static, _reuse_source_index=source_index)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_begin = time.perf_counter()
            begin.record()
            output_video, output_mask = render_zbuf(store, ids, poses, K, height, width, **run_options)
            end.record(); end.synchronize()
            host_seconds = time.perf_counter()-host_begin
            cuda_seconds = begin.elapsed_time(end)/1000

            video_equal = torch.equal(output_video, reference_video)
            mask_equal = torch.equal(output_mask, reference_mask)
            rng_equal = torch.equal(torch.cuda.get_rng_state(device), rng_before)
            if not (video_equal and mask_equal and rng_equal):
                raise RuntimeError('Warp optimization microbenchmark regression: '
                    f'gpu={device.index} variant={name} phase={phase} '
                    f'videoEqual={video_equal} maskEqual={mask_equal} rngEqual={rng_equal}')
            return {'variant':name, 'phase':phase, 'order':order_index,
                    'cudaSeconds':cuda_seconds, 'hostSeconds':host_seconds,
                    'videoEqual':video_equal, 'maskEqual':mask_equal, 'rngEqual':rng_equal}

        try:
            warmups = [run_variant(index, 'warmup', index) for index in range(len(variants))]


            order = list(range(len(variants))) + list(reversed(range(len(variants))))
            for position, index in enumerate(order):
                measurements.append(run_variant(index, 'measured', position))
            result = {'gpu':device.index, 'frames':len(poses),
                      'order':[variants[index][0] for index in order],
                      'warmups':warmups, 'samples':measurements,
                      'totalHostSeconds':time.perf_counter()-started,
                      'videoEqual':True, 'maskEqual':True, 'rngEqual':True}
            result['variants'] = {
                name:{'cudaMeanSeconds':sum(s['cudaSeconds'] for s in measurements if s['variant']==name)/2,
                      'hostMeanSeconds':sum(s['hostSeconds'] for s in measurements if s['variant']==name)/2,
                      'samples':2}
                for name, _, _, _, _ in variants}
            return result
        finally:


            stream.synchronize()
            torch.cuda.set_rng_state(rng_before, device)
