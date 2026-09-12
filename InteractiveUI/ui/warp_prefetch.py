from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from types import SimpleNamespace
import numpy as np


def pose_key(poses):
    return hashlib.sha256(np.asarray(poses,dtype=np.float32).tobytes()).hexdigest()


def candidate_matches(candidate, chunk, poses):
    return candidate is not None and candidate['chunk']==chunk and candidate['poseKey']==pose_key(poses)


class WarpPrefetch:
    def __init__(self,live,pipe,state,device):
        import torch
        self.torch=torch;self.live=live;self.pipe=pipe;self.state=state;self.device=device
        self.mode=live.geometry_mode;self.aux=torch.device('cuda',int(os.environ['EVOKE_AUX_GPU']))
        if self.aux==torch.device(device):raise ValueError('Warp needs a separate GPU')
        policy=live.root.parent.parent/'geometry-policy.json'
        settings=json.loads(policy.read_text()) if policy.exists() else {}


        self.camera_diagnostics_armed=bool((settings.get('cameraDiagnostics') or {}).get('enabled',False))
        self.camera_diagnostics=None
        self.camera_trial_armed=bool((settings.get('cameraControlTrial') or {}).get('enabled',False))
        self.camera_trial=None;self.camera_recoveries=0
        self.reuse_first=bool(settings.get('reuseFirstEncode',False))
        self.encode_graph=bool(settings.get('encodeGraph',False))
        self.verify_encode=bool(settings.get('verifyEncode',False))
        self.encode_optimizations=dict(settings.get('vaeEncodeOptimizations',{}))
        self.encode_calls=0
        devices=settings.get('warpGpus',[])
        self.encoder=torch.device('cuda',int(settings.get('encodeGpu',self.aux.index)))
        if not 4<=self.encoder.index<torch.cuda.device_count():raise ValueError('Warp encoder must stay outside DiT ranks 0–3')
        self.multirender=None
        if devices:
            if self.state.get('da3_render_mode')!='backward_zbuf':raise ValueError('Multiwarp is validated for the production backward_zbuf recipe only')
            if len(set(devices))!=len(devices) or any(int(d)<4 or int(d)>=torch.cuda.device_count() for d in devices):
                raise ValueError('Warp GPUs must be distinct and outside DiT ranks 0–3')
            from .multiwarp import MultiWarp
            pool=getattr(pipe,'_evoke_multiwarp',None)
            if pool is None:
                pool=MultiWarp(devices);pipe._evoke_multiwarp=pool
            if tuple(d.index for d in pool.devices)!=tuple(devices):raise ValueError('Restart required to change warp devices')
            pool.clear();pool.audit_path=live.root/'multiwarp-audit.jsonl' if settings.get('verifyWarp') else None
            pool.optimizations=dict(settings.get('warpOptimizations',{}))
            pool.sampling_benchmark_path=live.root/'source-sampling-benchmark.json'
            pool.covis_benchmark_path=live.root/'covis-benchmark.json'
            pool.benchmark_optimizations=bool(settings.get('warpOptimizationBenchmark',False))
            self.multirender=pool;self.state['multisrc_renderer']=pool
        cfg=pipe._geo_vsnoise_cfg
        if cfg.get('warp_keep_clean_anchor') or cfg.get('geo_warp_warm_encode'):
            raise ValueError('Prefetch does not support latest-frame anchors or warm warp encoding')
        self.state['da3_fixed_lag']=True
        self.stream=torch.cuda.Stream(device=self.aux)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='evoke-warp-gpu')
        self.future=None;self.pending=None;self.candidate=None;self.chunk=0;self.wait_seconds=0.
        self.cancelled=None;self.closed=False;self.last_kwargs=None;self.records=[];self.last_status={}
        vae=getattr(pipe,'_evoke_warp_vae',None)
        if vae is None:
            with torch.random.fork_rng(devices=[]):
                vae=type(pipe.vae).from_config(pipe.vae.config)
            vae.load_state_dict(pipe.vae.state_dict())
            vae.to(device=self.encoder,dtype=pipe.vae.dtype).eval().requires_grad_(False)
            if getattr(pipe.vae,'use_tiling',False):vae.enable_tiling()
            if getattr(pipe.vae,'use_slicing',False):vae.enable_slicing()
            pipe._evoke_warp_vae=vae
        if vae.device!=self.encoder:vae.to(self.encoder)
        self.vae=vae
        self.encode_stream=torch.cuda.Stream(device=self.encoder)
        self.facade=SimpleNamespace(_geo_vsnoise_cfg=dict(cfg),_geo_dump_dir=None,_geo_quiet_chunk_logs=True,
            _geo_chunk0_ref_warp=pipe._geo_chunk0_ref_warp,
            _geo_chunk0_target_disparity_px=pipe._geo_chunk0_target_disparity_px)

    def _log(self,result):
        if result is None:return
        record=result['timing']
        self.live.append_artifact('warp-prefetch.jsonl',record)
        self.records.append(record)

    def before_warp(self,chunk):
        self.chunk=chunk;start=time.perf_counter();self.candidate=None;self.cancelled=None
        if self.future is not None:
            future=self.future;self.future=None
            result=future.result();self._log(result);self.candidate=result.get('warp');self.cancelled=result.get('cancelled')
        if chunk>0 and chunk==getattr(self.live,'warmup_chunks',0) and self.pending is not None:


            payload=self.pending;self.pending=None
            ready=self.torch.cuda.Event();ready.record()
            self._log(self._work(payload,None,None,ready))
            self.candidate=None
            self.live.append_artifact('warmup-history.jsonl',{'visibleStartChunk':chunk,
                'ingestedThroughChunk':payload[0],'historySources':len(self.state['da3_bank'].frames)})
        self.wait_seconds=time.perf_counter()-start
        self.state['da3_lag']=0 if chunk<2 else 1

    def _prediction_current(self,key):
        plan=self.live.preview_poses()
        return plan is not None and pose_key(plan[:33])==key

    def _render(self,kwargs,chunk,speculative=False):
        torch=self.torch;state=dict(self.state)
        if self.camera_diagnostics_armed and self.camera_diagnostics is None:
            from .camera_diagnostics import CameraDiagnostics
            policy=self.live.root.parent.parent/'geometry-policy.json'
            diagnostic_settings=json.loads(policy.read_text()).get('cameraDiagnostics')
            self.camera_diagnostics=CameraDiagnostics.create(diagnostic_settings,self.live.root)
        state['da3_lag']=kwargs.pop('_lag_override',0 if chunk<2 else 1)
        state['da3_target_start']=kwargs.pop('_target_start')
        poses=kwargs['camera_poses_pix_window']
        if isinstance(poses,np.ndarray):poses=torch.from_numpy(poses)
        poses=poses.to(self.aux,dtype=torch.float32)
        key=pose_key(poses.detach().cpu().numpy())
        kwargs.update(camera_poses_pix_window=poses,geo_state=state,chunk_idx=chunk,device=self.aux)
        if kwargs.get('source_frame_pix') is not None:
            kwargs['source_frame_pix']=kwargs['source_frame_pix'].to(self.aux)
        started=time.perf_counter()
        if speculative and not self._prediction_current(key):
            self._cancel_stage='before-render';return None


        rng_before=torch.cuda.get_rng_state(self.aux)
        try:
            video,mask=type(self.pipe)._geo_render_chunk_da3(self.facade,**kwargs)
            render_done=torch.cuda.Event(enable_timing=True);render_done.record()
            render_done.synchronize()
            rendered=time.perf_counter()
            if speculative and not self._prediction_current(key):
                self._render_timing['renderSeconds']=rendered-started
                self._cancel_stage='before-encode';return None


            with torch.cuda.device(self.encoder),torch.cuda.stream(self.encode_stream):
                self.encode_stream.wait_event(render_done)
                encoding_vae=self.vae
                if getattr(self.live,'vae_precision','fp32') == 'fp16_mixed':
                    from .vae_precision_runtime import get_encoder
                    encoding_vae=get_encoder(self.vae)
                elif getattr(self.live,'precision_trial',{}).get('encoder'):
                    from .vae_precision_probe import get_trial_encoder
                    encoding_vae=get_trial_encoder(self.vae,self.live)
                enc_begin=torch.cuda.Event(enable_timing=True);enc_end=torch.cuda.Event(enable_timing=True)
                enc_begin.record()
                video=video.to(device=self.encoder,dtype=self.vae.dtype)
                enc_copy=torch.cuda.Event(enable_timing=True);enc_copy.record()
                from .vae_latency import encode_warp_moments,resolve_encode_timing
                self.encode_calls+=1
                first,full,encode_stats=encode_warp_moments(encoding_vae,video,
                    reuse_first=self.reuse_first,
                    verify=self.verify_encode and self.encode_calls in (1,8,32,55,95),
                    graph=self.encode_graph,
                    optimizations=({**self.encode_optimizations,
                        'allowGraphInitialization':chunk==0 and not speculative}
                        if self.encode_optimizations else None))
                enc_end.record();self.encode_stream.synchronize()
                encode_stats=resolve_encode_timing(encode_stats)
                encode_stats['vaePrecision']=getattr(self.live,'vae_precision','fp32')
                if getattr(self.live,'vae_precision','fp32') == 'fp16_mixed':
                    encode_stats['precisionVariant']='head_fp16'
                elif encoding_vae is not self.vae:
                    if not torch.isfinite(full).all().item():
                        raise AssertionError('Precision trial produced nonfinite posterior')
                    encode_stats['precisionTrial']=self.live.precision_trial['encoder']
                    encode_stats['precisionTrialFinite']=True
                if (getattr(self.live,'precision_probe',None)
                        and self.live.precision_probe.get('encoder',True) and chunk==0 and not speculative
                        and not getattr(self.live,'_precision_encoder_done',False)):
                    from .vae_precision_probe import encoder_probe
                    encoder_probe(self.vae,video,full,self.live,self.encode_optimizations)
                    self.live._precision_encoder_done=True
            if self.camera_diagnostics is not None:
                self.camera_diagnostics.capture(state,video,mask,poses,chunk,key,speculative,
                                                self.facade._geo_last_render_stats)
            return {'chunk':chunk,'poseKey':key,'moments':(first,full),'mask':mask,
                    'renderStats':self.facade._geo_last_render_stats,
                    'multiwarp':self.multirender.last_stats if self.multirender else None,
                    'encodeCudaSeconds':enc_begin.elapsed_time(enc_end)/1000,
                    'encodeCopyCudaSeconds':enc_begin.elapsed_time(enc_copy)/1000,
                    'encodeProfile':encode_stats,
                    'rngBefore':rng_before,'rngAfter':torch.cuda.get_rng_state(self.aux),
                    'renderHostSeconds':rendered-started,'encodeHostSeconds':time.perf_counter()-rendered}
        finally:
            torch.cuda.set_rng_state(rng_before,self.aux)


    def _work(self,payload,kwargs,chunk,ready,speculative=False):
        torch=self.torch;start=time.perf_counter();ingest=0.;self._cancel_stage=None;self._render_timing={'renderSeconds':0.}
        with torch.cuda.device(self.aux),torch.cuda.stream(self.stream),torch.no_grad():
            if ready is not None:self.stream.wait_event(ready)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            source=None
            if payload is not None:
                source,frames,pix_start,poses,produced=payload
                self.stream.wait_event(produced)
                before=time.perf_counter()
                self.pipe._geo_da3_ingest(self.state,frames,pix_start,poses,self.aux)
                self.stream.synchronize();ingest=time.perf_counter()-before
                if getattr(self.live,'precision_trial',None):
                    from .vae_precision_probe import save_trial_geometry
                    save_trial_geometry(self.state,source,self.live)
            warp=self._render(dict(kwargs),chunk,speculative) if kwargs is not None else None
            end.record();self.stream.synchronize()
        return {'warp':warp,'cancelled':self._cancel_stage,'timing':{'cancelled':self._cancel_stage,'sourceChunk':source,'preparedChunk':chunk if kwargs is not None else None,
            'hostSeconds':time.perf_counter()-start,'cudaSeconds':begin.elapsed_time(end)/1000,
            'ingestSeconds':ingest,'renderSeconds':warp['renderHostSeconds'] if warp else self._render_timing['renderSeconds'],
            'encodeSeconds':warp['encodeHostSeconds'] if warp else 0.,'auxGpu':self.aux.index,
            'renderStats':warp['renderStats'] if warp else None,
            'encodeGpu':self.encoder.index,'encodeCopyCudaSeconds':warp['encodeCopyCudaSeconds'] if warp else 0.,
            'encodeCudaSeconds':warp['encodeCudaSeconds'] if warp else 0.,
            'encodeProfile':warp['encodeProfile'] if warp else None,
            'multiwarp':warp.get('multiwarp') if warp else None}}

    def render_and_encode(self,kwargs,encode_kwargs):
        torch=self.torch
        kwargs=dict(kwargs);kwargs.pop('geo_state',None);kwargs.pop('device',None);kwargs.pop('chunk_idx',None)
        kwargs['_target_start']=self.state['da3_target_start']
        actual=kwargs['camera_poses_pix_window'].detach().cpu().numpy()
        hit=candidate_matches(self.candidate,self.chunk,actual)
        selected=self.candidate if hit else None
        stale=(self.candidate is not None and not hit) or self.cancelled is not None
        self.candidate=None
        started=time.perf_counter()
        if selected is None:
            ready=torch.cuda.Event();ready.record()
            result=self._work(None,kwargs,self.chunk,ready);self._log(result);selected=result['warp']
        recovery=None
        if self.camera_trial_armed:
            from .fresh_history_trial import resolve_trial,should_recover
            if self.camera_trial is None:
                policy=self.live.root.parent.parent/'geometry-policy.json'
                self.camera_trial=resolve_trial(json.loads(policy.read_text()).get('cameraControlTrial'),self.live.root.name)
            span=float(np.linalg.norm(actual[:,:3,3]-actual[0,:3,3],axis=1).max())
            pending_chunk=self.pending[0] if self.pending is not None else None
            coverage=selected['renderStats'].get('cov')
            if should_recover(self.camera_trial,self.camera_recoveries,self.chunk,pending_chunk,coverage,span):
                if self.future is not None:raise RuntimeError('Fresh-history recovery requires a fenced worker')
                if not torch.equal(torch.cuda.get_rng_state(self.aux),selected['rngBefore']):
                    raise RuntimeError('Warp RNG changed before fresh-history recovery')
                recovery_started=time.perf_counter()
                old_stats=dict(selected['renderStats'])
                bank_before=set(self.state['da3_bank'].frames)


                payload=self.pending;self.pending=None
                ready=torch.cuda.Event();ready.record()
                ingested=self._work(payload,None,None,ready);self._log(ingested)
                added_sources=sorted(set(self.state['da3_bank'].frames)-bank_before)
                if not added_sources:raise RuntimeError('Fresh-history ingest added no source frames')
                fresh_kwargs=dict(kwargs,_target_start=self.state['da3_target_start'],_lag_override=0)
                ready=torch.cuda.Event();ready.record()
                refreshed=self._work(None,fresh_kwargs,self.chunk,ready);self._log(refreshed)
                selected=refreshed['warp'];self.camera_recoveries+=1
                recovery={'mode':'fresh_history_low_coverage','sourceChunk':pending_chunk,
                          'coverageBefore':coverage,'coverageAfter':selected['renderStats'].get('cov'),
                          'historyCutoffBefore':old_stats.get('historyCutoff'),
                          'historyCutoffAfter':selected['renderStats'].get('historyCutoff'),
                          'addedSources':added_sources,'poolMaxAfter':selected['renderStats'].get('poolMax'),
                          'preRecoveryRngValidated':True,
                          'translationSpan':span,'seconds':time.perf_counter()-recovery_started,
                          'speculativeHitBeforeRecovery':hit,'count':self.camera_recoveries}
                with (self.live.root/'camera-recovery.jsonl').open('a') as handle:
                    handle.write(json.dumps({'chunk':self.chunk,'poseKey':pose_key(actual),**recovery})+'\n')
                hit=False
        if not torch.equal(torch.cuda.get_rng_state(self.aux),selected['rngBefore']):
            raise RuntimeError('Warp RNG changed before candidate commit')
        torch.cuda.set_rng_state(selected['rngAfter'],self.aux)
        mask=selected['mask'].to(device=self.device)
        moments=tuple(x.to(device=self.device) for x in selected['moments'])
        self.pipe._geo_last_render_stats=selected['renderStats']
        latent=self.pipe._geo_encode_warp_to_latents(warp_video=None,visibility_mask_pix=mask,
            encoded_moments=moments,**encode_kwargs)
        self.last_kwargs=kwargs
        self.last_status={'hit':hit,'discardedStale':stale,'cancelledStale':self.cancelled,'rngCommitValidated':True,'actualPoseKey':pose_key(actual),
                          'usedPoseKey':selected['poseKey'],'exposedSeconds':time.perf_counter()-started}
        if recovery is not None:self.last_status['freshHistoryRecovery']=recovery
        if self.last_status['actualPoseKey']!=self.last_status['usedPoseKey']:raise RuntimeError('Stale warp pose used')
        if self.camera_diagnostics is not None:
            self.camera_diagnostics.commit(self.chunk,selected['poseKey'],selected['renderStats'])
        return mask,latent

    def start_dit(self):
        if self.future is not None:raise RuntimeError('Warp worker not fenced')
        kwargs=None
        if self.mode=='prefetch' and self.chunk>=1:
            plan=self.live.preview_poses()
            if plan is not None:
                kwargs=dict(self.last_kwargs)
                kwargs.update(camera_poses_pix_window=plan[:33],source_frame_pix=None,anchor_c2w=None,
                              _target_start=self.state['da3_target_start']+36)
        payload=self.pending;self.pending=None
        if payload is not None or kwargs is not None:
            ready=self.torch.cuda.Event();ready.record()
            self.future=self.executor.submit(self._work,payload,kwargs,self.chunk+1,ready,True)

    def end_dit(self):pass

    def ingest(self,frames,start,poses):
        ready=self.torch.cuda.Event();ready.record()
        payload=(self.chunk,frames.detach(),start,poses,ready)
        if self.chunk==0:
            result=self._work(payload,None,None,ready);self._log(result)
            self.live.ingest_seconds+=result['timing']['ingestSeconds']
        else:
            if self.pending is not None:raise RuntimeError('Geometry queue overflow')
            self.pending=payload

    def summary(self):
        result={'mode':self.mode,'lag':self.state['da3_lag'],'auxGpu':self.aux.index,
                'boundaryWaitSeconds':self.wait_seconds,'warpPrefetch':self.last_status,'completed':self.records}
        self.records=[]
        return result

    def close(self,failed=False):
        if self.closed:return
        try:
            if self.future is not None:self._log(self.future.result())
            if not failed and self.pending is not None:self._log(self._work(self.pending,None,None,None))
        finally:
            self.executor.shutdown(wait=True);self.vae.clear_cache()
            if self.multirender is not None:self.multirender.clear()
            self.state.pop('multisrc_renderer',None)
            self.closed=True;self.future=None;self.pending=None;self.candidate=None
            self.last_kwargs=None;self.state=None;self.pipe=None;self.live=None;self.vae=None
