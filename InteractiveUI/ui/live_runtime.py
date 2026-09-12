from __future__ import annotations
import json
import io
import math
import sys
import os
import time
from pathlib import Path
import numpy as np
from .controls import Camera, parse_input
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False))
    tmp.replace(path)


def plan_poses(camera, pending, history_count, applied_seq, control, frame_count=36):

    import copy
    camera=copy.copy(camera);pending=copy.deepcopy(pending)
    control={'speed':1.5,'lookSpeed':15,**control}
    keys,speed,look=parse_input(control)


    pause_sequence=int(control.get('pauseSequence',-1))
    if pause_sequence>applied_seq:
        pending.clear();applied_seq=pause_sequence
    events=control.get('events',[]) or ([control['tap']] if control.get('tap') else [])
    for event in events:
        if event.get('sequence',-1)>applied_seq:
            ekeys,espeed,elook=parse_input({**control,**event})
            duration=float(event.get('duration',.1))
            if not math.isfinite(duration) or duration<0:raise ValueError('Invalid input duration')
            pending.append([ekeys,min(.33,duration),espeed,elook,int(event["sequence"])])


    excess=sum(event[1] for event in pending)-.5
    while excess>1e-10 and pending:
        removed=min(excess,pending[0][1]);excess-=removed
        pending[0][1]-=removed
        if pending[0][1]<1e-10:pending.popleft()
    applied_seq=int(control.get('sequence',0))
    applied_keys=set(keys)
    sample=[]; input_frames={}


    for index in range(frame_count):
        remaining=1/24 if history_count or index else 0
        while remaining>1e-10:
            if pending:
                active,duration,velocity,angular=pending[0][:4]
                input_sequence=pending[0][4] if len(pending[0])>4 else None
                dt=min(remaining,duration)
                pending[0][1]-=dt
                if pending[0][1]<1e-10:pending.popleft()
            else:
                active,velocity,angular,dt=keys,speed,look,remaining
                input_sequence=applied_seq
            if input_sequence is not None and input_sequence not in input_frames:
                input_frames[input_sequence]={'sequence':input_sequence,'firstPoseOffset':index,'keys':sorted(active)}
            applied_keys.update(active)
            camera.step(active,dt,velocity,angular)
            remaining-=dt
        yaw,pitch=camera.yaw,camera.pitch
        cy,sy,cp,sp=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
        pose=np.eye(4,dtype=np.float32)
        pose[:3,:3]=[[cy,sy*sp,sy*cp],[0,cp,-sp],[-sy,cy*sp,cy*cp]]
        pose[:3,3]=[camera.x,camera.y,camera.z]
        sample.append(pose)
    return {'camera':camera,'pending':pending,'sequence':applied_seq,'keys':sorted(applied_keys),'poses':np.stack(sample),'inputFrames':list(input_frames.values())}


class LiveRollout:
    def __init__(self, directory, pipeline_id=None, pipeline_load_count=None, progress_path=None):
        self.root=Path(directory)
        self.frames=self.root/'frames';self.frames.mkdir(exist_ok=True)
        self.camera=Camera();self.pending=deque();self.poses=[];self.frame=0;self.chunk=-1
        self.decode_seconds=0.;self.output_seconds=0.;self.ingest_seconds=0.
        self.decode_events=[];self.output_futures=deque();self.output_pool=None
        self.decode_host_seconds=[];self.output_cuda_events=[]
        self.publication_events=[];self.output_wait_events=[];self.output_task_events=[]
        self.publish_lock=threading.Lock()
        self.frame_publisher=None
        if os.environ.get('EVOKE_FRAME_SOCKET'):
            from .frame_hub import Publisher
            self.frame_publisher=Publisher(os.environ['EVOKE_FRAME_SOCKET'], self.root.name)
        session_path=self.root/"session.json"
        self.options=json.loads(session_path.read_text()) if session_path.exists() else {}
        self.playback_mode=self.options.get('playbackMode','legacy')
        if self.playback_mode not in {'legacy','continuous'}:
            raise ValueError('Invalid playback mode')
        policy=self.root.parent.parent/'geometry-policy.json'
        settings=json.loads(policy.read_text()) if policy.exists() else {}
        from .playback_settings import playback_settings
        self.playback_settings=playback_settings(self.options.get('warmupChunks',0),
            self.options.get('aheadChunks'),settings.get('playbackLeadFrames',60))
        self.warmup_chunks=self.playback_settings['warmupChunks']
        self.hidden_frames=0;self.warmup_completed=0
        self.generation_lead=self.playback_settings['generationLeadFrames']
        self.publication_limit=self.playback_settings['publicationLimitFrames']
        self.publication_gate=None
        self.output_stopped=False;self.cancelled_output_frames=0;self.chunk_first_published=0
        if self.playback_mode=='continuous' and (self.generation_lead>60 or self.options.get('aheadChunks') is not None):
            from .playback_flow import PlaybackPublicationGate
            self.publication_gate=PlaybackPublicationGate(lambda:self._read_request('control.json',{}),limit=self.publication_limit)
        self.geometry_mode=settings.get('mode','overlap' if os.environ.get('EVOKE_FRAME_SOCKET') else 'sync')
        self.latency_profile=bool(settings.get('latencyProfile',False))
        self.file_read_events=deque(maxlen=4096)
        self.reuse_jpeg_buffer=bool(settings.get('reuseJpegBuffer',False))
        self.benchmark_jpeg_buffer=bool(settings.get('benchmarkJpegBuffer',False))
        self._jpeg_bgr=None
        self.decode_graph=bool(settings.get('decodeGraph',False))
        self.verify_decode=bool(settings.get('verifyDecode',False))
        self.decode_graph_stats=[]
        self.vae_decode_options=dict(settings.get('vaeDecodeOptimizations',{}))
        self.precision_probe=dict(settings.get('vaePrecisionProbe',{}))
        self.precision_trial=dict(settings.get('vaePrecisionTrial',{}))
        self.vae_precision=self.options.get('vaePrecision','fp32')
        if self.vae_precision not in {'fp32','fp16_mixed'}:
            raise ValueError('Invalid VAE precision')

        if 'vaePrecision' in self.options:
            self.precision_probe={};self.precision_trial={}
        if self.vae_decode_options and self.decode_graph:
            raise ValueError('Select either decoder memory optimization or legacy decoder graph')
        self.vae_decode_stats=[]
        if self.geometry_mode not in {'sync','lag','overlap','dedicated','prefetch'}:raise ValueError('Invalid geometry mode')
        self.geometry=None;self.input_windows=deque(maxlen=8)
        self.prompt_revision=0;self.prompt_attempted=0;self.prompt_windows=deque(maxlen=32)
        self.prompt_encode_seconds=0.
        self.latent_window=int(self.options.get("latentWindowSize",9))
        if self.latent_window != 9:raise ValueError("Live inference requires the trained 9-latent window")
        self.metadata_writer=None
        self.artifact_writer=None
        if settings.get('asyncMetadata',False):
            from .live_metadata import LiveMetadataWriter
            self.metadata_writer=LiveMetadataWriter(self.root)
        self.applied_seq=-1;self.state={"workerPid":os.getpid(),"pipelineId":pipeline_id,"pipelineLoadCount":pipeline_load_count};self.started=time.time();self.chunk_started=self.started
        try:
            if settings.get('asyncArtifacts',False):
                from .live_artifacts import LiveArtifactWriter
                self.artifact_writer=LiveArtifactWriter(self.root,progress_path=progress_path)
            self.publish('starting',geometryMode=self.geometry_mode,vaePrecision=self.vae_precision,playbackMode=self.playback_mode,promptRevision=0,promptWindows=[],promptError=None)
            if self.precision_trial:
                from .vae_precision_probe import begin_trial_rng
                begin_trial_rng(self,settings)
        except BaseException:
            try:
                if self.artifact_writer is not None:self.artifact_writer.close()
            finally:
                if self.metadata_writer is not None:self.metadata_writer.close()
            raise

    def append_artifact(self, name, value):
        serialized=json.dumps(value,ensure_ascii=False)+'\n'
        if self.artifact_writer is not None:
            self.artifact_writer.append_jsonl(name,serialized)
        else:
            with (self.root/name).open('a') as handle:handle.write(serialized)

    def _read_request(self, name, default):
        started=time.perf_counter()
        try:return json.loads((self.root/name).read_text())
        except (OSError,ValueError):return default
        finally:
            self.file_read_events.append({'name':name,'seconds':time.perf_counter()-started})

    def control(self):
        control=self._read_request('control.json',{})
        if self.publication_gate is not None:self.publication_gate.observe(control)
        return control

    def publish(self, phase, **extra):
        if self.artifact_writer is not None:self.artifact_writer.check()
        profiled=self.latency_profile
        if profiled:started=time.perf_counter()
        with self.publish_lock:
            if profiled:acquired=time.perf_counter()
            output_chunk=extra.pop('_output_chunk',None)
            if output_chunk is not None and output_chunk!=self.chunk:
                phase=self.state.get('phase',phase)
            if self.chunk<self.warmup_chunks and self.warmup_completed<self.warmup_chunks and phase not in {'stopped','error','paused'}:
                phase='warming'
            self.state.update(phase=phase,chunk=self.chunk,latestFrame=self.frame,camera=self.camera.sample(),
                              appliedSequence=self.applied_seq,updatedAt=time.time(),
                              warmupChunks=self.warmup_chunks,warmupCompleted=self.warmup_completed,
                              hiddenFrames=self.hidden_frames,playbackSettings=self.playback_settings,**extra)
            if self.metadata_writer is not None:
                self.metadata_writer.publish_state(json.dumps(self.state))
            else:
                atomic_json(self.root/'state.json',self.state)
            if profiled:
                elapsed=time.perf_counter()-acquired
                self.publication_events.append({'phase':phase,'chunk':self.chunk,
                    'lockWaitSeconds':acquired-started,
                    'stateWriteSeconds':elapsed if self.metadata_writer is None else 0.,
                    'stateSubmitSeconds':elapsed if self.metadata_writer is not None else 0.})

    def _wait_output(self, reason):
        if self.latency_profile:started=time.perf_counter()
        self.output_futures.popleft().result()
        if self.latency_profile:
            self.output_wait_events.append({'reason':reason,'seconds':time.perf_counter()-started})

    def flush_output(self):
        while self.output_futures:self._wait_output('flush')

    def next_poses(self, chunk, frame_count=36):
        if frame_count != 36:raise ValueError("The trained live window decodes 36 frames")
        self.file_read_events.clear()
        while True:
            if self.output_stopped:
                self.publish('stopped');return None
            control=self.control()
            age=time.time()-control.get('updatedAt',0)
            if control.get('stop') or age>15:
                self.publish('stopped');return None
            if control.get('paused') or not control.get('connected') or age>1:
                self.pending.clear()
                if self.state.get('phase') != 'paused':self.publish('paused')
                time.sleep(.1);continue
            lead=self.frame-control.get('playedFrame',0)
            blocked=lead>self.generation_lead if self.options.get('aheadChunks') is not None else lead>=self.generation_lead
            if chunk>=self.warmup_chunks and self.playback_mode == 'continuous' and blocked:
                if self.state.get('phase') != 'playback_wait':self.publish('playback_wait')
                time.sleep(.02);continue
            break
        self.chunk=chunk;self.chunk_started=time.time()
        self.chunk_first_published=self.frame
        self.decode_seconds=self.output_seconds=self.ingest_seconds=0.
        self.decode_events=[]
        self.decode_host_seconds=[];self.output_cuda_events=[]
        self.publication_events=[];self.output_wait_events=[];self.output_task_events=[]
        self.decode_graph_stats=[]
        self.vae_decode_stats=[]
        hidden=chunk<self.warmup_chunks
        if hidden:
            control={**control,'keys':[],'events':[],'tap':None,'sequence':-1,'pauseSequence':-1}
        elif self.warmup_chunks and chunk==self.warmup_chunks:

            control={**control,'events':[],'tap':None}
        plan=plan_poses(self.camera,self.pending,len(self.poses),self.applied_seq,control,frame_count)
        self.camera=plan['camera'];self.pending=plan['pending'];self.applied_seq=plan['sequence']
        keys,_,_=parse_input({'speed':1.5,'lookSpeed':15,**control})
        applied_keys=plan['keys'];sample=plan['poses']
        accepted_at=time.time()
        output_start=(chunk-self.warmup_chunks)*36
        input_window=None if hidden else {'chunk':chunk,'sequence':self.applied_seq,'acceptedAt':accepted_at,
            'controlUpdatedAt':control.get('updatedAt'),'firstOutputFrame':output_start+1,'lastOutputFrame':output_start+36,
            'inputs':[{**entry,'firstOutputFrame':output_start+entry['firstPoseOffset']+1} for entry in plan['inputFrames']]}
        if input_window is not None:self.input_windows.append(input_window)
        self.poses.extend(sample)
        if self.artifact_writer is not None:
            buffer=io.BytesIO();np.save(buffer,np.stack(sample))
            self.artifact_writer.replace_bytes('last_chunk_poses.npy',buffer.getvalue())
        else:
            np.save(self.root/'last_chunk_poses.npy',np.stack(sample))
        self.append_artifact('actions.jsonl',{'chunk':chunk,'hiddenWarmup':hidden,'sequence':self.applied_seq,'keys':sorted(applied_keys),'poseFps':24,'firstPoseIndex':chunk*frame_count,
                            'cameraEnd':self.camera.sample(),'acceptedAt':accepted_at,'inputWindow':input_window})
        self.publish('generating',appliedKeys=sorted(keys),inputAcceptedAt=accepted_at,inputWindows=list(self.input_windows))
        return np.stack(self.poses)

    def next_prompt_request(self):

        self.prompt_encode_seconds=0.
        if self.chunk<self.warmup_chunks:return None
        request=self._read_request('prompt-request.json',None)
        if not isinstance(request,dict):return None
        revision=request.get('revision')
        if type(revision) is not int or revision<=self.prompt_attempted:return None
        self.prompt_attempted=revision
        if not isinstance(request.get('prompt'),str) or not request['prompt'].strip():
            self.prompt_failed(request,'新的场景描述不能为空。');return None
        return request

    def prompt_applied(self, request, encode_seconds, cached):
        self.prompt_revision=request['revision'];self.prompt_encode_seconds=encode_seconds
        window={'revision':self.prompt_revision,'requestId':request.get('requestId'),
                'chunk':self.chunk,'firstOutputFrame':(self.chunk-self.warmup_chunks)*36+1,'lastOutputFrame':(self.chunk-self.warmup_chunks+1)*36,
                'mode':request.get('mode','replace'),'text':request.get('text',request['prompt']),
                'prompt':request['prompt'],'submittedAt':request.get('submittedAt'),
                'appliedAt':time.time(),'encodeSeconds':encode_seconds,'cacheHit':cached}
        self.append_artifact('prompt-events.jsonl',{'status':'applied',**window})


        self.prompt_windows.append({key:(value[:240] if key=='text' else value)
                                    for key,value in window.items() if key!='prompt'})
        self.publish('generating',promptRevision=self.prompt_revision,promptError=None,
                     promptWindows=list(self.prompt_windows))

    def prompt_failed(self, request, message):
        error={'revision':request['revision'],'requestId':request.get('requestId'),'message':message}
        self.append_artifact('prompt-events.jsonl',{'status':'failed','chunk':self.chunk,**error})
        self.publish('generating',promptError=error)

    def preview_poses(self):
        control=self.control()
        if control.get('stop') or control.get('paused') or not control.get('connected') or time.time()-control.get('updatedAt',0)>1:
            return None
        if self.chunk+1<self.warmup_chunks:
            control={**control,'keys':[],'events':[],'tap':None,'sequence':-1,'pauseSequence':-1}
        elif self.warmup_chunks and self.chunk+1==self.warmup_chunks:
            control={**control,'events':[],'tap':None}
        return plan_poses(self.camera,self.pending,len(self.poses),self.applied_seq,control)['poses']

    def decoded(self, frames):

        if self.chunk<self.warmup_chunks:


            self.hidden_frames+=frames.shape[2]
            return
        import torch
        if self.output_pool is None:
            self.output_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='live-jpeg')

        if len(self.output_futures)>=3:self._wait_output('backpressure')
        if self.latency_profile:
            begin=torch.cuda.Event(enable_timing=True);converted=torch.cuda.Event(enable_timing=True)
            copied=torch.cuda.Event(enable_timing=True);begin.record()
        pixels=frames[0].detach().float().clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8).permute(1,2,3,0).contiguous()
        if self.latency_profile:converted.record()
        host=torch.empty(pixels.shape,dtype=torch.uint8,device='cpu',pin_memory=True)
        host.copy_(pixels,non_blocking=True)
        if self.latency_profile:
            copied.record();self.output_cuda_events.append((begin,converted,copied))
        ready=torch.cuda.Event();ready.record()
        context={'chunk':self.chunk,'sequence':self.applied_seq,'started':self.chunk_started}
        if self.latency_profile:context['submittedAt']=time.perf_counter()
        self.output_futures.append(self.output_pool.submit(self.write_frames,host,ready,context))

    def write_frames(self, host, ready, context):
        import cv2
        if self.latency_profile:task_started=time.perf_counter()
        ready.synchronize()
        start=time.perf_counter()
        if self.latency_profile:
            self.output_task_events.append({'chunk':context['chunk'],
                'queueSeconds':task_started-context['submittedAt'],'readyWaitSeconds':start-task_started})
        first_index=self.frame+1
        first_published=None
        for offset,pixel in enumerate(host.numpy()):
            if (self.output_stopped or (self.publication_gate is not None
                    and not self.publication_gate.wait_for_slot(self.frame))):


                self.output_stopped=True
                self.cancelled_output_frames+=len(host)-offset
                break
            if self.benchmark_jpeg_buffer and context['chunk']==2 and not getattr(self,'_jpeg_buffer_bench_done',False):
                self._jpeg_buffer_bench_done=True
                from .jpeg_output import benchmark_buffer
                atomic_json(self.root/'jpeg-buffer-benchmark.json',benchmark_buffer(pixel))
            index=self.frame+1
            if self.reuse_jpeg_buffer:
                if self._jpeg_bgr is None or self._jpeg_bgr.shape!=pixel.shape:
                    self._jpeg_bgr=np.empty_like(pixel)
                bgr=cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR,dst=self._jpeg_bgr)
            else:
                bgr=cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR)
            ok,jpeg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise RuntimeError('JPEG encoding failed')
            if self.frame_publisher is not None:
                self.frame_publisher.send(index,jpeg.tobytes())
            else:
                path=self.frames/f'{index:09d}.jpg';tmp=path.with_suffix('.tmp')
                tmp.write_bytes(jpeg.tobytes());tmp.replace(path)
                if index>self.publication_limit:(self.frames/f'{index-self.publication_limit:09d}.jpg').unlink(missing_ok=True)
            if first_published is None:first_published=time.time()
            self.frame=index
        if first_published is None:return
        if self.metadata_writer is not None:
            self.metadata_writer.append_frame(json.dumps({'chunk':context['chunk'],'sequence':context['sequence'],
                'firstFrame':first_index,'lastFrame':self.frame,'firstPublishedAt':first_published,
                'lastPublishedAt':time.time(),'hostSeconds':time.perf_counter()-start})+'\n')
        else:
            with (self.root/'frame-publication.jsonl').open('a') as handle:
                handle.write(json.dumps({'chunk':context['chunk'],'sequence':context['sequence'],
                    'firstFrame':first_index,'lastFrame':self.frame,'firstPublishedAt':first_published,
                    'lastPublishedAt':time.time(),'hostSeconds':time.perf_counter()-start})+'\n')
        self.output_seconds+=time.perf_counter()-start
        self.publish('streaming',_output_chunk=context['chunk'],displayedSequence=context['sequence'],
                     firstFrameSeconds=self.state.get('firstFrameSeconds',time.time()-self.started),
                     chunkFrameSeconds=time.time()-context['started'])

    def chunk_done(self, warp_seconds=None, diffusion_seconds=None):
        self.flush_output()
        elapsed=time.time()-self.chunk_started
        if self.chunk<self.warmup_chunks:self.warmup_completed=self.chunk+1
        timings={'promptRevision':self.prompt_revision,'promptEncodeSeconds':self.prompt_encode_seconds,'vaePrecision':self.vae_precision,'asyncMetadata':self.metadata_writer is not None,'chunk':self.chunk,'totalSeconds':elapsed,'warpSeconds':warp_seconds,'diffusionSeconds':diffusion_seconds,'decodeAndIngestSeconds':max(0,elapsed-(warp_seconds or 0)-(diffusion_seconds or 0)),
                 'hiddenWarmup':self.chunk<self.warmup_chunks,'hiddenFrames':self.hidden_frames,
                 'vaeDecodeSeconds':sum(a.elapsed_time(b) for a,b in self.decode_events)/1000,'imageOutputSeconds':self.output_seconds,
                 'geometryIngestSeconds':self.ingest_seconds,'framesPerWindow':self.latent_window*4}
        reads=list(self.file_read_events)
        timings['fileReadProfile']={name:{'count':len(values),'totalSeconds':sum(values),
            'maxSeconds':max(values)} for name in {r['name'] for r in reads}
            if (values:=[r['seconds'] for r in reads if r['name']==name])}
        if self.latency_profile:
            timings['decodeProfile']={'sliceCudaSeconds':[a.elapsed_time(b)/1000 for a,b in self.decode_events],
                'sliceHostSeconds':self.decode_host_seconds,
                'pixelConvertCudaSeconds':sum(a.elapsed_time(b) for a,b,c in self.output_cuda_events)/1000,
                'd2hCudaSeconds':sum(b.elapsed_time(c) for a,b,c in self.output_cuda_events)/1000}
        if self.latency_profile:
            timings['publicationProfile']={'statePublishes':list(self.publication_events),
                'futureWaits':list(self.output_wait_events),'tasks':list(self.output_task_events)}
        if self.decode_graph:
            timings['decodeGraph']={'usedSlices':sum(s['graphUsed'] for s in self.decode_graph_stats),
                'totalSlices':len(self.decode_graph_stats),
                'diagnostics':[s for s in self.decode_graph_stats
                    if s.get('verified') or s.get('graphInitialized') or not s.get('graphUsed')]}
        if self.vae_decode_options:
            timings['vaeDecodeOptimizations']={'options':self.vae_decode_options,
                'slices':self.vae_decode_stats}
        if self.geometry is not None:timings['geometryOverlap']=self.geometry.summary()
        if self.metadata_writer is not None:timings['metadataWriter']=self.metadata_writer.summary()
        if self.artifact_writer is not None:timings['artifactWriter']=self.artifact_writer.summary()
        if self.publication_gate is not None:
            timings['publicationGate']=self.publication_gate.summary()
            timings['publishedFramesThisChunk']=self.frame-self.chunk_first_published
            timings['cancelledOutputFrames']=self.cancelled_output_frames
        self.append_artifact('timings.jsonl',timings)
        self.publish('waiting',chunkSeconds=elapsed,timings=timings)

    def close(self):
        error = sys.exc_info()[1]
        if error is not None and self.publication_gate is not None:self.publication_gate.abort()
        try:
            if self.geometry is not None:
                try:self.geometry.close(failed=error is not None)
                except Exception as geometry_error:
                    if error is None:error=geometry_error
                    if self.publication_gate is not None:self.publication_gate.abort()
            try:self.flush_output()
            except Exception as output_error:
                if error is None:error=output_error
                if self.publication_gate is not None:self.publication_gate.abort()
            finally:
                try:
                    if self.output_pool is not None:self.output_pool.shutdown(wait=True)
                finally:
                    try:
                        if self.frame_publisher is not None:self.frame_publisher.close()
                    finally:
                        if getattr(self,'precision_probe',None) or getattr(self,'precision_trial',None):
                            from .vae_precision_probe import close_probe
                            close_probe(self)
            self.publish('error' if error else 'stopped', message=str(error) if error else '探索已结束')
        finally:
            try:
                if self.artifact_writer is not None:
                    self.artifact_writer.close()
                    atomic_json(self.root/'artifacts-summary.json',self.artifact_writer.summary())
            finally:
                if self.metadata_writer is not None:
                    self.metadata_writer.close()
                    atomic_json(self.root/'metadata-summary.json',self.metadata_writer.summary())
        atomic_json(self.root/'finished.json',{'frames':self.frame,'chunks':self.chunk+1,
                    'hiddenWarmupFrames':self.hidden_frames,'warmupCompleted':self.warmup_completed,
                    'generatedFrames':(self.chunk+1)*36,'cancelledOutputFrames':self.cancelled_output_frames,
                    'finishedAt':time.time()})
        if error is not None and sys.exc_info()[1] is None:raise error
