from concurrent.futures import ThreadPoolExecutor
import json
import time


def history_cutoff(target_start, stride, lag):
    return int(target_start)-int(stride)*int(lag)-1


class LaggedQueue:

    def __init__(self, run):
        self.run=run
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='evoke-geometry')
        self.pending=None
        self.future=None
        self.closed=False

    def defer(self, payload):
        if self.closed or self.pending is not None:raise RuntimeError('Geometry queue overflow or closed')
        self.pending=payload

    def before_warp(self):
        if self.future is None:return None
        future=self.future;self.future=None
        return future.result()

    def after_warp(self, context):
        if self.future is not None:raise RuntimeError('Previous geometry update was not fenced')
        if self.pending is not None:
            payload=self.pending;self.pending=None
            self.future=self.executor.submit(self.run,payload,context)

    def close(self, context=None, discard_pending=False):
        results=[]
        try:
            result=self.before_warp()
            if result is not None:results.append(result)
            if not discard_pending and self.pending is not None:
                payload=self.pending;self.pending=None
                results.append(self.run(payload,context))
            return results
        finally:
            self.closed=True;self.pending=None
            self.executor.shutdown(wait=True)


class GeometryOverlap:
    def __init__(self, live, pipe, state, device):
        import torch
        self.torch=torch;self.live=live;self.pipe=pipe;self.state=state;self.device=device
        self.mode=live.geometry_mode
        if state.get('da3_bank') is None:raise ValueError('Geometry overlap requires the DA3 frame bank')
        if self.mode not in {'sync','lag','overlap'}:raise ValueError('Invalid geometry mode')
        self.state['da3_fixed_lag']=True
        self.stream=torch.cuda.Stream(device=device) if self.mode=='overlap' else None
        self.origin=torch.cuda.Event(enable_timing=True);self.origin.record()
        self.queue=LaggedQueue(self._run) if self.mode=='overlap' else None
        self.dit={};self.completed=[];self.wait_seconds=0.;self.chunk=0;self.closed=False

    def _record(self, result):
        if result is None:return
        begin,end=result.pop('_events')
        result['cudaStartMs']=self.origin.elapsed_time(begin)
        result['cudaEndMs']=self.origin.elapsed_time(end)
        result['cudaSeconds']=(result['cudaEndMs']-result['cudaStartMs'])/1000
        pair=self.dit.pop(result['overlapWindow'],None)
        if pair is not None and pair[1] is not None:
            pair[1].synchronize()
            lo,hi=(self.origin.elapsed_time(event) for event in pair)
            result.update(ditStartMs=lo,ditEndMs=hi,
                cudaIntervalOverlapWithDitMs=max(0.,min(hi,result['cudaEndMs'])-max(lo,result['cudaStartMs'])))
        with (self.live.root/'geometry-overlap.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        self.completed.append(result)

    def before_warp(self, chunk):
        self.chunk=chunk;start=time.perf_counter()
        if self.queue is not None:self._record(self.queue.before_warp())
        if chunk>0 and chunk==getattr(self.live,'warmup_chunks',0) and self.queue is not None and self.queue.pending is not None:
            payload=self.queue.pending;self.queue.pending=None
            self._record(self._run(payload,None))
        self.wait_seconds=time.perf_counter()-start

        self.state['da3_lag']=0 if self.mode=='sync' or chunk<2 else 1

    def start_dit(self):
        torch=self.torch
        begin=torch.cuda.Event(enable_timing=True)
        if self.queue is not None:
            ready=torch.cuda.Event();ready.record()
            self.queue.after_warp((self.chunk,ready))
        begin.record();self.dit[self.chunk]=(begin,None)

    def end_dit(self):
        end=self.torch.cuda.Event(enable_timing=True);end.record()
        self.dit[self.chunk]=(self.dit[self.chunk][0],end)

        for k in list(self.dit):
            if k<self.chunk-1:self.dit.pop(k)

    def ingest(self, frames, start, poses):
        if self.queue is None or self.chunk==0:
            begin=time.perf_counter()
            self.pipe._geo_da3_ingest(self.state,frames,start,poses,self.device)
            self.live.ingest_seconds+=time.perf_counter()-begin
        else:
            ready=self.torch.cuda.Event();ready.record()
            self.queue.defer((self.chunk,frames.detach(),start,poses,ready))

    def _run(self, payload, context):
        torch=self.torch
        source,frames,start,poses,ready=payload
        window,warp_ready=context if context is not None else (None,None)
        began=time.perf_counter()
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream),torch.inference_mode():
            self.stream.wait_event(self.origin)
            self.stream.wait_event(ready)
            if warp_ready is not None:self.stream.wait_event(warp_ready)
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
            first.record()
            self.pipe._geo_da3_ingest(self.state,frames,start,poses,self.device)
            last.record()
            self.stream.synchronize()
        return {'sourceChunk':source,'overlapWindow':window,'hostSeconds':time.perf_counter()-began,'_events':(first,last)}

    def summary(self):
        result={'mode':self.mode,'lag':self.state['da3_lag'],'boundaryWaitSeconds':self.wait_seconds,
                'completed':self.completed,'pending':bool(self.queue and self.queue.pending is not None)}
        self.completed=[]
        return result

    def close(self, failed=False):
        if self.closed:return
        try:
            if self.queue is not None:
                for result in self.queue.close(discard_pending=failed):self._record(result)
        finally:
            self.closed=True


            self.queue=None;self.dit.clear();self.state=None;self.pipe=None;self.live=None
            self.stream=None;self.origin=None
