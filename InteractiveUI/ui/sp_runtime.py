from __future__ import annotations
import datetime
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist


def initialize():
    size = int(os.environ.get('EVOKE_SP_SIZE', '1'))
    if size == 1:
        return False
    if int(os.environ.get('WORLD_SIZE', '0')) != size:
        raise RuntimeError('EVOKE_SP_SIZE requires a matching torchrun world')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    from evoke.modules import student_sp
    student_sp.init_student_sp(size, chunk_parallel=False, ulysses_size=size)
    return True


class DistributedDiT:
    def __init__(self, transformer, queue):
        from evoke.modules.student_sp import make_ulysses_ctx
        self.model = transformer
        self.forward = transformer.forward
        self.clear = transformer.clear_kv_cache
        self.ctx = make_ulysses_ctx()
        self.rank = dist.get_rank()
        self.meta_group = dist.new_group(backend='gloo', timeout=datetime.timedelta(days=7))
        self.queue = Path(queue)
        self.checked = set()
        self.verify = os.environ.get('EVOKE_SP_VERIFY', '0') == '1'
        self.profile_every = max(0, int(os.environ.get('EVOKE_SP_PROFILE_EVERY', '0')))
        self.profile_limit = max(1, int(os.environ.get('EVOKE_SP_PROFILE_LIMIT', '30')))
        self.profile_count = 0
        self.forward_count = 0
        self._active_profile = None
        self.verify_qkv = os.environ.get('EVOKE_SP_VERIFY_QKV', '0') == '1'
        self.qkv_checked = set()
        self.timestep_cast_checked = set()
        self.policy_path = self.queue.parent.parent / 'latency-policy.json'
        self._policy_signature = None
        self._policy_values = {}

        marker = self.queue / f'sp-rank-{self.rank}.json'
        marker.write_text(json.dumps({'rank':self.rank,'worldSize':dist.get_world_size(),
            'pid':os.getpid(),'device':torch.cuda.current_device(),'loadedAt':time.time()}))
        dist.barrier()

    def _read_policy(self):

        try:
            stat = self.policy_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = None
        if signature != self._policy_signature:
            values = json.loads(self.policy_path.read_text()) if signature is not None else {}
            if not isinstance(values, dict):
                raise ValueError('latency-policy.json must contain an object')
            for key in ('fusedQkv', 'verifyFusedQkv', 'hoistTimestepCast', 'verifyTimestepCast', 'benchmarkTimestepCast'):
                if key in values and not isinstance(values[key], bool):
                    raise ValueError(f'{key} must be a boolean')
            if values.get('verifyFusedQkv') and (values.get('verifyTimestepCast') or values.get('benchmarkTimestepCast')):
                raise ValueError('Select only one DiT exact audit at a time')
            if values.get('benchmarkTimestepCast') and values.get('profileEvery', self.profile_every):
                raise ValueError('Disable SP profiling during the timestep benchmark')
            for key in ('profileEvery', 'profileLimit'):
                if key in values and (type(values[key]) is not int or values[key] < 0):
                    raise ValueError(f'{key} must be a nonnegative integer')
            if (values.get('profileEpoch') != self._policy_values.get('profileEpoch')
                    or (values.get('verifyFusedQkv') and not self._policy_values.get('verifyFusedQkv'))
                    or (values.get('verifyTimestepCast') and not self._policy_values.get('verifyTimestepCast'))
                    or (values.get('benchmarkTimestepCast') and not self._policy_values.get('benchmarkTimestepCast'))):
                self.profile_count = 0
                self.qkv_checked.clear()
                self.timestep_cast_checked.clear()
            self._policy_values = values
            self._policy_signature = signature
        return self._policy_values

    def exchange(self, payload=None):
        profile = self._active_profile
        started = time.perf_counter() if profile else 0
        tensors=[]
        def pack(value):
            if isinstance(value,torch.Tensor):
                index=len(tensors); tensors.append(value.detach().contiguous())
                return ('tensor',index,tuple(value.shape),str(value.dtype).split('.')[-1],value.device.type)
            if isinstance(value,dict): return ('dict',[(k,pack(v)) for k,v in value.items()])
            if isinstance(value,tuple): return ('tuple',[pack(v) for v in value])
            if isinstance(value,list): return ('list',[pack(v) for v in value])
            if value is None or isinstance(value,(str,int,float,bool)): return ('value',value)
            raise TypeError(f'Unsupported distributed forward argument: {type(value)}')
        wire=[pack(payload) if self.rank==0 else None]
        if profile:
            profile.cpu['packSeconds'] = time.perf_counter() - started
            started = time.perf_counter()
        dist.broadcast_object_list(wire,src=0,group=self.meta_group)
        if profile:
            profile.cpu['metadataSeconds'] = time.perf_counter() - started
            started = time.perf_counter()
        def unpack(node):
            kind=node[0]
            if kind=='tensor':
                _,index,shape,dtype,device=node
                tensor=tensors[index] if self.rank==0 else torch.empty(shape,dtype=getattr(torch,dtype),device='cuda' if device=='cuda' else 'cpu')
                broadcast = lambda: dist.broadcast(tensor,src=0,group=None if device=='cuda' else self.meta_group)
                if profile and device == 'cuda':
                    profile.collective('broadcast', tensor, broadcast)
                else:
                    broadcast()
                return tensor
            if kind=='dict': return {k:unpack(v) for k,v in node[1]}
            if kind=='tuple': return tuple(unpack(v) for v in node[1])
            if kind=='list': return [unpack(v) for v in node[1]]
            return node[1]
        result = unpack(wire[0])
        if profile:
            profile.cpu['tensorDispatchSeconds'] = time.perf_counter() - started
        return result

    @torch.inference_mode()
    def distributed_forward(self,*args,**kwargs):


        try:
            from evoke.modules import student_sp
            policy = self._read_policy()
            profile_every = policy.get('profileEvery', self.profile_every)
            profile_limit = policy.get('profileLimit', self.profile_limit)
            fused_qkv = policy.get('fusedQkv', os.environ.get('EVOKE_SP_FUSED_QKV', '0') == '1')
            hoist_cast = policy.get('hoistTimestepCast', False)
            self.forward_count += 1
            profile = None
            if (profile_every and self.forward_count % profile_every == 0
                    and self.profile_count < profile_limit):
                from ui.sp_profile import SPProfile
                profile = SPProfile(self.forward_count)
                self.profile_count += 1
            self._active_profile = profile
            signature = tuple((key, tuple(value.shape)) for key, value in kwargs.items()
                              if isinstance(value, torch.Tensor))
            verify_qkv = policy.get('verifyFusedQkv', self.verify_qkv) and signature not in self.qkv_checked
            if verify_qkv:
                self.qkv_checked.add(signature)
            benchmark_cast = policy.get('benchmarkTimestepCast', False) and signature not in self.timestep_cast_checked
            verify_cast = (policy.get('verifyTimestepCast', False) or benchmark_cast) and signature not in self.timestep_cast_checked
            if verify_cast and verify_qkv:
                raise ValueError('Select only one DiT exact audit at a time')
            if verify_cast:
                self.timestep_cast_checked.add(signature)
            _, args, kwargs, verify_qkv, fused_qkv, verify_cast, hoist_cast, benchmark_cast = self.exchange(
                ('forward', args, kwargs, verify_qkv, fused_qkv, verify_cast, hoist_cast, benchmark_cast))
            fused_token = student_sp._fused_qkv.set(fused_qkv)
            cast_token = student_sp._hoist_timestep_cast.set(hoist_cast)
            cast_observations = [] if profile or verify_cast else None
            observation_token = student_sp._timestep_cast_observations.set(cast_observations)
            sync_start = time.perf_counter()
            torch.cuda.synchronize(); start=time.perf_counter()
            if profile:
                profile.cpu['preForwardSyncSeconds'] = start - sync_start
                output = profile.run_model(self.model, lambda: self._call_forward(args, kwargs, verify_qkv, verify_cast, benchmark_cast))
            else:
                output=self._call_forward(args, kwargs, verify_qkv, verify_cast, benchmark_cast)
            sync_start = time.perf_counter()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            shape=tuple(kwargs['hidden_states'].shape)
            student_sp._fused_qkv.reset(fused_token)
            student_sp._hoist_timestep_cast.reset(cast_token)
            student_sp._timestep_cast_observations.reset(observation_token)
            self._active_profile = None
            if profile:
                profile.cpu['postForwardSyncSeconds'] = time.perf_counter() - sync_start
                profile.cpu['forwardSeconds'] = elapsed
                record = profile.result()
                record.update({'shape': shape, 'pid': os.getpid(), 'time': time.time(),
                               'fusedQkv': fused_qkv, 'profileEpoch': policy.get('profileEpoch'),
                               'hoistTimestepCast': hoist_cast, 'timestepCast': cast_observations,
                               'benchmarkTimestepCast': benchmark_cast,
                               'verificationDoubleForward': verify_qkv or verify_cast})
                with (self.queue/'sp-profile.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
            if self.verify and shape not in self.checked:
                self.checked.add(shape)
                torch.cuda.synchronize(); start=time.perf_counter()
                reference=self.forward(*args,**kwargs)
                torch.cuda.synchronize(); single=time.perf_counter()-start
                actual=output[0].float(); expected=reference[0].float()
                delta=actual-expected
                record={'shape':shape,'spSeconds':elapsed,'singleSeconds':single,
                    'maxAbs':delta.abs().max().item(),
                    'relativeMax':(delta.abs().max()/expected.abs().max().clamp_min(1e-8)).item(),
                    'relativeRms':(delta.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-8)).item()}
                with (self.queue/'sp-verification.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('[SP verify]',record,flush=True)


                if not torch.isfinite(actual).all() or record['relativeRms']>.02 or record['relativeMax']>.10:
                    raise RuntimeError(f'SP numerical verification failed: {record}')
            return output
        except BaseException:
            import traceback
            traceback.print_exc()


            os._exit(70)

    def _call_forward(self, args, kwargs, verify_qkv=False, verify_cast=False, benchmark_cast=False):
        verify_cast = verify_cast or benchmark_cast
        if not verify_qkv and not verify_cast:
            return self.forward(*args, **kwargs, sf_student_sp_ctx=self.ctx)
        from evoke.modules import student_sp
        if verify_qkv and verify_cast:
            raise ValueError('Select only one DiT exact audit at a time')
        audit_name = 'timestep-cast' if verify_cast else 'qkv'
        audit_context = student_sp._hoist_timestep_cast if verify_cast else student_sp._fused_qkv
        if getattr(self.model, '_cache_config', None) is not None:
            raise RuntimeError(f'{audit_name} audit does not snapshot optional diffusers stateful cache hooks')
        processors = [m.processor for m in self.model.modules()
                      if hasattr(getattr(m, 'processor', None), 'kv_cache')]
        scale_modules = [m for m in self.model.modules() if hasattr(m, '_scale_cache')]
        def snapshot():


            return ([p.kv_cache for p in processors], torch.get_rng_state(), torch.cuda.get_rng_state(),
                    [m._scale_cache for m in scale_modules])
        def restore(state):
            for p, cache in zip(processors, state[0]):
                p.kv_cache = cache
            torch.set_rng_state(state[1])
            torch.cuda.set_rng_state(state[2])
            for module, cache in zip(scale_modules, state[3]):
                module._scale_cache = cache
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return type(a) is type(b) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a == b
        before = snapshot()
        selected = audit_context.get()
        outputs, states = [], []


        for fused in (False, True):
            restore(before)
            token = audit_context.set(fused)
            try:
                outputs.append(self.forward(*args, **kwargs, sf_student_sp_ctx=self.ctx))
                states.append(snapshot())
            finally:
                audit_context.reset(token)
        outputs_equal = equal(outputs[0], outputs[1])
        states_equal = equal(states[0], states[1])
        actual, expected = outputs[1][0].float(), outputs[0][0].float()
        record = {'rank': self.rank, 'shape': tuple(kwargs['hidden_states'].shape),
                  'outputsEqual': outputs_equal, 'cacheAndRngEqual': states_equal,
                  'maxAbs': (actual - expected).abs().max().item(), 'time': time.time()}
        if verify_cast:
            record['timestepCast'] = student_sp._timestep_cast_observations.get()
        if benchmark_cast:


            warmup_ranks = [None] * dist.get_world_size(self.meta_group)


            with torch.inference_mode(False):
                dist.all_gather_object(warmup_ranks, {
                    'rank': self.rank, 'outputsEqual': outputs_equal, 'cacheAndRngEqual': states_equal,
                }, group=self.meta_group)
            outputs_equal = all(rank['outputsEqual'] for rank in warmup_ranks)
            states_equal = all(rank['cacheAndRngEqual'] for rank in warmup_ranks)
            record.update(outputsEqual=outputs_equal, cacheAndRngEqual=states_equal,
                          benchmarkWarmupRanks=warmup_ranks)
        if benchmark_cast and outputs_equal and states_equal:


            samples = []
            for enabled in (False, True, True, False):
                restore(before)
                token = audit_context.set(enabled)
                observations = []
                observation_token = student_sp._timestep_cast_observations.set(observations)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                try:
                    with torch.inference_mode(False):
                        dist.barrier(group=self.meta_group)
                    torch.cuda.synchronize()
                    start_event.record()
                    started = time.perf_counter()
                    measured_output = self.forward(*args, **kwargs, sf_student_sp_ctx=self.ctx)
                    end_event.record()
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - started) * 1000
                    cuda_ms = start_event.elapsed_time(end_event)
                    measured_state = snapshot()
                finally:
                    student_sp._timestep_cast_observations.reset(observation_token)
                    audit_context.reset(token)
                samples.append({'hoistTimestepCast': enabled, 'cudaMilliseconds': cuda_ms,
                                'wallMilliseconds': wall_ms, 'timestepCast': observations,
                                'outputsEqual': equal(outputs[0], measured_output),
                                'cacheAndRngEqual': equal(states[0], measured_state)})
                del measured_output, measured_state
            local = {'rank': self.rank, 'shape': tuple(kwargs['hidden_states'].shape),
                     'samples': samples}
            ranks = [None] * dist.get_world_size(self.meta_group)
            with torch.inference_mode(False):
                dist.all_gather_object(ranks, local, group=self.meta_group)
            all_outputs_equal = all(s['outputsEqual'] for rank in ranks for s in rank['samples'])
            all_states_equal = all(s['cacheAndRngEqual'] for rank in ranks for s in rank['samples'])
            all_exact = all_outputs_equal and all_states_equal
            record['benchmark'] = {
                'order': ['original', 'hoisted', 'hoisted', 'original'],
                'warmup': 'untimed exact original/hoisted audit pair',
                'allOutputsCacheAndRngEqual': all_exact, 'ranks': ranks,
                'timingScope': 'full local SP forward; CUDA event and wall include forward completion; '
                               'exclude input broadcast, restore, barrier, audit comparisons and report gather',
            }
            outputs_equal = outputs_equal and all_outputs_equal
            states_equal = states_equal and all_states_equal
            record.update(outputsEqual=outputs_equal, cacheAndRngEqual=states_equal)
        if self.rank == 0:
            with (self.queue/f'sp-{audit_name}-verification.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(f'[SP {audit_name} verify]', record, flush=True)
        restore(states[int(selected)])
        if not outputs_equal or not states_equal:
            raise RuntimeError(f'{audit_name} exact numerical verification failed: {record}')
        return outputs[int(selected)]

    def clear_cache(self):
        self.exchange(('clear',))
        return self.clear()

    def install(self):
        self.model.forward=self.distributed_forward
        self.model.clear_kv_cache=self.clear_cache

    @torch.inference_mode()
    def serve(self):
        while True:
            command=self.exchange()
            if command[0]=='clear': self.clear()
            elif command[0]=='forward':
                from evoke.modules import student_sp
                _,args,kwargs,verify_qkv,fused_qkv,verify_cast,hoist_cast,benchmark_cast=command
                token = student_sp._fused_qkv.set(fused_qkv)
                cast_token = student_sp._hoist_timestep_cast.set(hoist_cast)
                try:
                    with self.model.cache_context('cond'):
                        self._call_forward(args, kwargs, verify_qkv, verify_cast, benchmark_cast)
                finally:
                    student_sp._fused_qkv.reset(token)
                    student_sp._hoist_timestep_cast.reset(cast_token)
            else:raise RuntimeError(f'Unknown SP command {command[0]}')
