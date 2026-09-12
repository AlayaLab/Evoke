from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
import time

import torch


class ViGeoAuditFailure(BaseException):
    pass


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and a.device == b.device and torch.equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


@contextmanager
def _score_mode(model, enabled, audit=False):

    encoder = model.pretrained
    modules = [encoder] + [module for module in encoder.modules()
                           if module is not encoder and hasattr(module, 'eviction')]
    missing = object()
    saved = []
    trace = {'forwards': [], 'evictions': []} if audit else None
    try:
        for module in modules:
            for key, value in (('_evoke_defer_scores', enabled), ('_evoke_score_trace', trace)):
                saved.append((module, key, getattr(module, key, missing)))
                setattr(module, key, value)
        yield trace
    finally:
        for module, key, value in reversed(saved):
            if value is missing:
                delattr(module, key)
            else:
                setattr(module, key, value)


class _HeadProfile:
    def __init__(self, model):
        self.model = model
        self.events = []
        self.handles = []
        self.pending = {}

    def __enter__(self):
        for name in ('pretrained', 'decoder', 'point_head', 'camera_head', 'normal_head', 'conf_head', 'mask_head'):
            module = getattr(self.model, name, None)
            if not isinstance(module, torch.nn.Module):
                continue
            def before(module, inputs, name=name):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(self.model.device))
                self.pending.setdefault(name, []).append((start, end, time.perf_counter()))
            def after(module, inputs, output, name=name):
                start, end, host_start = self.pending[name].pop()
                end.record(torch.cuda.current_stream(self.model.device))
                self.events.append((name, start, end, time.perf_counter() - host_start))
            self.handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def result(self):


        torch.cuda.current_stream(self.model.device).synchronize()
        stages = {}
        for name, start, end, host in self.events:
            stage = stages.setdefault(name, {'count': 0, 'gpuMilliseconds': 0., 'hostLaunchMilliseconds': 0.})
            stage['count'] += 1
            stage['gpuMilliseconds'] += start.elapsed_time(end)
            stage['hostLaunchMilliseconds'] += host * 1000
        return stages


class ViGeoLatency:
    def __init__(self, model, queue):
        self.model = model
        self.original_infer = model.infer
        self.queue = Path(queue)
        self.policy_path = self.queue.parent.parent / 'geometry-policy.json'
        self.policy_signature = None
        self.options = {}
        self.calls = 0
        self.profile_count = 0

    def _policy(self):
        try:
            stat = self.policy_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = None
        if signature != self.policy_signature:
            whole = json.loads(self.policy_path.read_text()) if signature else {}
            options = whole.get('vigeoOptimizations', {})
            if not isinstance(options, dict):
                raise ValueError('vigeoOptimizations must be an object')
            for key in ('skipUnusedHeads', 'verify', 'profile', 'deferScores',
                        'deferScoresVerify', 'deferScoresBenchmark'):
                if key in options and type(options[key]) is not bool:
                    raise ValueError(f'vigeoOptimizations.{key} must be boolean')
            if 'profileLimit' in options and (type(options['profileLimit']) is not int or options['profileLimit'] < 0):
                raise ValueError('vigeoOptimizations.profileLimit must be a nonnegative integer')
            if options.get('verify') and (options.get('deferScoresVerify') or options.get('deferScoresBenchmark')):
                raise ValueError('Run head verification and score verification in separate epochs')
            if (options.get('epoch') != self.options.get('epoch')
                    or any(options.get(key) and not self.options.get(key)
                           for key in ('verify', 'deferScoresVerify', 'deferScoresBenchmark'))):
                self.calls = 0
                self.profile_count = 0
            self.options = options
            self.policy_signature = signature
        return self.options

    def _snapshot(self):


        scores = getattr(self.model.pretrained, 'last_scores', None)
        if not isinstance(scores, torch.Tensor):
            raise ViGeoAuditFailure('Streaming ViGeo audit requires a tensor last_scores buffer')
        anchors = [(module, module.num_anchor_tokens) for module in self.model.modules()
                   if hasattr(module, 'num_anchor_tokens')]
        return {'scores': scores.clone(),
                'anchors': anchors, 'cpuRng': torch.get_rng_state(),
                'cudaRng': torch.cuda.get_rng_state(self.model.device)}

    def _restore(self, state):
        self.model.pretrained.last_scores.copy_(state['scores'])
        for module, count in state['anchors']:
            module.num_anchor_tokens = count
        torch.set_rng_state(state['cpuRng'])
        torch.cuda.set_rng_state(state['cudaRng'], self.model.device)

    @staticmethod
    def _state_values(state):
        return {'scores': state['scores'], 'anchors': [value for module, value in state['anchors']],
                'cpuRng': state['cpuRng'], 'cudaRng': state['cudaRng']}

    def _write(self, name, record):
        with (self.queue / name).open('a') as file:
            file.write(json.dumps(record) + '\n')

    def _score_experiment(self, run, selected, benchmark, info):


        before = self._snapshot()
        reference = None
        reference_state = None
        reference_trace = None
        selected_output = None
        selected_state = None
        records = []
        passed = False
        try:
            order = [(False, 'audit'), (True, 'audit')]
            if benchmark:
                order += [(mode, 'measured') for mode in (False, True, True, False)]
            for mode, phase in order:
                self._restore(before)
                output, timing, trace = run(mode, phase == 'audit', phase == 'measured')
                after = self._snapshot()
                if reference is None:
                    reference, reference_state, reference_trace = output, after, trace
                exact = {key: _equal(reference[key], output[key]) for key in
                         ('points_pred', 'depth_pred', 'pose_pred', 'conf_pred', 'kv_caches')}
                exact['scores'] = _equal(reference_state['scores'], after['scores'])
                exact['anchors'] = _equal([v for _, v in reference_state['anchors']],
                                          [v for _, v in after['anchors']])
                exact['cpuRng'] = _equal(reference_state['cpuRng'], after['cpuRng'])
                exact['cudaRng'] = _equal(reference_state['cudaRng'], after['cudaRng'])
                if phase == 'audit':
                    exact['perForwardScoresAndBudgets'] = _equal(reference_trace, trace)
                    def next_budgets(snapshot):
                        result = []
                        for item in snapshot['forwards']:
                            score, budget = item.get('scores'), item['totalBudget']
                            result.append((torch.softmax((1.0-score)/0.5, dim=0)*budget).int().tolist()
                                          if score is not None and budget and budget > 0 else None)
                        return result
                    exact['nextBudgets'] = next_budgets(reference_trace) == next_budgets(trace)
                records.append({'deferScores': mode, 'phase': phase, **timing, 'exact': exact})
                if not all(exact.values()):
                    raise ViGeoAuditFailure(f'ViGeo deferred-score exact audit failed: {exact}')
                if phase == 'audit' and mode == selected:
                    selected_output, selected_state = output, after

            forwards = []
            for item in reference_trace['forwards']:
                score = item.get('scores')
                next_budgets = None
                if score is not None and item['totalBudget'] and item['totalBudget'] > 0:

                    next_budgets = (torch.softmax((1.0-score)/0.5, dim=0)
                                    * item['totalBudget']).int().tolist()
                forwards.append({**{k: v for k, v in item.items() if k != 'scores'},
                                 'scores': score.tolist() if score is not None else None,
                                 'nextBudgets': next_budgets})
            self._write('vigeo-score-verification.jsonl', {
                **info, 'passed': True, 'selectedDeferScores': selected,
                'skipUnusedHeadsHeldFixed': info['skipUnusedHeads'],
                'evictionCount': len(reference_trace['evictions']),
                'evictions': reference_trace['evictions'], 'forwards': forwards,
                'runs': records,
                'timingNote': 'Diagnostic same-input ABBA; measured runs omit trace/hooks. Restore, comparisons, state capture and output serialization excluded. CUDA interval includes original host gaps; not production rollout latency.'})
            passed = True
            return selected_output, records
        except BaseException as error:
            self._write('vigeo-score-verification.jsonl', {
                **info, 'passed': False, 'runs': records, 'error': str(error)})
            raise
        finally:
            self._restore(selected_state if passed else before)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        options = self._policy()
        self.calls += 1
        skip = options.get('skipUnusedHeads', False)
        defer = options.get('deferScores', False)
        score_audit = (options.get('deferScoresVerify', False)
                       or options.get('deferScoresBenchmark', False)) and self.calls in (1, 2, 8, 32, 55, 95)
        verify = options.get('verify', False) and self.calls in (1, 2, 8, 32, 55, 95)
        profile = options.get('profile', False) and self.profile_count < options.get('profileLimit', 30)
        image = args[0] if args else kwargs['image']
        info = {'call': self.calls, 'epoch': options.get('epoch'), 'time': time.time(),
                'shape': list(image.shape), 'skipUnusedHeads': skip, 'deferScores': defer, 'verificationDoubleForward': verify,
                'maskHeadPresent': self.model.mask_head is not None}
        def run(selected, sampled):
            call_kwargs = dict(kwargs, skip_unused_heads=selected)
            started = time.perf_counter()
            if sampled:
                with _HeadProfile(self.model) as measurement:
                    with (_score_mode(self.model, True) if defer else nullcontext()):
                        output = self.original_infer(*args, **call_kwargs)
                stages = measurement.result()
            else:
                with (_score_mode(self.model, True) if defer else nullcontext()):
                    output = self.original_infer(*args, **call_kwargs)
                stages = None
            return output, {'skipUnusedHeads': selected, 'inferWallMilliseconds': (time.perf_counter() - started) * 1000,
                            'gpuStages': stages}
        if score_audit:
            def score_run(mode, audit, measured):
                with _score_mode(self.model, mode, audit=audit) as trace:
                    if measured:
                        stream = torch.cuda.current_stream(self.model.device)
                        stream.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(stream)
                        started = time.perf_counter()
                    output = self.original_infer(*args, **dict(kwargs, skip_unused_heads=skip))
                    timing = {}
                    if measured:
                        end.record(stream)
                        end.synchronize()
                        timing = {'inferWallMilliseconds': (time.perf_counter()-started)*1000,
                                  'inferCudaMilliseconds': begin.elapsed_time(end)}
                return output, timing, trace
            output, timings = self._score_experiment(
                score_run, defer, options.get('deferScoresBenchmark', False), info)
        elif verify:
            before = self._snapshot()
            outputs, states, timings = [], [], []
            for selected in (False, True):
                self._restore(before)
                output, timing = run(selected, profile)
                outputs.append(output)
                states.append(self._snapshot())
                timings.append(timing)
            consumed = ('points_pred', 'depth_pred', 'pose_pred', 'conf_pred', 'kv_caches')
            checks = {key: _equal(outputs[0][key], outputs[1][key]) for key in consumed}
            checks['stateAndRng'] = _equal(self._state_values(states[0]), self._state_values(states[1]))
            self._restore(states[int(skip)])
            self._write('vigeo-verification.jsonl', {**info, 'exact': checks, 'passed': all(checks.values())})
            if not all(checks.values()):
                raise ViGeoAuditFailure(f'ViGeo consumed-output/cache/RNG exact audit failed: {checks}')
            output = outputs[int(skip)]
        else:
            output, timing = run(skip, profile)
            timings = [timing]
        if profile and not score_audit:
            self.profile_count += 1
            self._write('vigeo-profile.jsonl', {**info, 'runs': timings,
                        'timingNote': 'Module events include dependent stream waits. inferWall includes existing D2H and CPU postprocessing; audit runs are not production latency.'})
        return output


def install(estimator, queue):
    model = estimator._model
    if model is None:
        raise RuntimeError('Load the resident ViGeo model before installing its latency wrapper')
    if getattr(model, '_evoke_latency_wrapper', None) is None:
        wrapper = ViGeoLatency(model, queue)
        model._evoke_latency_wrapper = wrapper
        model.infer = wrapper
