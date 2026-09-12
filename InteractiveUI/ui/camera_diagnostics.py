from __future__ import annotations

import json
import math
from pathlib import Path
import re


class CameraDiagnostics:
    MAX_CHUNKS = 12
    MAX_SAMPLES_PER_CHUNK = 2
    MAX_DEPTH_SAMPLES = 4096
    FRAME_INDICES = (0, 16, 32)

    @classmethod
    def create(cls, settings, session_root):


        if not isinstance(settings, dict) or settings.get('enabled') is not True:
            return None
        root = Path(session_root)
        if settings.get('sessionId') != root.name:
            return None
        chunks = settings.get('chunks')
        if (not isinstance(chunks, list) or not 1 <= len(chunks) <= cls.MAX_CHUNKS
                or any(type(c) is not int or not 0 <= c <= 1_000_000 for c in chunks)
                or len(set(chunks)) != len(chunks)):
            raise ValueError('cameraDiagnostics.chunks requires 1–12 distinct nonnegative integers')
        return cls(root, chunks)

    def __init__(self, root, chunks):
        self.session_id = root.name
        self.root = root / 'camera-diagnostics'
        self.chunks = frozenset(chunks)
        self.samples = {}
        self.commits = {}

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or re.fullmatch(r'[0-9a-f]{64}', key) is None:
            raise ValueError('Camera diagnostic pose key must be a SHA256 hex digest')
        return key

    @staticmethod
    def _array(value):
        import numpy as np
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        if hasattr(value, 'item'):
            value = value.item()
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _uniform(ids, limit):
        if len(ids) <= limit:
            return list(ids)
        return [ids[(len(ids) - 1) * i // (limit - 1)] for i in range(limit)]

    @staticmethod
    def _cutoff(state, render_stats):

        stats = render_stats or {}
        if stats.get('historyCutoff') is not None:
            return int(stats['historyCutoff']), 'renderStats.historyCutoff'
        if state.get('da3_fixed_lag'):
            return (int(state['da3_target_start']) - int(state['da3_pix_stride']) * int(state['da3_lag']) - 1,
                    'fixed target_start-stride*lag-1')
        return (max(state['da3_bank'].frames, default=-1) - int(state['da3_pix_stride']) * int(state['da3_lag']),
                'legacy max_source-stride*lag')

    @staticmethod
    def _sample_id(key, cutoff):
        return f'{key}-cutoff-{cutoff}'

    @staticmethod
    def _atomic_json(path, value):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def capture(self, state, video, mask, poses, chunk, key, speculative, render_stats):


        if chunk not in self.chunks:
            return None
        key = self._key(key)
        cutoff, cutoff_origin = self._cutoff(state, render_stats)
        sample_id = self._sample_id(key, cutoff)
        existing = self.samples.get(chunk, {})
        if sample_id in existing or len(existing) >= self.MAX_SAMPLES_PER_CHUNK:
            return None
        import numpy as np

        if tuple(video.shape) != (1, 3, 33, 384, 640):
            raise ValueError(f'Unexpected diagnostic warp shape: {tuple(video.shape)}')
        if tuple(mask.shape) != (1, 1, 33, 384, 640):
            raise ValueError(f'Unexpected diagnostic mask shape: {tuple(mask.shape)}')
        if tuple(poses.shape) != (33, 4, 4):
            raise ValueError(f'Unexpected target pose shape: {tuple(poses.shape)}')
        targets = self._array(poses).copy()
        intrinsics = self._array(state['da3_K_pix']).copy()
        if intrinsics.shape != (3, 3):
            raise ValueError('Diagnostic render intrinsics must be 3x3')

        arrays = {
            'warp': np.stack([self._array(video[0, :, i]) for i in self.FRAME_INDICES]),
            'mask': np.stack([self._array(mask[0, 0, i]) for i in self.FRAME_INDICES]),
            'frame_indices': np.asarray(self.FRAME_INDICES, dtype=np.int64),
            'target_poses': targets,
            'K_pix': intrinsics,
        }
        for name in ('warp', 'mask', 'target_poses', 'K_pix'):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f'Nonfinite camera diagnostic {name}')
        frames = state['da3_bank'].frames
        ids = sorted(frames)
        if any(type(g) is not int for g in ids):
            raise ValueError('Diagnostic bank source ids must be integers')
        pool = [g for g in ids if g <= cutoff]
        recent = pool[-16:]
        history = self._uniform(pool[:-16], 16)
        selected = sorted(set(recent + history))
        source_records = []
        for g in selected:
            depth, intr, c2w, _rgb = frames[g]
            shape = tuple(depth.shape)
            if len(shape) != 2 or not all(1 <= size <= 2048 for size in shape):
                raise ValueError(f'Unexpected bank depth shape: {shape}')
            count = math.prod(shape)
            stride = max(1, math.ceil(count / self.MAX_DEPTH_SAMPLES))
            sampled = self._array(depth.reshape(-1)[::stride][:self.MAX_DEPTH_SAMPLES]).copy()
            source_pose = self._array(c2w).copy()
            source_intr = self._array(intr).copy()
            if source_pose.shape != (4, 4) or source_intr.shape != (3, 3):
                raise ValueError('Unexpected bank source pose/intrinsics')
            if not np.isfinite(source_pose).all() or not np.isfinite(source_intr).all():
                raise ValueError('Nonfinite bank source pose/intrinsics')
            valid = np.isfinite(sampled) & (sampled > 1e-4)
            quantiles = np.quantile(sampled[valid], [0, .1, .5, .9, 1]).tolist() if valid.any() else None
            arrays[f'source_{g}_depth_sample'] = sampled
            arrays[f'source_{g}_pose'] = source_pose
            arrays[f'source_{g}_K'] = source_intr
            source_records.append({
                'sourceId': g, 'selection': 'latest16' if g in recent else 'uniformOlderHistory',
                'depthShape': list(shape), 'flatStride': stride, 'sampleCount': int(sampled.size),
                'validSampleCount': int(valid.sum()), 'validSampleFraction': float(valid.mean()),
                'depthQuantiles_p0_p10_p50_p90_p100': quantiles,
                'ingestPose': source_pose.tolist(), 'intrinsics': source_intr.tolist(),
                'targetTranslationDistances': np.linalg.norm(targets[list(self.FRAME_INDICES), :3, 3] - source_pose[:3, 3], axis=1).tolist(),
            })
        estimator = state.get('da3_est')
        record = {
            'sessionId': self.session_id, 'chunk': chunk, 'poseKey': key, 'sampleId': sample_id,
            'speculativeAtCapture': bool(speculative), 'adoption': 'see chunk commit record',
            'frameIndices': list(self.FRAME_INDICES), 'warpLayout': 'selected_frame,channel,height,width',
            'warpRange': 'original FP32 [-1,1]; not JPEG quantized',
            'maskLayout': 'selected_frame,height,width',
            'selectedFrameCoverage': arrays['mask'].mean(axis=(1, 2)).tolist(),
            'renderMode': state.get('da3_render_mode'),
            'targetStart': state.get('da3_target_start'), 'lag': state.get('da3_lag'),
            'fixedLag': bool(state.get('da3_fixed_lag')), 'historyCutoff': cutoff,
            'cutoffOrigin': cutoff_origin, 'bankSourceCount': len(ids), 'poolSourceCount': len(pool),
            'poolMin': min(pool, default=None), 'poolMax': max(pool, default=None),
            'K_pix': intrinsics.tolist(), 'scaleLocked': self._scalar(getattr(estimator, '_scale_locked', None)),
            'scaleMode': getattr(estimator, 'scale_mode', None),
            'depthMedianTarget': self._scalar(getattr(estimator, 'depth_median_target', None)),
            'depthSampling': 'deterministic flattened stride, at most 4096 elements/source; CPU quantiles over finite depth >1e-4; sampled estimates, not full-map statistics',
            'sources': source_records,
            'budget': {'maxChunks': self.MAX_CHUNKS, 'maxSamplesPerChunk': self.MAX_SAMPLES_PER_CHUNK,
                       'maxSourcesPerSample': 32, 'maxDepthElementsPerSource': self.MAX_DEPTH_SAMPLES},
            'diagnosticOnly': True, 'timingIncludesExtraCopyAndDiskIO': True,
        }
        directory = self.root / f'chunk-{chunk:06d}-{sample_id}'
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'sample.npz.tmp'
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(directory / 'sample.npz')
        self._atomic_json(directory / 'summary.json', record)
        relative = str((directory / 'summary.json').relative_to(self.root))
        self.samples.setdefault(chunk, {})[sample_id] = relative
        return str(directory / 'summary.json')

    def commit(self, chunk, key, render_stats=None):

        if chunk not in self.chunks:
            return None
        key = self._key(key)
        existing = self.samples.get(chunk, {})
        cutoff = (render_stats or {}).get('historyCutoff')
        if cutoff is not None:
            cutoff = int(cutoff)
            sample_id = self._sample_id(key, cutoff)
        else:
            matches = [sample for sample in existing if sample.startswith(key + '-cutoff-')]
            if len(matches) > 1:
                raise ValueError('Ambiguous camera diagnostic commit: provide renderStats.historyCutoff')
            sample_id = matches[0] if matches else None
            if sample_id is not None:
                cutoff = int(sample_id.split('-cutoff-', 1)[1])
        identity = (key, sample_id)
        if chunk in self.commits:
            if self.commits[chunk] != identity:
                raise ValueError('A diagnostic chunk cannot commit two different samples')
            return None
        captured = existing.get(sample_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f'chunk-{chunk:06d}-commit.json'
        self._atomic_json(path, {'sessionId': self.session_id, 'chunk': chunk,
                                'poseKey': key, 'sampleId': sample_id, 'historyCutoff': cutoff,
                                'captured': captured is not None,
                                'sampleSummary': captured,
                                'missingSampleReason': None if captured else 'not captured or two-sample budget exhausted'})
        self.commits[chunk] = identity
        return str(path)
