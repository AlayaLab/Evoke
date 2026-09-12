from __future__ import annotations

from collections import OrderedDict, deque
from itertools import groupby
from pathlib import Path
import threading
import time


class LiveArtifactWriter:
    APPEND_NAMES = frozenset(('actions.jsonl', 'timings.jsonl', 'warp-prefetch.jsonl', 'prompt-events.jsonl', 'warmup-history.jsonl'))
    REPLACE_NAMES = frozenset(('last_chunk_poses.npy',))
    PROGRESS_SLOT = '<explicit-progress>'

    @classmethod
    def create(cls, root, *, enabled=False, **kwargs):
        if enabled is not True:
            return None
        return cls(root, **kwargs)

    def __init__(self, root, *, progress_path=None, max_rows=4096,
                 max_bytes=16 * 1024 * 1024, max_latest_bytes=4 * 1024 * 1024,
                 batch_rows=64):
        for name, value in (('max_rows', max_rows), ('max_bytes', max_bytes),
                            ('max_latest_bytes', max_latest_bytes), ('batch_rows', batch_rows)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.root = Path(root)
        self.progress_path = Path(progress_path) if progress_path is not None else None


        if self.progress_path is not None:
            target = self.progress_path.absolute()
            if target in {(self.root / name).absolute() for name in self.APPEND_NAMES | self.REPLACE_NAMES}:
                raise ValueError('progress_path must not overwrite a session append/pose artifact')
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.max_latest_bytes, self.batch_rows = max_latest_bytes, min(batch_rows, max_rows)
        self._condition = threading.Condition()
        self._append = deque()
        self._latest = OrderedDict()
        self._latest_inflight = None
        self._rows = self._bytes = 0
        self._closing = False
        self._error = None
        self._stats = {
            'appendSubmitted': 0, 'appendWritten': 0,
            'latestSubmitted': 0, 'latestWritten': 0, 'latestCoalesced': 0,
            'highWaterRows': 0, 'highWaterBytes': 0,
            'highWaterLatestSnapshots': 0, 'highWaterLatestBytes': 0,
            'backpressureCount': 0, 'backpressureSeconds': 0.,
            'maxAppendWriteSeconds': 0., 'maxReplaceWriteSeconds': 0.,
            'slowWriteCount': 0, 'writeSeconds': 0.,
        }
        self._writes = {}
        self._thread = threading.Thread(target=self._run, name='live-artifacts', daemon=True)
        self._thread.start()

    @staticmethod
    def _serialized(value):
        if isinstance(value, str):
            return value.encode('utf-8')
        if isinstance(value, bytes):
            return value
        raise TypeError('Artifact must be already-serialized immutable bytes or str')

    def _check_locked(self):
        if self._error is not None:
            raise RuntimeError('Live artifact writer failed') from self._error

    def _accepting_locked(self):
        self._check_locked()
        if self._closing:
            raise RuntimeError('Live artifact writer is closing or closed')

    def append_jsonl(self, name, serialized):
        if name not in self.APPEND_NAMES:
            raise ValueError('Unsupported append artifact name')
        data = self._serialized(serialized)
        if not data.endswith(b'\n'):
            data += b'\n'
        if b'\n' in data[:-1] or b'\r' in data:
            raise ValueError('Append artifact must contain exactly one JSONL record')
        if len(data) > self.max_bytes:
            raise ValueError('Append record exceeds max_bytes')
        with self._condition:
            self._accepting_locked()
            wait_started = None
            try:
                while self._rows >= self.max_rows or self._bytes + len(data) > self.max_bytes:
                    if wait_started is None:
                        wait_started = time.perf_counter()
                        self._stats['backpressureCount'] += 1
                    self._condition.wait()
                    self._accepting_locked()
            finally:
                if wait_started is not None:
                    self._stats['backpressureSeconds'] += time.perf_counter() - wait_started
            self._append.append((name, data))
            self._rows += 1; self._bytes += len(data)
            self._stats['appendSubmitted'] += 1
            self._stats['highWaterRows'] = max(self._stats['highWaterRows'], self._rows)
            self._stats['highWaterBytes'] = max(self._stats['highWaterBytes'], self._bytes)
            self._condition.notify()

    def _publish_latest(self, name, data):
        if len(data) > self.max_latest_bytes:
            raise ValueError('Latest artifact exceeds max_latest_bytes')
        with self._condition:
            self._accepting_locked()
            self._stats['latestSubmitted'] += 1
            if name in self._latest:
                self._stats['latestCoalesced'] += 1
            self._latest[name] = data
            count = len(self._latest) + int(self._latest_inflight is not None)
            size = sum(len(v) for v in self._latest.values()) + (self._latest_inflight[1] if self._latest_inflight else 0)
            self._stats['highWaterLatestSnapshots'] = max(self._stats['highWaterLatestSnapshots'], count)
            self._stats['highWaterLatestBytes'] = max(self._stats['highWaterLatestBytes'], size)
            self._condition.notify()

    def replace_bytes(self, name, data):
        if name not in self.REPLACE_NAMES:
            raise ValueError('Unsupported replacement artifact name')
        if not isinstance(data, bytes):
            raise TypeError('Replacement must be immutable serialized bytes')
        self._publish_latest(name, data)

    def publish_progress(self, serialized):
        if self.progress_path is None:
            raise ValueError('publish_progress requires an explicit constructor progress_path')
        self._publish_latest(self.PROGRESS_SLOT, self._serialized(serialized))

    def check(self):
        with self._condition:
            self._check_locked()

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self.check()

    def summary(self):

        with self._condition:
            return {**self._stats, 'pendingRows': self._rows, 'pendingBytes': self._bytes,
                    'pendingLatestSnapshots': len(self._latest),
                    'pendingLatestBytes': sum(len(v) for v in self._latest.values()),
                    'inflightLatestSnapshots': int(self._latest_inflight is not None),
                    'inflightLatestBytes': self._latest_inflight[1] if self._latest_inflight else 0,
                    'closing': self._closing, 'failed': self._error is not None,
                    'writerAlive': self._thread.is_alive(),
                    'writesByName': {name: dict(value) for name, value in self._writes.items()}}

    def _timing(self, kind, name, started, success):
        elapsed = time.perf_counter() - started
        with self._condition:
            key = 'maxAppendWriteSeconds' if kind == 'append' else 'maxReplaceWriteSeconds'
            self._stats[key] = max(self._stats[key], elapsed)
            self._stats['writeSeconds'] += elapsed
            self._stats['slowWriteCount'] += int(elapsed >= .1)
            stats = self._writes.setdefault(name, {'attempts': 0, 'completed': 0, 'seconds': 0., 'maxSeconds': 0., 'slowWrites': 0})
            stats['attempts'] += 1; stats['completed'] += int(success)
            stats['seconds'] += elapsed; stats['maxSeconds'] = max(stats['maxSeconds'], elapsed)
            stats['slowWrites'] += int(elapsed >= .1)

    def _append_file(self, name, records):
        with (self.root / name).open('ab') as handle:
            handle.writelines(records)

    def _replace_file(self, name, data):
        target = self.progress_path if name == self.PROGRESS_SLOT else self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.artifacts-{id(self)}.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)

    def _run(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            while True:
                with self._condition:
                    while not self._append and not self._latest and not self._closing:
                        self._condition.wait()
                    if not self._append and not self._latest and self._closing:
                        return
                    records = [self._append.popleft() for _ in range(min(self.batch_rows, len(self._append)))]
                    latest = self._latest.popitem(last=False) if self._latest else None
                    self._latest_inflight = (latest[0], len(latest[1])) if latest else None


                for name, group in groupby(records, key=lambda item: item[0]):
                    lines = tuple(item[1] for item in group)
                    started = time.perf_counter(); success = False
                    try:
                        self._append_file(name, lines); success = True
                    finally:
                        self._timing('append', name, started, success)
                if records:
                    count, size = len(records), sum(len(data) for _, data in records)

                    lines = group = None
                    records = []
                    with self._condition:
                        self._rows -= count; self._bytes -= size
                        self._stats['appendWritten'] += count
                        self._condition.notify_all()
                if latest is not None:
                    name, data = latest
                    started = time.perf_counter(); success = False
                    try:
                        self._replace_file(name, data); success = True
                    finally:
                        self._timing('replace', name, started, success)
                    latest = data = None
                    with self._condition:
                        self._latest_inflight = None
                        self._stats['latestWritten'] += 1
        except BaseException as error:
            with self._condition:
                self._error = error
                self._closing = True
                self._condition.notify_all()
