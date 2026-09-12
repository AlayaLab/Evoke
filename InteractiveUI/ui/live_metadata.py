from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time


class LiveMetadataWriter:


    def __init__(self, root, *, max_frame_rows=4096, max_frame_bytes=8 * 1024 * 1024,
                 max_state_bytes=1024 * 1024, batch_rows=64):
        for name, value in (('max_frame_rows', max_frame_rows),
                            ('max_frame_bytes', max_frame_bytes),
                            ('max_state_bytes', max_state_bytes), ('batch_rows', batch_rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.root = Path(root)
        self.max_frame_rows = max_frame_rows
        self.max_frame_bytes = max_frame_bytes
        self.max_state_bytes = max_state_bytes
        self.batch_rows = min(batch_rows, max_frame_rows)
        self._condition = threading.Condition()
        self._frames = deque()
        self._state = None
        self._state_inflight = False
        self._outstanding_rows = 0
        self._outstanding_bytes = 0
        self._stats = {'stateSnapshotsSubmitted': 0, 'stateSnapshotsWritten': 0,
                       'stateSnapshotsCoalesced': 0, 'frameLinesSubmitted': 0,
                       'frameLinesWritten': 0, 'highWaterFrameRows': 0,
                       'highWaterFrameBytes': 0, 'maxStateWriteSeconds': 0.,
                       'maxFrameWriteSeconds': 0.}
        self._closing = False
        self._error = None
        self._thread = threading.Thread(target=self._run, name='live-metadata', daemon=True)
        self._thread.start()

    def _check_locked(self):
        if self._error is not None:
            raise RuntimeError('Live metadata writer failed') from self._error

    def _accepting_locked(self):
        self._check_locked()
        if self._closing:
            raise RuntimeError('Live metadata writer is closing or closed')

    def publish_state(self, serialized_json_str):

        if not isinstance(serialized_json_str, str):
            raise TypeError('State must be an already-serialized JSON string')
        if len(serialized_json_str.encode('utf-8')) > self.max_state_bytes:
            raise ValueError('State snapshot exceeds max_state_bytes')
        with self._condition:
            self._accepting_locked()
            self._stats['stateSnapshotsSubmitted'] += 1
            if self._state is not None:
                self._stats['stateSnapshotsCoalesced'] += 1
            self._state = serialized_json_str
            self._condition.notify()

    def append_frame(self, serialized_json_line):

        if not isinstance(serialized_json_line, str):
            raise TypeError('Frame must be an already-serialized JSON line')
        line = serialized_json_line if serialized_json_line.endswith('\n') else serialized_json_line + '\n'
        if '\n' in line[:-1] or '\r' in line:
            raise ValueError('Frame record must be a single JSON line')
        size = len(line.encode('utf-8'))
        if size > self.max_frame_bytes:
            raise ValueError('Frame record exceeds max_frame_bytes')
        with self._condition:
            self._accepting_locked()
            while (self._outstanding_rows >= self.max_frame_rows
                   or self._outstanding_bytes + size > self.max_frame_bytes):
                self._condition.wait()
                self._accepting_locked()
            self._frames.append((line, size))
            self._outstanding_rows += 1
            self._outstanding_bytes += size
            self._stats['frameLinesSubmitted'] += 1
            self._stats['highWaterFrameRows'] = max(self._stats['highWaterFrameRows'], self._outstanding_rows)
            self._stats['highWaterFrameBytes'] = max(self._stats['highWaterFrameBytes'], self._outstanding_bytes)
            self._condition.notify()

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
            return {**self._stats,
                    'pendingFrameRows': self._outstanding_rows,
                    'pendingFrameBytes': self._outstanding_bytes,
                    'pendingStateSnapshots': int(self._state is not None),
                    'inflightStateSnapshots': int(self._state_inflight),
                    'closing': self._closing, 'writerAlive': self._thread.is_alive(),
                    'failed': self._error is not None}

    def _write_state(self, serialized):
        target = self.root / 'state.json'
        temporary = self.root / '.state-metadata.tmp'
        temporary.write_text(serialized, encoding='utf-8')
        temporary.replace(target)

    def _append_frames(self, lines):
        with (self.root / 'frame-publication.jsonl').open('a', encoding='utf-8') as handle:
            handle.writelines(lines)

    def _run(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            while True:
                with self._condition:
                    while not self._frames and self._state is None and not self._closing:
                        self._condition.wait()
                    if not self._frames and self._state is None and self._closing:
                        return
                    frames = [self._frames.popleft()
                              for _ in range(min(len(self._frames), self.batch_rows))]
                    state, self._state = self._state, None
                    self._state_inflight = state is not None

                if frames:
                    started = time.perf_counter()
                    try:
                        self._append_frames(tuple(line for line, _ in frames))
                    finally:
                        elapsed = time.perf_counter() - started
                        with self._condition:
                            self._stats['maxFrameWriteSeconds'] = max(self._stats['maxFrameWriteSeconds'], elapsed)
                    count, size = len(frames), sum(size for _, size in frames)
                    frames = []
                    with self._condition:
                        self._stats['frameLinesWritten'] += count
                        self._outstanding_rows -= count
                        self._outstanding_bytes -= size
                        self._condition.notify_all()
                if state is not None:
                    started = time.perf_counter()
                    try:
                        self._write_state(state)
                    finally:
                        elapsed = time.perf_counter() - started
                        with self._condition:
                            self._stats['maxStateWriteSeconds'] = max(self._stats['maxStateWriteSeconds'], elapsed)
                    with self._condition:
                        self._stats['stateSnapshotsWritten'] += 1
                        self._state_inflight = False
        except BaseException as error:
            with self._condition:
                self._error = error
                self._closing = True
                self._condition.notify_all()
