from __future__ import annotations

import math
import threading
import time


class PlaybackPublicationGate:


    def __init__(self, read_control, *, limit=96, wall_clock=None):
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer')
        self.limit = limit
        self._read_control = read_control
        self._clock = wall_clock or time.time
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._ack = 0
        self._updated_at = None
        self._stop = False
        self._terminal_reason = None
        self._paused = False
        self._connected = False
        self._stats = dict(waitCalls=0, allowed=0, denied=0, capacityWaits=0,
                           capacityWaitSeconds=0., maxCapacityWaitSeconds=0.,
                           refreshes=0, readFailures=0, invalidSnapshots=0,
                           staleSnapshots=0, observations=0)

    def observe(self, control):

        with self._lock:
            self._stats['observations'] += 1
            if not isinstance(control, dict):
                self._stats['invalidSnapshots'] += 1
                return

            if control.get('stop') is True:
                self._stop = True
            stamp = control.get('updatedAt')
            ack = control.get('playedFrame', 0)
            if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                    or type(ack) is not int or ack < 0):
                self._stats['invalidSnapshots'] += 1
                return
            self._ack = max(self._ack, ack)
            if self._updated_at is not None and stamp < self._updated_at:
                self._stats['staleSnapshots'] += 1
                return
            self._updated_at = stamp
            self._paused = bool(control.get('paused', False))
            self._connected = bool(control.get('connected', False))

    def _refresh(self):
        with self._lock:
            self._stats['refreshes'] += 1

        try:
            control = self._read_control()
        except (OSError, ValueError):
            with self._lock:
                self._stats['readFailures'] += 1
            return
        self.observe(control)

    def abort(self):

        self._abort.set()

    def wait_for_slot(self, published):
        if type(published) is not int or published < 0:
            raise ValueError('published must be a nonnegative integer')
        with self._lock:
            self._stats['waitCalls'] += 1
        waited_since = None
        try:
            while True:
                with self._lock:
                    stamp, ack, stop = self._updated_at, self._ack, self._stop
                    terminal = self._terminal_reason
                if self._abort.is_set() or stop or terminal is not None:
                    return self._deny('aborted' if self._abort.is_set() else ('stop' if stop else terminal))
                now = self._clock()


                if stamp is None or now - stamp >= .8 or published - ack >= self.limit:
                    self._refresh()
                with self._lock:
                    stamp, ack, stop = self._updated_at, self._ack, self._stop
                if self._abort.is_set() or stop:
                    return self._deny('aborted' if self._abort.is_set() else 'stop')
                if stamp is None or self._clock() - stamp > 15:
                    return self._deny('expired')
                if published - ack < self.limit:
                    with self._lock:
                        self._stats['allowed'] += 1
                    return True
                if waited_since is None:
                    waited_since = time.perf_counter()
                    with self._lock:
                        self._stats['capacityWaits'] += 1


                self._abort.wait(.02)
        finally:
            if waited_since is not None:
                duration = time.perf_counter() - waited_since
                with self._lock:
                    self._stats['capacityWaitSeconds'] += duration
                    self._stats['maxCapacityWaitSeconds'] = max(self._stats['maxCapacityWaitSeconds'], duration)

    def _deny(self, reason):
        with self._lock:
            if self._terminal_reason is None:
                self._terminal_reason = reason
            self._stats['denied'] += 1
        return False

    def summary(self):
        with self._lock:
            return dict(self._stats, limit=self.limit, playedAck=self._ack,
                        controlUpdatedAt=self._updated_at, paused=self._paused,
                        connected=self._connected, stopLatched=self._stop,
                        aborted=self._abort.is_set(), terminalReason=self._terminal_reason)
