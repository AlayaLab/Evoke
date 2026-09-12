from collections import OrderedDict, deque
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import threading


class FrameGap(ValueError):
    def __init__(self, expected, oldest, latest):
        self.details = {'expectedFrame': expected, 'oldestFrame': oldest, 'latestFrame': latest}
        super().__init__('Playback frames are no longer available')


class FrameHub:
    def __init__(self):
        self.frames = OrderedDict()
        self.lock = threading.Lock()

    def put(self, sid, index, jpeg):
        with self.lock:
            ring = self.frames.setdefault(sid, deque(maxlen=96))
            self.frames.move_to_end(sid)
            while len(self.frames) > 4:
                self.frames.popitem(last=False)
            if ring and index <= ring[-1][0]:
                raise ValueError('Frames must be ordered')
            ring.append((index, jpeg))

    def latest_index(self, sid):
        with self.lock:
            ring = self.frames.get(sid)
            return ring[-1][0] if ring else 0

    def next(self, sid, sent, continuous=False):
        with self.lock:
            ring = self.frames.get(sid)
            if not ring:
                return None
            if continuous and sent + 1 < ring[0][0]:
                raise FrameGap(sent + 1, ring[0][0], ring[-1][0])
            minimum = sent + 1 if continuous else max(sent + 1, ring[-1][0] - 8)
            if continuous:
                entry = next((entry for entry in ring if entry[0] >= minimum), None)
                if entry and entry[0] != minimum:
                    raise FrameGap(minimum, entry[0], ring[-1][0])
                return entry
            return next((entry for entry in ring if entry[0] >= minimum), None)


hub = FrameHub()


def serve(path):
    address = Path(path)
    address.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    keyfile = address.with_suffix('.key')
    key = os.urandom(32)
    fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(key)
    address.unlink(missing_ok=True)
    listener = Listener(str(address), family='AF_UNIX', authkey=key)
    os.chmod(address, 0o600)

    def receive(connection):
        try:
            while True:
                data = connection.recv_bytes(4 * 1024 * 1024)
                header, jpeg = data.split(b'\n', 1)
                sid, index = json.loads(header)
                if not isinstance(sid, str) or len(sid) != 32 or not isinstance(index, int):
                    raise ValueError('Invalid frame metadata')
                hub.put(sid, index, jpeg)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            connection.close()

    def accept():
        while True:
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=receive, args=(connection,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return listener


class Publisher:
    def __init__(self, path, sid):
        self.path, self.sid, self.connection = Path(path), sid, None

    def send(self, index, jpeg):
        if self.connection is None:
            self.connection = Client(str(self.path), family='AF_UNIX', authkey=self.path.with_suffix('.key').read_bytes())
        self.connection.send_bytes(json.dumps([self.sid, index]).encode() + b'\n' + jpeg)

    def close(self):
        if self.connection is not None:
            self.connection.close()
