import asyncio
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterator

_EVENT_END = re.compile(rb"\r\n\r\n|\n\n")

_PARTIAL_ENDS = (b"\r\n\r", b"\r\n", b"\r", b"\n")


class _EventBoundaryPadding:
    def __init__(self, padding: bytes, padding_due):
        self.padding = padding
        self.padding_due = padding_due
        self.tail = b""
        self.at_boundary = True

    def feed(self, block: bytes) -> Iterator[bytes]:
        data = self.tail + block
        self.tail = b""
        offset = 0
        for match in _EVENT_END.finditer(data):
            yield data[offset:match.end()]
            self.at_boundary = True
            if self.padding_due():
                yield self.padding
            offset = match.end()
        remainder = data[offset:]
        if remainder:
            self.at_boundary = False
            keep = next((len(s) for s in _PARTIAL_ENDS if remainder.endswith(s)), 0)
            if keep:
                self.tail = remainder[-keep:]
                remainder = remainder[:-keep]
            if remainder:
                yield remainder

    def finish(self) -> bytes:
        tail, self.tail = self.tail, b""
        return tail


async def relay_sse(
    source: AsyncIterable[bytes], *, idle_seconds: float = 0.25,
    padding_bytes: int = 16 * 1024, event_padding_interval: float = 0.25,
) -> AsyncIterator[bytes]:


    if idle_seconds <= 0 or padding_bytes < 3 or event_padding_interval < 0:
        raise ValueError("positive idle_seconds, padding_bytes >= 3, nonnegative event_padding_interval required")
    padding = b":" + b" " * (padding_bytes - 3) + b"\n\n"
    last_padding = float("-inf")
    def padding_due():
        nonlocal last_padding
        now = time.monotonic()
        if now - last_padding < event_padding_interval:
            return False
        last_padding = now
        return True
    parser = _EventBoundaryPadding(padding, padding_due)
    iterator = source.__aiter__()
    pending = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            ready, _ = await asyncio.wait({pending}, timeout=idle_seconds)
            if not ready:


                if parser.at_boundary:
                    last_padding = time.monotonic()
                    yield padding
                continue
            finished, pending = pending, None
            try:
                block = finished.result()
            except StopAsyncIteration:
                tail = parser.finish()
                if tail:
                    yield tail
                break
            if block:
                for output in parser.feed(block):
                    yield output
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
