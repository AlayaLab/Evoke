from collections import OrderedDict
import time


class LivePromptConditioning:
    def __init__(self, live, initial_prompt, initial_embeddings, encode, cache_size=4):
        self.live = live
        self.current = initial_embeddings
        self.encode = encode
        self.cache_size = cache_size
        self.cache = OrderedDict()
        if isinstance(initial_prompt, str):
            self.cache[initial_prompt] = initial_embeddings

    def for_chunk(self):
        request = self.live.next_prompt_request()
        if request is None:
            return self.current
        started = time.perf_counter()
        text = request['prompt']
        cached = text in self.cache
        try:
            if cached:
                candidate = self.cache[text]
                self.cache.move_to_end(text)
            else:
                candidate = self.encode(text)
                self.cache[text] = candidate
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        except Exception as error:


            self.live.prompt_failed(request, str(error)[:400] or type(error).__name__)
            return self.current
        self.current = candidate
        self.live.prompt_applied(request, time.perf_counter() - started, cached)
        return self.current
