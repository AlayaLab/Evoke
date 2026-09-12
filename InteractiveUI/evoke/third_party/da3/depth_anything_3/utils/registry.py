

from typing import Any
from addict import Dict


class Registry(Dict[str, Any]):
    def __init__(self):
        super().__init__()
        self._map = Dict({})

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__
            self._map[key] = cls
            return cls

        return decorator

    def get(self, name):
        return self._map[name]

    def all(self):
        return self._map
