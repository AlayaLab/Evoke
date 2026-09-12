

import importlib
import pkgutil
import threading

from depth_anything_3.utils.registry import Registry

__all__ = ["METRIC_REGISTRY", "MONO_REGISTRY", "MV_REGISTRY", "NVS_REGISTRY"]


_loaded = False
_lock = threading.Lock()


def _import_all_datasets_once():


    global _loaded
    if _loaded:
        return

    with _lock:
        if _loaded:
            return

        pkg_name = "depth_anything_3.bench.datasets"
        pkg = importlib.import_module(pkg_name)
        pkg_paths = list(getattr(pkg, "__path__", []))

        for finder, name, ispkg in pkgutil.walk_packages(pkg_paths, prefix=pkg_name + "."):
            base = name.rsplit(".", 1)[-1]
            if base.startswith("_"):
                continue
            try:
                importlib.import_module(name)
            except Exception as e:
                print(f"[datasets auto-import] Failed to import {name}: {e}")

        _loaded = True


class AutoRegistry(Registry):


    def get(self, name):
        _import_all_datasets_once()
        return super().get(name)

    def all(self):
        _import_all_datasets_once()
        return super().all()

    def has(self, name):
        _import_all_datasets_once()
        return name in self._map


METRIC_REGISTRY = AutoRegistry()
MONO_REGISTRY = AutoRegistry()
MV_REGISTRY = AutoRegistry()
NVS_REGISTRY = AutoRegistry()
