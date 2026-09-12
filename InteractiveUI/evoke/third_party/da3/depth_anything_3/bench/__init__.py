

from depth_anything_3.bench.registries import MV_REGISTRY, MONO_REGISTRY


def __getattr__(name):

    if name == "Evaluator":
        from depth_anything_3.bench.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Evaluator", "MV_REGISTRY", "MONO_REGISTRY"]
