
import os

from omegaconf import DictConfig, OmegaConf


def load_yaml_with_inheritance(path: str) -> DictConfig:
    return _load(os.path.abspath(path), frozenset())


def _load(abspath: str, seen: frozenset) -> DictConfig:
    if abspath in seen:
        raise ValueError(f"circular _base_ dependency at {abspath}")
    seen = seen | {abspath}

    raw = OmegaConf.load(abspath)
    bases = raw.pop("_base_", None) if isinstance(raw, DictConfig) else None
    if not bases:
        return raw

    if isinstance(bases, str):
        bases = [bases]

    here = os.path.dirname(abspath)
    merged = OmegaConf.create({})
    for b in bases:
        bp = b if os.path.isabs(b) else os.path.join(here, b)
        merged = OmegaConf.merge(merged, _load(os.path.abspath(bp), seen))
    return OmegaConf.merge(merged, raw)
