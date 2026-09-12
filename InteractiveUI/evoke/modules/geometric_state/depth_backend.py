

from __future__ import annotations

from typing import Optional

_VIGEO_KEYS = ("num_tokens", "mode", "chunk_size", "intr_source", "conf_transform",
               "scale_mode", "anchor_windows", "total_budget", "cache_keep_frames",


               "scale_value", "depth_median_target")


def build_estimator(backend: str, device, process_res: int,
                    weights: Optional[str] = None, src: Optional[str] = None,
                    vigeo_opts: Optional[dict] = None):


    b = (backend or "da3").lower()
    if b == "vigeo":
        from .vigeo_cloud import ViGeoDepthEstimator, _VIGEO_WEIGHTS
        kwargs = dict(device=device, process_res=int(process_res),
                      weights=(weights or str(_VIGEO_WEIGHTS)))
        if src:
            kwargs["src"] = src
        unknown = sorted(set(vigeo_opts or {}) - set(_VIGEO_KEYS))
        if unknown:
            raise ValueError(f"unknown ViGeo option(s) {unknown}; expected any of {sorted(_VIGEO_KEYS)}")
        for k in _VIGEO_KEYS:
            v = (vigeo_opts or {}).get(k)
            if v is not None:
                kwargs[k] = v
        return ViGeoDepthEstimator(**kwargs)
    if b != "da3":
        raise ValueError(f"unknown cloud_warp depth backend {backend!r}; expected 'da3' or 'vigeo'")
    from .da3_cloud import DA3DepthEstimator, _DA3_WEIGHTS
    kwargs = dict(device=device, process_res=int(process_res),
                  weights=(weights or str(_DA3_WEIGHTS)))
    if src:
        kwargs["src"] = src
    return DA3DepthEstimator(**kwargs)


def reset_stream(estimator) -> None:


    fn = getattr(estimator, "reset_stream", None)
    if callable(fn):
        fn()


def check_assets(backend: str, weights: Optional[str] = None, src: Optional[str] = None) -> None:


    if (backend or "da3").lower() != "vigeo":
        return
    from .vigeo_cloud import check_assets as _check
    _check(src, weights)
