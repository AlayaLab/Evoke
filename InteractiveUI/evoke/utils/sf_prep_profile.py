

import os
import time

_ON = os.environ.get("SF_PREP_PROFILE", "0") == "1"
_acc: dict = {}
_meta: dict = {}


def enabled() -> bool:
    return _ON


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def mark():

    if not _ON:
        return None
    _sync()
    return time.time()


def accum(key: str, t0, note: str = ""):

    if not _ON or t0 is None:
        return
    _sync()
    _acc[key] = _acc.get(key, 0.0) + (time.time() - t0)
    if note:
        _meta[key] = note


def step_reset():
    if _ON:
        _acc.clear()
        _meta.clear()


def report(prefix: str = "") -> str:

    if not _ON or not _acc:
        return ""
    tot = sum(_acc.values())
    parts = []
    for k in sorted(_acc, key=lambda x: -_acc[x]):
        v = _acc[k]
        n = f"({_meta[k]})" if k in _meta else ""
        parts.append(f"{k}={v:.2f}s{n}")
    return f"[PREP-PROFILE]{prefix} total={tot:.2f}s | " + " ".join(parts)


_GT_VERIFY_N = int(os.environ.get("SF_GT_VERIFY_N", "3") or 3)
_gt_verify_done: dict = {}


def gt_verify_take(key=None) -> bool:


    n = _gt_verify_done.get(key, 0)
    if n >= _GT_VERIFY_N:
        return False
    _gt_verify_done[key] = n + 1
    return True


def vae_rf_probe(vae, img, T_px: int, latents_mean, latents_std) -> str:


    import torch
    assert img.dim() == 5, f"[VAE-RF-PROBE] expected raw_video as [B,C,T,H,W], got {tuple(img.shape)}"
    with torch.no_grad():
        z = torch.zeros(1, img.shape[1], T_px, img.shape[3], img.shape[4],
                        device=img.device, dtype=img.dtype)
        a = vae.encode(z.clone()).latent_dist.mode()
        z[:, :, :1] = img[:1, :, :1]
        b = vae.encode(z).latent_dist.mode()

    per_t = (a - b).abs().amax(dim=(0, 1, 3, 4))
    T_lat = per_t.numel()
    first_zero = next((i for i in range(T_lat) if bool((per_t[i:] == 0).all())), None)
    nz = [i for i in range(T_lat) if float(per_t[i]) != 0.0]
    return (f"[VAE-RF-PROBE] T_px={T_px} T_lat={T_lat} | latent frames affected by the first frame={nz[:12]}"
            f"{'...' if len(nz) > 12 else ''} {len(nz)} in total | "
            f"bit-identical to all-zero input from frame {first_zero} on -> "
            f"{'the last ' + str(T_lat - first_zero) + ' frames are cacheable (' + f'{(T_lat-first_zero)/T_lat:.0%}' + ')' if first_zero is not None else 'no safe boundary: the whole sequence is affected'}"
            f" | largest differences={[round(float(per_t[i]),6) for i in range(min(6,T_lat))]}")
