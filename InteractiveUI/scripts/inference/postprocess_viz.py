

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _apply_joystick_hud(video_np: np.ndarray, p: dict) -> np.ndarray:


    npy = p.get("joystick_c2ws_npy")
    if not npy or not Path(npy).is_file():
        return video_np
    try:
        from evoke.utils.ev_validation import add_joystick_overlay_from_c2ws
        c2ws = np.load(npy)

        s = int(p.get("joystick_ref_pix", 0))
        e = min(s + int(video_np.shape[0]), int(c2ws.shape[0]))
        if e <= s:
            return video_np
        n = e - s
        out = add_joystick_overlay_from_c2ws(
            list(video_np[:n]), c2ws[s:e],
            move_scale=p.get("joystick_move_scale"), rot_scale=p.get("joystick_rot_scale"),
            label_left="Move", label_right="Rot",
        )

        merged = np.stack(list(out) + list(video_np[n:])) if n < video_np.shape[0] else np.stack(list(out))
        print(f"[postproc] joystick HUD drawn on {n}/{video_np.shape[0]} pred frames", flush=True)
        return merged
    except Exception as e:
        print(f"[postproc] [WARN] joystick HUD skipped: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return video_np


def _write_pred_outputs(output_dir: Path, video_np: np.ndarray, fps: int = 24, params: dict = None) -> None:


    if os.environ.get("EVOKE_DUMP_PRED_PNG", "0") == "1":
        pf_dir = output_dir / "pred_frames"
        pf_dir.mkdir(parents=True, exist_ok=True)
        for i, f_rgb in enumerate(video_np):


            Image.fromarray(f_rgb).save(pf_dir / f"{i:05d}.png", compress_level=1)
        print(f"[postproc] pred frames (lossless png): {pf_dir} ({len(video_np)} frames)", flush=True)
    p = params or {}

    def _encode(frames, path):
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
        for f_rgb in frames:
            writer.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[postproc] pred mp4 saved: {path}", flush=True)

    pred_mp4 = output_dir / "geo_pred.mp4"
    if p.get("joystick_dual"):


        _encode(_apply_joystick_hud(video_np, p), output_dir / "geo_pred_hud.mp4")
        _encode(video_np, pred_mp4)
    else:
        video_np = _apply_joystick_hud(video_np, p)
        _encode(video_np, pred_mp4)


    if os.environ.get("EVOKE_DUMP_SAMPLE_FRAMES", "1") == "1":
        sf_dir = output_dir / "sample_frames"
        sf_dir.mkdir(parents=True, exist_ok=True)
        n = video_np.shape[0]
        for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            Image.fromarray(video_np[i]).save(sf_dir / f"pred_{i:03d}.png")


def _concat_chunk_mp4s(geo_dir: Path, suffix: str) -> list:

    frames: list = []
    if not geo_dir.is_dir():
        return frames
    for mp4 in sorted(geo_dir.glob(f"chunk_*_{suffix}.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        while True:
            ret, bf = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bf, cv2.COLOR_BGR2RGB))
        cap.release()
    return frames


def _append_warp_mask_columns(output_dir: Path, combined: list) -> list:


    if not combined:
        return combined
    geo_dir = output_dir / "geo_debug"
    warp = _concat_chunk_mp4s(geo_dir, "warp")
    vis = _concat_chunk_mp4s(geo_dir, "vis")
    if not warp and not vis:
        return combined
    H, W = combined[0].shape[:2]
    pw = W // 2
    black = np.zeros((H, pw, 3), dtype=np.uint8)

    def _fit(seq, i):
        f = seq[i] if i < len(seq) else black
        return cv2.resize(f, (pw, H)) if (f.shape[0] != H or f.shape[1] != pw) else f

    out = []
    for i, base in enumerate(combined):
        parts = [base]
        if warp:
            parts.append(_fit(warp, i))
        if vis:
            parts.append(_fit(vis, i))
        out.append(np.concatenate(parts, axis=1))
    print(f"[postproc] appended warp/mask cols: warp={len(warp)}f vis={len(vis)}f "
          f"-> {len(parts)} panels", flush=True)
    return out


def _write_combined(output_dir: Path, video_np: np.ndarray, p: dict) -> None:

    from evoke.utils.ev_validation import combine_gt_pred_with_cam_viz

    _t_combine = time.time()
    print("[postproc] combining GT|Pred + joystick HUD ...", flush=True)
    combined = combine_gt_pred_with_cam_viz(
        pred_video=video_np,
        gt_video_path=str(p["ref_video_for_viz"]),
        pose_path=str(p["lingbot_pose_path"]),
        height=int(p["height"]), width=int(p["width"]),
        num_frames=int(p["num_frames"]),
        ref_seconds=float(p["ref_seconds"]), start_seconds=float(p["start_seconds"]),
        latent_window_size=9, vae_stride_t=4,
        target_fps=24, source_fps=int(p["lingbot_pose_source_fps"]),
        source_resolution=tuple(p["lingbot_pose_source_resolution"]),
        pose_type=p["lingbot_pose_type"],
        fallback_default_intrinsic=bool(p.get("lingbot_fallback_default_intrinsic", False)),
        panel_label_gt="GT", panel_label_pred="Pred(GEO)" if p["use_geometric_state"] else "Pred",
    )

    combined = _append_warp_mask_columns(output_dir, combined)
    sbs_mp4 = output_dir / "gt_vs_pred_cam_viz.mp4"
    H_out, W_out = combined[0].shape[:2]
    _t_write = time.time()
    w2 = cv2.VideoWriter(str(sbs_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (int(W_out), int(H_out)))
    for f_rgb in combined:
        w2.write(cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR))
    w2.release()
    print(f"[postproc] GT|Pred mp4 saved: {sbs_mp4} ({len(combined)} frames, {H_out}x{W_out}) | "
          f"combine {_t_write - _t_combine:.1f}s + write {time.time() - _t_write:.1f}s = "
          f"{time.time() - _t_combine:.1f}s total", flush=True)


def _ffmpeg_concat(seg_paths: list, out_mp4: Path, list_txt: Path) -> bool:


    import subprocess
    list_txt.write_text("".join(f"file '{p.resolve()}'\n" for p in seg_paths))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out_mp4)]
    try:
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode
    except FileNotFoundError:
        print("[assemble] ffmpeg unavailable -> falling back to per-frame cv2 re-encode", flush=True)
        return False
    if rc == 0 and out_mp4.is_file() and out_mp4.stat().st_size > 0:
        return True
    print(f"[assemble] ffmpeg -c copy failed (rc={rc}) -> falling back to per-frame cv2 re-encode", flush=True)
    return False


def _cv2_concat(seg_paths: list, out_mp4: Path, fps: int) -> bool:


    writer = None
    n_written = 0
    for seg in seg_paths:
        cap = cv2.VideoCapture(str(seg))
        while True:
            ret, bf = cap.read()
            if not ret:
                break
            if writer is None:
                h, w = bf.shape[:2]
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (int(w), int(h)))
            writer.write(bf)
            n_written += 1
        cap.release()
    if writer is not None:
        writer.release()
    return n_written > 0


def _mp4_frame_count(mp4: Path) -> int:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def assemble_from_segments(output_dir, fps: int = 24, sample_n: int = 8) -> dict:


    output_dir = Path(output_dir)
    seg_dir = output_dir / "segments"
    result = {"pred": None, "sbs": None, "n_pred_seg": 0, "n_sbs_seg": 0}
    if not seg_dir.is_dir():
        print(f"[assemble] [WARN] no segments directory: {seg_dir} - cannot stitch", flush=True)
        return result

    def _segments(kind: str) -> list:


        pat = re.compile(rf"^segment_(\d+)_{re.escape(kind)}\.mp4$")
        found = []
        for p in seg_dir.iterdir():
            m = pat.match(p.name)
            if m and p.is_file() and p.stat().st_size > 0:
                found.append((int(m.group(1)), p))
        return [p for _, p in sorted(found, key=lambda t: t[0])]

    jobs = [("pred", "pred", output_dir / "geo_pred.mp4"),
            ("sbs", "gt_vs_pred", output_dir / "gt_vs_pred_cam_viz.mp4")]
    for key, kind, out_mp4 in jobs:
        segs = _segments(kind)
        result[f"n_{key}_seg"] = len(segs)
        if not segs:
            print(f"[assemble] segment_*_{kind}.mp4: 0 segments, skipping {out_mp4.name}", flush=True)
            continue
        t0 = time.time()
        ok = _ffmpeg_concat(segs, out_mp4, seg_dir / f".concat_{key}.txt")
        if not ok:
            ok = _cv2_concat(segs, out_mp4, fps)
        if ok:
            n = _mp4_frame_count(out_mp4)
            size_mb = out_mp4.stat().st_size / 1024**2
            result[key] = out_mp4
            print(f"[assemble] {out_mp4.name}: {len(segs)} segments -> {n} frames "
                  f"({n / max(fps, 1) / 60.0:.1f} min, {size_mb:.0f} MB, {time.time() - t0:.1f}s)", flush=True)
        else:
            print(f"[assemble] [WARN] {out_mp4.name} stitching failed", flush=True)


    pred_segs = _segments("pred") if os.environ.get("EVOKE_DUMP_SAMPLE_FRAMES", "1") == "1" else []
    if pred_segs:
        sf_dir = output_dir / "sample_frames"
        sf_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, len(pred_segs) // max(1, sample_n))
        for seg in pred_segs[::step][:sample_n]:
            cap = cv2.VideoCapture(str(seg))
            ret, bf = cap.read()
            cap.release()
            if ret:
                Image.fromarray(cv2.cvtColor(bf, cv2.COLOR_BGR2RGB)).save(
                    sf_dir / f"pred_{seg.stem.split('_')[1]}.png")
    return result


def run_postprocess(output_dir, video_np, params: dict) -> None:

    output_dir = Path(output_dir)
    _write_pred_outputs(output_dir, video_np, fps=int(params.get("fps", 24)), params=params)
    if params.get("ref_video_for_viz") and params.get("lingbot_pose_path"):
        try:
            _write_combined(output_dir, video_np, params)
        except Exception as e:
            print(f"[postproc] [WARN] combine_gt_pred_with_cam_viz failed: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)


def _main_cli() -> None:

    out_dir = Path(sys.argv[1])
    params = json.loads((out_dir / ".postproc_params.json").read_text())
    npy = out_dir / ".pred_frames.npy"
    video_np = np.load(npy)
    print(f"[postproc] worker start: {out_dir} (pred {tuple(video_np.shape)})", flush=True)
    try:
        run_postprocess(out_dir, video_np, params)
    finally:

        for f in (npy, out_dir / ".postproc_params.json"):
            try:
                f.unlink()
            except OSError:
                pass
        marker = params.get("_worker_marker")
        if marker:
            try:
                Path(marker).unlink()
            except OSError:
                pass
    print("[postproc] DONE.", flush=True)


if __name__ == "__main__":
    _main_cli()
