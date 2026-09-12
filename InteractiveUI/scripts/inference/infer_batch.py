

import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)

def _env(k, d): return os.environ.get(k, d)


def _drain_bg_postprocess(out_root: Path, timeout_s: float = 1800.0, poll: float = 3.0) -> int:


    import time as _t
    wdir = out_root / ".pp_workers"
    if not wdir.is_dir():
        return 0
    deadline = _t.time() + timeout_s
    while True:
        live = 0
        for m in list(wdir.glob("*.lock")):
            try:
                pid = int((m.read_text().strip() or "0"))
            except (ValueError, OSError):
                pid = 0
            if pid <= 0:
                live += 1
                continue
            try:
                os.kill(pid, 0)
                live += 1
            except ProcessLookupError:
                try:
                    m.unlink()
                except OSError:
                    pass
            except PermissionError:
                live += 1
        if live == 0:
            return 0
        if _t.time() >= deadline:
            print(f"[warn] {live} postprocess worker(s) still in flight after {timeout_s:.0f}s; "
                  f"their cases will be reported FAIL and can be resumed", flush=True)
            return live
        print(f"[bg-drain] waiting on {live} postprocess worker(s) ...", flush=True)
        _t.sleep(poll)


SHARD = int(_env("SHARD", "0")); NSHARD = int(_env("NSHARD", "1"))


VERBOSE = _env("INFER_VERBOSE", "1" if NSHARD == 1 else "0") == "1"


JSONL = _env("JSONL", "examples/v2v/cases.jsonl")
MAX_CASES = int(_env("MAX_CASES", "0"))


VROOT = _env("VROOT", ""); AROOT = _env("AROOT", "")
OUT_ROOT = Path(_env("OUT_ROOT", "output_evoke/infer/batch"))


BASE_CKPT = _env("BASE_CKPT", "models/evoke-base")

TRANSFORMER_PATH = _env("TRANSFORMER_PATH", "")


HEIGHT = _env("HEIGHT", "384"); WIDTH = _env("WIDTH", "640"); FPS = _env("FPS", "24")
NUM_CHUNKS = int(_env("NUM_CHUNKS", "0"))
NUM_FRAMES = str(33 * NUM_CHUNKS) if NUM_CHUNKS > 0 else _env("NUM_FRAMES", "1437")
NUM_INFERENCE_STEPS = _env("NUM_INFERENCE_STEPS", "3")
GUIDANCE_SCALE = _env("GUIDANCE_SCALE", "1.0"); SEED = _env("SEED", "44")
START_SECONDS = _env("START_SECONDS", "0.0"); REF_VIDEO_SEC = _env("REF_VIDEO_SEC", "5.0")


MODE = _env("MODE", "v2v")
if MODE not in ("v2v", "i2v", "t2v"):
    sys.exit(f"[ERROR] MODE must be one of v2v/i2v/t2v, got {MODE!r}")


WARP = _env("WARP", "on")
if WARP not in ("on", "off"):
    sys.exit(f"[ERROR] WARP must be on|off, got {WARP!r}")
GEO_ON = MODE != "t2v" and WARP == "on"
IS_STAGE2 = _env("IS_STAGE2", "1")
STAGE2_NUM_STAGES = _env("STAGE2_NUM_STAGES", "3")
STAGE2_STEPS = _env("STAGE2_STEPS", "1 1 1").split()
STAGE2_STAGE_RANGE = _env("STAGE2_STAGE_RANGE", "0 0.3333333333333333 0.6666666666666666 1").split()


USE_DMD = _env("USE_DMD", "0")
AMPLIFY_FIRST_CHUNK = _env("AMPLIFY_FIRST_CHUNK", "0")


IN_PROCESS_BATCH = _env("IN_PROCESS_BATCH", "1")


ARGV_SERVER_DIR = _env("EVOKE_ARGV_SERVER_DIR", "")
LIVE_CONTROL_PATH = _env("EVOKE_LIVE_CONTROL_PATH", "")
SERVER_PRELOAD = _env("EVOKE_SERVER_PRELOAD", "0") == "1"
SERVER_REQUEST_ID = _env("EVOKE_SERVER_REQUEST_ID", "")


DUMP_GEO = _env("DUMP_GEO", "1")
SAVE_SEGMENTS = _env("SAVE_SEGMENTS", "1")


BG_POSTPROC = _env("BG_POSTPROC", "0")
BG_POSTPROC_MAX = _env("BG_POSTPROC_MAX", "4")
if BG_POSTPROC == "1" and IN_PROCESS_BATCH != "1":


    sys.exit("[ERROR] BG_POSTPROC=1 requires IN_PROCESS_BATCH=1 (the per-case path checks DONE_MARK "
             "before the detached worker has written it).")
WARP_MODE = _env("WARP_MODE", "fixed_mem")
NOISE_CENTER = _env("NOISE_CENTER", "1")


RENDER_MODE = _env("RENDER_MODE", "backward_zbuf")
BW_FILL_ITERS = _env("BW_FILL_ITERS", "12")
ZBUF_KSIZE = _env("ZBUF_KSIZE", "3"); ZBUF_FILL = _env("ZBUF_FILL", "4")
WARP_SIGMA_MAX = _env("WARP_SIGMA_MAX", "0.333")
WARP_LAG = _env("WARP_LAG", "0")
POSE_TYPE = _env("POSE_TYPE", "vipe")
DA3_SRC = _env("DA3_SRC", os.environ.get("EVOKE_DA3_SRC", ""))
DA3_WEIGHTS = _env("DA3_WEIGHTS", "models/DA3"); DA3_PROCESS_RES = _env("DA3_PROCESS_RES", "644")


DEPTH_BACKEND = _env("DEPTH_BACKEND", "vigeo")
VIGEO_SRC = _env("VIGEO_SRC", os.environ.get("EVOKE_VIGEO_SRC", ""))
VIGEO_WEIGHTS = _env("VIGEO_WEIGHTS", "models/ViGeo1.1")
VIGEO_MODE = _env("VIGEO_MODE", "chunk"); VIGEO_SCALE_MODE = _env("VIGEO_SCALE_MODE", "auto")


VIGEO_SCALE_VALUE = _env("VIGEO_SCALE_VALUE", "0.0")


VIGEO_DEPTH_MEDIAN_TARGET = _env("VIGEO_DEPTH_MEDIAN_TARGET", "5")
VIGEO_ANCHOR_WINDOWS = _env("VIGEO_ANCHOR_WINDOWS", "4")
VIGEO_CACHE_KEEP_FRAMES = _env("VIGEO_CACHE_KEEP_FRAMES", "6")
VIGEO_INTR_SOURCE = _env("VIGEO_INTR_SOURCE", "gt")
RESTRICT = _env("RESTRICT", "1")


PLUCKER = _env("PLUCKER", "0")


JOYSTICK_HUD = _env("JOYSTICK_HUD", "auto")
STAGE0_ONLY = _env("GEO_WARP_STAGE0_ONLY", "0")


CHUNK0_REF_WARP = _env("GEO_CHUNK0_REF_WARP", "1")


CHUNK0_TARGET_DISP_PX = _env("GEO_CHUNK0_TARGET_DISP_PX", "")
ZBUF_DESPECKLE = _env("ZBUF_DESPECKLE", "1")


VAE_DECODE_TYPE = _env("VAE_DECODE_TYPE", "persistent")


STREAM_LONG = _env("STREAM_LONG", "0")


GEO_HIST_MAX = _env("GEO_HIST_MAX_FRAMES", "0")


DONE_MARK = "geo_pred.mp4"
VIZ_MARK = "gt_vs_pred_cam_viz.mp4"


CAP_MAX_DEG = _env("CAP_MAX_DEG", "10"); CAP_MAX_TRANS = _env("CAP_MAX_TRANS", "3")
CAP_MODE = _env("CAP_MODE", "clamp"); POSE_SMOOTH_WIN = _env("POSE_SMOOTH_WIN", "5")


POSE_EXTEND_MODE = _env("POSE_EXTEND_MODE", "relative_replay")

NEG = _env("NEGATIVE_PROMPT", "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background")


def resolve_under(root, rel):


    joined = os.path.join(root, rel)
    if os.path.isfile(joined):
        return joined
    return rel if os.path.isfile(rel) else joined


def load_prompt(prompt_path):

    try:
        cj = json.load(open(prompt_path))
        ov = cj.get("overall", {}) if isinstance(cj, dict) else {}
        for k in ("full_prompt", "short_prompt", "description"):
            v = ov.get(k)
            if isinstance(v, str) and len(v) > 10:
                return v
    except Exception as e:
        print(f"[warn] failed to read caption {prompt_path}: {e}", flush=True)
    return "A first-person walk through an outdoor scene with continuous forward camera motion."


def main():
    if NOISE_CENTER == "1" and WARP_MODE != "fixed_mem":
        sys.exit(f"[ERROR] --warp_rope_noise_center_align requires WARP_MODE=fixed_mem (got {WARP_MODE})")
    if not TRANSFORMER_PATH:
        sys.exit("[ERROR] TRANSFORMER_PATH is required (the parent of the transformer/ dir). "
                 "Run one of the scripts/inference/infer_*.sh launchers, which set it for you.")
    if not (Path(TRANSFORMER_PATH) / "transformer").is_dir():
        sys.exit(f"[ERROR] transformer not found: {Path(TRANSFORMER_PATH) / 'transformer'}\n"
                 f"        TRANSFORMER_PATH must be the PARENT of the transformer/ dir -- the weights are\n"
                 f"        loaded with from_pretrained(TRANSFORMER_PATH, subfolder=\"transformer\").\n"
                 f"        Got TRANSFORMER_PATH={TRANSFORMER_PATH!r}.")


    if int(NUM_FRAMES) > 7200 and STREAM_LONG != "1":
        _gb = int(NUM_FRAMES) * 2.95 / 1024
        sys.exit(f"[ERROR] NUM_FRAMES={NUM_FRAMES} would accumulate the whole clip on GPU (~{_gb:.0f} GB fp32).\n"
                 f"        Set STREAM_LONG=1 to decode per chunk and stitch the final mp4 from segments/\n"
                 f"        (requires VAE_DECODE_TYPE=persistent, the launcher default). For hour-scale\n"
                 f"        rollouts also consider GEO_HIST_MAX_FRAMES -- see scripts/inference/README.md.")
    if STREAM_LONG == "1" and VAE_DECODE_TYPE != "persistent":
        sys.exit(f"[ERROR] STREAM_LONG=1 requires VAE_DECODE_TYPE=persistent (got {VAE_DECODE_TYPE!r}); "
                 f"the full video is stitched from per-chunk segments.")

    recs = [json.loads(l) for l in open(JSONL) if l.strip()]
    if MAX_CASES > 0:
        recs = recs[:MAX_CASES]
    mine = [(i, r) for i, r in enumerate(recs) if i % NSHARD == SHARD]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[{MODE} shard {SHARD}/{NSHARD}] total={len(recs)} mine={len(mine)} frames={NUM_FRAMES} "
          f"warp={'on' if GEO_ON else 'OFF'} transformer={TRANSFORMER_PATH} out={OUT_ROOT}", flush=True)


    if SHARD == 0:
        (OUT_ROOT / "run_info.json").write_text(json.dumps({
            "transformer_path": TRANSFORMER_PATH, "base_ckpt": BASE_CKPT, "mode": MODE, "jsonl": JSONL,
            "geometric_state": GEO_ON,
            "num_frames": NUM_FRAMES, "num_chunks": NUM_CHUNKS or None, "height": HEIGHT, "width": WIDTH,

            "fps": FPS, "seed_default": SEED,
            "seed_per_case": sum(1 for r in recs if r.get("seed") is not None),
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": NUM_INFERENCE_STEPS, "stage2_steps": STAGE2_STEPS if IS_STAGE2 == "1" else None,
            "use_dmd": USE_DMD, "amplify_first_chunk": AMPLIFY_FIRST_CHUNK,


            "vigeo_scale_mode": (VIGEO_SCALE_MODE if VIGEO_SCALE_MODE != "auto"
                                 else ("anchor" if MODE == "v2v" else "depth_median")),
            "vigeo_scale_mode_requested": VIGEO_SCALE_MODE,
            "vigeo_scale_value": VIGEO_SCALE_VALUE,
            "vigeo_depth_median_target": VIGEO_DEPTH_MEDIAN_TARGET,


            "chunk0_target_disparity_px": CHUNK0_TARGET_DISP_PX or "engine-default",
            "cap_max_deg": CAP_MAX_DEG, "cap_max_trans": CAP_MAX_TRANS, "pose_smooth_win": POSE_SMOOTH_WIN,
            "ref_video_sec": REF_VIDEO_SEC if MODE == "v2v" else "0.0", "start_seconds": START_SECONDS,
            "warp_sigma_max": WARP_SIGMA_MAX, "warp_stage0_only": STAGE0_ONLY, "restrict_self_attn": RESTRICT,
            "vae_decode_type": VAE_DECODE_TYPE, "stream_long": STREAM_LONG,
            "joystick_hud": JOYSTICK_HUD,
        }, indent=2) + "\n")

    _rst = ["--restrict_self_attn", "--use_kv_cache"] if RESTRICT == "1" else []
    _plk = ["--geo_warp_plucker_enabled"] if PLUCKER == "1" else []
    _nc = ["--warp_rope_noise_center_align"] if NOISE_CENTER == "1" else []
    done = skip = fail = skip_missing = 0
    pending: list = []
    pending_meta: list = []
    for n, (i, r) in enumerate(mine):


        video = image = pose = None
        if MODE == "v2v":


            name = r.get("name") or Path(r["video_path"]).stem
            video = resolve_under(VROOT, r["video_path"])
            pose = resolve_under(AROOT, r["pose_path"])
            prompt = load_prompt(resolve_under(AROOT, r["prompt_path"]))
            p_fps = str(int(round(float(r.get("video_fps", 30)))))

            src_h, src_w = [str(int(x)) for x in r.get("pose_source_resolution", [720, 1280])]
        else:
            name = r.get("name") or Path(r.get("image_path", f"case_{i:04d}")).parent.name
            prompt = (Path(r["prompt_path"]).read_text().strip() if r.get("prompt_path")
                      else r.get("prompt", ""))
            p_fps = str(int(r.get("pose_fps", 24)))
            src_h, src_w = [str(int(x)) for x in r.get("pose_source_resolution", [480, 832])]
            if MODE == "i2v":
                image = r["image_path"]

                pose = r["pose_path"] if GEO_ON else None
                if GEO_ON and not pose:
                    sys.exit(f"[ERROR] {name}: i2v with warp on needs \"pose_path\" in the jsonl row. "
                             f"Run with WARP=off to generate from image + prompt only.")


        _sched = r.get("segment_prompts")
        if _sched is None and r.get("segment_prompts_path"):
            _sched = json.loads(Path(r["segment_prompts_path"]).read_text())


        _seed = str(int(r["seed"])) if r.get("seed") is not None else SEED


        _events = ",".join(str(int(x)) for x in (r.get("event_chunks") or []))
        if _events and MODE == "t2v":
            print(f"[warn] {name}: event_chunks ignored, t2v has no warp to drop", flush=True)
            _events = ""
        out_dir = OUT_ROOT / name
        _required = [(video, "video"), (image, "image"), (pose, "pose")]
        _missing = [(pth, tag) for pth, tag in _required if pth and not os.path.isfile(pth)]
        if _missing:
            print(f"[SKIP] {name}: missing {_missing[0][1]} {_missing[0][0]}", flush=True)
            skip += 1; skip_missing += 1; continue
        if (out_dir / DONE_MARK).is_file():
            print(f"[SKIP] {name}: exists", flush=True); skip += 1; continue
        out_dir.mkdir(parents=True, exist_ok=True)

        argv = [


            sys.executable, "scripts/inference/infer_single.py",
            "--ckpt_path", BASE_CKPT, "--transformer_path", TRANSFORMER_PATH,
            *(["--is_enable_stage2", "--stage2_num_stages", STAGE2_NUM_STAGES,
               "--stage2_steps", *STAGE2_STEPS,
               "--stage2_stage_range", *STAGE2_STAGE_RANGE,
               "--stage2_warp_compression_mode", WARP_MODE] if IS_STAGE2 == "1" else []),
            *(["--use_dmd"] if USE_DMD == "1" else []),
            *(["--is_amplify_first_chunk"] if AMPLIFY_FIRST_CHUNK == "1" else []),
            "--height", HEIGHT, "--width", WIDTH, "--num_frames", NUM_FRAMES, "--fps", FPS,
            "--num_inference_steps", NUM_INFERENCE_STEPS, "--guidance_scale", GUIDANCE_SCALE, "--seed", _seed,
            "--vae_decode_type", VAE_DECODE_TYPE,
            "--start_seconds", START_SECONDS,

            "--ref_seconds", REF_VIDEO_SEC if MODE == "v2v" else "0.0",
            "--no_raw_sink_frames",


            *(["--use_geometric_state",
               "--lingbot_pose_path", pose, "--lingbot_pose_source_fps", p_fps,
               "--lingbot_pose_source_resolution", src_h, src_w, "--lingbot_pose_type", POSE_TYPE,
               "--visibility_aware_noise"] if GEO_ON else []),
            *(["--ref_video_for_viz", video] if MODE == "v2v" else []),
            "--warp_noise_sigma_invisible", "1.0", "--warp_noise_sigma_min", "0.0", "--warp_noise_sigma_max", WARP_SIGMA_MAX,
            *(["--visible_token_threshold", "0.5",
               "--prefix_idx_mode", "zero", "--warp_rope_mode", "overlap_noise",
               "--warp_lag_chunks", WARP_LAG,
               "--geo_recon_backend", "da3", "--geo_cloud_update_n", "12",
               *(["--geo_da3_src", DA3_SRC] if DA3_SRC else []),
               "--geo_da3_weights", DA3_WEIGHTS, "--geo_da3_process_res", DA3_PROCESS_RES,
               "--geo_depth_backend", DEPTH_BACKEND,
               *(["--geo_vigeo_weights", VIGEO_WEIGHTS,
                  "--geo_vigeo_mode", VIGEO_MODE, "--geo_vigeo_scale_mode", VIGEO_SCALE_MODE,
                  "--geo_vigeo_scale_value", VIGEO_SCALE_VALUE,
                  "--geo_vigeo_depth_median_target", VIGEO_DEPTH_MEDIAN_TARGET,
                  "--geo_vigeo_anchor_windows", VIGEO_ANCHOR_WINDOWS,
                  "--geo_vigeo_cache_keep_frames", VIGEO_CACHE_KEEP_FRAMES,
                  "--geo_vigeo_intr_source", VIGEO_INTR_SOURCE,
                  *(["--geo_vigeo_src", VIGEO_SRC] if VIGEO_SRC else [])]
                 if DEPTH_BACKEND == "vigeo" else []),
               "--geo_da3_render_mode", RENDER_MODE, "--geo_bw_fill_iters", BW_FILL_ITERS,
               *(["--geo_warp_stage0_only"] if STAGE0_ONLY == "1" else []),
               *(["--no_geo_chunk0_ref_warp"] if CHUNK0_REF_WARP == "0" else []),
               *(["--geo_chunk0_target_disparity_px", CHUNK0_TARGET_DISP_PX]
                 if CHUNK0_TARGET_DISP_PX else []),
               *_plk, *_nc] if GEO_ON else []),


            *(["--save_chunk_segments"]
              if VAE_DECODE_TYPE == "persistent" and SAVE_SEGMENTS == "1" else []),
            *(["--stream_long_video"] if STREAM_LONG == "1" else []),
            *(["--geo_hist_max_frames", GEO_HIST_MAX] if int(GEO_HIST_MAX) > 0 else []),
            *(["--dump_geo_intermediates"] if DUMP_GEO == "1" else []),
            *(["--bg_postprocess", "--bg_postprocess_max", BG_POSTPROC_MAX]
              if BG_POSTPROC == "1" else []),
            "--joystick_hud", JOYSTICK_HUD,
            *(["--prompt_schedule", json.dumps(_sched, ensure_ascii=False)] if _sched else []),
            *(["--event_chunks", _events] if _events else []),
            "--sample_type", MODE, "--prompt", prompt, "--negative_prompt", NEG,
            "--image_noise_sigma_min", "0.0", "--image_noise_sigma_max", "0.0",
            *(["--video_path", video,
               "--video_noise_sigma_min", "0.0", "--video_noise_sigma_max", "0.0"] if MODE == "v2v" else []),
            *(["--image_path", image] if MODE == "i2v" else []),
            "--output_folder", str(out_dir),
        ]
        if GEO_ON and RENDER_MODE == "backward_zbuf" and ZBUF_DESPECKLE == "1":
            argv += ["--geo_zbuf_despeckle", "--geo_zbuf_despeckle_ksize", ZBUF_KSIZE, "--geo_zbuf_despeckle_fill_iters", ZBUF_FILL]
        argv += _rst
        if LIVE_CONTROL_PATH:
            argv += ["--live_control_path", LIVE_CONTROL_PATH]
        if MODE != "t2v":
            if float(CAP_MAX_DEG) > 0: argv += ["--max_deg_per_chunk", CAP_MAX_DEG]
            if float(CAP_MAX_TRANS) > 0: argv += ["--max_trans_per_chunk", CAP_MAX_TRANS]
            if CAP_MAX_DEG != "0" or CAP_MAX_TRANS != "0": argv += ["--cap_mode", CAP_MODE]
            if float(POSE_SMOOTH_WIN) > 1: argv += ["--pose_smooth_win", POSE_SMOOTH_WIN]
            argv += ["--pose_extend_mode", POSE_EXTEND_MODE]

        _log_dir = OUT_ROOT / "_logs"; _log_dir.mkdir(parents=True, exist_ok=True)

        if IN_PROCESS_BATCH == "1":


            pending.append({"argv": argv[2:], "name": name,
                            "log": str(_log_dir / f"{name}.log")})
            pending_meta.append((name, out_dir, _log_dir / f"{name}.log"))
            continue

        print(f"\n==== [shard {SHARD}] {n+1}/{len(mine)} idx{i} {name} "
              f"[{TRANSFORMER_PATH}] ====", flush=True)
        env = dict(os.environ); env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"


        env["EVOKE_INFER_PROGRESS"] = "1" if VERBOSE else "0"


        env["DA3_LOG_LEVEL"] = os.environ.get(
            "DA3_LOG_LEVEL", "INFO" if _env("EVOKE_INFER_DEBUG", "0") == "1" else "WARN")
        log_dir = OUT_ROOT / "_logs"; log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"
        if VERBOSE:


            proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=0)
            with open(log_path, "wb") as lg:
                tail = b""
                while True:
                    buf = os.read(proc.stdout.fileno(), 65536)
                    if not buf:
                        break

                    sys.stdout.buffer.write(buf); sys.stdout.buffer.flush()


                    tail += buf
                    if b"\n" in tail:
                        *lines, tail = tail.split(b"\n")
                        lg.write(b"".join(l.rsplit(b"\r", 1)[-1] + b"\n" for l in lines)); lg.flush()
                if tail.rsplit(b"\r", 1)[-1]:
                    lg.write(tail.rsplit(b"\r", 1)[-1] + b"\n")
            proc.stdout.close()
            rc = proc.wait()
        else:
            with open(log_path, "w") as lg:
                rc = subprocess.run(argv, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode


        if rc == 0 and (out_dir / DONE_MARK).is_file():
            _extra = "" if (out_dir / VIZ_MARK).is_file() or MODE != "v2v" else f" (no {VIZ_MARK})"
            print(f"[DONE] {name}{_extra}", flush=True); done += 1
        else:
            print(f"[FAIL] {name} rc={rc} - see {log_path}", flush=True); fail += 1


            try:
                if not any(out_dir.iterdir()):
                    out_dir.rmdir()
            except OSError:
                pass


    if IN_PROCESS_BATCH == "1" and pending:
        batch_file = OUT_ROOT / f"_argv_batch_shard{SHARD}.jsonl"
        batch_file.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in pending))
        env = dict(os.environ); env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
        env["EVOKE_INFER_PROGRESS"] = "1" if VERBOSE else "0"
        env["DA3_LOG_LEVEL"] = os.environ.get(
            "DA3_LOG_LEVEL", "INFO" if _env("EVOKE_INFER_DEBUG", "0") == "1" else "WARN")


        _share = min(64, max(1, (os.cpu_count() or 8) // max(1, NSHARD)))
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "CV_NUM_THREADS"):
            env.setdefault(_v, str(_share))
        env.setdefault("EVOKE_CPU_THREADS", str(_share))
        shard_log = OUT_ROOT / "_logs" / f"_shard{SHARD}_batch.log"
        if ARGV_SERVER_DIR:
            server_dir = Path(ARGV_SERVER_DIR).resolve()
            requests_dir = server_dir / "requests"
            responses_dir = server_dir / "responses"
            requests_dir.mkdir(parents=True, exist_ok=True)
            responses_dir.mkdir(parents=True, exist_ok=True)
            state_path = server_dir / "state.json"

            def _atomic_json(path: Path, payload: dict) -> None:
                temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
                temporary.replace(path)

            if SERVER_PRELOAD:
                template_path = server_dir / "template.json"
                _atomic_json(template_path, pending[0])
                print(f"\n==== [shard {SHARD}] PRELOADING persistent pipeline "
                      f"[{TRANSFORMER_PATH}] ====\n     queue: {server_dir}", flush=True)


                command = [sys.executable]
                sp_size = int(env.get("EVOKE_SP_SIZE", "1"))
                if sp_size > 1:
                    command += ["-m", "torch.distributed.run", "--standalone",
                                "--nproc_per_node", str(sp_size), "--max_restarts", "0"]
                command += ["scripts/inference/infer_single.py", "--argv_server",
                            str(template_path), str(server_dir)]
                rc_batch = subprocess.run(command, env=env).returncode
            else:
                if len(pending) != 1:
                    sys.exit("[ERROR] EVOKE argv server submissions must contain exactly one case")
                request_id = SERVER_REQUEST_ID or f"shard{SHARD}-{os.getpid()}-{int(time.time() * 1000)}"
                request_path = requests_dir / f"{request_id}.json"
                response_path = responses_dir / f"{request_id}.json"
                response_path.unlink(missing_ok=True)
                _atomic_json(request_path, pending[0])
                print(f"\n==== [shard {SHARD}] submitted {pending[0]['name']} to persistent worker "
                      f"request={request_id} ====", flush=True)
                last_notice = 0.0
                while not response_path.is_file():
                    now = time.time()
                    if now - last_notice >= 10:
                        phase = "starting"
                        worker_pid = None
                        try:
                            state = json.loads(state_path.read_text(encoding="utf-8"))
                            phase = state.get("phase", phase)
                            worker_pid = state.get("pid")
                        except (OSError, json.JSONDecodeError):
                            pass
                        if worker_pid:
                            try:
                                os.kill(int(worker_pid), 0)
                            except (ProcessLookupError, ValueError):
                                sys.exit(f"[ERROR] persistent worker exited while request {request_id} was pending")
                            except PermissionError:
                                pass
                        print(f"[worker] request={request_id} phase={phase}", flush=True)
                        last_notice = now
                    time.sleep(0.5)
                response = json.loads(response_path.read_text(encoding="utf-8"))
                rc_batch = int(response.get("returnCode", 1))
                print(f"[worker] request={request_id} finished rc={rc_batch} "
                      f"elapsed={float(response.get('elapsed', 0)):.1f}s", flush=True)
        else:
            print(f"\n==== [shard {SHARD}] {len(pending)} case(s) in ONE process "
                  f"[{TRANSFORMER_PATH}] ====\n     argv batch: {batch_file}", flush=True)


            with open(shard_log, "w") as lg:
                rc_batch = subprocess.run(
                    [sys.executable, "scripts/inference/infer_single.py", "--argv_jsonl", str(batch_file)],
                    env=env, stdout=lg, stderr=subprocess.STDOUT).returncode


        if BG_POSTPROC == "1":
            _drain_bg_postprocess(OUT_ROOT)


        for name, out_dir, log_path in pending_meta:
            if (out_dir / DONE_MARK).is_file() or (LIVE_CONTROL_PATH and rc_batch == 0):
                _extra = "" if (out_dir / VIZ_MARK).is_file() or MODE != "v2v" else f" (no {VIZ_MARK})"
                print(f"[DONE] {name}{_extra}", flush=True); done += 1
            else:
                print(f"[FAIL] {name} - see {log_path} (batch rc={rc_batch}, "
                      f"shard log {shard_log})", flush=True); fail += 1
                try:
                    if not any(out_dir.iterdir()):
                        out_dir.rmdir()
                except OSError:
                    pass

    print(f"\n[shard {SHARD}/{NSHARD}] FINISHED done={done} skip={skip} fail={fail}", flush=True)


    if mine and skip_missing == len(mine):
        sys.exit(f"[ERROR] all {len(mine)} cases skipped: no input file resolved.\n"
                 f"        jsonl paths are tried under the dataset roots first, then repo-relative.\n"
                 f"        VROOT={VROOT!r} AROOT={AROOT!r} JSONL={JSONL!r}\n"
                 f"        Set VROOT / AROOT for a mounted dataset, or use a jsonl with repo-relative "
                 f"paths (examples/*/cases.jsonl).")


    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
