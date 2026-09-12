set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${EVOKE_PYTHON_BIN:-}" ] && export PATH="$EVOKE_PYTHON_BIN:$PATH"

if [ "${EVOKE_REMOTE_CLIENT:-0}" = "1" ] && [ "${IN_PROCESS_BATCH:-0}" = "1" ] \
    && [ -n "${EVOKE_ARGV_SERVER_DIR:-}" ] && [ "${EVOKE_SERVER_PRELOAD:-0}" != "1" ]; then
  echo "[preflight] remote queue client; CUDA is checked by the GPU worker"
elif ! python -c 'import torch, cv2' 2>/dev/null; then
  echo "[preflight] FATAL: the python on PATH ($(command -v python || echo none)) cannot import torch + cv2." >&2
  echo "[preflight]        Set EVOKE_PYTHON_BIN=<env>/bin to prepend the right interpreter." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
infer_post_distill.sh -- post-distilled model (models/evoke/stage3_post_distillation)

  MODE=v2v      reference video + pose track  -> camera-controlled continuation (needs shared dataset)
  MODE=i2v      first frame + pose track      -> camera-controlled generation (runs on bundled examples)
  MODE=t2v      prompt only                   -> no camera control (warp is off; the CLI forbids warp+t2v)
  MODE=segment  i2v + a prompt that switches mid-rollout (engine still runs i2v)

  NOTE: trained on v2v conditioning only -> i2v / t2v / segment are ZERO-SHOT here.

Examples
  MODE=i2v MAX_CASES=1 NUM_FRAMES=721  bash scripts/inference/infer_post_distill.sh
  MODE=segment NUM_CHUNKS=6            bash scripts/inference/infer_post_distill.sh
  MODE=v2v LOCAL_GPUS=4                bash scripts/inference/infer_post_distill.sh
  TRANSFORMER_PATH=<other-ckpt>        bash scripts/inference/infer_post_distill.sh

Full sweep (needs the shared dataset mounted)
  MODE=v2v JSONL=<your-sweep>.jsonl MAX_CASES=0 \
    VROOT=<video-root> AROOT=<annotation-root> bash <this script>

Knobs   LOCAL_GPUS (shard by case)  MAX_CASES (0=all)
        NUM_CHUNKS  exact chunk count (1 chunk = 36 frames = 1.5s @24fps) -- prefer this
        NUM_FRAMES  picks the chunk count only, it does NOT bound the output:
                    chunks = ceil(NUM_FRAMES/33), frames = 36*chunks - 3
                    721 -> 22 chunks -> 789 frames (32.9s);  2877 -> 88 -> 3165 (131.9s)
        v2v only: REF_VIDEO_SEC (default 5) = how much of the reference video conditions the
                  model, i.e. generation continues from that point. Forced to 0 for i2v/t2v.
        START_SECONDS (default 0) offsets where that window is taken from; it also shifts the
                  pose track, so it applies to i2v too (NOT forced to 0).
        HEIGHT WIDTH FPS  GUIDANCE_SCALE  SEED  OUT_ROOT  JSONL  TRANSFORMER_PATH
        EVOKE_PYTHON_BIN  prepended to PATH, for picking the interpreter
Logs    LOCAL_GPUS=1 streams the run to the terminal and to logs/ (chunk-level progress bar
        included). QUIET=1 redirects it to the log only; multi-shard runs always redirect.
Long    Rollouts above ~5min must stream: STREAM_LONG=1 decodes per chunk and stitches the final
        mp4 from segments/ (the driver refuses to accumulate more than 7200 frames on GPU).
        GEO_HIST_MAX_FRAMES bounds the DA3 point cloud for hour-scale runs -- note it changes the
        warp input, so it is a recipe change, not just a memory knob. See the README.
Output  $OUT_ROOT/<case>/geo_pred.mp4                      generated video (always)
        $OUT_ROOT/<case>/gt_vs_pred_cam_viz.mp4            4-panel gt|warp|vis|pred (v2v only)
        $OUT_ROOT/<case>/segments/segment_NNN_pred.mp4     per-chunk segments (persistent decode)
        $OUT_ROOT/_logs/<case>.log                         per-case engine log
EOF
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }

if [ -n "${NUM_INFERENCE_STEPS:-}" ] && [ "${NUM_INFERENCE_STEPS}" != "3" ]; then
  echo "[post_distill] WARNING: NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS is ignored under the pyramid;"
  echo "[post_distill]          the step count comes from STAGE2_STEPS. Override that instead."
fi

export MODE=${MODE:-v2v}
LOCAL_GPUS=${LOCAL_GPUS:-1}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}
NSHARD=${NSHARD:-$((LOCAL_GPUS * SHARDS_PER_GPU))}

TAG=$MODE
if [ "$MODE" = "segment" ]; then
  export JSONL=${JSONL:-"examples/segment_prompts/cases.jsonl"}
  export MODE=i2v
fi

if [ "$TAG" != "v2v" ]; then
  echo "[post_distill] WARNING: MODE=$TAG is zero-shot for this model (trained on v2v conditioning only)."
fi

case "$MODE" in
  v2v) export JSONL=${JSONL:-"examples/v2v/cases.jsonl"} ;;
  i2v) export JSONL=${JSONL:-"examples/i2v/cases.jsonl"} ;;
  t2v) export JSONL=${JSONL:-"examples/t2v/cases.jsonl"} ;;
esac

TP_ORIGIN="this is YOUR override (TRANSFORMER_PATH was set)"
[ -z "${TRANSFORMER_PATH:-}" ] && TP_ORIGIN="this is the LAUNCHER DEFAULT -- you did not set TRANSFORMER_PATH, so another ckpt you meant to test is NOT running"
export TRANSFORMER_PATH=${TRANSFORMER_PATH:-"models/evoke/stage3_post_distillation"}
export MAX_CASES=${MAX_CASES:-8}
export NUM_FRAMES=${NUM_FRAMES:-2877}
export HEIGHT=${HEIGHT:-384} WIDTH=${WIDTH:-640} FPS=${FPS:-24}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
export VAE_DECODE_TYPE=${VAE_DECODE_TYPE:-persistent}

export IS_STAGE2=1 STAGE2_NUM_STAGES=3 STAGE2_STEPS="1 1 1"
export NUM_INFERENCE_STEPS=3
export RESTRICT=0
export GEO_WARP_STAGE0_ONLY=1
export WARP_SIGMA_MAX=0.135
export NOISE_CENTER=0
export RENDER_MODE=backward_zbuf BW_FILL_ITERS=12
export DEPTH_BACKEND=${DEPTH_BACKEND:-vigeo}
export ZBUF_DESPECKLE=0
export WARP_MODE=fixed_mem

export OUT_ROOT=${OUT_ROOT:-"output_evoke/infer/post_distill_$TAG"}
case "$OUT_ROOT" in
  *//*) echo "[post_distill] FATAL: OUT_ROOT='$OUT_ROOT' has an empty path segment -- a" >&2
        echo "[post_distill]        variable in it expanded to nothing (unexported \$TAG etc.)." >&2
        echo "[post_distill]        Export it, or drop it from OUT_ROOT." >&2
        exit 1 ;;
  */)   echo "[post_distill] WARNING: OUT_ROOT='$OUT_ROOT' ends in a slash; if that is a"
        echo "[post_distill]          variable that expanded to nothing, results land one level up."
        export OUT_ROOT="${OUT_ROOT%/}" ;;
esac
EFF_FRAMES=$NUM_FRAMES; EFF_NOTE=""
if [ -n "${NUM_CHUNKS:-}" ] && [ "${NUM_CHUNKS}" != "0" ]; then
  EFF_FRAMES=$((33 * NUM_CHUNKS)); EFF_NOTE=" (NUM_CHUNKS=$NUM_CHUNKS -> $((36 * NUM_CHUNKS - 3)) px frames)"
fi

SLOG="logs/infer_post_distill_$TAG"; mkdir -p "$SLOG"
echo "=============================================================================="
echo "[post_distill] CHECKPOINT : $TRANSFORMER_PATH"
echo "[post_distill]              ^-- $TP_ORIGIN"
if [ "$GUIDANCE_SCALE" = "1" ] || [ "$GUIDANCE_SCALE" = "1.0" ]; then
    CFG_STATE=off
else
    CFG_STATE=on
fi
MODE_NOTE=""
if [ "$TAG" != "$MODE" ]; then
    MODE_NOTE=" (engine sample_type=$MODE)"
fi
echo "[post_distill] mode=$TAG$MODE_NOTE steps=3 (pyramid 1+1+1, guidance_scale=$GUIDANCE_SCALE, CFG $CFG_STATE)"
echo "[post_distill] jsonl=$JSONL max_cases=$MAX_CASES frames=$EFF_FRAMES$EFF_NOTE shards=$NSHARD"
echo "[post_distill] out=$OUT_ROOT"
echo "=============================================================================="

if [ "$LOCAL_GPUS" = "1" ] && [ "${QUIET:-0}" != "1" ]; then
  g=${GPU_OFFSET:-0}
  if [ "${EVOKE_SP_SIZE:-1}" -gt 1 ]; then
    g=$(seq -s, "${GPU_OFFSET:-0}" "$(( ${GPU_OFFSET:-0} + EVOKE_SP_SIZE - 1 ))")
  fi
  g=${EVOKE_VISIBLE_GPUS:-$g}
  if [ "${IN_PROCESS_BATCH:-0}" = "1" ] && [ -n "${EVOKE_ARGV_SERVER_DIR:-}" ] \
      && [ -e "$EVOKE_ARGV_SERVER_DIR/visible-gpus" ]; then
    if ! g=$(python -m ui.gpu_visibility --file "$EVOKE_ARGV_SERVER_DIR/visible-gpus" \
        --visible "${EVOKE_VISIBLE_GPUS:-}" --sp-size "${EVOKE_SP_SIZE:-1}"); then
      echo "[post_distill] FATAL: invalid resident-worker GPU visibility" >&2
      exit 1
    fi
  fi
  echo "[post_distill] shard 0 -> gpu $g -> terminal + $SLOG/shard_0.log"
  set +e
  CUDA_VISIBLE_DEVICES=$g SHARD=0 NSHARD=$NSHARD \
      python scripts/inference/infer_batch.py 2>&1 | tee "$SLOG/shard_0.log"
  fail=${PIPESTATUS[0]}
  set -e
else
  SHARD_STAGGER_S=${SHARD_STAGGER_S:-0}
  export EVOKE_MAX_CONCURRENT_LOADS=${EVOKE_MAX_CONCURRENT_LOADS:-$LOCAL_GPUS}
  pids=()
  for s in $(seq 0 $((NSHARD-1))); do
      g=$(( (s % LOCAL_GPUS) + ${GPU_OFFSET:-0} ))
      CUDA_VISIBLE_DEVICES=$g SHARD=$s NSHARD=$NSHARD \
          python scripts/inference/infer_batch.py > "$SLOG/shard_${s}.log" 2>&1 &
      pids+=($!); echo "[post_distill] shard $s -> gpu $g -> $SLOG/shard_${s}.log"
      [ "$s" -lt "$((NSHARD-1))" ] && [ "$SHARD_STAGGER_S" -gt 0 ] && sleep "$SHARD_STAGGER_S"
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
fi
echo "[post_distill] DONE failed_shards=$fail -> $OUT_ROOT"; exit $fail
