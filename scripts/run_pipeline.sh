#!/usr/bin/env bash
# Bash wrapper around `python -m pipeline.run` that activates conda and tees
# the orchestrator output to a logfile. Designed for tmux:
#
#   tmux new -s magmar
#   bash scripts/run_pipeline.sh
#
# Override anything via env vars:
#   INPUT=my_events.json VIDEOS=/data/videos OUT=out/ GPUS=0,1,2,3 \
#     bash scripts/run_pipeline.sh
#
# Run only some parts (composable subset via PART env var):
#   PART=1    bash scripts/run_pipeline.sh   # preprocessing only
#   PART=2    bash scripts/run_pipeline.sh   # claim generation only
#   PART=3    bash scripts/run_pipeline.sh   # aggregation only
#   PART=12   bash scripts/run_pipeline.sh   # preprocessing + claim-gen
#   PART=23   bash scripts/run_pipeline.sh   # claim-gen + aggregation
#   PART=all  bash scripts/run_pipeline.sh   # default — all three parts
set -u

INPUT=${INPUT:-examples/events.example.json}
VIDEOS=${VIDEOS:?set VIDEOS=/path/to/videos containing <video_id>.mp4}
OUT=${OUT:-out}
GPUS=${GPUS:-0,1}
ASR_GPUS=${ASR_GPUS:-$GPUS}
CLAIM_GPUS=${CLAIM_GPUS:-$GPUS}
FPS=${FPS:-1.0}
WORKERS=${WORKERS:-8}
WHISPER_MODEL=${WHISPER_MODEL:-large-v3}
YOLO_MODEL=${YOLO_MODEL:-yolo12x.pt}
OCR_MODEL=${OCR_MODEL:-tencent/HunyuanOCR}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
LVLM_MODEL=${LVLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}
FRAME_MODE=${FRAME_MODE:-guided}

AGG_MODEL=${AGG_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
AGG_EMBED=${AGG_EMBED:-Qwen/Qwen3-Embedding-8B}
AGG_TP=${AGG_TP:-1}
AGG_TAU=${AGG_TAU:-0.9}
AGG_TEAM=${AGG_TEAM:-123456}
AGG_TASK=${AGG_TASK:-oracle}
AGG_PREFIX=${AGG_PREFIX:-penkil}
AGG_METHOD_B=${AGG_METHOD_B:-0}      # 1 = also run method B ablation

PART=${PART:-all}                     # all | 1 | 2 | 3 | 12 | 23

CONDA_ENV=${CONDA_ENV:-qwen_3}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$OUT/logs"

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
PY=$(command -v python)
echo "[info] python      : $PY"
echo "[info] input       : $INPUT"
echo "[info] videos      : $VIDEOS"
echo "[info] out         : $OUT"
echo "[info] gpus        : $GPUS  (asr: $ASR_GPUS  claim: $CLAIM_GPUS  agg_tp: $AGG_TP)"
echo "[info] fps         : $FPS"
echo "[info] frame_mode  : $FRAME_MODE"
echo "[info] part        : $PART"
echo

# Translate PART -> --skip / --only flags
case "$PART" in
    all) EXTRA=() ;;
    1)   EXTRA=(--skip claim_step1 claim_step2 aggregate) ;;
    2)   EXTRA=(--only claim_step1 claim_step2) ;;
    3)   EXTRA=(--only aggregate) ;;
    12)  EXTRA=(--skip aggregate) ;;
    23)  EXTRA=(--only claim_step1 claim_step2 aggregate) ;;
    *)   echo "[fatal] PART must be one of: all | 1 | 2 | 3 | 12 | 23"; exit 2 ;;
esac

# Optional --run-method-b
EXTRA_AGG=()
[[ "$AGG_METHOD_B" == "1" ]] && EXTRA_AGG+=(--aggregator-run-method-b)

"$PY" -m pipeline.run \
    --input         "$INPUT" \
    --videos-dir    "$VIDEOS" \
    --output-dir    "$OUT" \
    --gpus          "$GPUS" \
    --asr-gpus      "$ASR_GPUS" \
    --claim-gpus    "$CLAIM_GPUS" \
    --fps           "$FPS" \
    --workers       "$WORKERS" \
    --whisper-model "$WHISPER_MODEL" \
    --yolo-model    "$YOLO_MODEL" \
    --ocr-model     "$OCR_MODEL" \
    --llm-model     "$LLM_MODEL" \
    --lvlm-model    "$LVLM_MODEL" \
    --frame-mode    "$FRAME_MODE" \
    --aggregator-model            "$AGG_MODEL" \
    --aggregator-embed-model      "$AGG_EMBED" \
    --aggregator-tensor-parallel  "$AGG_TP" \
    --aggregator-cluster-tau      "$AGG_TAU" \
    --aggregator-team-id          "$AGG_TEAM" \
    --aggregator-task             "$AGG_TASK" \
    --aggregator-run-id-prefix    "$AGG_PREFIX" \
    "${EXTRA[@]}" \
    "${EXTRA_AGG[@]}" \
    2>&1 | tee -a "$OUT/logs/run.log"
