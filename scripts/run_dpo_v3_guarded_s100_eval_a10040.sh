#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-results/sft/qwen25_7b_finground_sft}"
export DPO_TRAIN_FILE="${DPO_TRAIN_FILE:-data/dpo/dpo_guarded_v3.jsonl}"
export DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-results/dpo/qwen25_7b_finground_dpo_v3_guarded_s100_candidate}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
export BETA="${BETA:-0.05}"
export LR="${LR:-1e-5}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export REPORT_TO="${REPORT_TO:-none}"
export RUN_NAME="${RUN_NAME:-finground-dpo-v3-guarded-s100-candidate}"
# v3 has Codex structural audit but no human sign-off yet. Run as candidate/experimental, not formal acceptance.
export ALLOW_UNREVIEWED_DPO=1

mkdir -p logs results reports
main_log="logs/dpo_v3_guarded_s100_pipeline.log"
train_log="logs/dpo_v3_guarded_s100_train.log"
eval_log="logs/dpo_v3_guarded_s100_eval.log"

{
  echo "[$(date '+%F %T')] DPO v3 guarded s100 candidate pipeline started"
  echo "EXPERIMENTAL_CANDIDATE_RUN=1"
  echo "FORMAL_HUMAN_V3_AUDIT_PASS=0"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SFT_ADAPTER_PATH=$SFT_ADAPTER_PATH"
  echo "DPO_TRAIN_FILE=$DPO_TRAIN_FILE"
  echo "DPO_OUTPUT_DIR=$DPO_OUTPUT_DIR"
  echo "MAX_STEPS=$MAX_STEPS BETA=$BETA LR=$LR"
} >> "$main_log"

python scripts/build_dpo_guarded_v3.py >> "$main_log" 2>&1

if { time bash scripts/run_dpo_a10040.sh; } > "$train_log" 2>&1; then
  train_rc=0
else
  train_rc=$?
fi
echo "[$(date '+%F %T')] DPO v3 guarded s100 train finished rc=$train_rc" >> "$main_log"
tail -n 80 "$train_log" >> "$main_log" 2>/dev/null || true
if [ "$train_rc" -ne 0 ]; then
  echo "[$(date '+%F %T')] DPO_V3_GUARDED_S100_DONE rc=$train_rc" >> "$main_log"
  exit "$train_rc"
fi

export DPO_ADAPTER_PATH="$DPO_OUTPUT_DIR"
export EVAL_FILE="${EVAL_FILE:-data/eval/eval.jsonl}"
export PREDICTIONS="${PREDICTIONS:-results/dpo_v3_guarded_s100_predictions.jsonl}"
export METRICS="${METRICS:-results/dpo_v3_guarded_s100_metrics.json}"
export SCORED="${SCORED:-results/dpo_v3_guarded_s100_scored.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export LIMIT="${LIMIT:--1}"

if { time bash scripts/run_dpo_eval_a10040.sh; } > "$eval_log" 2>&1; then
  eval_rc=0
else
  eval_rc=$?
fi
echo "[$(date '+%F %T')] DPO v3 guarded s100 eval finished rc=$eval_rc" >> "$main_log"
tail -n 80 "$eval_log" >> "$main_log" 2>/dev/null || true
echo "[$(date '+%F %T')] DPO_V3_GUARDED_S100_DONE rc=$eval_rc" >> "$main_log"
exit "$eval_rc"
