#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-results/sft/qwen25_7b_finground_sft}"
export DPO_TRAIN_FILE="${DPO_TRAIN_FILE:-data/dpo/dpo_targeted_v4.jsonl}"
export DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-results/dpo/qwen25_7b_finground_dpo_v4_targeted_s100_candidate}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-25}"
export EVAL_STEPS="${EVAL_STEPS:-25}"
export BETA="${BETA:-0.05}"
export LR="${LR:-1e-5}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export REPORT_TO="${REPORT_TO:-none}"
export RUN_NAME="${RUN_NAME:-finground-dpo-v4-targeted-sweep}"
# run_dpo_a10040.sh only knows the original v1 audit file, so v4 performs its
# own gate below and then bypasses the legacy gate.
export ALLOW_UNREVIEWED_DPO=1

mkdir -p logs results reports
main_log="logs/dpo_v4_targeted_sweep_pipeline.log"
train_log="logs/dpo_v4_targeted_sweep_train.log"

{
  echo "[$(date '+%F %T')] DPO v4 targeted sweep started"
  echo "EXPERIMENTAL_CANDIDATE_RUN=1"
  echo "FORMAL_HUMAN_V4_AUDIT_PASS=0"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SFT_ADAPTER_PATH=$SFT_ADAPTER_PATH"
  echo "DPO_TRAIN_FILE=$DPO_TRAIN_FILE"
  echo "DPO_OUTPUT_DIR=$DPO_OUTPUT_DIR"
  echo "MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS BETA=$BETA LR=$LR"
} >> "$main_log"

python scripts/build_dpo_targeted_v4.py >> "$main_log" 2>&1

if [[ "${ALLOW_UNREVIEWED_V4_DPO:-0}" != "1" ]]; then
  python - <<'PY'
import json
from pathlib import Path

path = Path("reports/dpo_targeted_v4_audit_report.json")
if not path.exists():
    raise SystemExit("Missing reports/dpo_targeted_v4_audit_report.json. Review v4 audit sample before training, or set ALLOW_UNREVIEWED_V4_DPO=1 for smoke only.")
report = json.loads(path.read_text(encoding="utf-8"))
if not report.get("gate_pass"):
    raise SystemExit("V4 audit gate has not passed. Do not run formal v4 candidate training. Set ALLOW_UNREVIEWED_V4_DPO=1 only for smoke/debug.")
PY
fi

if { time bash scripts/run_dpo_a10040.sh; } > "$train_log" 2>&1; then
  train_rc=0
else
  train_rc=$?
fi
echo "[$(date '+%F %T')] DPO v4 targeted train finished rc=$train_rc" >> "$main_log"
tail -n 80 "$train_log" >> "$main_log" 2>/dev/null || true
if [ "$train_rc" -ne 0 ]; then
  echo "[$(date '+%F %T')] DPO_V4_TARGETED_SWEEP_DONE rc=$train_rc" >> "$main_log"
  exit "$train_rc"
fi

export EVAL_FILE="${EVAL_FILE:-data/eval/eval.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export LIMIT="${LIMIT:--1}"

overall_rc=0
for step in ${V4_CKPT_STEPS:-50 75 100}; do
  adapter="$DPO_OUTPUT_DIR/checkpoint-${step}"
  if [ "$step" = "$MAX_STEPS" ] && [ ! -f "$adapter/adapter_model.safetensors" ]; then
    adapter="$DPO_OUTPUT_DIR"
  fi
  if [ ! -f "$adapter/adapter_model.safetensors" ]; then
    echo "[$(date '+%F %T')] missing $adapter/adapter_model.safetensors; skip v4 checkpoint-$step" >> "$main_log"
    continue
  fi
  export DPO_ADAPTER_PATH="$adapter"
  export PREDICTIONS="results/dpo_v4_targeted_s${step}_predictions.jsonl"
  export METRICS="results/dpo_v4_targeted_s${step}_metrics.json"
  export SCORED="results/dpo_v4_targeted_s${step}_scored.jsonl"
  eval_log="logs/dpo_v4_targeted_s${step}_eval.log"
  echo "[$(date '+%F %T')] start DPO v4 checkpoint-$step eval" >> "$main_log"
  if { time bash scripts/run_dpo_eval_a10040.sh; } > "$eval_log" 2>&1; then
    rc=0
  else
    rc=$?
    overall_rc=$rc
  fi
  echo "[$(date '+%F %T')] finish DPO v4 checkpoint-$step eval rc=$rc" >> "$main_log"
  tail -n 80 "$eval_log" >> "$main_log" 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    break
  fi
done

echo "[$(date '+%F %T')] DPO_V4_TARGETED_SWEEP_DONE rc=$overall_rc" >> "$main_log"
exit "$overall_rc"
