#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export EVAL_FILE="${EVAL_FILE:-data/eval/eval.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export LIMIT="${LIMIT:--1}"

steps="${CKPT_STEPS:-50 100 150 200 250 300 350 400 450}"
mkdir -p logs results/checkpoint_sweep reports
main_log="logs/dpo_checkpoint_sweep.log"

{
  echo "[$(date '+%F %T')] DPO checkpoint sweep started"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "EVAL_FILE=$EVAL_FILE"
  echo "LIMIT=$LIMIT"
  echo "steps=$steps"
} >> "$main_log"

overall_rc=0
for step in $steps; do
  adapter="results/dpo/qwen25_7b_finground_dpo/checkpoint-${step}"
  if [ ! -f "$adapter/adapter_model.safetensors" ]; then
    echo "[$(date '+%F %T')] missing $adapter/adapter_model.safetensors; skip checkpoint-$step" >> "$main_log"
    continue
  fi
  export DPO_ADAPTER_PATH="$adapter"
  export PREDICTIONS="results/checkpoint_sweep/dpo_ckpt_${step}_predictions.jsonl"
  export METRICS="results/checkpoint_sweep/dpo_ckpt_${step}_metrics.json"
  export SCORED="results/checkpoint_sweep/dpo_ckpt_${step}_scored.jsonl"
  eval_log="logs/dpo_ckpt_${step}_eval.log"
  echo "[$(date '+%F %T')] start checkpoint-$step eval" >> "$main_log"
  if { time bash scripts/run_dpo_eval_a10040.sh; } > "$eval_log" 2>&1; then
    rc=0
  else
    rc=$?
    overall_rc=$rc
  fi
  echo "[$(date '+%F %T')] finish checkpoint-$step eval rc=$rc" >> "$main_log"
  tail -n 30 "$eval_log" >> "$main_log" 2>/dev/null || true
  python scripts/summarize_dpo_checkpoint_sweep.py >> "$main_log" 2>&1 || true
  if [ "$rc" -ne 0 ]; then
    break
  fi
done

python scripts/summarize_dpo_checkpoint_sweep.py >> "$main_log" 2>&1 || true
echo "[$(date '+%F %T')] CHECKPOINT_SWEEP_DONE rc=$overall_rc" >> "$main_log"
exit "$overall_rc"
