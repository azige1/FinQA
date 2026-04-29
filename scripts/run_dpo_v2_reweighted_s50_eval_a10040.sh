#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export DPO_ADAPTER_PATH="${DPO_ADAPTER_PATH:-results/dpo/qwen25_7b_finground_dpo_v2_reweighted_s100/checkpoint-50}"
export EVAL_FILE="${EVAL_FILE:-data/eval/eval.jsonl}"
export PREDICTIONS="${PREDICTIONS:-results/dpo_v2_reweighted_s50_predictions.jsonl}"
export METRICS="${METRICS:-results/dpo_v2_reweighted_s50_metrics.json}"
export SCORED="${SCORED:-results/dpo_v2_reweighted_s50_scored.jsonl}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export LIMIT="${LIMIT:--1}"

mkdir -p logs results
main_log="logs/dpo_v2_reweighted_s50_eval_pipeline.log"
eval_log="logs/dpo_v2_reweighted_s50_eval.log"

{
  echo "[$(date '+%F %T')] DPO v2 reweighted checkpoint-50 eval started"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "DPO_ADAPTER_PATH=$DPO_ADAPTER_PATH"
  echo "EVAL_FILE=$EVAL_FILE"
} >> "$main_log"

if { time bash scripts/run_dpo_eval_a10040.sh; } > "$eval_log" 2>&1; then
  rc=0
else
  rc=$?
fi
echo "[$(date '+%F %T')] DPO v2 reweighted checkpoint-50 eval finished rc=$rc" >> "$main_log"
tail -n 80 "$eval_log" >> "$main_log" 2>/dev/null || true
echo "[$(date '+%F %T')] DPO_V2_REWEIGHTED_S50_EVAL_DONE rc=$rc" >> "$main_log"
exit "$rc"
