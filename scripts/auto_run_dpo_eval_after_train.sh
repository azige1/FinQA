#!/usr/bin/env bash
set -euo pipefail
cd /root/FinGround-QA
export PATH=/home/vipuser/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export DPO_ADAPTER_PATH=results/dpo/qwen25_7b_finground_dpo
export EVAL_FILE=data/eval/eval.jsonl
export PREDICTIONS=results/dpo_predictions.jsonl
export METRICS=results/dpo_metrics.json
export SCORED=results/dpo_scored.jsonl
export MAX_NEW_TOKENS=512
export LIMIT=-1
monitor_log=logs/dpo_eval_auto.log
{
  echo "[$(date '+%F %T')] auto eval watcher started"
  echo "waiting for DPO train process to finish..."
  while ps -eo cmd | grep -F 'python -m src.finground_qa.train_dpo' | grep -F 'results/dpo/qwen25_7b_finground_dpo' | grep -v grep >/dev/null; do
    tail -n 3 logs/dpo_train.log 2>/dev/null || true
    sleep 60
  done
  echo "[$(date '+%F %T')] DPO train process ended"
  for i in $(seq 1 20); do
    if grep -q 'DPO_DONE rc=' logs/dpo_train.log 2>/dev/null; then
      break
    fi
    echo "waiting for DPO_DONE marker... $i"
    sleep 15
  done
  tail -n 30 logs/dpo_train.log 2>/dev/null || true
  if ! grep -q 'DPO_DONE rc=0' logs/dpo_train.log 2>/dev/null; then
    echo "[$(date '+%F %T')] DPO did not finish cleanly; eval not started"
    exit 1
  fi
  if [ ! -f "$DPO_ADAPTER_PATH/adapter_model.safetensors" ]; then
    echo "[$(date '+%F %T')] Missing $DPO_ADAPTER_PATH/adapter_model.safetensors; eval not started"
    exit 1
  fi
  echo "[$(date '+%F %T')] starting DPO eval"
  { time bash scripts/run_dpo_eval_a10040.sh; } > logs/dpo_eval.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DPO eval finished rc=$rc"
  tail -n 80 logs/dpo_eval.log 2>/dev/null || true
  exit $rc
} >> "$monitor_log" 2>&1
