#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DPO_ADAPTER_PATH="${DPO_ADAPTER_PATH:-results/dpo/qwen25_7b_finground_dpo}"

python -m src.finground_qa.generate \
  --model "$MODEL_PATH" \
  --adapter "$DPO_ADAPTER_PATH" \
  --input "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --output "${PREDICTIONS:-results/dpo_predictions.jsonl}" \
  --limit "${LIMIT:--1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --load-in-4bit

python -m src.finground_qa.pipeline evaluate \
  --eval-file "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --predictions "${PREDICTIONS:-results/dpo_predictions.jsonl}" \
  --metrics "${METRICS:-results/dpo_metrics.json}" \
  --scored "${SCORED:-results/dpo_scored.jsonl}"
