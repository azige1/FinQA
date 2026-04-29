#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-results/sft/qwen25_7b_finground_sft}"

python -m src.finground_qa.generate \
  --model "$MODEL_PATH" \
  --adapter "$SFT_ADAPTER_PATH" \
  --input "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --output "${PREDICTIONS:-results/sft_predictions.jsonl}" \
  --limit "${LIMIT:--1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --load-in-4bit

python -m src.finground_qa.pipeline evaluate \
  --eval-file "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --predictions "${PREDICTIONS:-results/sft_predictions.jsonl}" \
  --metrics "${METRICS:-results/sft_metrics.json}" \
  --scored "${SCORED:-results/sft_scored.jsonl}"
