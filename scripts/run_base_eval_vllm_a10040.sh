#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

python -m src.finground_qa.generate_vllm \
  --model "$MODEL_PATH" \
  --input "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --output "${PREDICTIONS:-results/base_predictions.jsonl}" \
  --limit "${LIMIT:--1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-0.95}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}"

python -m src.finground_qa.pipeline evaluate \
  --eval-file "${EVAL_FILE:-data/eval/eval.jsonl}" \
  --predictions "${PREDICTIONS:-results/base_predictions.jsonl}" \
  --metrics "${METRICS:-results/base_metrics.json}" \
  --scored "${SCORED:-results/base_scored.jsonl}"
