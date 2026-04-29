#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-results/sft/qwen25_7b_finground_sft}"
MINING_SOURCE_ROWS="${MINING_SOURCE_ROWS:-data/unified/train_unified.jsonl}"
MINING_EVAL_FILE="${MINING_EVAL_FILE:-data/eval/train_mining_eval.jsonl}"
MINING_LIMIT="${MINING_LIMIT:-500}"

python -m src.finground_qa.pipeline build-mining-eval \
  --input "$MINING_SOURCE_ROWS" \
  --output "$MINING_EVAL_FILE" \
  --limit "$MINING_LIMIT" \
  --report reports/train_mining_eval_report.json

python -m src.finground_qa.generate_vllm \
  --model "$MODEL_PATH" \
  --input "$MINING_EVAL_FILE" \
  --output results/sft_mining_predictions.jsonl \
  --limit "$MINING_LIMIT" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}"

python -m src.finground_qa.pipeline mine-rejected \
  --source-rows "$MINING_SOURCE_ROWS" \
  --predictions results/sft_mining_predictions.jsonl \
  --max-pairs "${MAX_MINED_PAIRS:-500}" \
  --output data/dpo/model_mined_pairs.jsonl \
  --report reports/model_mined_rejected_report.json
