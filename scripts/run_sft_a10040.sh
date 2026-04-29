#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

python -m src.finground_qa.train_sft \
  --model "$MODEL_PATH" \
  --train-file "${SFT_TRAIN_FILE:-data/sft/sft_train.jsonl}" \
  --val-file "${SFT_VAL_FILE:-data/sft/sft_val.jsonl}" \
  --output-dir "${SFT_OUTPUT_DIR:-results/sft/qwen25_7b_finground_sft}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --epochs "${EPOCHS:-1}" \
  --learning-rate "${LR:-1e-4}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --save-steps "${SAVE_STEPS:-200}" \
  --eval-steps "${EVAL_STEPS:-200}" \
  --report-to "${REPORT_TO:-wandb}" \
  --run-name "${RUN_NAME:-finground-sft-qwen25-7b}"
