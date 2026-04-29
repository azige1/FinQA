#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-results/sft/qwen25_7b_finground_sft}"

ADAPTER_ARGS=()
if [[ -d "$SFT_ADAPTER_PATH" ]]; then
  ADAPTER_ARGS+=(--sft-adapter "$SFT_ADAPTER_PATH")
fi

GC_ARGS=(--gradient-checkpointing)
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "0" ]]; then
  GC_ARGS=(--no-gradient-checkpointing)
fi

if [[ "${ALLOW_UNREVIEWED_DPO:-0}" != "1" ]]; then
  python - <<'PY'
import json
from pathlib import Path

path = Path("reports/preference_pair_audit_report.json")
if not path.exists():
    raise SystemExit("Missing reports/preference_pair_audit_report.json. Run preference audit before DPO, or set ALLOW_UNREVIEWED_DPO=1 for smoke only.")
report = json.loads(path.read_text(encoding="utf-8"))
if not report.get("gate_pass"):
    raise SystemExit("Preference audit gate has not passed. Do not run formal DPO. Set ALLOW_UNREVIEWED_DPO=1 only for smoke/debug.")
PY
fi

python -m src.finground_qa.train_dpo \
  --model "$MODEL_PATH" \
  "${ADAPTER_ARGS[@]}" \
  --train-file "${DPO_TRAIN_FILE:-data/dpo/dpo_balanced_v1.jsonl}" \
  --output-dir "${DPO_OUTPUT_DIR:-results/dpo/qwen25_7b_finground_dpo}" \
  --max-source-length "${MAX_SOURCE_LENGTH:-1024}" \
  --max-target-length "${MAX_TARGET_LENGTH:-512}" \
  --max-steps "${MAX_STEPS:-500}" \
  --learning-rate "${LR:-1e-5}" \
  --beta "${BETA:-0.1}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "${GC_ARGS[@]}" \
  --save-steps "${SAVE_STEPS:-100}" \
  --eval-steps "${EVAL_STEPS:-100}" \
  --report-to "${REPORT_TO:-wandb}" \
  --run-name "${RUN_NAME:-finground-dpo-qwen25-7b}"
