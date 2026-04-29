#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline build-answerability-eval \
  --input data/unified/all_unified.jsonl \
  --answerable-size "${ANSWERABLE_SIZE:-100}" \
  --unanswerable-size "${UNANSWERABLE_SIZE:-100}" \
  --seed "${SEED:-42}" \
  --output data/eval/answerability_eval.jsonl \
  --report reports/answerability_eval_report.json
