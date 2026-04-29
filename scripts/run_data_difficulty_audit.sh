#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline data-difficulty-audit \
  --input "${INPUT:-data/unified/all_unified.jsonl}" \
  --limit "${LIMIT:-100}" \
  --seed "${SEED:-42}" \
  --output results/data_difficulty_audit_100.jsonl \
  --report reports/data_difficulty_report.json
