#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline prepare-data --output-dir .
python -m src.finground_qa.pipeline validate-sft \
  --file data/sft/sft_train.jsonl \
  --output reports/validate_sft_train.json
python -m src.finground_qa.pipeline build-rule-dpo \
  --unified-train data/unified/train_unified.jsonl \
  --target 600 \
  --output data/dpo/rule_dpo_pairs.jsonl \
  --report reports/preference_pair_quality_report.json \
  --difficulty-report reports/pair_difficulty_report.json
python -m src.finground_qa.pipeline audit-pairs \
  --pairs data/dpo/rule_dpo_pairs.jsonl \
  --output results/preference_pair_audit_100.jsonl

echo "Phase 0 local preparation complete."
