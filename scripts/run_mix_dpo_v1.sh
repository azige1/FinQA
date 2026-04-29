#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${ALLOW_RULE_ONLY_DPO:-0}" != "1" && ! -s data/dpo/model_mined_pairs.jsonl ]]; then
  echo "Missing data/dpo/model_mined_pairs.jsonl. Final DPO v1 must include model-mined rejected. Set ALLOW_RULE_ONLY_DPO=1 only for smoke/debug." >&2
  exit 1
fi

python -m src.finground_qa.pipeline mix-dpo \
  --rule-pairs data/dpo/rule_dpo_pairs.jsonl \
  --mined-pairs data/dpo/model_mined_pairs.jsonl \
  --target "${DPO_TARGET:-1000}" \
  --output data/dpo/dpo_balanced_v1.jsonl \
  --quality-report reports/preference_pair_quality_report.json \
  --difficulty-report reports/pair_difficulty_report.json

python -m src.finground_qa.pipeline audit-pairs \
  --pairs data/dpo/dpo_balanced_v1.jsonl \
  --audit-size 100 \
  --output results/preference_pair_audit_100.jsonl
