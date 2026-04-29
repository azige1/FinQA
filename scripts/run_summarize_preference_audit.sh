#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline summarize-audit \
  --audit "${AUDIT_FILE:-results/preference_pair_audit_100.jsonl}" \
  --min-reviewed "${MIN_REVIEWED:-100}" \
  --output "${AUDIT_REPORT:-reports/preference_pair_audit_report.json}"
