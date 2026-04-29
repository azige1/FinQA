#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline error-delta \
  --names base sft dpo \
  --metrics results/base_metrics.json results/sft_metrics.json results/dpo_metrics.json \
  --output reports/error_delta_report.json
