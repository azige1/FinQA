#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.finground_qa.pipeline export-badcases \
  --names base sft dpo \
  --scored results/base_scored.jsonl results/sft_scored.jsonl results/dpo_scored.jsonl \
  --limit "${BADCASE_LIMIT:-50}" \
  --output results/badcase_50.jsonl
