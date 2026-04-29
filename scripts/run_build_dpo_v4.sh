#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_dpo_targeted_v4.py "$@"
