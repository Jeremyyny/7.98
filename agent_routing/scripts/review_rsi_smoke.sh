#!/usr/bin/env bash
# Upload an existing smoke's metrics and original text; no model or training.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_ENTITY="${WANDB_ENTITY:-yuningyangaillm}"
export WANDB_PROJECT="${WANDB_PROJECT:-MATH_rsi}"
export MARGENT_WANDB_MODE=online
export MARGENT_WANDB_TEXT=1
exec /workspace/margent-venv/bin/python -u scripts/review_rsi_smoke.py "${1:-/workspace/margent-rsi-smoke-actions-01}"
