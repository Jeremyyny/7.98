#!/usr/bin/env bash
# Start the Qwen3-0.6B frozen advisor in a separate terminal before running this.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MARGENT_RUN_ROOT="${MARGENT_RUN_ROOT:-/workspace/margent-runs}"
if [[ ! -f "$MARGENT_RUN_ROOT/smoke-data/manifest.json" ]]; then
  python scripts/make_math_smoke_data.py --out "$MARGENT_RUN_ROOT/smoke-data"
fi
python -m src.verifiable doctor --config configs/math_smoke.json --out "$MARGENT_RUN_ROOT/smoke-environment.json"
python -m src.verifiable loop --config configs/math_smoke.json \
  --data-dir "$MARGENT_RUN_ROOT/smoke-data" --out "$MARGENT_RUN_ROOT/smoke-loop" \
  --arm dynamic_rl --rounds 2 --resume
