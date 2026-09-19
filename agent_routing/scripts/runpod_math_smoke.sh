#!/usr/bin/env bash
# Start the matching frozen advisor first; MARGENT_SMOKE_CONFIG can select 9B.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MARGENT_RUN_ROOT="${MARGENT_RUN_ROOT:-/workspace/margent-runs}"
if [[ ! -f "$MARGENT_RUN_ROOT/smoke-data/manifest.json" ]]; then
  python scripts/make_math_smoke_data.py --out "$MARGENT_RUN_ROOT/smoke-data"
fi
MARGENT_SMOKE_CONFIG="${MARGENT_SMOKE_CONFIG:-$MARGENT_RUN_ROOT/frozen_smoke.json}"
if [[ ! -f "$MARGENT_SMOKE_CONFIG" ]]; then
  python -m src.verifiable freeze-config --config configs/math_smoke.json --out "$MARGENT_SMOKE_CONFIG"
fi
python -m src.verifiable doctor --config "$MARGENT_SMOKE_CONFIG" --out "$MARGENT_RUN_ROOT/smoke-environment.json"
python -m src.verifiable loop --config "$MARGENT_SMOKE_CONFIG" \
  --data-dir "$MARGENT_RUN_ROOT/smoke-data" --out "$MARGENT_RUN_ROOT/smoke-loop" \
  --arm dynamic_sft --rounds 2 --resume
