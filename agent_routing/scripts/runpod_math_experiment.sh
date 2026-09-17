#!/usr/bin/env bash
# Run one predeclared arm/seed, then locked initial/final external tests.
# Start the frozen advisor separately; set CUDA_VISIBLE_DEVICES for the manager.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MARGENT_ARM="${1:-dynamic_rl}"
MARGENT_SEED="${2:-42}"
MARGENT_RUN_ROOT="${MARGENT_RUN_ROOT:-/workspace/margent-runs}"
MARGENT_DATA="${MARGENT_DATA:-/workspace/margent-data}"
case "$MARGENT_ARM" in dynamic_rl|dynamic_sft|static_rl|success_rl) ;; *) printf 'Unknown arm: %s\n' "$MARGENT_ARM" >&2; exit 2 ;; esac
MARGENT_CONFIG="$MARGENT_RUN_ROOT/configs/${MARGENT_ARM}_s${MARGENT_SEED}.json"
python - "$MARGENT_CONFIG" "$MARGENT_SEED" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path("configs/math_rsi.json").read_text())
value["seed"] = int(sys.argv[2])
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and json.loads(path.read_text()) != value:
    raise ValueError("Existing frozen configuration differs; choose a new run root")
path.write_text(json.dumps(value, indent=2))
PY
MARGENT_OUTPUT="$MARGENT_RUN_ROOT/${MARGENT_ARM}_s${MARGENT_SEED}"
python -m src.verifiable doctor --config "$MARGENT_CONFIG" --out "$MARGENT_RUN_ROOT/${MARGENT_ARM}_s${MARGENT_SEED}_environment.json"
python -m src.verifiable loop --config "$MARGENT_CONFIG" --data-dir "$MARGENT_DATA" \
  --out "$MARGENT_OUTPUT" --arm "$MARGENT_ARM" --rounds 2 --resume
python -m src.verifiable evaluate-suite --run-dir "$MARGENT_OUTPUT" --data-dir "$MARGENT_DATA"
