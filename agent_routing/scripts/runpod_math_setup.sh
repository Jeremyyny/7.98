#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MARGENT_VENV="${MARGENT_VENV:-/workspace/margent-venv}"
MARGENT_RUN_ROOT="${MARGENT_RUN_ROOT:-/workspace/margent-runs}"
mkdir -p "$MARGENT_RUN_ROOT" /workspace/hf-cache
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
python3 -c 'import torch; assert torch.cuda.is_available(), "Choose a RunPod CUDA PyTorch image first"; print(torch.__version__, torch.cuda.get_device_name(0))'
if [[ ! -x "$MARGENT_VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$MARGENT_VENV"
fi
"$MARGENT_VENV/bin/python" -m pip install -r requirements-math.txt 'pytest>=8,<9'
"$MARGENT_VENV/bin/python" -m pip check
"$MARGENT_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable after installation"'
"$MARGENT_VENV/bin/python" -m pip freeze > "$MARGENT_RUN_ROOT/environment.lock.txt"
nvidia-smi > "$MARGENT_RUN_ROOT/nvidia-smi.txt"
MARGENT_WANDB_MODE=disabled "$MARGENT_VENV/bin/python" -m pytest -q tests/test_verifiable.py tests/test_math_reporting.py tests/test_math_wandb.py tests/test_marginal_value.py tests/test_benchmark_loaders.py tests/test_routing_anchor.py
printf 'Ready. Activate with: source %s/bin/activate\n' "$MARGENT_VENV"
