#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASELINE_ROOT="${BASELINE_ROOT:-/workspace/margent-aime-baseline-01}"
BASELINE_SESSION="${BASELINE_SESSION:-margent-aime}"
BASELINE_PYTHON="${BASELINE_PYTHON:-/workspace/margent-venv/bin/python}"
case "${1:-start}" in
  status)
    cat "$BASELINE_ROOT/baseline_status.json" 2>/dev/null || true
    cat "$BASELINE_ROOT/status.json" 2>/dev/null || true
    tail -n 20 "$BASELINE_ROOT.log" 2>/dev/null || true
    ;;
  start)
    command -v tmux >/dev/null
    test -x "$BASELINE_PYTHON"
    if tmux has-session -t "$BASELINE_SESSION" 2>/dev/null; then
      echo "Session $BASELINE_SESSION already exists. Use: tmux attach -t $BASELINE_SESSION"
      exit 1
    fi
    if [[ -f "$BASELINE_ROOT/baseline_report.json" ]]; then
      echo "Baseline already completed: $BASELINE_ROOT/baseline_report.json"
      exit 0
    fi
    printf -v BASELINE_COMMAND '%q -u %q --out %q >> %q 2>&1' "$BASELINE_PYTHON" "$PWD/scripts/runpod_aime_baseline.py" "$BASELINE_ROOT" "$BASELINE_ROOT.log"
    tmux new-session -d -s "$BASELINE_SESSION" -c "$PWD"
    tmux send-keys -t "$BASELINE_SESSION" -l "$BASELINE_COMMAND"
    tmux send-keys -t "$BASELINE_SESSION" Enter
    echo "Started in tmux: $BASELINE_SESSION"
    echo "Progress: bash scripts/runpod_aime_baseline.sh status"
    echo "Live log: tail -f $BASELINE_ROOT.log"
    ;;
  *) echo 'Usage: bash scripts/runpod_aime_baseline.sh [start|status]' >&2; exit 2 ;;
esac
