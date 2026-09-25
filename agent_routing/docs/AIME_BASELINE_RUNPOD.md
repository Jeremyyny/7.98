# Full AIME2026 baseline on the existing RunPod

From `agent_routing`, update main and run:

```bash
git pull --ff-only origin main
bash scripts/runpod_aime_baseline.sh
```

This starts a retained tmux shell named `margent-aime`. Closing the browser
terminal does not end the experiment while the Pod remains running. It uses
Manager GPU 1 and starts its own frozen advisor on GPU 0, port 8003. Existing
services are neither reused nor killed. Busy Manager GPU / insufficient VRAM
or an occupied port fails before model startup. No packages are reinstalled.

The fixed output directory is `/workspace/margent-aime-baseline-01` and the
parent log is `/workspace/margent-aime-baseline-01.log`. Repeated launch refuses
an existing tmux session; it does not silently create a duplicate GPU job.

The stages are:

1. Start the parent W&B `aime_baseline` run, validate the frozen dataset and
   available GPUs, then start the owned advisor.
2. Assess three deterministic dev questions. Interrupt the child after it has
   saved a question, then restart it with resume enabled. Verify that the saved
   question files are unchanged and all three questions finish exactly once.
   This tests evaluation recovery, not interrupted training optimizer recovery.
3. Record dev generation timings and an explicitly rough 30-question estimate.
   If two or more dev questions have invalid or truncated Manager output, stop
   before the benchmark. One failed sample and wrong mathematical answers do
   not fail this preflight. If the estimate plus 25% exceeds the remaining
   budget, stop before starting AIME. Dev timing does not predict all AIME costs.
4. Evaluate **all 30 AIME2026 questions**, both Manager independent answers and
   deployed routing, with no test-based selection or training. The baseline
   uses the untrained base model, not the smoke's trained adapter.
5. Produce `baseline_report.json` only after verifying the exact full question
   set. The report includes integer correct counts and frozen configuration.

The configuration is `math_rsi_actions.json`: Manager answer budget 2048 tokens,
finite action decisions, greedy evaluation, frozen sampled advisor settings.
This is a complete dataset evaluation under those budgets, not an unrestricted
model leaderboard result. It makes no RSI improvement claim. Reuse the frozen
settings and test set for the later prespecified final checkpoint; do not tune
settings or choose checkpoints on AIME results.

## Progress and completion

```bash
bash scripts/runpod_aime_baseline.sh status
```

Or `tail -f /workspace/margent-aime-baseline-01.log`. The parent and child runs
share a W&B group beginning `margent-aime-baseline-01-`. The parent job type is
`aime_baseline`; the final child is `aime2026` / `evaluate`. Parent `baseline_complete`
and a report with `n=30` mean the full baseline is complete. A child Finished
alone does not establish whole-run completion. The deliberate dev interruption
may briefly show an interrupted/crashed child before it resumes.

The controller has a **persisted 120-minute wall deadline**, including startup
and preflight. Timeout kills only owned children and advisor, preserves saved
questions and writes `baseline_status.json`; shutdown may need a short grace
period. It does not stop the Pod or its billing. The deadline is not renewed by
restarting. After a budget expiry, extending the budget requires an explicit
follow-up rather than an automatic restart. Until the deadline, the same Python
entry point with the same output and exact code/data/config can resume saved
questions after the old controller and its owned GPU processes have stopped.

Exceptions within the controller are recorded in the parent W&B summary and
sent to W&B's alert service. Actual email delivery depends on the user's W&B
notification settings and has **not** been verified. Abrupt Pod loss/SIGKILL
cannot run Python cleanup or send an in-process alert. External periodic
monitoring is a separate fallback, not a guarantee of instantaneous detection.
Original dataset or manifest errors before W&B initialization appear in the
parent log. Keep the recorded code version when resuming: changed harness or
configuration is deliberately rejected to preserve comparability.
