# Repairing malformed Manager decisions without changing math rewards

The first A100 smoke reached all training stages, but all four sampled GRPO
rollouts were invalid. Saved examples end with an unclosed tool-call tag; one
also names `COMMIT` as a tool. Rewards and advantages are all zero. The nonzero
gradient therefore does not demonstrate outcome learning. The two collection
questions were already solved independently, so their four selected SFT targets
contained COMMIT and solution examples, without positive tool-call examples.
One SFT update was only a plumbing check, not evidence of protocol acquisition.

## 1. Read-only check using the existing SFT adapter

Update `main`, keep the original smoke directory, and run the following once:

```bash
tmux new-session -d -s rsi-decision-check -c /workspace/margent-restart-20260925/agent_routing 'timeout --signal=TERM --kill-after=30s 20m env CUDA_VISIBLE_DEVICES=1 /workspace/margent-venv/bin/python -u scripts/runpod_rsi_decision_check.py --source /workspace/margent-rsi-smoke-01 --out /workspace/margent-rsi-decision-check-01 > /workspace/margent-rsi-decision-check-01.log 2>&1'
tail -n 40 /workspace/margent-rsi-decision-check-01.log
```

This loads `/workspace/margent-rsi-smoke-01/sft` once, reuses its two saved
training decision states, and generates one greedy plus four sampled actions
per state under each condition: original decoding and `finite_actions_v1`.
It makes no optimizer updates, regenerates no solutions, and calls no advisors.
The Manager GPU must be idle. W&B is online under `yuningyangaillm/MATH_rsi`.
The exact sampled token IDs (including EOS) and original output text are saved
in `decision_outputs.json`; `decision_report.json` summarizes valid/total.
Both modes use the same checkpoint, histories, sampling temperature and seeds.

Expected target: the finite-action condition is 10/10 syntactically valid.
This is an engineering check of a changed decoder, **not learned improvement**.
It does not verify the eventual math answer, later advisor turns, the GPU
backward pass with constraints, or a benchmark gain. It does not backfill a
passing report into the failed original run.

## 2. The explicit new policy

`configs/math_rsi_actions.json` opts in to `decision_constraint=finite_actions_v1`.
The original pilot config retains unrestricted decisions. In the new mode,
generation follows a token trie containing exactly COMMIT and canonical calls
to unused tools. EOS is allowed only after a complete action. No gold answers
or rewards are consulted. Answers and revisions remain free-form. The parser
is still strict; malformed old output is never repaired after generation.

Sampling, greedy evaluation, old-policy scoring, reference scoring and gradient
scoring use the same allowed-token sets for decision turns. GRPO stores those
sets as token paths with each turn. Changing only generation without changing
the probability calculation would produce an incorrect policy-gradient loss.
Local tests compare actual generation scores with training scores and perform
backpropagation through a real small Transformer, including forced tokens.

Because this is a new decoding policy, baseline and all comparison arms must
use it consistently and be evaluated again. Old and new results must not be
combined as if they used the same harness. Restart the advisor on matching
code before a new GPU training experiment; use fresh output directories.

## 3. After the decision check passes

The next full smoke can use the new config in a new directory:

```bash
CUDA_VISIBLE_DEVICES=1 /workspace/margent-venv/bin/python -u scripts/runpod_rsi_smoke.py --config configs/math_rsi_actions.json --out /workspace/margent-rsi-smoke-actions-01
```

The orchestrator assigns physical GPUs to its subprocesses itself. It creates
its own advisor, runs SFT, then requires the decision check to pass before GRPO.
Any all-invalid GRPO group saves `invalid_group.json` and continues with its
original zero rewards and zero advantages. KL regularization may still update
weights; that is not outcome learning. Mixed valid/invalid outcomes keep their
original binary rewards; there is no hidden resampling. Zero-advantage groups are still
identified as lacking outcome signal; they are not recast as successful RL.
Reports separate policy loss from weighted KL loss and record the old/reference
log-probability discrepancy. Failures now also produce `smoke_report.json`.

This does not add email alerts or stop Pod billing. Run the short read-only
check first; do not launch the full benchmark before reviewing its report.

## 4. View saved results without rerunning training

After updating main, run `bash scripts/review_rsi_smoke.sh` from `agent_routing`.
The default source is `/workspace/margent-rsi-smoke-actions-01`; pass another
source directory as the first argument if needed. This CPU-only review uploads
reward mean/std, validity rates, advantages, per-rollout text and failure causes
to a new `smoke_review` run in W&B. Exact saved JSON evidence is attached as an
artifact. Original run files, statuses and rewards are never overwritten.

New smoke runs report execution completion separately from rollout quality and
outcome-learning evidence. Invalid samples produce warnings and retain zero
rewards; they no longer fail the entire completed smoke. Missing checkpoints,
nonfinite training or runtime exceptions still fail. This smoke has one GRPO
update and a second SFT update, not a complete two-round RSI benchmark.
