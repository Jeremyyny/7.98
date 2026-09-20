# Math supplementary experiment — protocol v2

This is the SFT-only supplementary experiment for MARGENT: measure changing
need for delegation while the Manager learns from successful generated solutions.
The supported arms are `dynamic_sft`, `success_sft`, and `static_sft`.
The legacy GRPO command raises an error because its environment does not implement
immutable COMMIT. It must not be mixed into this experiment.

For executable setup and launch commands, use
[MATH_RUNPOD_QUICKSTART.md](MATH_RUNPOD_QUICKSTART.md).
See [MATH_AUDIT.md](MATH_AUDIT.md) for audit findings and validation boundaries.

## Claim and operational definitions

Within each checkpoint, an initial complete solution is the shared candidate.
COMMIT submits that candidate unchanged. A subagent call obtains advice and is
followed by one explicit Manager revision. The next decision uses that revised
candidate. At the call limit the current candidate is submitted automatically.
Verifier inputs are bound by the environment to the full stored candidate; a
routing action cannot substitute another derivation.

Collection and deployment share the same revision function, prompts, token limits
and sequence-based generation seeds. The model only emits `COMMIT` or a single
JSON native tool call in a decision turn. Every tool has empty arguments; task
inputs are supplied by the environment. Invalid decisions count as failures.

Across checkpoints, match the same held-out questions, not identical generated
states. A question initially solved only through a measured delegation branch
that later becomes independently correct is operationally internalized. This is
answer-level evidence on held-out questions, not step correctness or a proof that
a general mathematical capability has been acquired.

## Data and grading

`prepare` pins dataset revisions, filters NuminaMath-1.5 to valid non-proof,
non-MCQ, text-only, parseable-answer examples, then deduplicates before splitting.
Source solutions never enter Manager or subagent prompts. AIME 2026 (30) and
BeyondAIME (100) are locked external tests. Their upstream split names are not
their role in this experiment. Official tests must load intact.

Train, dev and test files are checksum-locked and checked for overlap. Deduplication
normalizes case, whitespace and Unicode; it is not semantic decontamination and
cannot remove unknown pretraining exposure. The answer grader requires one final
line `FINAL_ANSWER: \boxed{...}`. Plain and parsed rational/decimal values use exact
numeric comparison; other supported expressions use Math-Verify. Correct final
answers do not verify every step of a generated solution. Inspect pilot traces
for incorrect reasoning with a correct answer and for answer-format failures.

## Collection and training

For depth two, collection evaluates all three one-call and six distinct two-call
sequences, even when the initial solution is correct. This is finite measured
coverage, not an unrestricted oracle or an estimate of all possible trajectories.

MARGENT chooses an empty path for a correct initial candidate; otherwise it chooses
a shortest successful path, with deterministic tie-breaking. Failed trajectories
are not solution targets. Call decisions, the successful terminal revision,
COMMIT, and a question-only/full-solution distillation example are exported. Earlier
incorrect intermediate revisions are context only, not supervised solution targets.
The revised solution is requested to be self-contained; manual quality inspection
is still needed.

| Arm | Selector | Collection | Training |
| --- | --- | --- | --- |
| `dynamic_sft` | Shortest successful trajectory | Refresh each round | Continue SFT weights |
| `success_sft` | Uniform random successful trajectory | Refresh each round | Continue SFT weights |
| `static_sft` | Shortest successful trajectory | Reuse round-one data | Continue SFT weights |

The same selected question cohort and direct/rescue sampling rule are used before
choosing shortest vs random-success trajectories at a common checkpoint. The
first-round branch pool is deterministically reproducible across arms because
generation uses the fixed `generation_seed`, independent of training seed. Later
pools differ because the learned Managers differ. Subagent weights remain frozen.

Use the same base revision, data split, subagent, prompt protocol, two-round plan,
positive `sft_max_steps`, accumulation, learning rate, LoRA settings and sequence
limits in comparisons. Longer trajectories contain more supervised turns; equal
optimizer updates are not equal tokens, FLOPs or GPU hours. Actual input and
supervised tokens are recorded in `usage.jsonl`, `sft_data_report.json` and report
cost tables. Do not claim equal-compute superiority from these controls.
Oversized SFT targets cause a failure rather than silently changing the cohort.

Two-call search is required for the shortest-vs-random trajectory comparison:
at depth one all rescue paths have length one. GRPO is not required for this
supplement. Training starts from the same base model and continues adapters
between rounds; it does not update subagent weights.

## Measurements

Each dev checkpoint records independent accuracy, deployed policy accuracy,
bounded search coverage, validity, truncation, mean calls and token costs. It also
runs one self-revision control whose maximum generated-token allowance equals one
subagent response plus one Manager revision. This does not match the compute of
the entire multi-call search or deployed policy.

`delegation_behavior.csv` and W&B `delegation/*` include group sizes and behavior
on currently independent, currently rescuable, and unresolved questions. A fixed
initially rescuable cohort is tracked through all nine state transitions. For the
subset that becomes independently correct, report calls both before and after
on exactly those same questions. Also report regressions, invalid decisions and
policy accuracy on questions still rescuable. Empty groups have missing rates,
not zero rates. These changing-subset analyses are descriptive, not causal proof
that internalization alone caused call reductions.

External tests evaluate only the predeclared initial and final checkpoints after
the loop. Do not tune budgets, seeds, data size or checkpoints against test scores.
`arm_comparisons.csv` pairs final question outcomes between `dynamic_sft` and the
controls within each seed/protocol; `arm_summary.csv` reports mean and SD of those
seed-level differences. Within-run changes are in `paired_changes.csv`.

Intervals are question-level 95% Wilson intervals and paired bootstrap intervals.
They do not replace training-seed variability. McNemar p-values are exploratory
and unadjusted. Small competitive test sets and small rescued cohorts can yield
wide uncertainty. Always report all predeclared seeds, including negative results.

## Reproducibility and completion

`freeze-config` resolves a base-model revision. Run manifests also lock the
executable harness hash, dataset hashes, checkpoints and configuration. Frozen
subagent identity is checked across stages and HTTP responses. Use a new output
directory after any code, prompt, data or protocol change. Protocol-v1 checkpoints
and unfinished runs must not be resumed as protocol-v2 experiments.

Each example is saved atomically before appending the combined record file, so a
torn append can be rebuilt on resume. Stage completion requires saved artifacts.
Adapter continuation restores the last optimizer checkpoint after interruption.
Retries remain in the usage ledger; interruption can make full cost accounting
incomplete, which the report flags instead of inventing cost values.

`paper-check` regenerates the report and checks both primary arms with at least
two matched training seeds, pinned official sources, the same protocol/budgets,
completed SFT update budgets and locked initial/final external tests. It emits
`paper_readiness.json` and exits nonzero on missing requirements. Passing it means
protocol completeness, not positive results, statistical significance or a
publication guarantee. It is not a validator of all generated reasoning steps.

W&B is optional (`MARGENT_WANDB_MODE=online`, `offline`, or `disabled`). Scalar
metrics and configuration are the default. `MARGENT_WANDB_TEXT=1` additionally
uploads question/answer and per-generation text tables; weights remain local.
Local `generations.jsonl` is written before truncation errors so incomplete
questions retain their advisor output. Text-upload errors do not turn into math
labels or replace the original generation error. Tables use incremental uploads,
separate attempt suffixes on resume, and explicit display limits (10000 rows and
20000 characters per cell by default; full text stays local).
`wandb-upload-records --run-dir STAGE --out NEW_REVIEW_DIR` creates a model-free
review run from existing records without altering the source experiment.
Older cost records lack some prompts/phase details; these remain unknown, and
outputs never saved by the old code cannot be recovered. The review preserves
source hashes, configuration and harness identity without recomputing labels.
See `MATH_RUNPOD_QUICKSTART.md` for table fields, limits and migration commands.
Online runs resume their stage identity; offline attempts have separate IDs
under a shared group. Loss uses `trainer_step`; dev and conditional delegation
metrics use `diagnostic_step`.
