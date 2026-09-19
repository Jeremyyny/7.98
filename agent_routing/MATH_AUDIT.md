# Math experiment audit — protocol v2

Scope: the free-response math supplement. No experimental gains are fabricated or
inferred from synthetic tests. GPU memory, throughput and real-model behavior are
still runtime validation gates.

| Component | Finding and resolution | Remaining boundary |
| --- | --- | --- |
| Data | Reject proof/MCQ/image-required Numina records; preserve zero answers; lock source revisions, split hashes and disjoint question identities | Exact normalized dedup only; dataset/model pretraining contamination is unknown |
| Answers | Require a single terminal boxed declaration; compare parsed numeric near-misses exactly | Symbolic grading follows Math-Verify; correct outcomes do not validate reasoning steps |
| COMMIT | Old evaluation could revise a candidate while claiming no delegation; v2 separates routing and revision and submits stored candidates unchanged | Initial base-model adherence must be checked on GPU |
| Same-state comparisons | Collection and policy now call the same revision function with identical prompts, bound candidate and sequence-based seed | Finite depth/role search only |
| Verifier | Environment supplies the exact stored full candidate; tool arguments cannot replace it | Verifier is a fallible LM, not the gold grader |
| Native tools | A fixed JSON template is used; preserve tool markers during decoding and reject malformed, repeated or multi-call decisions | Real Qwen3.5 tokenizer checked; GPU model integration still requires smoke test |
| Loss masking | Tokenize the exact inference prefix and target suffix separately, avoiding whitespace BPE merges across the supervision boundary | A non-prefix-preserving chat template fails explicitly |
| Supervision | Train call decisions, a successful terminal revision, COMMIT and complete question-only solutions; never supervise failed intermediate revisions | Manually inspect successful derivations for spurious correctness/context dependence |
| Controls | Add success_sft and static_sft; all paper arms use SFT, fixed rounds and positive optimizer-step budgets | Actual supervised/input tokens differ; no equal-FLOP claim |
| GRPO | Explicitly disabled for protocol v2; archived implementation remains for reference | Implement and separately validate a compatible environment before any RL extension |
| Mechanism metrics | Fixed-cohort internalization, group denominators, all state transitions, calls before/after learning, remaining rescue success and regressions | Changing-subset results are descriptive |
| Statistics | Add paired final comparisons between arms and summarize seed-level deltas; retain question CIs and exploratory p-values | At least two seeds is a small repeated experiment; three or more is preferable if predeclared and affordable |
| External tests | Complete initial/final evaluations only; require stage completion and verify raw answers against saved labels | Small competitive sets may have floor effects and wide CIs |
| Provenance | Pin model revisions and hash the executable harness; check frozen subagent identity; reject changed-resume settings | Reproduce with recorded software/hardware, not just a code branch name |
| Recovery | Atomic per-question shards rebuild a torn JSONL append; validate stage/adapter artifacts | Interrupted compute accounting may be incomplete and is flagged |
| Reporting | Export condition-specific behavior, arm comparisons, costs and readiness checks | A completed report does not establish the research claim without favorable, reliable observations |
| RunPod | Separate frozen subagent and Manager GPUs; actual-model smoke test, limited math pilot and generation-time estimate before full runs | No A100 execution has been performed by this audit |

The default executable route is two rounds of SFT with depth-two search. For a
seven-day window, start with 128 train / 64 dev questions and two matched seeds
for the two primary arms; use observed pilot throughput to decide whether to
increase size or add the static control before opening external test results.

Validation commands:

```bash
python -m pytest -q tests/test_verifiable.py tests/test_math_reporting.py tests/test_math_paper_protocol.py tests/test_math_wandb.py tests/test_marginal_value.py tests/test_routing_anchor.py tests/test_benchmark_loaders.py
MARGENT_CPU_INTEGRATION=1 python -m pytest -q tests/test_math_training_integration.py
```

Validation observed in this audit: 64 regression tests and 3 CPU integration
tests passed with torch 2.8.0+cpu, Transformers 5.3.0, TRL 0.29.0 and PEFT 0.18.1.
The integration exercised LoRA training, continued training and adapter reload,
plus Qwen3.5 text-weight loading. The actual Qwen3.5-9B tokenizer at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` passed prefix/masking and decoded-target
checks for call, revision, COMMIT and independent-solution rows. `pip check` and
shell syntax checks passed.

The CPU integration uses tiny randomly initialized models and synthetic fixtures,
not paper benchmark results. No A100 model execution or online W&B upload was
performed; those are the next runtime gates in the RunPod guide.
