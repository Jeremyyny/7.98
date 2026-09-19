# Subagent prompts and math experiment status

The three subagents are Extractor, Reasoner, and Verifier. The math module also
uses `advisor` in configuration keys and Python identifiers for these same
subagents; it does not introduce a separate agent type.

## Prompt entry points

| Path | Use |
| --- | --- |
| `src/subagents/prompts/extractor.py`, `reasoner.py`, `verifier.py` | Teacher prompts for generating structured subagent training targets |
| `src/subagents/prompts/runtime_prompts.py` | Structured subagent prompts used by local inference, remote inference, and synthesized SFT input rows |
| `src/subagents/runtime.py` | Both local and remote clients use `build_runtime_messages`; no separate system prompt is embedded here |
| `src/verifiable/protocol.py` | Subagent prompts and tool descriptions for free-response math |
| `src/manager/prompt.py` | Generic MCQ Manager prompt, with an optional explicit task description |

Subagent instructions use the supplied question's subject matter. The structured
Reasoner accepts optional choices; without choices it leaves
`candidate_considerations` empty. Existing JSON field names and answer-disclosure
rules are preserved. The math subagents retain their free-text assistance
contract. The math Manager retains mathematical reasoning instructions and the
boxed final-answer format required by that experiment.

Runnable task-description examples in the guides and training script are generic.
Benchmark names such as MedQA and dataset identifiers still identify their actual
datasets.

## Existing data and runs

Prompt changes apply to newly built requests. Archived JSONL prompts, generated
responses, trained adapters, and evaluation records under `outputs/` are historical
artifacts and have not been rewritten. Scripts that consume existing prompt JSONL
files replay those stored messages; regenerate those input files to use the new
prompts. Existing trained weights do not become domain-neutral through a prompt
edit.

Use a fresh output directory for experiments with these prompts. Record the Git
commit with each run, and restart existing subagent processes to load the changes.
Do not resume a partial experiment across prompt versions or combine its scores
with earlier runs as if the prompts were unchanged.

## Math protocol v2

The math experiment now uses immutable COMMIT, separate revision turns, and
an environment-bound Verifier candidate. Tools have empty arguments. SFT-only
controls, fixed-cohort internalization, conditional call metrics, state transitions
and paired arm comparisons are implemented. See [MATH_RUNPOD.md](MATH_RUNPOD.md)
and [MATH_AUDIT.md](MATH_AUDIT.md) for definitions and validation boundaries.
The legacy GRPO path is blocked under this protocol.
