# Defect Log

Append-only after Screenshot Review Cycle 1.

## Pass

| pass | screenshot | capture | resolution | outcome |
|---|---|---|---|---|
| 1 | `paper-pass-1.png` | canvas-only | 1718×744 | Three P0 layout defects found. |
| 2 | `paper-pass-2.png` | canvas-only | 1718×744 | P0 defects fixed; bottom lanes refined. |
| 3 | `paper-pass-3.png` | canvas-only | 1718×744 | Semantic normalization gap found. |
| 4 | `paper-pass-4.png` | canvas-only | 1718×744 | Final reviewed artifact; no P0/P1 blocker. |
| 5 | `paper-binary-pass-1.png` | canvas-only | 1718×744 | Binary reward correction; one label defect found. |
| 6 | `paper-binary-final.png` | canvas-only | 1718×744 | Binary reward final; no P0/P1 blocker. |

## User-Found Failure From Previous Version

- The previous figure used an engineering-dashboard visual grammar rather than the supplied hand-drawn research-figure grammar.
- It mixed Chinese explanatory text with English code identifiers despite the user's expectation of English-only labels.
- The correction is structural: rebuild the composition from the style reference instead of recoloring the old layout.
- The user later corrected the reward semantics: the final reward is binary. All
  visible ADC, anytime-draft, missing-draft, and consultation-cost reward terms
  were removed and replaced by `correct → 1 / incorrect → 0`.

## Preflight Review

- `validate_visual_quality.py`: 0 FAIL, 1 WARN.
- Reviewed `spacing-inconsistent-v`: accepted. The checker combines the three Stage-1 advisor rows and four Stage-2 reward rows because their x coordinates are similar even though they are in separate stage bands. Spacing is internally uniform inside each group (9 px and 8 px respectively).

## Screenshot Review Cycle 1

Evidence: `paper-pass-1.png`, canvas-only, 1718×744. Compared side-by-side with
`topconf-handdrawn-rl-pipeline.png`.

| id | zone | element | issue | severity |
|---|---|---|---|---|
| C1-01 | Text | `goal_panel` | Multiline content is rendered as one line and clips off the left canvas edge. | P0 |
| C1-02 | Text | `task_panel` | Benchmark/task content is rendered as one line and clips off the left canvas edge. | P0 |
| C1-03 | Arrows | `trace_to_reward` | The episode-trace route and label cross the Stage-2 title. | P0 |
| C1-04 | Text | `evaluation` | The narrow evaluation panel makes the last metric line visually cramped. | P1 |
| C1-05 | Text | `grpo_to_eval` | “trained policy” is partially obscured near the evaluation path. | P1 |
| C1-06 | Arrows | bottom feedback | “signal + revised draft” and “policy update” sit on nearby parallel routes. | P1 |
| C1-07 | Arrows | `task_to_teacher` | The long vertical route nearly merges with the Stage-1 left border. | P1 |
| C1-08 | Boxes | `rollout_manager` | The model box identifies the adapter but not the Qwen3-8B base. | P1 |
| C1-09 | Boxes | `manager_grpo` | The trained-manager box does not identify the Qwen3-8B base. | P1 |
| C1-10 | Text | `reward_*` | Reward-term labels are readable but smaller than the reference’s comparable reward rows. | P2 |
| C1-11 | Text | `frozen_*` | Frozen-advisor row text is slightly smaller than the reference’s model/result rows. | P2 |
| C1-12 | Text | arrow labels | `generate`, `accept`, `train`, and `freeze` are close to adjacent boxes. | P2 |
| C1-13 | Typography | overall | Comic Sans hierarchy matches the reference, but some labels remain too uniformly bold. | P2 |
| C1-14 | Typography | subtitle | The global subtitle is visually strong enough to compete with Stage-1 heading. | P2 |
| C1-15 | Boxes | `advisor_sft` | Tall LoRA-SFT block is somewhat sparse compared with the reference model blocks. | P2 |
| C1-16 | Boxes | `coldstart_demo` | Cold-start trajectory bar is wide and visually flat. | P2 |
| C1-17 | Spacing | Stage 1 | Vertical white space between advisor panels and cold-start bar is tight. | P2 |
| C1-18 | Spacing | Stage 2 | Rollout Manager and Rollout Trace have a smaller gap than later panels. | P2 |
| C1-19 | Color | Stage 1 | Green advisor toolkit border is slightly darker than nearby pale blue data elements. | P2 |
| C1-20 | Color | Stage 2 | Orange routing and reward containers dominate more than the reference’s pale peach borders. | P2 |
| C1-21 | Layout | left column | `MCQ Task Stream` heading floats between two large dashed panels. | P2 |
| C1-22 | Layout | Stage 1 | The long cold-start lane has less internal structure than the reference’s instruction/result lane. | P2 |
| C1-23 | Layout | Stage 2 | The COMMIT-to-evaluation route is long and visually detached from the action card. | P2 |
| C1-24 | Layout | contribution | Red contribution sentence sits close to the Stage-2 lower border. | P2 |
| C1-25 | Icons | manager blocks | Purple star marker is meaningful but small compared with the reference’s LLM logo. | P2 |
| C1-26 | Icons | task pool | Database cylinder is clear but visually heavier than the reference’s small source icons. | P2 |
| C1-27 | Style | overall | The composition is now reference-like, but still slightly more geometric/regular than hand-drawn. | P2 |
| C1-28 | Style | left input | The reference uses a concrete text excerpt; this figure uses an abstract task summary. | P2 |
| C1-29 | Semantics | `frozen_toolkit` | “signals only” is explicit only in Stage 2, not Stage 1. | P1 |
| C1-30 | Semantics | `routing_actions` | COMMIT is clear, but the one-call-per-advisor constraint is not shown. | P1 |

## Fix Verification Cycle 1

- Fixed C1-01/C1-02: multiline dashed-panel text now renders inside both left panels.
- Fixed C1-03: the episode-trace route now runs through the inter-stage gutter.
- Fixed C1-04/C1-05: the evaluation panel is wider and its incoming policy path is routed.
- Fixed C1-07: the sample path is separated from the Stage-1 border.
- Fixed C1-08/C1-09: all Manager blocks now name the Qwen3-8B base.
- Fixed C1-29/C1-30: “Signals Only” and the one-call-per-advisor constraint are explicit.

## Screenshot Review Cycle 2

Evidence: `paper-pass-2.png`, canvas-only, 1718×744.

| id | zone | element | issue | severity |
|---|---|---|---|---|
| C2-01 | Layout | `contribution` | The red contribution sentence sits on the Stage-2 bottom border. | P0 |
| C2-02 | Arrows | bottom feedback | The green signal-return path and black COMMIT path are only five pixels apart. | P1 |
| C2-03 | Arrows | `grpo_to_eval` | The long “trained policy” label is cramped beside the evaluation panel. | P1 |
| C2-04 | Arrows | `commit_to_eval` | The COMMIT route spans several modules and competes with the return path. | P1 |
| C2-05 | Typography | `evaluation` | Metric rows are readable but visually dense at the right edge. | P1 |
| C2-06 | Typography | `frozen_toolkit` | The long Stage-1 toolkit title nearly reaches both dashed borders. | P2 |
| C2-07 | Typography | `routing_actions` | The ≤1 constraint is clear but the compact header has limited breathing room. | P2 |
| C2-08 | Arrows | `questions` | The pale-gray label has low contrast against the long cold-start bar. | P2 |
| C2-09 | Arrows | `episode trace` | The top inter-stage route is deliberately long and visually prominent. | P2 |
| C2-10 | Arrows | `tool traces` | The Stage-1 feedback route is close to the toolkit’s lower boundary. | P2 |
| C2-11 | Spacing | Stage 1 | The teacher-to-data row is compact compared with the large white stage area. | P2 |
| C2-12 | Spacing | Stage 2 | Evaluation begins higher than Manager_GRPO, weakening their relationship. | P2 |
| C2-13 | Hierarchy | Stage 2 | The orange reward heading is stronger than the green trained-policy block. | P2 |
| C2-14 | Hierarchy | Stage 1 | The global subtitle and Stage-1 heading remain close in visual weight. | P2 |
| C2-15 | Semantics | `coldstart_demo` | The trajectory communicates sequence but not that these are manager demonstrations. | P2 |
| C2-16 | Semantics | `quality_gates` | Leakage protection is shown, but the reason for symmetric auditing is implicit. | P2 |
| C2-17 | Semantics | `evaluation` | Cross-domain transfer is named without clarifying it uses held-out tasks. | P2 |
| C2-18 | Style | overall | The figure is cleaner and more geometric than the intentionally irregular reference. | P2 |

## Fix Verification Cycle 2

- Fixed C2-01: the contribution is now a separate caption below the Stage-2 boundary.
- Fixed C2-02/C2-04: signal return, COMMIT, and policy-update routes use distinct vertical lanes.
- Fixed C2-03: the deployment label is shortened and routed into the evaluation panel.
- Fixed C2-06/C2-08/C2-15/C2-17: headings and labels were shortened or made more explicit.

## Screenshot Review Cycle 3

Evidence: `paper-pass-3.png`, canvas-only, 1718×744.

| id | zone | element | issue | severity |
|---|---|---|---|---|
| C3-01 | Semantics | left input | The answer-space card and normalized task pool are not explicitly connected. | P1 |
| C3-02 | Layout | `contribution` | Caption has a small bottom margin, though it is fully visible. | P2 |
| C3-03 | Arrows | bottom feedback | Three bottom lanes remain visually dense but no longer overlap. | P2 |
| C3-04 | Arrows | `trace_to_reward` | Long route leaves Stage 2 and re-enters at the reward block. | P2 |
| C3-05 | Arrows | `toolkit_to_demo` | Route originates from the verifier row although its label represents all tool traces. | P2 |
| C3-06 | Semantics | `teacher_synth` | Vendor examples are shown, while teacher selection/fallback policy is omitted. | P2 |
| C3-07 | Semantics | `advisor_sft` | “×3” is compact but relies on adjacency to identify the three specialists. | P2 |
| C3-08 | Semantics | `manager_sft` | Qwen3-8B is clear; LoRA/adapter detail is intentionally omitted. | P2 |
| C3-09 | Typography | tiny labels | `generate`, `accept`, `train`, and `freeze` are legible only at full-resolution viewing. | P2 |
| C3-10 | Typography | `ground truth` note | The red note is deliberately small and subordinate. | P2 |
| C3-11 | Hierarchy | Stage 2 | Reward and routing containers have equal visual weight despite different roles. | P2 |
| C3-12 | Layout | evaluation | Evaluation is close to the right stage boundary but has sufficient padding. | P2 |
| C3-13 | Color | action rows | Pale-blue action rows and synthetic-data rows share a color for “structured artifacts.” | P2 |
| C3-14 | Style | arrows | Orthogonal routing is cleaner than the reference’s hand-drawn arrow irregularity. | P2 |
| C3-15 | Scope | overall | Operational commands, caches, and checkpoint filenames are omitted from the paper-scale view. | P2 |

## Fix Verification Cycle 3

- Fixed C3-01: the answer-space card now feeds the normalized task pool through an explicit `normalize` edge.
- Re-rendered at 1718×744; all text remains visible and all feedback lanes remain separated.

## Requirement And Semantic Audit

All user-visible text is English and the composition follows the supplied
paper-figure grammar rather than an engineering dashboard. The figure preserves
the repository’s central authority boundary: frozen advisors return signals, while
only the Manager can revise its draft and COMMIT. Teacher synthesis and quality
gates appear only in offline Stage 1. Stage 2 explicitly contains the constrained
routing action space, binary final-answer reward, GRPO policy update, and held-out
evaluation. The operational implementation details intentionally omitted from this
paper-scale view do not change the scientific method shown.

## Red-Team Visual Audit

| finding | zone | adversarial check | result |
|---|---|---|---|
| RT-01 | Z1 canvas | Is any content clipped by the 1718×744 canvas? | No; final canvas-only screenshot shows full extents. |
| RT-02 | Z1 canvas | Does the bottom caption touch the image edge? | It retains a small but visible bottom margin. |
| RT-03 | Z1 canvas | Is the visual center pulled away from the two stages? | No; the large dashed bands dominate the canvas. |
| RT-04 | Z2 title | Is the core contribution discoverable in five seconds? | Yes; title, goal, Stage-2 heading, and caption repeat it at different levels. |
| RT-05 | Z2 title | Does the subtitle compete with Stage 1? | Slightly, but Stage 1 remains larger and structurally bounded. |
| RT-06 | Z2 title | Are all visible labels English? | Yes. |
| RT-07 | Z3 input | Are the benchmark sources explicit? | Yes; MedQA, LegalBench, MMLU-Pro, and GPQA are listed. |
| RT-08 | Z3 input | Is the answer format explicit? | Yes; multiple-choice options A–D are shown. |
| RT-09 | Z3 input | Is normalization visually connected? | Yes; a labeled edge feeds the normalized task pool. |
| RT-10 | Z3 input | Could ground truth leak into routing? | The red note explicitly says it is hidden. |
| RT-11 | Z4 Stage 1 | Could teacher APIs appear to be required online? | No; teacher synthesis is confined to Stage 1. |
| RT-12 | Z4 Stage 1 | Are synthesis quality requirements visible? | Yes; JSON, schema, coverage, and leakage gates are listed. |
| RT-13 | Z4 Stage 1 | Are three specialist datasets distinguishable? | Yes; Extractor, Reasoner, and Verifier have separate rows. |
| RT-14 | Z4 Stage 1 | Is the three-adapter training multiplicity visible? | Yes; LoRA-SFT ×3 is explicit. |
| RT-15 | Z4 Stage 1 | Could advisors be mistaken for answer owners? | No; the toolkit title says Signals Only. |
| RT-16 | Z4 Stage 1 | Is Manager cold start linked to tool traces? | Yes; demonstrations receive questions and frozen-advisor traces. |
| RT-17 | Z5 Stage 2 | Is the rollout model identity clear? | Yes; Manager_SFT and Qwen3-8B are named. |
| RT-18 | Z5 Stage 2 | Is iterative draft evolution visible? | Yes; Draft₀, action, signal, Draft₁, and COMMIT form a trace. |
| RT-19 | Z5 Stage 2 | Is the one-call constraint explicit? | Yes; the action header says ≤1 each. |
| RT-20 | Z5 Stage 2 | Is COMMIT visually distinct from consult actions? | Yes; it uses a peach action card. |
| RT-21 | Z6 advisors | Are advisor responsibilities distinguishable? | Yes; evidence, scaffold, and draft audit are separately labeled. |
| RT-22 | Z6 advisors | Do runtime advisors remain frozen? | Yes; the group is titled Frozen Advisors. |
| RT-23 | Z6 advisors | Do signals return to the rollout trace? | Yes; a dedicated green return lane is labeled. |
| RT-24 | Z7 reward | Is the reward explicitly binary? | Yes; the panel is titled Binary Outcome Reward. |
| RT-25 | Z7 reward | Is a correct final answer mapped to the right value? | Yes; correct maps to 1. |
| RT-26 | Z7 reward | Is an incorrect final answer mapped to the right value? | Yes; incorrect maps to 0. |
| RT-27 | Z7 reward | Are obsolete ADC/cost terms still visible? | No; only the two terminal outcomes remain. |
| RT-28 | Z8 evaluation | Are quality and efficiency both evaluated? | Yes; accuracy and average consultations are both listed. |
| RT-29 | Z8 evaluation | Are regret, calibration, and transfer covered? | Yes; oracle regret, risk–coverage, and held-out transfer are present. |
| RT-30 | Z9 integrity | Is the artifact editable and self-contained? | Yes; native draw.io primitives only, with no raster or external images. |

## Final Self-score

| dimension | score / 10 |
|---|---:|
| Semantic correctness | 9 |
| Flow clarity | 9 |
| Legibility | 8 |
| Reference-style fidelity | 8 |
| Editability and technical integrity | 10 |
| **TOTAL** | **44/50** |

Final validation:

- `validate_drawio.py --strict --json`: 0 errors, 0 warnings.
- `validate_visual_quality.py`: 0 FAIL, 1 reviewed WARN. The warning groups Stage-1 advisor rows and Stage-2 reward rows into one x-column; spacing inside each actual group is uniform.

## Remaining Gaps

- The figure intentionally omits CLI commands, cache filenames, and checkpoint
  directories; those belong in an implementation appendix rather than this
  method overview.
- The native draw.io geometry is cleaner than the reference’s hand-drawn
  irregularity, but preserves its typography, palette, density, and stage grammar.
- The single visual-quality warning is a cross-stage x-coordinate grouping
  artifact, not an actual spacing inconsistency inside either component group.

## Screenshot Review — Binary Reward Correction

Evidence: `paper-binary-pass-1.png`, canvas-only, 1718×744.

| id | zone | element | issue | severity |
|---|---|---|---|---|
| BR-01 | Semantics | reward panel | Correct and incorrect outcomes now map unambiguously to 1 and 0. | verified |
| BR-02 | Semantics | reward panel | No ADC, draft-accuracy, missing-draft, or consultation-cost term remains visible. | verified |
| BR-03 | Text | global subtitle | Subtitle now describes learning from binary task outcomes. | verified |
| BR-04 | Text | contribution caption | Caption no longer claims cost-aware optimization. | verified |
| BR-05 | Arrows | final-answer route | Rollout trace feeds the binary reward through a `final answer` edge. | verified |
| BR-06 | Arrows | reward update | Binary reward feeds Manager_GRPO. | verified |
| BR-07 | Text | `grpo_to_eval` | The deployment label is clipped beside the evaluation border. | P1 |
| BR-08 | Boxes | binary reward | Two outcome rows are balanced and fit inside the shortened panel. | verified |
| BR-09 | Spacing | Manager_GRPO | Moving the manager upward creates a clear reward-to-policy gap. | verified |
| BR-10 | Regression | full canvas | No unrelated Stage-1 or routing label regressed. | verified |

## Fix Verification — Binary Reward Correction

- Fixed BR-07 by routing the trained Manager to the bottom of the Evaluation
  panel and shortening the edge label to `eval`.
- Verified in `paper-binary-final.png`: both binary outcomes are legible, the
  final-answer and reward-update edges are unobstructed, and no obsolete reward
  term appears in the final canvas.
