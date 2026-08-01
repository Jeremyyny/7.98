# Visual Spec

## Source

The semantic source is the local `agent_routing` repository. The visual source is
the user-supplied paper figure, matched by the bundled
`topconf-handdrawn-rl-pipeline.png` reference at 1718×744.

## Global Style

- Canvas: 1718×744, white background.
- Typeface: Comic Sans MS.
- Main text: `#020202`; stage titles: `#0D2507`; red accents: `#B5150B`.
- Muted borders: `#C4C4C4`; pale blue: `#D6E1F5`; peach: `#FCE1D6`;
  orange: `#DBB69E`; green: `#6A884E`; purple model marker: `#6762E3`.
- Major containers use 1.5 px dashed strokes with 14 px rounded corners.

## Regions

1. Left input column: research goal, benchmark stream, answer space, task pool.
2. Stage 1 band: advisor-data synthesis, quality gates, LoRA-SFT, frozen advisors,
   Manager cold start.
3. Stage 2 band: Manager rollout, delegate-or-commit actions, frozen advisor
   signals, binary outcome reward, GRPO update, evaluation.

## Text Blocks

Stage titles are 20 pt bold dark green. Subsection titles are 14–17 pt bold,
using red for data/reward emphasis. Body labels are 11–13 pt; connector labels
are 9.5 pt. All visible text is English.

## Shapes

Dashed rounded regions group stages and repeated artifacts. Pale-blue rows encode
structured data/actions, pale-green rows encode frozen signals or trained policy,
and peach rows encode reward or commit semantics. A cylinder encodes the normalized
task pool. Purple stars identify learned LLM components.

## Connectors

Black connectors encode task/control flow, blue connectors encode dataset/training
flow, green connectors encode signals and learned-policy feedback, and orange
connectors encode reward flow. Long feedback routes are orthogonal and occupy
separate horizontal lanes.

## Semantic Relations And Flow

Benchmark questions and A–D options are normalized. Stage 1 uses teacher models
and schema/leakage gates to create three specialist datasets, trains three frozen
advisors, and constructs Manager demonstrations. Stage 2 lets Qwen3-8B Manager_SFT
iteratively consult each advisor at most once or COMMIT. Episode traces receive an
end-of-episode binary reward—1 for a correct final answer and 0 otherwise—and update Manager_GRPO, which is evaluated on
accuracy, consultation efficiency, regret, calibration, and held-out transfer.

## Icons And Images

The figure contains only editable draw.io primitives and Unicode arrow/star
markers. It contains no embedded raster, base64 payload, branded logo, or external
image reference.
