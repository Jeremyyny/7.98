# Diagram Brief

## User Goal

- Output: an editable, English-only draw.io research framework figure.
- Audience: paper readers who should understand the contribution without reading the code.
- Must communicate: a manager is trained from a binary final-answer reward to decide whether to consult another frozen advisor or commit now.
- Must not do: look like an infrastructure diagram; use Chinese labels; imply that advisors answer the question or that teacher APIs are used during deployment.

## Source Inventory

| id | source | type | role | priority | notes |
|---|---|---|---|---|---|
| S1 | repository README and experiment plan | text/code | content | high | Defines the research objective and stages. |
| S2 | `src/subagents/*` | code | structure | high | Defines teacher synthesis, schemas, leakage gates, frozen advisors, and caching. |
| S3 | `src/manager/*` | code | structure | high | Defines manager cold start, delegate/commit loop, GRPO, binary outcome reward, and evolve path. |
| S4 | user-supplied reference image | image | style + layout | high | Defines the paper-figure visual language, not scientific content. |
| S5 | bundled `topconf-handdrawn-rl-pipeline.png` | local image | measurable style source | high | Pixel-identical visual family used for color/layout extraction. |

## Requirement Traceability

| id | requirement | source evidence | priority | visual encoding |
|---|---|---|---|---|
| R1 | English only | user feedback | must | All visible labels are English. |
| R2 | Explain what the project is doing | user feedback, S1-S3 | must | One-sentence contribution banner plus two staged bands. |
| R3 | Match the reference's figure grammar | user image, S4-S5 | must | Left input column, dashed stage bands, handwritten font, muted fills, dense annotated flow. |
| R4 | Preserve manager/advisor authority | S1-S3 | must | Advisors labeled “signals only”; only Manager has a COMMIT path. |
| R5 | Show training objective | user correction | must | Explicit binary reward: correct final answer = 1, incorrect final answer = 0. |
| R6 | Show offline vs runtime separation | S2-S3 | must | Teacher synthesis appears only in Stage 1; frozen advisors appear in Stage 2. |

## Semantic Model

| id | entity / relationship | direction | visual encoding | uncertainty |
|---|---|---|---|---|
| M1 | MCQ benchmarks → normalized task pool | many-to-one | left input panel and database | none |
| M2 | task pool + teacher → gated advisor SFT data | fan-in | Stage 1 data-generation flow | none |
| M3 | gated data → three frozen advisor adapters | one-to-three | stacked advisor toolkit | none |
| M4 | cold-start demonstrations → manager SFT | one-to-one | Stage 1 right-side model | none |
| M5 | manager draft → consult or commit | one-to-four action | central Stage 2 routing decision | none |
| M6 | advisor signals → revised draft | fan-in feedback | green return path | none |
| M7 | final answer → binary reward → GRPO update | feedback | orange reward loop | none |
| M8 | trained manager → evaluation | one-to-one | right-side report panel | none |

## Style Contract

- Use the exact palette and dimensions in `style-extraction.md`.
- Font: Comic Sans MS throughout.
- Canvas: 1718 × 744, matching the reference aspect ratio.
- Major section containers: dashed gray, transparent fill, 14 px radius.
- Stage headings: dark green, centered, underlined.
- Subsection headings: sampled red.
- Body boxes: mostly open/white; selective pale blue, peach, and green.
- Primary arrows: black/gray; rewards orange; policy/update green.

## Open Assumptions

| assumption | risk | verification |
|---|---|---|
| The figure should explain the method contribution rather than every CLI stage. | Some operational details are omitted. | Keep exact module names out of the main figure and provide them separately if requested. |
| A single paper-scale figure is preferred over multiple pages. | Dense content may need 10–12 pt labels. | Validate at the reference's 1718×744 resolution and iterate from screenshots. |
