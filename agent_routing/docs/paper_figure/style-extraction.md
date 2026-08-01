# Style Extraction: topconf-handdrawn-rl-pipeline.png

Reference: the bundled local copy that visually matches the user's supplied image.
Measured image size: 1718 × 744 px. Colors below were sampled from the local PNG.

## 1. Palette

| role | hex | used on |
|---|---|---|
| background | `#FFFFFF` | canvas and open stage bands |
| primary neutral fill | `#F2F2F2` | light instruction/data cards |
| secondary fill | `#D6E1F5` | result rows and cool highlights |
| accent heading | `#B5150B` | red subsection headings |
| border / muted stroke | `#C4C4C4` | dashed containers and secondary connectors |
| main text / arrow | `#020202` | body text and primary flow |
| stage heading | `#0D2507` | stage titles |
| orange reward fill | `#FCE1D6` | reward and optimization components |
| orange reward stroke | `#DBB69E` | reward borders and update flow |
| green system stroke | `#6A884E` | learned policy and evaluation |
| model accent | `#6762E3` | LLM / learned-model markers |

The implementation uses these sampled colors consistently; antialiased variants from
the raster reference are not introduced as new semantic colors.

## 2. Typography

- Heading font: Comic Sans MS, 20 pt, bold.
- Subheading font: Comic Sans MS, 16 pt, bold.
- Body text font: Comic Sans MS, 12 pt.
- Small label / caption font: Comic Sans MS, 9–10 pt.
- Code / exact interface tokens: Comic Sans MS, 10–11 pt, bold where needed.

## 3. Shape Language

- Corner radius: 14 px, with occasional pill-shaped model boxes.
- Stroke width for boxes: 1.5 px.
- Stroke width for arrows: 1.5 px.
- Dash pattern for containers: `10 8`.
- Shadow: no.
- Background regions: transparent/white.
- Important model/reward blocks: pale fills with colored outlines.

## 4. Layout Rhythm

- Outer margin: 70 px.
- Gap between major regions: 22–28 px.
- Gap between same-row elements: 18–24 px.
- Padding inside boxes: 10 px vertical, 14 px horizontal.
- Typical box: 150–210 px wide, 70–110 px high.
- Grid: 5 px.
- Composition: narrow left input column plus two full-width stacked stage bands.

## 5. Arrow Grammar

- Default arrow: small classic arrowhead.
- Main arrow color: `#020202`; secondary flow: `#C4C4C4`.
- Routing: straight for short flows, orthogonal for stage-spanning flows.
- Labels: yes, 9–10 pt, one or two words.
- Orange arrows: reward/update.
- Green arrows: learned-policy feedback or deployment.
- Dashed arrows: optional/failure-recycling paths.

## 6. Icon Language

- Minimal editable primitives and built-in draw.io document/cylinder/note shapes.
- Typical icon size: 28–42 px.
- Stroke width: 1.5 px.
- Model marker: purple star/diamond carrying the semantic meaning “learned LLM policy”.
- No branded logos and no embedded raster assets.

## 7. Density & Composition

- Diagram type: dense staged pipeline with a feedback loop.
- Major regions: left input/task panel, Stage 1, Stage 2, right evaluation/deployment.
- Content density: dense.
- Whitespace: moderate; stage bands are full but not card-grid heavy.
- Panel labels: named stages.
- Legend: integrated through repeated colors and direct labels.
- Caption: none.

## Semantic Justification

| element | visual form | what it represents | each unit corresponds to | justified? |
|---|---|---|---|---|
| benchmark list | stacked document cards | the four supported MCQ benchmarks | one card = one benchmark family | YES |
| A/B/C/D row | four labeled cells | the multiple-choice action space | one cell = one answer option | YES |
| three advisor cards | vertically stacked specialist cards | Extractor, Reasoner, Verifier | one card = one frozen advisor | YES |
| purple model marker | star/diamond inside model boxes | a learned LLM policy or adapter | one marker = one trainable model component | YES |
| database cylinder | cylinder | cached normalized data / traces | one cylinder = one persistent artifact store | YES |
| circular loop arrows | feedback connectors | repeated delegate → observe → revise steps | one loop = one manager episode | YES |
| binary reward rows | labeled rows | the final-answer binary outcome | one row = one terminal outcome (correct or incorrect) | YES |
