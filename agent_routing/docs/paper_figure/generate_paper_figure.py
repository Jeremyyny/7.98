from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
DRAWIO = HERE / "agent-routing-paper-framework.drawio"
SVG = HERE / "agent-routing-paper-framework-local.svg"

W, H = 1718, 744
FONT = "Comic Sans MS"

BLACK = "#020202"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F2F2F2"
GRAY = "#C4C4C4"
RED = "#B5150B"
DARK_GREEN = "#0D2507"
BLUE_FILL = "#D6E1F5"
BLUE = "#426B9D"
PEACH = "#FCE1D6"
ORANGE = "#DBB69E"
GREEN = "#6A884E"
GREEN_FILL = "#EFF7EA"
PURPLE = "#6762E3"


@dataclass
class Node:
    id: str
    value: str
    x: float
    y: float
    w: float
    h: float
    style: str
    kind: str = "box"
    parent: str = "1"
    font_size: int = 12
    font_color: str = BLACK
    bold: bool = False
    fill: str = WHITE
    stroke: str = GRAY
    dashed: bool = False
    align: str = "center"
    radius: float = 12


@dataclass
class Edge:
    id: str
    source: str
    target: str
    value: str
    color: str = BLACK
    dashed: bool = False
    width: float = 1.5
    points: list[tuple[float, float]] = field(default_factory=list)


nodes: dict[str, Node] = {}
edges: list[Edge] = []


mxfile = ET.Element(
    "mxfile",
    {"host": "app.diagrams.net", "agent": "Codex", "version": "26.0.9", "pages": "1"},
)
diagram = ET.SubElement(mxfile, "diagram", {"id": "learning-when-to-commit", "name": "Paper Framework"})
model = ET.SubElement(
    diagram,
    "mxGraphModel",
    {
        "dx": str(W),
        "dy": str(H),
        "grid": "1",
        "gridSize": "5",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": str(W),
        "pageHeight": str(H),
        "math": "0",
        "shadow": "0",
    },
)
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})


def text_style(size: int, color: str, *, bold: bool = False, align: str = "center") -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=visible;"
        f"fontFamily={FONT};fontSize={size};fontColor={color};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
    )


def box_style(
    fill: str,
    stroke: str,
    size: int,
    *,
    bold: bool = False,
    dashed: bool = False,
    radius: int = 14,
    align: str = "center",
    shape: Optional[str] = None,
) -> str:
    shape_part = f"shape={shape};" if shape else ""
    fill_part = "" if fill == "none" else f"fillColor={fill};"
    return (
        f"{shape_part}rounded=1;arcSize={radius};whiteSpace=wrap;html=1;"
        f"{fill_part}strokeColor={stroke};strokeWidth=1.5;"
        f"dashed={1 if dashed else 0};dashPattern=10 8;"
        f"fontFamily={FONT};fontSize={size};fontColor={BLACK};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=8;"
    )


def add_node(
    node_id: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    kind: str = "box",
    fill: str = WHITE,
    stroke: str = GRAY,
    size: int = 12,
    color: str = BLACK,
    bold: bool = False,
    dashed: bool = False,
    radius: int = 14,
    align: str = "center",
    style: Optional[str] = None,
) -> None:
    if style is None:
        shape = None
        if kind == "cylinder":
            shape = "cylinder3"
        elif kind == "document":
            shape = "document"
        elif kind == "hexagon":
            shape = "hexagon"
        style = box_style(
            fill, stroke, size, bold=bold, dashed=dashed, radius=radius, align=align, shape=shape
        )
        if color != BLACK:
            style = style.replace(f"fontColor={BLACK}", f"fontColor={color}")
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": node_id, "value": value, "style": style, "vertex": "1", "parent": "1"},
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
            "as": "geometry",
        },
    )
    nodes[node_id] = Node(
        node_id, value, x, y, w, h, style, kind, "1", size, color, bold, fill, stroke,
        dashed, align, radius,
    )


def add_text(
    node_id: str,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: int = 12,
    color: str = BLACK,
    bold: bool = False,
    align: str = "center",
) -> None:
    add_node(
        node_id, value, x, y, w, h, kind="text", fill="none", stroke="none",
        size=size, color=color, bold=bold, align=align,
        style=text_style(size, color, bold=bold, align=align),
    )


def edge_style(color: str, *, dashed: bool, width: float) -> str:
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;endArrow=classic;endFill=1;"
        f"strokeColor={color};strokeWidth={width};"
        f"dashed={1 if dashed else 0};dashPattern=7 5;"
        f"fontFamily={FONT};fontSize=10;fontColor={color};labelBackgroundColor=#FFFFFF;"
    )


def add_edge(
    edge_id: str,
    source: str,
    target: str,
    value: str,
    *,
    color: str = BLACK,
    dashed: bool = False,
    width: float = 1.5,
    points: Optional[list[tuple[float, float]]] = None,
) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": edge_id,
            "value": value,
            "style": edge_style(color, dashed=dashed, width=width),
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
    edges.append(Edge(edge_id, source, target, value, color, dashed, width, points or []))


# ---------------------------------------------------------------------------
# Global contribution and left task column
# ---------------------------------------------------------------------------

add_text(
    "figure_title",
    "Learning When to Commit",
    48, 8, 305, 46,
    size=21, color=RED, bold=True, align="left",
)
add_text(
    "figure_subtitle",
    "Delegate-or-commit routing learned from binary task outcomes",
    310, 16, 1050, 28,
    size=16, color=DARK_GREEN, bold=True,
)

add_node(
    "goal_panel",
    "Research Goal\n\nA Manager decides whether to\nconsult another frozen advisor\nor COMMIT its current answer.",
    62, 62, 218, 152,
    fill="none", stroke=GRAY, size=14, dashed=True, bold=True, align="left",
)
add_text("task_heading", "MCQ Task Stream", 76, 226, 185, 28, size=17, color=RED, bold=True)
add_node(
    "task_panel",
    "Benchmarks\n\nMedQA\nLegalBench\nMMLU-Pro\nGPQA\n\nQuestion + Context",
    62, 255, 218, 294,
    fill="none", stroke=GRAY, size=14, dashed=True, bold=True, align="left",
)
add_text("choices_label", "Answer Space", 82, 459, 90, 22, size=11, color=BLACK, bold=True)
add_node(
    "answer_space", "Multiple-choice options  A–D",
    88, 488, 156, 34,
    fill=BLUE_FILL, stroke=BLUE, size=11, bold=True, radius=6,
)
add_node(
    "task_cache",
    "Normalized\nTask Pool",
    102, 572, 138, 72,
    kind="cylinder", fill=LIGHT_GRAY, stroke=BLACK, size=13, bold=True,
)
add_text(
    "hidden_gt",
    "Ground truth is hidden\nfrom the routing policy",
    82, 650, 178, 36,
    size=10, color=RED, bold=True,
)
add_edge("choices_to_cache", "answer_space", "task_cache", "normalize", color=BLUE)

# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

add_node(
    "stage1",
    "Stage 1: Build Frozen Advisors & Cold-start the Manager",
    300, 62, 1350, 310,
    fill="none", stroke=GRAY, size=20, color=DARK_GREEN, dashed=True,
    style=box_style("none", GRAY, 20, bold=True, dashed=True)
    .replace(f"fontColor={BLACK}", f"fontColor={DARK_GREEN}")
    .replace("verticalAlign=middle", "verticalAlign=top")
    .replace("spacing=8", "spacingTop=6"),
)

add_node(
    "teacher_synth",
    "Teacher Synthesis\n\nGPT / Claude / DeepSeek\ncreates specialist signals\nfrom benchmark questions",
    330, 105, 205, 118,
    fill=LIGHT_GRAY, stroke=GRAY, size=12, bold=True, align="left",
)
add_node(
    "quality_gates",
    "Quality Gates\n\nJSON parse\nPydantic schema\nchoice coverage\nsymmetric leakage audit",
    570, 105, 190, 118,
    fill=WHITE, stroke=GRAY, size=13, bold=True, dashed=True, align="left",
)
add_node(
    "advisor_data",
    "Synthetic Advisor Data",
    800, 90, 225, 172,
    fill="none", stroke=BLUE, size=15, color=RED, bold=True, dashed=True,
)
add_node(
    "data_extractor", "Extractor\nEvidence signals",
    820, 128, 185, 36,
    fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=5,
)
add_node(
    "data_reasoner", "Reasoner\nNeutral decision scaffold",
    820, 171, 185, 36,
    fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=5,
)
add_node(
    "data_verifier", "Verifier\nCurrent-draft audit",
    820, 214, 185, 36,
    fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=5,
)
add_node(
    "advisor_sft",
    "✦\nLoRA-SFT\n× 3",
    1065, 112, 120, 128,
    fill="#FDFAF5", stroke=ORANGE, size=14, color=PURPLE, bold=True, radius=18,
)
add_node(
    "frozen_toolkit",
    "Frozen Advisors (Signals Only)",
    1225, 86, 245, 176,
    fill="none", stroke=GREEN, size=13, color=DARK_GREEN, bold=True, dashed=True,
)
add_node(
    "frozen_extractor", "Extractor  →  evidence",
    1245, 126, 205, 34,
    fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=8,
)
add_node(
    "frozen_reasoner", "Reasoner  →  scaffold",
    1245, 169, 205, 34,
    fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=8,
)
add_node(
    "frozen_verifier", "Verifier  →  draft audit",
    1245, 212, 205, 34,
    fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=8,
)
add_node(
    "coldstart_demo",
    "Manager Cold-start Demonstrations\nDRAFT_ANSWER  →  delegate(tool)  →  advisor signal  →  ...  →  ANSWER",
    480, 284, 760, 62,
    fill="#FDFAF5", stroke=ORANGE, size=12, bold=True, radius=10,
)
add_node(
    "manager_sft",
    "✦  Manager_SFT\nQwen3-8B",
    1290, 286, 210, 58,
    fill=WHITE, stroke=ORANGE, size=14, color=DARK_GREEN, bold=True, radius=18,
)

add_edge("task_to_teacher", "task_panel", "teacher_synth", "sample", color=BLACK,
         points=[(290, 402), (315, 402), (315, 164)])
add_edge("teacher_to_gates", "teacher_synth", "quality_gates", "generate", color=BLACK)
add_edge("gates_to_data", "quality_gates", "data_reasoner", "accept", color=BLUE)
add_edge("data_to_sft", "data_reasoner", "advisor_sft", "train", color=BLUE)
add_edge("sft_to_toolkit", "advisor_sft", "frozen_reasoner", "freeze", color=GREEN)
add_edge("task_to_demo", "task_panel", "coldstart_demo", "questions", color=BLACK,
         points=[(285, 360), (290, 360), (290, 315), (460, 315)])
add_edge("toolkit_to_demo", "frozen_verifier", "coldstart_demo", "tool traces", color=GREEN,
         points=[(1348, 274), (1348, 268), (860, 268), (860, 280)])
add_edge("demo_to_manager", "coldstart_demo", "manager_sft", "SFT", color=ORANGE)

# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

add_node(
    "stage2",
    "Stage 2: Learn the Delegate-or-Commit Policy with GRPO",
    300, 410, 1350, 310,
    fill="none", stroke=GRAY, size=20, color=DARK_GREEN, dashed=True,
    style=box_style("none", GRAY, 20, bold=True, dashed=True)
    .replace(f"fontColor={BLACK}", f"fontColor={DARK_GREEN}")
    .replace("verticalAlign=middle", "verticalAlign=top")
    .replace("spacing=8", "spacingTop=6"),
)

add_node(
    "rollout_manager",
    "✦\nManager_SFT\nQwen3-8B\n\nrollout",
    330, 478, 112, 160,
    fill="#FDFAF5", stroke=ORANGE, size=16, color=PURPLE, bold=True, radius=18,
)
add_node(
    "rollout_trace",
    "Rollout Trace\n\nDraft₀\n→ action₀\n→ signal₀\n→ Draft₁\n→ ...\n→ COMMIT",
    475, 475, 190, 160,
    fill="none", stroke=ORANGE, size=16, color=BLACK, bold=True, dashed=True, align="left",
)
add_node(
    "routing_actions",
    "Routing Actions (≤1 each)",
    700, 447, 205, 222,
    fill="none", stroke=ORANGE, size=14, color=DARK_GREEN, bold=True, dashed=True,
)
add_node("act_extractor", "Consult Extractor", 720, 490, 165, 32,
         fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=4)
add_node("act_reasoner", "Consult Reasoner", 720, 530, 165, 32,
         fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=4)
add_node("act_verifier", "Consult Verifier", 720, 570, 165, 32,
         fill=BLUE_FILL, stroke=WHITE, size=11, bold=True, radius=4)
add_node("act_commit", "COMMIT current answer", 720, 620, 165, 32,
         fill=PEACH, stroke=ORANGE, size=11, bold=True, radius=4)

add_node(
    "runtime_advisors",
    "Frozen Advisors — Signals Only",
    940, 447, 245, 222,
    fill="none", stroke=GREEN, size=15, color=DARK_GREEN, bold=True, dashed=True,
)
add_node("sig_extractor", "Extractor\nkey evidence", 960, 490, 205, 42,
         fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=7)
add_node("sig_reasoner", "Reasoner\nchoice-aware scaffold", 960, 542, 205, 42,
         fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=7)
add_node("sig_verifier", "Verifier\naudits current_draft", 960, 594, 205, 42,
         fill=GREEN_FILL, stroke=GREEN, size=11, bold=True, radius=7)

add_node(
    "binary_reward",
    "Binary Outcome Reward",
    1215, 447, 245, 130,
    fill="none", stroke=ORANGE, size=16, color=RED, bold=True, dashed=True,
)
add_node("reward_correct", "Correct final answer  →  1", 1235, 489, 205, 34,
         fill=PEACH, stroke=WHITE, size=12, bold=True, radius=5)
add_node("reward_incorrect", "Incorrect final answer  →  0", 1235, 533, 205, 34,
         fill=LIGHT_GRAY, stroke=WHITE, size=12, bold=True, radius=5)
add_node(
    "manager_grpo",
    "✦  Manager_GRPO\nQwen3-8B",
    1225, 600, 225, 54,
    fill=GREEN_FILL, stroke=GREEN, size=14, color=DARK_GREEN, bold=True, radius=18,
)

add_node(
    "evaluation",
    "Evaluation\n\nAccuracy\nAvg. consultations\nOracle regret\nRisk–coverage\nHeld-out transfer",
    1455, 475, 165, 160,
    fill=WHITE, stroke=GREEN, size=16, color=BLACK, bold=True, align="left",
)

add_edge("task_to_rollout", "task_cache", "rollout_manager", "input", color=BLACK,
         points=[(270, 632), (310, 632), (310, 558)])
add_edge("manager_to_trace", "rollout_manager", "rollout_trace", "rollout", color=BLACK)
add_edge("trace_to_actions", "rollout_trace", "act_reasoner", "parse", color=BLACK)
add_edge("actions_to_advisors", "act_reasoner", "sig_reasoner", "delegate", color=GREEN)
add_edge("advisors_to_trace", "sig_verifier", "rollout_trace", "signal + revised draft",
         color=GREEN, points=[(1060, 666), (570, 666), (570, 640)])
add_edge("trace_to_reward", "rollout_trace", "reward_correct", "final answer", color=ORANGE,
         points=[(570, 392), (1338, 392), (1338, 443)])
add_edge("reward_to_grpo", "reward_incorrect", "manager_grpo", "binary reward", color=ORANGE)
add_edge("grpo_to_manager", "manager_grpo", "rollout_manager", "policy update", color=GREEN,
         points=[(1210, 700), (385, 700), (385, 665)])
add_edge("commit_to_eval", "act_commit", "evaluation", "COMMIT", color=BLACK,
         points=[(905, 688), (1450, 688)])
add_edge("grpo_to_eval", "manager_grpo", "evaluation", "eval", color=GREEN,
         points=[(1465, 665), (1538, 665)])

add_text(
    "contribution",
    "Binary outcome feedback teaches the policy whether to consult again or COMMIT.",
    590, 718, 820, 18,
    size=12, color=RED, bold=True,
)


# ---------------------------------------------------------------------------
# Write editable draw.io XML
# ---------------------------------------------------------------------------

ET.indent(mxfile, space="  ")
ET.ElementTree(mxfile).write(DRAWIO, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Privacy-safe local SVG mirror for visual review
# ---------------------------------------------------------------------------

def center(n: Node) -> tuple[float, float]:
    return n.x + n.w / 2, n.y + n.h / 2


def boundary(n: Node, toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = center(n)
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    tx = (n.w / 2) / abs(dx) if dx else float("inf")
    ty = (n.h / 2) / abs(dy) if dy else float("inf")
    scale = min(tx, ty)
    return cx + dx * scale, cy + dy * scale


def edge_points(e: Edge) -> list[tuple[float, float]]:
    s, t = nodes[e.source], nodes[e.target]
    if e.points:
        pts = [center(s), *e.points, center(t)]
    else:
        sx, sy = center(s)
        tx, ty = center(t)
        mx = (sx + tx) / 2
        pts = [(sx, sy), (mx, sy), (mx, ty), (tx, ty)]
    pts[0] = boundary(s, pts[1])
    pts[-1] = boundary(t, pts[-2])
    return pts


def midpoint(pts: list[tuple[float, float]]) -> tuple[float, float]:
    lengths = [abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(pts, pts[1:])]
    total = sum(lengths)
    target = total / 2
    walked = 0.0
    for ((x1, y1), (x2, y2)), length in zip(zip(pts, pts[1:]), lengths):
        if walked + length >= target and length:
            ratio = (target - walked) / length
            return x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio
        walked += length
    return pts[-1]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


svg: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
]
edge_colors = sorted({e.color for e in edges})
for i, color in enumerate(edge_colors):
    svg.append(
        f'<marker id="arrow{i}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3.5 L0,7 z" fill="{color}"/></marker>'
    )
svg.extend(["</defs>", f'<rect x="0" y="0" width="{W}" height="{H}" fill="{WHITE}"/>'])

# Large dashed containers.
for n in nodes.values():
    if not n.dashed or n.w < 200 or n.h < 120:
        continue
    svg.append(
        f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" rx="{n.radius}" '
        f'fill="{WHITE}" fill-opacity="0.22" stroke="{n.stroke}" stroke-width="1.5" '
        'stroke-dasharray="10 8"/>'
    )
    if n.value:
        lines = n.value.splitlines() or [""]
        line_h = n.font_size * 1.22
        anchor = "middle" if n.align == "center" else "start"
        tx = n.x + n.w / 2 if anchor == "middle" else n.x + 12
        for i, line in enumerate(lines):
            svg.append(
                f'<text x="{tx:.0f}" y="{n.y + 24 + i * line_h:.0f}" text-anchor="{anchor}" '
                f'font-family="{FONT}" font-size="{n.font_size}" font-weight="700" '
                f'fill="{n.font_color}">{esc(line)}</text>'
            )

# Connectors behind ordinary nodes.
for e in edges:
    pts = edge_points(e)
    pstr = " ".join(f"{x:.0f},{y:.0f}" for x, y in pts)
    marker = edge_colors.index(e.color)
    dash = ' stroke-dasharray="7 5"' if e.dashed else ""
    svg.append(
        f'<polyline points="{pstr}" fill="none" stroke="{e.color}" stroke-width="{e.width}" '
        f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow{marker})"{dash}/>'
    )
    if e.value:
        lx, ly = midpoint(pts)
        label_w = max(34, len(e.value) * 6.2)
        svg.append(
            f'<rect x="{lx - label_w / 2:.0f}" y="{ly - 10:.0f}" width="{label_w:.0f}" '
            'height="16" rx="3" fill="#FFFFFF" fill-opacity="0.94"/>'
        )
        svg.append(
            f'<text x="{lx:.0f}" y="{ly + 2:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="9.5" fill="{e.color}">{esc(e.value)}</text>'
        )

# Ordinary nodes and text.
for n in nodes.values():
    if n.dashed and n.w >= 200 and n.h >= 120:
        continue
    if n.kind != "text":
        if n.kind == "cylinder":
            svg.append(
                f'<path d="M {n.x},{n.y + 10} C {n.x},{n.y} {n.x + n.w},{n.y} {n.x + n.w},{n.y + 10} '
                f'L {n.x + n.w},{n.y + n.h - 10} C {n.x + n.w},{n.y + n.h} {n.x},{n.y + n.h} '
                f'{n.x},{n.y + n.h - 10} Z" fill="{n.fill}" stroke="{n.stroke}" stroke-width="1.5"/>'
            )
            svg.append(
                f'<ellipse cx="{n.x + n.w / 2}" cy="{n.y + 10}" rx="{n.w / 2}" ry="10" '
                f'fill="{n.fill}" stroke="{n.stroke}" stroke-width="1.5"/>'
            )
        elif n.kind == "document":
            svg.append(
                f'<path d="M {n.x},{n.y} H {n.x + n.w} V {n.y + n.h - 10} '
                f'Q {n.x + n.w * .75},{n.y + n.h + 2} {n.x + n.w * .5},{n.y + n.h - 8} '
                f'Q {n.x + n.w * .25},{n.y + n.h - 18} {n.x},{n.y + n.h - 8} Z" '
                f'fill="{n.fill}" stroke="{n.stroke}" stroke-width="1.5"/>'
            )
        else:
            dash = ' stroke-dasharray="10 8"' if n.dashed else ""
            svg.append(
                f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" rx="{n.radius}" '
                f'fill="{n.fill if n.fill != "none" else WHITE}" fill-opacity="{0.25 if n.fill == "none" else 1}" '
                f'stroke="{n.stroke}" stroke-width="1.5"{dash}/>'
            )
    lines = n.value.splitlines() or [""]
    line_h = n.font_size * 1.22
    y0 = n.y + (n.h - line_h * len(lines)) / 2 + n.font_size
    anchor = "middle" if n.align == "center" else "start"
    tx = n.x + n.w / 2 if anchor == "middle" else n.x + 10
    for i, line in enumerate(lines):
        svg.append(
            f'<text x="{tx:.0f}" y="{y0 + i * line_h:.0f}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{n.font_size}" '
            f'font-weight="{"700" if n.bold else "400"}" fill="{n.font_color}">{esc(line)}</text>'
        )

svg.append("</svg>")
SVG.write_text("\n".join(svg), encoding="utf-8")
print(DRAWIO)
print(SVG)
