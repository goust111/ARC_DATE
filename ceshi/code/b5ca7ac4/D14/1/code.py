# Variant of b5ca7ac4
# Dimension: D14 — 源-目标基数关系 (meta)
# Change: 多对二——N个框架分为两组，每组按各自规则堆叠到左/右边缘（many:2 目标区域）... → 多对四——四种边框颜色分四组

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

"""
design_analysis:
  source_task:
    reference_solver: none
  concretization:
    object_shape: fixed 5x5 framed square with a uniform 3x3 interior
    frame_colors:
      8: left group
      2: right group
      3: top group
      4: bottom group
    preserved_axis:
      left_right: keep original row, move horizontally
      top_bottom: keep original column, move vertically
    processing_order:
      left: smaller input column first
      right: larger input column first
      top: smaller input row first
      bottom: larger input row first
    stacking_rule:
      default_lane: flush to the assigned edge
      inward_lane: one square-width inward from that edge
      trigger: if the current square overlaps any already-placed default-lane square on the preserved axis
  behavior_trace:
    original:
      - only teal and red existed, so the solver only needed horizontal motion.
      - overlap was tested only on row spans because rows were preserved.
    variant:
      - green and yellow add vertical motion groups, so overlap must be checked on columns for top/bottom.
      - example:
        - a green square at input row 3 and another green square at input row 12 can share the same column band.
        - the upper one reaches row 0 first.
        - when the lower one is processed later, its 5-column span overlaps that edge square, so it is diverted to row 5 instead of row 0.
      - difference: the same "edge first, overlap goes inward" logic now exists independently on two axes.
  complexity_calibration:
    repeated_support:
      - every generated example contains all four frame colors.
      - every color group contains 2 or 3 squares and always includes at least one inward placement.
    observability:
      - the edge mapping is visible because four distinct frame colors consistently target four different borders.
      - ordering never depends on hidden values such as color ids or interior colors; it depends only on visible distance-to-edge along the motion axis.
  legality_and_layout:
    required_adjustment:
      - keeping the original 22x22 grid would make four two-lane edge bands collide in the corners.
      - the canvas is enlarged to 30x30 so left/right occupy cols 0,5,20,25 while top/bottom occupy rows 0,5,20,25, leaving a 10x10 central safe band.
    output_safety:
      - left/right groups use only middle rows 10..15 for their anchored positions.
      - top/bottom groups use only middle cols 10..15 for their anchored positions.
      - this prevents cross-edge overlaps in the output.
    input_safety:
      - generator places all input squares with explicit rectangle-overlap checks.
      - output-first validation keeps only inputs whose solver output matches the constructed target exactly.
  implicit_assumptions_review:
    solver_assumptions:
      - scanning 5x5 windows assumes every meaningful object is exactly one bordered 5x5 square.
      - only the four frame colors participate in motion grouping.
      - max stacking depth stays at two lanes per edge; extra overlapping inward squares would collide, so the generator never creates them.
      - top/bottom groups require column-overlap checks instead of the old row-only checks.
    generator_guarantees:
      - no two input squares overlap.
      - each inward square overlaps at least one default-lane square on the preserved axis and is ordered after such a square.
      - each group has visible support for both its edge target and its inward fallback lane.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack four color-coded families of framed 5x5 squares onto four different edges.

    Input structure:
    - bg is the dominant color.
    - Four frame colors appear: teal/8, red/2, green/3, yellow/4.
    - Each object is a 5x5 border square with a non-frame 3x3 interior, scattered without overlap.

    Rule:
    - Teal squares slide to the left edge, red squares to the right edge, green squares to the top edge, yellow squares to the bottom edge.
    - Left/right groups keep their original row; top/bottom groups keep their original column.
    - Within each color, process the squares nearest to that target edge first.
    - A square uses the edge lane unless its preserved-axis span overlaps an already placed edge-lane square of the same color, in which case it uses the second lane one square inward.

    Key insight:
    - Each color defines both an edge destination and an ordering direction.
    - The only reason a square misses the edge lane is visible overlap with an earlier same-color square along the axis that stays fixed.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    square_size = 5
    frame_configs = {
        8: {
            "move_axis": "c",
            "keep_axis": "r",
            "ascending": True,
            "default_primary": 0,
            "inward_primary": square_size,
        },
        2: {
            "move_axis": "c",
            "keep_axis": "r",
            "ascending": False,
            "default_primary": W - square_size,
            "inward_primary": W - 2 * square_size,
        },
        3: {
            "move_axis": "r",
            "keep_axis": "c",
            "ascending": True,
            "default_primary": 0,
            "inward_primary": square_size,
        },
        4: {
            "move_axis": "r",
            "keep_axis": "c",
            "ascending": False,
            "default_primary": H - square_size,
            "inward_primary": H - 2 * square_size,
        },
    }

    def spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - square_size + 1):
        for c0 in range(W - square_size + 1):
            patch = crop_bbox(g, (r0, c0, r0 + square_size - 1, c0 + square_size - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_color = int(border[0])
            if frame_color not in frame_configs:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_color).any()):
                continue
            squares.append({"frame": frame_color, "r0": r0, "c0": c0, "patch": patch})

    for frame_color, cfg in frame_configs.items():
        move_axis = str(cfg["move_axis"])
        keep_axis = str(cfg["keep_axis"])
        sort_axis = "c0" if move_axis == "c" else "r0"
        group = [s for s in squares if int(s["frame"]) == frame_color]
        group = sorted(group, key=lambda s: int(s[sort_axis]), reverse=not bool(cfg["ascending"]))

        placed_default_spans: list[tuple[int, int]] = []
        for square in group:
            keep0 = int(square[f"{keep_axis}0"])
            keep1 = keep0 + square_size - 1
            has_overlap = any(spans_overlap(keep0, keep1, p0, p1) for p0, p1 in placed_default_spans)
            dest_primary = int(cfg["inward_primary"] if has_overlap else cfg["default_primary"])

            if move_axis == "c":
                dest_r, dest_c = int(square["r0"]), dest_primary
            else:
                dest_r, dest_c = dest_primary, int(square["c0"])
            paste(out, square["patch"], dest_r, dest_c, transparent=bg)

            if not has_overlap:
                placed_default_spans.append((keep0, keep1))

    return out


# --------------- generator (c) ---------------

GRID_H, GRID_W = 30, 30
SQ = 5
LEFT, RIGHT, TOP, BOTTOM = 8, 2, 3, 4
COUNT_RANGE = (2, 3)
INPUT_ASSIGN_RETRIES = 120
GEN_RETRIES = 800
MIDDLE_EDGE_STARTS = (10, 15)
INWARD_OPTIONS = {
    10: (11, 12),
    15: (13, 14),
}

FRAME_CONFIGS = {
    LEFT: {
        "move_axis": "c",
        "fixed_axis": "r",
        "ascending": True,
        "default_primary": 0,
        "inward_primary": SQ,
    },
    RIGHT: {
        "move_axis": "c",
        "fixed_axis": "r",
        "ascending": False,
        "default_primary": GRID_W - SQ,
        "inward_primary": GRID_W - 2 * SQ,
    },
    TOP: {
        "move_axis": "r",
        "fixed_axis": "c",
        "ascending": True,
        "default_primary": 0,
        "inward_primary": SQ,
    },
    BOTTOM: {
        "move_axis": "r",
        "fixed_axis": "c",
        "ascending": False,
        "default_primary": GRID_H - SQ,
        "inward_primary": GRID_H - 2 * SQ,
    },
}


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _span_overlap(a: int, b: int) -> bool:
    return abs(int(a) - int(b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        r1, c1, _ = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2, _ = positions[j]
            if _rects_overlap(r1, c1, r2, c2):
                return True
    return False


def _occupied_overlap(candidate: list[tuple[int, int, np.ndarray]], occupied: list[tuple[int, int, np.ndarray]]) -> bool:
    for r1, c1, _ in candidate:
        for r2, c2, _ in occupied:
            if _rects_overlap(r1, c1, r2, c2):
                return True
    return False


def _assign_group_fixed_positions(n: int) -> tuple[list[int], list[int]]:
    edge_count = n - 1
    edge_fixed = sorted(random.sample(list(MIDDLE_EDGE_STARTS), edge_count))

    inward_fixed: list[int] = []
    for _ in range(n - edge_count):
        anchor = random.choice(edge_fixed)
        inward_fixed.append(random.choice(list(INWARD_OPTIONS[anchor])))
    return edge_fixed, inward_fixed


def _place_group_output(
    output: np.ndarray,
    frame_color: int,
    edge_fixed: list[int],
    inward_fixed: list[int],
    inner_colors: list[int],
) -> list[dict[str, int | bool | np.ndarray]]:
    cfg = FRAME_CONFIGS[frame_color]
    placed: list[dict[str, int | bool | np.ndarray]] = []
    color_idx = 0

    for fixed in edge_fixed:
        patch = _make_patch(frame_color, inner_colors[color_idx])
        color_idx += 1
        if cfg["move_axis"] == "c":
            paste(output, patch, fixed, int(cfg["default_primary"]))
        else:
            paste(output, patch, int(cfg["default_primary"]), fixed)
        placed.append(
            {
                "fixed": int(fixed),
                "dest_primary": int(cfg["default_primary"]),
                "is_edge": True,
                "patch": patch,
            }
        )

    for fixed in inward_fixed:
        patch = _make_patch(frame_color, inner_colors[color_idx])
        color_idx += 1
        if cfg["move_axis"] == "c":
            paste(output, patch, fixed, int(cfg["inward_primary"]))
        else:
            paste(output, patch, int(cfg["inward_primary"]), fixed)
        placed.append(
            {
                "fixed": int(fixed),
                "dest_primary": int(cfg["inward_primary"]),
                "is_edge": False,
                "patch": patch,
            }
        )

    return placed


def _derive_input_positions(
    squares: list[dict[str, int | bool | np.ndarray]],
    frame_color: int,
    occupied: list[tuple[int, int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    cfg = FRAME_CONFIGS[frame_color]
    move_axis = str(cfg["move_axis"])
    dimension = GRID_W if move_axis == "c" else GRID_H
    primary_range = list(range(dimension - SQ + 1))

    edges = [s for s in squares if bool(s["is_edge"])]
    inwards = [s for s in squares if not bool(s["is_edge"])]
    inward_deps = [
        [i for i, edge in enumerate(edges) if _span_overlap(int(inward["fixed"]), int(edge["fixed"]))]
        for inward in inwards
    ]

    for _ in range(INPUT_ASSIGN_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
            latest_dep = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_dep + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
        chosen_primary = sorted(random.sample(primary_range, len(ordered)))
        if not bool(cfg["ascending"]):
            chosen_primary = chosen_primary[::-1]

        candidate: list[tuple[int, int, np.ndarray]] = []
        valid = True
        for idx, square in enumerate(ordered):
            input_primary = chosen_primary[idx]
            if input_primary == int(square["dest_primary"]):
                valid = False
                break

            fixed = int(square["fixed"])
            patch = np.array(square["patch"], dtype=int)
            if move_axis == "c":
                r0, c0 = fixed, input_primary
            else:
                r0, c0 = input_primary, fixed
            candidate.append((r0, c0, patch))

        if not valid:
            continue
        if _any_pairwise_overlap(candidate):
            continue
        if _occupied_overlap(candidate, occupied):
            continue
        return candidate

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {LEFT, RIGHT, TOP, BOTTOM}])
    group_counts = {frame_color: random.randint(*COUNT_RANGE) for frame_color in FRAME_CONFIGS}

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    output_groups: dict[int, list[dict[str, int | bool | np.ndarray]]] = {}

    for frame_color in (LEFT, RIGHT, TOP, BOTTOM):
        n = group_counts[frame_color]
        edge_fixed, inward_fixed = _assign_group_fixed_positions(n)
        avail_inner = [c for c in range(10) if c not in {bg, frame_color}]
        if len(avail_inner) < n:
            return None
        inner_colors = random.sample(avail_inner, n)
        output_groups[frame_color] = _place_group_output(output, frame_color, edge_fixed, inward_fixed, inner_colors)

    input_positions: list[tuple[int, int, np.ndarray]] = []
    for frame_color in (LEFT, RIGHT, TOP, BOTTOM):
        derived = _derive_input_positions(output_groups[frame_color], frame_color, input_positions)
        if derived is None:
            return None
        input_positions.extend(derived)

    if _any_pairwise_overlap(input_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r0, c0, patch in input_positions:
        paste(inp, patch, r0, c0)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
