# Combo variant of b5ca7ac4 (index 26)
# combo_spec: D10/1+D14/1
# Name: 半侧裁剪输出+四组四方向堆叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

"""
design_analysis:
  source_task:
    original_behavior:
      - two frame colors moved horizontally toward left/right edges.
      - the full 22x22 canvas remained visible after repacking.
    fused_change:
      - extend to four frame colors and four edge targets.
      - after building that full four-edge arrangement, crop away everything except one selected group.
  concretization:
    object_shape:
      - every meaningful object is still a 5x5 framed square with a uniform 3x3 interior.
    frame_to_edge:
      8: left
      2: right
      3: top
      4: bottom
    preserved_axis:
      - left/right groups keep input row and change column.
      - top/bottom groups keep input column and change row.
    crop_selector:
      - the only displayed group is the one whose frame color appears most often in the input.
      - generation enforces a unique maximum: one group has 3 squares, the other three groups have 2 squares each.
  behavior_trace:
    original_example:
      - if two teal squares sit at rows 10 and 12, the leftmost one reaches col 0 first.
      - the later teal square overlaps that row band, so it moves to col 5.
    fused_example:
      - keep that same left-edge logic for teal.
      - add green squares that analogously move upward by row order while preserving their columns.
      - after all four color groups are stacked on a hidden 30x30 canvas, suppose red is the only color with 3 squares.
      - the solver then discards teal/green/yellow results and crops tightly around the stacked red group only.
    difference:
      - D14 adds a second motion axis and doubles the number of independently inferred edge rules.
      - D10 then hides three of the four finished groups, so the learner must infer which group was chosen from input-visible evidence.
  complexity_calibration:
    repeated_support:
      - every sample contains all four frame colors.
      - every group has both an edge-lane square and an inward-lane square.
      - the selected group varies across samples because the unique 3-square group is randomized among the four colors.
    controlled_freedom:
      - counts use only two visible levels, 2 and 3, avoiding ambiguous large count ranges.
      - crop choice depends only on object count, not on hidden numeric ordering of colors.
  implicit_assumptions_review:
    solver_side:
      - scanning 5x5 windows assumes all relevant objects are exact framed 5x5 squares.
      - only frame colors {8,2,3,4} participate in movement.
      - each edge still has only two lanes; generator must prevent any need for a third lane.
      - selecting the cropped group by maximum count assumes there is a unique maximum.
    generator_response:
      - generation fixes counts to {3,2,2,2}, so the crop selector is always unambiguous.
      - output positions are restricted to middle bands so left/right groups never collide with top/bottom groups.
      - inward placements are constructed to overlap some edge placement on the preserved axis and are ordered later in the input.
  legality_and_layout:
    full_canvas:
      - enlarge the hidden working canvas to 30x30.
      - left/right occupy cols 0,5 and 20,25.
      - top/bottom occupy rows 0,5 and 20,25.
      - anchored positions use middle starts 10 or 15, leaving corners conflict-free.
    crop_shapes:
      - left/right crops are at most 10 columns wide with variable height.
      - top/bottom crops are at most 10 rows tall with variable width.
      - tight cropping keeps the output visually clean instead of showing empty unused bands.
  visual_observability:
    what_a_human_sees:
      - input: four clearly different frame colors, each already repeated enough to count.
      - output: a single two-lane stack whose frame color matches exactly one input group.
      - across examples, whichever color has 3 instances is the one that survives cropping.
    non_hidden_features:
      - ordering uses geometric closeness to the destination edge.
      - crop choice uses visible multiplicity, not color ids or interior-color arithmetic.
  cleanliness_and_inferability:
    - corner collisions are eliminated by using only middle anchor bands on the full canvas.
    - the crop contains only one group, so no mixed overlays remain in the final output.
    - using four fixed frame colors plus count-based selection gives enough repeated structure for 2-4 ARC examples.
"""

SQ = 5
GRID_H = 30
GRID_W = 30
LEFT = 8
RIGHT = 2
TOP = 3
BOTTOM = 4
FRAME_ORDER = (LEFT, RIGHT, TOP, BOTTOM)
INPUT_ASSIGN_RETRIES = 160
GEN_RETRIES = 1000
MIDDLE_EDGE_STARTS = (10, 15)
INWARD_OPTIONS = {
    10: (11, 12),
    15: (13, 14),
}
_FRAME_CYCLE: list[int] = []

FRAME_CONFIGS = {
    LEFT: {
        "move_axis": "c",
        "keep_axis": "r",
        "ascending": True,
        "default_primary": 0,
        "inward_primary": SQ,
    },
    RIGHT: {
        "move_axis": "c",
        "keep_axis": "r",
        "ascending": False,
        "default_primary": GRID_W - SQ,
        "inward_primary": GRID_W - 2 * SQ,
    },
    TOP: {
        "move_axis": "r",
        "keep_axis": "c",
        "ascending": True,
        "default_primary": 0,
        "inward_primary": SQ,
    },
    BOTTOM: {
        "move_axis": "r",
        "keep_axis": "c",
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


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


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


def _detect_squares(grid: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    H, W = grid.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(grid, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_color = int(border[0])
            if frame_color not in FRAME_CONFIGS:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_color).any()):
                continue
            squares.append({"frame": frame_color, "r0": r0, "c0": c0, "patch": patch})
    return squares


def _stack_all_groups(
    squares: list[dict[str, int | np.ndarray]],
    bg: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[int, list[dict[str, int | np.ndarray]]]]:
    H, W = map(int, shape)
    out = np.full((H, W), int(bg), dtype=int)
    placements: dict[int, list[dict[str, int | np.ndarray]]] = {frame: [] for frame in FRAME_ORDER}

    for frame_color in FRAME_ORDER:
        cfg = FRAME_CONFIGS[frame_color]
        move_axis = str(cfg["move_axis"])
        keep_axis = str(cfg["keep_axis"])
        sort_axis = "c0" if move_axis == "c" else "r0"
        group = [s for s in squares if int(s["frame"]) == frame_color]
        group = sorted(group, key=lambda s: int(s[sort_axis]), reverse=not bool(cfg["ascending"]))

        placed_default_spans: list[tuple[int, int]] = []
        for square in group:
            keep0 = int(square[f"{keep_axis}0"])
            keep1 = keep0 + SQ - 1
            has_overlap = any(_spans_overlap(keep0, keep1, p0, p1) for p0, p1 in placed_default_spans)
            dest_primary = int(cfg["inward_primary"] if has_overlap else cfg["default_primary"])

            if move_axis == "c":
                dest_r, dest_c = int(square["r0"]), dest_primary
            else:
                dest_r, dest_c = dest_primary, int(square["c0"])
            patch = np.array(square["patch"], dtype=int)
            paste(out, patch, dest_r, dest_c, transparent=bg)
            placements[frame_color].append({"r0": dest_r, "c0": dest_c, "patch": patch})

            if not has_overlap:
                placed_default_spans.append((keep0, keep1))

    return out, placements


def _selected_frame_from_counts(groups: dict[int, list[object]]) -> int:
    return max(FRAME_ORDER, key=lambda frame: (len(groups.get(frame, [])), -FRAME_ORDER.index(frame)))


def _crop_selected_group(
    full_output: np.ndarray,
    placements: dict[int, list[dict[str, int | np.ndarray]]],
    selected_frame: int,
) -> np.ndarray:
    group = placements[selected_frame]
    if not group:
        return full_output.copy()
    r0 = min(int(item["r0"]) for item in group)
    c0 = min(int(item["c0"]) for item in group)
    r1 = max(int(item["r0"]) + SQ - 1 for item in group)
    c1 = max(int(item["c0"]) + SQ - 1 for item in group)
    return crop_bbox(full_output, (r0, c0, r1, c1))


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, int, dict[int, list[dict[str, int | np.ndarray]]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    squares = _detect_squares(g)
    grouped = {frame: [s for s in squares if int(s["frame"]) == frame] for frame in FRAME_ORDER}
    full_output, placements = _stack_all_groups(squares, bg, g.shape)
    selected_frame = _selected_frame_from_counts(grouped)
    cropped = _crop_selected_group(full_output, placements, selected_frame)
    return cropped, selected_frame, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the four-edge stacking result, then reveal only the busiest color group.

    Input structure:
    - bg is the dominant color on a 30x30 canvas.
    - Four frame colors appear: teal/8, red/2, green/3, yellow/4.
    - Every object is a separated 5x5 framed square with a non-frame 3x3 interior.

    Rule:
    - Teal stacks toward the left edge, red toward the right edge, green toward the top edge, yellow toward the bottom edge.
    - Left/right groups preserve row and sort by nearness to their target side; top/bottom preserve column and sort by nearness to their target side.
    - Within each color, a square uses the edge lane unless its preserved-axis span overlaps an already placed edge-lane square of the same color, in which case it uses the second lane one square inward.
    - After all four groups are stacked on the full canvas, keep only the group whose frame color occurs most often in the input and crop tightly around that group.

    Key insight:
    - There are two coupled inferences: color decides which edge and motion axis, while visible group count decides which finished stack survives the crop.
    - The inward lane is still explained purely by overlap with earlier same-color edge placements.
    """
    cropped, _selected_frame, _placements = _solve_with_metadata(grid)
    return cropped


def _assign_group_fixed_positions(n: int) -> tuple[list[int], list[int]]:
    if n == 3:
        edge_fixed = list(MIDDLE_EDGE_STARTS)
    elif n == 2:
        edge_fixed = [random.choice(MIDDLE_EDGE_STARTS)]
    else:
        raise ValueError(f"unsupported group size {n}")

    inward_anchor = random.choice(edge_fixed)
    inward_fixed = [random.choice(INWARD_OPTIONS[inward_anchor])]
    return edge_fixed, inward_fixed


def _next_selected_frame() -> int:
    global _FRAME_CYCLE
    if not _FRAME_CYCLE:
        _FRAME_CYCLE = list(FRAME_ORDER)
        random.shuffle(_FRAME_CYCLE)
    return int(_FRAME_CYCLE.pop())


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


def _gen_attempt(selected_frame: int) -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in FRAME_ORDER])
    group_counts = {frame_color: 2 for frame_color in FRAME_ORDER}
    group_counts[selected_frame] = 3

    full_output = np.full((GRID_H, GRID_W), bg, dtype=int)
    output_groups: dict[int, list[dict[str, int | bool | np.ndarray]]] = {}

    for frame_color in FRAME_ORDER:
        n = group_counts[frame_color]
        edge_fixed, inward_fixed = _assign_group_fixed_positions(n)
        avail_inner = [c for c in range(10) if c not in {bg, frame_color}]
        if len(avail_inner) < n:
            return None
        inner_colors = random.sample(avail_inner, n)
        output_groups[frame_color] = _place_group_output(full_output, frame_color, edge_fixed, inward_fixed, inner_colors)

    cropped_output = _crop_selected_group(
        full_output,
        {
            frame: [
                {
                    "r0": int(item["fixed"]) if FRAME_CONFIGS[frame]["move_axis"] == "c" else int(item["dest_primary"]),
                    "c0": int(item["dest_primary"]) if FRAME_CONFIGS[frame]["move_axis"] == "c" else int(item["fixed"]),
                    "patch": np.array(item["patch"], dtype=int),
                }
                for item in items
            ]
            for frame, items in output_groups.items()
        },
        selected_frame,
    )

    input_positions: list[tuple[int, int, np.ndarray]] = []
    for frame_color in FRAME_ORDER:
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
        produced, produced_frame, _placements = _solve_with_metadata(inp)
        if produced_frame != selected_frame:
            return None
        if np.array_equal(produced, cropped_output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    selected_frame = _next_selected_frame()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(selected_frame)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
