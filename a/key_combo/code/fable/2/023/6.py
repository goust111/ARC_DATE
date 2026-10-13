from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
LEFT_FRAME = Color.TEAL
RIGHT_FRAME = Color.RED
OBSTACLE = Color.GREY
CHANNEL = Color.GREEN
SQ = 5

INNER_COLORS = [Color.BLUE, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]
INNER_TEMPLATES = [
    (),
    ((1, 1),),
    ((0, 0), (2, 2)),
    ((0, 2), (2, 0)),
    ((0, 1), (1, 0), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (2, 0), (2, 2)),
    ((0, 1), (1, 1), (2, 1)),
    ((1, 0), (1, 1), (1, 2)),
]

HEIGHT_RANGE = (16, 24)
WIDTH_RANGE = (24, 30)
SIDE_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 600
POSITION_ATTEMPTS = 220


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _rows_overlap(r0: int, r1: int) -> bool:
    return abs(int(r0) - int(r1)) < SQ


def _box(r0: int, c0: int, h: int = SQ, w: int = SQ) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _frame_color(patch: np.ndarray) -> int | None:
    if patch.shape != (SQ, SQ):
        return None
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    if int(border.min()) != int(border.max()):
        return None
    frame = int(border[0])
    if frame not in {LEFT_FRAME, RIGHT_FRAME}:
        return None
    inner = patch[1:-1, 1:-1]
    if np.any(inner == BG) or np.any(inner == OBSTACLE) or np.any(inner == CHANNEL):
        return None
    if not np.any(inner != frame):
        return None
    return frame


def _extract_squares(grid: np.ndarray) -> list[dict]:
    assert get_bg(grid) == BG, "expected black background"
    assert not np.any(grid == CHANNEL), "green is reserved for the output channel"

    squares: list[dict] = []
    components = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        patch = crop_bbox(grid, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        colors = set(map(int, np.unique(patch)))
        if h == SQ and w == SQ and int(comp["area"]) == SQ * SQ:
            frame = _frame_color(patch)
            if frame is not None:
                squares.append(
                    {
                        "frame": frame,
                        "r0": int(r0),
                        "c0": int(c0),
                        "patch": patch,
                    }
                )
                continue
        assert colors == {OBSTACLE}, "non-square components must be grey obstacle blocks"

    left_count = sum(int(sq["frame"]) == LEFT_FRAME for sq in squares)
    right_count = sum(int(sq["frame"]) == RIGHT_FRAME for sq in squares)
    assert left_count >= 1 and right_count >= 1, "expected both teal and red bordered squares"
    return squares


def _park_squares(
    squares: list[dict],
    width: int,
    *,
    reverse_left: bool = False,
    reverse_right: bool = False,
    ignore_overlap: bool = False,
) -> list[dict]:
    left = [sq for sq in squares if int(sq["frame"]) == LEFT_FRAME]
    right = [sq for sq in squares if int(sq["frame"]) == RIGHT_FRAME]

    parked: list[dict] = []
    side_specs = [
        ("left", sorted(left, key=lambda sq: int(sq["c0"]), reverse=bool(reverse_left)), 0, SQ),
        (
            "right",
            sorted(right, key=lambda sq: int(sq["c0"]), reverse=not bool(reverse_right)),
            width - SQ,
            width - 2 * SQ,
        ),
    ]

    for side, ordered, edge_col, inward_col in side_specs:
        edge_rows: list[tuple[int, int]] = []
        for sq in ordered:
            r0 = int(sq["r0"])
            r1 = r0 + SQ - 1
            overlaps_edge = any(_intervals_overlap(r0, r1, er0, er1) for er0, er1 in edge_rows)
            use_inward = (not ignore_overlap) and overlaps_edge
            dest_c = inward_col if use_inward else edge_col
            parked.append(
                {
                    "side": side,
                    "r0": r0,
                    "dest_c": int(dest_c),
                    "patch": sq["patch"],
                }
            )
            if not use_inward:
                edge_rows.append((r0, r1))

    return parked


def _row_channel_spans(parked: list[dict], height: int) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    for r in range(height):
        left = [sq for sq in parked if sq["side"] == "left" and int(sq["r0"]) <= r <= int(sq["r0"]) + SQ - 1]
        right = [sq for sq in parked if sq["side"] == "right" and int(sq["r0"]) <= r <= int(sq["r0"]) + SQ - 1]
        if not left or not right:
            continue
        left_wall = max(int(sq["dest_c"]) + SQ - 1 for sq in left)
        right_wall = min(int(sq["dest_c"]) for sq in right)
        if left_wall + 1 <= right_wall - 1:
            spans.append((int(r), int(left_wall + 1), int(right_wall - 1)))
    return spans


def _paint_channel(out: np.ndarray, parked: list[dict], *, ignore_obstacles: bool = False) -> None:
    for r, c0, c1 in _row_channel_spans(parked, out.shape[0]):
        seg = out[r, c0 : c1 + 1]
        if ignore_obstacles:
            seg[seg == BG] = CHANNEL
        elif np.all(seg == BG):
            seg[:] = CHANNEL


def _render_from_squares(
    shape: tuple[int, int],
    squares: list[dict],
    obstacle_mask: np.ndarray,
    *,
    reverse_left: bool = False,
    reverse_right: bool = False,
    ignore_overlap: bool = False,
    ignore_obstacles: bool = False,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[obstacle_mask] = OBSTACLE
    parked = _park_squares(
        squares,
        shape[1],
        reverse_left=reverse_left,
        reverse_right=reverse_right,
        ignore_overlap=ignore_overlap,
    )
    for sq in parked:
        paste(out, sq["patch"], int(sq["r0"]), int(sq["dest_c"]))
    _paint_channel(out, parked, ignore_obstacles=ignore_obstacles)
    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    reverse_left: bool = False,
    reverse_right: bool = False,
    ignore_overlap: bool = False,
    ignore_obstacles: bool = False,
) -> np.ndarray:
    squares = _extract_squares(grid)
    return _render_from_squares(
        grid.shape,
        squares,
        grid == OBSTACLE,
        reverse_left=reverse_left,
        reverse_right=reverse_right,
        ignore_overlap=ignore_overlap,
        ignore_obstacles=ignore_obstacles,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Park the teal and red bordered 5x5 squares at opposite sides, then paint green channels only across unobstructed shared rows.

    Input structure:
    - The background is black.
    - Several disconnected 5x5 bordered squares appear, with teal (8) or red (2) borders and non-grey interior colors.
    - Separate grey obstacle blocks may sit in the middle; they do not touch the bordered squares.

    Rule:
    - Move every teal-bordered square to the left parking side (col 0) and every red-bordered square to the right parking side (col W-5), preserving each square's row.
    - Process teal squares from nearest-to-left to farthest, and red squares from nearest-to-right to farthest; if a square's row span overlaps a same-side square already parked on the edge lane, place it one square-width inward instead.
    - After parking, inspect each row: if that row passes through at least one parked square on each side and every cell between the two innermost facing square edges is background, recolor that whole middle span green.
    - Leave grey obstacle cells unchanged.

    Key insight:
    - The green channel is determined only after the parking simulation: ordering and row-overlap decide which squares occupy the inner lanes, and those parked lanes define both the corridor endpoints and whether a grey obstacle blocks the row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    squares = _extract_squares(g)
    return _render_from_squares(g.shape, squares, g == OBSTACLE)


def _make_patch(frame_color: int, used_signatures: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        base = random.choice(INNER_COLORS)
        inner = np.full((SQ - 2, SQ - 2), base, dtype=int)
        if random.random() < 0.9:
            accent = random.choice([c for c in INNER_COLORS if c != base])
            for rr, cc in random.choice(INNER_TEMPLATES):
                inner[rr, cc] = accent
        if random.random() < 0.35:
            accent2 = random.choice([c for c in INNER_COLORS if c not in {base, int(inner[1, 1])}])
            rr, cc = random.choice([(0, 1), (1, 0), (1, 2), (2, 1)])
            inner[rr, cc] = accent2
        sig = tuple(map(int, inner.flatten()))
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        patch = np.full((SQ, SQ), int(frame_color), dtype=int)
        patch[1:-1, 1:-1] = inner
        return patch
    raise RuntimeError("failed to build a unique bordered-square interior")


def _assign_side_rows(count: int, height: int) -> tuple[list[int], list[int]] | None:
    starts = list(range(height - SQ + 1))
    max_inward = min(2, count - 1)
    for _ in range(120):
        n_inward = random.randint(1, max_inward)
        n_edge = count - n_inward

        random.shuffle(starts)
        edge_rows: list[int] = []
        for r0 in starts:
            if all(not _rows_overlap(r0, prev) for prev in edge_rows):
                edge_rows.append(r0)
                if len(edge_rows) == n_edge:
                    break
        if len(edge_rows) != n_edge:
            continue

        random.shuffle(starts)
        inward_rows: list[int] = []
        for r0 in starts:
            if not any(_rows_overlap(r0, edge_r) for edge_r in edge_rows):
                continue
            if any(_rows_overlap(r0, prev) for prev in inward_rows):
                continue
            inward_rows.append(r0)
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows
    return None


def _build_side_plan(
    side: str,
    frame_color: int,
    count: int,
    height: int,
    width: int,
    used_signatures: set[tuple[int, ...]],
) -> list[dict] | None:
    assigned = _assign_side_rows(count, height)
    if assigned is None:
        return None
    edge_rows, inward_rows = assigned
    edge_col = 0 if side == "left" else width - SQ
    inward_col = SQ if side == "left" else width - 2 * SQ

    plan: list[dict] = []
    for r0 in edge_rows:
        plan.append(
            {
                "side": side,
                "r0": int(r0),
                "out_c": int(edge_col),
                "is_edge": True,
                "patch": _make_patch(frame_color, used_signatures),
            }
        )
    for r0 in inward_rows:
        plan.append(
            {
                "side": side,
                "r0": int(r0),
                "out_c": int(inward_col),
                "is_edge": False,
                "patch": _make_patch(frame_color, used_signatures),
            }
        )
    return plan


def _planned_parked(left_plan: list[dict], right_plan: list[dict]) -> list[dict]:
    parked = []
    for sq in left_plan + right_plan:
        parked.append(
            {
                "side": sq["side"],
                "r0": int(sq["r0"]),
                "dest_c": int(sq["out_c"]),
                "patch": sq["patch"],
            }
        )
    return parked


def _choose_obstacles(
    shape: tuple[int, int],
    parked: list[dict],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[int, tuple[int, int]], set[int], set[int]] | None:
    height, width = shape
    row_infos: dict[int, tuple[int, int]] = {}
    safe_lo = 2 * SQ
    safe_hi = width - 2 * SQ - 1
    usable_rows: list[int] = []

    for r, c0, c1 in _row_channel_spans(parked, height):
        lo = max(c0, safe_lo)
        hi = min(c1, safe_hi)
        if lo <= hi:
            row_infos[int(r)] = (int(lo), int(hi))
            usable_rows.append(int(r))

    if len(usable_rows) < 4:
        return None

    blocked_count = random.randint(1, len(usable_rows) - 1)
    blocked_rows = set(random.sample(usable_rows, blocked_count))
    clear_rows = set(usable_rows) - blocked_rows
    if not clear_rows:
        return None

    obstacle_mask = np.zeros(shape, dtype=bool)
    obstacle_boxes: list[tuple[int, int, int, int]] = []
    for r in blocked_rows:
        lo, hi = row_infos[r]
        max_w = min(2, hi - lo + 1)
        width_here = random.randint(1, max_w)
        c0 = random.randint(lo, hi - width_here + 1)
        obstacle_mask[r, c0 : c0 + width_here] = True
        obstacle_boxes.append(_box(r, c0, 1, width_here))

    return obstacle_mask, obstacle_boxes, row_infos, blocked_rows, clear_rows


def _derive_input_positions(
    plan: list[dict],
    width: int,
    forbidden_boxes: list[tuple[int, int, int, int]],
    *,
    ascending: bool,
) -> list[dict] | None:
    edges = [sq for sq in plan if bool(sq["is_edge"])]
    inwards = [sq for sq in plan if not bool(sq["is_edge"])]
    inward_deps = [
        [idx for idx, edge_sq in enumerate(edges) if _rows_overlap(int(sq["r0"]), int(edge_sq["r0"]))]
        for sq in inwards
    ]
    parking_cols = {0, SQ, width - 2 * SQ, width - SQ}
    candidate_cols = [c for c in range(width - SQ + 1) if c not in parking_cols]
    if len(candidate_cols) < len(plan):
        return None

    for _ in range(POSITION_ATTEMPTS):
        edge_order = list(range(len(edges)))
        inward_order = list(range(len(inwards)))
        random.shuffle(edge_order)
        random.shuffle(inward_order)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_order]
        for inward_idx in inward_order:
            latest = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            order.insert(random.randint(latest + 1, len(order)), ("inward", inward_idx))

        ordered_cols = sorted(candidate_cols, reverse=not ascending)

        def search(
            step: int,
            start_pos: int,
            boxes: list[tuple[int, int, int, int]],
            placed: list[dict],
        ) -> list[dict] | None:
            if step == len(order):
                return placed
            kind, idx = order[step]
            sq = edges[idx] if kind == "edge" else inwards[idx]
            positions = list(range(start_pos, len(ordered_cols)))
            random.shuffle(positions)
            for pos in positions:
                col = ordered_cols[pos]
                box = _box(int(sq["r0"]), int(col))
                if any(_boxes_overlap(box, other, pad=1) for other in boxes):
                    continue
                if any(_boxes_overlap(box, other, pad=1) for other in forbidden_boxes):
                    continue
                result = search(
                    step + 1,
                    pos + 1,
                    boxes + [box],
                    placed
                    + [
                        {
                            "side": sq["side"],
                            "r0": int(sq["r0"]),
                            "c0": int(col),
                            "patch": sq["patch"],
                        }
                    ],
                )
                if result is not None:
                    return result
            return None

        placed = search(0, 0, [], [])
        if placed is not None:
            return placed
    return None


def _boxes_for_input(squares: list[dict]) -> list[tuple[int, int, int, int]]:
    return [_box(int(sq["r0"]), int(sq["c0"])) for sq in squares]


def _place_inputs(
    left_plan: list[dict],
    right_plan: list[dict],
    width: int,
    obstacle_boxes: list[tuple[int, int, int, int]],
) -> tuple[list[dict], list[dict]] | None:
    orders = [("left", "right"), ("right", "left")]
    random.shuffle(orders)
    for first_side, second_side in orders:
        first_plan = left_plan if first_side == "left" else right_plan
        second_plan = right_plan if second_side == "right" else left_plan
        first = _derive_input_positions(
            first_plan,
            width,
            obstacle_boxes,
            ascending=(first_side == "left"),
        )
        if first is None:
            continue
        second = _derive_input_positions(
            second_plan,
            width,
            obstacle_boxes + _boxes_for_input(first),
            ascending=(second_side == "left"),
        )
        if second is None:
            continue
        if first_side == "left":
            return first, second
        return second, first
    return None


def _render_planned_output(
    shape: tuple[int, int],
    parked: list[dict],
    obstacle_mask: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[obstacle_mask] = OBSTACLE
    for sq in parked:
        paste(out, sq["patch"], int(sq["r0"]), int(sq["dest_c"]))
    _paint_channel(out, parked)
    return out


def _green_span_matches(
    out: np.ndarray,
    row_infos: dict[int, tuple[int, int]],
    blocked_rows: set[int],
    clear_rows: set[int],
) -> bool:
    for r, (c0, c1) in row_infos.items():
        seg = out[r, c0 : c1 + 1]
        if r in clear_rows:
            if not np.all(seg == CHANNEL):
                return False
        elif r in blocked_rows:
            if np.any(seg == CHANNEL):
                return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    left_count = random.randint(*SIDE_COUNT_RANGE)
    right_count = random.randint(*SIDE_COUNT_RANGE)

    used_signatures: set[tuple[int, ...]] = set()
    left_plan = _build_side_plan("left", LEFT_FRAME, left_count, height, width, used_signatures)
    right_plan = _build_side_plan("right", RIGHT_FRAME, right_count, height, width, used_signatures)
    if left_plan is None or right_plan is None:
        return None

    parked = _planned_parked(left_plan, right_plan)
    obstacle_choice = _choose_obstacles((height, width), parked)
    if obstacle_choice is None:
        return None
    obstacle_mask, obstacle_boxes, row_infos, blocked_rows, clear_rows = obstacle_choice

    inputs = _place_inputs(left_plan, right_plan, width, obstacle_boxes)
    if inputs is None:
        return None
    left_input, right_input = inputs

    all_boxes: list[tuple[int, int, int, int]] = []
    for sq in left_input + right_input:
        box = _box(int(sq["r0"]), int(sq["c0"]))
        if any(_boxes_overlap(box, other, pad=1) for other in all_boxes):
            return None
        if any(_boxes_overlap(box, obstacle, pad=1) for obstacle in obstacle_boxes):
            return None
        all_boxes.append(box)

    inp = np.full((height, width), BG, dtype=int)
    inp[obstacle_mask] = OBSTACLE
    for sq in left_input + right_input:
        paste(inp, sq["patch"], int(sq["r0"]), int(sq["c0"]))

    expected = _render_planned_output((height, width), parked, obstacle_mask)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _green_span_matches(actual, row_infos, blocked_rows, clear_rows):
        return None
    if not np.array_equal(actual[obstacle_mask], np.full(int(obstacle_mask.sum()), OBSTACLE, dtype=int)):
        return None

    reversed_order = _solve_variant(inp, reverse_left=True, reverse_right=True)
    no_overlap = _solve_variant(inp, ignore_overlap=True)
    no_obstacle_gate = _solve_variant(inp, ignore_obstacles=True)
    if np.array_equal(actual, reversed_order):
        return None
    if np.array_equal(actual, no_overlap):
        return None
    if np.array_equal(actual, no_obstacle_gate):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
