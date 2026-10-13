from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
AXIS = 2
MARKER = 3
RECT_COLORS = (1, 4, 5, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_axis(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    candidates: list[tuple[str, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            candidates.append(("row", r, color))
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            candidates.append(("col", c, color))
    assert len(candidates) == 1, "expected exactly one full row/column reflection axis"
    return candidates[0]


def _reflect(kind: str, index: int, r: int, c: int) -> tuple[int, int]:
    if kind == "row":
        return 2 * index - r, c
    assert kind == "col"
    return r, 2 * index - c


def _extract_rectangles(grid: np.ndarray, bg: int, axis_color: int) -> list[dict[str, int]]:
    rects: list[dict[str, int]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in {bg, axis_color, MARKER}):
        coords = np.argwhere(grid == color)
        assert len(coords) > 0, "missing rectangle color"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 4 and w >= 4, "rectangles must be at least 4x4"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert coords.shape[0] == h * w, "expected filled rectangle"
        assert np.all(patch == color), "rectangle bbox must be monochrome"
        rects.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )
    assert 3 <= len(rects) <= 5, "expected 3 to 5 rectangles"
    return rects


def _target_index(rects: list[dict[str, int]], reflected: tuple[int, int]) -> int:
    rr, cc = reflected
    hits = [
        idx
        for idx, rect in enumerate(rects)
        if rect["r0"] < rr < rect["r1"] and rect["c0"] < cc < rect["c1"]
    ]
    assert len(hits) == 1, "reflected marker must land in exactly one rectangle interior"
    return hits[0]


def _carve_rectangle(out: np.ndarray, rect: dict[str, int], bg: int, inverted: bool) -> None:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    rows, cols = np.indices(interior.shape)
    parity = (rows + cols) % 2
    interior[parity == (1 if inverted else 0)] = bg


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    kind, axis_index, axis_color = _find_axis(g, bg)
    assert axis_color == AXIS, "axis color must be red"

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one green marker"
    marker_r, marker_c = map(int, marker_coords[0])

    rects = _extract_rectangles(g, bg, axis_color)

    rr, cc = _reflect(kind, axis_index, marker_r, marker_c)
    h, w = g.shape
    assert 0 <= rr < h and 0 <= cc < w, "reflected marker must stay in bounds"
    target = _target_index(rects, (rr, cc))

    out = g.copy()
    out[marker_r, marker_c] = bg
    for idx, rect in enumerate(rects):
        _carve_rectangle(out, rect, bg, inverted=idx == target)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the lone green marker across the red axis to choose one rectangle whose checkerboard carving flips phase.

    Input structure:
    - A black background contains exactly one full red row or full red column acting as a reflection axis.
    - There are 3 to 5 solid monochrome rectangles of distinct non-black, non-red, non-green colors.
    - Exactly one green pixel lies outside all rectangles, and its reflection across the red axis lands strictly inside one rectangle.

    Rule:
    - Find the red row/column axis and reflect the green pixel across it.
    - Identify the unique rectangle whose interior contains the reflected location.
    - Erase the green pixel to black.
    - Preserve every rectangle border, but carve each interior into a checkerboard anchored at that interior's top-left cell:
      non-target rectangles erase even-parity interior cells to black, while the reflected target rectangle erases odd-parity interior cells instead.

    Key insight:
    - The green pixel is only a hidden pointer: its mirrored landing position selects which rectangle gets the opposite checkerboard phase.
    """
    return _solve(grid)


# --------------- generator (o5x) ---------------

GEN_HEIGHT_RANGE = (16, 23)
GEN_WIDTH_RANGE = (16, 23)
GEN_RECT_COUNT_RANGE = (3, 5)
GEN_RECT_SIDE_RANGE = (4, 7)
GEN_RETRY_LIMIT = 600
GEN_AXIS_MARGIN = 6
GEN_RECT_AXIS_GAP = 1
GEN_RECT_PAD = 1
GEN_MARKER_PAD = 1


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _rect_bbox(rect: dict[str, int]) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r1"], rect["c1"]


def _side_of_rect(kind: str, axis_index: int, rect: dict[str, int]) -> str:
    if kind == "row":
        return "near_low" if rect["r0"] > axis_index else "near_high"
    return "near_low" if rect["c0"] > axis_index else "near_high"


def _valid_axis_positions(length: int) -> list[int]:
    return list(range(GEN_AXIS_MARGIN, length - GEN_AXIS_MARGIN))


def _allowed_ranges(kind: str, axis_index: int, side: str, h: int, w: int, H: int, W: int) -> tuple[range, range]:
    if kind == "row":
        row_hi = axis_index - GEN_RECT_AXIS_GAP - h
        row_lo = axis_index + GEN_RECT_AXIS_GAP + 1
        rows = range(0, row_hi + 1) if side == "near_high" else range(row_lo, H - h + 1)
        cols = range(0, W - w + 1)
        return rows, cols
    col_hi = axis_index - GEN_RECT_AXIS_GAP - w
    col_lo = axis_index + GEN_RECT_AXIS_GAP + 1
    rows = range(0, H - h + 1)
    cols = range(0, col_hi + 1) if side == "near_high" else range(col_lo, W - w + 1)
    return rows, cols


def _can_place_rect(
    rect: dict[str, int],
    placed: list[dict[str, int]],
    marker_cell: tuple[int, int] | None,
    axis_kind: str,
    axis_index: int,
) -> bool:
    r0, c0, r1, c1 = _rect_bbox(rect)
    if axis_kind == "row":
        if not (r1 <= axis_index - GEN_RECT_AXIS_GAP - 1 or r0 >= axis_index + GEN_RECT_AXIS_GAP + 1):
            return False
    else:
        if not (c1 <= axis_index - GEN_RECT_AXIS_GAP - 1 or c0 >= axis_index + GEN_RECT_AXIS_GAP + 1):
            return False
    for other in placed:
        if not _boxes_clear(_rect_bbox(rect), _rect_bbox(other), GEN_RECT_PAD):
            return False
    if marker_cell is not None:
        mr, mc = marker_cell
        if r0 - GEN_MARKER_PAD <= mr <= r1 + GEN_MARKER_PAD and c0 - GEN_MARKER_PAD <= mc <= c1 + GEN_MARKER_PAD:
            return False
    return True


def _sample_target_rectangle(
    H: int,
    W: int,
    axis_kind: str,
    axis_index: int,
    side: str,
    color: int,
) -> tuple[dict[str, int], tuple[int, int], tuple[int, int]] | None:
    for _ in range(150):
        h = random.randint(*GEN_RECT_SIDE_RANGE)
        w = random.randint(*GEN_RECT_SIDE_RANGE)
        rows, cols = _allowed_ranges(axis_kind, axis_index, side, h, w, H, W)
        rows = list(rows)
        cols = list(cols)
        if not rows or not cols:
            continue
        r0 = random.choice(rows)
        c0 = random.choice(cols)
        rect = {"color": color, "r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}
        rr = random.randint(rect["r0"] + 1, rect["r1"] - 1)
        cc = random.randint(rect["c0"] + 1, rect["c1"] - 1)
        marker = _reflect(axis_kind, axis_index, rr, cc)
        mr, mc = marker
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        if axis_kind == "row" and mr == axis_index:
            continue
        if axis_kind == "col" and mc == axis_index:
            continue
        return rect, marker, (rr, cc)
    return None


def _place_other_rectangle(
    H: int,
    W: int,
    axis_kind: str,
    axis_index: int,
    side: str,
    color: int,
    placed: list[dict[str, int]],
    marker_cell: tuple[int, int],
) -> dict[str, int] | None:
    for _ in range(200):
        h = random.randint(*GEN_RECT_SIDE_RANGE)
        w = random.randint(*GEN_RECT_SIDE_RANGE)
        row_choices, col_choices = _allowed_ranges(axis_kind, axis_index, side, h, w, H, W)
        row_choices = list(row_choices)
        col_choices = list(col_choices)
        if not row_choices or not col_choices:
            continue
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for r0 in row_choices[:20]:
            for c0 in col_choices[:20]:
                rect = {"color": color, "r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}
                if _can_place_rect(rect, placed, marker_cell, axis_kind, axis_index):
                    return rect
    return None


def _render_input(
    H: int,
    W: int,
    axis_kind: str,
    axis_index: int,
    rects: list[dict[str, int]],
    marker: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    if axis_kind == "row":
        grid[axis_index, :] = AXIS
    else:
        grid[:, axis_index] = AXIS
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    grid[marker] = MARKER
    return grid


def _expected_rect_pattern(shape: tuple[int, int], color: int, bg: int, inverted: bool) -> np.ndarray:
    h, w = shape
    patch = np.full((h, w), color, dtype=int)
    interior = patch[1:-1, 1:-1]
    rows, cols = np.indices(interior.shape)
    interior[(rows + cols) % 2 == (1 if inverted else 0)] = bg
    return patch


def _validate_sample(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        if bg != BG:
            return False
        axis_kind, axis_index, axis_color = _find_axis(grid, bg)
        if axis_color != AXIS:
            return False
        marker_coords = np.argwhere(grid == MARKER)
        if marker_coords.shape != (1, 2):
            return False
        marker_r, marker_c = map(int, marker_coords[0])
        rects = _extract_rectangles(grid, bg, axis_color)
        colors = [rect["color"] for rect in rects]
        if len(set(colors)) != len(colors):
            return False
        if any(color not in RECT_COLORS for color in colors):
            return False
        for i, rect in enumerate(rects):
            if marker_r >= rect["r0"] and marker_r <= rect["r1"] and marker_c >= rect["c0"] and marker_c <= rect["c1"]:
                return False
            for other in rects[i + 1 :]:
                if not _boxes_clear(_rect_bbox(rect), _rect_bbox(other), GEN_RECT_PAD):
                    return False
        side_counts = Counter(_side_of_rect(axis_kind, axis_index, rect) for rect in rects)
        if len(side_counts) != 2:
            return False
        rr, cc = _reflect(axis_kind, axis_index, marker_r, marker_c)
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return False
        target = _target_index(rects, (rr, cc))

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if axis_kind == "row" and not np.all(out[axis_index, :] == AXIS):
            return False
        if axis_kind == "col" and not np.all(out[:, axis_index] == AXIS):
            return False
        if int(out[marker_r, marker_c]) != BG:
            return False

        normal_only = grid.copy()
        normal_only[marker_r, marker_c] = BG
        for rect in rects:
            _carve_rectangle(normal_only, rect, BG, inverted=False)
        if np.array_equal(out, normal_only):
            return False

        for idx, rect in enumerate(rects):
            patch = out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1]
            expected = _expected_rect_pattern(
                (rect["r1"] - rect["r0"] + 1, rect["c1"] - rect["c0"] + 1),
                rect["color"],
                BG,
                inverted=idx == target,
            )
            if not np.array_equal(patch, expected):
                return False
        return True
    except AssertionError:
        return False


def _build_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_HEIGHT_RANGE)
    W = random.randint(*GEN_WIDTH_RANGE)
    axis_kind = random.choice(("row", "col"))
    axis_positions = _valid_axis_positions(H if axis_kind == "row" else W)
    if not axis_positions:
        return None
    axis_index = random.choice(axis_positions)

    rect_count = random.randint(*GEN_RECT_COUNT_RANGE)
    colors = random.sample(RECT_COLORS, rect_count)
    target_side = random.choice(("near_high", "near_low"))

    target_data = _sample_target_rectangle(H, W, axis_kind, axis_index, target_side, colors[0])
    if target_data is None:
        return None
    target_rect, marker, _ = target_data

    rects = [target_rect]
    other_sides = [random.choice(("near_high", "near_low")) for _ in range(rect_count - 1)]
    if rect_count > 1 and len({target_side, *other_sides}) == 1:
        other_sides[-1] = "near_low" if target_side == "near_high" else "near_high"
    random.shuffle(other_sides)

    for color, side in zip(colors[1:], other_sides):
        rect = _place_other_rectangle(H, W, axis_kind, axis_index, side, color, rects, marker)
        if rect is None:
            return None
        rects.append(rect)

    grid = _render_input(H, W, axis_kind, axis_index, rects, marker)
    if not _validate_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid reflected-phase rectangle puzzle")
