from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste
import random


def _rectangle_moves(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int, int, int]]:
    marker = Color.TEAL
    marker_coords = {tuple(map(int, p)) for p in np.argwhere(g == marker)}
    assert marker_coords, "expected at least one teal control pixel"

    moves: list[tuple[int, int, int, int, int, int, int]] = []
    used_markers: set[tuple[int, int]] = set()
    colors = colors_present(g, exclude={bg, marker})

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=colors):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = region.shape

        marks = np.argwhere(region == marker)
        assert marks.shape == (1, 2), "each colored rectangle has one teal corner"
        mr, mc = map(int, marks[0])
        assert mr in (0, h - 1) and mc in (0, w - 1), "teal pixel must be a rectangle corner"

        expected = np.full((h, w), color, dtype=int)
        expected[mr, mc] = marker
        assert np.array_equal(region, expected), "teal corner must complete a filled rectangle"

        dr = -1 if mr == 0 else 1
        dc = -1 if mc == 0 else 1
        tr0, tr1 = r0 + dr, r1 + dr
        tc0, tc1 = c0 + dc, c1 + dc
        H, W = g.shape
        assert 0 <= tr0 <= tr1 < H and 0 <= tc0 <= tc1 < W, "shifted rectangle must stay in bounds"

        used_markers.add((r0 + mr, c0 + mc))
        moves.append((r0, c0, r1, c1, color, dr, dc))

    assert used_markers == marker_coords, "every teal control pixel must belong to one rectangle"
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each teal-cornered filled rectangle one cell diagonally toward its teal corner.

    Input structure:
    - Dominant background surrounds several filled rectangular colored objects.
    - Each object has exactly one teal control pixel replacing one corner cell of
      its rectangle.

    Rule:
    - Treat the teal pixel as the missing corner of its colored rectangle.
    - Erase each original rectangle footprint, including the teal pixel.
    - Repaint the completed rectangle shifted one cell diagonally in the direction
      of that teal corner.

    Key insight:
    - The control pixel's corner position gives both the missing cell to restore
      and the diagonal motion vector.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    moves = _rectangle_moves(g, bg)

    out = g.copy()
    for r0, c0, r1, c1, _color, _dr, _dc in moves:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    occupied = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1, color, dr, dc in moves:
        tr0, tr1 = r0 + dr, r1 + dr
        tc0, tc1 = c0 + dc, c1 + dc
        assert not occupied[tr0 : tr1 + 1, tc0 : tc1 + 1].any(), "shifted rectangles must not overlap"
        assert np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == bg), "target region must be background"
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = color
        occupied[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    return out

# --------------- generator (att2, o5x) ---------------

# Colors and palette.
_BG = Color.ORANGE
_TEAL = Color.TEAL
_RECT_COLORS = [c for c in Color.ALL_COLORS if c not in {_BG, _TEAL}]
# Canvas/object sampling.
_GRID_SIZES = [10, 11]
_RECT_COUNT_RANGE = (2, 4)
_RECT_SIZE_RANGE = range(2, 6)
_OCCUPANCY_PCT_RANGE = (30, 36)
_PCT_DENOMINATOR = 100
_DIMS = [(h, w, h * w) for h in _RECT_SIZE_RANGE for w in _RECT_SIZE_RANGE]
# Shift directions encode the teal-corner motion.
_SHIFT_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# Placement and retry limits.
_OUTPUT_GAP = 1
_PLACEMENT_NODE_BUDGET = 8000
_MAX_GENERATION_ATTEMPTS = 300

_BBox = tuple[int, int, int, int]


def _bbox_intersects(a: _BBox, b: _BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(b: _BBox, n: int) -> _BBox:
    gap = _OUTPUT_GAP
    return (max(0, b[0] - gap), max(0, b[1] - gap), min(n - 1, b[2] + gap), min(n - 1, b[3] + gap))


def _choose_sizes(k: int, n: int) -> list[tuple[int, int]] | None:
    min_pct, max_pct = _OCCUPANCY_PCT_RANGE
    area = n * n
    lo = (min_pct * area + _PCT_DENOMINATOR - 1) // _PCT_DENOMINATOR
    hi = (max_pct * area) // _PCT_DENOMINATOR
    sums = [{0}]
    for _ in range(k):
        sums.append({prev + area for prev in sums[-1] for _, _, area in _DIMS})

    targets = [s for s in sums[k] if lo <= s <= hi]
    if not targets:
        return None

    target = random.choice(targets)
    sizes: list[tuple[int, int]] = []
    for remaining in range(k, 0, -1):
        choices = [(h, w, area) for h, w, area in _DIMS if target - area in sums[remaining - 1]]
        h, w, area = random.choice(choices)
        sizes.append((h, w))
        target -= area

    random.shuffle(sizes)
    return sizes


def _candidate_boxes(n: int, h: int, w: int, dr: int, dc: int) -> list[tuple[_BBox, _BBox]]:
    r0_min = 0 if dr == -1 else 1
    r0_max = n - h - 1 if dr == -1 else n - h
    c0_min = 0 if dc == -1 else 1
    c0_max = n - w - 1 if dc == -1 else n - w
    boxes = []
    for r0 in range(r0_min, r0_max + 1):
        for c0 in range(c0_min, c0_max + 1):
            out_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            in_bbox = (r0 - dr, c0 - dc, r0 - dr + h - 1, c0 - dc + w - 1)
            boxes.append((out_bbox, in_bbox))
    random.shuffle(boxes)
    return boxes


def _overlaps_placed(out_bbox: _BBox, in_bbox: _BBox, placed: list[dict[str, int | _BBox] | None], n: int) -> bool:
    out_with_gap = _expand_bbox(out_bbox, n)
    for other in placed:
        if other is None:
            continue
        if _bbox_intersects(out_with_gap, other["out_bbox"]):
            return True
        if _bbox_intersects(in_bbox, other["in_bbox"]):
            return True
    return False


def _place_rectangles(rects: list[dict[str, int]], n: int) -> list[dict[str, int | _BBox]] | None:
    placed: list[dict[str, int | _BBox] | None] = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda i: rects[i]["h"] * rects[i]["w"], reverse=True)
    node_budget = _PLACEMENT_NODE_BUDGET

    def rec(pos: int) -> bool:
        nonlocal node_budget
        if pos == len(order):
            return True
        if node_budget <= 0:
            return False
        node_budget -= 1

        i = order[pos]
        rect = rects[i]
        for out_bbox, in_bbox in _candidate_boxes(n, rect["h"], rect["w"], rect["dr"], rect["dc"]):
            if _overlaps_placed(out_bbox, in_bbox, placed, n):
                continue

            placed[i] = {**rect, "out_bbox": out_bbox, "in_bbox": in_bbox}
            if rec(pos + 1):
                return True
            placed[i] = None
        return False

    if not rec(0):
        return None
    return [p for p in placed if p is not None]


def _render_pair(placed: list[dict[str, int | _BBox]], n: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((n, n), _BG, dtype=int)
    inp = np.full((n, n), _BG, dtype=int)

    for rect in placed:
        h, w = int(rect["h"]), int(rect["w"])
        color = int(rect["color"])
        dr, dc = int(rect["dr"]), int(rect["dc"])
        out_r0, out_c0, _, _ = rect["out_bbox"]
        in_r0, in_c0, _, _ = rect["in_bbox"]

        completed = np.full((h, w), color, dtype=int)
        paste(out, completed, int(out_r0), int(out_c0))

        marked = completed.copy()
        marked[0 if dr == -1 else h - 1, 0 if dc == -1 else w - 1] = _TEAL
        paste(inp, marked, int(in_r0), int(in_c0))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_GRID_SIZES)
    k = random.randint(*_RECT_COUNT_RANGE)
    sizes = _choose_sizes(k, n)
    if sizes is None:
        return None

    colors = random.sample(_RECT_COLORS, k)
    rects: list[dict[str, int]] = []
    for (h, w), color in zip(sizes, colors):
        dr, dc = random.choice(_SHIFT_DIRS)
        rects.append({"h": h, "w": w, "color": color, "dr": dr, "dc": dc})

    placed = _place_rectangles(rects, n)
    if placed is None:
        return None

    inp, expected = _render_pair(placed, n)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 470c91de input")
