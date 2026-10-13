from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

GEN_RETRIES = 400
RECT_COUNT_RANGE = (3, 6)  # upper-exclusive for rng.integers
STEP_RANGE = (1, 3)  # generates 1 or 2
RECT_SIDE_MAX = 14
EXTRA_MARGIN_MAX = 8


def _scan_colors(grid: np.ndarray, bg: int = BG) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in map(int, grid.ravel(order="C")):
        if value == bg or value in seen:
            continue
        seen.add(value)
        colors.append(value)
    return colors


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected visible cells for color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _rect_key(rect: dict[str, object]) -> tuple[int, int, int, int]:
    return rect["bbox"]  # type: ignore[return-value]


def _paint_rect(grid: np.ndarray, rect: dict[str, object]) -> None:
    r0, c0, r1, c1 = rect["bbox"]  # type: ignore[misc]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])


def _topological_order(rects: list[dict[str, object]], edges: list[set[int]]) -> list[int]:
    indegree = [0] * len(rects)
    for src in range(len(rects)):
        for dst in edges[src]:
            indegree[dst] += 1

    ready = [idx for idx, deg in enumerate(indegree) if deg == 0]
    order: list[int] = []
    while ready:
        pick = min(ready, key=lambda idx: _rect_key(rects[idx]))
        ready.remove(pick)
        order.append(pick)
        for nxt in sorted(edges[pick], key=lambda idx: _rect_key(rects[idx])):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == len(rects), "expected an acyclic rectangle layer graph"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden solid rectangles and redraw the stack in reverse layer order.

    Input structure:
    - The grid has a black background and several distinct-colored solid rectangles.
    - Rectangles partially overlap, with higher rectangles occluding lower ones.
    - Every rectangle still exposes at least one visible segment on each of its four sides.

    Rule:
    - For each color, recover the full rectangle as the bounding box of that color's visible cells.
    - For every overlapping pair, inspect their shared area: the one whose color remains visible there is the upper rectangle.
    - Reconstruct the original stacking order, then repaint all full rectangles on a black canvas in the opposite order.

    Key insight:
    - The visible fragments encode both missing geometry and z-order: side fragments reveal each full box, and overlap fragments reveal which rectangle was on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = [{"color": color, "bbox": _bbox_of_color(g, color)} for color in _scan_colors(g, BG)]
    assert rects, "expected at least one non-background rectangle"

    edges: list[set[int]] = [set() for _ in rects]  # lower -> upper
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            overlap = _intersect(_rect_key(rects[i]), _rect_key(rects[j]))
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            visible_i = bool(np.any(region == int(rects[i]["color"])))
            visible_j = bool(np.any(region == int(rects[j]["color"])))
            assert not (visible_i and visible_j), "pairwise overlap should expose at most one top color"
            if visible_i == visible_j:
                continue
            lower, upper = (j, i) if visible_i else (i, j)
            edges[lower].add(upper)

    original_order = _topological_order(rects, edges)  # bottom -> top
    out = np.zeros_like(g)
    for idx in reversed(original_order):
        _paint_rect(out, rects[idx])
    return out


def _directional_offsets(count: int, step: int, sign: int) -> list[int]:
    span = count - 1
    offsets = [idx * step for idx in range(count)]
    if sign < 0:
        return [span * step - off for off in offsets]
    return offsets


def _split_margin(total: int, rng: np.random.Generator) -> tuple[int, int]:
    lead = int(rng.integers(1, total))
    return lead, total - lead


def _render_stack(shape: tuple[int, int], rects: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        _paint_rect(grid, rect)
    return grid


def _generated_layout_is_valid(grid: np.ndarray, rects: list[dict[str, object]]) -> bool:
    for rect in rects:
        color = int(rect["color"])
        if _bbox_of_color(grid, color) != _rect_key(rect):
            return False
        r0, c0, r1, c1 = _rect_key(rect)
        if not bool(np.any(grid[r0, c0 : c1 + 1] == color)):
            return False
        if not bool(np.any(grid[r1, c0 : c1 + 1] == color)):
            return False
        if not bool(np.any(grid[r0 : r1 + 1, c0] == color)):
            return False
        if not bool(np.any(grid[r0 : r1 + 1, c1] == color)):
            return False

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            overlap = _intersect(_rect_key(rects[i]), _rect_key(rects[j]))
            if overlap is None:
                return False
            r0, c0, r1, c1 = overlap
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            top_color = int(rects[j]["color"])
            low_color = int(rects[i]["color"])
            if bool(np.any(region == low_color)):
                return False
            if not bool(np.any(region == top_color)):
                return False
    return True


def _expected_output(shape: tuple[int, int], rects: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in reversed(rects):
        _paint_rect(out, rect)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    count = int(rng.integers(*RECT_COUNT_RANGE))
    span = count - 1
    step_r = int(rng.integers(*STEP_RANGE))
    step_c = int(rng.integers(*STEP_RANGE))

    rect_h_min = span * step_r + 4
    rect_w_min = span * step_c + 4
    rect_h_max = min(RECT_SIDE_MAX, 28 - span * step_r)
    rect_w_max = min(RECT_SIDE_MAX, 28 - span * step_c)
    if rect_h_min > rect_h_max or rect_w_min > rect_w_max:
        return None

    rect_h = int(rng.integers(rect_h_min, rect_h_max + 1))
    rect_w = int(rng.integers(rect_w_min, rect_w_max + 1))
    stack_h = rect_h + span * step_r
    stack_w = rect_w + span * step_c

    extra_h_max = min(EXTRA_MARGIN_MAX, 30 - stack_h)
    extra_w_max = min(EXTRA_MARGIN_MAX, 30 - stack_w)
    if extra_h_max < 2 or extra_w_max < 2:
        return None

    extra_h = int(rng.integers(2, extra_h_max + 1))
    extra_w = int(rng.integers(2, extra_w_max + 1))
    top_margin, bottom_margin = _split_margin(extra_h, rng)
    left_margin, right_margin = _split_margin(extra_w, rng)

    sign_r, sign_c = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
    row_offsets = _directional_offsets(count, step_r, sign_r)
    col_offsets = _directional_offsets(count, step_c, sign_c)
    colors = [int(c) for c in rng.choice(PALETTE, size=count, replace=False)]

    shape = (stack_h + top_margin + bottom_margin, stack_w + left_margin + right_margin)
    rects: list[dict[str, object]] = []
    for idx, color in enumerate(colors):
        r0 = top_margin + row_offsets[idx]
        c0 = left_margin + col_offsets[idx]
        rects.append(
            {
                "color": color,
                "bbox": (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
            }
        )

    grid = _render_stack(shape, rects)
    if not _generated_layout_is_valid(grid, rects):
        return None

    expected = _expected_output(shape, rects)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reverse-occlusion-stack input")
