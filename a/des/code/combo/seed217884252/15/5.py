from __future__ import annotations

import random

import numpy as np


BG = 7
PALETTE = (0, 1, 2, 3, 4, 5, 6, 8, 9)
GRID_MIN = 14
GRID_MAX = 22
RECT_COUNT_CHOICES = (2, 3, 4)
SIDE_MIN = 2
SIDE_MAX = 10
OUTPUT_MAX = 30
GENERATION_ATTEMPTS = 500
CHAIN_RECT_ATTEMPTS = 80

BBox = tuple[int, int, int, int]


def _bbox_from_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_shape(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _intersect(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _covers(box: BBox, r: int, c: int) -> bool:
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _render_scene(shape: tuple[int, int], rects: list[BBox], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color, box in zip(colors, rects):
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _reconstruct_bboxes(grid: np.ndarray) -> dict[int, BBox]:
    boxes: dict[int, BBox] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        boxes[color] = _bbox_from_mask(grid == color)
    assert boxes, "expected at least one foreground rectangle"
    return boxes


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, higher in edges:
        assert lower != higher, "self-ordering is invalid"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must define a unique layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _infer_layer_order(grid: np.ndarray, boxes: dict[int, BBox]) -> list[int]:
    h, w = grid.shape
    colors = sorted(boxes)
    edges: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if _covers(boxes[color], r, c)]
            if not covering:
                assert int(grid[r, c]) == BG, "background cell cannot be inside any reconstructed rectangle"
                continue

            top = int(grid[r, c])
            assert top in covering, "visible color must be one of the rectangles covering that cell"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    if len(colors) == 1:
        return colors
    assert edges, "multiple rectangles must generate visible overlap constraints"
    return _unique_topological_order(colors, edges)


def _pack_rectangles(order: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    heights = []
    widths = []
    for color in order:
        h, w = _bbox_shape(boxes[color])
        heights.append(h)
        widths.append(w)

    out_h = max(heights)
    out_w = sum(widths) + len(widths) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for color, h, w in zip(order, heights, widths):
        out[:h, c : c + w] = int(color)
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose stacked solid rectangles, recover their complete shapes, and list them left-to-right by layer.

    The orange background is fixed. Every non-background color is one solid rectangle whose visible pixels may be
    partially occluded by higher rectangles. Each rectangle's complete bbox is reconstructed from the visible extent
    of its color, and overlap cells reveal which color is above which. The output is a fresh orange canvas where the
    rectangles are redrawn in bottom-to-top order with one background column between neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    boxes = _reconstruct_bboxes(g)
    order = _infer_layer_order(g, boxes)
    return _pack_rectangles(order, boxes)


def _random_rect(h: int, w: int, min_h: int, min_w: int, max_h: int, max_w: int) -> BBox | None:
    max_h = min(max_h, h - 2)
    max_w = min(max_w, w - 2)
    if max_h < min_h or max_w < min_w:
        return None
    rect_h = random.randint(min_h, max_h)
    rect_w = random.randint(min_w, max_w)
    r0 = random.randint(1, h - rect_h - 1)
    c0 = random.randint(1, w - rect_w - 1)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _sample_nested_rects(shape: tuple[int, int], n_rects: int) -> list[BBox] | None:
    h, w = shape
    min_bottom_h = SIDE_MIN + 2 * (n_rects - 1)
    min_bottom_w = SIDE_MIN + 2 * (n_rects - 1)
    bottom = _random_rect(h, w, min_bottom_h, min_bottom_w, SIDE_MAX + 2 * (n_rects - 1), SIDE_MAX + 2 * (n_rects - 1))
    if bottom is None:
        return None

    rects = [bottom]
    for i in range(1, n_rects):
        prev = rects[-1]
        prev_h, prev_w = _bbox_shape(prev)
        remaining = n_rects - i - 1
        min_h = SIDE_MIN + 2 * remaining
        min_w = SIDE_MIN + 2 * remaining
        if prev_h < min_h + 2 or prev_w < min_w + 2:
            return None

        next_h = random.randint(min_h, prev_h - 2)
        next_w = random.randint(min_w, prev_w - 2)
        top_shrink = random.randint(1, prev_h - next_h - 1)
        left_shrink = random.randint(1, prev_w - next_w - 1)
        bottom_shrink = prev_h - next_h - top_shrink
        right_shrink = prev_w - next_w - left_shrink
        pr0, pc0, pr1, pc1 = prev
        rects.append(
            (
                pr0 + top_shrink,
                pc0 + left_shrink,
                pr1 - bottom_shrink,
                pc1 - right_shrink,
            )
        )
    return rects


def _axis_start_range(anchor: int, size: int, limit: int, prev_lo: int, prev_hi: int, mode: int) -> tuple[int, int] | None:
    lo = max(0, anchor - size + 1)
    hi = min(anchor, limit - size)
    if mode < 0:
        hi = min(hi, prev_lo - 1)
    elif mode > 0:
        lo = max(lo, prev_hi - size + 2)
    if lo > hi:
        return None
    return lo, hi


def _sample_chain_successor(prev: BBox, shape: tuple[int, int]) -> BBox | None:
    h, w = shape
    pr0, pc0, pr1, pc1 = prev
    vertical_modes = (-1, 0, 1)
    horizontal_modes = (-1, 0, 1)

    for _ in range(CHAIN_RECT_ATTEMPTS):
        rect_h = random.randint(SIDE_MIN, min(SIDE_MAX, h - 2))
        rect_w = random.randint(SIDE_MIN, min(SIDE_MAX, w - 2))
        vr = random.choice(vertical_modes)
        hc = random.choice(horizontal_modes)
        if vr == 0 and hc == 0:
            continue

        anchor_r_lo = pr0 + (1 if pr1 > pr0 else 0)
        anchor_r_hi = pr1 - (1 if pr1 > pr0 else 0)
        anchor_c_lo = pc0 + (1 if pc1 > pc0 else 0)
        anchor_c_hi = pc1 - (1 if pc1 > pc0 else 0)
        if anchor_r_lo > anchor_r_hi or anchor_c_lo > anchor_c_hi:
            return None

        anchor_r = random.randint(anchor_r_lo, anchor_r_hi)
        anchor_c = random.randint(anchor_c_lo, anchor_c_hi)
        row_range = _axis_start_range(anchor_r, rect_h, h, pr0, pr1, vr)
        col_range = _axis_start_range(anchor_c, rect_w, w, pc0, pc1, hc)
        if row_range is None or col_range is None:
            continue

        r0 = random.randint(row_range[0], row_range[1])
        c0 = random.randint(col_range[0], col_range[1])
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if rect == prev or _intersect(rect, prev) is None:
            continue
        return rect
    return None


def _sample_chain_rects(shape: tuple[int, int], n_rects: int) -> list[BBox] | None:
    h, w = shape
    first = _random_rect(h, w, max(4, SIDE_MIN), max(4, SIDE_MIN), SIDE_MAX, SIDE_MAX)
    if first is None:
        return None

    rects = [first]
    for _ in range(1, n_rects):
        nxt = _sample_chain_successor(rects[-1], shape)
        if nxt is None:
            return None
        rects.append(nxt)
    return rects


def _has_any_overlap(rects: list[BBox]) -> bool:
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersect(rects[i], rects[j]) is not None:
                return True
    return False


def _intended_output(rects: list[BBox], colors: list[int]) -> np.ndarray:
    boxes = {color: box for color, box in zip(colors, rects)}
    return _pack_rectangles(colors, boxes)


def _scene_is_valid(grid: np.ndarray, rects: list[BBox], colors: list[int]) -> bool:
    if not _has_any_overlap(rects):
        return False
    if np.count_nonzero(grid == BG) < max(grid.shape):
        return False

    try:
        visible_boxes = _reconstruct_bboxes(grid)
    except Exception:
        return False
    if set(visible_boxes) != set(colors):
        return False
    for color, box in zip(colors, rects):
        if visible_boxes[color] != box:
            return False

    expected = _intended_output(rects, colors)
    if expected.shape[0] > OUTPUT_MAX or expected.shape[1] > OUTPUT_MAX:
        return False

    try:
        recovered_order = _infer_layer_order(grid, visible_boxes)
    except Exception:
        return False
    if recovered_order != colors:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_rects = random.choice(RECT_COUNT_CHOICES)
    colors = random.sample(list(PALETTE), n_rects)
    mode = "nested" if random.random() < 0.5 else "chain"

    if mode == "nested":
        rects = _sample_nested_rects((h, w), n_rects)
    else:
        rects = _sample_chain_rects((h, w), n_rects)
    if rects is None:
        return None

    grid = _render_scene((h, w), rects, colors)
    if not _scene_is_valid(grid, rects, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GENERATION_ATTEMPTS} attempts")
