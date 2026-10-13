from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GRID_MIN = 16
GRID_MAX = 24
MIN_SIDE = 5
MAX_SIDE = 12
MIN_RECTS = 3
MAX_RECTS = 5
GEN_RETRIES = 200
CHAIN_CASE_RETRIES = 25
SHIFT_RETRIES = 30
SHIFT_CHOICES = (-3, -2, -1, 0, 1, 2, 3)

BBox = tuple[int, int, int, int]


def _bbox_height(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_width(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _bbox_area(box: BBox) -> int:
    return _bbox_height(box) * _bbox_width(box)


def _bbox_of_color(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _visible_bboxes(grid: np.ndarray, bg: int) -> dict[int, BBox]:
    return {color: _bbox_of_color(grid, color) for color in _colors_present(grid, bg)}


def _overlap_box(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _shrink_box(box: BBox) -> BBox | None:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1))


def _render_rectangles(
    shape: tuple[int, int], bboxes: dict[int, BBox], order: tuple[int, ...], bg: int
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        _paint_rect(out, bboxes[int(color)], int(color))
    return out


def _render_shrunken(
    shape: tuple[int, int], bboxes: dict[int, BBox], order: tuple[int, ...], bg: int
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        shrunk = _shrink_box(bboxes[int(color)])
        assert shrunk is not None
        _paint_rect(out, shrunk, int(color))
    return out


def _matching_orders(grid: np.ndarray, bboxes: dict[int, BBox], bg: int) -> list[tuple[int, ...]]:
    colors = tuple(sorted(int(color) for color in bboxes))
    matches: list[tuple[int, ...]] = []
    for order in itertools.permutations(colors):
        if np.array_equal(_render_rectangles(grid.shape, bboxes, order, bg), grid):
            matches.append(tuple(int(color) for color in order))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle bboxes and layer order, then shrink each rectangle by one cell
    per side and redraw them in the same back-to-front order.

    The key observation is that each rectangle keeps a visible L/U-shaped outline, so the visible
    pixels for one color still reach that rectangle's true top, bottom, left, and right extent.
    Therefore each rectangle's full bbox is simply the bbox of its currently visible color.
    Once the bboxes are known, the rectangle count is small enough to brute-force the few possible
    layer orders and keep the orders whose full-rectangle repaint exactly reproduces the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(BG if np.any(g == BG) else np.bincount(g.ravel()).argmax())
    bboxes = _visible_bboxes(g, bg)
    assert bboxes

    matches = _matching_orders(g, bboxes, bg)
    assert matches, "visible bboxes should admit at least one consistent stacking order"

    outputs: dict[bytes, np.ndarray] = {}
    for order in matches:
        out = _render_shrunken(g.shape, bboxes, order, bg)
        outputs[out.tobytes()] = out

    assert len(outputs) == 1, "stacking order should be uniquely recoverable for this puzzle"
    return next(iter(outputs.values()))


def _overlap_graph_connected(bboxes: dict[int, BBox]) -> bool:
    colors = list(bboxes)
    if not colors:
        return False

    seen = {colors[0]}
    stack = [colors[0]]
    while stack:
        color = stack.pop()
        for other in colors:
            if other in seen:
                continue
            if _overlap_box(bboxes[color], bboxes[other]) is None:
                continue
            seen.add(other)
            stack.append(other)
    return len(seen) == len(colors)


def _split_reduction(current: int, minimum_next: int) -> tuple[int, int]:
    max_total = int(current - minimum_next)
    assert max_total >= 2
    total = random.randint(2, max_total)
    first = random.randint(1, total - 1)
    return int(first), int(total - first)


def _build_nested_case() -> tuple[np.ndarray, dict[int, BBox], tuple[int, ...]]:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)
    min_outer = MIN_SIDE + 2 * (n_rects - 1)

    height = random.randint(max(GRID_MIN, min_outer + 2), GRID_MAX)
    width = random.randint(max(GRID_MIN, min_outer + 2), GRID_MAX)

    max_outer_h = min(height - 2, MAX_SIDE + 2 * (n_rects - 1) + 2)
    max_outer_w = min(width - 2, MAX_SIDE + 2 * (n_rects - 1) + 2)
    outer_h = random.randint(min_outer, max_outer_h)
    outer_w = random.randint(min_outer, max_outer_w)
    outer_r0 = random.randint(1, height - outer_h - 1)
    outer_c0 = random.randint(1, width - outer_w - 1)

    boxes = [(outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)]
    current = boxes[0]
    for idx in range(1, n_rects):
        remain = n_rects - idx - 1
        min_h = MIN_SIDE + 2 * remain
        min_w = MIN_SIDE + 2 * remain
        top, bottom = _split_reduction(_bbox_height(current), min_h)
        left, right = _split_reduction(_bbox_width(current), min_w)
        current = (
            int(current[0] + top),
            int(current[1] + left),
            int(current[2] - bottom),
            int(current[3] - right),
        )
        boxes.append(current)

    order = tuple(int(color) for color in random.sample(COLORS, n_rects))
    bboxes = {color: box for color, box in zip(order, boxes, strict=False)}
    grid = _render_rectangles((height, width), bboxes, order, BG)
    return grid, bboxes, order


def _sample_shifted_box(anchor: BBox, height: int, width: int) -> BBox | None:
    anchor_h = _bbox_height(anchor)
    anchor_w = _bbox_width(anchor)
    min_h = max(MIN_SIDE, anchor_h - 2)
    max_h = min(MAX_SIDE, anchor_h + 2, height - 2)
    min_w = max(MIN_SIDE, anchor_w - 2)
    max_w = min(MAX_SIDE, anchor_w + 2, width - 2)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(SHIFT_RETRIES):
        box_h = random.randint(min_h, max_h)
        box_w = random.randint(min_w, max_w)
        dr = random.choice(SHIFT_CHOICES)
        dc = random.choice(SHIFT_CHOICES)
        if dr == 0 and dc == 0:
            continue

        r0 = int(anchor[0] + dr)
        c0 = int(anchor[1] + dc)
        r1 = int(r0 + box_h - 1)
        c1 = int(c0 + box_w - 1)
        if r0 < 1 or c0 < 1 or r1 > height - 2 or c1 > width - 2:
            continue

        candidate = (r0, c0, r1, c1)
        overlap = _overlap_box(anchor, candidate)
        if overlap is None:
            continue

        overlap_h = _bbox_height(overlap)
        overlap_w = _bbox_width(overlap)
        if overlap_h < 2 or overlap_w < 2:
            continue

        overlap_area = _bbox_area(overlap)
        smaller = min(_bbox_area(anchor), _bbox_area(candidate))
        if overlap_area < max(6, smaller // 5):
            continue
        if overlap_area > (4 * smaller) // 5:
            continue
        return candidate
    return None


def _build_chain_case() -> tuple[np.ndarray, dict[int, BBox], tuple[int, ...]] | None:
    for _ in range(CHAIN_CASE_RETRIES):
        n_rects = random.randint(MIN_RECTS, MAX_RECTS)
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)

        max_h = min(MAX_SIDE, height - 2)
        max_w = min(MAX_SIDE, width - 2)
        if max_h < MIN_SIDE or max_w < MIN_SIDE:
            continue

        first_h = random.randint(MIN_SIDE, max_h)
        first_w = random.randint(MIN_SIDE, max_w)
        first_r0 = random.randint(1, height - first_h - 1)
        first_c0 = random.randint(1, width - first_w - 1)
        boxes = [(first_r0, first_c0, first_r0 + first_h - 1, first_c0 + first_w - 1)]

        ok = True
        for _idx in range(1, n_rects):
            anchors = boxes[max(0, len(boxes) - 2) :]
            anchor = random.choice(anchors)
            candidate = _sample_shifted_box(anchor, height, width)
            if candidate is None:
                ok = False
                break
            boxes.append(candidate)

        if not ok:
            continue

        order = tuple(int(color) for color in random.sample(COLORS, n_rects))
        bboxes = {color: box for color, box in zip(order, boxes, strict=False)}
        grid = _render_rectangles((height, width), bboxes, order, BG)
        return grid, bboxes, order
    return None


def _valid_case(grid: np.ndarray, bboxes: dict[int, BBox], order: tuple[int, ...]) -> bool:
    if not np.any(grid == BG):
        return False
    if not _overlap_graph_connected(bboxes):
        return False

    visible = _visible_bboxes(grid, BG)
    if visible != bboxes:
        return False

    matches = _matching_orders(grid, visible, BG)
    if len(matches) != 1 or matches[0] != order:
        return False

    expected = _render_shrunken(grid.shape, bboxes, order, BG)
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        builders = (
            (_build_chain_case, _build_nested_case)
            if random.random() < 0.5
            else (_build_nested_case, _build_chain_case)
        )
        for builder in builders:
            case = builder()
            if case is None:
                continue
            grid, bboxes, order = case
            if _valid_case(grid, bboxes, order):
                return grid
    raise AssertionError("failed to generate a valid overlapping-rectangles instance")
