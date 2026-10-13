from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]

BG = 0
PALETTE = list(range(1, 10))
DIAGONALS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
RECT_SIZE_RANGE = (4, 8)
RECT_COUNT_RANGE = (2, 7)
SHIFT_MAX = 3
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> BBox | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _color_bboxes(g: np.ndarray) -> dict[int, BBox]:
    boxes: dict[int, BBox] = {}
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        box = _bbox(g == color)
        assert box is not None, "every foreground color must remain visible"
        boxes[color] = box
    assert boxes, "input must contain at least one colored rectangle"
    return boxes


def _layer_order(g: np.ndarray) -> list[int]:
    boxes = _color_bboxes(g)
    colors = list(boxes)
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ov = _overlap(boxes[a], boxes[b])
            if ov is None:
                continue
            r0, c0, r1, c1 = ov
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(sub == a))
            b_visible = bool(np.any(sub == b))
            assert not (a_visible and b_visible), "two overlapping opaque rectangles cannot both be visible there"
            if a_visible:
                lower, higher = b, a
            elif b_visible:
                lower, higher = a, b
            else:
                continue
            if higher not in above[lower]:
                above[lower].add(higher)
                indegree[higher] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "occlusion relations must determine a unique layer order"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _rings(order: list[int]) -> np.ndarray:
    n = len(order)
    size = 2 * n - 1
    center = n - 1
    rr, cc = np.indices((size, size))
    dist = np.maximum(np.abs(rr - center), np.abs(cc - center))
    out = np.full((size, size), BG, dtype=int)
    for depth, color in enumerate(order):
        radius = center - depth
        out[dist == radius] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stacking order of overlapping filled rectangles and redraw it as concentric color rings.

    Input structure:
    - The background is black.
    - The foreground consists of several different-colored filled rectangles whose visible parts may be partially occluded.
    - Each rectangle's full extent is exactly the bounding box of its visible pixels.

    Rule:
    - Reconstruct one full rectangle per color from that color's visible-pixel bounding box.
    - For every pair of reconstructed rectangles that geometrically overlap, inspect their overlap area: if only color A is visible there, then A lies above the other color.
    - Topologically sort the resulting occlusion graph from bottom layer to top layer.
    - Output a square grid of side 2N-1, where N is the number of colors: paint the bottom color on the outer ring, the next color on the next inner ring, and continue until the top color fills the center.

    Key insight:
    - The input geometry matters only as evidence for relative depth; once the global depth order is recovered from local overlap visibility, the output ignores the original layout and encodes only that layer sequence as nested rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order = _layer_order(g)
    return _rings(order)


def _sample_rectangles() -> tuple[list[int], list[BBox]]:
    count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    height = random.randint(*RECT_SIZE_RANGE)
    width = random.randint(*RECT_SIZE_RANGE)
    step_r_max = min(SHIFT_MAX, height - 2)
    step_c_max = min(SHIFT_MAX, width - 2)
    assert step_r_max >= 1 and step_c_max >= 1

    dr_sign, dc_sign = random.choice(DIAGONALS)
    rects: list[BBox] = [(0, 0, height - 1, width - 1)]
    for _ in range(1, count):
        prev = rects[-1]
        step_r = random.randint(1, step_r_max)
        step_c = random.randint(1, step_c_max)
        r0 = int(prev[0] + dr_sign * step_r)
        c0 = int(prev[1] + dc_sign * step_c)
        rects.append((r0, c0, r0 + height - 1, c0 + width - 1))
    return colors, rects


def _split_margin(extra: int) -> tuple[int, int]:
    total = random.randint(2, min(8, extra))
    before = random.randint(1, total - 1)
    after = total - before
    return before, after


def _render(colors: list[int], rects: list[BBox]) -> tuple[np.ndarray, list[BBox]]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    max_r = max(r1 for _r0, _c0, r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    max_c = max(c1 for _r0, _c0, _r1, c1 in rects)
    span_h = int(max_r - min_r + 1)
    span_w = int(max_c - min_c + 1)
    assert span_h <= 28 and span_w <= 28

    top, bottom = _split_margin(30 - span_h)
    left, right = _split_margin(30 - span_w)
    out = np.full((span_h + top + bottom, span_w + left + right), BG, dtype=int)

    shifted: list[BBox] = []
    row_shift = top - min_r
    col_shift = left - min_c
    for color, (r0, c0, r1, c1) in zip(colors, rects, strict=False):
        rr0 = int(r0 + row_shift)
        cc0 = int(c0 + col_shift)
        rr1 = int(r1 + row_shift)
        cc1 = int(c1 + col_shift)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
        shifted.append((rr0, cc0, rr1, cc1))
    return out, shifted


def _valid_example(g: np.ndarray, order: list[int], rects: list[BBox]) -> bool:
    try:
        if max(g.shape) > 30:
            return False
        for color, rect in zip(order, rects, strict=False):
            if _bbox(g == color) != rect:
                return False
        if _layer_order(g) != order:
            return False
        if not np.array_equal(main(g), _rings(order)):
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        colors, rects = _sample_rectangles()
        grid, shifted = _render(colors, rects)
        if _valid_example(grid, colors, shifted):
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
