from __future__ import annotations

import random
from collections import deque

import numpy as np


Rect = tuple[int, int, int, int]  # top, left, height, width
BBox = tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive


_BG = 0
_PALETTE = tuple(range(1, 10))
_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_to_bbox(rect: Rect) -> BBox:
    top, left, height, width = rect
    return top, left, top + height - 1, left + width - 1


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    top, left, height, width = rect
    grid[top : top + height, left : left + width] = color


def _overlap_bbox(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _topological_total_order(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in colors}
    for lower in colors:
        for upper in edges[lower]:
            indegree[upper] += 1

    order: list[int] = []
    available = deque(sorted(color for color in colors if indegree[color] == 0))
    while available:
        if len(available) != 1:
            raise AssertionError("expected a unique stacking order")
        color = available.popleft()
        order.append(color)
        for upper in sorted(edges[color]):
            indegree[upper] -= 1
            if indegree[upper] == 0:
                available.append(upper)

    if len(order) != len(colors):
        raise AssertionError("stacking relations must form an acyclic total order")
    return order


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    colors = sorted(int(color) for color in np.unique(g) if int(color) != _BG)
    if len(colors) < 2:
        raise AssertionError("expected at least two visible rectangles")

    rects: dict[int, BBox] = {}
    for color in colors:
        box = _bbox(g == color)
        if box is None:
            raise AssertionError("every rectangle must be at least partially visible")
        rects[color] = box

    edges = {color: set() for color in colors}
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = _overlap_bbox(rects[color_a], rects[color_b])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            overlap_view = g[r0 : r1 + 1, c0 : c1 + 1]
            winners = {int(v) for v in np.unique(overlap_view) if int(v) in (color_a, color_b)}
            if len(winners) != 1:
                raise AssertionError("each overlap must visibly reveal the higher rectangle")
            upper = winners.pop()
            lower = color_b if upper == color_a else color_a
            edges[lower].add(upper)

    order = _topological_total_order(colors, edges)  # bottom -> top
    rotated = [order[-1], *order[:-1]]

    out = np.zeros_like(g)
    for color in rotated:
        r0, c0, r1, c1 = rects[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full rectangle stacking order from visible overlaps, then rotate that order by one layer.

    Input structure:
    - The background is black.
    - Each non-black color is a distinct solid axis-aligned rectangle, possibly partly occluded by higher rectangles.
    - Every rectangle is still visible enough that the bbox of its visible pixels equals its full rectangle.

    Rule:
    - Reconstruct each full rectangle from the bbox of its visible pixels.
    - For every pair of overlapping reconstructed rectangles, inspect the overlap region: the visible color there is the higher layer.
    - Combine these pairwise relations into the unique bottom-to-top order, move the original top rectangle to the bottom, then redraw all full rectangles on an empty canvas in that rotated order.

    Key insight:
    - The geometry of each rectangle is hidden only by occlusion, so visible bboxes recover the shapes, and the overlap colors reveal the complete stacking order needed for the cyclic layer shift.
    """
    return _solve(grid)


def _render_input(shape: tuple[int, int], rects: list[Rect], colors: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect, color in zip(rects, colors):
        _paint_rect(grid, rect, color)
    return grid


def _render_output(shape: tuple[int, int], rects: list[Rect], colors: list[int]) -> np.ndarray:
    rotated = [colors[-1], *colors[:-1]]
    grid = np.zeros(shape, dtype=int)
    rect_by_color = dict(zip(colors, rects))
    for color in rotated:
        _paint_rect(grid, rect_by_color[color], color)
    return grid


def _shift_positions(rects: list[Rect]) -> tuple[list[Rect], tuple[int, int]]:
    min_r = min(top for top, _left, _h, _w in rects)
    min_c = min(left for _top, left, _h, _w in rects)
    max_r = max(top + height for top, _left, height, _w in rects)
    max_c = max(left + width for _top, left, _h, width in rects)

    extent_h = max_r - min_r
    extent_w = max_c - min_c
    if extent_h > 30 or extent_w > 30:
        raise ValueError("shape exceeds ARC canvas bounds")

    free_h = 30 - extent_h
    free_w = 30 - extent_w
    pad_top = random.randint(0, free_h)
    pad_left = random.randint(0, free_w)

    shifted = [
        (top - min_r + pad_top, left - min_c + pad_left, height, width)
        for top, left, height, width in rects
    ]
    return shifted, (extent_h + free_h, extent_w + free_w)


def _adjacent_only_overlaps(rects: list[Rect]) -> bool:
    boxes = [_rect_to_bbox(rect) for rect in rects]
    for i, box_a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            overlap = _overlap_bbox(box_a, boxes[j])
            if abs(i - j) == 1:
                if overlap is None:
                    return False
            elif overlap is not None:
                return False
    return True


def _visible_bboxes_match(grid: np.ndarray, rects: list[Rect], colors: list[int]) -> bool:
    for rect, color in zip(rects, colors):
        if _bbox(grid == color) != _rect_to_bbox(rect):
            return False
    return True


def _sample_rectangles() -> list[Rect] | None:
    count = random.randint(2, 5)
    dr_sign, dc_sign = random.choice(_DIRECTIONS)

    base_h = random.randint(5, 8)
    base_w = random.randint(5, 8)
    sizes = []
    for _ in range(count):
        height = max(4, min(9, base_h + random.randint(-1, 1)))
        width = max(4, min(9, base_w + random.randint(-1, 1)))
        sizes.append((height, width))

    rects: list[Rect] = [(0, 0, sizes[0][0], sizes[0][1])]
    for idx in range(1, count):
        prev_top, prev_left, prev_h, prev_w = rects[-1]
        height, width = sizes[idx]
        max_oh = min(3, prev_h - 1, height - 1)
        max_ow = min(3, prev_w - 1, width - 1)
        if max_oh < 2 or max_ow < 2:
            return None
        overlap_h = random.randint(2, max_oh)
        overlap_w = random.randint(2, max_ow)

        top = prev_top + (prev_h - overlap_h) if dr_sign > 0 else prev_top - (height - overlap_h)
        left = prev_left + (prev_w - overlap_w) if dc_sign > 0 else prev_left - (width - overlap_w)
        rects.append((top, left, height, width))

    if not _adjacent_only_overlaps(rects):
        return None
    try:
        shifted, _shape = _shift_positions(rects)
    except ValueError:
        return None
    return shifted


def _gen_attempt() -> np.ndarray | None:
    rects = _sample_rectangles()
    if rects is None:
        return None

    max_r = max(top + height for top, _left, height, _w in rects)
    max_c = max(left + width for _top, left, _h, width in rects)
    shape = (max_r, max_c)
    if max(shape) > 30:
        return None

    colors = random.sample(_PALETTE, len(rects))
    inp = _render_input(shape, rects, colors)
    if not _visible_bboxes_match(inp, rects, colors):
        return None

    expected = _render_output(shape, rects, colors)
    if not np.array_equal(_solve(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_ATTEMPTS} attempts")
