from __future__ import annotations

from itertools import permutations
import random

import numpy as np


BG = 0
AVAILABLE_COLORS = tuple(range(1, 10))
RECT_COUNT_RANGE = (3, 5)
BASE_SIZE_BY_COUNT = {
    3: (8, 13),
    4: (9, 14),
    5: (10, 15),
}
MODE_FAMILIES = ("inner", "corner", "side")
CORNER_SIDES = ("tl", "tr", "bl", "br")
EDGE_SIDES = ("top", "bottom", "left", "right")
OUTER_EXTENSION_MAX = 3
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _rect_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _paint_rect(canvas: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    canvas[r0 : r1 + 1, c0 : c1 + 1] = color


def _render(shape: tuple[int, int], layers: list[tuple[int, tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color, box in layers:
        _paint_rect(out, box, int(color))
    return out


def _colors_in_scan_order(grid: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in grid.ravel():
        color = int(value)
        if color == BG or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _candidate_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in _colors_in_scan_order(grid):
        box = _bbox(grid == color)
        r0, c0, r1, c1 = box
        assert np.any(grid[r0, c0 : c1 + 1] == color)
        assert np.any(grid[r1, c0 : c1 + 1] == color)
        assert np.any(grid[r0 : r1 + 1, c0] == color)
        assert np.any(grid[r0 : r1 + 1, c1] == color)
        rects.append((color, box))
    assert rects
    return rects


def _reverse_stacking(grid: np.ndarray) -> np.ndarray:
    rects = _candidate_rectangles(grid)
    matches: list[tuple[tuple[int, tuple[int, int, int, int]], ...]] = []
    for perm in permutations(rects):
        if np.array_equal(_render(grid.shape, list(perm)), grid):
            matches.append(perm)
    assert matches, "could not recover a valid rectangle stacking order"

    outputs = [_render(grid.shape, list(reversed(perm))) for perm in matches]
    first = outputs[0]
    for other in outputs[1:]:
        assert np.array_equal(first, other), "ambiguous order produced different reversed scenes"
    return first


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the stacking order of overlapping solid rectangles after reconstructing their full bounding boxes.

    Input structure:
    - The background is black.
    - Each non-black color belongs to one solid axis-aligned rectangle, but lower rectangles can be partially hidden by higher ones.
    - The remaining visible pixels of a hidden rectangle still expose its extreme rows and columns.

    Rule:
    - For each visible color, infer the full rectangle as the bounding box of all pixels of that color.
    - Find a bottom-to-top painting order of these full rectangles whose forward rendering exactly reproduces the input.
    - Repaint the same rectangles on a black canvas in the reversed order.

    Key insight:
    - The hidden parts do not destroy the rectangle's outer extent, so the visible fragment still determines the full box; once all boxes are known, the true stacking order is the order whose rendering matches the observed occlusions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert np.any(g == BG), "expected black background"
    return _reverse_stacking(g)


def _sample_inner(rng: random.Random, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    h, w = _rect_h(box), _rect_w(box)
    if h < 5 or w < 5:
        return None
    for _ in range(40):
        nh = rng.randint(3, h - 2)
        nw = rng.randint(3, w - 2)
        nr0 = rng.randint(r0 + 1, r1 - nh + 1)
        nc0 = rng.randint(c0 + 1, c1 - nw + 1)
        nr1 = nr0 + nh - 1
        nc1 = nc0 + nw - 1
        if nr1 <= r1 - 1 and nc1 <= c1 - 1:
            return int(nr0), int(nc0), int(nr1), int(nc1)
    return None


def _sample_corner(rng: random.Random, box: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    h, w = _rect_h(box), _rect_w(box)
    if h < 4 or w < 4:
        return None
    if side == "br":
        nr0 = rng.randint(r0 + 1, r1 - 2)
        nc0 = rng.randint(c0 + 1, c1 - 2)
        nr1 = rng.randint(r1, r1 + OUTER_EXTENSION_MAX)
        nc1 = rng.randint(c1, c1 + OUTER_EXTENSION_MAX)
        return int(nr0), int(nc0), int(nr1), int(nc1)
    if side == "tr":
        nr0 = rng.randint(r0 - OUTER_EXTENSION_MAX, r0)
        nc0 = rng.randint(c0 + 1, c1 - 2)
        nr1 = rng.randint(r0 + 2, r1 - 1)
        nc1 = rng.randint(c1, c1 + OUTER_EXTENSION_MAX)
        return int(nr0), int(nc0), int(nr1), int(nc1)
    if side == "bl":
        nr0 = rng.randint(r0 + 1, r1 - 2)
        nc0 = rng.randint(c0 - OUTER_EXTENSION_MAX, c0)
        nr1 = rng.randint(r1, r1 + OUTER_EXTENSION_MAX)
        nc1 = rng.randint(c0 + 2, c1 - 1)
        return int(nr0), int(nc0), int(nr1), int(nc1)
    if side == "tl":
        nr0 = rng.randint(r0 - OUTER_EXTENSION_MAX, r0)
        nc0 = rng.randint(c0 - OUTER_EXTENSION_MAX, c0)
        nr1 = rng.randint(r0 + 2, r1 - 1)
        nc1 = rng.randint(c0 + 2, c1 - 1)
        return int(nr0), int(nc0), int(nr1), int(nc1)
    raise ValueError(side)


def _sample_side(rng: random.Random, box: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    h, w = _rect_h(box), _rect_w(box)
    if h < 4 or w < 4:
        return None
    if side in {"top", "bottom"}:
        nw = rng.randint(2, w - 2)
        nc0 = rng.randint(c0 + 1, c1 - nw)
        nc1 = nc0 + nw - 1
        if side == "top":
            nr0 = rng.randint(r0 - OUTER_EXTENSION_MAX, r0)
            nr1 = rng.randint(r0 + 1, r1 - 1)
        else:
            nr0 = rng.randint(r0 + 1, r1 - 1)
            nr1 = rng.randint(r1, r1 + OUTER_EXTENSION_MAX)
        return int(nr0), int(nc0), int(nr1), int(nc1)

    nh = rng.randint(2, h - 2)
    nr0 = rng.randint(r0 + 1, r1 - nh)
    nr1 = nr0 + nh - 1
    if side == "left":
        nc0 = rng.randint(c0 - OUTER_EXTENSION_MAX, c0)
        nc1 = rng.randint(c0 + 1, c1 - 1)
    elif side == "right":
        nc0 = rng.randint(c0 + 1, c1 - 1)
        nc1 = rng.randint(c1, c1 + OUTER_EXTENSION_MAX)
    else:
        raise ValueError(side)
    return int(nr0), int(nc0), int(nr1), int(nc1)


def _sample_child(rng: random.Random, parent: tuple[int, int, int, int], family: str) -> tuple[int, int, int, int] | None:
    if family == "inner":
        return _sample_inner(rng, parent)
    if family == "corner":
        return _sample_corner(rng, parent, rng.choice(CORNER_SIDES))
    if family == "side":
        return _sample_side(rng, parent, rng.choice(EDGE_SIDES))
    raise ValueError(family)


def _shift_rect(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                cell = (int(nr), int(nc))
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(mask.sum())


def _scene_valid(
    rects: list[tuple[int, tuple[int, int, int, int]]],
    shape: tuple[int, int],
    expected_output: np.ndarray,
) -> np.ndarray | None:
    grid = _render(shape, rects)
    occluded = 0
    for color, box in rects:
        r0, c0, r1, c1 = box
        visible = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        if not np.any(visible):
            return None
        if visible.sum() < _rect_h(box) * _rect_w(box):
            occluded += 1
        if not np.any(grid[r0, c0 : c1 + 1] == color):
            return None
        if not np.any(grid[r1, c0 : c1 + 1] == color):
            return None
        if not np.any(grid[r0 : r1 + 1, c0] == color):
            return None
        if not np.any(grid[r0 : r1 + 1, c1] == color):
            return None
        if _bbox(grid == color) != box:
            return None
        if not _connected(visible):
            return None

    if occluded < 2:
        return None

    try:
        solved = _reverse_stacking(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected_output):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*RECT_COUNT_RANGE)
    size_lo, size_hi = BASE_SIZE_BY_COUNT[count]
    base_h = rng.randint(size_lo, size_hi)
    base_w = rng.randint(size_lo, size_hi)

    local_boxes = [(0, 0, base_h - 1, base_w - 1)]
    families: list[str] = []
    for _ in range(count - 1):
        family = rng.choice(MODE_FAMILIES)
        child = _sample_child(rng, local_boxes[-1], family)
        if child is None:
            return None
        families.append(family)
        local_boxes.append(child)

    if all(family == "inner" for family in families):
        return None

    min_r = min(box[0] for box in local_boxes)
    min_c = min(box[1] for box in local_boxes)
    max_r = max(box[2] for box in local_boxes)
    max_c = max(box[3] for box in local_boxes)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h > 28 or span_w > 28:
        return None

    extra_h = 30 - span_h
    extra_w = 30 - span_w
    top_margin = rng.randint(0, min(3, extra_h))
    left_margin = rng.randint(0, min(3, extra_w))
    bottom_margin = rng.randint(0, min(3, extra_h - top_margin))
    right_margin = rng.randint(0, min(3, extra_w - left_margin))

    shape = (span_h + top_margin + bottom_margin, span_w + left_margin + right_margin)
    dr = top_margin - min_r
    dc = left_margin - min_c
    shifted_boxes = [_shift_rect(box, dr, dc) for box in local_boxes]

    colors = rng.sample(list(AVAILABLE_COLORS), count)
    rects = list(zip(colors, shifted_boxes))
    expected = _render(shape, list(reversed(rects)))
    return _scene_valid(rects, shape, expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-rectangle scene")
