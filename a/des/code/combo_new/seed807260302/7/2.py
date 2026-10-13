from __future__ import annotations

import math
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Ordered by area, with monotonically growing near-square dimensions.
RECT_SHAPES = (
    (2, 3),
    (3, 3),
    (3, 4),
    (4, 4),
    (4, 5),
    (5, 5),
    (5, 6),
    (6, 6),
    (6, 7),
    (7, 7),
    (7, 8),
    (8, 8),
)

COLOR_COUNT_RANGE = (3, 6)
GENERATION_RETRIES = 300


def _closest_factor_rect(area: int) -> tuple[int, int]:
    """Return the factor pair with the smallest side-length difference."""
    assert area > 0
    best_h, best_w = 1, int(area)
    best_delta = best_w - best_h
    for h in range(1, math.isqrt(int(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        delta = w - h
        if delta < best_delta:
            best_h, best_w, best_delta = h, w, delta
    return int(best_h), int(best_w)


def _orient_to_fit(height: int, width: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    """Use the canonical wide orientation unless only the transpose fits."""
    if height <= grid_h and width <= grid_w:
        return int(height), int(width)
    if width <= grid_h and height <= grid_w:
        return int(width), int(height)
    raise AssertionError("reassembled rectangle must fit inside the grid")


def _center_slice(grid_h: int, grid_w: int, rect_h: int, rect_w: int) -> tuple[slice, slice]:
    r0 = (grid_h - rect_h) // 2
    c0 = (grid_w - rect_w) // 2
    return slice(r0, r0 + rect_h), slice(c0, c0 + rect_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored pixels into centered nested solid rectangles.

    Input structure:
    - The background is black.
    - Each non-black color appears as scattered foreground fragments, not as a meaningful placed object.
    - Every foreground color has a distinct pixel count, and each count determines one near-square solid rectangle.

    Rule:
    - Count the pixels of each non-black color.
    - Convert each count to the factor pair whose side lengths are closest; use the fitting orientation for the grid.
    - Sort the reconstructed rectangles by area to define nested layers from inner to outer.
    - Draw all rectangles centered on the original grid, painting larger layers first and smaller layers last.

    Key insight:
    - The fragment positions are irrelevant: the color histogram encodes rectangle areas, and area order encodes the centered nesting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    grid_h, grid_w = map(int, g.shape)
    assert grid_h <= 30 and grid_w <= 30, "ARC grids must be at most 30x30"

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors, "expected at least one foreground color"

    seen_areas: set[int] = set()
    rectangles: list[tuple[int, int, int, int]] = []
    for color in colors:
        area = int(np.count_nonzero(g == color))
        assert area not in seen_areas, "foreground colors must have distinct counts"
        seen_areas.add(area)

        rect_h, rect_w = _closest_factor_rect(area)
        rect_h, rect_w = _orient_to_fit(rect_h, rect_w, grid_h, grid_w)
        rectangles.append((area, color, rect_h, rect_w))

    out = np.full((grid_h, grid_w), BG, dtype=int)
    for _area, color, rect_h, rect_w in sorted(rectangles, key=lambda item: item[0], reverse=True):
        rs, cs = _center_slice(grid_h, grid_w, rect_h, rect_w)
        out[rs, cs] = int(color)
    return out


def _choose_rectangles() -> list[tuple[int, int]]:
    count = random.randint(*COLOR_COUNT_RANGE)
    chosen = random.sample(list(RECT_SHAPES), count)
    chosen.sort(key=lambda shape: shape[0] * shape[1])
    return chosen


def _choose_grid_size(rectangles: list[tuple[int, int]]) -> tuple[int, int]:
    total_foreground = sum(h * w for h, w in rectangles)
    max_h = max(h for h, _w in rectangles)
    max_w = max(w for _h, w in rectangles)
    min_h = max(12, max_h + 4)
    min_w = max(12, max_w + 4)
    candidates = [
        (h, w)
        for h in range(min_h, 31)
        for w in range(min_w, 31)
        if h * w >= total_foreground * 2
    ]
    assert candidates, "rectangle set must fit in an ARC-sized grid"
    return random.choice(candidates)


def _scatter_pixels(
    grid_h: int, grid_w: int, color_rectangles: list[tuple[int, tuple[int, int]]]
) -> np.ndarray:
    total = sum(rect_h * rect_w for _color, (rect_h, rect_w) in color_rectangles)
    positions = random.sample(range(grid_h * grid_w), total)

    labels: list[int] = []
    for color, (rect_h, rect_w) in color_rectangles:
        labels.extend([int(color)] * int(rect_h * rect_w))
    random.shuffle(labels)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for pos, color in zip(positions, labels):
        grid[pos // grid_w, pos % grid_w] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rectangles = _choose_rectangles()
    grid_h, grid_w = _choose_grid_size(rectangles)
    colors = random.sample(list(COLORS), len(rectangles))

    color_rectangles = list(zip(colors, rectangles))
    random.shuffle(color_rectangles)
    grid = _scatter_pixels(grid_h, grid_w, color_rectangles)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    observed = sorted(
        (int(np.count_nonzero(grid == color)), int(color))
        for color, _rect in color_rectangles
    )
    return grid if len(observed) == len(set(observed)) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-rectangle input")
