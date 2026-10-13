from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_RETRIES = 400
REGION_RETRIES = 120


@dataclass(frozen=True)
class Region:
    top: int
    left: int
    bottom: int
    right: int
    color: int
    seed_count: int
    boundary: tuple[tuple[int, int], ...]


def _boundary_clockwise(top: int, left: int, bottom: int, right: int) -> tuple[tuple[int, int], ...]:
    pts: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        pts.append((top, c))
    for r in range(top + 1, bottom + 1):
        pts.append((r, right))
    for c in range(right - 1, left - 1, -1):
        pts.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        pts.append((r, left))
    return tuple(pts)


def _parse_region(grid: np.ndarray, top: int, left: int) -> Region:
    h, w = grid.shape
    assert int(grid[top, left]) == MAROON

    right = left
    while right + 1 < w and int(grid[top, right + 1]) == GREY:
        right += 1

    bottom = top
    while bottom + 1 < h and int(grid[bottom + 1, left]) == GREY:
        bottom += 1

    assert right - left >= 2 and bottom - top >= 2
    assert np.all(grid[top, left + 1 : right + 1] == GREY)
    assert np.all(grid[top + 1 : bottom + 1, left] == GREY)
    assert np.all(grid[bottom, left : right + 1] == GREY)
    assert np.all(grid[top : bottom + 1, right] == GREY)
    if right + 1 < w:
        assert int(grid[top, right + 1]) == BLACK
    if bottom + 1 < h:
        assert int(grid[bottom + 1, left]) == BLACK

    interior = grid[top + 1 : bottom, left + 1 : right]
    assert interior.size > 0
    assert not np.any(interior == GREY)
    assert not np.any(interior == MAROON)

    seed_mask = interior != BLACK
    seed_values = interior[seed_mask]
    assert seed_values.size > 0
    colors = np.unique(seed_values)
    assert len(colors) == 1
    color = int(colors[0])

    boundary = _boundary_clockwise(top, left, bottom, right)
    return Region(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        color=color,
        seed_count=min(int(seed_values.size), len(boundary)),
        boundary=boundary,
    )


def _find_regions(grid: np.ndarray) -> list[Region]:
    markers = sorted(map(tuple, np.argwhere(grid == MAROON)))
    assert markers
    return [_parse_region(grid, int(r), int(c)) for r, c in markers]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each enclosed seed set into a same-color clockwise border band on its rectangular wall.

    Input structure:
    - The background is black.
    - One or more 1-pixel-thick rectangular walls are drawn in grey, with each rectangle enclosing a black interior.
    - The top-left wall cell of every rectangle is replaced by a single maroon marker.
    - Inside each rectangle, several isolated 1x1 seed pixels share one color; different rectangles use different seed colors.

    Rule:
    - For each maroon-marked rectangle, count the number N of non-black seed pixels inside it.
    - Read the rectangle wall as a clockwise sequence that starts at the maroon top-left marker, then goes across the top edge, down the right edge, across the bottom edge, and up the left edge.
    - Recolor the first N wall cells in that sequence to the interior seed color, and set every remaining wall cell to grey.
    - Erase all interior seed pixels to black.

    Key insight:
    - Each region encodes two independent signals: seed count becomes border-band length, while seed color becomes the repaint color, and the maroon corner fixes the unique starting point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region in _find_regions(g):
        inner = out[region.top + 1 : region.bottom, region.left + 1 : region.right]
        inner[inner != BLACK] = BLACK

        for r, c in region.boundary:
            out[r, c] = GREY
        for r, c in region.boundary[: region.seed_count]:
            out[r, c] = region.color

    return out


def _sample_seed_count(rng: random.Random, max_seed_count: int) -> int:
    if max_seed_count == 1:
        return 1
    mode = rng.randrange(3)
    if mode == 0:
        high = max(1, max_seed_count // 3)
        return rng.randint(1, high)
    if mode == 1:
        low = max(1, max_seed_count // 3)
        high = max(low, (2 * max_seed_count) // 3)
        return rng.randint(low, high)
    low = max(1, (2 * max_seed_count) // 3)
    return rng.randint(low, max_seed_count)


def _place_region(
    grid: np.ndarray,
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    seed_count: int,
    rng: random.Random,
) -> None:
    bottom = top + height - 1
    right = left + width - 1

    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left] = MAROON

    interior_cells = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    for r, c in rng.sample(interior_cells, seed_count):
        grid[r, c] = color

    pad_top = max(0, top - 1)
    pad_left = max(0, left - 1)
    pad_bottom = min(grid.shape[0], bottom + 2)
    pad_right = min(grid.shape[1], right + 2)
    occupied[pad_top:pad_bottom, pad_left:pad_right] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 30)
    width = rng.randint(12, 30)
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    region_count = rng.randint(1, min(4, len(SEED_COLORS)))
    colors = rng.sample(SEED_COLORS, region_count)

    for color in colors:
        placed = False
        for _ in range(REGION_RETRIES):
            rect_h = rng.randint(4, min(9, height - 2))
            rect_w = rng.randint(4, min(10, width - 2))
            if height - rect_h - 1 < 1 or width - rect_w - 1 < 1:
                continue

            top = rng.randint(1, height - rect_h - 1)
            left = rng.randint(1, width - rect_w - 1)
            bottom = top + rect_h - 1
            right = left + rect_w - 1

            pad_top = top - 1
            pad_left = left - 1
            pad_bottom = bottom + 2
            pad_right = right + 2
            if occupied[pad_top:pad_bottom, pad_left:pad_right].any():
                continue

            perimeter = 2 * (rect_h + rect_w) - 4
            interior_area = (rect_h - 2) * (rect_w - 2)
            max_seed_count = min(perimeter, interior_area)
            if max_seed_count < 1:
                continue

            seed_count = _sample_seed_count(rng, max_seed_count)
            _place_region(grid, occupied, top, left, rect_h, rect_w, color, seed_count, rng)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
