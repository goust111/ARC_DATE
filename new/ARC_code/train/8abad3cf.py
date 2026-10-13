import math

import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Repack each non-background color group as a solid square ordered by increasing size.

    Input structure
    The grid has one dominant background color, and each other color has a
    perfect-square pixel count. Those square side lengths are unique.

    Rule
    Convert every non-background color into a filled square whose side length is
    the square root of its pixel count. Place the squares from left to right in
    increasing side length, bottom-aligned, with one background column between
    neighboring squares.

    Key insight
    The original geometry of each color is discarded; only the color's total
    pixel count determines the size of its packed output square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    specs: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        count = int(np.count_nonzero(g == color))
        side = math.isqrt(count)
        assert side * side == count
        specs.append((color, side))

    assert specs
    sides = [side for _color, side in specs]
    assert len(set(sides)) == len(sides)

    ordered = sorted(specs, key=lambda item: item[1])
    height = max(side for _color, side in ordered)
    width = sum(side for _color, side in ordered) + len(ordered) - 1
    out = np.full((height, width), bg, dtype=int)

    col = 0
    for i, (color, side) in enumerate(ordered):
        out[height - side : height, col : col + side] = color
        col += side
        if i != len(ordered) - 1:
            col += 1

    assert col == width
    return out

# --------------- generator (att1) ---------------

# Fixed task palette/background.
BG_COLOR = 7
ALL_COLORS = range(10)

# Color-group structure.
COLOR_COUNTS = [2, 3]
SIDE_LENGTHS = range(1, 6)

# Input canvas and density bounds.
ROW_RANGE = range(5, 8)
COL_RANGE = range(4, 11)
MIN_BG_FRACTION = 0.42
MAX_BG_FRACTION = 0.58

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _sample_groups() -> tuple[list[int], list[int]]:
    """Return colors and distinct square sides using the original sampling order."""
    n_colors = random.choice(COLOR_COUNTS)
    sides = sorted(random.sample(SIDE_LENGTHS, n_colors))
    available_colors = [c for c in ALL_COLORS if c != BG_COLOR]
    colors = random.sample(available_colors, n_colors)
    return colors, sides


def _valid_dimensions(total_pixels: int) -> list[tuple[int, int]]:
    min_area = max(total_pixels + 1, int(np.ceil(total_pixels / MAX_BG_FRACTION)))
    max_area = int(np.floor(total_pixels / MIN_BG_FRACTION))
    if min_area > max_area:
        return []

    return [
        (rows, cols)
        for rows in ROW_RANGE
        for cols in COL_RANGE
        if min_area <= rows * cols <= max_area
    ]


def _scatter_pixels(rows: int, cols: int, colors: list[int], sides: list[int]) -> np.ndarray:
    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_positions)

    idx = 0
    for color, side in zip(colors, sides):
        count = side * side
        for _ in range(count):
            r, c = all_positions[idx]
            grid[r, c] = color
            idx += 1
    return grid


def _expected_output(colors: list[int], sides: list[int]) -> np.ndarray:
    ordered = sorted(zip(colors, sides), key=lambda x: x[1])
    height = max(sides)
    width = sum(sides) + len(sides) - 1
    expected = np.full((height, width), BG_COLOR, dtype=int)
    col = 0
    for i, (color, side) in enumerate(ordered):
        expected[height - side:height, col:col + side] = color
        col += side
        if i != len(ordered) - 1:
            col += 1
    return expected


def _gen_attempt() -> np.ndarray | None:
    colors, sides = _sample_groups()
    total_pixels = sum(s * s for s in sides)
    valid_dims = _valid_dimensions(total_pixels)
    if not valid_dims:
        return None

    rows, cols = random.choice(valid_dims)
    grid = _scatter_pixels(rows, cols, colors, sides)

    try:
        output = main(grid)
    except (AssertionError, Exception):
        return None

    if not np.array_equal(output, _expected_output(colors, sides)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
