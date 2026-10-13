from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

PAIR_COLORS = (BLUE, GREEN, YELLOW, PINK)
GENERATION_RETRIES = 400


def _extract_pairs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return [(color, left_row, right_row)] for the unique border pair of each color."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 5 and w % 2 == 1, "expected an odd-width canvas with room to meet"

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    pairs: list[tuple[int, int, int]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"

        left = [(int(r), int(c)) for r, c in coords if int(c) == 0]
        right = [(int(r), int(c)) for r, c in coords if int(c) == w - 1]
        assert len(left) == 1 and len(right) == 1, "each color must appear once on each side edge"

        left_row = left[0][0]
        right_row = right[0][0]
        assert abs(right_row - left_row) == 1, "paired edge pixels must differ by exactly one row"
        assert 0 < left_row < h - 1 or 0 < right_row < h - 1
        pairs.append((color, left_row, right_row))

    assert 3 <= len(pairs) <= 4, "expected three or four color pairs"
    return sorted(pairs, key=lambda item: (item[1], item[2], item[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color left-right border pair inward, mark the offset-side meeting point gray, and emit a vertical gray ray from it.

    Input structure:
    - The background is black.
    - Three or four colors each appear as isolated single pixels, with one pixel on the left edge and one on the right edge.
    - For each color pair, the left and right pixels lie on adjacent rows, differing by exactly one cell vertically.

    Rule:
    - Pair pixels by color.
    - For each pair, extend the left pixel horizontally rightward and the right pixel horizontally leftward until both reach the common meeting column at the canvas center.
    - Mark the meeting cell on the offset-side row as gray.
    - From that gray meeting cell, extend a gray vertical ray further in the same up/down direction as the right-side offset, stopping at the grid edge and never overwriting colored trail pixels.

    Key insight:
    - The tiny one-cell vertical asymmetry on the right edge is not noise: it selects both which row hosts the gray meeting point and whether the gray ray goes upward or downward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    meet_col = w // 2
    pairs = _extract_pairs(g)

    out = np.zeros_like(g)

    # Draw all horizontal colored trails first so they always take precedence over rays.
    for color, left_row, right_row in pairs:
        out[left_row, : meet_col + 1] = int(color)
        out[right_row, meet_col:] = int(color)
        out[right_row, meet_col] = GREY

    # The right-edge row is the visible one-cell offset from the left-side baseline.
    for _color, left_row, right_row in pairs:
        step = -1 if right_row < left_row else 1
        ray_row = right_row + step
        while 0 <= ray_row < h:
            if int(out[ray_row, meet_col]) == BLACK:
                out[ray_row, meet_col] = GREY
            ray_row += step

    return out


def _choose_left_rows(height: int, pair_count: int) -> list[int] | None:
    rows = list(range(2, height - 2))
    for _ in range(200):
        chosen = sorted(random.sample(rows, pair_count))
        if all(b - a >= 4 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _pick_directions(pair_count: int) -> list[int]:
    directions = [random.choice((-1, 1)) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(directions)) == 1:
        directions[random.randrange(pair_count)] *= -1
    return directions


def _build_input(height: int, width: int, colors: list[int], left_rows: list[int], directions: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for color, left_row, direction in zip(colors, left_rows, directions):
        right_row = left_row + direction
        grid[left_row, 0] = int(color)
        grid[right_row, width - 1] = int(color)
    return grid


def _is_isolated_input(grid: np.ndarray) -> bool:
    coords = np.argwhere(grid != BLACK)
    cells = {tuple(map(int, rc)) for rc in coords}
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(3, 4)
    min_height = 4 * pair_count + 1
    height = random.randint(min_height, 23)
    width = random.choice([size for size in range(13, 22) if size % 2 == 1])

    left_rows = _choose_left_rows(height, pair_count)
    if left_rows is None:
        return None

    directions = _pick_directions(pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)
    grid = _build_input(height, width, colors, left_rows, directions)

    if not _is_isolated_input(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == GREY)) < pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
