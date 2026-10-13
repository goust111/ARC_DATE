from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 500
GRID_ROWS_RANGE = (13, 18)
GRID_COLS_RANGE = (13, 18)
STRIPE_SPACINGS = (3, 4)
BRANCH_TYPES = ("equal", "row_major", "col_major")


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _sign(delta: int) -> int:
    return (delta > 0) - (delta < 0)


def _stripe_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert rows, "expected at least one full grey stripe row"
    return rows


def _chebyshev_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    end_r, end_c = end
    path: list[tuple[int, int]] = []

    step_r = _sign(end_r - r)
    step_c = _sign(end_c - c)

    while r != end_r and c != end_c:
        r += step_r
        c += step_c
        if (r, c) != end:
            path.append((r, c))

    while r != end_r:
        r += step_r
        if (r, c) != end:
            path.append((r, c))

    while c != end_c:
        c += step_c
        if (r, c) != end:
            path.append((r, c))

    return path


def _paint_path(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    stripe_rows: set[int],
) -> int:
    path = _chebyshev_path(start, end)
    crossings = 0
    use_orange = False
    prev = start

    for cell in path:
        if prev[0] not in stripe_rows and cell[0] in stripe_rows:
            use_orange = not use_orange
            crossings += 1
        out[cell] = ORANGE if use_orange else GREEN
        prev = cell

    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the blue-to-red Chebyshev path and alternate its color whenever it enters a periodic stripe row.

    Input structure:
    - A black background grid contains exactly one blue start pixel and one red end pixel.
    - Several full horizontal grey stripe rows cross the whole grid at a fixed period.
    - The two markers are off the stripe rows and differ in both row and column.

    Rule:
    - Follow the Chebyshev-shortest king-move path from blue to red: take diagonal steps until one axis aligns, then continue straight along the remaining axis.
    - Paint only the intermediate path cells, starting in green; whenever the next path cell enters a grey stripe row from a non-stripe row, switch the path color, and paint that entry cell in the new color.
    - Count those stripe-entry events and draw a yellow 1xC summary bar from the bottom-right corner leftward, while keeping the blue and red markers unchanged.

    Key insight:
    - The path geometry comes only from the marker pair, but the visible green/orange segmentation is created by where that fixed path intersects the independent grey stripe rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start = _single_coord(g, BLUE)
    end = _single_coord(g, RED)
    assert start[0] != end[0] and start[1] != end[1], "markers must differ on both axes"

    stripe_rows = set(_stripe_rows(g))
    assert start[0] not in stripe_rows and end[0] not in stripe_rows, "markers must be off stripe rows"

    out = g.copy()
    crossings = _paint_path(out, start, end, stripe_rows)
    if crossings > 0:
        out[-1, out.shape[1] - crossings :] = YELLOW

    out[start] = BLUE
    out[end] = RED
    return out


def _rows_with_required_crossings(height: int, stripe_rows: list[int], required: int) -> list[tuple[int, int]]:
    stripe_set = set(stripe_rows)
    candidates = []
    for r0 in range(1, height - 1):
        if r0 in stripe_set:
            continue
        for r1 in range(1, height - 1):
            if r1 in stripe_set or r0 == r1:
                continue
            low, high = sorted((r0, r1))
            crossings = sum(low < stripe < high for stripe in stripe_rows)
            if crossings >= required:
                candidates.append((r0, r1))
    return candidates


def _col_pairs(width: int, distance: int) -> list[tuple[int, int]]:
    pairs = []
    for c0 in range(1, width - 1):
        for sign in (-1, 1):
            c1 = c0 + sign * distance
            if 1 <= c1 < width - 1:
                pairs.append((c0, c1))
    return pairs


def _pick_branch_distance(rng: random.Random, row_diff: int, width: int) -> tuple[str, int] | None:
    choices: list[tuple[str, int]] = []

    if row_diff <= width - 3:
        choices.append(("equal", row_diff))
    if row_diff >= 2:
        for col_diff in range(1, min(row_diff - 1, width - 3) + 1):
            choices.append(("row_major", col_diff))
    for col_diff in range(row_diff + 1, width - 2):
        choices.append(("col_major", col_diff))

    if not choices:
        return None

    weighted: list[tuple[str, int]] = []
    for branch in BRANCH_TYPES:
        branch_choices = [choice for choice in choices if choice[0] == branch]
        if branch_choices:
            weighted.append(rng.choice(branch_choices))
    return rng.choice(weighted) if weighted else None


def _choose_stripe_rows(rng: random.Random, height: int) -> list[int]:
    spacing = rng.choice(STRIPE_SPACINGS)
    max_offset = min(spacing + 1, height - 5)
    min_offset = 2
    if max_offset < min_offset:
        return []
    offset = rng.randint(min_offset, max_offset)
    rows = list(range(offset, height - 1, spacing))
    return [r for r in rows if 0 < r < height - 1]


def _build_input(
    height: int,
    width: int,
    stripe_rows: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for row in stripe_rows:
        grid[row, :] = GREY
    grid[start] = BLUE
    grid[end] = RED
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_ROWS_RANGE)
    width = rng.randint(*GRID_COLS_RANGE)

    stripe_rows = _choose_stripe_rows(rng, height)
    if len(stripe_rows) < 2:
        return None

    row_pairs = _rows_with_required_crossings(height, stripe_rows, required=2)
    if not row_pairs:
        return None

    start_r, end_r = rng.choice(row_pairs)
    row_diff = abs(end_r - start_r)
    branch = _pick_branch_distance(rng, row_diff, width)
    if branch is None:
        return None

    _branch_name, col_diff = branch
    col_pairs = _col_pairs(width, col_diff)
    if not col_pairs:
        return None
    start_c, end_c = rng.choice(col_pairs)

    start = (start_r, start_c)
    end = (end_r, end_c)
    grid = _build_input(height, width, stripe_rows, start, end)

    try:
        out = main(grid)
    except AssertionError:
        return None

    path = _chebyshev_path(start, end)
    crossings = sum(
        prev[0] not in set(stripe_rows) and cell[0] in set(stripe_rows)
        for prev, cell in zip([start] + path[:-1], path)
    )

    if crossings < 2:
        return None
    if np.array_equal(out, grid):
        return None
    if np.any(grid[-1, width - crossings :] != BLACK):
        return None
    if not np.all(out[-1, width - crossings :] == YELLOW):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
