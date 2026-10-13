from __future__ import annotations

import random

import numpy as np


BG = 0

# --------------- generator controls ---------------

HEIGHT_RANGE = (7, 15)
WIDTH_RANGE = (8, 18)
NONEMPTY_ROW_RANGE = (3, 8)
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


def _row_side(cols: np.ndarray, width: int) -> str:
    axis = (width - 1) / 2.0
    if np.all(cols < axis):
        return "left"
    if np.all(cols > axis):
        return "right"
    raise AssertionError("each non-empty row must stay entirely on one side of the midpoint")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each non-empty row toward its empty half across the row midpoint.

    Input structure:
    - The background is black.
    - Only some rows contain non-background pixels; every other row is fully black.
    - In each non-empty row, all colored pixels lie entirely in the left half or entirely in the right half of that row.

    Rule:
    - Keep every empty row unchanged.
    - For a non-empty row whose colored pixels are all in the left half, copy those pixels to their horizontal mirror positions in the right half.
    - For a non-empty row whose colored pixels are all in the right half, copy those pixels to their horizontal mirror positions in the left half.
    - Preserve every original colored pixel and keep the output size unchanged.

    Key insight:
    - The mirror axis is the same vertical midpoint for the whole grid, but the copy direction is decided independently for each row by which side already contains color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = g.copy()
    width = g.shape[1]
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] != BG)
        if cols.size == 0:
            continue

        _row_side(cols, width)
        mirror_cols = width - 1 - cols
        assert np.all(g[r, mirror_cols] == BG), "mirror targets must be empty in the input"
        out[r, mirror_cols] = g[r, cols]

    return out


def _side_columns(width: int, side: str) -> list[int]:
    mid = width // 2
    if side == "left":
        return list(range(mid))
    start = mid if width % 2 == 0 else mid + 1
    return list(range(start, width))


def _sample_local_positions(side_width: int) -> list[int]:
    max_cells = side_width if side_width <= 3 else side_width - 1
    target = random.randint(1, max_cells)
    cells = {random.randrange(side_width)}

    while len(cells) < target:
        frontier = {
            nxt
            for cell in cells
            for nxt in (cell - 1, cell + 1)
            if 0 <= nxt < side_width and nxt not in cells
        }
        free = [idx for idx in range(side_width) if idx not in cells]
        if frontier and random.random() < 0.7:
            cells.add(random.choice(tuple(frontier)))
        else:
            cells.add(random.choice(free))

    return sorted(cells)


def _sample_colors(length: int, palette: list[int]) -> list[int]:
    subset = random.sample(palette, random.randint(1, min(3, len(palette), length)))
    if len(subset) == 1 or length == 1:
        return [subset[0]] * length

    mode = random.choice(("cycle", "bands", "random"))
    if mode == "cycle":
        start = random.randrange(len(subset))
        return [subset[(start + idx) % len(subset)] for idx in range(length)]
    if mode == "bands":
        colors: list[int] = []
        remaining = length
        for idx, color in enumerate(subset):
            left = len(subset) - idx - 1
            run = random.randint(1, remaining - left)
            colors.extend([color] * run)
            remaining -= run
        while len(colors) < length:
            colors.append(colors[-1])
        return colors[:length]
    return [random.choice(subset) for _ in range(length)]


def _build_row(width: int, side: str, palette: list[int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    columns = _side_columns(width, side)
    local_positions = _sample_local_positions(len(columns))
    colors = _sample_colors(len(local_positions), palette)

    for local_idx, color in zip(local_positions, colors):
        row[columns[local_idx]] = color
    return row


def _grid_sides(nonempty_count: int) -> list[str]:
    left_count = random.randint(1, nonempty_count - 1)
    sides = ["left"] * left_count + ["right"] * (nonempty_count - left_count)
    random.shuffle(sides)
    return sides


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    max_nonempty = min(NONEMPTY_ROW_RANGE[1], height - 1)
    min_nonempty = min(NONEMPTY_ROW_RANGE[0], max_nonempty)
    if min_nonempty < 2:
        return None

    nonempty_count = random.randint(min_nonempty, max_nonempty)
    row_indices = sorted(random.sample(range(height), nonempty_count))
    palette = random.sample(list(PALETTE), random.randint(2, 5))

    grid = np.zeros((height, width), dtype=int)
    row_signatures: set[tuple[int, ...]] = set()
    used_colors: set[int] = set()

    for row_idx, side in zip(row_indices, _grid_sides(nonempty_count)):
        for _ in range(20):
            row = _build_row(width, side, palette)
            sig = tuple(int(v) for v in row)
            if sig not in row_signatures:
                break
        else:
            row = _build_row(width, side, palette)
            sig = tuple(int(v) for v in row)

        row_signatures.add(sig)
        grid[row_idx] = row
        used_colors.update(int(v) for v in row if v != BG)

    if len(used_colors) < 2:
        filled = np.argwhere(grid != BG)
        if filled.shape[0] < 2:
            return None
        grid[int(filled[0, 0]), int(filled[0, 1])] = palette[0]
        grid[int(filled[1, 0]), int(filled[1, 1])] = palette[1]

    row_sides = []
    for row in grid:
        cols = np.flatnonzero(row != BG)
        if cols.size:
            row_sides.append(_row_side(cols, width))
    if "left" not in row_sides or "right" not in row_sides:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-row ARC input")
