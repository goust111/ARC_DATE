from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SOURCE_COLORS = (BLUE, RED, GREEN, ORANGE, AZURE)
GREY_SOURCE_PAIRS = (
    (RED, ORANGE),
    (BLUE, ORANGE),
    (GREEN, ORANGE),
    (BLUE, AZURE),
    (GREEN, AZURE),
    (AZURE, ORANGE),
)
BRANCHES = ("magenta", "yellow", "azure", "maroon", "grey")
MAX_GENERATION_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (6, 12)
GRID_WIDTH_RANGE = (6, 12)

MIX_TABLE = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, GREEN)): YELLOW,
    frozenset((BLUE, GREEN)): AZURE,
    frozenset((RED, AZURE)): MAROON,
}


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _mix_colors(a: int, b: int) -> int:
    if a == b:
        return int(a)
    return int(MIX_TABLE.get(frozenset((int(a), int(b))), GREY))


def _project_from_bars(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()
    interior = np.full((height - 1, width - 1), bg, dtype=int)

    top_sources = g[0, 1:]
    left_sources = g[1:, 0]
    active_cols = np.flatnonzero(top_sources != bg)
    active_rows = np.flatnonzero(left_sources != bg)

    if active_cols.size:
        interior[:, active_cols] = top_sources[active_cols]
    if active_rows.size:
        interior[active_rows, :] = left_sources[active_rows, None]

    for row_idx in active_rows:
        row_color = int(left_sources[row_idx])
        for col_idx in active_cols:
            interior[row_idx, col_idx] = _mix_colors(int(top_sources[col_idx]), row_color)

    out[1:, 1:] = interior
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project row/column edge bars into the interior and recolor crossings with a fixed mixing table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "grid must have an interior"

    bg = _background_color(g)
    return _project_from_bars(g, bg)


def _required_pair() -> tuple[str, tuple[int, int]]:
    branch = random.choice(BRANCHES)
    if branch == "magenta":
        pair = (RED, BLUE)
    elif branch == "yellow":
        pair = (RED, GREEN)
    elif branch == "azure":
        pair = (BLUE, GREEN)
    elif branch == "maroon":
        pair = (RED, AZURE)
    else:
        pair = random.choice(GREY_SOURCE_PAIRS)
    return branch, pair


def _sample_color_sets() -> tuple[str, tuple[int, int], list[int], list[int]]:
    branch, pair = _required_pair()
    if random.random() < 0.5:
        top_colors = [pair[0]]
        left_colors = [pair[1]]
    else:
        top_colors = [pair[1]]
        left_colors = [pair[0]]

    remaining = [color for color in SOURCE_COLORS if color not in pair]
    random.shuffle(remaining)
    for color in remaining:
        roll = random.random()
        if roll < 0.35:
            top_colors.append(color)
        elif roll < 0.70:
            left_colors.append(color)

    if len(top_colors) + len(left_colors) < 3:
        extra = random.choice(remaining)
        if random.random() < 0.5:
            top_colors.append(extra)
        else:
            left_colors.append(extra)

    random.shuffle(top_colors)
    random.shuffle(left_colors)
    assert set(top_colors).isdisjoint(left_colors)
    return branch, pair, top_colors, left_colors


def _build_input(top_colors: list[int], left_colors: list[int]) -> np.ndarray | None:
    min_height = max(GRID_HEIGHT_RANGE[0], len(left_colors) + 2)
    min_width = max(GRID_WIDTH_RANGE[0], len(top_colors) + 2)
    if min_height > GRID_HEIGHT_RANGE[1] or min_width > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    grid = np.full((height, width), BLACK, dtype=int)

    row_positions = random.sample(range(1, height), k=len(left_colors))
    col_positions = random.sample(range(1, width), k=len(top_colors))

    for col, color in zip(col_positions, top_colors):
        grid[0, col] = color
    for row, color in zip(row_positions, left_colors):
        grid[row, 0] = color

    return grid


def _attempt() -> np.ndarray | None:
    branch, pair, top_colors, left_colors = _sample_color_sets()
    grid = _build_input(top_colors, left_colors)
    if grid is None:
        return None

    out = main(grid)
    required_color = _mix_colors(*pair)
    if required_color not in set(map(int, np.unique(out[1:, 1:]))):
        return None

    bg = BLACK
    assert np.all(grid[1:, 1:] == bg)
    assert len(np.flatnonzero(grid[0, 1:] != bg)) < grid.shape[1] - 1
    assert len(np.flatnonzero(grid[1:, 0] != bg)) < grid.shape[0] - 1

    top_sources = grid[0, 1:]
    left_sources = grid[1:, 0]
    active_cols = np.flatnonzero(top_sources != bg)
    active_rows = np.flatnonzero(left_sources != bg)
    if not active_cols.size or not active_rows.size:
        return None

    # Keep both single-layer branches available in generated samples.
    if len(active_cols) >= grid.shape[1] - 1 or len(active_rows) >= grid.shape[0] - 1:
        return None

    expected = _project_from_bars(grid, bg)
    if not np.array_equal(out, expected):
        return None

    # `branch` is sampled to balance long-run coverage across all mix outcomes.
    _ = branch
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
