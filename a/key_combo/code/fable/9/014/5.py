from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_RNG = np.random.default_rng()

MIN_PANEL_HEIGHT = 4
MAX_PANEL_HEIGHT = 8
MIN_PANEL_WIDTH = 4
MAX_PANEL_WIDTH = 8
MAX_GENERATION_ATTEMPTS = 400


def _non_bg_colors(panel: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(panel) if int(color) != bg)


def _find_divider_row(grid: np.ndarray, bg: int) -> int:
    height, width = map(int, grid.shape)
    colors, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    candidates = []
    for row in range(height):
        color = int(grid[row, 0])
        if color == bg:
            continue
        if np.all(grid[row] == color) and count_by_color[color] == width:
            candidates.append(row)
    assert len(candidates) == 1, "expected exactly one unique full-width divider row"
    return int(candidates[0])


def _column_counts(panel: np.ndarray, fg: int) -> np.ndarray:
    return np.count_nonzero(panel == fg, axis=0)


def _sorted_upper_panel(upper: np.ndarray, bg: int) -> np.ndarray:
    colors = _non_bg_colors(upper, bg)
    assert len(colors) == 1, "upper panel must contain exactly one foreground color"
    fg = colors[0]
    counts = _column_counts(upper, fg)
    assert len(set(map(int, counts))) == upper.shape[1], "upper column counts must be pairwise distinct"
    order = np.argsort(counts, kind="stable")
    return upper[:, order]


def _solve(sorted_upper: np.ndarray, lower: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(sorted_upper.shape, bg, dtype=int)
    out[(sorted_upper == bg) & (lower == bg)] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the upper mask columns by their foreground counts, then mark the cells empty in both panels.

    Input structure:
    - A single full-width divider row splits the grid into equal-height upper and lower panels.
    - The upper panel is a one-color foreground mask on black, and the lower panel is another one-color foreground mask on black.
    - The upper-panel foreground counts are pairwise distinct across columns.

    Rule:
    - Treat each upper-panel column as a whole unit and stably reorder those columns from fewest foreground pixels to most.
    - Leave the lower panel unchanged, remove the divider row, and align the reordered upper panel with the lower panel cell by cell.
    - Output one panel-sized grid: paint green where both aligned cells are black, and black everywhere else.

    Key insight:
    - The visible boolean merge is only correct after a hidden preprocessing step: the upper mask must be column-sorted by foreground count first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid is too small"

    bg = BLACK
    divider_row = _find_divider_row(g, bg)
    assert divider_row > 0 and divider_row < g.shape[0] - 1, "divider cannot touch the edge"
    assert 2 * divider_row + 1 == g.shape[0], "divider must split the grid into equal-height panels"

    upper = g[:divider_row]
    lower = g[divider_row + 1 :]
    assert upper.shape == lower.shape

    upper_colors = _non_bg_colors(upper, bg)
    lower_colors = _non_bg_colors(lower, bg)
    assert len(upper_colors) == 1, "upper panel must use exactly one foreground color"
    assert len(lower_colors) == 1, "lower panel must use exactly one foreground color"
    assert GREEN not in upper_colors and GREEN not in lower_colors, "green is reserved for the output"

    sorted_upper = _sorted_upper_panel(upper, bg)
    return _solve(sorted_upper, lower, bg)


def _make_binary_column(height: int, count: int) -> np.ndarray:
    col = np.zeros(height, dtype=int)
    if count:
        rows = GEN_RNG.choice(height, size=count, replace=False)
        col[rows] = 1
    return col


def _sample_panel_shape() -> tuple[int, int]:
    height = random.randint(MIN_PANEL_HEIGHT, MAX_PANEL_HEIGHT)
    width = random.randint(MIN_PANEL_WIDTH, min(MAX_PANEL_WIDTH, height + 1))
    return height, width


def _sample_sorted_counts(height: int, width: int) -> np.ndarray:
    values = np.arange(height + 1, dtype=int)
    for _ in range(100):
        if random.random() < 0.4:
            start = random.randint(0, height + 1 - width)
            counts = np.arange(start, start + width, dtype=int)
        else:
            counts = np.sort(GEN_RNG.choice(values, size=width, replace=False))
        free_cells = height * width - int(counts.sum())
        if free_cells >= 5 and int(counts.sum()) >= 5:
            return counts
    raise RuntimeError("failed to sample valid distinct upper-column counts")


def _build_sorted_upper(height: int, counts: Iterable[int], fg: int) -> np.ndarray:
    columns = []
    for count in counts:
        col = _make_binary_column(height, int(count))
        columns.append(np.where(col == 1, fg, BLACK))
    return np.stack(columns, axis=1)


def _sample_non_identity_permutation(width: int) -> np.ndarray:
    perm = np.arange(width, dtype=int)
    while np.array_equal(perm, np.arange(width, dtype=int)):
        perm = GEN_RNG.permutation(width)
    return perm


def _choose_cells(cells: np.ndarray, count: int) -> np.ndarray:
    assert 0 <= count <= len(cells)
    if count == 0:
        return np.empty((0,), dtype=int)
    idx = GEN_RNG.choice(len(cells), size=count, replace=False)
    return cells[np.asarray(idx, dtype=int)]


def _build_lower(sorted_upper: np.ndarray, lower_fg: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, sorted_upper.shape)
    upper_bg = np.flatnonzero(sorted_upper.ravel() == BLACK)
    upper_fg = np.flatnonzero(sorted_upper.ravel() != BLACK)
    assert len(upper_bg) >= 5 and len(upper_fg) >= 5

    max_green = len(upper_bg) - 2
    green_count = random.randint(2, max_green)
    green_flats = _choose_cells(upper_bg, green_count)

    remaining_bg = np.setdiff1d(upper_bg, green_flats, assume_unique=False)
    lower_only_cap = min(len(remaining_bg), max(2, (height * width) // 3))
    lower_only_count = random.randint(2, lower_only_cap)
    lower_only_flats = _choose_cells(remaining_bg, lower_only_count)

    both_count = 0
    if len(upper_fg) >= 2 and random.random() < 0.55:
        both_cap = min(len(upper_fg), max(2, (height * width) // 4))
        both_count = random.randint(1, both_cap)
    both_flats = _choose_cells(upper_fg, both_count)

    lower = np.full(sorted_upper.shape, BLACK, dtype=int)
    lower.ravel()[lower_only_flats] = lower_fg
    lower.ravel()[both_flats] = lower_fg

    expected = np.full(sorted_upper.shape, BLACK, dtype=int)
    expected.ravel()[green_flats] = GREEN
    return lower, expected


def _count_color(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == color))


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    divider_row = _find_divider_row(inp, BLACK)
    upper = inp[:divider_row]
    lower = inp[divider_row + 1 :]
    upper_fg = _non_bg_colors(upper, BLACK)[0]

    counts = _column_counts(upper, upper_fg)
    if len(set(map(int, counts))) != upper.shape[1]:
        return False
    if np.all(counts[:-1] <= counts[1:]):
        return False

    sorted_upper = _sorted_upper_panel(upper, BLACK)
    naive = _solve(upper, lower, BLACK)
    diff = expected != naive
    if _count_color(expected, GREEN) < 2:
        return False
    if _count_color(expected, BLACK) < 2:
        return False
    if int(np.count_nonzero(diff)) < 2:
        return False
    if int(np.count_nonzero((sorted_upper == BLACK) & (lower != BLACK))) < 2:
        return False
    if int(np.count_nonzero((sorted_upper != BLACK) & (lower == BLACK))) < 2:
        return False
    if np.array_equal(expected, np.where(lower == BLACK, GREEN, BLACK)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_panel_shape()
    sep_color, upper_fg, lower_fg = random.sample(INPUT_COLORS, 3)

    counts = _sample_sorted_counts(height, width)
    sorted_upper = _build_sorted_upper(height, counts, upper_fg)
    lower, expected = _build_lower(sorted_upper, lower_fg)

    perm = _sample_non_identity_permutation(width)
    upper_input = sorted_upper[:, perm]
    divider = np.full((1, width), sep_color, dtype=int)
    inp = np.vstack([upper_input, divider, lower])

    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/9/014/5")
