import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_SIDE = 3
MAX_SIDE = 7
GEN_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the colored row/column rulers on the top and left border to relocate each
    interior marker to the cell whose row ruler color and column ruler color both
    match the marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4, "expected rulers plus a nontrivial interior"

    row_scale = [int(v) for v in g[1:, 0]]
    col_scale = [int(v) for v in g[0, 1:]]
    assert len(set(row_scale)) == len(row_scale), "row ruler colors must be unique"
    assert len(set(col_scale)) == len(col_scale), "column ruler colors must be unique"

    interior = g[1:, 1:]
    bg = int(np.bincount(interior.reshape(-1), minlength=10).argmax())

    row_by_color = {color: r for r, color in enumerate(row_scale)}
    col_by_color = {color: c for c, color in enumerate(col_scale)}

    out = g.copy()
    out[1:, 1:] = bg

    for r, c in np.argwhere(interior != bg):
        color = int(interior[r, c])
        assert color in row_by_color, "every marker must appear on the row ruler"
        assert color in col_by_color, "every marker must appear on the column ruler"
        out[1 + row_by_color[color], 1 + col_by_color[color]] = color

    return out


def _choose_dimensions() -> tuple[int, int]:
    if random.random() < 0.5:
        return random.randint(MIN_SIDE, MAX_SIDE), random.randint(MIN_SIDE, MAX_SIDE)
    short_side = random.randint(MIN_SIDE, 4)
    long_side = random.randint(5, MAX_SIDE)
    if random.random() < 0.5:
        return short_side, long_side
    return long_side, short_side


def _choose_shared_count(n_rows: int, n_cols: int) -> int:
    shared_min = max(2, n_rows + n_cols - len(PALETTE))
    shared_max = min(n_rows, n_cols)
    if shared_min == shared_max:
        return shared_min
    mode = random.random()
    if mode < 1.0 / 3.0:
        return shared_min
    if mode < 2.0 / 3.0:
        return shared_max
    return random.randint(shared_min, shared_max)


def _pick_scale_colors(n_rows: int, n_cols: int) -> tuple[list[int], list[int], list[int]]:
    shared_count = _choose_shared_count(n_rows, n_cols)
    shared_colors = random.sample(PALETTE, shared_count)

    leftovers = [color for color in PALETTE if color not in shared_colors]
    row_only_count = n_rows - shared_count
    row_only = random.sample(leftovers, row_only_count)

    remaining = [color for color in leftovers if color not in row_only]
    col_only_count = n_cols - shared_count
    col_only = random.sample(remaining, col_only_count)

    row_colors = list(shared_colors) + row_only
    col_colors = list(shared_colors) + col_only
    random.shuffle(row_colors)
    random.shuffle(col_colors)
    return row_colors, col_colors, shared_colors


def _choose_mark_colors(shared_colors: list[int]) -> list[int]:
    if len(shared_colors) == 2 or random.random() < 0.5:
        mark_count = len(shared_colors)
    else:
        mark_count = random.randint(2, len(shared_colors) - 1)
    return random.sample(shared_colors, mark_count)


def _gen_attempt() -> np.ndarray:
    n_rows, n_cols = _choose_dimensions()
    row_colors, col_colors, shared_colors = _pick_scale_colors(n_rows, n_cols)
    mark_colors = _choose_mark_colors(shared_colors)

    row_by_color = {color: r for r, color in enumerate(row_colors)}
    col_by_color = {color: c for c, color in enumerate(col_colors)}
    target_cells = {(row_by_color[color], col_by_color[color]) for color in mark_colors}

    candidates = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if (r, c) not in target_cells
    ]
    assert len(candidates) >= len(mark_colors), "need enough off-target source cells"
    source_cells = random.sample(candidates, len(mark_colors))

    grid = np.full((n_rows + 1, n_cols + 1), BG, dtype=int)
    grid[0, 0] = SEPARATOR
    grid[1:, 0] = np.array(row_colors, dtype=int)
    grid[0, 1:] = np.array(col_colors, dtype=int)

    for color, (r, c) in zip(mark_colors, source_cells):
        grid[1 + r, 1 + c] = int(color)

    out = main(grid)
    assert not np.array_equal(grid, out), "instances should require relocation"
    assert np.array_equal(out[0, :], grid[0, :]), "top ruler must stay fixed"
    assert np.array_equal(out[:, 0], grid[:, 0]), "left ruler must stay fixed"
    assert np.array_equal(main(out), out), "transform should be idempotent"

    for color in mark_colors:
        tr = 1 + row_by_color[color]
        tc = 1 + col_by_color[color]
        assert int(out[tr, tc]) == int(color), "marker must land on its ruler-coded target"

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ruler-relocation instance")
