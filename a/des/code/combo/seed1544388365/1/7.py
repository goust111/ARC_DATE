from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _draw_block(
    out: np.ndarray, row: int, col: int, row_color: int, col_color: int
) -> None:
    if row_color == col_color:
        pattern = ((0, 0, row_color), (0, 1, row_color), (1, 0, row_color), (1, 1, row_color))
    else:
        pattern = ((0, 0, row_color), (0, 1, col_color), (1, 0, col_color), (1, 1, row_color))

    height, width = out.shape
    for dr, dc, color in pattern:
        rr = row + dr
        cc = col + dc
        if rr < height and cc < width:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw 2x2 mixed-color blocks at the intersections of row and column labels.

    Input structure:
    - The background color is dominant.
    - Some cells on the top edge label columns.
    - Some cells on the left edge label rows.
    - The interior starts as background.

    Rule:
    - For every labeled row r with color A and every labeled column c with color B,
      draw a 2x2 block anchored at (r, c), clipped to the grid bounds.
    - If A == B, the block is solid A.
    - Otherwise the block is a checkerboard with A on the main diagonal.
    - Later intersections overwrite earlier ones in row-major order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    bg = _get_bg(g)

    out = g.copy()
    row_labels = [(r, int(g[r, 0])) for r in range(1, height) if int(g[r, 0]) != bg]
    col_labels = [(c, int(g[0, c])) for c in range(1, width) if int(g[0, c]) != bg]

    for r, row_color in row_labels:
        for c, col_color in col_labels:
            _draw_block(out, r, c, row_color, col_color)

    return out


_RNG = random.Random()
_GRID_MIN = 4
_GRID_MAX = 11
_MAX_LABELS = 4
_GEN_RETRIES = 300


def _sample_positions(
    size: int, count: int, want_adjacent: bool, want_last: bool
) -> list[int] | None:
    choices = list(range(1, size))
    for _ in range(200):
        picked = sorted(_RNG.sample(choices, count))
        if want_last and picked[-1] != size - 1:
            continue
        has_adjacent = any(b - a == 1 for a, b in zip(picked, picked[1:]))
        if has_adjacent != want_adjacent:
            continue
        return picked
    return None


def _assign_colors(n_rows: int, n_cols: int, palette: list[int]) -> tuple[list[int], list[int]]:
    shared = _RNG.choice(palette)
    others = [color for color in palette if color != shared]
    different = _RNG.choice(others)

    row_colors = [_RNG.choice(palette) for _ in range(n_rows)]
    col_colors = [_RNG.choice(palette) for _ in range(n_cols)]

    row_colors[_RNG.randrange(n_rows)] = shared
    col_colors[_RNG.randrange(n_cols)] = shared

    if n_rows >= 2:
        row_colors[(_RNG.randrange(n_rows - 1) + 1) % n_rows] = different
    else:
        col_colors[(_RNG.randrange(n_cols - 1) + 1) % n_cols] = different

    if all(color == shared for color in col_colors):
        col_colors[_RNG.randrange(n_cols)] = different
    if all(color == shared for color in row_colors):
        row_colors[_RNG.randrange(n_rows)] = different

    return row_colors, col_colors


def _branch_flags(colors_a: list[int], colors_b: list[int]) -> tuple[bool, bool]:
    same = any(a == b for a in colors_a for b in colors_b)
    diff = any(a != b for a in colors_a for b in colors_b)
    return same, diff


def _gen_attempt() -> np.ndarray | None:
    height = _RNG.randint(_GRID_MIN, _GRID_MAX)
    width = _RNG.randint(_GRID_MIN, _GRID_MAX)

    dense_rows = _RNG.random() < 0.5
    dense_cols = _RNG.random() < 0.5
    clip_rows = _RNG.random() < 0.5
    clip_cols = _RNG.random() < 0.5

    max_sparse_rows = max(2, height // 2)
    max_sparse_cols = max(2, width // 2)
    row_cap = min(_MAX_LABELS, height - 1 if dense_rows else max_sparse_rows)
    col_cap = min(_MAX_LABELS, width - 1 if dense_cols else max_sparse_cols)
    if row_cap < 2 or col_cap < 2:
        return None

    row_count = _RNG.randint(2, row_cap)
    col_count = _RNG.randint(2, col_cap)

    row_positions = _sample_positions(height, row_count, dense_rows, clip_rows)
    col_positions = _sample_positions(width, col_count, dense_cols, clip_cols)
    if row_positions is None or col_positions is None:
        return None

    bg = _RNG.randrange(10)
    palette_pool = [color for color in range(10) if color != bg]
    palette_size = _RNG.randint(2, min(4, len(palette_pool)))
    palette = _RNG.sample(palette_pool, palette_size)
    row_colors, col_colors = _assign_colors(len(row_positions), len(col_positions), palette)

    same_branch, diff_branch = _branch_flags(row_colors, col_colors)
    if not (same_branch and diff_branch):
        return None

    grid = np.full((height, width), bg, dtype=int)
    for pos, color in zip(row_positions, row_colors):
        grid[pos, 0] = color
    for pos, color in zip(col_positions, col_colors):
        grid[0, pos] = color

    if int(grid[0, 0]) != bg:
        return None
    if not np.all(grid[1:, 1:] == bg):
        return None
    if int((grid == bg).sum()) <= max(int((grid == color).sum()) for color in np.unique(grid) if int(color) != bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
