from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
FRAME = 2
ORANGE = 7
TEAL = 8

MACRO_COUNT = 4
MACRO_SIZE = 3
STRIDE = 4
GRID_SIZE = MACRO_COUNT * MACRO_SIZE + (MACRO_COUNT - 1)

COUNT_PROFILES = (
    (4, 3, 2, 1),
    (4, 3, 2, 0),
    (4, 3, 1, 0),
    (4, 2, 1, 0),
    (3, 2, 1, 0),
)
COUNT_PROFILE_WEIGHTS = (4, 4, 2, 1, 1)
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort macro-columns by filled center count, then flip teal/orange only in columns that changed position.

    Input structure:
    - The grid is a square macro-grid of red 3x3 cells separated by black gutter rows and black gutter columns.
    - Only the center pixel of each macro-cell varies; it is black for empty cells or teal/orange for filled cells.
    - Each macro-column consists of the full 3-pixel-wide band for one cell column, with gutter columns fixed in place.

    Rule:
    - Count filled centers in each macro-column.
    - Reorder the full macro-columns from highest filled-count to lowest, leaving gutter rows/columns unchanged.
    - Detect which original macro-columns moved to a different slot during that sort.
    - In the output, swap teal and orange only for the filled centers belonging to those moved columns; keep black centers and the red/black scaffold unchanged.

    Key insight:
    - The color swap trigger is not marked directly in the input: it is a byproduct of the sorting step, so you must recover the sorted column order before knowing which columns flip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_bands, center_rows = _macro_layout(g, axis=0)
    col_bands, center_cols = _macro_layout(g, axis=1)
    assert len(row_bands) == len(col_bands) >= 2, "expected a square macro-grid"

    _validate_palette(g, center_rows, center_cols)

    centers = g[np.ix_(center_rows, center_cols)]
    filled_counts = np.count_nonzero((centers == ORANGE) | (centers == TEAL), axis=0)
    assert len(set(map(int, filled_counts))) == len(center_cols), "expected unique macro-column counts"
    order = np.argsort(-filled_counts, kind="stable")

    out = g.copy()
    for dst, src in enumerate(order):
        dst0, dst1 = col_bands[dst]
        src0, src1 = col_bands[int(src)]
        out[:, dst0 : dst1 + 1] = g[:, src0 : src1 + 1]
        if int(src) != dst:
            out[center_rows, center_cols[dst]] = _swap_center_colors(out[center_rows, center_cols[dst]])
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/009/4")


def _macro_layout(grid: np.ndarray, axis: int) -> tuple[list[tuple[int, int]], list[int]]:
    length = grid.shape[axis]
    gutters = []
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == BLACK):
            gutters.append(idx)

    cuts = [-1] + gutters + [length]
    bands: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        start = left + 1
        stop = right - 1
        if start <= stop:
            bands.append((start, stop))

    assert all(stop - start + 1 == MACRO_SIZE for start, stop in bands), "expected 3-pixel macro bands"
    assert all(gutters[i + 1] - gutters[i] == STRIDE for i in range(len(gutters) - 1)), "expected evenly spaced gutters"
    centers = [start + 1 for start, _stop in bands]
    return bands, centers


def _validate_palette(grid: np.ndarray, center_rows: list[int], center_cols: list[int]) -> None:
    frame_mask = np.ones(grid.shape, dtype=bool)
    center_mask = np.zeros(grid.shape, dtype=bool)
    center_mask[np.ix_(center_rows, center_cols)] = True
    frame_mask &= ~center_mask

    gutter_rows = np.flatnonzero(np.all(grid == BLACK, axis=1))
    gutter_cols = np.flatnonzero(np.all(grid == BLACK, axis=0))
    frame_mask[gutter_rows, :] = False
    frame_mask[:, gutter_cols] = False

    assert np.all(grid[frame_mask] == FRAME), "expected fixed red macro frames"
    centers = grid[np.ix_(center_rows, center_cols)]
    valid = (centers == BLACK) | (centers == ORANGE) | (centers == TEAL)
    assert np.all(valid), "expected centers to use only black/orange/teal"


def _swap_center_colors(values: np.ndarray) -> np.ndarray:
    out = np.array(values, copy=True)
    out[values == ORANGE] = TEAL
    out[values == TEAL] = ORANGE
    return out


def _render_grid(columns: list[np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), FRAME, dtype=int)
    gutters = [STRIDE - 1 + STRIDE * i for i in range(MACRO_COUNT - 1)]
    grid[gutters, :] = BLACK
    grid[:, gutters] = BLACK

    center_rows = [1 + STRIDE * i for i in range(MACRO_COUNT)]
    center_cols = [1 + STRIDE * i for i in range(MACRO_COUNT)]
    for c, column in enumerate(columns):
        grid[center_rows, center_cols[c]] = np.asarray(column, dtype=int)
    return grid


def _sample_center_column(count: int) -> np.ndarray:
    column = np.full(MACRO_COUNT, BLACK, dtype=int)
    if count == 0:
        return column

    rows = random.sample(range(MACRO_COUNT), count)
    if count == 1:
        column[rows[0]] = random.choice((ORANGE, TEAL))
        return column

    colors = [ORANGE, TEAL]
    colors.extend(random.choice((ORANGE, TEAL)) for _ in range(count - 2))
    random.shuffle(colors)
    for row, color in zip(rows, colors):
        column[row] = color
    return column


def _all_permutations(require_fixed: bool, sorted_counts: tuple[int, ...]) -> list[tuple[int, ...]]:
    identity = tuple(range(MACRO_COUNT))
    choices = []
    for perm in itertools.permutations(range(MACRO_COUNT)):
        if perm == identity:
            continue
        fixed_points = [idx for idx in range(MACRO_COUNT) if perm[idx] == idx]
        has_visible_fixed = any(sorted_counts[idx] >= 2 for idx in fixed_points)
        if require_fixed:
            if has_visible_fixed:
                choices.append(tuple(int(x) for x in perm))
        else:
            if not fixed_points:
                choices.append(tuple(int(x) for x in perm))
    return choices


def _build_expected(sorted_columns: list[np.ndarray], placement: tuple[int, ...]) -> np.ndarray:
    source_pos = {rank: pos for pos, rank in enumerate(placement)}
    output_columns: list[np.ndarray] = []
    for rank, column in enumerate(sorted_columns):
        out_col = np.array(column, copy=True)
        if source_pos[rank] != rank:
            out_col = _swap_center_colors(out_col)
        output_columns.append(out_col)
    return _render_grid(output_columns)


def _sort_only_output(sorted_columns: list[np.ndarray]) -> np.ndarray:
    return _render_grid([np.array(column, copy=True) for column in sorted_columns])


def _flip_all_sorted_output(sorted_columns: list[np.ndarray]) -> np.ndarray:
    return _render_grid([_swap_center_colors(column) for column in sorted_columns])


def _global_color_coverage(columns: list[np.ndarray]) -> bool:
    flat = np.concatenate(columns)
    return ORANGE in flat and TEAL in flat


def _gen_attempt() -> np.ndarray | None:
    require_fixed = random.random() < 0.5
    sorted_counts = random.choices(COUNT_PROFILES, weights=COUNT_PROFILE_WEIGHTS, k=1)[0]
    permutations = _all_permutations(require_fixed, tuple(sorted_counts))
    if not permutations:
        return None

    sorted_columns = [_sample_center_column(int(count)) for count in sorted_counts]
    if not _global_color_coverage(sorted_columns):
        return None

    placement = random.choice(permutations)
    input_columns = [np.array(sorted_columns[rank], copy=True) for rank in placement]
    input_grid = _render_grid(input_columns)
    expected = _build_expected(sorted_columns, placement)
    pure_sorted = _sort_only_output(sorted_columns)
    flip_all_sorted = _flip_all_sorted_output(sorted_columns)

    if np.array_equal(input_grid, expected):
        return None
    if require_fixed:
        if np.array_equal(expected, pure_sorted) or np.array_equal(expected, flip_all_sorted):
            return None
    else:
        if not np.array_equal(expected, flip_all_sorted):
            return None

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid
