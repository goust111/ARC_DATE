import random
from typing import Sequence

import numpy as np


BG = 0
GRID = 5
ACTIVE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

ROW_RANGE = (5, 8)
COL_RANGE = (5, 8)
CELL_SIZE_RANGE = (1, 3)
PALETTE_SIZES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [idx for idx in range(grid.shape[0]) if np.all(grid[idx, :] == color)]
    return [idx for idx in range(grid.shape[1]) if np.all(grid[:, idx] == color)]


def _is_regular_line_set(lines: Sequence[int], limit: int) -> bool:
    if len(lines) < 3 or lines[0] != 0 or lines[-1] != limit - 1:
        return False
    gaps = np.diff(lines)
    return len(set(map(int, gaps))) == 1 and int(gaps[0]) >= 2


def _find_grid_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        row_lines = _line_indices(grid, color, axis=0)
        col_lines = _line_indices(grid, color, axis=1)
        if not _is_regular_line_set(row_lines, grid.shape[0]):
            continue
        if not _is_regular_line_set(col_lines, grid.shape[1]):
            continue
        row_spans = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1)]
        col_spans = [(col_lines[i] + 1, col_lines[i + 1]) for i in range(len(col_lines) - 1)]
        if min(r1 - r0 for r0, r1 in row_spans) <= 0:
            continue
        if min(c1 - c0 for c0, c1 in col_spans) <= 0:
            continue
        candidates.append((len(row_spans) * len(col_spans), color, row_spans, col_spans))

    assert candidates, "expected visible grey grid lines"
    _, grid_color, row_spans, col_spans = max(candidates, key=lambda item: item[0])
    return grid_color, row_spans, col_spans


def _extract_cells(
    grid: np.ndarray, row_spans: Sequence[tuple[int, int]], col_spans: Sequence[tuple[int, int]]
) -> np.ndarray:
    cells = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = grid[r0:r1, c0:c1]
            values = np.unique(block)
            assert len(values) == 1, "each small cell should have a uniform fill"
            cells[r, c] = int(values[0])
    return cells


def _render_cells(
    base: np.ndarray,
    cells: np.ndarray,
    row_spans: Sequence[tuple[int, int]],
    col_spans: Sequence[tuple[int, int]],
) -> np.ndarray:
    out = np.array(base, copy=True)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0:r1, c0:c1] = int(cells[r, c])
    return out


def _logical_fill(cells: np.ndarray, sample_row: int, sample_col: int) -> np.ndarray:
    out = np.array(cells, copy=True)
    row_colors = cells[:, sample_col]
    col_colors = cells[sample_row, :]
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            if out[r, c] != BG:
                continue
            if row_colors[r] != BG:
                out[r, c] = int(row_colors[r])
            elif col_colors[c] != BG:
                out[r, c] = int(col_colors[c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate sample row/column colors across the whole gridded scene."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, row_spans, col_spans = _find_grid_structure(g)
    cells = _extract_cells(g, row_spans, col_spans)

    colored = cells != BG
    row_counts = np.count_nonzero(colored, axis=1)
    col_counts = np.count_nonzero(colored, axis=0)
    assert row_counts.max() > 0 and col_counts.max() > 0, "expected colored sample row and column"

    sample_rows = np.flatnonzero(row_counts == row_counts.max())
    sample_cols = np.flatnonzero(col_counts == col_counts.max())
    assert len(sample_rows) == 1, "sample row should be uniquely densest"
    assert len(sample_cols) == 1, "sample column should be uniquely densest"

    sample_row = int(sample_rows[0])
    sample_col = int(sample_cols[0])
    out_cells = _logical_fill(cells, sample_row, sample_col)
    return _render_cells(g, out_cells, row_spans, col_spans)


def _resolved_color(row_color: int, col_color: int) -> int:
    if row_color != BG:
        return int(row_color)
    if col_color != BG:
        return int(col_color)
    return BG


def _sample_sequences(
    rng: random.Random, rows: int, cols: int, sample_row: int, sample_col: int, palette: Sequence[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    min_row_density = max(3, rows // 2)
    min_col_density = max(3, cols // 2)

    for _ in range(300):
        row_zero_prob = rng.choice((0.3, 0.4, 0.5))
        col_zero_prob = rng.choice((0.3, 0.4, 0.5))
        intersection = int(rng.choice(palette))

        row_colors = np.full(rows, BG, dtype=int)
        col_colors = np.full(cols, BG, dtype=int)
        row_colors[sample_row] = intersection
        col_colors[sample_col] = intersection

        for r in range(rows):
            if r == sample_row:
                continue
            if rng.random() > row_zero_prob:
                row_colors[r] = int(rng.choice(palette))

        for c in range(cols):
            if c == sample_col:
                continue
            if rng.random() > col_zero_prob:
                col_colors[c] = int(rng.choice(palette))

        if int(np.count_nonzero(row_colors)) < min_row_density:
            continue
        if int(np.count_nonzero(col_colors)) < min_col_density:
            continue

        used_colors = set(map(int, row_colors[row_colors != BG])) | set(map(int, col_colors[col_colors != BG]))
        if len(used_colors) != len(set(palette)):
            continue

        other_rows = [r for r in range(rows) if r != sample_row]
        other_cols = [c for c in range(cols) if c != sample_col]

        has_both = any(row_colors[r] != BG and col_colors[c] != BG for r in other_rows for c in other_cols)
        has_row_only = any(row_colors[r] != BG and col_colors[c] == BG for r in other_rows for c in other_cols)
        has_col_only = any(row_colors[r] == BG and col_colors[c] != BG for r in other_rows for c in other_cols)
        has_neither = any(row_colors[r] == BG and col_colors[c] == BG for r in other_rows for c in other_cols)
        has_conflict = any(
            row_colors[r] != BG and col_colors[c] != BG and row_colors[r] != col_colors[c]
            for r in other_rows
            for c in other_cols
        )
        if has_both and has_row_only and has_col_only and has_neither and has_conflict:
            return row_colors, col_colors

    return None


def _choose_clues(
    rng: random.Random, row_colors: np.ndarray, col_colors: np.ndarray, sample_row: int, sample_col: int
) -> list[tuple[int, int]] | None:
    rows, cols = row_colors.shape[0], col_colors.shape[0]
    conflict = []
    row_only = []
    col_only = []
    both = []
    neither = []

    for r in range(rows):
        if r == sample_row:
            continue
        for c in range(cols):
            if c == sample_col:
                continue
            rr = int(row_colors[r])
            cc = int(col_colors[c])
            if rr != BG and cc != BG:
                both.append((r, c))
                if rr != cc:
                    conflict.append((r, c))
            elif rr != BG:
                row_only.append((r, c))
            elif cc != BG:
                col_only.append((r, c))
            else:
                neither.append((r, c))

    rng.shuffle(conflict)
    rng.shuffle(row_only)
    rng.shuffle(col_only)

    for a in conflict:
        used_rows = {a[0]}
        used_cols = {a[1]}
        for b in row_only:
            if b[0] in used_rows or b[1] in used_cols:
                continue
            used_rows_2 = used_rows | {b[0]}
            used_cols_2 = used_cols | {b[1]}
            for c in col_only:
                if c[0] in used_rows_2 or c[1] in used_cols_2:
                    continue
                remaining = {
                    "both": [pos for pos in both if pos not in (a, b, c)],
                    "row_only": [pos for pos in row_only if pos not in (a, b, c)],
                    "col_only": [pos for pos in col_only if pos not in (a, b, c)],
                    "neither": list(neither),
                }
                if all(remaining[key] for key in ("both", "row_only", "col_only", "neither")):
                    return [a, b, c]

    return None


def _build_input_cells(
    row_colors: np.ndarray,
    col_colors: np.ndarray,
    sample_row: int,
    sample_col: int,
    clues: Sequence[tuple[int, int]],
) -> np.ndarray:
    cells = np.full((len(row_colors), len(col_colors)), BG, dtype=int)
    cells[sample_row, :] = col_colors
    cells[:, sample_col] = row_colors
    for r, c in clues:
        cells[r, c] = _resolved_color(int(row_colors[r]), int(col_colors[c]))
    return cells


def _unique_dense_axes(cells: np.ndarray, sample_row: int, sample_col: int) -> bool:
    row_counts = np.count_nonzero(cells != BG, axis=1)
    col_counts = np.count_nonzero(cells != BG, axis=0)
    row_max = int(row_counts.max())
    col_max = int(col_counts.max())
    return (
        int(row_counts[sample_row]) == row_max
        and int(col_counts[sample_col]) == col_max
        and int(np.sum(row_counts == row_max)) == 1
        and int(np.sum(col_counts == col_max)) == 1
    )


def _render_input(cells: np.ndarray, cell_size: int) -> np.ndarray:
    rows, cols = cells.shape
    height = rows * cell_size + rows + 1
    width = cols * cell_size + cols + 1
    grid = np.full((height, width), GRID, dtype=int)
    for r in range(rows):
        r0 = 1 + r * (cell_size + 1)
        r1 = r0 + cell_size
        for c in range(cols):
            c0 = 1 + c * (cell_size + 1)
            c1 = c0 + cell_size
            grid[r0:r1, c0:c1] = int(cells[r, c])
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows = rng.randint(*ROW_RANGE)
    cols = rng.randint(*COL_RANGE)
    cell_size = rng.randint(*CELL_SIZE_RANGE)
    sample_row = rng.randint(1, rows - 2)
    sample_col = rng.randint(1, cols - 2)
    palette_size = rng.choice(PALETTE_SIZES)
    palette = rng.sample(list(ACTIVE_POOL), palette_size)

    sequences = _sample_sequences(rng, rows, cols, sample_row, sample_col, palette)
    if sequences is None:
        return None
    row_colors, col_colors = sequences

    clues = _choose_clues(rng, row_colors, col_colors, sample_row, sample_col)
    if clues is None:
        return None

    cells = _build_input_cells(row_colors, col_colors, sample_row, sample_col, clues)
    if not _unique_dense_axes(cells, sample_row, sample_col):
        return None

    rendered = _render_input(cells, cell_size)

    # Validate against the solver so the generator only emits consistent scenes.
    expected_cells = _logical_fill(cells, sample_row, sample_col)
    expected = _render_input(expected_cells, cell_size)
    if not np.array_equal(main(rendered), expected):
        return None

    return rendered


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
