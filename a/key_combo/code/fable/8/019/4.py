from __future__ import annotations

import numpy as np


BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MIN_CELL_COUNT = 3
MAX_CELL_COUNT = 5
CELL_SIZE_CHOICES = (3, 4)
MAX_MARKERS = 3
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _interior_spans(lines: np.ndarray, limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines[:-1], lines[1:]):
        start = int(a) + 1
        end = int(b) - 1
        if start > end:
            return []
        spans.append((start, end))
    if not spans or int(lines[0]) != 0 or int(lines[-1]) != limit - 1:
        return []
    return spans


def _find_table_structure(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []

    for color in colors:
        mask = g == color
        row_lines = np.flatnonzero(mask.sum(axis=1) >= 2)
        col_lines = np.flatnonzero(mask.sum(axis=0) >= 2)
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue

        row_spans = _interior_spans(row_lines, g.shape[0])
        col_spans = _interior_spans(col_lines, g.shape[1])
        if not row_spans or not col_spans:
            continue

        line_mask = np.zeros_like(g, dtype=bool)
        line_mask[row_lines, :] = True
        line_mask[:, col_lines] = True

        if np.any(mask & ~line_mask):
            continue
        if np.any((g != bg) & line_mask & (g != color)):
            continue

        try:
            markers, masks = _cell_contents(g, bg, color, row_spans, col_spans)
            if not markers or not masks:
                continue
            for ri, cj, _active in masks:
                _unique_source_color(ri, cj, markers)
        except AssertionError:
            continue

        candidates.append((color, row_lines, col_lines))

    assert len(candidates) == 1, "expected exactly one sparse separator color"
    return candidates[0]


def _cell_contents(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int, np.ndarray]]]:
    markers: dict[tuple[int, int], int] = {}
    masks: list[tuple[int, int, np.ndarray]] = []

    for ri, (r0, r1) in enumerate(row_spans):
        for cj, (c0, c1) in enumerate(col_spans):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            active = (tile != bg) & (tile != line_color)
            active_count = int(active.sum())
            if active_count == 0:
                continue
            colors = [int(color) for color in np.unique(tile[active])]
            if active_count == 1:
                assert len(colors) == 1
                markers[(ri, cj)] = colors[0]
            else:
                assert len(colors) == 1, "mask cells should contain one mask color"
                masks.append((ri, cj, active))

    return markers, masks


def _unique_source_color(ri: int, cj: int, markers: dict[tuple[int, int], int]) -> int:
    hits = {
        cell: color
        for cell, color in markers.items()
        if cell[0] == ri or cell[1] == cj
    }
    assert len(hits) == 1, "each mask cell must align with exactly one marker cell"
    return next(iter(hits.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse separator traces into a full table, then recolor each mask region from its aligned marker cell.

    Input structure:
    - A black background contains sparse pixels of one separator color; any true separator row or column already has at least two such pixels.
    - The completed separator rows and columns would form a full table whose cells contain either a single-pixel colored marker, a mask-colored region, or only background.
    - Markers and masks are always placed strictly inside table cells, never on separator positions.

    Rule:
    - Find the unique color whose sparse pixels identify multiple separator rows and columns, and paint those entire rows and columns in that color.
    - For every cell containing a mask region, locate the unique marker cell that shares its table row or its table column.
    - Recolor the whole mask region to that marker's color while preserving the completed separators and the original marker pixels.

    Key insight:
    - The sparse separator pixels must be densified into the hidden table first; only then do row/column cell alignments reveal which marker color each mask region should inherit.
    """
    g = np.asarray(grid, dtype=int)
    line_color, row_lines, col_lines = _find_table_structure(g)
    bg = _mode_color(g)
    row_spans = _interior_spans(row_lines, g.shape[0])
    col_spans = _interior_spans(col_lines, g.shape[1])
    markers, masks = _cell_contents(g, bg, line_color, row_spans, col_spans)

    out = g.copy()
    out[row_lines, :] = line_color
    out[:, col_lines] = line_color

    for ri, cj, active in masks:
        color = _unique_source_color(ri, cj, markers)
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[cj]
        tile = out[r0 : r1 + 1, c0 : c1 + 1]
        tile[active] = color

    return out


def _line_positions(sizes: list[int]) -> tuple[int, list[int], list[tuple[int, int]]]:
    cursor = 0
    lines = [0]
    spans: list[tuple[int, int]] = []
    for size in sizes:
        start = cursor + 1
        end = cursor + size
        spans.append((start, end))
        cursor = end + 1
        lines.append(cursor)
    return cursor + 1, lines, spans


def _sample_sparse_intersections(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    while True:
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        for r in range(n_rows):
            chosen = rng.choice(n_cols, size=2, replace=False)
            mask[r, chosen] = True
        for c in range(n_cols):
            chosen = rng.choice(n_rows, size=2, replace=False)
            mask[chosen, c] = True
        mask |= rng.random((n_rows, n_cols)) < rng.uniform(0.05, 0.3)
        if np.all(mask.sum(axis=1) >= 2) and np.all(mask.sum(axis=0) >= 2):
            return mask


def _random_horizontal_mask(
    rng: np.random.Generator,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
) -> tuple[int, int, int]:
    r0, r1 = row_span
    c0, c1 = col_span
    width = c1 - c0 + 1
    length = int(rng.integers(2, min(3, width) + 1))
    rr = int(rng.integers(r0, r1 + 1))
    cc = int(rng.integers(c0, c1 - length + 2))
    return rr, cc, length


def _marker_pixel(
    rng: np.random.Generator,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
) -> tuple[int, int]:
    r0, r1 = row_span
    c0, c1 = col_span
    rr = int(rng.integers(r0, r1 + 1))
    cc = int(rng.integers(c0, c1 + 1))
    return rr, cc


def _expected_output(
    inp: np.ndarray,
    line_color: int,
    line_rows: list[int],
    line_cols: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    marker_cells: dict[tuple[int, int], int],
    mask_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    out[line_rows, :] = line_color
    out[:, line_cols] = line_color

    for ri, cj in mask_cells:
        source = _unique_source_color(ri, cj, marker_cells)
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[cj]
        mask = inp[r0 : r1 + 1, c0 : c1 + 1] != BACKGROUND
        tile = out[r0 : r1 + 1, c0 : c1 + 1]
        tile[mask] = source

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_cell_rows = int(rng.integers(MIN_CELL_COUNT, MAX_CELL_COUNT + 1))
    n_cell_cols = int(rng.integers(MIN_CELL_COUNT, MAX_CELL_COUNT + 1))
    row_sizes = rng.choice(CELL_SIZE_CHOICES, size=n_cell_rows, replace=True).tolist()
    col_sizes = rng.choice(CELL_SIZE_CHOICES, size=n_cell_cols, replace=True).tolist()

    height, line_rows, row_spans = _line_positions(row_sizes)
    width, line_cols, col_spans = _line_positions(col_sizes)
    if height > 30 or width > 30:
        return None

    max_markers = min(MAX_MARKERS, n_cell_rows - 1, n_cell_cols - 1)
    if max_markers < 1:
        return None
    marker_count = int(rng.integers(1, max_markers + 1))

    palette = rng.choice(COLOR_POOL, size=marker_count + 2, replace=False).tolist()
    line_color = int(palette[0])
    mask_color = int(palette[1])
    marker_colors = [int(color) for color in palette[2:]]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    intersections = _sample_sparse_intersections(rng, len(line_rows), len(line_cols))
    for rr, cc in np.argwhere(intersections):
        grid[line_rows[int(rr)], line_cols[int(cc)]] = line_color

    marked_rows = sorted(rng.choice(n_cell_rows, size=marker_count, replace=False).tolist())
    marked_cols = sorted(rng.choice(n_cell_cols, size=marker_count, replace=False).tolist())
    marked_row_set = set(marked_rows)
    marked_col_set = set(marked_cols)
    col_perm = rng.permutation(marker_count)
    marker_cells: dict[tuple[int, int], int] = {}
    for idx, row_idx in enumerate(marked_rows):
        col_idx = marked_cols[int(col_perm[idx])]
        marker_cells[(int(row_idx), int(col_idx))] = marker_colors[idx]

    for (ri, cj), color in marker_cells.items():
        rr, cc = _marker_pixel(rng, row_spans[ri], col_spans[cj])
        grid[rr, cc] = color

    unmarked_rows = [ri for ri in range(n_cell_rows) if ri not in marked_row_set]
    unmarked_cols = [cj for cj in range(n_cell_cols) if cj not in marked_col_set]
    if not unmarked_rows or not unmarked_cols:
        return None

    row_mask_count = int(rng.integers(1, min(3, len(unmarked_cols)) + 1))
    col_mask_count = int(rng.integers(1, min(3, len(marked_cols)) + 1))
    row_mask_cols = rng.choice(unmarked_cols, size=row_mask_count, replace=False).tolist()
    col_mask_cols = rng.choice(marked_cols, size=col_mask_count, replace=False).tolist()

    mask_cells: set[tuple[int, int]] = set()
    for col_idx in row_mask_cols:
        row_idx = int(rng.choice(marked_rows))
        mask_cells.add((row_idx, int(col_idx)))
    for col_idx in col_mask_cols:
        row_idx = int(rng.choice(unmarked_rows))
        mask_cells.add((row_idx, int(col_idx)))

    for ri, cj in mask_cells:
        rr, cc, length = _random_horizontal_mask(rng, row_spans[ri], col_spans[cj])
        grid[rr, cc : cc + length] = mask_color

    expected = _expected_output(
        grid,
        line_color,
        line_rows,
        line_cols,
        row_spans,
        col_spans,
        marker_cells,
        mask_cells,
    )

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-table ARC input")
