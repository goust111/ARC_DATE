import random
from collections import Counter

import numpy as np


def _group_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for raw in indices[1:]:
        cur = int(raw)
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def _intervals_between_bands(length: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in bands:
        if start > prev_end + 1:
            intervals.append((prev_end + 1, start - 1))
        prev_end = end
    if prev_end < length - 1:
        intervals.append((prev_end + 1, length - 1))
    return [(start, end) for start, end in intervals if start <= end]


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    _, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    c = 0
    while c < w:
        color = int(grid[0, c])
        if not np.all(grid[:, c] == color):
            c += 1
            continue
        start = c
        c += 1
        while c < w and np.all(grid[:, c] == color):
            c += 1
        end = c - 1
        left_w = start
        right_w = w - end - 1
        if left_w > 0 and left_w == right_w:
            candidates.append((start, end, color))
    assert candidates, "no equal-width vertical separator found"
    candidates.sort(key=lambda item: (item[1] - item[0] + 1, item[0]), reverse=True)
    return candidates[0]


def _extract_grid(panel: np.ndarray) -> dict:
    best = None
    for raw_color in np.unique(panel):
        color = int(raw_color)
        row_bands = _group_runs(np.flatnonzero(np.all(panel == color, axis=1)))
        col_bands = _group_runs(np.flatnonzero(np.all(panel == color, axis=0)))
        if not row_bands or not col_bands:
            continue
        row_cells = _intervals_between_bands(panel.shape[0], row_bands)
        col_cells = _intervals_between_bands(panel.shape[1], col_bands)
        if not row_cells or not col_cells:
            continue
        row_sizes = {end - start + 1 for start, end in row_cells}
        col_sizes = {end - start + 1 for start, end in col_cells}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        cell_size = next(iter(row_sizes))
        if cell_size <= 0 or cell_size != next(iter(col_sizes)):
            continue
        row_line_sizes = {end - start + 1 for start, end in row_bands}
        col_line_sizes = {end - start + 1 for start, end in col_bands}
        score = (
            len(row_cells) * len(col_cells),
            len(row_bands) + len(col_bands),
            -abs(len(row_line_sizes) - 1) - abs(len(col_line_sizes) - 1),
        )
        candidate = {
            "grid_color": color,
            "row_cells": row_cells,
            "col_cells": col_cells,
            "cell_size": cell_size,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    assert best is not None, "could not recover template grid"
    return best


def _build_index_map(length: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    index_map = np.full(length, -1, dtype=int)
    for index, (start, end) in enumerate(intervals):
        index_map[start : end + 1] = index
    return index_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use control-panel marker coordinates to color template-panel cells.

    Input structure:
    - A vertical separator splits the grid into equal left/right panels.
    - The left panel is a blank square-cell template with visible grid lines.
    - The right panel has sparse single-pixel colored markers.

    Rule:
    - Each marker's row and column, measured in template-cell coordinates,
      selects one cell in the left panel.
    - Fill that template cell's interior with the marker color.
    - If several markers resolve to the same cell, the last marker in
      top-to-bottom, left-to-right order wins.
    - Separator and grid lines stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    sep_start, sep_end, _ = _find_separator(g)
    left = g[:, :sep_start]
    right = g[:, sep_end + 1 :]
    assert left.shape == right.shape, "panels must be equal-sized"

    grid_info = _extract_grid(left)
    row_cells = grid_info["row_cells"]
    col_cells = grid_info["col_cells"]
    row_map = _build_index_map(left.shape[0], row_cells)
    col_map = _build_index_map(left.shape[1], col_cells)

    valid_mask = (row_map[:, None] >= 0) & (col_map[None, :] >= 0)
    control_values = right[valid_mask]
    assert control_values.size > 0, "control panel has no interior"
    control_bg = Counter(map(int, control_values.tolist())).most_common(1)[0][0]

    out = g.copy()
    for r in range(right.shape[0]):
        row_index = int(row_map[r])
        if row_index < 0:
            continue
        for c in range(right.shape[1]):
            col_index = int(col_map[c])
            if col_index < 0:
                continue
            color = int(right[r, c])
            if color == control_bg:
                continue
            r0, r1 = row_cells[row_index]
            c0, c1 = col_cells[col_index]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


MAX_RETRIES = 300
GRID_HEIGHT_LIMIT = 30
GRID_WIDTH_LIMIT = 30


def _sample_geometry() -> tuple[int, int, int, int, int]:
    for _ in range(200):
        rows = random.randint(2, 5)
        cols = random.randint(2, 4)
        cell_size = random.randint(2, 4)
        line_size = random.randint(1, 2)
        sep_size = random.randint(1, 2)
        panel_h = rows * cell_size + (rows + 1) * line_size
        panel_w = cols * cell_size + (cols + 1) * line_size
        if panel_h <= GRID_HEIGHT_LIMIT and 2 * panel_w + sep_size <= GRID_WIDTH_LIMIT:
            return rows, cols, cell_size, line_size, sep_size
    raise RuntimeError("failed to sample geometry")


def _make_panel(rows: int, cols: int, cell_size: int, line_size: int, line_color: int, fill_color: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    panel_h = rows * cell_size + (rows + 1) * line_size
    panel_w = cols * cell_size + (cols + 1) * line_size
    panel = np.full((panel_h, panel_w), line_color, dtype=int)

    row_cells: list[tuple[int, int]] = []
    col_cells: list[tuple[int, int]] = []

    cursor = line_size
    for _ in range(rows):
        row_cells.append((cursor, cursor + cell_size - 1))
        cursor += cell_size + line_size

    cursor = line_size
    for _ in range(cols):
        col_cells.append((cursor, cursor + cell_size - 1))
        cursor += cell_size + line_size

    for r0, r1 in row_cells:
        for c0, c1 in col_cells:
            panel[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return panel, row_cells, col_cells


def _place_markers(
    control: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    palette: list[int],
    allow_overwrite: bool,
) -> list[tuple[int, int, int, int, int]]:
    rows = len(row_cells)
    cols = len(col_cells)
    logical_cells = [(r, c) for r in range(rows) for c in range(cols)]
    max_active = min(len(logical_cells), max(1, rows + cols - 1))
    active_count = random.randint(1, max_active)
    active_cells = random.sample(logical_cells, active_count)

    markers_per_cell = {cell: 1 for cell in active_cells}
    overwrite_cell = None
    cell_area = (row_cells[0][1] - row_cells[0][0] + 1) * (col_cells[0][1] - col_cells[0][0] + 1)
    if allow_overwrite and cell_area >= 2:
        overwrite_cell = random.choice(active_cells)
        markers_per_cell[overwrite_cell] = random.randint(2, min(3, cell_area))
        extra_dup_cells = [cell for cell in active_cells if cell != overwrite_cell]
        if extra_dup_cells and random.random() < 0.4:
            other = random.choice(extra_dup_cells)
            markers_per_cell[other] = random.randint(2, min(2, cell_area))

    placed: list[tuple[int, int, int, int, int]] = []
    for cell in active_cells:
        rr, cc = cell
        r0, r1 = row_cells[rr]
        c0, c1 = col_cells[cc]
        positions = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        chosen_positions = random.sample(positions, markers_per_cell[cell])
        colors = [random.choice(palette) for _ in chosen_positions]
        if cell == overwrite_cell and len(colors) >= 2 and len(palette) >= 2 and len(set(colors)) == 1:
            alternatives = [color for color in palette if color != colors[0]]
            colors[-1] = random.choice(alternatives)
        for (r, c), color in zip(chosen_positions, colors):
            control[r, c] = color
            placed.append((r, c, color, rr, cc))
    return placed

def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_size, line_size, sep_size = _sample_geometry()
    show_control_grid = random.random() < 0.5
    allow_overwrite = random.random() < 0.5

    colors = list(range(10))
    random.shuffle(colors)
    bg_color, line_color, sep_color = colors[:3]
    palette = colors[3 : 3 + random.randint(2, 5)]

    left, row_cells, col_cells = _make_panel(rows, cols, cell_size, line_size, line_color, bg_color)
    if show_control_grid:
        right, _, _ = _make_panel(rows, cols, cell_size, line_size, line_color, bg_color)
    else:
        right = np.full_like(left, bg_color)

    markers = _place_markers(right, row_cells, col_cells, palette, allow_overwrite)
    if not markers:
        return None

    sep = np.full((left.shape[0], sep_size), sep_color, dtype=int)
    grid = np.concatenate([left, sep, right], axis=1)

    expected = grid.copy()
    for _, _, color, row_index, col_index in sorted(markers, key=lambda item: (item[0], item[1])):
        r0, r1 = row_cells[row_index]
        c0, c1 = col_cells[col_index]
        expected[r0 : r1 + 1, c0 : c1 + 1] = color

    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(produced, grid):
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
