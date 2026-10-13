from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
SEP = 5
FG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _line_groups(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    if not indices:
        return groups
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _bands_from_lines(indices: list[int]) -> list[tuple[int, int]]:
    groups = _line_groups(sorted(indices))
    bands: list[tuple[int, int]] = []
    for (_, end_a), (start_b, _) in zip(groups, groups[1:]):
        r0 = int(end_a + 1)
        r1 = int(start_b - 1)
        if r0 <= r1:
            bands.append((r0, r1))
    return bands


def _majority(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.tolist())
    best = counts.most_common()
    assert best, "expected at least one value"
    if len(best) >= 2:
        assert best[0][1] > best[1][1], "expected a unique majority value"
    return int(best[0][0])


def _grid_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_hits: dict[int, list[int]] = defaultdict(list)
    col_hits: dict[int, list[int]] = defaultdict(list)
    for r in range(h):
        if np.all(g[r, :] == g[r, 0]):
            row_hits[int(g[r, 0])].append(int(r))
    for c in range(w):
        if np.all(g[:, c] == g[0, c]):
            col_hits[int(g[0, c])].append(int(c))

    candidates = [
        color
        for color in row_hits
        if len(row_hits[color]) >= 2 and len(col_hits.get(color, [])) >= 2
    ]
    assert len(candidates) == 1, "expected one separator color with full rows and columns"
    sep = int(candidates[0])

    row_bands = _bands_from_lines(row_hits[sep])
    col_bands = _bands_from_lines(col_hits[sep])
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected at least a 2x2 cell grid"

    row_heights = {int(r1 - r0 + 1) for r0, r1 in row_bands}
    col_widths = {int(c1 - c0 + 1) for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "cell sizes must be uniform"
    return sep, row_bands, col_bands


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep, row_bands, col_bands = _grid_structure(g)
    cell_h = int(row_bands[0][1] - row_bands[0][0] + 1)
    cell_w = int(col_bands[0][1] - col_bands[0][0] + 1)

    cells: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cells.append(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy())
    stack = np.stack(cells, axis=0)

    consensus = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            consensus[r, c] = _majority(stack[:, r, c])

    mismatch_counts = np.array([int(np.sum(cell != consensus)) for cell in cells], dtype=int)
    odd_cells = np.flatnonzero(mismatch_counts > 0)
    assert odd_cells.size == 1, "expected exactly one anomalous cell"

    odd_index = int(odd_cells[0])
    diff_positions = np.argwhere(cells[odd_index] != consensus)
    assert 1 <= len(diff_positions) <= 2, "expected one or two corrupted pixels"

    error_colors = {int(cells[odd_index][int(r), int(c)]) for r, c in diff_positions}
    assert len(error_colors) == 1, "all corrupted pixels must share one error color"
    error_color = int(next(iter(error_colors)))

    out = g.copy()
    odd_row = odd_index // len(col_bands)
    odd_col = odd_index % len(col_bands)
    cell_r0, _cell_r1 = row_bands[odd_row]
    cell_c0, _cell_c1 = col_bands[odd_col]

    for r, c in diff_positions:
        out[int(cell_r0) + int(r), int(cell_c0) + int(c)] = int(consensus[int(r), int(c)])

    band_r0, band_r1 = row_bands[odd_row]
    band_c0, band_c1 = col_bands[odd_col]
    mid_r = int((band_r0 + band_r1) // 2)
    mid_c = int((band_c0 + band_c1) // 2)
    out[0, mid_c] = error_color
    out[-1, mid_c] = error_color
    out[mid_r, 0] = error_color
    out[mid_r, -1] = error_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the lone corrupted repeated cell pattern and move its wrong color to four border markers.

    Input structure:
    - Black is the background inside a rectangular grey lattice with a one-cell outer frame.
    - The grey rows and columns partition the grid into equal-sized rectangular cells.
    - Every cell contains the same multi-color pixel pattern except one cell, where one or two pixels use a wrong color.

    Rule:
    - Identify the separator rows and columns, crop every cell, and compare corresponding pixel positions across all cells.
    - At each cell-relative position, take the majority color as the consensus pattern; exactly one cell disagrees at one or two positions.
    - Recolor those corrupted pixels in the anomalous cell back to the consensus colors.
    - Reuse the anomalous cell's original wrong color as four border markers: top and bottom at the midpoint of that cell's column span, and left and right at the midpoint of its row span.

    Key insight:
    - The anomaly is detected by cross-cell majority voting, and the discarded wrong color is not removed but relocated onto the outer frame to encode the anomalous cell's row and column.
    """
    return _solve(np.asarray(grid, dtype=int))


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _random_connected_cells(
    h: int, w: int, size: int, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    if len(free) < size:
        return None

    for _ in range(80):
        start = random.choice(free)
        cells = [start]
        used = {start}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors(r, c, h, w):
                    if (nr, nc) in blocked or (nr, nc) in used:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            pick = random.choice(frontier)
            used.add(pick)
            cells.append(pick)
        if len(cells) == size:
            return cells
    return None


def _build_template(cell_h: int, cell_w: int) -> tuple[np.ndarray, list[int]] | None:
    area = int(cell_h * cell_w)
    max_fill = max(4, min(area - 1, area // 2 + 2))
    min_fill = 3
    if max_fill < min_fill:
        return None

    for _ in range(120):
        n_colors = random.randint(2, min(4, len(FG_COLORS)))
        palette = random.sample(FG_COLORS, n_colors)
        fill_total = random.randint(max(n_colors + 1, min_fill), max_fill)

        sizes = [2] + [1] * (n_colors - 1)
        extra = int(fill_total - sum(sizes))
        for _ in range(extra):
            sizes[random.randrange(n_colors)] += 1

        tail = sizes[1:]
        random.shuffle(tail)
        sizes = [sizes[0]] + tail

        template = np.full((cell_h, cell_w), BG, dtype=int)
        occupied: set[tuple[int, int]] = set()
        color_counts: dict[int, int] = {}
        ok = True
        for color, size in zip(palette, sizes):
            cells = _random_connected_cells(cell_h, cell_w, int(size), occupied)
            if cells is None:
                ok = False
                break
            for r, c in cells:
                template[int(r), int(c)] = int(color)
                occupied.add((int(r), int(c)))
            color_counts[int(color)] = int(size)

        if not ok:
            continue
        if len(occupied) != fill_total:
            continue
        if not np.any(template == BG):
            continue
        if len(color_counts) < 2:
            continue
        if max(color_counts.values()) < 2:
            continue
        return template, palette
    return None


def _expected_output(
    grid: np.ndarray,
    template: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    odd_row: int,
    odd_col: int,
    diff_positions: list[tuple[int, int]],
    error_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    cell_r0, _ = row_bands[odd_row]
    cell_c0, _ = col_bands[odd_col]
    for r, c in diff_positions:
        out[int(cell_r0) + int(r), int(cell_c0) + int(c)] = int(template[int(r), int(c)])

    band_r0, band_r1 = row_bands[odd_row]
    band_c0, band_c1 = col_bands[odd_col]
    mid_r = int((band_r0 + band_r1) // 2)
    mid_c = int((band_c0 + band_c1) // 2)
    out[0, mid_c] = int(error_color)
    out[-1, mid_c] = int(error_color)
    out[mid_r, 0] = int(error_color)
    out[mid_r, -1] = int(error_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 5)
    n_cols = random.randint(2, 5)

    max_cell_h = (30 - (n_rows + 1)) // n_rows
    max_cell_w = (30 - (n_cols + 1)) // n_cols
    odd_heights = [size for size in range(3, max_cell_h + 1, 2)]
    odd_widths = [size for size in range(3, max_cell_w + 1, 2)]
    if not odd_heights or not odd_widths:
        return None

    cell_h = random.choice(odd_heights)
    cell_w = random.choice(odd_widths)
    built = _build_template(cell_h, cell_w)
    if built is None:
        return None
    template, palette = built

    available_error_colors = [color for color in FG_COLORS if color not in palette]
    if not available_error_colors:
        return None
    error_color = int(random.choice(available_error_colors))

    h = int(n_rows * cell_h + (n_rows + 1))
    w = int(n_cols * cell_w + (n_cols + 1))
    grid = np.full((h, w), BG, dtype=int)

    sep_rows = [0] + [int(1 + k * (cell_h + 1) - 1) for k in range(1, n_rows + 1)]
    sep_cols = [0] + [int(1 + k * (cell_w + 1) - 1) for k in range(1, n_cols + 1)]
    for r in sep_rows:
        grid[int(r), :] = SEP
    for c in sep_cols:
        grid[:, int(c)] = SEP

    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []
    for i in range(n_rows):
        r0 = int(1 + i * (cell_h + 1))
        r1 = int(r0 + cell_h - 1)
        row_bands.append((r0, r1))
    for j in range(n_cols):
        c0 = int(1 + j * (cell_w + 1))
        c1 = int(c0 + cell_w - 1)
        col_bands.append((c0, c1))

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = template

    diff_count = random.choice([1, 2])
    color_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(cell_h):
        for c in range(cell_w):
            value = int(template[r, c])
            if value != BG:
                color_positions[value].append((int(r), int(c)))

    candidates = [color for color, pts in color_positions.items() if len(pts) >= diff_count]
    if not candidates:
        return None
    source_color = int(random.choice(candidates))
    diff_positions = random.sample(color_positions[source_color], diff_count)

    odd_row = random.randrange(n_rows)
    odd_col = random.randrange(n_cols)
    cell_r0, _ = row_bands[odd_row]
    cell_c0, _ = col_bands[odd_col]
    for r, c in diff_positions:
        grid[int(cell_r0) + int(r), int(cell_c0) + int(c)] = error_color

    expected = _expected_output(
        grid,
        template,
        row_bands,
        col_bands,
        odd_row,
        odd_col,
        diff_positions,
        error_color,
    )

    try:
        solved = _solve(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input for puzzle 7")
