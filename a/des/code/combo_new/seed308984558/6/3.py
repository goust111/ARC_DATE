from __future__ import annotations

import random

import numpy as np


BG = 0
GRID = 8
LEGEND = 5
DATA_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 300


def _most_common(values: np.ndarray) -> int:
    vals, cnts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _find_grid_structure(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], int, np.ndarray, np.ndarray] | None = None

    for color in np.unique(grid):
        rows = np.flatnonzero(np.all(grid == int(color), axis=1))
        cols = np.flatnonzero(np.all(grid == int(color), axis=0))
        if len(rows) < 3 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue

        row_steps = np.diff(rows)
        col_steps = np.diff(cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) < 2 or int(col_steps[0]) < 2:
            continue
        if not np.array_equal(rows, np.arange(0, h, int(row_steps[0]))):
            continue
        if not np.array_equal(cols, np.arange(0, w, int(col_steps[0]))):
            continue

        row_cells = len(rows) - 1
        col_cells = len(cols) - 1
        score = (row_cells * col_cells, row_cells + col_cells, int(color))
        if best is None or score > best[0]:
            best = (score, int(color), rows, cols)

    if best is None:
        raise AssertionError("failed to detect the separator grid")
    return best[1], best[2], best[3]


def _background_color(grid: np.ndarray, line_color: int) -> int:
    non_line = grid[grid != int(line_color)]
    if non_line.size == 0:
        raise AssertionError("grid cannot be entirely separator color")
    return _most_common(non_line)


def _cell_bounds(lines: np.ndarray, index: int) -> tuple[int, int]:
    return int(lines[index] + 1), int(lines[index + 1])


def _connected_pattern(cell_h: int, cell_w: int, size: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    start = random.choice(cells)
    pattern = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and (nr, nc) not in pattern:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(pattern) < size:
        if not frontier:
            raise AssertionError("pattern growth stalled")
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        pattern.add(nxt)
        add_frontier(*nxt)
    return sorted(pattern)


def _legend_patterns(
    grid: np.ndarray, row_lines: np.ndarray, col_lines: np.ndarray, bg: int
) -> dict[int, list[tuple[int, int]]]:
    patterns: dict[int, list[tuple[int, int]]] = {}
    r0, r1 = _cell_bounds(row_lines, 0)
    for idx in range(len(col_lines) - 1):
        c0, c1 = _cell_bounds(col_lines, idx)
        cell = grid[r0:r1, c0:c1]
        coords = [tuple(map(int, rc)) for rc in np.argwhere(cell != int(bg))]
        count = idx + 1
        if len(coords) != count:
            raise AssertionError("legend cells must be ordered by pixel count")
        patterns[count] = coords
    return patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each seeded data cell with the legend pattern indexed by its seed count.

    Input structure:
    - The background is black, and a single-pixel-thick separator color forms a
      regular rectangular grid with equal spacing between all horizontal lines and
      all vertical lines.
    - The top row of cells is a legend row: the kth legend cell from the left
      contains exactly k grey pixels arranged as that cell's template.
    - Every cell below the legend is either empty black or contains several
      same-colored seed pixels that are neither background nor separator.

    Rule:
    - Detect the separator grid and treat the first cell row as the legend.
    - For each non-empty data cell, count its seed pixels N and read the local
      template coordinates from the Nth legend cell.
    - Clear the data cell, then redraw that template in the data cell using the
      seed color; empty cells and the legend row stay unchanged.

    Key insight:
    - The legend row is a visible lookup table from seed-count to template shape,
      while each data cell supplies only the lookup index N and the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    line_color, row_lines, col_lines = _find_grid_structure(g)
    bg = _background_color(g, line_color)
    patterns = _legend_patterns(g, row_lines, col_lines, bg)
    out = g.copy()

    for row_idx in range(1, len(row_lines) - 1):
        r0, r1 = _cell_bounds(row_lines, row_idx)
        for col_idx in range(len(col_lines) - 1):
            c0, c1 = _cell_bounds(col_lines, col_idx)
            cell = g[r0:r1, c0:c1]
            seeds = np.argwhere(cell != int(bg))
            if len(seeds) == 0:
                continue
            colors = np.unique(cell[cell != int(bg)])
            if len(colors) != 1:
                raise AssertionError("each data cell must contain one seed color")
            seed_color = int(colors[0])
            count = len(seeds)
            if count not in patterns:
                raise AssertionError("seed count must map to an existing legend cell")

            out[r0:r1, c0:c1] = int(bg)
            for pr, pc in patterns[count]:
                out[r0 + int(pr), c0 + int(pc)] = seed_color

    return out


def _paint_coords(
    grid: np.ndarray, top: int, left: int, coords: list[tuple[int, int]], color: int
) -> None:
    for r, c in coords:
        grid[top + int(r), left + int(c)] = int(color)


def _seed_coords(
    cell_h: int, cell_w: int, count: int, pattern: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    forbidden = set(pattern)
    for _ in range(50):
        coords = random.sample(all_cells, count)
        if set(coords) != forbidden:
            return sorted(coords)
    raise AssertionError("failed to place non-template seed coordinates")


def _choose_geometry() -> tuple[int, int, int, int]:
    for _ in range(100):
        cell_h = random.randint(3, 5)
        cell_w = random.randint(3, 5)
        max_row_cells = (30 - 1) // (cell_h + 1)
        max_col_cells = (30 - 1) // (cell_w + 1)
        max_patterns = min(6, max_col_cells, cell_h * cell_w - 1)
        max_data_rows = min(4, max_row_cells - 1)
        if max_patterns >= 3 and max_data_rows >= 2:
            n_cols = random.randint(3, max_patterns)
            data_rows = random.randint(2, max_data_rows)
            return cell_h, cell_w, n_cols, data_rows
    raise AssertionError("failed to choose a valid geometry")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    cell_h, cell_w, n_cols, data_rows = _choose_geometry()
    row_cells = data_rows + 1
    height = row_cells * (cell_h + 1) + 1
    width = n_cols * (cell_w + 1) + 1
    if height > 30 or width > 30:
        raise AssertionError("geometry exceeded ARC bounds")

    inp = np.full((height, width), BG, dtype=int)
    for r in range(0, height, cell_h + 1):
        inp[r, :] = GRID
    for c in range(0, width, cell_w + 1):
        inp[:, c] = GRID

    patterns = {k: _connected_pattern(cell_h, cell_w, k) for k in range(1, n_cols + 1)}
    for col_idx in range(n_cols):
        top = 1
        left = int(col_idx * (cell_w + 1) + 1)
        _paint_coords(inp, top, left, patterns[col_idx + 1], LEGEND)

    all_data_cells = [(r, c) for r in range(1, row_cells) for c in range(n_cols)]
    mandatory_cells = random.sample(all_data_cells, n_cols)
    remaining_cells = [cell for cell in all_data_cells if cell not in mandatory_cells]
    extra_active_max = len(remaining_cells) - 1
    if extra_active_max < 0:
        raise AssertionError("need at least one empty data cell")
    extra_active = random.randint(0, extra_active_max)
    extra_cells = random.sample(remaining_cells, extra_active)

    palette_size = random.randint(2, min(4, len(DATA_COLORS)))
    palette = random.sample(DATA_COLORS, palette_size)
    assignments: list[tuple[tuple[int, int], int, int]] = []

    for count, cell in enumerate(mandatory_cells, start=1):
        assignments.append((cell, count, random.choice(palette)))
    for cell in extra_cells:
        assignments.append((cell, random.randint(1, n_cols), random.choice(palette)))

    if len(assignments) > 1 and len({color for _cell, _count, color in assignments}) == 1:
        first_cell, first_count, first_color = assignments[0]
        alt_colors = [color for color in DATA_COLORS if color != first_color]
        assignments[0] = (first_cell, first_count, random.choice(alt_colors))

    out = inp.copy()
    for (row_idx, col_idx), count, color in assignments:
        top = int(row_idx * (cell_h + 1) + 1)
        left = int(col_idx * (cell_w + 1) + 1)
        seeds = _seed_coords(cell_h, cell_w, count, patterns[count])
        _paint_coords(inp, top, left, seeds, color)
        out[top : top + cell_h, left : left + cell_w] = BG
        _paint_coords(out, top, left, patterns[count], color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _build_case()
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        sig = {int(v) for v in np.unique(inp)}
        if not {BG, GRID, LEGEND}.issubset(sig):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
