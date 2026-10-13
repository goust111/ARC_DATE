import random

import numpy as np


BG = 0
SEP = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS = ("top", "bottom", "left", "right")
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_separator_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = [color for color in row_colors & col_colors if color != bg]
    assert len(candidates) == 1, "expected a unique full-row/full-column separator color"
    return int(candidates[0])


def _bands(size: int, lines: list[int]) -> list[tuple[int, int]]:
    borders = [-1] + sorted(int(x) for x in lines) + [int(size)]
    bands: list[tuple[int, int]] = []
    for left, right in zip(borders, borders[1:]):
        start = left + 1
        stop = right
        if start < stop:
            bands.append((start, stop))
    return bands


def _reflect_cell_pattern(cell: np.ndarray, bg: int) -> np.ndarray:
    h, w = cell.shape
    assert h >= 3 and w >= 3

    boundary_pixels = []
    for r in range(h):
        for c in range(w):
            if cell[r, c] == bg:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                boundary_pixels.append((r, c))
    assert len(boundary_pixels) == 1, "each cell must have exactly one boundary marker"

    mr, mc = boundary_pixels[0]
    marker_color = int(cell[mr, mc])

    out = np.full_like(cell, bg)
    for r, c in np.argwhere(cell != bg):
        r = int(r)
        c = int(c)
        if (r, c) == (mr, mc):
            continue
        if mr in (0, h - 1):
            rr, cc = h - 1 - r, c
        else:
            rr, cc = r, w - 1 - c
        out[rr, cc] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each cell's interior pattern across the center axis indicated by its edge marker, then repaint it with the marker color.

    Input structure:
    - Black background with one-pixel Azure separator rows and columns that form a regular lattice of equal cells.
    - Each cell interior contains one connected monochrome pattern that stays away from the cell boundary.
    - Each cell also contains exactly one single-pixel marker on the first or last interior row or column, in a color different from the pattern and separators.

    Rule:
    - Detect which cell edge the marker touches: top or bottom means reflect the pattern across the cell's horizontal centerline; left or right means reflect across the vertical centerline.
    - Repaint every reflected pattern pixel with the marker color.
    - Remove the marker itself and keep all separator lines unchanged.

    Key insight:
    - The marker carries two visible signals at once: its position selects the reflection axis, and its color becomes the output color for that cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sep = _find_separator_color(g, bg)
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "separator lattice must include cell boundaries"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands
    assert len({stop - start for start, stop in row_bands}) == 1, "row cells must have equal height"
    assert len({stop - start for start, stop in col_bands}) == 1, "column cells must have equal width"

    out = g.copy()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _reflect_cell_pattern(cell, bg)
    return out


def _grow_connected_shape(height: int, width: int, target_size: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = {start}
    while len(cells) < target_size:
        choices = []
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    choices.append((rr, cc))
        if not choices:
            break
        nxt = random.choice(choices)
        cells.add(nxt)
        frontier.add(nxt)
    return cells


def _reflected_shape(cells: set[tuple[int, int]], height: int, width: int, direction: str) -> set[tuple[int, int]]:
    if direction in ("top", "bottom"):
        return {(height - 1 - r, c) for r, c in cells}
    return {(r, width - 1 - c) for r, c in cells}


def _make_cell(height: int, width: int, direction: str) -> np.ndarray:
    cell = np.full((height, width), BG, dtype=int)
    core_h = height - 2
    core_w = width - 2

    for _ in range(100):
        area_max = min(6, core_h * core_w - 1)
        area = random.randint(2, area_max)
        shape = _grow_connected_shape(core_h, core_w, area)
        if shape != _reflected_shape(shape, core_h, core_w, direction):
            break
    else:
        raise RuntimeError("failed to create a non-symmetric cell pattern")

    pattern_color, marker_color = random.sample(PALETTE, 2)
    for r, c in shape:
        cell[r + 1, c + 1] = pattern_color

    if direction == "top":
        marker = (0, random.randint(1, width - 2))
    elif direction == "bottom":
        marker = (height - 1, random.randint(1, width - 2))
    elif direction == "left":
        marker = (random.randint(1, height - 2), 0)
    else:
        marker = (random.randint(1, height - 2), width - 1)
    cell[marker] = marker_color
    return cell


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)

    max_cell_h = (30 - (n_rows + 1)) // n_rows
    max_cell_w = (30 - (n_cols + 1)) // n_cols
    if max_cell_h < 5 or max_cell_w < 5:
        return None

    cell_h = random.randint(5, min(7, max_cell_h))
    cell_w = random.randint(5, min(7, max_cell_w))
    height = n_rows * cell_h + (n_rows + 1)
    width = n_cols * cell_w + (n_cols + 1)

    grid = np.full((height, width), BG, dtype=int)
    sep_rows = [r * (cell_h + 1) for r in range(n_rows + 1)]
    sep_cols = [c * (cell_w + 1) for c in range(n_cols + 1)]
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    n_cells = n_rows * n_cols
    full_cycles, extra = divmod(n_cells, len(DIRS))
    directions = list(DIRS) * full_cycles + random.sample(DIRS, extra)
    random.shuffle(directions)

    idx = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            r0 = sep_rows[ri] + 1
            r1 = sep_rows[ri + 1]
            c0 = sep_cols[ci] + 1
            c1 = sep_cols[ci + 1]
            grid[r0:r1, c0:c1] = _make_cell(cell_h, cell_w, directions[idx])
            idx += 1

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
