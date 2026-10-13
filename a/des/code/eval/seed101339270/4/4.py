from __future__ import annotations

import random

import numpy as np


BG = 0
LINE = 3
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)

PARTITION_ROWS_RANGE = (2, 4)
PARTITION_COLS_RANGE = (2, 4)
CELL_HEIGHT_RANGE = (4, 6)
CELL_WIDTH_RANGE = (4, 6)
PALETTE_SIZE_RANGE = (3, 6)
GEN_ATTEMPTS = 400

DIR_UP = "up"
DIR_DOWN = "down"
DIR_LEFT = "left"
DIR_RIGHT = "right"
ALL_DIRS = (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)


def _dominant_non_bg_color(g: np.ndarray, bg: int) -> int:
    values, counts = np.unique(g, return_counts=True)
    best_count = -1
    best_colors: list[int] = []
    for value, count in zip(values, counts):
        color = int(value)
        if color == bg:
            continue
        count = int(count)
        if count > best_count:
            best_count = count
            best_colors = [color]
        elif count == best_count:
            best_colors.append(color)
    assert len(best_colors) == 1, "expected a unique lattice color"
    return best_colors[0]


def _grid_line_indices(counts: np.ndarray) -> list[int]:
    max_count = int(counts.max())
    for frac in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5):
        threshold = max(2, int(np.ceil(max_count * frac)))
        indices = [int(i) for i, count in enumerate(counts) if int(count) >= threshold]
        if len(indices) < 3:
            continue
        diffs = np.diff(indices)
        if np.all(diffs == diffs[0]) and int(diffs[0]) > 1:
            return indices
    raise AssertionError("failed to recover evenly spaced lattice lines")


def _partition_lines(g: np.ndarray, line_color: int) -> tuple[list[int], list[int]]:
    mask = g == int(line_color)
    row_lines = _grid_line_indices(np.count_nonzero(mask, axis=1))
    col_lines = _grid_line_indices(np.count_nonzero(mask, axis=0))
    row_step = row_lines[1] - row_lines[0]
    col_step = col_lines[1] - col_lines[0]
    assert all(b - a == row_step for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a == col_step for a, b in zip(col_lines, col_lines[1:]))
    return row_lines, col_lines


def _cell_bounds(row_lines: list[int], col_lines: list[int], i: int, j: int) -> tuple[int, int, int, int]:
    return row_lines[i], row_lines[i + 1], col_lines[j], col_lines[j + 1]


def _cell_seed(g: np.ndarray, top: int, bottom: int, left: int, right: int, bg: int, line_color: int) -> tuple[int, int, int]:
    interior = g[top + 1 : bottom, left + 1 : right]
    seed_coords = np.argwhere((interior != int(bg)) & (interior != int(line_color)))
    assert seed_coords.shape == (1, 2), "expected exactly one seed in each partition"
    dr, dc = map(int, seed_coords[0])
    sr = top + 1 + dr
    sc = left + 1 + dc
    return sr, sc, int(g[sr, sc])


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG
    line_color = _dominant_non_bg_color(g, bg)
    row_lines, col_lines = _partition_lines(g, line_color)

    row_step = row_lines[1] - row_lines[0]
    col_step = col_lines[1] - col_lines[0]
    assert all(b - a == row_step for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a == col_step for a, b in zip(col_lines, col_lines[1:]))

    out = g.copy()
    height, width = g.shape

    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            top, bottom, left, right = _cell_bounds(row_lines, col_lines, i, j)
            sr, sc, color = _cell_seed(g, top, bottom, left, right, bg, line_color)

            # A side is active when the seed's straight growth ray reaches a hole on that side.
            if g[top, sc] == bg:
                out[top + 1 : sr, sc] = color
                if top > 0:
                    assert g[top - 1, sc] == bg
                    out[top - 1, sc] = color

            if g[bottom, sc] == bg:
                out[sr + 1 : bottom, sc] = color
                if bottom + 1 < height:
                    assert g[bottom + 1, sc] == bg
                    out[bottom + 1, sc] = color

            if g[sr, left] == bg:
                out[sr, left + 1 : sc] = color
                if left > 0:
                    assert g[sr, left - 1] == bg
                    out[sr, left - 1] = color

            if g[sr, right] == bg:
                out[sr, sc + 1 : right] = color
                if right + 1 < width:
                    assert g[sr, right + 1] == bg
                    out[sr, right + 1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored bars inside lattice cells according to boundary gaps.

    Input structure:
    - bg is black.
    - A single-pixel green lattice partitions the grid into equal rectangles.
    - Each cell contains exactly one non-black, non-green seed pixel.
    - Some lattice segments contain one-pixel gaps.

    Rule:
    - A gap aligned with the seed's row/column activates growth toward that side.
    - Growth fills the straight in-cell segment from the seed to that boundary.
    - The gap itself stays unchanged; growth only leaks one more pixel into the
      neighboring cell on the far side of the gap.
    - Seeds and intact lattice pixels remain unchanged.
    """
    return _transform(grid)


def _available_dirs(i: int, j: int, n_rows: int, n_cols: int) -> list[str]:
    dirs = []
    if i > 0:
        dirs.append(DIR_UP)
    if i + 1 < n_rows:
        dirs.append(DIR_DOWN)
    if j > 0:
        dirs.append(DIR_LEFT)
    if j + 1 < n_cols:
        dirs.append(DIR_RIGHT)
    return dirs


def _hole_position(cell: dict[str, int], direction: str, row_lines: list[int], col_lines: list[int]) -> tuple[int, int]:
    i = int(cell["i"])
    j = int(cell["j"])
    top, bottom, left, right = _cell_bounds(row_lines, col_lines, i, j)
    sr = int(cell["sr"])
    sc = int(cell["sc"])
    if direction == DIR_UP:
        return top, sc
    if direction == DIR_DOWN:
        return bottom, sc
    if direction == DIR_LEFT:
        return sr, left
    return sr, right


def _claims_for_cell(
    cell: dict[str, int],
    directions: set[str],
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[int, int]]:
    i = int(cell["i"])
    j = int(cell["j"])
    top, bottom, left, right = _cell_bounds(row_lines, col_lines, i, j)
    sr = int(cell["sr"])
    sc = int(cell["sc"])
    claims: list[tuple[int, int]] = []

    if DIR_UP in directions:
        claims.extend((r, sc) for r in range(top + 1, sr))
        claims.append((top - 1, sc))
    if DIR_DOWN in directions:
        claims.extend((r, sc) for r in range(sr + 1, bottom))
        claims.append((bottom + 1, sc))
    if DIR_LEFT in directions:
        claims.extend((sr, c) for c in range(left + 1, sc))
        claims.append((sr, left - 1))
    if DIR_RIGHT in directions:
        claims.extend((sr, c) for c in range(sc + 1, right))
        claims.append((sr, right + 1))

    return claims


def _build_candidate() -> np.ndarray | None:
    n_rows = random.randint(*PARTITION_ROWS_RANGE)
    n_cols = random.randint(*PARTITION_COLS_RANGE)
    cell_h = random.randint(*CELL_HEIGHT_RANGE)
    cell_w = random.randint(*CELL_WIDTH_RANGE)

    height = n_rows * cell_h + n_rows + 1
    width = n_cols * cell_w + n_cols + 1
    row_lines = [r * (cell_h + 1) for r in range(n_rows + 1)]
    col_lines = [c * (cell_w + 1) for c in range(n_cols + 1)]

    grid = np.full((height, width), BG, dtype=int)
    grid[row_lines, :] = LINE
    grid[:, col_lines] = LINE

    n_cells = n_rows * n_cols
    palette_size = random.randint(PALETTE_SIZE_RANGE[0], min(PALETTE_SIZE_RANGE[1], n_cells, len(SEED_COLORS)))
    palette = random.sample(SEED_COLORS, palette_size)
    seed_colors = palette.copy()
    while len(seed_colors) < n_cells:
        seed_colors.append(random.choice(palette))
    random.shuffle(seed_colors)

    cells: list[dict[str, int]] = []
    color_iter = iter(seed_colors)
    for i in range(n_rows):
        for j in range(n_cols):
            top, bottom, left, right = _cell_bounds(row_lines, col_lines, i, j)
            sr = top + 1 + random.randint(1, cell_h - 2)
            sc = left + 1 + random.randint(1, cell_w - 2)
            color = int(next(color_iter))
            grid[sr, sc] = color
            cells.append({"i": i, "j": j, "sr": sr, "sc": sc, "color": color})

    directions_by_cell: dict[tuple[int, int], set[str]] = {}
    dir_counts = {direction: 0 for direction in ALL_DIRS}
    multi_direction = False
    for cell in cells:
        avail = _available_dirs(int(cell["i"]), int(cell["j"]), n_rows, n_cols)
        size = random.randint(1, len(avail))
        chosen = set(random.sample(avail, size))
        directions_by_cell[(int(cell["i"]), int(cell["j"]))] = chosen
        multi_direction |= len(chosen) >= 2
        for direction in chosen:
            dir_counts[direction] += 1

    if not multi_direction:
        return None
    if any(count == 0 for count in dir_counts.values()):
        return None

    holes: set[tuple[int, int]] = set()
    for cell in cells:
        chosen = directions_by_cell[(int(cell["i"]), int(cell["j"]))]
        for direction in chosen:
            pos = _hole_position(cell, direction, row_lines, col_lines)
            if pos in holes:
                return None
            holes.add(pos)
            grid[pos] = BG

    seed_positions = {(int(cell["sr"]), int(cell["sc"])) for cell in cells}
    claims: dict[tuple[int, int], int] = {}
    for cell in cells:
        color = int(cell["color"])
        chosen = directions_by_cell[(int(cell["i"]), int(cell["j"]))]
        for pos in _claims_for_cell(cell, chosen, row_lines, col_lines):
            r, c = pos
            if not (0 <= r < height and 0 <= c < width):
                return None
            if (r, c) in seed_positions:
                return None
            if int(grid[r, c]) != BG:
                return None
            if (r, c) in claims:
                return None
            claims[(r, c)] = color

    # Keep the grid visually readable: avoid over-dense fill patterns.
    if len(claims) < n_cells * 2:
        return None

    try:
        transformed = _transform(grid)
    except AssertionError:
        return None
    assert transformed.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-growth input")
