from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)

UP = "U"
DOWN = "D"
LEFT = "L"
RIGHT = "R"

DIR_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# 3-pixel L corners. The "opening side" encodes the growth direction.
DIR_TO_SHAPE = {
    UP: frozenset({(0, 1), (1, 0), (1, 1)}),
    DOWN: frozenset({(0, 0), (0, 1), (1, 0)}),
    LEFT: frozenset({(0, 0), (0, 1), (1, 1)}),
    RIGHT: frozenset({(0, 0), (1, 0), (1, 1)}),
}
SHAPE_TO_DIR = {shape: direction for direction, shape in DIR_TO_SHAPE.items()}


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == color))]
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == color))]


def _parse_cell_layout(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    grey_rows = _full_line_indices(grid, GREY, axis=0)
    grey_cols = _full_line_indices(grid, GREY, axis=1)
    assert len(grey_rows) >= 3 and len(grey_cols) >= 3, "expected a gridded board"
    assert grey_rows[0] == 0 and grey_cols[0] == 0
    assert grey_rows[-1] == grid.shape[0] - 1 and grey_cols[-1] == grid.shape[1] - 1

    row_steps = {b - a for a, b in zip(grey_rows, grey_rows[1:])}
    col_steps = {b - a for a, b in zip(grey_cols, grey_cols[1:])}
    assert len(row_steps) == 1 and len(col_steps) == 1, "grid lines must be evenly spaced"

    row_step = row_steps.pop()
    col_step = col_steps.pop()
    assert row_step == col_step and row_step >= 4, "cells must be square with size >= 3"
    cell_size = int(row_step - 1)
    return grey_rows, grey_cols, cell_size


def _decode_seed(cell: np.ndarray) -> tuple[int, str]:
    coords = np.argwhere(cell != BLACK)
    assert len(coords) == 3, "each cell must contain one 3-pixel seed"

    colors = np.unique(cell[cell != BLACK])
    assert len(colors) == 1, "seed must be monochrome"
    color = int(colors[0])
    assert color in SEED_COLORS, "seed color must come from the visible seed palette"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2, "seed must be a 2x2-corner L"
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert shape in SHAPE_TO_DIR, "unknown seed orientation"
    return color, SHAPE_TO_DIR[shape]


def _draw_strip(cell: np.ndarray, direction: str, color: int) -> None:
    n = int(cell.shape[0])
    center = n // 2
    if direction == UP:
        cell[: center + 1, center] = color
    elif direction == DOWN:
        cell[center:, center] = color
    elif direction == LEFT:
        cell[center, : center + 1] = color
    else:
        cell[center, center:] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each seed L with a 1-pixel growth strip whose direction comes from the seed opening.

    Input structure:
    - Black background with a 1-pixel grey lattice splitting the canvas into equal square cells.
    - Every cell contains exactly one monochrome 3-pixel L seed in one of five visible colors.
    - The seed opening encodes one of four growth directions.

    Rule:
    - Decode the L opening in each cell to get that cell's direction.
    - A cell keeps its own seed color if it points to an in-bounds neighbor.
    - A cell becomes maroon if it points outside the board.
    - If two neighboring cells point at each other, both are overridden to azure.
    - Output keeps the grey grid and replaces every seed by a 1-pixel strip from the cell center to the cell edge.

    Key insight:
    - The seed is not copied; it only encodes direction. The azure override depends on scanning adjacent pairs globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_rows, grey_cols, cell_size = _parse_cell_layout(g)
    rows = len(grey_rows) - 1
    cols = len(grey_cols) - 1

    seed_colors = np.zeros((rows, cols), dtype=int)
    directions = np.empty((rows, cols), dtype="<U1")

    for rr in range(rows):
        for cc in range(cols):
            cell = g[
                grey_rows[rr] + 1 : grey_rows[rr + 1],
                grey_cols[cc] + 1 : grey_cols[cc + 1],
            ]
            assert cell.shape == (cell_size, cell_size)
            color, direction = _decode_seed(cell)
            seed_colors[rr, cc] = color
            directions[rr, cc] = direction

    out_colors = seed_colors.copy()
    for rr in range(rows):
        for cc in range(cols):
            dr, dc = DIR_TO_DELTA[str(directions[rr, cc])]
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                out_colors[rr, cc] = MAROON

    for rr in range(rows):
        for cc in range(cols):
            direction = str(directions[rr, cc])
            dr, dc = DIR_TO_DELTA[direction]
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < rows and 0 <= nc < cols and str(directions[nr, nc]) == OPPOSITE[direction]:
                out_colors[rr, cc] = AZURE
                out_colors[nr, nc] = AZURE

    out = np.full_like(g, BLACK)
    out[grey_rows, :] = GREY
    out[:, grey_cols] = GREY

    for rr in range(rows):
        for cc in range(cols):
            cell = out[
                grey_rows[rr] + 1 : grey_rows[rr + 1],
                grey_cols[cc] + 1 : grey_cols[cc + 1],
            ]
            _draw_strip(cell, str(directions[rr, cc]), int(out_colors[rr, cc]))

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_ROWS = (2, 3, 4, 5)
GRID_COLS = (2, 3, 4, 5)
CELL_SIZES = (3, 5, 7)


def _adjacent_pairs(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int], str, str]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int], str, str]] = []
    for rr in range(rows):
        for cc in range(cols):
            if cc + 1 < cols:
                pairs.append(((rr, cc), (rr, cc + 1), RIGHT, LEFT))
            if rr + 1 < rows:
                pairs.append(((rr, cc), (rr + 1, cc), DOWN, UP))
    return pairs


def _outward_directions(rr: int, cc: int, rows: int, cols: int) -> list[str]:
    directions: list[str] = []
    if rr == 0:
        directions.append(UP)
    if rr == rows - 1:
        directions.append(DOWN)
    if cc == 0:
        directions.append(LEFT)
    if cc == cols - 1:
        directions.append(RIGHT)
    return directions


def _classify_cells(directions: np.ndarray) -> dict[str, int]:
    rows, cols = map(int, directions.shape)
    counts = {"mutual": 0, "boundary": 0, "normal": 0}
    for rr in range(rows):
        for cc in range(cols):
            direction = str(directions[rr, cc])
            dr, dc = DIR_TO_DELTA[direction]
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                counts["boundary"] += 1
            elif str(directions[nr, nc]) == OPPOSITE[direction]:
                counts["mutual"] += 1
            else:
                counts["normal"] += 1
    return counts


def _stamp_seed(cell: np.ndarray, direction: str, color: int) -> None:
    n = int(cell.shape[0])
    r0 = random.randint(0, n - 2)
    c0 = random.randint(0, n - 2)
    for dr, dc in DIR_TO_SHAPE[direction]:
        cell[r0 + dr, c0 + dc] = int(color)


def _build_input(directions: np.ndarray, colors: np.ndarray, cell_size: int) -> np.ndarray:
    rows, cols = map(int, directions.shape)
    height = rows * cell_size + rows + 1
    width = cols * cell_size + cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = list(range(0, height, cell_size + 1))
    col_lines = list(range(0, width, cell_size + 1))
    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    for rr in range(rows):
        for cc in range(cols):
            cell = grid[
                row_lines[rr] + 1 : row_lines[rr + 1],
                col_lines[cc] + 1 : col_lines[cc + 1],
            ]
            _stamp_seed(cell, str(directions[rr, cc]), int(colors[rr, cc]))
    return grid


def _sample_colors(rows: int, cols: int) -> np.ndarray:
    total = rows * cols
    distinct_count = random.randint(2, min(len(SEED_COLORS), max(3, min(total, len(SEED_COLORS)))))
    palette = random.sample(SEED_COLORS, distinct_count)
    colors = np.array([random.choice(palette) for _ in range(total)], dtype=int).reshape(rows, cols)

    chosen_cells = random.sample(range(total), distinct_count)
    for idx, color in zip(chosen_cells, palette):
        colors[idx // cols, idx % cols] = int(color)
    return colors


def _sample_directions(rows: int, cols: int) -> np.ndarray | None:
    directions = np.full((rows, cols), "", dtype="<U1")
    fixed = np.zeros((rows, cols), dtype=bool)

    pair = random.choice(_adjacent_pairs(rows, cols))
    (a_r, a_c), (b_r, b_c), a_dir, b_dir = pair
    directions[a_r, a_c] = a_dir
    directions[b_r, b_c] = b_dir
    fixed[a_r, a_c] = True
    fixed[b_r, b_c] = True

    boundary_cells = [
        (rr, cc)
        for rr in range(rows)
        for cc in range(cols)
        if _outward_directions(rr, cc, rows, cols) and not fixed[rr, cc]
    ]
    if not boundary_cells:
        return None
    boundary_r, boundary_c = random.choice(boundary_cells)
    directions[boundary_r, boundary_c] = random.choice(_outward_directions(boundary_r, boundary_c, rows, cols))
    fixed[boundary_r, boundary_c] = True

    candidates: list[tuple[int, int, str, int, int]] = []
    for rr in range(rows):
        for cc in range(cols):
            if fixed[rr, cc]:
                continue
            for direction, (dr, dc) in DIR_TO_DELTA.items():
                nr = int(rr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if fixed[nr, nc] and str(directions[nr, nc]) == OPPOSITE[direction]:
                    continue
                candidates.append((rr, cc, direction, nr, nc))
    if not candidates:
        return None

    normal_r, normal_c, normal_dir, normal_nr, normal_nc = random.choice(candidates)
    directions[normal_r, normal_c] = normal_dir
    fixed[normal_r, normal_c] = True

    if not fixed[normal_nr, normal_nc]:
        safe_dirs = [direction for direction in DIR_TO_DELTA if direction != OPPOSITE[normal_dir]]
        directions[normal_nr, normal_nc] = random.choice(safe_dirs)
        fixed[normal_nr, normal_nc] = True

    for rr in range(rows):
        for cc in range(cols):
            if not fixed[rr, cc]:
                directions[rr, cc] = random.choice((UP, DOWN, LEFT, RIGHT))

    counts = _classify_cells(directions)
    if min(counts.values()) == 0:
        return None
    return directions


def _gen_attempt() -> np.ndarray | None:
    rows = random.choice(GRID_ROWS)
    cols = random.choice(GRID_COLS)
    cell_size = random.choice(CELL_SIZES)

    directions = _sample_directions(rows, cols)
    if directions is None:
        return None
    colors = _sample_colors(rows, cols)

    grid = _build_input(directions, colors, cell_size)

    try:
        solved = main(grid)
    except Exception:
        return None

    counts = _classify_cells(directions)
    if min(counts.values()) == 0:
        return None
    if not np.any(solved == AZURE):
        return None
    if not np.any(solved == MAROON):
        return None
    if not np.any(np.isin(solved, SEED_COLORS)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-growth board")
