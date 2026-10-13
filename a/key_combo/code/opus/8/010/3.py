from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9
CONTROL_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_chambers(grid: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
    for coords in _components_of_color(grid, BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            continue
        if h != w or h not in (2, 3):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if coords.shape[0] != h * w or not np.all(patch == BLACK):
            continue
        candidates.append(((h, w), (r0, c0, r1, c1)))

    by_shape: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for shape, box in candidates:
        by_shape[shape].append(box)

    matches = []
    for (h, w), boxes in by_shape.items():
        rows = sorted({r0 for r0, _c0, _r1, _c1 in boxes})
        cols = sorted({c0 for _r0, c0, _r1, _c1 in boxes})
        if not (3 <= len(rows) <= 5 and 3 <= len(cols) <= 5):
            continue
        if len(boxes) != len(rows) * len(cols):
            continue
        if any(rows[i + 1] - rows[i] != h + 1 for i in range(len(rows) - 1)):
            continue
        if any(cols[i + 1] - cols[i] != w + 1 for i in range(len(cols) - 1)):
            continue
        box_map = {(r0, c0): box for box in boxes for r0, c0, _r1, _c1 in [box]}
        if any((r, c) not in box_map for r in rows for c in cols):
            continue
        matches.append((rows, cols, box_map))

    assert len(matches) == 1, "expected one regular chamber lattice"
    rows, cols, box_map = matches[0]
    indexed = {(i, j): box_map[(r, c)] for i, r in enumerate(rows) for j, c in enumerate(cols)}
    return rows, cols, indexed


def _find_control_matrix(grid: np.ndarray, rows: int, cols: int) -> tuple[int, int]:
    mask = np.isin(grid, CONTROL_COLORS)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a control matrix"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (rows, cols), "control matrix shape mismatch"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "control matrix must be dense"
    return int(r0), int(c0)


def _chamber_is_activated(box: tuple[int, int, int, int], markers: np.ndarray) -> bool:
    r0, c0, r1, c1 = box
    for mr, mc in markers:
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if abs(int(rr - mr)) == abs(int(cc - mc)):
                    return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Use diagonal rays from maroon lattice markers to select chambers, then fill each selected chamber from the same-position control-matrix color.

    Input structure:
    - A teal lattice with maroon markers at some intersection nodes encloses a regular grid of black square chambers.
    - A dense colored control matrix sits outside the lattice, and its row and column counts match the chamber lattice.
    - Outside the lattice, the remaining background is black.

    Rule:
    - Identify the regular chamber rows and columns from the enclosed black square cavities.
    - For every maroon marker, activate the four full diagonals through that marker out to the grid edges.
    - A chamber is selected iff at least one cell of its black interior lies on one of those diagonals.
    - Fill every selected chamber with the color from the control matrix at the same chamber-row and chamber-column index.
    - Leave unselected chambers, the teal lattice, the maroon markers, and the control matrix unchanged.

    Key insight:
    - The diagonals do not supply colors; they only choose chamber coordinates, and the control matrix supplies the fill color independently by matching lattice position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    chamber_rows, chamber_cols, chambers = _find_chambers(g)
    matrix_r0, matrix_c0 = _find_control_matrix(g, len(chamber_rows), len(chamber_cols))
    markers = np.argwhere(g == MAROON)
    assert 1 <= len(markers) <= 2, "expected one or two markers"

    out = g.copy()
    for (i, j), box in chambers.items():
        if not _chamber_is_activated(box, markers):
            continue
        r0, c0, r1, c1 = box
        color = int(g[matrix_r0 + i, matrix_c0 + j])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _draw_lattice(grid: np.ndarray, top: int, left: int, rows: int, cols: int, cell: int) -> None:
    lattice_h = rows * cell + (rows + 1)
    lattice_w = cols * cell + (cols + 1)
    for i in range(rows + 1):
        r = top + i * (cell + 1)
        grid[r, left : left + lattice_w] = TEAL
    for j in range(cols + 1):
        c = left + j * (cell + 1)
        grid[top : top + lattice_h, c] = TEAL


def _control_matrix(rows: int, cols: int) -> np.ndarray:
    palette_size = random.randint(2, min(len(CONTROL_COLORS), rows * cols))
    palette = random.sample(CONTROL_COLORS, palette_size)
    flat = [palette[i] for i in range(palette_size)]
    flat.extend(random.choice(palette) for _ in range(rows * cols - palette_size))
    random.shuffle(flat)
    return np.asarray(flat, dtype=int).reshape(rows, cols)


def _marker_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    nodes = [(i, j) for i in range(1, rows) for j in range(1, cols)]
    marker_count = 1 if random.random() < 0.5 else 2
    return random.sample(nodes, marker_count)


def _activated_indices(
    rows: int,
    cols: int,
    cell: int,
    lattice_top: int,
    lattice_left: int,
    markers: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    marker_pixels = [
        (lattice_top + i * (cell + 1), lattice_left + j * (cell + 1))
        for i, j in markers
    ]
    active: set[tuple[int, int]] = set()
    for i in range(rows):
        r0 = lattice_top + 1 + i * (cell + 1)
        for j in range(cols):
            c0 = lattice_left + 1 + j * (cell + 1)
            for mr, mc in marker_pixels:
                hit = False
                for rr in range(r0, r0 + cell):
                    for cc in range(c0, c0 + cell):
                        if abs(rr - mr) == abs(cc - mc):
                            active.add((i, j))
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
    return active


def _build_input() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    cell = random.choice((2, 3))

    lattice_h = rows * cell + (rows + 1)
    lattice_w = cols * cell + (cols + 1)
    top_margin = rows + random.randint(1, 3)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    height = top_margin + lattice_h + bottom_margin
    width = left_margin + lattice_w + right_margin
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    lattice_top = top_margin
    lattice_left = left_margin
    _draw_lattice(grid, lattice_top, lattice_left, rows, cols, cell)

    matrix = _control_matrix(rows, cols)
    matrix_r0 = random.randint(0, top_margin - rows)
    matrix_c0 = width - right_margin - cols
    grid[matrix_r0 : matrix_r0 + rows, matrix_c0 : matrix_c0 + cols] = matrix

    markers = _marker_positions(rows, cols)
    for i, j in markers:
        r = lattice_top + i * (cell + 1)
        c = lattice_left + j * (cell + 1)
        grid[r, c] = MAROON

    active = _activated_indices(rows, cols, cell, lattice_top, lattice_left, markers)
    if not active or len(active) == rows * cols:
        return None
    active_colors = {int(matrix[i, j]) for i, j in active}
    if len(active_colors) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
