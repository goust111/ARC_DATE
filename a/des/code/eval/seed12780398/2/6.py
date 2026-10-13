from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

DIRS = [
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1),  # up-left
]

RING_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
POINTER_COLORS = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 200


def _consecutive_groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _separator_groups(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    if axis == 0:
        indices = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    else:
        indices = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    groups = _consecutive_groups(indices)
    assert len(groups) >= 2, "expected outer frame and internal separators"
    return groups


def _cell_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_groups = _separator_groups(grid, axis=0)
    col_groups = _separator_groups(grid, axis=1)
    assert row_groups[0][0] == 0 and row_groups[-1][1] == grid.shape[0] - 1
    assert col_groups[0][0] == 0 and col_groups[-1][1] == grid.shape[1] - 1

    row_spans = [(a[1] + 1, b[0] - 1) for a, b in zip(row_groups, row_groups[1:]) if a[1] + 1 <= b[0] - 1]
    col_spans = [(a[1] + 1, b[0] - 1) for a, b in zip(col_groups, col_groups[1:]) if a[1] + 1 <= b[0] - 1]
    assert row_spans and col_spans

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    cell_h = row_sizes.pop()
    cell_w = col_sizes.pop()
    assert cell_h == cell_w and cell_h % 2 == 1 and cell_h >= 3
    return row_spans, col_spans


def _cell_center(r_span: tuple[int, int], c_span: tuple[int, int]) -> tuple[int, int]:
    r0, r1 = r_span
    c0, c1 = c_span
    return (r0 + r1) // 2, (c0 + c1) // 2


def _palette_mapping(
    grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], int], tuple[int, int] | None]:
    candidates: list[tuple[dict[tuple[int, int], int], tuple[int, int]]] = []
    for ri, r_span in enumerate(row_spans):
        for ci, c_span in enumerate(col_spans):
            cr, cc = _cell_center(r_span, c_span)
            if int(grid[cr, cc]) != MAROON:
                continue
            ring = [int(grid[cr + dr, cc + dc]) for dr, dc in DIRS]
            if all(color not in (BLACK, GREY) for color in ring) and len(set(ring)) == 8:
                mapping = {direction: int(grid[cr + direction[0], cc + direction[1]]) for direction in DIRS}
                candidates.append((mapping, (ri, ci)))
    assert len(candidates) == 1, "expected exactly one palette cell"
    return candidates[0]


def _pointer_direction(grid: np.ndarray, r_span: tuple[int, int], c_span: tuple[int, int]) -> tuple[int, int] | None:
    cr, cc = _cell_center(r_span, c_span)
    if int(grid[cr, cc]) != MAROON:
        return None

    r0, r1 = r_span
    c0, c1 = c_span
    found: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = cr + dr
        cc2 = cc + dc
        if not (r0 <= rr <= r1 and c0 <= cc2 <= c1):
            continue
        first = int(grid[rr, cc2])
        if first in (BLACK, GREY, MAROON):
            continue
        length = 0
        while r0 <= rr <= r1 and c0 <= cc2 <= c1 and int(grid[rr, cc2]) == first:
            length += 1
            rr += dr
            cc2 += dc
        if length > 0:
            found.append((dr, dc))
    if not found:
        return None
    assert len(found) == 1, "pointer cell should have exactly one ray"
    return found[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each clock-hand direction, look up the matching color on the corner wheel,
    and fill the remainder of that hand's cell with the looked-up color.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _cell_spans(g)
    palette, palette_idx = _palette_mapping(g, row_spans, col_spans)

    out = g.copy()
    for ri, r_span in enumerate(row_spans):
        for ci, c_span in enumerate(col_spans):
            if (ri, ci) == palette_idx:
                continue
            direction = _pointer_direction(g, r_span, c_span)
            if direction is None:
                continue
            fill_color = palette[direction]
            r0, r1 = r_span
            c0, c1 = c_span
            cell = out[r0 : r1 + 1, c0 : c1 + 1]
            cell[cell == BLACK] = fill_color
    return out


def _assign_pointer_colors(chosen_dirs: list[tuple[int, int]], wheel_map: dict[tuple[int, int], int]) -> dict[tuple[int, int], int] | None:
    for _ in range(100):
        colors = random.sample(POINTER_COLORS, len(chosen_dirs))
        if all(colors[i] != wheel_map[chosen_dirs[i]] for i in range(len(chosen_dirs))):
            return dict(zip(chosen_dirs, colors))
    return None


def _cell_origin(cell_inner: int, cell_r: int, cell_c: int) -> tuple[int, int]:
    return 1 + cell_r * (cell_inner + 1), 1 + cell_c * (cell_inner + 1)


def _draw_grid_lines(grid: np.ndarray, cell_inner: int, n_cell_rows: int, n_cell_cols: int) -> None:
    for r in range(n_cell_rows + 1):
        grid[r * (cell_inner + 1), :] = GREY
    for c in range(n_cell_cols + 1):
        grid[:, c * (cell_inner + 1)] = GREY


def _gen_attempt() -> np.ndarray | None:
    n_cell_rows = random.randint(3, 5)
    n_cell_cols = random.randint(3, 5)
    cell_inner = random.choice([5, 7])
    total_cells = n_cell_rows * n_cell_cols
    max_pointers = min(len(POINTER_COLORS), total_cells - 1)
    if max_pointers <= 0:
        return None

    grid_h = n_cell_rows * cell_inner + n_cell_rows + 1
    grid_w = n_cell_cols * cell_inner + n_cell_cols + 1
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    _draw_grid_lines(grid, cell_inner, n_cell_rows, n_cell_cols)

    wheel_map = dict(zip(DIRS, random.sample(RING_COLORS, len(DIRS))))
    wheel_corner = random.choice(
        [(0, 0), (0, n_cell_cols - 1), (n_cell_rows - 1, 0), (n_cell_rows - 1, n_cell_cols - 1)]
    )
    wr0, wc0 = _cell_origin(cell_inner, wheel_corner[0], wheel_corner[1])
    wcr = wr0 + cell_inner // 2
    wcc = wc0 + cell_inner // 2
    grid[wcr, wcc] = MAROON
    for dr, dc in DIRS:
        grid[wcr + dr, wcc + dc] = wheel_map[(dr, dc)]

    available_cells = [(r, c) for r in range(n_cell_rows) for c in range(n_cell_cols) if (r, c) != wheel_corner]
    num_pointers = random.randint(1, min(7, max_pointers))
    chosen_cells = random.sample(available_cells, num_pointers)
    chosen_dirs = random.sample(DIRS, num_pointers)
    pointer_colors = _assign_pointer_colors(chosen_dirs, wheel_map)
    if pointer_colors is None:
        return None

    max_len = cell_inner // 2
    for (cell_r, cell_c), direction in zip(chosen_cells, chosen_dirs):
        pr0, pc0 = _cell_origin(cell_inner, cell_r, cell_c)
        center_r = pr0 + cell_inner // 2
        center_c = pc0 + cell_inner // 2
        color = pointer_colors[direction]
        length = random.randint(2, max_len)
        grid[center_r, center_c] = MAROON
        dr, dc = direction
        for step in range(1, length + 1):
            grid[center_r + dr * step, center_c + dc * step] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
