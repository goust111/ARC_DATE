from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
PINK = 6
TEAL = 8

CELL_SIZE = 2
CELL_COUNTS = (4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _cell_ranges(length: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    walls = sorted(set(int(w) for w in wall_indices))
    bands: list[tuple[int, int]] = []
    start = 0
    for wall in walls:
        if start <= wall - 1:
            bands.append((start, wall - 1))
        start = wall + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= pos <= hi:
            return idx
    raise ValueError(f"position {pos} is not inside any cell band")


def _paint_cell(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell: tuple[int, int],
    color: int,
) -> None:
    row_idx, col_idx = cell
    r0, r1 = row_bands[row_idx]
    c0, c1 = col_bands[col_idx]
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _swap_red_green(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    red_mask = out == RED
    green_mask = out == GREEN
    out[red_mask] = GREEN
    out[green_mask] = RED
    return out


def _split_input_halves(grid: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[1] % 2:
        raise ValueError("expected a 2D grid with even width")
    half_w = g.shape[1] // 2
    left = g[:, :half_w]
    right = g[:, half_w:]
    left_has_teal = bool(np.any(left == TEAL))
    right_has_teal = bool(np.any(right == TEAL))
    if left_has_teal == right_has_teal:
        raise ValueError("expected exactly one lattice half")
    if left_has_teal:
        if np.any(right != BLACK):
            raise ValueError("expected the opposite half to be blank")
        return "left", left.copy(), right.copy()
    if np.any(left != BLACK):
        raise ValueError("expected the opposite half to be blank")
    return "right", right.copy(), left.copy()


def _extract_lattice(half: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_walls = [r for r in range(half.shape[0]) if np.all(half[r, :] == TEAL)]
    col_walls = [c for c in range(half.shape[1]) if np.all(half[:, c] == TEAL)]
    row_bands = _cell_ranges(half.shape[0], row_walls)
    col_bands = _cell_ranges(half.shape[1], col_walls)
    if len(row_bands) < 3 or len(col_bands) < 3:
        raise ValueError("expected at least a 3x3 lattice")
    return row_bands, col_bands


def _marker_cells(
    half: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> dict[int, set[tuple[int, int]]]:
    markers: dict[int, set[tuple[int, int]]] = {RED: set(), GREEN: set()}
    marker_pixels: dict[int, list[tuple[int, int]]] = {RED: [], GREEN: []}
    for r_raw, c_raw in np.argwhere((half != BLACK) & (half != TEAL)):
        r = int(r_raw)
        c = int(c_raw)
        color = int(half[r, c])
        if color not in (RED, GREEN):
            raise ValueError("unexpected non-marker color in input half")
        marker_pixels[color].append((r, c))
        markers[color].add((_band_index(r, row_bands), _band_index(c, col_bands)))
    if len(markers[RED]) < 3 or len(markers[GREEN]) < 3:
        raise ValueError("expected at least three red and three green marker cells")
    return markers


def _fill_cells_from_markers(markers: dict[int, set[tuple[int, int]]]) -> dict[tuple[int, int], set[int]]:
    filled: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, cells in markers.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for row_idx, col_idx in cells:
            by_row[row_idx].append(col_idx)
            by_col[col_idx].append(row_idx)
        for row_idx, cols in by_row.items():
            if len(cols) >= 2:
                for col_idx in range(min(cols), max(cols) + 1):
                    filled[(row_idx, col_idx)].add(color)
        for col_idx, rows in by_col.items():
            if len(rows) >= 2:
                for row_idx in range(min(rows), max(rows) + 1):
                    filled[(row_idx, col_idx)].add(color)
    return filled


def _filled_half(half: np.ndarray) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]], dict[tuple[int, int], set[int]]]:
    row_bands, col_bands = _extract_lattice(half)
    markers = _marker_cells(half, row_bands, col_bands)
    filled_cells = _fill_cells_from_markers(markers)

    out = half.copy()
    for cell, colors in filled_cells.items():
        color = PINK if colors == {RED, GREEN} else next(iter(colors))
        _paint_cell(out, row_bands, col_bands, cell, color)

    # Preserve the original marker pixels on top of the cell fills.
    marker_mask = (half == RED) | (half == GREEN)
    out[marker_mask] = half[marker_mask]
    return out, markers, filled_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the lattice half by marker-aligned cell runs, then mirror it into the blank half with red/green swapped.

    Input structure:
    - The grid width is split into two equal halves.
    - Exactly one half contains a teal lattice whose vertical border touches exactly one outer grid edge; the other half is blank black background.
    - Red and green marker pixels sit inside lattice cells, with at least three marker cells per color.

    Rule:
    - Inside the lattice half, for each color independently, fill every lattice-cell row segment and column segment whose endpoints are same-color marker cells.
    - Cells reached by both red and green fills become pink, while teal lattice lines stay teal.
    - Keep the original marker pixels visible on top of the filled cells.
    - Horizontally mirror the completed lattice half into the blank half and swap red with green in that mirrored copy; pink, teal, and black stay unchanged.

    Key insight:
    - The markers operate on coarse lattice cells rather than raw pixels, and the touched outer vertical border determines which side receives the mirrored color-swapped copy.
    """
    side, occupied_half, _blank_half = _split_input_halves(np.asarray(grid, dtype=int))
    completed_half, _markers, _filled_cells = _filled_half(occupied_half)
    mirrored_half = _swap_red_green(np.fliplr(completed_half))
    if side == "left":
        return np.concatenate((completed_half, mirrored_half), axis=1)
    return np.concatenate((mirrored_half, completed_half), axis=1)


def _wall_indices(n_cells: int, anchor: int) -> list[int]:
    pitch = CELL_SIZE + 1
    return [anchor + pitch * i for i in range(n_cells)]


def _marker_pixel(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    row_idx, col_idx = cell
    r0, _r1 = row_bands[row_idx]
    c0, _c1 = col_bands[col_idx]
    return r0 + offset[0], c0 + offset[1]


def _coarse_template(n_cells: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    rows = sorted(random.sample(range(n_cells), 3))
    cols = sorted(random.sample(range(n_cells), 4))
    r0, r1, r2 = rows
    c0, c1, c2, c3 = cols
    red_cells = {(r1, c0), (r1, c2), (r0, c0)}
    green_cells = {(r0, c1), (r2, c1), (r0, c3)}
    if random.random() < 0.5:
        return red_cells, green_cells
    return green_cells, red_cells


def _gen_attempt() -> np.ndarray | None:
    n_cells = random.choice(CELL_COUNTS)
    half_size = n_cells * (CELL_SIZE + 1)
    height = half_size
    width = half_size * 2
    side = random.choice(("left", "right"))
    row_anchor = random.choice((0, CELL_SIZE))
    col_anchor = 0 if side == "left" else CELL_SIZE

    occupied = np.zeros((height, half_size), dtype=int)
    occupied[_wall_indices(n_cells, row_anchor), :] = TEAL
    occupied[:, _wall_indices(n_cells, col_anchor)] = TEAL

    row_bands, col_bands = _extract_lattice(occupied)
    offset = (random.randint(0, CELL_SIZE - 1), random.randint(0, CELL_SIZE - 1))

    red_cells, green_cells = _coarse_template(n_cells)
    red_fill = _fill_cells_from_markers({RED: set(red_cells), GREEN: set()})
    green_fill = _fill_cells_from_markers({RED: set(), GREEN: set(green_cells)})
    red_fill_cells = {cell for cell, colors in red_fill.items() if colors}
    green_fill_cells = {cell for cell, colors in green_fill.items() if colors}

    intersection = red_fill_cells & green_fill_cells
    if not intersection:
        return None
    if not (red_fill_cells - green_fill_cells) or not (green_fill_cells - red_fill_cells):
        return None
    if not (red_fill_cells - set(red_cells)) or not (green_fill_cells - set(green_cells)):
        return None
    if intersection & (set(red_cells) | set(green_cells)):
        return None
    if (red_fill_cells & set(green_cells)) or (green_fill_cells & set(red_cells)):
        return None

    for cell in red_cells:
        r, c = _marker_pixel(row_bands, col_bands, cell, offset)
        occupied[r, c] = RED
    for cell in green_cells:
        r, c = _marker_pixel(row_bands, col_bands, cell, offset)
        occupied[r, c] = GREEN

    grid = np.zeros((height, width), dtype=int)
    if side == "left":
        grid[:, :half_size] = occupied
    else:
        grid[:, half_size:] = occupied

    if not _semantic_check(grid):
        return None
    expected = _expected_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    side, occupied_half, _blank_half = _split_input_halves(grid)
    completed_half, _markers, _filled_cells = _filled_half(occupied_half)
    mirrored_half = _swap_red_green(np.fliplr(completed_half))
    if side == "left":
        return np.concatenate((completed_half, mirrored_half), axis=1)
    return np.concatenate((mirrored_half, completed_half), axis=1)


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        side, occupied_half, blank_half = _split_input_halves(grid)
        completed_half, markers, filled_cells = _filled_half(occupied_half)
    except Exception:
        return False

    if np.any(blank_half != BLACK):
        return False
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False

    magenta_cells = {cell for cell, colors in filled_cells.items() if colors == {RED, GREEN}}
    red_only_cells = {cell for cell, colors in filled_cells.items() if colors == {RED}}
    green_only_cells = {cell for cell, colors in filled_cells.items() if colors == {GREEN}}
    if not magenta_cells or not red_only_cells or not green_only_cells:
        return False

    if len(markers[RED]) < 3 or len(markers[GREEN]) < 3:
        return False
    if not np.any(completed_half != occupied_half):
        return False

    output = main(grid)
    half_w = occupied_half.shape[1]
    if side == "left":
        left_out = output[:, :half_w]
        right_out = output[:, half_w:]
        if not np.array_equal(left_out, completed_half):
            return False
        if not np.array_equal(right_out, _swap_red_green(np.fliplr(completed_half))):
            return False
    else:
        left_out = output[:, :half_w]
        right_out = output[:, half_w:]
        if not np.array_equal(right_out, completed_half):
            return False
        if not np.array_equal(left_out, _swap_red_green(np.fliplr(completed_half))):
            return False

    # Require a non-marker pink cell so the intersection must be inferred from fills.
    row_bands, col_bands = _extract_lattice(occupied_half)
    pink_non_marker = False
    for cell in magenta_cells:
        rr, cc = _marker_pixel(row_bands, col_bands, cell, (0, 0))
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        sub = occupied_half[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any((sub == RED) | (sub == GREEN)):
            pink_non_marker = True
            break
    return pink_non_marker


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
