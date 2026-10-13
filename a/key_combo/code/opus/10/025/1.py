from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    TEAL = 8


Cell = tuple[int, int]
PixelMarker = tuple[int, int, int]

ORIENTATIONS = ("horizontal", "vertical")
CELL_INTERIOR_SIZES = (2, 3)
GRID_CELL_COUNT = 4
MAX_GENERATION_ATTEMPTS = 300


def _trace_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False


def _full_line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _cell_bands(walls: list[int]) -> list[tuple[int, int]]:
    assert len(walls) >= 2, "need at least two lattice walls"
    gaps = [b - a for a, b in zip(walls, walls[1:])]
    assert len(set(gaps)) == 1, "lattice spacing must be uniform"
    bands = []
    for a, b in zip(walls, walls[1:]):
        assert b - a >= 2, "each lattice cell needs positive interior area"
        bands.append((a + 1, b - 1))
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int | None:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= pos <= hi:
            return idx
    return None


def _lattice_structure(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    wall_rows = _full_line_positions(grid, Color.TEAL, axis=0)
    wall_cols = _full_line_positions(grid, Color.TEAL, axis=1)
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected a teal lattice"
    row_bands = _cell_bands(wall_rows)
    col_bands = _cell_bands(wall_cols)
    return wall_rows, wall_cols, row_bands, col_bands


def _decode_vector(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> tuple[int, int]:
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.YELLOW)]
    grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == Color.GREY)}
    assert len(yellow) == 2, "expected exactly two yellow path endpoints"
    assert 3 <= len(grey) <= 5, "expected a short grey connector path"
    start = min(yellow, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    end = max(yellow, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    corner_row = wall_rows[0]
    corner_col = wall_cols[0]
    for r, c in yellow:
        assert r < corner_row and c < corner_col, "path endpoints must stay in the corner outside the lattice"
    for r, c in grey:
        assert r < corner_row and c < corner_col, "grey path must stay in the corner outside the lattice"
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in grey | set(yellow):
        mask[r, c] = True
    assert _trace_path(mask, start, end), "path pixels must connect the two endpoints"
    return int(end[0] - start[0]), int(end[1] - start[1])


def _collect_original_markers(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> list[PixelMarker]:
    markers: list[PixelMarker] = []
    for color in (Color.RED, Color.GREEN):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) >= 2, "each marker color needs at least two pixels"
        for r, c in coords:
            row_idx = _band_index(r, row_bands)
            col_idx = _band_index(c, col_bands)
            assert row_idx is not None and col_idx is not None, "markers must lie inside lattice cells"
            markers.append((color, r, c))
    return markers


def _register_markers(
    markers: list[PixelMarker],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[dict[int, set[Cell]], list[PixelMarker]]:
    cells_by_color: dict[int, set[Cell]] = defaultdict(set)
    visible_markers: list[PixelMarker] = []
    occupied_pixels: set[tuple[int, int]] = set()

    def add_marker(color: int, r: int, c: int) -> bool:
        row_idx = _band_index(r, row_bands)
        col_idx = _band_index(c, col_bands)
        if row_idx is None or col_idx is None:
            return False
        if (r, c) not in occupied_pixels:
            occupied_pixels.add((r, c))
            visible_markers.append((color, r, c))
        cells_by_color[color].add((row_idx, col_idx))
        return True

    for color, r, c in markers:
        ok = add_marker(color, r, c)
        assert ok
    for color, r, c in markers:
        add_marker(color, r + dr, c + dc)
    return cells_by_color, visible_markers


def _fill_cells(cells_by_color: dict[int, set[Cell]]) -> dict[Cell, set[int]]:
    fills: dict[Cell, set[int]] = defaultdict(set)
    for color, marker_cells in cells_by_color.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for row_idx, col_idx in sorted(marker_cells):
            by_row[row_idx].append(col_idx)
            by_col[col_idx].append(row_idx)

        for row_idx, cols in by_row.items():
            if len(cols) < 2:
                continue
            for col_idx in range(min(cols), max(cols) + 1):
                fills[(row_idx, col_idx)].add(color)
        for col_idx, rows in by_col.items():
            if len(rows) < 2:
                continue
            for row_idx in range(min(rows), max(rows) + 1):
                fills[(row_idx, col_idx)].add(color)
    return fills


def _paint_cell(grid: np.ndarray, band_r: tuple[int, int], band_c: tuple[int, int], color: int) -> None:
    r0, r1 = band_r
    c0, c1 = band_c
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _render_output(
    shape: tuple[int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    fills: dict[Cell, set[int]],
    visible_markers: list[PixelMarker],
) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    out[wall_rows, :] = Color.TEAL
    out[:, wall_cols] = Color.TEAL

    for (row_idx, col_idx), colors in fills.items():
        if colors == {Color.RED, Color.GREEN}:
            color = Color.PINK
        else:
            assert len(colors) == 1
            color = next(iter(colors))
        _paint_cell(out, row_bands[row_idx], col_bands[col_idx], color)

    for color, r, c in visible_markers:
        out[r, c] = color
    return out


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[Cell, set[int]], dict[Cell, set[int]], list[PixelMarker]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    wall_rows, wall_cols, row_bands, col_bands = _lattice_structure(g)
    dr, dc = _decode_vector(g, wall_rows, wall_cols)
    original_markers = _collect_original_markers(g, row_bands, col_bands)

    original_cells: dict[int, set[Cell]] = defaultdict(set)
    for color, r, c in original_markers:
        original_cells[color].add((_band_index(r, row_bands), _band_index(c, col_bands)))

    cells_by_color, visible_markers = _register_markers(original_markers, row_bands, col_bands, dr, dc)
    fills_with_echoes = _fill_cells(cells_by_color)
    fills_without_echoes = _fill_cells(original_cells)
    out = _render_output(g.shape, wall_rows, wall_cols, row_bands, col_bands, fills_with_echoes, visible_markers)
    return out, fills_with_echoes, fills_without_echoes, visible_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the path displacement, echo every lattice marker by that offset, and fill lattice rows and columns between same-color marked cells.

    Input structure:
    - A teal one-pixel lattice occupies the lower-right part of the canvas and divides it into equal black cells.
    - Red and green single-pixel markers lie inside lattice-cell interiors.
    - The top-left corner outside the lattice contains two yellow endpoints connected by a short grey path.

    Rule:
    - Take the yellow endpoint nearer the top-left corner as the path start and the other as the end; their coordinate difference is the displacement vector.
    - For each red or green marker, create a virtual echo marker at the displaced pixel position and keep it only if it lands inside another lattice-cell interior.
    - Combine original and echoed markers. For each color, any lattice row with at least two same-color marked cells is filled from the leftmost to the rightmost such cell, and any lattice column with at least two same-color marked cells is filled from the topmost to the bottommost such cell.
    - Fill whole cell interiors, preserve the teal lattice, color cells covered by both red and green fills pink, erase the grey path and yellow endpoints, and draw both original and valid echo markers on top of the fills.

    Key insight:
    - The grey path does not move existing objects; it only supplies one shared offset that determines which additional lattice cells become marked before the coarse row-and-column fill is recomputed.
    """
    out, _fills_with_echoes, _fills_without_echoes, _visible_markers = _solve_details(grid)
    return out


def _bands_from_walls(walls: list[int]) -> list[tuple[int, int]]:
    return [(a + 1, b - 1) for a, b in zip(walls, walls[1:])]


def _cell_pixel(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], cell: Cell, offset: tuple[int, int]) -> tuple[int, int]:
    row_idx, col_idx = cell
    band_r = row_bands[row_idx]
    band_c = col_bands[col_idx]
    return band_r[0] + offset[0], band_c[0] + offset[1]


def _path_points(start: tuple[int, int], end: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sr, sc = start
    er, ec = end
    assert sr == er or sc == ec
    endpoints = [start, end]
    grey: list[tuple[int, int]] = []
    if sr == er:
        step = 1 if ec > sc else -1
        for c in range(sc + step, ec, step):
            grey.append((sr, c))
    else:
        step = 1 if er > sr else -1
        for r in range(sr + step, er, step):
            grey.append((r, sc))
    return endpoints, grey


def _base_marker_cells(orientation: str, shift_cells: int) -> tuple[list[Cell], list[Cell]]:
    if orientation == "horizontal":
        anchor_col = 0 if shift_cells == 2 else random.randint(0, 1)
        red_cells = [(1, anchor_col), (2, anchor_col + 1)]
        green_cells = [(0, anchor_col + shift_cells), (3, anchor_col + shift_cells)]
        return red_cells, green_cells

    anchor_row = 0 if shift_cells == 2 else random.randint(0, 1)
    red_cells = [(anchor_row, 1), (anchor_row + 1, 2)]
    green_cells = [(anchor_row + shift_cells, 0), (anchor_row + shift_cells, 3)]
    return red_cells, green_cells


def _build_input() -> np.ndarray:
    orientation = random.choice(ORIENTATIONS)
    cell_h = random.choice(CELL_INTERIOR_SIZES)
    cell_w = random.choice(CELL_INTERIOR_SIZES)
    step_h = cell_h + 1
    step_w = cell_w + 1

    if orientation == "horizontal":
        shift_cells = 1 if step_w == 4 else 2
        dr, dc = 0, shift_cells * step_w
        row_margin = random.randint(4, 6)
        col_margin = random.randint(dc + 2, dc + 4)
    else:
        shift_cells = 1 if step_h == 4 else 2
        dr, dc = shift_cells * step_h, 0
        row_margin = random.randint(dr + 2, dr + 4)
        col_margin = random.randint(4, 6)

    n_rows = GRID_CELL_COUNT
    n_cols = GRID_CELL_COUNT
    height = row_margin + n_rows * step_h + 1
    width = col_margin + n_cols * step_w + 1
    assert height <= 30 and width <= 30

    row_walls = [row_margin + i * step_h for i in range(n_rows + 1)]
    col_walls = [col_margin + i * step_w for i in range(n_cols + 1)]
    row_bands = _bands_from_walls(row_walls)
    col_bands = _bands_from_walls(col_walls)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[row_walls, :] = Color.TEAL
    grid[:, col_walls] = Color.TEAL

    if orientation == "horizontal":
        start = (random.randint(1, row_margin - 2), random.randint(1, col_margin - dc - 1))
    else:
        start = (random.randint(1, row_margin - dr - 1), random.randint(1, col_margin - 2))
    end = (start[0] + dr, start[1] + dc)
    endpoints, grey = _path_points(start, end)
    for r, c in grey:
        grid[r, c] = Color.GREY
    for r, c in endpoints:
        grid[r, c] = Color.YELLOW

    red_cells, green_cells = _base_marker_cells(orientation, shift_cells)
    occupied_pixels: set[tuple[int, int]] = set(endpoints) | set(grey)
    for color, cells in ((Color.RED, red_cells), (Color.GREEN, green_cells)):
        for row_idx, col_idx in cells:
            offset = (random.randint(0, cell_h - 1), random.randint(0, cell_w - 1))
            r, c = _cell_pixel(row_bands, col_bands, (row_idx, col_idx), offset)
            assert (r, c) not in occupied_pixels
            occupied_pixels.add((r, c))
            grid[r, c] = color

    return grid


def _has_new_fill_due_to_echoes(fills_with_echoes: dict[Cell, set[int]], fills_without_echoes: dict[Cell, set[int]]) -> bool:
    cells = set(fills_with_echoes) | set(fills_without_echoes)
    for cell in cells:
        if fills_with_echoes.get(cell, set()) != fills_without_echoes.get(cell, set()):
            return True
    return False


def _semantic_check(inp: np.ndarray) -> None:
    out, fills_with_echoes, fills_without_echoes, visible_markers = _solve_details(inp)
    assert _has_new_fill_due_to_echoes(fills_with_echoes, fills_without_echoes), "echoes must create at least one new fill"
    assert any(colors == {Color.RED, Color.GREEN} for colors in fills_with_echoes.values()), "need at least one mixed red-green cell"

    red_pixels = sum(1 for color, _r, _c in visible_markers if color == Color.RED)
    green_pixels = sum(1 for color, _r, _c in visible_markers if color == Color.GREEN)
    assert red_pixels >= 3 and green_pixels >= 2, "valid echoes should stay visible in the output"

    assert np.count_nonzero(out == Color.YELLOW) == 0
    assert np.count_nonzero(out == Color.GREY) == 0


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _build_input()
        try:
            _semantic_check(inp)
            out = main(inp)
            assert out.shape == inp.shape
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid non-degenerate input")
