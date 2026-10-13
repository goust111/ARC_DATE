from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PINK = 6
TEAL = 8

CELL_SIZE = 3
WALL = 1
N_CELL_OPTIONS = (4, 5)
MAX_GENERATION_ATTEMPTS = 500
MARKER_COLORS = (RED, GREEN)
SYMMETRY_PATTERNS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.ones((3, 3), dtype=bool),
)


def _cell_ranges(length: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for wall in sorted(wall_indices):
        if start <= wall - 1:
            ranges.append((start, wall - 1))
        start = wall + 1
    if start <= length - 1:
        ranges.append((start, length - 1))
    return ranges


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    matches = [idx for idx, (lo, hi) in enumerate(bands) if lo <= pos <= hi]
    assert len(matches) == 1
    return matches[0]


def _wall_indices(n_cells: int, cell_size: int = CELL_SIZE) -> list[int]:
    return [i * (cell_size + WALL) for i in range(n_cells + 1)]


def _grid_size(n_cells: int, cell_size: int = CELL_SIZE) -> int:
    return n_cells * cell_size + (n_cells + 1) * WALL


def _render_base_canvas(n_cells: int, cell_size: int = CELL_SIZE) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    size = _grid_size(n_cells, cell_size)
    grid = np.full((size, size), BLACK, dtype=int)
    walls = _wall_indices(n_cells, cell_size)
    grid[walls, :] = TEAL
    grid[:, walls] = TEAL
    bands = _cell_ranges(size, walls)
    return grid, bands, bands


def _qualifying_groups(marker_cells: list[tuple[int, int]]) -> tuple[dict[int, list[int]], dict[int, list[int]], set[tuple[int, int]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in marker_cells:
        by_row[int(r)].append(int(c))
        by_col[int(c)].append(int(r))

    used: set[tuple[int, int]] = set()
    for r, cols in by_row.items():
        if len(cols) >= 2:
            used.update((r, c) for c in cols)
    for c, rows in by_col.items():
        if len(rows) >= 2:
            used.update((r, c) for r in rows)
    return by_row, by_col, used


def _base_fill_cells(marker_cells: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int, set[tuple[int, int]]]:
    by_row, by_col, used = _qualifying_groups(marker_cells)
    filled: set[tuple[int, int]] = set()
    row_groups = 0
    col_groups = 0
    for r, cols in by_row.items():
        if len(cols) >= 2:
            row_groups += 1
            for c in range(min(cols), max(cols) + 1):
                filled.add((r, c))
    for c, rows in by_col.items():
        if len(rows) >= 2:
            col_groups += 1
            for r in range(min(rows), max(rows) + 1):
                filled.add((r, c))
    return filled, row_groups, col_groups, used


def _orbit_cells(
    cells: set[tuple[int, int]],
    axis_r: int,
    axis_c: int,
    n_rows: int,
    n_cols: int,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr, cc in (
            (r, c),
            (2 * axis_r - r, c),
            (r, 2 * axis_c - c),
            (2 * axis_r - r, 2 * axis_c - c),
        ):
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                out.add((int(rr), int(cc)))
    return out


def _render_output_from_sets(
    shape: tuple[int, int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sym_cell: tuple[int, int],
    sym_pattern: np.ndarray,
    red_cells: set[tuple[int, int]],
    green_cells: set[tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[wall_rows, :] = TEAL
    out[:, wall_cols] = TEAL

    for cell in sorted(red_cells | green_cells):
        if cell == sym_cell:
            continue
        rr, cc = cell
        r0, r1 = row_bands[rr]
        c0, c1 = col_bands[cc]
        if cell in red_cells and cell in green_cells:
            fill_color = PINK
        elif cell in red_cells:
            fill_color = RED
        else:
            fill_color = GREEN
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    sr, sc = sym_cell
    r0, r1 = row_bands[sr]
    c0, c1 = col_bands[sc]
    out[r0 : r1 + 1, c0 : c1 + 1] = np.where(sym_pattern, GREY, BLACK)
    return out


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    H, W = g.shape

    wall_rows = [r for r in range(H) if np.all(g[r, :] == TEAL)]
    wall_cols = [c for c in range(W) if np.all(g[:, c] == TEAL)]
    assert len(wall_rows) >= 5 and len(wall_cols) >= 5
    assert wall_rows[0] == 0 and wall_rows[-1] == H - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == W - 1

    row_bands = _cell_ranges(H, wall_rows)
    col_bands = _cell_ranges(W, wall_cols)
    assert len(row_bands) == len(col_bands) in N_CELL_OPTIONS

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    assert next(iter(row_sizes)) == next(iter(col_sizes))

    sym_cells: list[tuple[int, int]] = []
    sym_pattern: np.ndarray | None = None
    markers_by_color: dict[int, list[tuple[int, int]]] = {RED: [], GREEN: []}

    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            has_grey = np.any(cell == GREY)
            has_red = np.any(cell == RED)
            has_green = np.any(cell == GREEN)

            if has_grey:
                assert not has_red and not has_green
                mask = cell == GREY
                assert np.array_equal(mask, np.flipud(mask))
                assert np.array_equal(mask, np.fliplr(mask))
                sym_cells.append((rr, cc))
                sym_pattern = mask.copy()
            elif has_red or has_green:
                assert not (has_red and has_green)
                color = RED if has_red else GREEN
                assert np.count_nonzero(cell == color) == 1
                assert np.count_nonzero((cell != BLACK) & (cell != color)) == 0
                markers_by_color[color].append((rr, cc))
            else:
                assert np.count_nonzero(cell != BLACK) == 0

    assert len(sym_cells) == 1 and sym_pattern is not None
    sym_cell = sym_cells[0]
    return {
        "grid": g,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "n_cells": len(row_bands),
        "sym_cell": sym_cell,
        "sym_pattern": sym_pattern,
        "markers_by_color": markers_by_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill lattice cells between same-color markers, then mirror those filled cells across the offset axes borrowed from the grey symmetric marker cell.

    Input structure:
    - A black canvas is split by one-pixel teal grid lines into a 4x4 or 5x5 array of equal square cells.
    - Exactly one cell contains a grey pattern that is symmetric both horizontally and vertically inside that cell.
    - Other cells may contain one centered red or green marker pixel; same-color markers align in shared cell rows and columns.

    Rule:
    - Treat the grey pattern's cell center as both the horizontal and vertical reflection axes in the coarse cell lattice.
    - For each color separately, fill every whole cell on each qualifying marker row and qualifying marker column between the extreme same-color markers, inclusive.
    - Reflect the filled red-cell set and the filled green-cell set across both borrowed axes; preserve only in-bounds reflected cells.
    - Paint red-only cells red, green-only cells green, red/green overlaps pink, keep teal grid lines, and leave the grey symmetric cell unchanged.

    Key insight:
    - The endpoints act on the coarse cell grid, but the mirror axes do not come from the grid center; they are borrowed from the off-center symmetric marker cell, so locating that cell is necessary before reflecting the fills.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    row_bands = parsed["row_bands"]
    col_bands = parsed["col_bands"]
    sym_r, sym_c = parsed["sym_cell"]
    n_cells = parsed["n_cells"]

    filled_by_color: dict[int, set[tuple[int, int]]] = {}
    total_row_groups = 0
    total_col_groups = 0
    for color in MARKER_COLORS:
        marker_cells = parsed["markers_by_color"][color]
        filled, row_groups, col_groups, used = _base_fill_cells(marker_cells)
        assert 2 <= row_groups + col_groups <= 3
        assert set(marker_cells) == used
        total_row_groups += row_groups
        total_col_groups += col_groups
        filled_by_color[color] = _orbit_cells(filled, sym_r, sym_c, n_cells, n_cells)

    assert total_row_groups >= 1 and total_col_groups >= 1
    return _render_output_from_sets(
        parsed["grid"].shape,
        row_bands,
        col_bands,
        parsed["sym_cell"],
        parsed["sym_pattern"],
        filled_by_color[RED],
        filled_by_color[GREEN],
        parsed["wall_rows"],
        parsed["wall_cols"],
    )


def _preferred_positions(index: int, n_cells: int) -> list[int]:
    far = [pos for pos in range(n_cells) if pos != index and abs(pos - index) >= 2]
    if far:
        return far
    return [pos for pos in range(n_cells) if pos != index]


def _symmetry_cell_choices(n_cells: int) -> list[tuple[int, int]]:
    if n_cells == 4:
        return [(1, 1), (1, 2), (2, 1), (2, 2)]
    return [(1, 1), (1, 3), (3, 1), (3, 3)]


def _build_l_markers(n_cells: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    for _ in range(80):
        r = random.randrange(n_cells)
        c = random.randrange(n_cells)
        pivot = (r, c)
        if pivot in blocked:
            continue
        h_choices = [cc for cc in _preferred_positions(c, n_cells) if (r, cc) not in blocked]
        v_choices = [rr for rr in _preferred_positions(r, n_cells) if (rr, c) not in blocked]
        if not h_choices or not v_choices:
            continue
        return [pivot, (r, random.choice(h_choices)), (random.choice(v_choices), c)]
    return None


def _build_parallel_markers(n_cells: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    orientation = random.choice(("rows", "cols"))
    for _ in range(120):
        if orientation == "rows":
            if n_cells < 4:
                return None
            r1, r2 = random.sample(range(n_cells), 2)
            cols = random.sample(range(n_cells), 4)
            c1, c2, c3, c4 = cols
            markers = [(r1, c1), (r1, c2), (r2, c3), (r2, c4)]
        else:
            if n_cells < 4:
                return None
            c1, c2 = random.sample(range(n_cells), 2)
            rows = random.sample(range(n_cells), 4)
            r1, r2, r3, r4 = rows
            markers = [(r1, c1), (r2, c1), (r3, c2), (r4, c2)]
        if len(set(markers)) != 4 or any(cell in blocked for cell in markers):
            continue
        _filled, groups, _other_groups, used = _base_fill_cells(markers) if orientation == "rows" else (
            lambda data: (data[0], data[2], data[1], data[3])
        )(_base_fill_cells(markers))
        if groups != 2 or used != set(markers):
            continue
        return markers
    return None


def _build_fork_markers(n_cells: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    orientation = random.choice(("row-heavy", "col-heavy"))
    for _ in range(160):
        if orientation == "row-heavy":
            if n_cells < 3:
                return None
            r1, r2, r3 = random.sample(range(n_cells), 3)
            c1, c2, c3 = random.sample(range(n_cells), 3)
            markers = [(r1, c1), (r1, c2), (r2, c1), (r2, c3), (r3, c1)]
        else:
            if n_cells < 3:
                return None
            r1, r2, r3 = random.sample(range(n_cells), 3)
            c1, c2, c3 = random.sample(range(n_cells), 3)
            markers = [(r1, c1), (r2, c1), (r1, c2), (r3, c2), (r1, c3)]
        if len(set(markers)) != 5 or any(cell in blocked for cell in markers):
            continue
        filled, row_groups, col_groups, used = _base_fill_cells(markers)
        if not filled or used != set(markers) or row_groups + col_groups != 3:
            continue
        return markers
    return None


def _sample_markers(n_cells: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    builders = [
        _build_l_markers,
        _build_parallel_markers,
        _build_fork_markers,
    ]
    random.shuffle(builders)
    for builder in builders:
        markers = builder(n_cells, blocked)
        if markers is not None:
            return markers
    return None


def _render_input_from_model(model: dict) -> np.ndarray:
    n_cells = model["n_cells"]
    sym_cell = model["sym_cell"]
    sym_pattern = model["sym_pattern"]
    grid, row_bands, col_bands = _render_base_canvas(n_cells)

    sr, sc = sym_cell
    r0, r1 = row_bands[sr]
    c0, c1 = col_bands[sc]
    grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(sym_pattern, GREY, BLACK)

    for color, markers in model["markers_by_color"].items():
        for rr, cc in markers:
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            grid[(r0 + r1) // 2, (c0 + c1) // 2] = color
    return grid


def _expected_output_from_model(model: dict) -> np.ndarray:
    n_cells = model["n_cells"]
    sym_r, sym_c = model["sym_cell"]
    grid, row_bands, col_bands = _render_base_canvas(n_cells)
    wall_rows = _wall_indices(n_cells)
    wall_cols = _wall_indices(n_cells)

    red_base, _, _, _ = _base_fill_cells(model["markers_by_color"][RED])
    green_base, _, _, _ = _base_fill_cells(model["markers_by_color"][GREEN])
    red_cells = _orbit_cells(red_base, sym_r, sym_c, n_cells, n_cells)
    green_cells = _orbit_cells(green_base, sym_r, sym_c, n_cells, n_cells)
    return _render_output_from_sets(
        grid.shape,
        row_bands,
        col_bands,
        model["sym_cell"],
        model["sym_pattern"],
        red_cells,
        green_cells,
        wall_rows,
        wall_cols,
    )


def _model_ok(model: dict) -> bool:
    n_cells = model["n_cells"]
    sym_r, sym_c = model["sym_cell"]
    total_row_groups = 0
    total_col_groups = 0
    all_base: set[tuple[int, int]] = set()
    h_added: set[tuple[int, int]] = set()
    v_added: set[tuple[int, int]] = set()

    red_base, red_rows, red_cols, red_used = _base_fill_cells(model["markers_by_color"][RED])
    green_base, green_rows, green_cols, green_used = _base_fill_cells(model["markers_by_color"][GREEN])

    if red_used != set(model["markers_by_color"][RED]) or green_used != set(model["markers_by_color"][GREEN]):
        return False
    if not (2 <= red_rows + red_cols <= 3 and 2 <= green_rows + green_cols <= 3):
        return False

    for base_cells, row_groups, col_groups in (
        (red_base, red_rows, red_cols),
        (green_base, green_rows, green_cols),
    ):
        total_row_groups += row_groups
        total_col_groups += col_groups
        all_base |= base_cells
        for r, c in base_cells:
            hr = 2 * sym_r - r
            vc = 2 * sym_c - c
            if 0 <= hr < n_cells:
                h_added.add((hr, c))
            if 0 <= vc < n_cells:
                v_added.add((r, vc))

    if total_row_groups < 1 or total_col_groups < 1:
        return False

    red_final = _orbit_cells(red_base, sym_r, sym_c, n_cells, n_cells)
    green_final = _orbit_cells(green_base, sym_r, sym_c, n_cells, n_cells)
    visible_cells = (red_final | green_final) - {model["sym_cell"]}
    reflection_gain = (red_final - red_base) | (green_final - green_base)

    if len(visible_cells) < 6 or len(visible_cells) > int(0.8 * n_cells * n_cells):
        return False
    if len(reflection_gain) < 2:
        return False
    if not (h_added - all_base) or not (v_added - all_base):
        return False
    if len((red_final & green_final) - {model["sym_cell"]}) > max(4, n_cells):
        return False

    inp = _render_input_from_model(model)
    expected = _expected_output_from_model(model)
    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_cells = random.choice(N_CELL_OPTIONS)
        sym_cell = random.choice(_symmetry_cell_choices(n_cells))
        model = {
            "n_cells": n_cells,
            "sym_cell": sym_cell,
            "sym_pattern": random.choice(SYMMETRY_PATTERNS).copy(),
            "markers_by_color": {},
        }

        blocked = {sym_cell}
        ok = True
        for color in MARKER_COLORS:
            markers = _sample_markers(n_cells, blocked)
            if markers is None:
                ok = False
                break
            model["markers_by_color"][color] = markers
            blocked.update(markers)
        if not ok:
            continue

        if not _model_ok(model):
            continue
        return _render_input_from_model(model)

    raise RuntimeError("failed to generate a valid input for ARC puzzle 6")
