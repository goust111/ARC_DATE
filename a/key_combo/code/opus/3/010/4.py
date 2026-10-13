from __future__ import annotations

import itertools
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

GEN_MAX_ATTEMPTS = 300
CELL_SIZE_CHOICES = (3, 4)
CELL_COUNT_RANGE = (5, 7)


def _bbox_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _open_runs(length: int, line_indices: np.ndarray | list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted({int(x) for x in line_indices}):
        if idx - prev > 1:
            runs.append((prev + 1, idx - 1))
        prev = idx
    if length - prev > 1:
        runs.append((prev + 1, length - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    v = int(value)
    for idx, (start, end) in enumerate(runs):
        if start <= v <= end:
            return idx
    raise ValueError(f"value {value} is not inside any run")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _is_connected(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    pts = {tuple(map(int, xy)) for xy in coords}
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _cell_runs_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 1
    for size in sizes:
        runs.append((start, start + int(size) - 1))
        start += int(size) + 1
    return runs


def _grid_extent(sizes: list[int]) -> int:
    return int(sum(sizes) + len(sizes) + 1)


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in sorted(set(map(int, np.unique(g)))):
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if (
            len(row_lines) >= 2
            and len(col_lines) >= 2
            and 0 in row_lines
            and h - 1 in row_lines
            and 0 in col_lines
            and w - 1 in col_lines
        ):
            row_cells = _open_runs(h, row_lines)
            col_cells = _open_runs(w, col_lines)
            if row_cells and col_cells:
                candidates.append((int(color), row_cells, col_cells))
    assert len(candidates) == 1, "expected a unique full-row/full-column lattice color"
    lattice_color, row_cells, col_cells = candidates[0]
    assert len(row_cells) >= 5 and len(col_cells) >= 5, "expected at least a 5x5 cell array"
    return lattice_color, row_cells, col_cells


def _locate_seed(
    grid: np.ndarray,
    lattice_color: int,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
) -> tuple[int, int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    seed_mask = (g != int(lattice_color)) & (g != GREY)
    seed_coords = np.argwhere(seed_mask)
    assert 2 <= len(seed_coords) <= 4, "seed motif should contain 2-4 colored pixels"
    seed_colors = {int(g[int(r), int(c)]) for r, c in seed_coords}
    assert seed_colors == {BLUE, YELLOW}, "seed motif should use blue and yellow"

    source_rows = {_run_index(row_cells, int(r)) for r, _c in seed_coords}
    source_cols = {_run_index(col_cells, int(c)) for _r, c in seed_coords}
    assert len(source_rows) == 1 and len(source_cols) == 1, "seed motif must sit in one cell"
    source_row = source_rows.pop()
    source_col = source_cols.pop()
    assert 0 < source_row < len(row_cells) - 1 and 0 < source_col < len(col_cells) - 1, "seed cell must be interior"

    cell_r0 = row_cells[source_row][0]
    cell_c0 = col_cells[source_col][0]
    offsets = [(int(r) - cell_r0, int(c) - cell_c0, int(g[int(r), int(c)])) for r, c in seed_coords]

    local_h = row_cells[source_row][1] - row_cells[source_row][0] + 1
    local_w = col_cells[source_col][1] - col_cells[source_col][0] + 1
    local_mask = np.zeros((local_h, local_w), dtype=bool)
    for dr, dc, _color in offsets:
        local_mask[int(dr), int(dc)] = True
    comps = _components(local_mask)
    assert len(comps) == 1, "seed motif must be 4-connected"
    return source_row, source_col, offsets


def _transform_with_metadata(
    grid: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    seed_row: int,
    seed_col: int,
    seed_pixels: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    last_row = len(row_cells) - 1
    last_col = len(col_cells) - 1

    for row_idx, (r0, r1) in enumerate(row_cells):
        for col_idx, (c0, c1) in enumerate(col_cells):
            if row_idx == seed_row and col_idx == seed_col:
                continue
            rs = slice(r0, r1 + 1)
            cs = slice(c0, c1 + 1)
            same_parity = (row_idx % 2 == seed_row % 2) and (col_idx % 2 == seed_col % 2)
            is_edge = row_idx in (0, last_row) or col_idx in (0, last_col)
            if same_parity and is_edge:
                out[rs, cs] = GREEN
            elif same_parity:
                out[rs, cs] = GREY
                for dr, dc, color in seed_pixels:
                    out[r0 + int(dr), c0 + int(dc)] = int(color)
            elif is_edge:
                out[rs, cs] = GREY
            else:
                out[rs, cs] = RED
    return out


def _cell_patch(grid: np.ndarray, row_cells: list[tuple[int, int]], col_cells: list[tuple[int, int]], row_idx: int, col_idx: int) -> np.ndarray:
    r0, r1 = row_cells[row_idx]
    c0, c1 = col_cells[col_idx]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _category_cells(row_count: int, col_count: int, seed_row: int, seed_col: int) -> dict[str, list[tuple[int, int]]]:
    categories = {
        "same_interior": [],
        "diff_interior": [],
        "same_edge": [],
        "diff_edge": [],
    }
    for row_idx in range(row_count):
        for col_idx in range(col_count):
            if row_idx == seed_row and col_idx == seed_col:
                continue
            same_parity = (row_idx % 2 == seed_row % 2) and (col_idx % 2 == seed_col % 2)
            is_edge = row_idx in (0, row_count - 1) or col_idx in (0, col_count - 1)
            if same_parity and is_edge:
                categories["same_edge"].append((row_idx, col_idx))
            elif same_parity:
                categories["same_interior"].append((row_idx, col_idx))
            elif is_edge:
                categories["diff_edge"].append((row_idx, col_idx))
            else:
                categories["diff_interior"].append((row_idx, col_idx))
    return categories


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the seed motif into same-parity interior cells while recoloring the other lattice cells by parity and edge status.

    Input structure:
    - Orange full-row and full-column lattice lines split the grid into a rectangular array of grey cell interiors.
    - Exactly one interior cell contains a small 4-connected blue/yellow seed motif; every other cell interior is grey.
    - The lattice array has at least five cell rows and five cell columns, so both edge and interior cells exist.

    Rule:
    - Detect the open cell runs between the orange lattice lines and index them from top to bottom and left to right.
    - Find the unique cell containing the blue/yellow motif and record its cell-row and cell-column parity together with each motif pixel's offset from that cell's upper-left corner.
    - Leave the seed cell unchanged, fill same-parity edge cells green, fill different-parity interior cells red, keep different-parity edge cells grey, and stamp the seed motif into every other same-parity interior cell.

    Key insight:
    - The output is determined by the Cartesian product of two independent classifications: whether a cell matches the seed cell's row/column parity pair, and whether that cell sits on the array perimeter or in the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    lattice_color, row_cells, col_cells = _parse_lattice(g)
    seed_row, seed_col, seed_pixels = _locate_seed(g, lattice_color, row_cells, col_cells)
    return _transform_with_metadata(g, row_cells, col_cells, seed_row, seed_col, seed_pixels)


def _build_pattern_shapes() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    shapes = set()
    for area in (2, 3, 4):
        for combo in itertools.combinations(cells, area):
            if not _is_connected(combo):
                continue
            r0, c0, _r1, _c1 = _bbox_coords(list(combo))
            norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in combo))
            shapes.add(norm)
    return sorted(shapes, key=lambda shape: (len(shape), max(r for r, _c in shape), max(c for _r, c in shape), shape))


PATTERN_SHAPES = _build_pattern_shapes()


def _sample_sizes() -> tuple[list[int], list[int]] | None:
    for _ in range(100):
        row_count = random.randint(CELL_COUNT_RANGE[0], CELL_COUNT_RANGE[1])
        col_count = random.randint(CELL_COUNT_RANGE[0], CELL_COUNT_RANGE[1])
        row_heights = [random.choice(CELL_SIZE_CHOICES) for _ in range(row_count)]
        col_widths = [random.choice(CELL_SIZE_CHOICES) for _ in range(col_count)]
        if _grid_extent(row_heights) <= 30 and _grid_extent(col_widths) <= 30:
            return row_heights, col_widths
    return None


def _choose_seed_cell(row_count: int, col_count: int) -> tuple[int, int, dict[str, list[tuple[int, int]]]] | None:
    candidates: list[tuple[int, int, dict[str, list[tuple[int, int]]]]] = []
    for seed_row in range(1, row_count - 1):
        for seed_col in range(1, col_count - 1):
            categories = _category_cells(row_count, col_count, seed_row, seed_col)
            if all(categories[name] for name in ("same_interior", "diff_interior", "same_edge", "diff_edge")):
                candidates.append((seed_row, seed_col, categories))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_seed_pixels(row_heights: list[int], col_widths: list[int]) -> tuple[int, int, list[tuple[int, int, int]]] | None:
    min_h = min(row_heights)
    min_w = min(col_widths)
    candidate_shapes = []
    for shape in PATTERN_SHAPES:
        pat_h = max(r for r, _c in shape) + 1
        pat_w = max(c for _r, c in shape) + 1
        if pat_h <= min_h and pat_w <= min_w:
            candidate_shapes.append(shape)
    if not candidate_shapes:
        return None

    shape = random.choice(candidate_shapes)
    pat_h = max(r for r, _c in shape) + 1
    pat_w = max(c for _r, c in shape) + 1
    off_r = random.randint(0, min_h - pat_h)
    off_c = random.randint(0, min_w - pat_w)

    colors = [BLUE] * len(shape)
    yellow_count = random.randint(1, len(shape) - 1)
    yellow_indices = random.sample(range(len(shape)), yellow_count)
    for idx in yellow_indices:
        colors[idx] = YELLOW
    seed_pixels = [(off_r + r, off_c + c, colors[idx]) for idx, (r, c) in enumerate(shape)]
    return off_r, off_c, seed_pixels


def _build_input(
    row_heights: list[int],
    col_widths: list[int],
    seed_row: int,
    seed_col: int,
    seed_pixels: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = _grid_extent(row_heights)
    w = _grid_extent(col_widths)
    grid = np.full((h, w), ORANGE, dtype=int)
    row_cells = _cell_runs_from_sizes(row_heights)
    col_cells = _cell_runs_from_sizes(col_widths)

    for r0, r1 in row_cells:
        for c0, c1 in col_cells:
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    seed_r0 = row_cells[seed_row][0]
    seed_c0 = col_cells[seed_col][0]
    for dr, dc, color in seed_pixels:
        grid[seed_r0 + int(dr), seed_c0 + int(dc)] = int(color)
    return grid, row_cells, col_cells


def _validate_generated(
    grid: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    seed_row: int,
    seed_col: int,
    seed_pixels: list[tuple[int, int, int]],
) -> bool:
    parsed_lattice, parsed_rows, parsed_cols = _parse_lattice(grid)
    assert parsed_lattice == ORANGE
    assert parsed_rows == row_cells
    assert parsed_cols == col_cells

    parsed_seed_row, parsed_seed_col, parsed_seed_pixels = _locate_seed(grid, parsed_lattice, parsed_rows, parsed_cols)
    assert parsed_seed_row == seed_row
    assert parsed_seed_col == seed_col
    assert sorted(parsed_seed_pixels) == sorted(seed_pixels)

    categories = _category_cells(len(row_cells), len(col_cells), seed_row, seed_col)
    assert all(categories[name] for name in ("same_interior", "diff_interior", "same_edge", "diff_edge"))

    expected = _transform_with_metadata(grid, row_cells, col_cells, seed_row, seed_col, seed_pixels)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.any(actual != grid), "output should not equal input"

    same_interior = categories["same_interior"][0]
    diff_interior = categories["diff_interior"][0]
    same_edge = categories["same_edge"][0]
    diff_edge = categories["diff_edge"][0]

    same_patch = _cell_patch(actual, row_cells, col_cells, *same_interior)
    diff_patch = _cell_patch(actual, row_cells, col_cells, *diff_interior)
    edge_same_patch = _cell_patch(actual, row_cells, col_cells, *same_edge)
    edge_diff_patch = _cell_patch(actual, row_cells, col_cells, *diff_edge)

    assert np.all(diff_patch == RED), "different-parity interior cells should be fully red"
    assert np.all(edge_same_patch == GREEN), "same-parity edge cells should be fully green"
    assert np.all(edge_diff_patch == GREY), "different-parity edge cells should stay grey"
    assert BLUE in same_patch and YELLOW in same_patch and GREY in same_patch, "stamped cells should show the seed on grey"

    assert not np.array_equal(same_patch, edge_same_patch), "position type must affect same-parity cells"
    assert not np.array_equal(diff_patch, edge_diff_patch), "position type must affect different-parity cells"
    assert not np.array_equal(same_patch, diff_patch), "parity must affect interior cells"
    assert not np.array_equal(edge_same_patch, edge_diff_patch), "parity must affect edge cells"

    seed_patch = _cell_patch(actual, row_cells, col_cells, seed_row, seed_col)
    for dr, dc, color in seed_pixels:
        assert seed_patch[int(dr), int(dc)] == int(color)
    assert seed_patch.shape == _cell_patch(grid, row_cells, col_cells, seed_row, seed_col).shape
    return True


def _gen_attempt() -> np.ndarray | None:
    size_choice = _sample_sizes()
    if size_choice is None:
        return None
    row_heights, col_widths = size_choice

    seed_choice = _choose_seed_cell(len(row_heights), len(col_widths))
    if seed_choice is None:
        return None
    seed_row, seed_col, _categories = seed_choice

    pattern_choice = _choose_seed_pixels(row_heights, col_widths)
    if pattern_choice is None:
        return None
    _off_r, _off_c, seed_pixels = pattern_choice

    grid, row_cells, col_cells = _build_input(row_heights, col_widths, seed_row, seed_col, seed_pixels)
    try:
        _validate_generated(grid, row_cells, col_cells, seed_row, seed_col, seed_pixels)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
