from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME = 1
SEP = 5
REF_COLORS = (2, 3, 4, 6, 7, 8, 9)
AREA_SHAPES = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2), (1, 4), (4, 1)],
}
NON_MONOTONIC_ORDERS = [
    perm
    for perm in itertools.permutations((1, 2, 3, 4))
    if perm not in ((1, 2, 3, 4), (4, 3, 2, 1))
]
MAX_GENERATION_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _parse_reference_line(line: np.ndarray) -> dict[int, int] | None:
    if np.any((line != BG) & (line == SEP)):
        return None

    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(line):
        color = int(line[i])
        if color == BG:
            i += 1
            continue
        j = i + 1
        while j < len(line) and int(line[j]) == color:
            j += 1
        runs.append((color, j - i))
        i = j

    if len(runs) != 4:
        return None
    lengths = sorted(length for _color, length in runs)
    if lengths != [1, 2, 3, 4]:
        return None
    if len({color for color, _length in runs}) != 4:
        return None
    return {length: color for color, length in runs}


def _reference_mapping(strip: np.ndarray, horizontal: bool) -> dict[int, int]:
    mappings = []
    if horizontal:
        for r in range(strip.shape[0]):
            mapping = _parse_reference_line(strip[r, :])
            if mapping is not None:
                mappings.append(mapping)
    else:
        for c in range(strip.shape[1]):
            mapping = _parse_reference_line(strip[:, c])
            if mapping is not None:
                mappings.append(mapping)
    assert len(mappings) == 1, "expected exactly one valid reference bar"
    return mappings[0]


def _cell_geometry(cell: np.ndarray) -> tuple[int, int, int, int, int]:
    coords = np.argwhere(cell == FRAME)
    assert len(coords) > 0, "each cell must contain a frame"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty hole"

    box = cell[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(box, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(box[border] == FRAME), "frame border must be a rectangle"
    assert np.all(box[~border] == BG), "hole must be black in the input"

    outside = cell.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BG
    assert not np.any(outside == FRAME), "cell must contain one hollow rectangle"

    area = int(np.count_nonzero(box[~border] == BG))
    assert 1 <= area <= 4, "hole area must be in 1..4"
    return r0, c0, r1, c1, area


def _solve_bottom(g: np.ndarray, full_rows: list[int]) -> np.ndarray:
    h, w = g.shape
    boundary_r = full_rows[-1]
    row_bands = _bands(boundary_r + 1, full_rows)
    assert len(row_bands) in (3, 4), "expected a 3x3 or 4x4 lattice"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    assert len(row_heights) == 1, "row bands must be uniform"

    col_separators = [c for c in range(w) if np.all(g[: boundary_r + 1, c] == SEP)]
    col_bands = _bands(w, col_separators)
    assert len(col_bands) == len(row_bands), "lattice must be square"
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(col_widths) == 1, "column bands must be uniform"

    mapping = _reference_mapping(g[boundary_r + 1 :, :], horizontal=True)
    out = g.copy()

    last = len(row_bands) - 1
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            fr0, fc0, fr1, fc1, area = _cell_geometry(g[r0 : r1 + 1, c0 : c1 + 1])
            color = mapping[area]
            ar0 = r0 + fr0
            ac0 = c0 + fc0
            ar1 = r0 + fr1
            ac1 = c0 + fc1

            if ri in (0, last) or ci in (0, last):
                out[ar0, ac0 : ac1 + 1] = color
                out[ar1, ac0 : ac1 + 1] = color
                out[ar0 : ar1 + 1, ac0] = color
                out[ar0 : ar1 + 1, ac1] = color
            else:
                out[ar0 + 1 : ar1, ac0 + 1 : ac1] = color
    return out


def _solve_right(g: np.ndarray, full_cols: list[int]) -> np.ndarray:
    h, w = g.shape
    boundary_c = full_cols[-1]
    col_bands = _bands(boundary_c + 1, full_cols)
    assert len(col_bands) in (3, 4), "expected a 3x3 or 4x4 lattice"
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(col_widths) == 1, "column bands must be uniform"

    row_separators = [r for r in range(h) if np.all(g[r, : boundary_c + 1] == SEP)]
    row_bands = _bands(h, row_separators)
    assert len(row_bands) == len(col_bands), "lattice must be square"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    assert len(row_heights) == 1, "row bands must be uniform"

    mapping = _reference_mapping(g[:, boundary_c + 1 :], horizontal=False)
    out = g.copy()

    last = len(row_bands) - 1
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            fr0, fc0, fr1, fc1, area = _cell_geometry(g[r0 : r1 + 1, c0 : c1 + 1])
            color = mapping[area]
            ar0 = r0 + fr0
            ac0 = c0 + fc0
            ar1 = r0 + fr1
            ac1 = c0 + fc1

            if ri in (0, last) or ci in (0, last):
                out[ar0, ac0 : ac1 + 1] = color
                out[ar1, ac0 : ac1 + 1] = color
                out[ar0 : ar1 + 1, ac0] = color
                out[ar0 : ar1 + 1, ac1] = color
            else:
                out[ar0 + 1 : ar1, ac0 + 1 : ac1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow frames by reading a hole-area code bar and by checking lattice-edge position.

    Input structure:
    - Gray separator lines divide one side of the grid into a square 3x3 or 4x4 lattice of rectangular cells, with one extra outside strip on the bottom or right.
    - Every cell contains exactly one blue hollow rectangle on black background, and the number of black hole pixels inside that frame is between 1 and 4.
    - The outside strip contains a one-pixel-thick reference bar made of four differently colored runs whose lengths are 1, 2, 3, and 4 in some order.

    Rule:
    - Read the reference bar so the run of length L supplies the target color for hole area L.
    - Split the lattice into cells with the gray separators and measure each cell's hole area by counting the black pixels inside its blue frame.
    - If the cell lies on the outer row or outer column of the lattice, recolor the frame border to the mapped color and keep the hole black.
    - If the cell is strictly interior, keep the blue frame border and fill the hole with the mapped color.
    - Preserve the gray separators, the outside strip, and all untouched background pixels.

    Key insight:
    - The colored strip is a visible lookup table keyed by segment length, and the same hole area causes different output edits depending on whether its cell is on the lattice perimeter or in the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(np.max(g)) <= 9 and int(np.min(g)) >= 0

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]

    if full_rows and not full_cols:
        return _solve_bottom(g, full_rows)
    if full_cols and not full_rows:
        return _solve_right(g, full_cols)
    raise AssertionError("expected either a bottom strip or a right strip")


def _sample_area_grid(n: int) -> np.ndarray:
    areas = np.zeros((n, n), dtype=int)
    edge_positions = [(r, c) for r in range(n) for c in range(n) if r in (0, n - 1) or c in (0, n - 1)]
    random.shuffle(edge_positions)

    if n == 4:
        interior_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        interior_values = [1, 2, 3, 4]
        random.shuffle(interior_values)
        for pos, area in zip(interior_positions, interior_values):
            areas[pos] = area

        edge_values = [1, 2, 3, 4] + [random.randint(1, 4) for _ in range(len(edge_positions) - 4)]
        random.shuffle(edge_values)
        for pos, area in zip(edge_positions, edge_values):
            areas[pos] = area
    else:
        center_area = random.randint(1, 4)
        areas[1, 1] = center_area
        edge_values = [1, 2, 3, 4, center_area] + [random.randint(1, 4) for _ in range(len(edge_positions) - 5)]
        random.shuffle(edge_values)
        for pos, area in zip(edge_positions, edge_values):
            areas[pos] = area

    assert set(map(int, np.unique(areas))) == {1, 2, 3, 4}
    return areas


def _fit_shapes(area: int, cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    return [
        (hh, ww)
        for hh, ww in AREA_SHAPES[area]
        if hh + 2 <= cell_h and ww + 2 <= cell_w
    ]


def _assign_shapes(areas: np.ndarray, cell_h: int, cell_w: int) -> dict[tuple[int, int], tuple[int, int]]:
    shape_map: dict[tuple[int, int], tuple[int, int]] = {}
    for area in range(1, 5):
        positions = [(r, c) for r in range(areas.shape[0]) for c in range(areas.shape[1]) if int(areas[r, c]) == area]
        options = _fit_shapes(area, cell_h, cell_w)
        assert options, "every sampled area must fit inside the cell"

        random.shuffle(positions)
        assigned: list[tuple[int, int]] = []
        while len(assigned) < len(positions):
            round_options = options[:]
            random.shuffle(round_options)
            assigned.extend(round_options)
        for pos, shape in zip(positions, assigned[: len(positions)]):
            shape_map[pos] = shape
    return shape_map


def _make_cell(cell_h: int, cell_w: int, hole_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    hole_h, hole_w = hole_shape
    outer_h = hole_h + 2
    outer_w = hole_w + 2
    top = random.randint(0, cell_h - outer_h)
    left = random.randint(0, cell_w - outer_w)
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    cell = np.zeros((cell_h, cell_w), dtype=int)
    cell[top, left : right + 1] = FRAME
    cell[bottom, left : right + 1] = FRAME
    cell[top : bottom + 1, left] = FRAME
    cell[top : bottom + 1, right] = FRAME
    return cell, (top, left, bottom, right)


def _paint_expected(
    grid: np.ndarray,
    frames: dict[tuple[int, int], tuple[int, int, int, int]],
    areas: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    n = areas.shape[0]
    last = n - 1
    for (r, c), (r0, c0, r1, c1) in frames.items():
        color = mapping[int(areas[r, c])]
        if r in (0, last) or c in (0, last):
            out[r0, c0 : c1 + 1] = color
            out[r1, c0 : c1 + 1] = color
            out[r0 : r1 + 1, c0] = color
            out[r0 : r1 + 1, c1] = color
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _semantic_checks(
    n: int,
    areas: np.ndarray,
    shape_map: dict[tuple[int, int], tuple[int, int]],
    mapping: dict[int, int],
    ref_order: tuple[int, int, int, int],
    cell_h: int,
    cell_w: int,
) -> bool:
    if set(mapping) != {1, 2, 3, 4}:
        return False
    if len(set(mapping.values())) != 4:
        return False
    if ref_order in ((1, 2, 3, 4), (4, 3, 2, 1)):
        return False

    edge_areas = {
        int(areas[r, c])
        for r in range(n)
        for c in range(n)
        if r in (0, n - 1) or c in (0, n - 1)
    }
    if edge_areas != {1, 2, 3, 4}:
        return False

    interior_areas = {
        int(areas[r, c])
        for r in range(1, n - 1)
        for c in range(1, n - 1)
    }
    if n == 4 and interior_areas != {1, 2, 3, 4}:
        return False
    if n == 3 and len(interior_areas) != 1:
        return False

    for area in range(2, 5):
        positions = [(r, c) for r in range(n) for c in range(n) if int(areas[r, c]) == area]
        used_shapes = {
            shape
            for (r, c), shape in shape_map.items()
            if int(areas[r, c]) == area
        }
        possible_shapes = set(_fit_shapes(area, cell_h, cell_w))
        if len(positions) >= 2 and len(possible_shapes) >= 2 and len(used_shapes) < 2:
            return False

    return True


def _build_bottom_scene(n: int, cell_h: int, cell_w: int, areas: np.ndarray, shape_map: dict[tuple[int, int], tuple[int, int]], mapping: dict[int, int], ref_order: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    width = n * cell_w + (n - 1)
    boundary_r = n * cell_h + (n - 1)
    height = boundary_r + 2
    grid = np.zeros((height, width), dtype=int)

    for k in range(n - 1):
        sr = cell_h + k * (cell_h + 1)
        grid[sr, :] = SEP
    grid[boundary_r, :] = SEP
    for k in range(n - 1):
        sc = cell_w + k * (cell_w + 1)
        grid[: boundary_r + 1, sc] = SEP

    frames: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(n):
        for c in range(n):
            cell, (fr0, fc0, fr1, fc1) = _make_cell(cell_h, cell_w, shape_map[(r, c)])
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
            frames[(r, c)] = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)

    start = random.randint(0, width - 10)
    pos = start
    for area in ref_order:
        grid[boundary_r + 1, pos : pos + area] = mapping[area]
        pos += area

    expected = _paint_expected(grid, frames, areas, mapping)
    return grid, expected


def _build_right_scene(n: int, cell_h: int, cell_w: int, areas: np.ndarray, shape_map: dict[tuple[int, int], tuple[int, int]], mapping: dict[int, int], ref_order: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    height = n * cell_h + (n - 1)
    boundary_c = n * cell_w + (n - 1)
    width = boundary_c + 2
    grid = np.zeros((height, width), dtype=int)

    for k in range(n - 1):
        sr = cell_h + k * (cell_h + 1)
        grid[sr, : boundary_c + 1] = SEP
    for k in range(n - 1):
        sc = cell_w + k * (cell_w + 1)
        grid[:, sc] = SEP
    grid[:, boundary_c] = SEP

    frames: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(n):
        for c in range(n):
            cell, (fr0, fc0, fr1, fc1) = _make_cell(cell_h, cell_w, shape_map[(r, c)])
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
            frames[(r, c)] = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)

    start = random.randint(0, height - 10)
    pos = start
    for area in ref_order:
        grid[pos : pos + area, boundary_c + 1] = mapping[area]
        pos += area

    expected = _paint_expected(grid, frames, areas, mapping)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((4, 4, 4, 3))
    cell_h, cell_w = random.choice(((5, 5), (5, 6), (6, 5), (6, 6)))
    areas = _sample_area_grid(n)
    shape_map = _assign_shapes(areas, cell_h, cell_w)
    ref_order = random.choice(NON_MONOTONIC_ORDERS)
    colors = random.sample(list(REF_COLORS), 4)
    mapping = {area: color for area, color in zip((1, 2, 3, 4), colors)}

    if not _semantic_checks(n, areas, shape_map, mapping, ref_order, cell_h, cell_w):
        return None

    if random.choice((False, True)):
        grid, expected = _build_bottom_scene(n, cell_h, cell_w, areas, shape_map, mapping, ref_order)
    else:
        grid, expected = _build_right_scene(n, cell_h, cell_w, areas, shape_map, mapping, ref_order)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
