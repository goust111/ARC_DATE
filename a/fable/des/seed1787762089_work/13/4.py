from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
PITCH = 4
CENTER_OFFSET = PITCH // 2
MAX_ATTEMPTS = 300


def _find_lattice_lines(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_counts = np.sum(grid == GREY, axis=1)
    col_counts = np.sum(grid == GREY, axis=0)
    assert int(row_counts.max()) > 0 and int(col_counts.max()) > 0

    line_rows = np.flatnonzero(row_counts > int(row_counts.max()) // 2)
    line_cols = np.flatnonzero(col_counts > int(col_counts.max()) // 2)
    assert line_rows.size >= 4 and line_cols.size >= 4
    assert np.unique(np.diff(line_rows)).size == 1
    assert np.unique(np.diff(line_cols)).size == 1
    return line_rows.astype(int), line_cols.astype(int)


def _extract_legend_mapping(left_band: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in left_band:
        colored = np.flatnonzero((row != BLACK) & (row != GREY))
        if colored.size == 0:
            continue
        assert colored.size == 2 and int(colored[1]) == int(colored[0]) + 1
        src = int(row[int(colored[0])])
        dst = int(row[int(colored[1])])
        assert src not in mapping
        mapping[src] = dst
    assert 2 <= len(mapping) <= 4
    return mapping


def _recolor_crop(crop: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = crop.copy()
    marker_mask = (crop != BLACK) & (crop != GREY)
    out[marker_mask] = BLACK
    for src, dst in mapping.items():
        out[crop == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle marked by four missing lattice intersections and recolor its markers via the left legend.

    Input structure
    - A grey lattice sits on the right; cell interiors are black and some cell centers hold single-pixel markers.
    - Exactly four lattice intersections are missing (black), forming the corners of the crop rectangle.
    - A left-side legend strip contains 2-4 horizontal color pairs encoding source -> replacement.

    Rule
    - Find the four missing intersections and crop the inclusive rectangle they define.
    - Inside that crop, replace each marker color using the legend; delete unlisted marker colors to black.
    - Output only the cropped lattice region, keeping the lattice lines grey.
    """
    g = np.asarray(grid, dtype=int)
    line_rows, line_cols = _find_lattice_lines(g)

    lattice_c0 = int(line_cols[0])
    mapping = _extract_legend_mapping(g[:, :lattice_c0])

    missing = [
        (int(r), int(c))
        for r in line_rows
        for c in line_cols
        if int(g[int(r), int(c)]) == BLACK
    ]
    assert len(missing) == 4

    corner_rows = sorted({r for r, _ in missing})
    corner_cols = sorted({c for _, c in missing})
    assert len(corner_rows) == 2 and len(corner_cols) == 2
    assert set(missing) == {
        (corner_rows[0], corner_cols[0]),
        (corner_rows[0], corner_cols[1]),
        (corner_rows[1], corner_cols[0]),
        (corner_rows[1], corner_cols[1]),
    }

    r0, r1 = corner_rows
    c0, c1 = corner_cols
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    out = _recolor_crop(crop, mapping)

    rel_rows = line_rows[(line_rows >= r0) & (line_rows <= r1)] - r0
    rel_cols = line_cols[(line_cols >= c0) & (line_cols <= c1)] - c0
    out[rel_rows, :] = GREY
    out[:, rel_cols] = GREY
    return out


def _sample_mapping() -> dict[int, int]:
    count = random.randint(2, 4)
    sources = random.sample(list(MARKER_COLORS), count)
    mapping: dict[int, int] = {}
    for src in sources:
        choices = [color for color in MARKER_COLORS if color != src]
        mapping[src] = random.choice(choices)
    return mapping


def _sample_crop_bounds(cell_rows: int, cell_cols: int) -> tuple[int, int, int, int]:
    while True:
        top = random.randint(0, cell_rows - 2)
        bottom = random.randint(top + 2, cell_rows)
        left = random.randint(0, cell_cols - 2)
        right = random.randint(left + 2, cell_cols)
        if top > 0 or left > 0 or bottom < cell_rows or right < cell_cols:
            return top, bottom, left, right


def _cell_lists(
    cell_rows: int,
    cell_cols: int,
    bounds: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top, bottom, left, right = bounds
    inside: list[tuple[int, int]] = []
    outside: list[tuple[int, int]] = []
    for r in range(cell_rows):
        for c in range(cell_cols):
            cell = (r, c)
            if top <= r < bottom and left <= c < right:
                inside.append(cell)
            else:
                outside.append(cell)
    return inside, outside


def _choose_color(mapped: list[int], unmapped: list[int], bias: float) -> int:
    if random.random() < bias:
        return random.choice(mapped)
    return random.choice(unmapped)


def _sample_markers(
    cell_rows: int,
    cell_cols: int,
    bounds: tuple[int, int, int, int],
    mapping: dict[int, int],
) -> dict[tuple[int, int], int]:
    inside, outside = _cell_lists(cell_rows, cell_cols, bounds)
    mapped = list(mapping)
    unmapped = [color for color in MARKER_COLORS if color not in mapping]
    assert len(inside) >= 4 and outside

    markers: dict[tuple[int, int], int] = {}
    must_mapped, must_unmapped = random.sample(inside, 2)
    markers[must_mapped] = random.choice(mapped)
    markers[must_unmapped] = random.choice(unmapped)

    remaining_inside = [cell for cell in inside if cell not in markers]
    remaining_outside = list(outside)

    extra_inside = random.randint(1, min(len(remaining_inside), max(1, len(inside) // 2)))
    extra_outside = random.randint(1, min(len(remaining_outside), max(1, len(outside) // 2)))

    for cell in random.sample(remaining_inside, extra_inside):
        markers[cell] = _choose_color(mapped, unmapped, 0.5)
    for cell in random.sample(remaining_outside, extra_outside):
        markers[cell] = _choose_color(mapped, unmapped, 0.5)
    return markers


def _build_clean_lattice(
    cell_rows: int,
    cell_cols: int,
    markers: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = cell_rows * PITCH + 1
    width = cell_cols * PITCH + 1
    lattice = np.full((height, width), BLACK, dtype=int)
    line_rows = np.arange(0, height, PITCH, dtype=int)
    line_cols = np.arange(0, width, PITCH, dtype=int)

    lattice[line_rows, :] = GREY
    lattice[:, line_cols] = GREY
    for (cell_r, cell_c), color in markers.items():
        rr = int(cell_r * PITCH + CENTER_OFFSET)
        cc = int(cell_c * PITCH + CENTER_OFFSET)
        lattice[rr, cc] = int(color)
    return lattice, line_rows, line_cols


def _legend_rows(height: int, count: int) -> list[int]:
    gaps = [random.randint(1, 3) for _ in range(count - 1)]
    block_height = count + sum(gaps)
    start = random.randint(0, height - block_height)
    rows = [start]
    current = start
    for gap in gaps:
        current += 1 + gap
        rows.append(current)
    return rows


def _expected_output(
    clean_lattice: np.ndarray,
    line_rows: np.ndarray,
    line_cols: np.ndarray,
    bounds: tuple[int, int, int, int],
    mapping: dict[int, int],
) -> np.ndarray:
    top, bottom, left, right = bounds
    r0 = int(line_rows[top])
    r1 = int(line_rows[bottom])
    c0 = int(line_cols[left])
    c1 = int(line_cols[right])
    crop = clean_lattice[r0 : r1 + 1, c0 : c1 + 1]
    return _recolor_crop(crop, mapping)


def _compose_input(
    clean_lattice: np.ndarray,
    line_rows: np.ndarray,
    line_cols: np.ndarray,
    bounds: tuple[int, int, int, int],
    mapping: dict[int, int],
) -> np.ndarray:
    lattice = clean_lattice.copy()
    top, bottom, left, right = bounds
    for rr in (int(line_rows[top]), int(line_rows[bottom])):
        for cc in (int(line_cols[left]), int(line_cols[right])):
            lattice[rr, cc] = BLACK

    lattice_h, lattice_w = lattice.shape
    legend_count = len(mapping)
    min_height = max(lattice_h + 1, legend_count * 2 + 2)
    height = random.randint(min_height, min_height + 6)

    min_width = lattice_w + 5
    width = random.randint(min_width, min_width + 6)

    max_lattice_c0 = width - lattice_w
    lattice_c0 = random.randint(4, max_lattice_c0)
    lattice_r0 = random.randint(0, height - lattice_h)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[lattice_r0 : lattice_r0 + lattice_h, lattice_c0 : lattice_c0 + lattice_w] = lattice

    legend_c0 = random.randint(0, lattice_c0 - 3)
    for row, (src, dst) in zip(_legend_rows(height, legend_count), mapping.items()):
        grid[row, legend_c0] = src
        grid[row, legend_c0 + 1] = dst

    return grid


def _gen_attempt() -> np.ndarray | None:
    cell_rows = random.randint(3, 5)
    cell_cols = random.randint(3, 5)
    bounds = _sample_crop_bounds(cell_rows, cell_cols)
    mapping = _sample_mapping()
    markers = _sample_markers(cell_rows, cell_cols, bounds, mapping)

    clean_lattice, line_rows, line_cols = _build_clean_lattice(cell_rows, cell_cols, markers)
    expected = _expected_output(clean_lattice, line_rows, line_cols, bounds, mapping)
    grid = _compose_input(clean_lattice, line_rows, line_cols, bounds, mapping)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
