from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _detect_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_counts = np.sum(grid == GREY, axis=1)
    col_counts = np.sum(grid == GREY, axis=0)

    row_max = int(row_counts.max())
    col_max = int(col_counts.max())
    rows = [int(r) for r in np.flatnonzero(row_counts >= row_max - 2)]
    cols = [int(c) for c in np.flatnonzero(col_counts >= col_max - 2)]

    assert len(rows) >= 4 and len(cols) >= 4
    row_gaps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_gaps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    assert row_gaps.issubset({3, 4}) and col_gaps.issubset({3, 4})
    return rows, cols


def _find_crop_bbox(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> tuple[int, int, int, int]:
    missing = [(r, c) for r in row_lines for c in col_lines if int(grid[r, c]) == BG]
    assert len(missing) == 4
    rows = sorted({r for r, _c in missing})
    cols = sorted({c for _r, c in missing})
    assert len(rows) == 2 and len(cols) == 2
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    assert set(missing) == expected
    return rows[0], cols[0], rows[1], cols[1]


def _extract_nodes(region: np.ndarray, row_lines: list[int], col_lines: list[int]) -> dict[tuple[int, int], tuple[int, int, int]]:
    nodes: dict[tuple[int, int], tuple[int, int, int]] = {}
    for ri in range(len(row_lines) - 1):
        for ci in range(len(col_lines) - 1):
            r0, r1 = row_lines[ri], row_lines[ri + 1]
            c0, c1 = col_lines[ci], col_lines[ci + 1]
            cell = region[r0 + 1 : r1, c0 + 1 : c1]
            coords = np.argwhere((cell != BG) & (cell != GREY))
            assert coords.shape[0] <= 1
            if coords.size == 0:
                continue
            dr, dc = map(int, coords[0])
            rr = int(r0 + 1 + dr)
            cc = int(c0 + 1 + dc)
            nodes[(ri, ci)] = (rr, cc, int(region[rr, cc]))
    return nodes


def _analyze_pairs(
    nodes: dict[tuple[int, int], tuple[int, int, int]],
    cell_rows: int,
    cell_cols: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    horiz_cells: set[tuple[int, int]] = set()
    horiz_pairs: list[tuple[int, int, int]] = []
    vert_allowed: list[tuple[int, int, int]] = []
    vert_blocked: list[tuple[int, int, int]] = []

    for ri in range(cell_rows):
        for ci in range(cell_cols - 1):
            a = nodes.get((ri, ci))
            b = nodes.get((ri, ci + 1))
            if a is None or b is None or a[2] != b[2]:
                continue
            horiz_cells.add((ri, ci))
            horiz_cells.add((ri, ci + 1))
            horiz_pairs.append((ri, ci, a[2]))

    for ri in range(cell_rows - 1):
        for ci in range(cell_cols):
            a = nodes.get((ri, ci))
            b = nodes.get((ri + 1, ci))
            if a is None or b is None or a[2] != b[2]:
                continue
            triple = (ri, ci, a[2])
            if (ri, ci) in horiz_cells or (ri + 1, ci) in horiz_cells:
                vert_blocked.append(triple)
            else:
                vert_allowed.append(triple)

    return horiz_cells, horiz_pairs, vert_allowed, vert_blocked


def _connect_region(region: np.ndarray, row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    out = region.copy()
    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1
    nodes = _extract_nodes(region, row_lines, col_lines)
    horiz_cells, _horiz_pairs, _vert_allowed, _vert_blocked = _analyze_pairs(nodes, cell_rows, cell_cols)

    for ri in range(cell_rows):
        for ci in range(cell_cols - 1):
            a = nodes.get((ri, ci))
            b = nodes.get((ri, ci + 1))
            if a is None or b is None or a[2] != b[2]:
                continue
            assert a[0] == b[0]
            r = a[0]
            c0 = min(a[1], b[1])
            c1 = max(a[1], b[1])
            out[r, c0 : c1 + 1] = a[2]

    for ri in range(cell_rows - 1):
        for ci in range(cell_cols):
            a = nodes.get((ri, ci))
            b = nodes.get((ri + 1, ci))
            if a is None or b is None or a[2] != b[2]:
                continue
            if (ri, ci) in horiz_cells or (ri + 1, ci) in horiz_cells:
                continue
            assert a[1] == b[1]
            c = a[1]
            r0 = min(a[0], b[0])
            r1 = max(a[0], b[0])
            out[r0 : r1 + 1, c] = a[2]

    return out


def _mirror_left_half(region: np.ndarray) -> np.ndarray:
    out = region.copy()
    center_c = (region.shape[1] - 1) // 2
    colored = np.argwhere((region != BG) & (region != GREY))
    for r, c in colored:
        rr = int(region.shape[0] - 1 - r)
        cc = int(region.shape[1] - 1 - c)
        if int(c) < center_c:
            out[rr, cc] = int(region[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the missing-corner lattice rectangle, connect same-color adjacent dots on its left half, then point-reflect that pattern across the crop center.

    Input structure:
    - A black background contains full grey lattice rows and columns with uniform 3- or 4-pixel spacing.
    - Exactly four lattice intersections are missing, appearing as black pixels at the corners of one axis-aligned rectangle.
    - Cell interiors may contain single colored dots; inside the crop rectangle only the left half contains dots, while the right half is empty.

    Rule:
    - Use the four missing intersections as the crop bbox and discard everything outside it.
    - Inside that crop, treat one dot per cell as a logical node; connect horizontally adjacent same-color nodes first.
    - Connect vertically adjacent same-color nodes only when neither endpoint belongs to any horizontal same-color pair.
    - Finally copy every non-grey colored pixel from the left half to its 180-degree rotated position around the crop's geometric center.

    Key insight:
    The missing lattice corners do more than mark the output crop: they cut adjacency at the crop boundary, so connecting before cropping would create visibly wrong bars.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_lines, col_lines = _detect_lattice_lines(g)
    r0, c0, r1, c1 = _find_crop_bbox(g, row_lines, col_lines)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    local_rows = [r - r0 for r in row_lines if r0 <= r <= r1]
    local_cols = [c - c0 for c in col_lines if c0 <= c <= c1]
    cell_cols = len(local_cols) - 1
    assert cell_cols % 2 == 0

    nodes = _extract_nodes(crop, local_rows, local_cols)
    for (_ri, ci), _node in nodes.items():
        assert ci < cell_cols // 2

    connected = _connect_region(crop, local_rows, local_cols)
    return _mirror_left_half(connected)


def _line_positions(cell_count: int, step: int) -> list[int]:
    return [i * step for i in range(cell_count + 1)]


def _cell_coord(line_positions: list[int], offsets: list[int], idx: int) -> int:
    return int(line_positions[idx] + offsets[idx])


def _paint_lattice(grid: np.ndarray, row_lines: Iterable[int], col_lines: Iterable[int]) -> None:
    for r in row_lines:
        grid[int(r), :] = GREY
    for c in col_lines:
        grid[:, int(c)] = GREY


def _cell_candidates(
    total_rows: int,
    total_cols: int,
    crop_top: int,
    crop_left: int,
    crop_rows: int,
    crop_cols: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for ri in range(total_rows):
        for ci in range(total_cols):
            if crop_top <= ri < crop_top + crop_rows and crop_left <= ci < crop_left + crop_cols:
                continue
            out.append((ri, ci))
    return out


def _adjacent_colors(cells: np.ndarray, ri: int, ci: int) -> set[int]:
    colors: set[int] = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = ri + dr
        cc = ci + dc
        if 0 <= rr < cells.shape[0] and 0 <= cc < cells.shape[1]:
            value = int(cells[rr, cc])
            if value != BG:
                colors.add(value)
    return colors


def _place_single(cells: np.ndarray, ri: int, ci: int, forbidden: set[int]) -> bool:
    if int(cells[ri, ci]) != BG:
        return False
    banned = _adjacent_colors(cells, ri, ci) | set(forbidden)
    choices = [color for color in PALETTE if color not in banned]
    if not choices:
        return False
    cells[ri, ci] = random.choice(choices)
    return True


def _render_cells(
    cells: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_offsets: list[int],
    col_offsets: list[int],
) -> np.ndarray:
    grid = np.zeros((row_lines[-1] + 1, col_lines[-1] + 1), dtype=int)
    _paint_lattice(grid, row_lines, col_lines)
    for ri in range(cells.shape[0]):
        for ci in range(cells.shape[1]):
            color = int(cells[ri, ci])
            if color == BG:
                continue
            r = _cell_coord(row_lines, row_offsets, ri)
            c = _cell_coord(col_lines, col_offsets, ci)
            grid[r, c] = color
    return grid


def _has_boundary_trap(cells: np.ndarray, crop_top: int, crop_left: int, crop_rows: int) -> bool:
    for local_r in range(crop_rows):
        global_r = crop_top + local_r
        inside = int(cells[global_r, crop_left])
        outside = int(cells[global_r, crop_left - 1])
        if inside != BG and inside == outside:
            return True
    return False


def _validate_instance(grid: np.ndarray) -> None:
    row_lines, col_lines = _detect_lattice_lines(grid)
    r0, c0, r1, c1 = _find_crop_bbox(grid, row_lines, col_lines)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    local_rows = [r - r0 for r in row_lines if r0 <= r <= r1]
    local_cols = [c - c0 for c in col_lines if c0 <= c <= c1]
    cell_rows = len(local_rows) - 1
    cell_cols = len(local_cols) - 1
    assert cell_cols % 2 == 0

    nodes = _extract_nodes(crop, local_rows, local_cols)
    colors = np.zeros((cell_rows, cell_cols), dtype=int)
    for (ri, ci), (_r, _c, color) in nodes.items():
        colors[ri, ci] = int(color)
    assert not np.any(colors[:, cell_cols // 2 :])

    horiz_cells, horiz_pairs, vert_allowed, vert_blocked = _analyze_pairs(nodes, cell_rows, cell_cols)
    assert len(horiz_pairs) >= 2
    assert vert_allowed
    assert vert_blocked
    assert horiz_cells

    connected_crop = _connect_region(crop, local_rows, local_cols)
    connected_full = _connect_region(grid, row_lines, col_lines)[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.array_equal(connected_crop, connected_full)

    out = main(grid)
    assert out.shape == crop.shape
    expected = _mirror_left_half(connected_crop)
    assert np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    step_r = random.choice((3, 4))
    step_c = random.choice((3, 4))

    max_row_cells = (30 - 1) // step_r
    max_col_cells = (30 - 1) // step_c
    total_row_cells_choices = [n for n in range(5, min(7, max_row_cells) + 1)]
    total_col_cells_choices = [n for n in range(5, min(7, max_col_cells) + 1)]
    if not total_row_cells_choices or not total_col_cells_choices:
        return None

    total_row_cells = random.choice(total_row_cells_choices)
    total_col_cells = random.choice(total_col_cells_choices)

    crop_rows_choices = [n for n in (4, 5) if n <= total_row_cells and total_row_cells - n >= 1]
    crop_cols_choices = [n for n in (4, 6) if n <= total_col_cells - 1]
    if not crop_rows_choices or not crop_cols_choices:
        return None

    crop_rows = random.choice(crop_rows_choices)
    crop_cols = random.choice(crop_cols_choices)
    left_half_cols = crop_cols // 2

    crop_top = random.randint(0, total_row_cells - crop_rows)
    if crop_top == 0 and crop_top + crop_rows == total_row_cells:
        return None
    crop_left = random.randint(1, total_col_cells - crop_cols)

    row_lines = _line_positions(total_row_cells, step_r)
    col_lines = _line_positions(total_col_cells, step_c)
    row_offsets = [random.randint(1, step_r - 1) for _ in range(total_row_cells)]
    col_offsets = [random.randint(1, step_c - 1) for _ in range(total_col_cells)]

    cells = np.zeros((total_row_cells, total_col_cells), dtype=int)
    blocked_color, vertical_color = random.sample(PALETTE, 2)

    block_c0 = max(0, left_half_cols - 2)
    for dr in (0, 1):
        for dc in (0, 1):
            cells[crop_top + dr, crop_left + block_c0 + dc] = blocked_color

    vr0 = crop_top + crop_rows - 2
    vc = crop_left
    cells[vr0, vc] = vertical_color
    cells[vr0 + 1, vc] = vertical_color
    cells[vr0, crop_left - 1] = vertical_color

    if left_half_cols >= 3 and random.random() < 0.7:
        extra_color = random.choice([color for color in PALETTE if color not in {blocked_color, vertical_color}])
        extra_row = crop_top + random.randint(1, crop_rows - 2)
        cells[extra_row, crop_left + 1] = extra_color
        cells[extra_row, crop_left + 2] = extra_color

    remaining_left = [
        (ri, ci)
        for ri in range(crop_top, crop_top + crop_rows)
        for ci in range(crop_left, crop_left + left_half_cols)
        if int(cells[ri, ci]) == BG
    ]
    random.shuffle(remaining_left)
    for ri, ci in remaining_left:
        if random.random() < 0.45:
            forbidden = set()
            if ci == crop_left:
                forbidden.add(int(cells[ri, ci - 1]))
            _place_single(cells, ri, ci, forbidden)

    outside_cells = _cell_candidates(total_row_cells, total_col_cells, crop_top, crop_left, crop_rows, crop_cols)
    random.shuffle(outside_cells)
    extra_outside = random.randint(3, min(8, len(outside_cells)))
    for ri, ci in outside_cells[:extra_outside]:
        forbidden = set()
        if ci + 1 == crop_left and crop_top <= ri < crop_top + crop_rows:
            forbidden.add(int(cells[ri, crop_left]))
        _place_single(cells, ri, ci, forbidden)

    grid = _render_cells(cells, row_lines, col_lines, row_offsets, col_offsets)

    corner_rows = (row_lines[crop_top], row_lines[crop_top + crop_rows])
    corner_cols = (col_lines[crop_left], col_lines[crop_left + crop_cols])
    for rr in corner_rows:
        for cc in corner_cols:
            grid[rr, cc] = BG

    if not _has_boundary_trap(cells, crop_top, crop_left, crop_rows):
        return None

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")
