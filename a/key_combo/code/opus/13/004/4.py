from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6
TEAL = 8

CELL_SIZE = 3
CELL_COUNTS = (4, 5)
GEN_RETRY_LIMIT = 500


def _cell_ranges(size: int, walls: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for wall in walls:
        if start <= wall - 1:
            bands.append((start, wall - 1))
        start = wall + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= pos <= hi:
            return idx
    raise AssertionError("position does not fall inside any lattice cell")


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _singleton_color(colors: set[int], label: str) -> int:
    if len(colors) != 1:
        raise AssertionError(f"{label} expected exactly one fill color")
    return next(iter(colors))


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    wall_rows = [r for r in range(h) if np.all(g[r, :] == TEAL)]
    wall_cols = [c for c in range(w) if np.all(g[:, c] == TEAL)]
    assert wall_rows and wall_cols, "expected a teal lattice"

    row_bands = _cell_ranges(h, wall_rows)
    col_bands = _cell_ranges(w, wall_cols)
    assert len(row_bands) >= 4 and len(col_bands) >= 4, "expected a coarse grid with at least 4 cells per axis"
    assert len({hi - lo + 1 for lo, hi in row_bands}) == 1, "row cell heights must be uniform"
    assert len({hi - lo + 1 for lo, hi in col_bands}) == 1, "column cell widths must be uniform"
    assert row_bands[0][1] - row_bands[0][0] + 1 >= 3, "cell interiors must be at least 3x3"

    anchors = []
    for comp in _components_of_color(g, GREEN):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        if len(comp) == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            anchors.append(comp)
    assert len(anchors) == 1, "expected exactly one 2x2 green anchor"
    anchor_pixels = anchors[0]
    ar0, ac0 = anchor_pixels.min(axis=0)
    anchor_row = _band_index(int(ar0), row_bands)
    anchor_col = _band_index(int(ac0), col_bands)
    assert _band_index(int(anchor_pixels[:, 0].max()), row_bands) == anchor_row
    assert _band_index(int(anchor_pixels[:, 1].max()), col_bands) == anchor_col

    markers = []
    occupied_cells: set[tuple[int, int]] = {(anchor_row, anchor_col)}
    for color in (BLUE, RED):
        for r_raw, c_raw in np.argwhere(g == color):
            r = int(r_raw)
            c = int(c_raw)
            row_idx = _band_index(r, row_bands)
            col_idx = _band_index(c, col_bands)
            r0, r1 = row_bands[row_idx]
            c0, c1 = col_bands[col_idx]
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            assert (r, c) == center, "markers must sit at cell centers"
            cell = (row_idx, col_idx)
            assert cell not in occupied_cells, "each lattice cell may hold at most one object"
            occupied_cells.add(cell)
            markers.append({"pixel": (r, c), "cell": cell, "color": color})

    assert markers, "expected at least one marker"
    return g, row_bands, col_bands, (anchor_row, anchor_col), anchor_pixels, markers


def _coarse_coverages(
    markers: list[dict[str, object]], anchor_cell: tuple[int, int]
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    anchor_row, anchor_col = anchor_cell
    ray_cov: dict[tuple[int, int], set[int]] = defaultdict(set)
    interval_cov: dict[tuple[int, int], set[int]] = defaultdict(set)

    aligned: list[dict[str, object]] = []
    unaligned_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for marker in markers:
        row_idx, col_idx = marker["cell"]
        color = int(marker["color"])
        if row_idx == anchor_row or col_idx == anchor_col:
            aligned.append(marker)
        else:
            unaligned_by_color[color].append((row_idx, col_idx))

    assert len(aligned) >= 2, "expected at least two anchor-aligned markers"

    for marker in aligned:
        row_idx, col_idx = marker["cell"]
        color = int(marker["color"])
        if row_idx == anchor_row:
            step = 1 if col_idx < anchor_col else -1
            for cc in range(col_idx, anchor_col, step):
                ray_cov[(row_idx, cc)].add(color)
        else:
            step = 1 if row_idx < anchor_row else -1
            for rr in range(row_idx, anchor_row, step):
                ray_cov[(rr, col_idx)].add(color)

    pair_count = 0
    used_unaligned: set[tuple[int, int]] = set()
    for color, cells in unaligned_by_color.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for row_idx, col_idx in cells:
            by_row[row_idx].append(col_idx)
            by_col[col_idx].append(row_idx)

        for row_idx, cols in by_row.items():
            if len(cols) >= 2:
                pair_count += 1
                for cc in range(min(cols), max(cols) + 1):
                    interval_cov[(row_idx, cc)].add(color)
                for cc in cols:
                    used_unaligned.add((row_idx, cc))

        for col_idx, rows in by_col.items():
            if len(rows) >= 2:
                pair_count += 1
                for rr in range(min(rows), max(rows) + 1):
                    interval_cov[(rr, col_idx)].add(color)
                for rr in rows:
                    used_unaligned.add((rr, col_idx))

    assert pair_count >= 2, "expected at least two non-anchor interval pairs"
    for cells in unaligned_by_color.values():
        assert set(cells) <= used_unaligned, "every non-anchor-aligned marker must participate in an interval run"

    return ray_cov, interval_cov


def _paint_output(
    base: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    anchor_pixels: np.ndarray,
    markers: list[dict[str, object]],
    ray_cov: dict[tuple[int, int], set[int]],
    interval_cov: dict[tuple[int, int], set[int]],
) -> np.ndarray:
    out = np.array(base, copy=True)
    for cell in sorted(set(ray_cov) | set(interval_cov)):
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        has_ray = cell in ray_cov
        has_interval = cell in interval_cov
        if has_ray and has_interval:
            fill_color = PINK
        elif has_ray:
            fill_color = _singleton_color(ray_cov[cell], "ray fill")
        else:
            fill_color = _singleton_color(interval_cov[cell], "interval fill")
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    for r, c in anchor_pixels:
        out[int(r), int(c)] = GREEN
    for marker in markers:
        r, c = marker["pixel"]
        out[int(r), int(c)] = int(marker["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill lattice cells by splitting markers into anchor-aligned rays and non-anchor interval runs, with pink at mode overlaps.

    Input structure
    - A teal lattice divides the canvas into uniform square cells with black interiors.
    - One cell contains a 2x2 green anchor square, and other occupied cells contain one centered blue or red marker pixel each.
    - Some markers share the anchor's coarse row or coarse column; the rest avoid both anchor axes but align with same-color partners by coarse row or coarse column.

    Rule
    - Keep the teal lattice, the green anchor, and every original marker pixel unchanged.
    - For each marker whose cell is on the anchor's coarse row or coarse column, fill every cell from that marker to the cell adjacent to the anchor with the marker color, including the marker cell and excluding the anchor cell.
    - For the remaining markers, fill inclusive same-color row spans and same-color column spans between their leftmost/rightmost or topmost/bottommost marker cells.
    - If a cell is covered by both the anchor-aligned ray mode and the non-anchor interval mode, color that whole cell pink.

    Key insight
    - The green anchor is both a geometric endpoint and the partition key: anchor alignment decides which markers use rays and which markers use interval filling, so pink marks where those two behaviors intersect.
    """
    g, row_bands, col_bands, anchor_cell, anchor_pixels, markers = _parse_input(grid)
    ray_cov, interval_cov = _coarse_coverages(markers, anchor_cell)
    return _paint_output(g, row_bands, col_bands, anchor_pixels, markers, ray_cov, interval_cov)


def _grid_size(n_cells: int) -> int:
    return n_cells * CELL_SIZE + (n_cells - 1)


def _build_lattice(row_cells: int, col_cells: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = _grid_size(row_cells)
    w = _grid_size(col_cells)
    grid = np.zeros((h, w), dtype=int)
    wall_rows = [CELL_SIZE + i * (CELL_SIZE + 1) for i in range(row_cells - 1)]
    wall_cols = [CELL_SIZE + i * (CELL_SIZE + 1) for i in range(col_cells - 1)]
    grid[wall_rows, :] = TEAL
    grid[:, wall_cols] = TEAL
    return grid, _cell_ranges(h, wall_rows), _cell_ranges(w, wall_cols)


def _cell_center(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], cell: tuple[int, int]) -> tuple[int, int]:
    r0, r1 = row_bands[cell[0]]
    c0, c1 = col_bands[cell[1]]
    return (r0 + r1) // 2, (c0 + c1) // 2


def _place_anchor(grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], cell: tuple[int, int]) -> None:
    r0, _ = row_bands[cell[0]]
    c0, _ = col_bands[cell[1]]
    grid[r0 : r0 + 2, c0 : c0 + 2] = GREEN


def _place_markers(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    markers: list[tuple[tuple[int, int], int]],
) -> None:
    for cell, color in markers:
        r, c = _cell_center(row_bands, col_bands, cell)
        grid[r, c] = color


def _render_input(
    row_cells: int,
    col_cells: int,
    anchor_cell: tuple[int, int],
    markers: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    grid, row_bands, col_bands = _build_lattice(row_cells, col_cells)
    _place_anchor(grid, row_bands, col_bands, anchor_cell)
    _place_markers(grid, row_bands, col_bands, markers)
    return grid


def _rows_same_side(anchor_row: int, row_cells: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    above = list(range(anchor_row))
    below = list(range(anchor_row + 1, row_cells))
    for group in (above, below):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i], group[j]))
    return pairs


def _cols_same_side(anchor_col: int, col_cells: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    left = list(range(anchor_col))
    right = list(range(anchor_col + 1, col_cells))
    for group in (left, right):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i], group[j]))
    return pairs


def _valid_anchor_cells(row_cells: int, col_cells: int, mode: str) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for ar in range(1, row_cells - 1):
        for ac in range(1, col_cells - 1):
            up = ar
            down = row_cells - 1 - ar
            left = ac
            right = col_cells - 1 - ac
            if mode == "row_overlap":
                if max(left, right) >= 2 and max(up, down) >= 2:
                    valid.append((ar, ac))
            else:
                if max(up, down) >= 2 and max(left, right) >= 2:
                    valid.append((ar, ac))
    return valid


def _attempt_description() -> tuple[int, int, tuple[int, int], list[tuple[tuple[int, int], int]]] | None:
    row_cells = random.choice(CELL_COUNTS)
    col_cells = random.choice(CELL_COUNTS)
    mode = random.choice(("row_overlap", "col_overlap"))
    anchors = _valid_anchor_cells(row_cells, col_cells, mode)
    if not anchors:
        return None
    anchor_cell = random.choice(anchors)
    ar, ac = anchor_cell

    occupied = {anchor_cell}
    markers: list[tuple[tuple[int, int], int]] = []

    def add_marker(cell: tuple[int, int], color: int) -> bool:
        if cell in occupied:
            return False
        occupied.add(cell)
        markers.append((cell, color))
        return True

    overlap_interval_color = random.choice((BLUE, RED))
    other_interval_color = BLUE if overlap_interval_color == RED else RED
    row_ray_color = random.choice((BLUE, RED))
    col_ray_color = random.choice((BLUE, RED))

    if mode == "row_overlap":
        horizontal_options = []
        if ac >= 2:
            horizontal_options.append(("left", ac))
        if col_cells - 1 - ac >= 2:
            horizontal_options.append(("right", col_cells - 1 - ac))
        if not horizontal_options:
            return None
        side, max_dist = random.choice(horizontal_options)
        ray_dist = random.randint(2, max_dist)
        row_marker = (ar, ac - ray_dist) if side == "left" else (ar, ac + ray_dist)
        if not add_marker(row_marker, row_ray_color):
            return None

        if side == "left":
            overlap_col = random.choice(list(range(row_marker[1] + 1, ac)))
        else:
            overlap_col = random.choice(list(range(ac + 1, row_marker[1])))

        vertical_options = []
        if ar >= 1:
            vertical_options.append(("up", ar))
        if row_cells - 1 - ar >= 1:
            vertical_options.append(("down", row_cells - 1 - ar))
        side2, max_dist2 = random.choice(vertical_options)
        ray_dist2 = random.randint(1, max_dist2)
        col_marker = (ar - ray_dist2, ac) if side2 == "up" else (ar + ray_dist2, ac)
        if not add_marker(col_marker, col_ray_color):
            return None

        top = random.choice(list(range(ar)))
        bottom = random.choice(list(range(ar + 1, row_cells)))
        if not add_marker((top, overlap_col), overlap_interval_color):
            return None
        if not add_marker((bottom, overlap_col), overlap_interval_color):
            return None

        second_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for col in range(col_cells):
            if col in (ac, overlap_col):
                continue
            for r0, r1 in _rows_same_side(ar, row_cells):
                if (r0, col) in occupied or (r1, col) in occupied:
                    continue
                second_candidates.append(((r0, col), (r1, col)))
        if not second_candidates:
            return None
        pair_a, pair_b = random.choice(second_candidates)
        if not add_marker(pair_a, other_interval_color):
            return None
        if not add_marker(pair_b, other_interval_color):
            return None
    else:
        vertical_options = []
        if ar >= 2:
            vertical_options.append(("up", ar))
        if row_cells - 1 - ar >= 2:
            vertical_options.append(("down", row_cells - 1 - ar))
        if not vertical_options:
            return None
        side, max_dist = random.choice(vertical_options)
        ray_dist = random.randint(2, max_dist)
        col_marker = (ar - ray_dist, ac) if side == "up" else (ar + ray_dist, ac)
        if not add_marker(col_marker, col_ray_color):
            return None

        if side == "up":
            overlap_row = random.choice(list(range(col_marker[0] + 1, ar)))
        else:
            overlap_row = random.choice(list(range(ar + 1, col_marker[0])))

        horizontal_options = []
        if ac >= 1:
            horizontal_options.append(("left", ac))
        if col_cells - 1 - ac >= 1:
            horizontal_options.append(("right", col_cells - 1 - ac))
        side2, max_dist2 = random.choice(horizontal_options)
        ray_dist2 = random.randint(1, max_dist2)
        row_marker = (ar, ac - ray_dist2) if side2 == "left" else (ar, ac + ray_dist2)
        if not add_marker(row_marker, row_ray_color):
            return None

        left = random.choice(list(range(ac)))
        right = random.choice(list(range(ac + 1, col_cells)))
        if not add_marker((overlap_row, left), overlap_interval_color):
            return None
        if not add_marker((overlap_row, right), overlap_interval_color):
            return None

        second_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for row in range(row_cells):
            if row in (ar, overlap_row):
                continue
            for c0, c1 in _cols_same_side(ac, col_cells):
                if (row, c0) in occupied or (row, c1) in occupied:
                    continue
                second_candidates.append(((row, c0), (row, c1)))
        if not second_candidates:
            return None
        pair_a, pair_b = random.choice(second_candidates)
        if not add_marker(pair_a, other_interval_color):
            return None
        if not add_marker(pair_b, other_interval_color):
            return None

    return row_cells, col_cells, anchor_cell, markers


def _count_magenta_cells(
    out: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> int:
    total = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = out[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == PINK):
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    desc = _attempt_description()
    if desc is None:
        return None
    row_cells, col_cells, anchor_cell, coarse_markers = desc
    inp = _render_input(row_cells, col_cells, anchor_cell, coarse_markers)

    try:
        parsed = _parse_input(inp)
        out = main(inp)
    except AssertionError:
        return None

    _, row_bands, col_bands, anchor_cell2, _anchor_pixels, markers = parsed
    if anchor_cell2 != anchor_cell:
        return None

    aligned_count = sum(1 for marker in markers if marker["cell"][0] == anchor_cell[0] or marker["cell"][1] == anchor_cell[1])
    if aligned_count < 2:
        return None

    ray_cov, interval_cov = _coarse_coverages(markers, anchor_cell)
    if not set(ray_cov) & set(interval_cov):
        return None
    if _count_magenta_cells(out, row_bands, col_bands) < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/13/004/4")
