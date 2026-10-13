from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

Coord = tuple[int, int]
Vector = tuple[int, int]

MAX_ATTEMPTS = 400


def _add(a: Coord, b: Vector) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _unit_step(start: Coord, end: Coord) -> Vector:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "segments must be axis-aligned or 45-degree"
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    return int(sr), int(sc)


def _segment_cells(start: Coord, end: Coord) -> set[Coord]:
    sr, sc = _unit_step(start, end)
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    return {(start[0] + i * sr, start[1] + i * sc) for i in range(steps + 1)}


def _decompose_displacement(a: Coord, c: Coord) -> tuple[Vector, Vector]:
    dr = c[0] - a[0]
    dc = c[1] - a[1]
    assert dr != 0 and dc != 0 and abs(dr) != abs(dc), "yellow markers must define a mixed diagonal/axis displacement"
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    diag_len = min(abs(dr), abs(dc))
    diag = (sr * diag_len, sc * diag_len)
    if abs(dr) < abs(dc):
        axis = (0, sc * (abs(dc) - diag_len))
    else:
        axis = (sr * (abs(dr) - diag_len), 0)
    return diag, axis


def _parallelogram_geometry(a: Coord, c: Coord) -> tuple[Vector, Vector, Coord, Coord]:
    diag, axis = _decompose_displacement(a, c)
    b = _add(a, diag)
    d = _add(a, axis)
    return diag, axis, b, d


def _classify_point(point: Coord, a: Coord, c: Coord) -> str:
    diag, axis = _decompose_displacement(a, c)
    det = diag[0] * axis[1] - diag[1] * axis[0]
    assert det != 0

    pr = point[0] - a[0]
    pc = point[1] - a[1]
    alpha = pr * axis[1] - pc * axis[0]
    beta = diag[0] * pc - diag[1] * pr
    if det < 0:
        det = -det
        alpha = -alpha
        beta = -beta

    if 0 < alpha < det and 0 < beta < det:
        return "inside"
    if ((alpha == 0 or alpha == det) and 0 <= beta <= det) or ((beta == 0 or beta == det) and 0 <= alpha <= det):
        return "boundary"
    return "outside"


def _scan_parallelogram(shape: tuple[int, int], a: Coord, c: Coord) -> tuple[set[Coord], set[Coord]]:
    diag, axis, b, d = _parallelogram_geometry(a, c)
    corners = [a, b, d, c]
    r0 = min(r for r, _ in corners)
    r1 = max(r for r, _ in corners)
    c0 = min(col for _, col in corners)
    c1 = max(col for _, col in corners)

    interior: set[Coord] = set()
    boundary: set[Coord] = set()
    for r in range(max(0, r0), min(shape[0] - 1, r1) + 1):
        for col in range(max(0, c0), min(shape[1] - 1, c1) + 1):
            kind = _classify_point((r, col), a, c)
            if kind == "inside":
                interior.add((r, col))
            elif kind == "boundary":
                boundary.add((r, col))
    return interior, boundary


def _boundary_edge_sets(a: Coord, c: Coord) -> tuple[set[Coord], set[Coord]]:
    diag, axis, b, d = _parallelogram_geometry(a, c)
    axis_edges = _segment_cells(a, d) | _segment_cells(b, c)
    diag_edges = _segment_cells(a, b) | _segment_cells(d, c)
    return axis_edges, diag_edges


def _stamp_footprint(cells: list[Coord], shape: tuple[int, int]) -> set[Coord]:
    H, W = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _pick_spaced(candidates: list[Coord], count: int, occupied: list[Coord], min_gap: int) -> list[Coord]:
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[Coord] = []
    blockers = list(occupied)
    for cell in pool:
        if all(_chebyshev(cell, other) >= min_gap for other in blockers + picked):
            picked.append(cell)
            if len(picked) == count:
                break
    return picked


def _adjacent_pairs(cells: list[Coord]) -> list[tuple[Coord, Coord]]:
    cell_set = set(cells)
    pairs: list[tuple[Coord, Coord]] = []
    for cell in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                other = (cell[0] + dr, cell[1] + dc)
                if other in cell_set and cell < other:
                    pairs.append((cell, other))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover an invisible parallelogram from two yellow markers, then recolor or stamp only the red pixels it selects.

    Input structure:
    - The grid has a black background, exactly two yellow marker pixels, and any number of red pixels.
    - The yellow markers differ in both row and column, and their displacement is neither axis-aligned nor a pure 45-degree diagonal.

    Rule:
    - Decompose the marker-to-marker displacement into one maximal 45-degree diagonal vector plus one remaining horizontal or vertical vector, and use those two vectors as adjacent edges of a hidden parallelogram whose opposite corners are the yellow markers.
    - Red pixels strictly inside that parallelogram keep their red centers and paint every in-bounds cell of their 3x3 neighborhood blue wherever the input was black.
    - Red pixels lying exactly on the hidden outline become green, while red pixels outside the outline stay red.

    Key insight:
    - The controlling region is never drawn; the only way to classify each red pixel is to reconstruct the precise lattice outline of the implicit diagonal-plus-axis parallelogram from the two yellow generators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, YELLOW}, "unexpected colors in input"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    assert len(markers) == 2, "expected exactly two yellow markers"
    a, c = markers
    _decompose_displacement(a, c)

    red_pixels = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    kinds = {coord: _classify_point(coord, a, c) for coord in red_pixels}

    out = g.copy()
    for coord, kind in kinds.items():
        if kind == "boundary":
            out[coord] = GREEN

    for r, col in red_pixels:
        if kinds[(r, col)] != "inside":
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = col + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == BLACK:
                    out[rr, cc] = BLUE

    return out


def _sample_geometry() -> tuple[tuple[int, int], Coord, Coord]:
    axis_kind = random.choice(("horizontal", "vertical"))
    diag_len = random.randint(2, 6)
    axis_len = random.randint(2, 7)
    row_sign = random.choice((-1, 1))
    col_sign = random.choice((-1, 1))

    diag = (row_sign * diag_len, col_sign * diag_len)
    axis = (0, col_sign * axis_len) if axis_kind == "horizontal" else (row_sign * axis_len, 0)

    offsets = [(0, 0), diag, axis, (diag[0] + axis[0], diag[1] + axis[1])]
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1

    H = min(30, max(12, bbox_h + random.randint(5, 12)))
    W = min(30, max(12, bbox_w + random.randint(5, 12)))
    assert H - bbox_h >= 4 and W - bbox_w >= 4

    top_margin = random.randint(2, H - bbox_h - 2)
    left_margin = random.randint(2, W - bbox_w - 2)
    anchor = (top_margin - min_r, left_margin - min_c)
    opposite = _add(anchor, (diag[0] + axis[0], diag[1] + axis[1]))
    return (H, W), anchor, opposite


def _select_interior_reds(interior_cells: list[Coord]) -> list[Coord]:
    pairs = _adjacent_pairs(interior_cells)
    if pairs and random.random() < 0.55:
        first, second = random.choice(pairs)
        selected = [first, second]
        extras = _pick_spaced(
            [cell for cell in interior_cells if cell not in selected],
            1,
            occupied=selected,
            min_gap=2,
        )
        return selected + extras

    target = random.randint(1, min(3, len(interior_cells)))
    selected = _pick_spaced(interior_cells, target, occupied=[], min_gap=2)
    if not selected:
        return []
    return selected


def _generate_attempt() -> np.ndarray | None:
    shape, anchor, opposite = _sample_geometry()
    H, W = shape
    interior_cells, boundary_cells = _scan_parallelogram(shape, anchor, opposite)
    if len(interior_cells) < 1 or len(boundary_cells) < 6:
        return None

    interior_reds = _select_interior_reds(sorted(interior_cells))
    if not interior_reds:
        return None
    footprint = _stamp_footprint(interior_reds, shape)

    yellow_set = {anchor, opposite}
    axis_edges, diag_edges = _boundary_edge_sets(anchor, opposite)
    axis_candidates = sorted((axis_edges - yellow_set) - footprint)
    diag_candidates = sorted((diag_edges - yellow_set) - footprint)
    if not axis_candidates or not diag_candidates:
        return None

    occupied = [anchor, opposite, *interior_reds]
    boundary_reds = _pick_spaced(diag_candidates, 1, occupied, min_gap=2)
    if not boundary_reds:
        return None
    occupied += boundary_reds

    axis_pick = _pick_spaced(axis_candidates, 1, occupied, min_gap=2)
    if not axis_pick:
        axis_pick = _pick_spaced(axis_candidates, 1, occupied, min_gap=1)
    if not axis_pick:
        return None
    boundary_reds += axis_pick
    occupied += axis_pick

    extra_boundary_pool = [cell for cell in boundary_cells if cell not in yellow_set and cell not in footprint and cell not in boundary_reds]
    if extra_boundary_pool and random.random() < 0.5:
        extra_boundary = _pick_spaced(extra_boundary_pool, 1, occupied, min_gap=2)
        if not extra_boundary:
            extra_boundary = _pick_spaced(extra_boundary_pool, 1, occupied, min_gap=1)
        boundary_reds += extra_boundary
        occupied += extra_boundary

    outside_candidates: list[Coord] = []
    for r in range(H):
        for c in range(W):
            cell = (r, c)
            if cell in yellow_set or cell in footprint:
                continue
            if _classify_point(cell, anchor, opposite) != "outside":
                continue
            if any(_chebyshev(cell, other) < 2 for other in occupied):
                continue
            outside_candidates.append(cell)
    if len(outside_candidates) < 2:
        return None

    outside_target = random.randint(2, min(4, len(outside_candidates)))
    outside_reds = _pick_spaced(outside_candidates, outside_target, occupied, min_gap=2)
    if len(outside_reds) < 2:
        outside_reds = _pick_spaced(outside_candidates, 2, occupied, min_gap=1)
    if len(outside_reds) < 2:
        return None

    grid = np.zeros(shape, dtype=int)
    grid[anchor] = YELLOW
    grid[opposite] = YELLOW
    for cell in interior_reds + boundary_reds + outside_reds:
        grid[cell] = RED

    out = main(grid)
    if np.count_nonzero(out == BLUE) < 5:
        return None
    if np.count_nonzero(out == GREEN) != len(boundary_reds):
        return None
    if np.count_nonzero(out != grid) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-parallelogram stamp input")
