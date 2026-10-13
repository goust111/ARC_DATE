from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 30)
TEMPLATE_SIDE_RANGE = (3, 6)
TEMPLATE_FILL_RANGE = (0.45, 0.80)
SINGLETON_COUNT_RANGE = (4, 7)
LATTICE_EXTENT_RANGE = (3, 5)
GRID_UNIT_RANGE = (2, 4)
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _non_bg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    return _component_from_mask(grid != bg)


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h_grid, w_grid = g.shape
    bg = _get_bg(g)

    comps = _non_bg_components(g, bg)
    assert comps

    template = max(comps, key=len)
    template_color = int(g[template[0]])
    r0, c0, r1, c1 = _bbox(template)
    templ_h = r1 - r0 + 1
    templ_w = c1 - c0 + 1
    templ_rel = [(r - r0, c - c0) for r, c in template]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if len(comp) != 1:
            continue
        rr, cc = comp[0]
        singletons.append((rr, cc, int(g[rr, cc])))

    origin_points = [(r, c) for r, c, color in singletons if color == template_color]
    assert len(origin_points) == 1
    origin_r, origin_c = origin_points[0]

    overlap_colors = [color for _r, _c, color in singletons if color != template_color]
    assert overlap_colors
    overlap_color = Counter(overlap_colors).most_common(1)[0][0]

    deltas_raw = [(r - origin_r, c - origin_c, color) for r, c, color in singletons]
    gcd_terms = [abs(v) for dr, dc, _ in deltas_raw for v in (dr, dc) if v != 0]
    assert gcd_terms
    grid_unit = int(math.gcd(*gcd_terms))
    assert grid_unit >= 1

    deltas: list[tuple[int, int, int]] = []
    for dr, dc, color in deltas_raw:
        assert dr % grid_unit == 0 and dc % grid_unit == 0
        deltas.append((dr // grid_unit, dc // grid_unit, color))

    cover_total = np.zeros_like(g, dtype=int)
    cover_origin = np.zeros_like(g, dtype=bool)
    for dr, dc, color in deltas:
        shift_r = dr * templ_h
        shift_c = dc * templ_w
        is_origin = color == template_color
        for rr, cc in templ_rel:
            nr = r0 + shift_r + rr
            nc = c0 + shift_c + cc
            if 0 <= nr < h_grid and 0 <= nc < w_grid:
                cover_total[nr, nc] += 1
                if is_origin:
                    cover_origin[nr, nc] = True

    out = np.full_like(g, bg)
    out[cover_total > 0] = template_color
    out[cover_origin] = overlap_color
    out[cover_total >= 2] = overlap_color
    return out


def _random_connected_subset(area: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.add((nr, nc))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.add((nr, nc))
    if len(cells) != area:
        return None
    return list(cells)


def _spanning_polyomino(area: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        cells = _random_connected_subset(area, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _generate_template(templ_h: int, templ_w: int) -> list[tuple[int, int]] | None:
    total = templ_h * templ_w
    lo = max(5, math.ceil(total * TEMPLATE_FILL_RANGE[0]))
    hi = min(total - 1, math.floor(total * TEMPLATE_FILL_RANGE[1]))
    if lo > hi:
        return None
    for _ in range(SHAPE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, templ_h, templ_w)
        if cells is not None:
            return cells
    return None


def _generate_lattice(count: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        rows = random.randint(*LATTICE_EXTENT_RANGE)
        cols = random.randint(*LATTICE_EXTENT_RANGE)
        if rows * cols < count:
            continue
        cells = _spanning_polyomino(count, rows, cols)
        if cells is None:
            continue
        interior = [(r, c) for r, c in cells if 0 < r < rows - 1 and 0 < c < cols - 1]
        candidates = interior if interior and random.random() < 0.7 else cells
        origin = random.choice(candidates)
        deltas = [(r - origin[0], c - origin[1]) for r, c in cells]
        if len({dr for dr, _ in deltas}) < 2 or len({dc for _, dc in deltas}) < 2:
            continue
        return cells, origin
    return None


def _translated_points(
    template_cells: list[tuple[int, int]],
    base_r: int,
    base_c: int,
    templ_h: int,
    templ_w: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for rr, cc in template_cells:
            points.add((base_r + dr * templ_h + rr, base_c + dc * templ_w + cc))
    return points


def _attempt_grid() -> np.ndarray | None:
    h_grid = random.randint(*GRID_SIDE_RANGE)
    w_grid = random.randint(*GRID_SIDE_RANGE)
    bg, template_color, overlap_color = random.sample(range(10), 3)

    templ_h = random.randint(*TEMPLATE_SIDE_RANGE)
    templ_w = random.randint(*TEMPLATE_SIDE_RANGE)
    if templ_h >= h_grid - 2 or templ_w >= w_grid - 2:
        return None

    template_cells = _generate_template(templ_h, templ_w)
    if template_cells is None:
        return None

    lattice_count = random.randint(*SINGLETON_COUNT_RANGE)
    lattice = _generate_lattice(lattice_count)
    if lattice is None:
        return None
    lattice_cells, origin = lattice
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]

    min_dr = min(dr for dr, _ in deltas)
    max_dr = max(dr for dr, _ in deltas)
    min_dc = min(dc for _, dc in deltas)
    max_dc = max(dc for _, dc in deltas)

    base_r_lo = -min_dr * templ_h
    base_r_hi = h_grid - templ_h - max_dr * templ_h
    base_c_lo = -min_dc * templ_w
    base_c_hi = w_grid - templ_w - max_dc * templ_w
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        base_r = random.randint(base_r_lo, base_r_hi)
        base_c = random.randint(base_c_lo, base_c_hi)
        stamped = _translated_points(template_cells, base_r, base_c, templ_h, templ_w, deltas)
        if len(stamped) != len(template_cells) * len(deltas):
            continue

        unit = random.randint(*GRID_UNIT_RANGE)
        origin_r_lo = -min_dr * unit
        origin_r_hi = h_grid - 1 - max_dr * unit
        origin_c_lo = -min_dc * unit
        origin_c_hi = w_grid - 1 - max_dc * unit
        if origin_r_lo > origin_r_hi or origin_c_lo > origin_c_hi:
            continue

        temp_r0, temp_c0 = base_r, base_c
        temp_r1, temp_c1 = base_r + templ_h - 1, base_c + templ_w - 1

        for _ in range(PLACEMENT_RETRIES):
            origin_r = random.randint(origin_r_lo, origin_r_hi)
            origin_c = random.randint(origin_c_lo, origin_c_hi)
            singletons = [(origin_r + dr * unit, origin_c + dc * unit) for dr, dc in deltas]

            sing_r0 = min(r for r, _ in singletons)
            sing_r1 = max(r for r, _ in singletons)
            sing_c0 = min(c for _, c in singletons)
            sing_c1 = max(c for _, c in singletons)

            if not (
                sing_r1 + 1 < temp_r0
                or temp_r1 + 1 < sing_r0
                or sing_c1 + 1 < temp_c0
                or temp_c1 + 1 < sing_c0
            ):
                continue

            if any(pt in stamped for pt in singletons):
                continue

            grid = np.full((h_grid, w_grid), bg, dtype=int)
            for rr, cc in template_cells:
                grid[base_r + rr, base_c + cc] = template_color
            for (rr, cc), (dr, dc) in zip(singletons, deltas):
                color = template_color if (dr == 0 and dc == 0) else overlap_color
                grid[rr, cc] = color

            try:
                out = main(grid)
            except Exception:
                continue
            if out.shape != grid.shape:
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
