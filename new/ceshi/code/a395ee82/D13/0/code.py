# Variant of a395ee82
# Dimension: D13 — 颜色数量与角色分配 (meta)
# Change: 3种颜色 — bg(背景), T(模板色/原点标识), O(晶格点色/原点stamp重着色)... → 4色方案：分离原点标识和重着色

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D13 variant
# - Concrete behavior:
#   Input still has one large monochrome template and one distant singleton lattice,
#   but the lattice now uses two colors instead of one shared with the template:
#   a unique singleton color M marks the lattice origin, while all other singleton
#   points use O. Output logic stays the same as the seed task: every lattice point
#   spawns a translated copy of the template; the copy indexed by the origin is
#   recolored to O, and overlaps also become O.
# - Behavior trace on a small example:
#   Suppose the template bbox is 3x3, and singleton offsets are
#   {(-1,0), (0,0), (0,1), (1,0)} in coarse lattice units. In the seed task, the
#   (0,0) singleton is color T, so solver finds the origin by matching the template
#   color. In this variant, the template remains T but the (0,0) singleton is M.
#   Solver must first isolate the large template component, then inspect only the
#   singleton colors: the unique singleton color is M, therefore that point is the
#   origin; the repeated singleton color is O, therefore it is also the recolor used
#   on the origin stamp and any overlaps.
# - Complexity calibration:
#   The lattice still has 5-6 points and a strictly interior origin. This keeps the
#   origin visually meaningful: M is not just an endpoint or corner, so a viewer can
#   see that it acts as the reference point of the layout. Using exactly four colors
#   preserves a visible D13 change without adding unrelated branching.
# - Old-code assumptions that break here:
#   `origin = ... if col == templ_col` is invalid because no singleton now shares the
#   template color. `other_cols = ... if col != templ_col` would mix O and M together.
#   The new solver therefore identifies origin by singleton frequency, not by matching
#   against the template color.
# - Placement legality:
#   Generator still enforces full in-bounds stamping, no stamp-stamp overlap, and a
#   one-cell gap between the template bbox and the singleton-cluster bbox. Singletons
#   are placed on a uniform gunit=2 lattice and never land on stamp pixels.
# - Visual observability:
#   Across examples, the template is always color T, the repeated lattice points are
#   always O, and exactly one interior singleton is M. Because output recolors the
#   template copy aligned to M into O, the role of M is visible from only a few pairs.
# - New solver assumptions and generator support:
#   Solver assumes the non-template components are all singletons, exactly one
#   singleton color occurs once (M), exactly one different singleton color repeats
#   (O), and the repeated color is the origin-stamp recolor. Generator constructs
#   every sample to satisfy these conditions.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template using a singleton lattice whose origin has its own marker color.

    Input structure:
    - `bg` is the background color.
    - One large 4-connected monochrome component is the template, colored `T`.
    - Separate singleton pixels form a uniform-spacing lattice: exactly one singleton is color `M` and the rest are color `O`.
    - `M` is only an origin marker; it does not appear in the template.

    Rule:
    - Take the unique `M` singleton as the lattice origin.
    - Convert every singleton position into row/column offsets from that origin, then normalize by the lattice spacing.
    - Stamp translated copies of the template using those coarse offsets scaled by the template bbox height and width.
    - Paint covered cells as `T` by default, but recolor the origin copy and all multi-covered cells to `O`.

    Key insight:
    - The origin is no longer the singleton that matches the template color. It is the only singleton whose color is unique within the lattice.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert singletons
    singleton_counts = Counter(col for _r, _c, col in singletons)
    assert int(templ_col) not in singleton_counts
    origin_cols = [col for col, cnt in singleton_counts.items() if int(cnt) == 1]
    assert len(origin_cols) == 1
    origin_col = int(origin_cols[0])
    overlap_cols = [col for col in singleton_counts if int(col) != int(origin_col)]
    assert len(overlap_cols) == 1
    overlap_col = int(overlap_cols[0])

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(origin_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    deltas = [(int(r - orr), int(c - orc), int(col) == int(origin_col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _is_origin in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in deltas]

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)

    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (origin_cover > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Polyomino of n cells that spans exactly rows x cols."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template shape in h x w, spanning bbox, within area/fill bounds."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with a strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_dr, max_dr, min_dc, max_dc) for coarse lattice deltas."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    M: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Place the singleton lattice and validate the resulting input grid."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * GUNIT
        sr1 = orr + max_dr * GUNIT
        sc0 = orc + min_dc * GUNIT
        sc1 = orc + max_dc * GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((orr + dr * GUNIT, orc + dc * GUNIT) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = M if (dr == 0 and dc == 0) else O

        try:
            main(grid)
        except Exception:
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, and place the objects."""
    bg, T, O, M = random.sample(range(10), 4)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (r0 + dr * h + tr, c0 + dc * w + tc)
            if pt in occupied:
                return None
            occupied.add(pt)

    return _place_singletons(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        r0,
        c0,
        h,
        w,
        bg,
        T,
        O,
        M,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
