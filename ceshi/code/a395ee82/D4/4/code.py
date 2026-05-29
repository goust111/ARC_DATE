# Variant of a395ee82
# Dimension: D4 — stamp着色变换规则 (transform)
# Change: recolor + pattern_stamping — 原点stamp着O色，其余stamp着T色；重叠区域(total>=2)着O色 → stamp擦除模式

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# variant_spec:
#   task_family: translated template stamping from a singleton lattice
#   preserved_structure:
#     - grid size stays 22x22
#     - input still uses exactly 3 colors: bg, template color T, lattice marker color O
#     - the large monochrome template in the input is still the same object the lattice refers to
#     - singleton count / lattice spacing / origin-inside-lattice constraints stay unchanged
#   d4_core_change:
#     before:
#       - every lattice point produced one template copy
#       - the origin copy was recolored from T to O
#       - overlap regions were also recolored to O
#     after:
#       - non-origin lattice points still stamp the template in T
#       - the origin lattice point no longer paints anything at all
#       - cells covered only by the origin copy remain bg, creating a missing-copy hole
#   behavior_trace:
#     small_example:
#       - template bbox 3x3, template area 7
#       - lattice deltas = {(-1,0), (0,0), (0,1), (1,0), (1,1)}
#     original_flow:
#       - stamp five copies
#       - the center copy at delta (0,0) becomes O, so output contains a visibly recolored center sprite
#     variant_flow:
#       - stamp only the four non-origin copies in T
#       - the delta (0,0) location is left as bg instead of becoming O
#     visible_difference:
#       - original output has a colored center object
#       - variant output has a template-shaped gap at that same location
#   complexity_calibration:
#     - keep exactly one T-colored singleton as the origin cue; this gives a single unambiguous missing position
#     - keep the other 4-5 lattice points marked by one shared O color; no color-id-based branching
#     - keep the origin strictly interior in the lattice polyomino so the missing copy is structurally informative
#     - keep 4-5 surviving copies so the user sees a repeated pattern with one absent member
#   implicit_assumption_review:
#     - template = largest component: still valid because the template area is 6-10 and all other components are singletons
#     - origin found by singleton color == templ_col: still valid and still the only role of the T singleton
#     - gcd of singleton offsets gives lattice unit: still valid because the lattice spacing is unchanged
#     - old solver line "paint everything T, then recolor origin/overlap to O" is no longer valid
#       because erase semantics depend on whether any non-origin copy covers a cell
#     - new solver therefore tracks origin coverage separately and paints a cell iff it has non-origin coverage
#   placement_legality:
#     - all translated stamps must stay inside the 22x22 grid
#     - the singleton cluster bbox stays disjoint from the template bbox with at least one bg cell gap
#     - non-overlap between stamps is intentionally preserved from the original generator
#       so the erased origin remains a clean full-shape hole instead of being partially refilled
#   observability:
#     - the output now uses only bg and T, so the D4 change is visible as topology rather than recolor
#     - because the missing copy sits at an interior lattice position, the gap is inferable from the repeated arrangement
#     - no hidden ordering by color id or other non-visual metadata is used
#   cleanliness:
#     - avoiding stamp overlaps keeps the output visually crisp
#     - the input still has just two object groups: one template and one singleton lattice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template from a singleton lattice, but leave the origin slot blank.

    Input structure:
    - bg is the background color.
    - One large monochrome component is the template sprite, colored T.
    - Elsewhere there is a lattice of singleton pixels: exactly one singleton is also T, while the others share a second color O.
    - The singleton lattice has uniform spacing and the T singleton marks the lattice origin.

    Rule:
    - Read every singleton's offset from the T-colored origin and normalize it by the lattice spacing.
    - Convert each lattice offset into a translation measured in template bbox heights and widths.
    - For every non-origin lattice point, stamp the template at the translated location using color T.
    - For the origin lattice point, do not draw the template at all, so cells that only this copy would occupy stay bg.

    Key insight: The unique T singleton no longer says "recolor this copy" — it identifies which translated template instance should be missing entirely.
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

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    marker_col = int(other_cols[0])
    assert all(int(col) == int(marker_col) for col in other_cols)

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    origin_cover = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin_cover[R, C] += 1

    non_origin_cover = total - origin_cover
    out = np.full((H, W), int(bg), dtype=int)
    out[non_origin_cover > 0] = int(templ_col)
    return out


# --------------- generator (a1) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]          # template bbox height options
TEMPLATE_W = 3                       # template bbox width (fixed)
TEMPLATE_AREA_RANGE = (6, 10)        # absolute cell-count bounds
TEMPLATE_FILL_RANGE = (0.65, 0.85)   # fill-ratio bounds within bbox
SINGLETON_COUNT_RANGE = (5, 6)       # number of lattice singletons
LATTICE_EXTENT_RANGE = (3, 4)        # rows/cols of lattice polyomino
GUNIT = 2                            # singleton spacing in pixels

# Retry budgets
SPANNING_RETRIES = 40                # attempts per spanning polyomino
TEMPLATE_RETRIES = 60                # attempts to build a valid template
SINGLETON_RETRIES = 50               # attempts to place singleton cluster
GENERATE_RETRIES = 300               # outer retry loop


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of *n* cells in a rows x cols grid."""
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
    """Polyomino of *n* cells that spans exactly rows x cols (touches all 4 edges)."""
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
    """Lattice polyomino of *n* cells with strictly interior origin."""
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
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
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
    templ_col: int,
    marker_col: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Try to find a valid singleton cluster position and build the input grid."""
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
            grid[r0 + tr, c0 + tc] = templ_col
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = templ_col if (dr == 0 and dc == 0) else marker_col

        try:
            out = main(grid)
        except Exception:
            continue

        # Visibility guard: the erased origin must create a real gap.
        if int((out == int(templ_col)).sum()) <= len(templ_cells):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
    bg, templ_col, marker_col = random.sample(range(10), 3)

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
        templ_col,
        marker_col,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
