# Variant of a395ee82
# Dimension: D6 — 晶格间距与缩放关系 (meta)
# Change: 晶格间距gunit=2(固定)，stamp间距=晶格偏移×模板尺寸(h,w)——晶格结构与stamp阵列之间存在尺寸缩放... → 间距编码在输入中

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis:
#   variant:
#     core_change:
#       - The lattice spacing is no longer inferred from a fixed convention or from the gcd
#         of all singleton offsets.
#       - Two extra ruler pixels are placed on one outer edge. Their axis-aligned separation
#         equals gunit, so the solver must decode spacing before reading the interior lattice.
#     input_visual:
#       - One monochrome template component of color T remains unchanged.
#       - The lattice still consists of 5-6 isolated singletons: one T-colored origin and the
#         rest O-colored points, but now all of them are kept strictly away from the border.
#       - A separate pair of O-colored singletons sits on the border, with at least one bg gap
#         around each marker so they remain isolated components.
#     behavior_trace:
#       - Original flow: collect every singleton -> use origin-centered gcd to recover gunit ->
#         convert pixel offsets into lattice offsets -> stamp copies.
#       - Variant flow: collect every singleton -> peel off the unique border pair -> read
#         gunit from its separation -> normalize only the interior singleton lattice -> stamp
#         copies exactly as before.
#       - Difference: if the border markers were left inside the gcd pool, old logic would often
#         collapse to gcd 1. The generator explicitly enforces that failure mode so the new
#         decoding step is genuinely required.
#     complexity_calibration:
#       - gunit is limited to {2, 3, 4}. This is visibly variable, but still leaves enough room
#         for the lattice, the template, and the edge ruler inside a 22x22 grid.
#       - The ruler uses the same visible color O as the overlap singletons, so no hidden
#         color-id convention is introduced; the only cue is its distinctive edge placement.
#     implicit_assumptions_review:
#       - `template = max(comps, key=area)` still holds because the ruler and lattice remain
#         isolated single-pixel components, far smaller than the template.
#       - The original `gcd` block relied on "every singleton belongs to the lattice". That
#         assumption is invalid now; the solver replaces it with "exactly two singleton
#         components lie on the border and form the ruler".
#       - The original origin detection by color T still holds because only one interior
#         singleton uses color T; the ruler uses O.
#       - Integer divisibility of lattice offsets by gunit is now an explicit invariant and is
#         asserted after ruler removal.
#     placement_legality:
#       - The input template is kept at least one cell away from the border so border markers
#         cannot merge with it under non-monochromatic connectivity.
#       - Interior lattice singletons are also kept off the border, ensuring the solver can
#         identify the ruler unambiguously as the only border singleton pair.
#       - The ruler search rejects placements that would let the old gcd solver recover the same
#         gunit; this quantifies that the D6 mechanism actually changes the reasoning path.
#     visual_observability:
#       - Across examples, the border pair visibly changes length while the stamp spacing in the
#         output stays tied to the recovered lattice unit. A human can notice that the two edge
#         pixels are the only new structure absent from the original task family.
#     cleanliness:
#       - No object classes overlap in the input: template, interior lattice, and ruler all have
#         separate bboxes/components with bg gaps, keeping the scene readable.
#     solver_assumptions_audit:
#       - Every loop that processes "singleton lattice points" now operates on `lattice_pts`,
#         not on the full singleton list.
#       - The ruler decoder assumes the two border singletons are axis-aligned and same-colored;
#         the generator guarantees this.
#       - The stamping loop assumes normalized offsets are integral. The generator guarantees each
#         interior singleton lies on the gunit lattice relative to the T-colored origin.


def _decode_ruler(
    singletons: list[tuple[int, int, int]],
    H: int,
    W: int,
) -> tuple[int, set[int]]:
    """Return (gunit, singleton indices used by the edge ruler)."""
    border = [
        (idx, r, c, col)
        for idx, (r, c, col) in enumerate(singletons)
        if r in (0, H - 1) or c in (0, W - 1)
    ]
    assert len(border) == 2
    i1, r1, c1, col1 = border[0]
    i2, r2, c2, col2 = border[1]
    assert int(col1) == int(col2)
    assert (r1 == r2) != (c1 == c2)
    gunit = abs(c1 - c2) if r1 == r2 else abs(r1 - r2)
    assert gunit >= 2
    return int(gunit), {int(i1), int(i2)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template using a lattice whose spacing is encoded by an edge ruler.

    Input structure:
    - bg is the background. One large monochrome component is the template, colored T.
    - Away from the template there is an interior singleton lattice: exactly one singleton is T (the origin) and the rest are O.
    - A separate pair of O-colored singleton pixels sits on the outer border; their gap is the lattice unit gunit.

    Rule:
    - Read gunit from the border ruler before interpreting singleton offsets.
    - Ignore the ruler pixels, then convert each interior singleton offset from the T-colored origin into lattice units by dividing by gunit.
    - Stamp template copies at those normalized offsets scaled by the template bbox size; the origin copy and any overlap are recolored O.

    Key insight:
    - The new difficulty is separating "metadata pixels" from the actual lattice. The border pair is not another stamp location; it only tells you how coarse one lattice step is.
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

    gunit, ruler_ids = _decode_ruler(singletons, H, W)
    lattice_pts = [pt for idx, pt in enumerate(singletons) if idx not in ruler_ids]
    assert lattice_pts
    assert all(0 < r < H - 1 and 0 < c < W - 1 for r, c, _col in lattice_pts)

    origin = [(r, c) for (r, c, col) in lattice_pts if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in lattice_pts if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas: list[tuple[int, int, int]] = []
    for r, c, col in lattice_pts:
        dr = int(r - orr)
        dc = int(c - orc)
        assert dr % gunit == 0 and dc % gunit == 0
        deltas.append((int(dr // gunit), int(dc // gunit), int(col)))

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

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
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator (a1, D6 variant) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT_CHOICES = [2, 3, 4]

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 60
RULER_RETRIES = 60
GENERATE_RETRIES = 600


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
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _old_solver_gcd(
    origin_rc: tuple[int, int],
    lattice_positions: list[tuple[int, int]],
    ruler_positions: list[tuple[int, int]],
) -> int:
    """gcd that the original solver would have recovered if it treated rulers as lattice points."""
    orr, orc = origin_rc
    gcd_vals = []
    for r, c in lattice_positions + ruler_positions:
        dr = int(r - orr)
        dc = int(c - orc)
        for x in (dr, dc):
            if x != 0:
                gcd_vals.append(abs(int(x)))
    assert gcd_vals
    return int(math.gcd(*gcd_vals))


def _find_ruler_positions(
    gunit: int,
    blocked: set[tuple[int, int]],
    origin_rc: tuple[int, int],
    lattice_positions: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """
    Find an isolated edge pair whose separation is gunit and whose inclusion would break
    the original gcd-based solver.
    """
    candidates = []
    for start in range(1, GRID_SIZE - gunit - 1):
        candidates.append([(0, start), (0, start + gunit)])
        candidates.append([(GRID_SIZE - 1, start), (GRID_SIZE - 1, start + gunit)])
        candidates.append([(start, 0), (start + gunit, 0)])
        candidates.append([(start, GRID_SIZE - 1), (start + gunit, GRID_SIZE - 1)])
    random.shuffle(candidates)

    for pair in candidates[:RULER_RETRIES * 8]:
        pair_set = set(pair)
        if any(pos in blocked for pos in pair):
            continue
        ok = True
        for r, c in pair:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) in blocked:
                    ok = False
                    break
                if (nr, nc) in pair_set and (nr, nc) != (r, c):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if _old_solver_gcd(origin_rc, lattice_positions, pair) == int(gunit):
            continue
        return pair
    return None


def _place_singletons_and_ruler(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
    gunit: int,
) -> np.ndarray | None:
    """Place the interior singleton lattice and the edge ruler, then validate with main()."""
    orr_lo = max(1, 1 - min_dr * gunit)
    orr_hi = min(GRID_SIZE - 2, GRID_SIZE - 2 - max_dr * gunit)
    orc_lo = max(1, 1 - min_dc * gunit)
    orc_hi = min(GRID_SIZE - 2, GRID_SIZE - 2 - max_dc * gunit)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + max(tr for tr, _tc in templ_cells)
    tc0, tc1 = c0, c0 + max(tc for _tr, tc in templ_cells)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * gunit
        sr1 = orr + max_dr * gunit
        sc0 = orc + min_dc * gunit
        sc1 = orc + max_dc * gunit
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        lattice_positions = [(orr + dr * gunit, orc + dc * gunit) for dr, dc in deltas]
        if any(pos in occupied for pos in lattice_positions):
            continue
        if any(not (0 < r < GRID_SIZE - 1 and 0 < c < GRID_SIZE - 1) for r, c in lattice_positions):
            continue

        blocked = set(occupied) | set(lattice_positions)
        ruler_positions = _find_ruler_positions(gunit, blocked, (orr, orc), lattice_positions)
        if ruler_positions is None:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            rr = orr + dr * gunit
            cc = orc + dc * gunit
            grid[rr, cc] = T if (dr == 0 and dc == 0) else O
        for rr, cc in ruler_positions:
            grid[rr, cc] = O

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, add a ruler-coded spacing."""
    bg, T, O = random.sample(range(10), 3)
    gunit = random.choice(GUNIT_CHOICES)

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

    r0_lo = max(1, -min_dr * h)
    r0_hi = min(GRID_SIZE - h - 2, GRID_SIZE - max_dr * h - h)
    c0_lo = max(1, -min_dc * w)
    c0_hi = min(GRID_SIZE - w - 2, GRID_SIZE - max_dc * w - w)
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

    return _place_singletons_and_ruler(
        deltas,
        min_dr,
        max_dr,
        min_dc,
        max_dc,
        templ_cells,
        r0,
        c0,
        bg,
        T,
        O,
        occupied,
        gunit,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
