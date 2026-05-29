# Variant of a395ee82
# Dimension: D6 — 晶格间距与缩放关系 (meta)
# Change: 晶格间距gunit=2(固定)，stamp间距=晶格偏移×模板尺寸(h,w)——晶格结构与stamp阵列之间存在尺寸缩放... → 间距由模板面积决定

from __future__ import annotations

from collections import Counter
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
import random


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   unchanged_core:
#     - grid stays 22x22 with one background color
#     - there is still one monochrome template component in color T
#     - there are still 5-6 singleton markers, exactly one in color T and the rest in color O
#     - output still stamps translated copies of the template on a coarse lattice, and recolors
#       the origin copy plus all overlaps to O
#   d6_change:
#     - the singleton lattice spacing is no longer a fixed 2 pixels
#     - instead gunit is derived from the template's filled-cell area:
#         area 6-7 -> gunit 2
#         area 8-9 -> gunit 3
#         area 10  -> gunit 4
#     - the lattice deltas used for stamping are recovered by dividing singleton offsets by
#       this area-driven gunit
#   concrete_trace:
#     - original task: template area 8 still had singleton gaps 2, so an offset of (+4,-2)
#       decoded to lattice delta (+2,-1)
#     - new task: the same area 8 now implies gunit 3, so the input shows the marker at (+6,-3);
#       the decoded lattice delta is still (+2,-1), but only after computing the template area
#       and applying the new spacing rule
#     - visually, larger templates produce visibly wider singleton spacing even when the output
#       stamping rule stays the same
#   complexity_calibration:
#     - only three spacing levels are used: 2, 3, 4
#     - the generator samples these levels intentionally so the relation repeats across examples
#     - this keeps the rule learnable while making the D6 change visible
#   implicit_assumptions_review:
#     - choosing the largest component as the template still works because singletons remain
#       isolated area-1 components
#     - the old solver's gcd-based spacing recovery is replaced with area-based spacing recovery;
#       therefore singleton offsets must be exact multiples of the chosen gunit
#     - the output translation still assumes the template bbox anchor is reused for every stamp
#     - overlap recoloring logic is unchanged, so coverage counting remains valid
#   placement_legality:
#     - all stamps are placed from normalized lattice deltas, exactly as before
#     - singleton positions are placed using the derived gunit and kept disjoint from the
#       template bbox with at least one background cell of separation
#     - the generator verifies that every singleton offset is divisible by the intended gunit and
#       that the constructed input reproduces the expected output
#   observability:
#     - the changed quantity is visible in the input itself: the singleton cluster breathes wider
#       when the template has more filled cells
#     - color ids do not control spacing; only the visible template size does
#   neatness:
#     - no extra markers are added, so the scene remains the original two-object layout:
#       one template block and one singleton cluster
#     - stamps are still constrained to avoid overlap in generated data, preventing clutter


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT_CHOICES = [2, 3, 4]
AREAS_BY_GUNIT = {
    2: [6, 7],
    3: [8, 9],
    4: [10],
}

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 80
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 500


def _gunit_from_area(area: int) -> int:
    area = int(area)
    if area <= 7:
        return 2
    if area <= 9:
        return 3
    return 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton layout whose spacing
    is determined by the template's area.

    Input structure:
    - bg is the background. One large monochrome component is the template in color T.
    - Elsewhere there are singleton pixels: exactly one also has color T and marks the lattice
      origin; the remaining singletons share color O.
    - The singleton cluster lies on a uniform lattice, but the lattice gap is not fixed. It is
      derived from the template's filled-cell count.

    Rule:
    - Measure the template area A and convert it to the singleton spacing gunit.
    - Divide each singleton offset from the T-colored origin by gunit to recover coarse lattice
      deltas.
    - Stamp the template at each recovered delta, scaled by the template bbox height/width.
    - Recolor the origin's stamped copy to O, and recolor any overlap region to O as well.

    Key insight:
    - The T-colored singleton still identifies the layout origin, but the missing scale of the
      singleton lattice must now be inferred from the template's area rather than assumed fixed.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_area = int(template["area"])
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    gunit = int(_gunit_from_area(templ_area))

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
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas: list[tuple[int, int, int]] = []
    for r, c, col in singletons:
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


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
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
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _area_choices_for_bbox(h: int, w: int, target_gunit: int) -> list[int]:
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    return [area for area in AREAS_BY_GUNIT[target_gunit] if lo <= area <= hi]


def _gen_template(h: int, w: int, allowed_areas: list[int]) -> list[tuple[int, int]] | None:
    if not allowed_areas:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.choice(allowed_areas)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
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


def _render_expected(
    bg: int,
    T: int,
    O: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    center = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for dr, dc in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = dr == 0 and dc == 0
        for tr, tc in templ_cells:
            rr = int(r0 + off_r + tr)
            cc = int(c0 + off_c + tc)
            total[rr, cc] += 1
            if is_origin:
                center[rr, cc] += 1

    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    out[total > 0] = int(T)
    out[(total == 1) & (center > 0)] = int(O)
    out[total >= 2] = int(O)
    return out


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
    occupied: set[tuple[int, int]],
    gunit: int,
    expected: np.ndarray,
) -> np.ndarray | None:
    orr_lo, orr_hi = -min_dr * gunit, GRID_SIZE - 1 - max_dr * gunit
    orc_lo, orc_hi = -min_dc * gunit, GRID_SIZE - 1 - max_dc * gunit
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * gunit
        sr1 = orr + max_dr * gunit
        sc0 = orc + min_dc * gunit
        sc1 = orc + max_dc * gunit
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        positions = [(orr + dr * gunit, orc + dc * gunit) for dr, dc in deltas]
        if any(pos in occupied for pos in positions):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            rr = orr + dr * gunit
            cc = orc + dc * gunit
            grid[rr, cc] = T if (dr == 0 and dc == 0) else O

        produced = main(grid)
        if np.array_equal(produced, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, T, O = random.sample(range(10), 3)
    target_gunit = random.choice(GUNIT_CHOICES)

    template_options: list[tuple[int, list[int]]] = []
    for h in TEMPLATE_H_CHOICES:
        allowed_areas = _area_choices_for_bbox(h, TEMPLATE_W, target_gunit)
        if allowed_areas:
            template_options.append((h, allowed_areas))
    if not template_options:
        return None

    h, allowed_areas = random.choice(template_options)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w, allowed_areas)
    if templ_cells is None:
        return None
    templ_area = len(templ_cells)
    gunit = int(_gunit_from_area(templ_area))
    if gunit != target_gunit:
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

    expected = _render_expected(bg, T, O, templ_cells, r0, c0, h, w, deltas)
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
        occupied,
        gunit,
        expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
