# Variant of a395ee82
# Dimension: D4 — stamp着色变换规则 (transform)
# Change: recolor + pattern_stamping — 原点stamp着O色，其余stamp着T色；重叠区域(total>=2)着O色
#   → 旋转变换替代重着色：所有stamp都保持模板色T，但原点对应的那一份要旋转180度后再放置

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D4 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   inherited_structure:
#     grid: 22x22, single bg color
#     colors: input still uses 3 colors {bg, T, O}; output uses only {bg, T}
#     template:
#       one monochrome 4-connected component, color T
#       bbox height in {3,4}, bbox width = 3
#       area 6-10, spanning all four bbox edges
#     lattice_markers:
#       5-6 singleton pixels on a gunit=2 lattice
#       exactly one singleton is color T and marks the origin
#       remaining singletons are color O and mark the other lattice sites
#       lattice shape is a 4-connected polyomino with strictly interior origin
#   d4_change:
#     original_output:
#       every lattice site receives an unrotated template copy
#       origin copy and overlaps are recolored to O
#     new_output:
#       every lattice site receives a template-colored copy
#       the unique origin copy is rotated 180 degrees inside its own bbox
#       there is no recolor-to-O step in the output
#   concrete_behavior_trace:
#     - Suppose the template bbox is 4x3 and the template cells are an asymmetric
#       "hook". In the input, that hook appears once in color T; the lattice
#       markers elsewhere contain one T singleton (origin) and several O singletons.
#     - Original logic would stamp the same hook silhouette at every translated
#       location, then recolor only the origin copy to O.
#     - New logic keeps all copies in color T. The copy at delta (0,0) relative to
#       the T singleton uses the 180-degree-rotated hook instead of recoloring.
#     - The solver therefore still uses the color-T singleton to identify the
#       special lattice site, but the distinguishing signal in the output is now
#       geometric rather than chromatic.
#   hidden_assumptions_audit:
#     - `template = max(comps, key=area)` assumes the multi-cell template is larger
#       than every singleton. This still holds.
#     - `templ_col` is used to locate the unique origin singleton. This still holds.
#     - The old solver computed `overlap_col` and wrote O into the origin/overlap
#       region. That assumption no longer holds, so the recolor bookkeeping is
#       removed entirely.
#     - The old `center` coverage array existed only to identify pixels belonging
#       to the origin copy. In the variant, we instead switch the relative stencil
#       from `templ_rel` to `templ_rot_rel` when processing the origin delta.
#     - `off_r = dr * h`, `off_c = dc * w` assumes stamps are aligned to the
#       template bbox dimensions. Rotating by 180 degrees keeps the same bbox, so
#       this alignment remains valid and needs no further change.
#   legality_and_generator_guarantees:
#     - The generator still places all copies on the same bbox-scaled lattice and
#       ensures every copy fits in bounds.
#     - The special origin copy is rotated only within its own bbox, so no new
#       out-of-bounds cases are introduced.
#     - To prevent visual degeneration, template generation rejects any shape that
#       is 180-degree symmetric and also rejects shapes whose rotated silhouette is
#       too similar to the original. This ensures `main(input)` differs from a
#       naive "all copies unrotated" solver on every generated sample.
#   observability:
#     - Humans can compare the input template orientation with the many output
#       copies and notice that exactly one copy is upside-down, at the position
#       indicated by the unique T-colored singleton in the lattice.
#     - The generator keeps 5-6 lattice sites so the upright orientation repeats
#       several times, providing evidence for the single rotated exception.
#   cleanliness:
#     - No extra overlapping color layers remain in the output; all output pixels
#       are either bg or T. This makes the special behavior easier to read than the
#       old recolor-based rule.


def _rot180_rel(cells: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    return [(int(h - 1 - r), int(w - 1 - c)) for r, c in cells]


def _rotation_diff(cells: list[tuple[int, int]], h: int, w: int) -> int:
    base = {(int(r), int(c)) for r, c in cells}
    rot = {(int(h - 1 - r), int(w - 1 - c)) for r, c in cells}
    return int(len(base ^ rot))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton lattice.

    Input structure:
    - bg = background. A large monochrome template component uses color T.
    - Singleton pixels elsewhere form a uniform-spacing lattice; exactly one singleton is color T and marks the lattice origin, while the rest share color O.

    Rule:
    - Convert each singleton offset from the T-colored origin into coarse lattice units.
    - Place template copies at those offsets, scaled by the template bbox height/width.
    - All copies keep color T, but the origin copy is placed as the template rotated 180 degrees inside the same bbox.

    Key insight: The unique T-colored singleton still identifies the special lattice site, but the output distinguishes that site through a geometric flip rather than a recolor.
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
    templ_rot_rel = _rot180_rel(templ_rel, h, w)

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
    assert any(int(col) != int(templ_col) for (_r, _c, col) in singletons)
    orr, orc = origin[0]

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = int(math.gcd(*gcd_vals))
    gunit = max(1, int(gunit))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    out = np.full((H, W), int(bg), dtype=int)
    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        rels = templ_rot_rel if int(col) == int(templ_col) else templ_rel
        for rr, cc in rels:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = int(templ_col)
    return out


# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
ROTATION_DIFF_MIN = 4

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 80
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of *n* cells in a rows x cols grid."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier: list[tuple[int, int]] = []
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
    if len(cells) != n:
        return None
    return sorted((int(r), int(c)) for r, c in cells)


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
    """4-connected template shape in h x w, spanning bbox, and visibly altered by 180-degree rotation."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is None:
            continue
        if _rotation_diff(cells, h, w) < ROTATION_DIFF_MIN:
            continue
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
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _all_stamp_points(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    templ_rot_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        rels = templ_rot_cells if (int(dr) == 0 and int(dc) == 0) else templ_cells
        for tr, tc in rels:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)
    return occupied


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
) -> np.ndarray | None:
    """Try to find a valid singleton cluster position and build the input grid."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr * GUNIT)
        sr1 = int(orr + max_dr * GUNIT)
        sc0 = int(orc + min_dc * GUNIT)
        sc1 = int(orc + max_dc * GUNIT)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((int(orr + dr * GUNIT), int(orc + dc * GUNIT)) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for dr, dc in deltas:
            grid[int(orr + dr * GUNIT), int(orc + dc * GUNIT)] = int(T if (dr == 0 and dc == 0) else O)

        try:
            out = main(grid)
        except Exception:
            continue

        origin_plain = {(int(r0 + tr), int(c0 + tc)) for tr, tc in templ_cells}
        origin_out = {(int(r), int(c)) for r, c in np.argwhere(out == int(T)) if r0 <= int(r) < r0 + h and c0 <= int(c) < c0 + w}
        if origin_out == origin_plain:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place template + singletons."""
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None
    templ_rot_cells = _rot180_rel(templ_cells, h, w)

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied = _all_stamp_points(deltas, templ_cells, templ_rot_cells, r0, c0, h, w)
    if occupied is None:
        return None

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
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
