# Variant of a395ee82
# Dimension: D14 — stamp几何变换 (transform)
# Change: 所有stamp使用相同朝向(无旋转/翻转)，仅有平移和重着色... → 晶格方向决定翻转

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior / trace:
# - Input still contains one large monochrome template component in color T plus a
#   separate singleton lattice with one T-colored origin and the rest colored O.
# - Output still stamps one copy per singleton delta, using bbox-sized steps
#   (dr * h, dc * w), recolors the origin copy to O, and recolors overlaps to O.
# - The new behavior is local to the stamping step: each copy is mirrored inside its
#   own bbox before placement. A point left of the origin (dc < 0) uses a horizontal
#   mirror; a point above the origin (dr < 0) uses a vertical mirror; upper-left
#   points apply both mirrors. Example with a 4x3 asymmetric template:
#   original old rule: delta (-1, 0) -> same shape shifted upward.
#   new D14 rule: delta (-1, 0) -> vertically mirrored shape shifted upward.
#   delta (0, -1) changes by horizontal mirroring, and delta (-1, -1) changes by both.
#
# b) Complexity calibration:
# - If the template were symmetric under horizontal or vertical reflection, the new
#   transform would be visually invisible. The generator therefore rejects templates
#   invariant under H, V, or HV reflection.
# - To make the rule inferable from 2-4 examples, the lattice is constrained so the
#   origin always has a same-row left neighbor, a same-column upper neighbor, and an
#   upper-left neighbor. This gives explicit evidence for horizontal, vertical, and
#   composed reflection in every sample.
#
# c) Implicit-assumption audit of the original main():
# - `templ_rel` was computed once and reused for every stamp. That relied on the old
#   "all copies have identical orientation" invariant; it is no longer valid.
# - Placement windows based on template bbox height/width still hold because flips do
#   not change bbox size.
# - Color parsing still works unchanged: the unique singleton with color T remains the
#   origin marker, and the majority non-T singleton color remains the overlap/origin
#   output color O.
# - Overlap recoloring cannot be done by directly painting per-copy colors, because a
#   later copy could overwrite an earlier overlap. Coverage counting remains the safe
#   approach, now with transformed cell sets per delta.
#
# d) Placement legality:
# - Reflections stay inside each copy's h x w bbox, so the original step size
#   (multiples of h and w) still guarantees non-overlapping bboxes when stamps fit.
# - The generator keeps the singleton cluster disjoint from the template by at least
#   one background cell, and validates that the resulting grid is accepted by main().
#
# e) Visual observability:
# - Every generated lattice shows left, up, and upper-left cases around the origin.
# - Every generated template visibly changes under H, V, and HV reflection.
# - No color-id ordering or hidden metadata is used; all reasoning comes from visible
#   geometry and the already-existing T/O color roles.
#
# f) Visual cleanliness:
# - The scene still has only two object families: one template component and isolated
#   singletons. Generated data keeps stamp bboxes non-overlapping, so the reflected
#   silhouettes remain easy to read.
#
# g) Color inference:
# - Colors are unchanged from the base task: T is the template color, O is the
#   recolor/overlap color, and bg is the background. The variant changes geometry
#   only; color behavior remains structurally signaled by the unique T singleton.


def _transform_cells(
    cells: list[tuple[int, int]], h: int, w: int, flip_v: bool, flip_h: bool
) -> list[tuple[int, int]]:
    """Reflect relative template cells inside their h x w bbox."""
    out: list[tuple[int, int]] = []
    for rr, cc in cells:
        nr = h - 1 - rr if flip_v else rr
        nc = w - 1 - cc if flip_h else cc
        out.append((int(nr), int(nc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp mirrored copies of a monochrome template guided by a singleton pixel layout.

    Input structure:
    - bg is the background. One large monochrome component is the template in color T.
    - A detached singleton lattice marks stamp locations; exactly one singleton is also
      color T and serves as the lattice origin, while the others are color O.
    - Singleton spacings are uniform, so lattice offsets can be read in coarse grid units.

    Rule:
    - Convert each singleton's offset from the T-colored origin into lattice deltas.
    - Place one template-sized copy per delta, using row steps of template height and
      column steps of template width.
    - Mirror the copy vertically when the lattice point is above the origin, mirror it
      horizontally when the point is left of the origin, and apply both mirrors for
      upper-left points.
    - Paint the origin copy in O instead of T, and recolor any overlap area to O.

    Key insight: The singleton lattice still gives translation offsets, but now the sign
    of each offset also selects the reflection applied inside that copy's bbox.
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
    transformed = {
        (flip_v, flip_h): _transform_cells(templ_rel, h, w, flip_v, flip_h)
        for flip_v in (False, True)
        for flip_h in (False, True)
    }

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

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        stamp_cells = transformed[(int(dr) < 0, int(dc) < 0)]
        is_origin = int(col) == int(templ_col)
        for rr, cc in stamp_cells:
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


# --------------- generator (D14) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (6, 6)   # need left / up / upper-left evidence in every sample
LATTICE_EXTENT_RANGE = (3, 3)    # 3x3 is the smallest clean extent supporting those roles
GUNIT = 2

SPANNING_RETRIES = 120
TEMPLATE_RETRIES = 120
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 500


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
    """Polyomino of n cells that spans exactly rows x cols (touches all 4 edges)."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _cell_set(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _has_visible_flips(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    """Require all reflection modes to visibly change the silhouette."""
    base = _cell_set(cells)
    return all(
        _cell_set(_transform_cells(cells, h, w, flip_v, flip_h)) != base
        for flip_v, flip_h in [(False, True), (True, False), (True, True)]
    )


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    """4-connected template shape in h x w, spanning bbox, asymmetric under reflections."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None and _has_visible_flips(cells, h, w):
            return cells
    return None


def _choose_origin(cells: list[tuple[int, int]], rows: int, cols: int) -> tuple[int, int] | None:
    """
    Pick an origin that visibly demonstrates the new rule:
    left, up, upper-left, and at least one right and one down witness.
    """
    cell_set = _cell_set(cells)
    candidates: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 < r < rows - 1 and 0 < c < cols - 1):
            continue
        if (r, c - 1) not in cell_set:
            continue
        if (r - 1, c) not in cell_set:
            continue
        if (r - 1, c - 1) not in cell_set:
            continue
        if not any(rr == r and cc > c for rr, cc in cell_set):
            continue
        if not any(rr > r and cc == c for rr, cc in cell_set):
            continue
        candidates.append((int(r), int(c)))
    return random.choice(candidates) if candidates else None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with a reflection-demonstrating interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    origin = _choose_origin(cells, ext_r, ext_c)
    if origin is None:
        return None
    return cells, origin


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _stamp_points(
    templ_cells: list[tuple[int, int]], h: int, w: int, r0: int, c0: int, deltas: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Real occupied output pixels for all reflected copies."""
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        stamp = _transform_cells(templ_cells, h, w, dr < 0, dc < 0)
        for tr, tc in stamp:
            occupied.add((int(r0 + dr * h + tr), int(c0 + dc * w + tc)))
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
    """Find a valid singleton-cluster position and build the input grid."""
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
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = T if (dr == 0 and dc == 0) else O

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, then place them."""
    bg, T, O = random.sample(range(10), 3)

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
    occupied = _stamp_points(templ_cells, h, w, r0, c0, deltas)

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
