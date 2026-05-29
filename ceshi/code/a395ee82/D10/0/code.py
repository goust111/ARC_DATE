# Variant of a395ee82
# Dimension: D10 — 输出网格尺寸与裁剪 (grid_shape)
# Change: 输出网格与输入相同(22x22)，stamp绘制在原地，未使用裁剪或扩展... → 紧凑裁剪

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D10 tight-crop variant
# ==============================================
# a) Concrete behavior and trace
#    Input stays the same as the source task: one large monochrome template in
#    color T, plus 5-6 singleton markers on a coarse lattice. Exactly one
#    singleton is also T and acts as the lattice origin; the rest are color O.
#    The solver still reconstructs the stamp layout from singleton offsets and
#    still paints the origin copy and any overlaps with O.
#    The D10 change happens after stamping. Originally the solver returned the
#    full 22x22 canvas with a large background frame still present. Here we
#    render on the same absolute coordinates first, then crop to the minimal
#    bbox containing every stamped pixel. Example: if the reconstructed stamps
#    occupy rows 4..15 and cols 6..14, the old output was 22x22; the new output
#    is only the 12x9 subgrid spanning those rows/cols.
# b) Complexity calibration
#    The variant should be visibly different, not a no-op. Tight crop is only
#    noticeable when the stamp union sits strictly inside the 22x22 canvas, so
#    the generator enforces a margin around the entire stamped union. This keeps
#    the rule human-visible across few examples while preserving the original
#    object vocabulary and reconstruction logic.
# c) Hidden assumptions audit in main()
#    - `template = max(comps, key=area)`: still valid because the input format
#      is unchanged and the template remains the only multi-cell large object.
#    - `templ_col` read from the template component: still valid because the
#      template is monochrome.
#    - origin detection by `col == templ_col`: still valid because exactly one
#      singleton keeps the template color.
#    - `gunit = gcd(nonzero singleton deltas)`: unchanged because lattice
#      spacing in the input is still uniform.
#    - old code allocated `out` as `(H, W)` and returned it directly. This is
#      the main old-shape assumption that breaks under D10, so the new solver
#      preserves the full render as an intermediate and then crops it.
# d) Placement legality in the generator
#    The generator still ensures: template and singleton cluster are disjoint,
#    all singleton positions lie in-bounds, all stamp copies fit in-bounds, and
#    stamps do not overlap. New for D10: the full stamped union must stay at
#    least `OUTPUT_MARGIN` cells away from all four borders so the crop is
#    guaranteed to shrink the canvas.
# e) Visual observability
#    A human now sees that output dimensions vary with the stamp union instead
#    of remaining 22x22. The crop is inferred from repeated evidence: the
#    internal stamp pattern stays the same as the source task, but all unused
#    background-only rows and columns disappear.
# f) Visual cleanliness
#    No extra marks are added. The crop removes empty border noise, so the
#    result is actually cleaner than the source task. Keeping no-overlap stamps
#    avoids dense ambiguous merged blobs unrelated to D10.
# g) Color-change inference
#    Color behavior is unchanged from the source task: T marks the template and
#    non-origin copies, O marks the origin copy and any overlaps. The variant
#    does not introduce any new color mapping, so the only new concept for the
#    solver is geometric cropping.


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton lattice, then tight-crop the result.

    Input structure:
    - Background color `bg`, one large monochrome template component in color `T`.
    - Separate singleton markers form a uniformly spaced lattice; exactly one singleton is `T` and the rest are `O`.
    - The `T` singleton is the lattice origin and also identifies which stamped copy should be recolored.

    Rule:
    - Recover lattice offsets by measuring each singleton relative to the `T` singleton and normalizing by the lattice spacing.
    - Stamp the template at those offsets, using template bbox height/width as the translation step.
    - Recolor the origin copy to `O`, and recolor any multiply covered cells to `O` as well.
    - After all stamping is done on the original canvas, crop away every all-background border row and column.

    Key insight:
    - The singleton lattice still determines the same translated stamp layout as the source task; the only new step is that the answer is the minimal bounding box of the stamped figure rather than the full 22x22 canvas.
    """
    out = _solve_uncropped(grid)
    bg = int(get_bg(np.array(grid, dtype=int)))
    r0, c0, r1, c1 = _foreground_bbox(out != int(bg))
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
OUTPUT_MARGIN = 2

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


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


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
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
            cropped = main(grid)
            if cropped.shape == (GRID_SIZE, GRID_SIZE):
                continue
            if cropped.shape[0] > GRID_SIZE - 2 * OUTPUT_MARGIN:
                continue
            if cropped.shape[1] > GRID_SIZE - 2 * OUTPUT_MARGIN:
                continue
        except Exception:
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
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

    r0_lo = OUTPUT_MARGIN - min_dr * h
    r0_hi = GRID_SIZE - OUTPUT_MARGIN - max_dr * h - h
    c0_lo = OUTPUT_MARGIN - min_dc * w
    c0_hi = GRID_SIZE - OUTPUT_MARGIN - max_dc * w - w
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
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GENERATE_RETRIES)
