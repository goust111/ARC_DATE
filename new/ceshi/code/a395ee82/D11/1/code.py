# Variant of a395ee82
# Dimension: D11 — stamp放置基准点 (meta)
# Change: stamp以模板外接矩形左上角为基准点，按晶格偏移×模板尺寸(h,w)进行平移放置
#   → 以单例像素位置为直接stamp中心，取消缩放关系

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D11 variant
# a) Concrete behavior:
#    - Input still contains one monochrome template component (color T) and a separate
#      lattice of singleton pixels with one T-colored origin and the rest color O.
#    - The changed part is only how output copies are positioned: every singleton is now
#      the center of a direct template stamp. There is no multiplication by template
#      height/width, so spacing is much tighter and overlap is expected.
#    - Example trace: for a 3x3 template centered at its middle cell, two singleton
#      points 2 pixels apart vertically used to yield two copies 3 rows apart; now they
#      are only 2 rows apart, so the bottom edge of the upper copy can collide with the
#      top edge of the lower copy.
# b) Complexity calibration:
#    - To make the "centered at singleton" rule visible, the template height is fixed to
#      3 so the center pixel is unambiguous and human-solvable.
#    - The singleton lattice still has 5-6 points on a 3-4 by 3-4 coarse grid, which
#      keeps repeated evidence without introducing too many free cases.
# c) Old assumptions that break:
#    - `off_r = dr * h` / `off_c = dc * w` depended on bbox-corner anchoring and sparse,
#      mostly non-overlapping copies. That is invalid here.
#    - Template-relative coordinates used to be measured from the bbox top-left. They now
#      must be measured from a guaranteed-filled center cell.
#    - Generator logic used to reject any stamp overlap; this would erase the D11 effect,
#      so the new generator instead requires visible overlap.
# d) Placement legality:
#    - The generator guarantees that every centered stamp fits inside 22x22, that the
#      template component and singleton cluster remain disjoint in the input with a bg gap,
#      and that the template center cell exists so the solver's anchor is well-defined.
# e) Visual observability:
#    - Samples always contain at least a few overlap-colored output pixels and at least a
#      few origin-only pixels, so both recoloring rules are visible from examples.
# f) Visual cleanliness:
#    - Overlap is limited to one compact cluster because all centers lie on the singleton
#      lattice; the input template stays separate, avoiding ambiguous merged components.
# g) Color inference:
#    - Colors remain exactly the original three roles: bg, template T, and overlap/origin
#      color O. No hidden ordering by color id is used.


def _template_anchor_and_offsets(template_bbox, templ_coords: np.ndarray):
    """Use the template bbox center cell as the direct-stamp anchor."""
    r0, c0, r1, c1 = map(int, template_bbox)
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    ar = int(r0 + h // 2)
    ac = int(c0 + w // 2)
    coords = {(int(r), int(c)) for r, c in templ_coords}
    assert (ar, ac) in coords
    rel = [(int(r) - ar, int(c) - ac) for r, c in templ_coords]
    return (ar, ac), rel


def _render_from_centers(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    templ_rel: list[tuple[int, int]],
    singletons: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return output plus total/origin coverage maps for centered stamping."""
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)
    for sr, sc, col in singletons:
        is_origin = int(col) == int(templ_col)
        for dr, dc in templ_rel:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W:
                total[rr, cc] += 1
                if is_origin:
                    center[rr, cc] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out, total, center


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp centered copies of a monochrome template according to a singleton lattice.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome component is the template in color T.
    - Separate singleton pixels form a lattice; exactly one singleton also has color T
      and marks the origin, while the remaining singletons are color O.

    Rule:
    - Treat the template's filled bbox-center cell as the stamp anchor.
    - For every singleton pixel, place one direct template copy so that this anchor lands
      exactly on that singleton position; do not scale the offset by template size.
    - Recolor the origin-centered copy to O, and recolor any multiply covered output
      pixels to O as well.

    Key insight:
    - The singleton lattice no longer tells you how many template-widths/heights to move;
      it gives the literal stamp centers. This makes overlap a first-class signal.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    _anchor, templ_rel = _template_anchor_and_offsets(template["bbox"], templ_coords)

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

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    out, _total, _center = _render_from_centers(H, W, bg, templ_col, overlap_col, templ_rel, singletons)
    return out


# --------------- generator (D11) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3]          # odd height keeps the stamp center pixel unambiguous
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
MIN_OVERLAP_CELLS = 2
MIN_ORIGIN_ONLY_CELLS = 2

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 80
SINGLETON_RETRIES = 80
TEMPLATE_PLACEMENT_RETRIES = 80
GENERATE_RETRIES = 400


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
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
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Polyomino of n cells that touches all four bbox edges."""
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
    """4-connected template that spans its bbox and contains the center anchor cell."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None

    center = (h // 2, w // 2)
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is None:
            continue
        if center not in set(cells):
            continue
        return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior)


def _delta_bounds(deltas: list[tuple[int, int]]):
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _bbox_separated_with_gap(a, b) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_singletons_and_measure(
    deltas: list[tuple[int, int]],
    templ_rel: list[tuple[int, int]],
    bg: int,
    T: int,
    O: int,
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample singleton positions that fit centered stamps and visibly exhibit D11 overlap."""
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)
    min_tr = min(dr for dr, _dc in templ_rel)
    max_tr = max(dr for dr, _dc in templ_rel)
    min_tc = min(dc for _dr, dc in templ_rel)
    max_tc = max(dc for _dr, dc in templ_rel)

    orr_lo = -min_dr * GUNIT - min_tr
    orr_hi = GRID_SIZE - 1 - max_dr * GUNIT - max_tr
    orc_lo = -min_dc * GUNIT - min_tc
    orc_hi = GRID_SIZE - 1 - max_dc * GUNIT - max_tc
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)
        singletons = []
        for dr, dc in deltas:
            rr = int(orr + dr * GUNIT)
            cc = int(orc + dc * GUNIT)
            col = int(T) if (dr == 0 and dc == 0) else int(O)
            singletons.append((rr, cc, col))

        _out, total, center = _render_from_centers(
            GRID_SIZE, GRID_SIZE, bg, T, O, templ_rel, singletons
        )
        overlap_cells = int(np.count_nonzero(total >= 2))
        origin_only = int(np.count_nonzero((total == 1) & (center > 0)))
        if overlap_cells < MIN_OVERLAP_CELLS or origin_only < MIN_ORIGIN_ONLY_CELLS:
            continue
        return singletons, _out, total, center
    return None


def _place_template(
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
    singleton_bbox,
) -> tuple[int, int] | None:
    """Place the input template away from the singleton cluster with a bg gap."""
    for _ in range(TEMPLATE_PLACEMENT_RETRIES):
        r0 = random.randint(0, GRID_SIZE - h)
        c0 = random.randint(0, GRID_SIZE - w)
        templ_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _bbox_separated_with_gap(templ_bbox, singleton_bbox):
            continue
        return r0, c0
    return None


def _build_input(
    bg: int,
    T: int,
    O: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    singletons: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for tr, tc in templ_cells:
        grid[int(r0 + tr), int(c0 + tc)] = int(T)
    for rr, cc, col in singletons:
        grid[int(rr), int(cc)] = int(col)
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: template + singleton lattice for centered direct stamping."""
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    templ_bbox = (0, 0, h - 1, w - 1)
    _anchor, templ_rel = _template_anchor_and_offsets(templ_bbox, np.array(templ_cells, dtype=int))

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]

    placed = _place_singletons_and_measure(deltas, templ_rel, bg, T, O)
    if placed is None:
        return None
    singletons, expected, _total, _center = placed

    rows = [r for r, _c, _col in singletons]
    cols = [c for _r, c, _col in singletons]
    singleton_bbox = (min(rows), min(cols), max(rows), max(cols))

    pos = _place_template(templ_cells, h, w, singleton_bbox)
    if pos is None:
        return None
    r0, c0 = pos

    grid = _build_input(bg, T, O, templ_cells, r0, c0, singletons)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
