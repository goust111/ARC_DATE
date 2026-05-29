# Variant of a395ee82
# Dimension: D3 — 目标模板结构 (dst)
# Change: 单连通单色精灵(色T)，外接矩形3-4行x3列，面积6-10，填充率0.65-0.85，触及外接矩形四边... → 含孔洞的模板

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg


# Design analysis
# ---------------
# spec:
#   preserved_backbone:
#     grid: 22x22, single bg color
#     colors: 3 distinct colors -> bg / T(template) / O(overlap-origin)
#     src: 5-6 singleton lattice points at spacing 2; exactly one singleton has color T
#     transform: convert singleton deltas into bbox-scaled offsets and stamp the template there
#   d3_change:
#     dst template is no longer a solid polyomino-only sprite.
#     The bbox is still 3-4 rows by 3 cols and still touches all four bbox sides,
#     but inside the bbox there must be 1-2 enclosed bg pixels.
#     Stamp semantics now depend on the exact filled mask:
#       - filled template pixels stamp as T
#       - the origin copy's filled pixels stamp as O
#       - hole pixels stay bg in every copy
#   concrete_pixel_behavior:
#     Example 4x3 template:
#       ###        filled pixels are stamped
#       #.#   ->   the middle '.' is an enclosed hole and remains bg
#       ###        after translation too
#       #..
#     The original task could be imagined as "copy every foreground coordinate".
#     The variant keeps that global motion logic, but the copied foreground mask now
#     contains interior voids, so rectangle-filling or bbox-based stamping becomes wrong.
#   complexity_calibration:
#     We keep colors, lattice logic, spacing, and placement style unchanged so the
#     innovation remains isolated to the template structure.
#     Template generation is switched to exhaustive legal-mask sampling because the
#     bbox is tiny; this guarantees actual holes and avoids accidental regression to
#     ordinary solid sprites.
#   hidden_assumptions_review:
#     - Largest component as template still holds: holes do not split the T component,
#       and singletons stay area-1.
#     - Bbox height/width scaling still holds because translations are still anchored
#       at the template bbox origin.
#     - The old "iterate template coords" pattern is safe only if we truly use the
#       filled mask, not the full bbox. This file now computes the bbox mask explicitly
#       and separately identifies enclosed bg, mirroring the hole-aware reasoning from
#       the reference task.
#     - Generator must guarantee enclosed bg rather than merely missing edge cells;
#       otherwise the change would visually collapse back into the old task family.
#   placement_legality:
#     - all stamped filled pixels remain in-bounds
#     - singleton cluster bbox stays disjoint from template bbox with >=1 bg gap
#     - stamps do not overlap on filled pixels
#     - the solver's assumptions are enforced by generator-side mask validation
#   observability:
#     - samples always contain at least 1 enclosed bg pixel in the template
#     - many legal masks also include edge-connected bg notches, so a human must
#       distinguish "inside hole" from ordinary outside background within the bbox
#     - hole-pixel count is controlled to 1 or 2, keeping the branch space small
#       enough to infer from a few examples


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bg cells inside the bbox that are not 4-connected to the bbox edge."""
    h, w = filled_mask.shape
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if touches_edge:
            continue
        holes[coords[:, 0], coords[:, 1]] = True
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a holey template using a singleton lattice as the guide.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome template of color T appears once; its bbox is 3-4 by 3.
    - The template may contain 1-2 enclosed bg pixels (holes).
    - Elsewhere, isolated lattice singletons appear: exactly one is color T (the origin),
      the others share color O.

    Rule:
    - Read singleton offsets relative to the unique T-colored origin and normalize them by
      the lattice spacing.
    - Convert each lattice offset into a translation scaled by the template bbox height and width.
    - Stamp only the template's filled pixels at every translated position, leaving hole pixels bg.
    - The origin copy's filled pixels become O instead of T, and any filled pixel covered by
      two or more copies also becomes O.

    Key insight: The object to replicate is the exact filled mask inside the template bbox, not
    the full rectangle. Enclosed bg cells are part of the template's structure and must stay empty.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_mask = (g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col))
    hole_mask = _enclosed_bg_mask(templ_mask)
    hole_pixels = int(hole_mask.sum())
    assert 1 <= hole_pixels <= 2

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

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


# --------------- generator ---------------

# Tunable configuration
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

# Retry budgets
SPANNING_RETRIES = 40
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


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


def _template_candidates(h: int, w: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate all legal holey template masks, grouped by enclosed-hole pixel count."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))

    grouped: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}
    for bits in range(1, 1 << total):
        cells = [(idx // w, idx % w) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < lo or area > hi:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        comps = components_mask(mask, connectivity=4)
        if len(comps) != 1:
            continue
        if not mask[0, :].any() or not mask[h - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, w - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels not in grouped:
            continue

        grouped[hole_pixels].append(cells)

    return grouped


TEMPLATE_LIBRARY = {
    h: _template_candidates(h, TEMPLATE_W)
    for h in TEMPLATE_H_CHOICES
}


def _gen_template() -> tuple[int, int, list[tuple[int, int]], int] | None:
    """Sample a legal holey template, balancing 1-hole-pixel and 2-hole-pixel cases."""
    hole_choices = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_choices)
    for desired_holes in hole_choices:
        valid_heights = [h for h in TEMPLATE_H_CHOICES if TEMPLATE_LIBRARY[h][desired_holes]]
        if not valid_heights:
            continue
        h = random.choice(valid_heights)
        cells = random.choice(TEMPLATE_LIBRARY[h][desired_holes])
        return h, TEMPLATE_W, cells, desired_holes

    valid_heights = [h for h in TEMPLATE_H_CHOICES if any(TEMPLATE_LIBRARY[h].values())]
    if not valid_heights:
        return None
    h = random.choice(valid_heights)
    all_cells = TEMPLATE_LIBRARY[h][1] + TEMPLATE_LIBRARY[h][2]
    cells = random.choice(all_cells)
    holes = int(_enclosed_bg_mask(_cells_to_mask(cells, h, TEMPLATE_W)).sum())
    return h, TEMPLATE_W, cells, holes


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


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
    """Try to place the singleton lattice and build the final input grid."""
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
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: choose colors, build a holey template, then place the lattice."""
    bg, T, O = random.sample(range(10), 3)

    templ = _gen_template()
    if templ is None:
        return None
    h, w, templ_cells, _hole_pixels = templ

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
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
