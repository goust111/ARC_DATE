# Variant of a395ee82
# Dimension: D8 — 原点副本与普通副本的区分方式 (transform)
# Change: 原点副本整体重着色为O色(与普通副本的T色形成颜色对比)... → 原点副本使用模板的补集形状

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D8 variant
# a) Concrete behavior and trace
# - Input stays structurally identical to the source task: one large monochrome
#   template in color T, plus a uniformly spaced singleton lattice where the
#   unique T singleton marks the origin and the remaining singletons use color O.
# - Output still places one translated copy of the template per lattice point,
#   using offsets scaled by the template bbox size (h, w).
# - The only semantic change is the treatment of the origin copy:
#   ordinary copies draw the template itself in color T;
#   the origin copy draws the complement of the template inside the same bbox,
#   meaning template cells become background and bbox-empty cells become O.
# - Small trace: if the template bbox is 3x3 and the template misses only the
#   center cell, the original task would turn the origin copy into an 8-cell O
#   shape. This variant instead leaves those 8 cells as background and paints
#   only the center cell O. The bbox anchor and all other copies stay unchanged.
#
# b) Complexity calibration
# - The rule change is visually strong without adding extra branching:
#   there is still exactly one distinguished copy, identified by the unique
#   T singleton. The user only needs to infer "this one becomes a negative".
# - The original generator already keeps template fill below 100%, so the
#   complement is always non-empty. With bbox sizes 3x3 or 4x3 and the given
#   fill bounds, the complement size is 2-4 cells, which is enough to make the
#   negative copy visible without introducing a second independent object type.
#
# c) Implicit-assumption audit of the original solver
# - `templ_rel` remains valid: we still need the template's occupied cells.
# - The old `total` array assumed every copy contributes the positive template
#   shape. That is no longer true for the origin copy, so the origin must not
#   be counted as template coverage.
# - The old `center` mask plus
#   `out[(total == 1) & (center > 0)] = overlap_col`
#   encoded "origin copy = same shape, different color". That line is invalid
#   here because the origin no longer uses the template support at all.
# - Writing `out[total > 0] = templ_col` is still correct only for non-origin
#   copies. The origin bbox must be rendered separately as a full bbox-level
#   negative image.
# - Rendering the origin bbox last is safe because the generation strategy keeps
#   translated stamp bboxes disjoint. In the source task this was only a benign
#   detail; in the variant it matters because the origin writes background over
#   its former template cells.
#
# d) Placement legality
# - The source generator already ensures every translated bbox fits the grid.
# - Because translations are integer multiples of (h, w), distinct copies have
#   disjoint bboxes rather than merely disjoint foreground pixels.
# - The singleton cluster remains disjoint from the template with a bg gap, so
#   component detection and origin recovery are unaffected.
#
# e) Visual observability
# - Humans can still identify the special lattice point from the unique T
#   singleton in the input.
# - In the output there is exactly one anomalous copy, and its anomaly is not a
#   hidden color-id trick: it is a visible positive/negative contrast against
#   the ordinary T copies.
#
# f) Visual cleanliness
# - Only one copy uses the complement. All other copies stay identical to the
#   template, so the grid is not cluttered by many different transforms.
# - The variant avoids overlap-heavy composites; each bbox contains either a
#   normal template or one clean negative rendering.
#
# g) Color inference
# - The task still uses exactly three colors: bg, T, O.
# - O retains a structural role already hinted by the singleton lattice, then
#   reappears in the output as the negative-origin foreground color.
# - No logic depends on ordering or arithmetic properties of the color ids.
#
# New solver assumptions to verify against the generator
# - There is exactly one singleton whose color equals the template color.
# - There is at least one non-origin singleton, so O is recoverable.
# - The template bbox is not full, so its complement contains at least one cell.
# - Copy bboxes are disjoint, making the origin overwrite deterministic.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of one template, but render the origin copy as a negative image.

    Input structure:
    - One large 4-connected monochrome sprite in color T defines the template and its bbox.
    - Separate singleton pixels form a uniform-spacing lattice; exactly one singleton is also T.
    - The remaining singletons share color O, which is reused for the distinguished output copy.

    Rule:
    - Read each singleton as a lattice offset from the unique T singleton, then scale offsets by the template bbox size.
    - Every non-origin offset stamps the template itself in color T; overlaps between such stamps become O.
    - The origin offset does not stamp the template. Instead, inside the template bbox it paints the template's complement in O and leaves the template cells as background.

    Key insight:
    The special T-colored singleton still marks the reference copy, but now that copy is rendered as the bbox-level inverse of the template rather than as a recolored duplicate.
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
    templ_set = set(templ_rel)
    assert len(templ_set) < h * w

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

    non_origin_total = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        if int(col) == int(templ_col):
            continue
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                non_origin_total[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[non_origin_total > 0] = int(templ_col)
    out[non_origin_total >= 2] = int(overlap_col)

    for rr in range(h):
        for cc in range(w):
            R = int(r0 + rr)
            C = int(c0 + cc)
            out[R, C] = int(bg) if (rr, cc) in templ_set else int(overlap_col)

    return out


# --------------- generator (a1) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]         # template bbox height options
TEMPLATE_W = 3                      # template bbox width (fixed)
TEMPLATE_AREA_RANGE = (6, 10)       # absolute cell-count bounds
TEMPLATE_FILL_RANGE = (0.65, 0.85)  # fill-ratio bounds within bbox
SINGLETON_COUNT_RANGE = (5, 6)      # number of lattice singletons
LATTICE_EXTENT_RANGE = (3, 4)       # rows/cols of lattice polyomino
GUNIT = 2                           # singleton spacing in pixels

# Retry budgets
SPANNING_RETRIES = 40               # attempts per spanning polyomino
TEMPLATE_RETRIES = 60               # attempts to build a valid template
SINGLETON_RETRIES = 50              # attempts to place singleton cluster
GENERATE_RETRIES = 300              # outer retry loop


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
        if cells is None:
            continue
        if len(cells) >= total:
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
    overlap_col: int,
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
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = templ_col if (dr == 0 and dc == 0) else overlap_col

        try:
            main(grid)
        except Exception:
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

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

    occupied = set()
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
        overlap_col,
        occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
