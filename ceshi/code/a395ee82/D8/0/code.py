# Variant of a395ee82
# Dimension: D8 — 原点副本与普通副本的区分方式 (transform)
# Change: 原点副本整体重着色为O色(与普通副本的T色形成颜色对比)... → 原点副本仅边框着色

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D8 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: the origin-guided copy was recolored entirely to O.
#     new: only the border cells of that one copy become O; its interior cells stay T.
#     unchanged: all non-origin copies stay T, and any pixel covered by 2+ copies is still O.
#   concrete_pixel_behavior:
#     template_border: template cells whose 4-neighborhood touches background/outside-template.
#     template_interior: template cells whose 4 orthogonal neighbors all belong to the template.
#     input: unchanged from the seed task -- one monochrome template in T, plus a singleton lattice
#       with exactly one T singleton as the origin and the rest in O.
#     output: translated template copies at lattice offsets. The copy at delta (0, 0) gets an O
#       outline only; any interior cavity-free center remains T.
#   behavior_trace:
#     example_template: 3x3 almost-full sprite with center cell present.
#     original_flow:
#       - stamp all copies in T
#       - recolor every pixel of the origin copy to O
#       - result: origin copy loses all T pixels
#     variant_flow:
#       - stamp all copies in T
#       - compute the template's border mask once from the input sprite
#       - recolor only the origin copy's border support to O
#       - keep origin-copy interior support as T unless overlap overrides it
#     visible_difference:
#       - the origin copy now mixes O and T inside one connected sprite instead of becoming solid O
#       - therefore the generator must ensure the template has at least one true interior cell
#   complexity_calibration:
#     - No new hidden attributes are introduced: the only new distinction is border vs interior.
#     - The template remains monochrome in the input, so the solver must infer the border partition
#       from shape geometry, which is visible and repeatable across copies.
#     - Templates without interior cells would collapse the variant back to the original behavior,
#       so generation rejects them.
#   implicit_assumption_audit:
#     - `template = max(comps, key=area)` still relies on the template being the only non-singleton
#       component. The generator preserves that.
#     - `templ_col` still comes from the template color, and exactly one singleton still shares it.
#     - The original line `out[(total == 1) & (center > 0)] = overlap_col` depended on every
#       origin-covered pixel needing recolor. That assumption is no longer valid; we now need a
#       finer `origin_border` mask instead of recoloring the full origin support.
#     - Overlap recoloring remains safe as a final overwrite because O is still the intended color
#       whenever coverage count is 2 or more.
#   placement_legality:
#     - Stamps must fit fully in-grid and remain non-overlapping for clean samples.
#     - Singleton cluster stays disjoint from the template bbox with at least one background cell gap.
#     - Because the generator constructs the template at the origin copy position, the output copy
#       that shows the border/interior contrast is guaranteed to be present.
#   visual_observability:
#     - The changed rule is human-visible only if the origin copy has both border and interior cells.
#     - Requiring at least one interior cell makes the output differ from the seed task on every
#       generated sample, without adding unrelated dimensions like rotations or extra colors.
#   visual_cleanliness:
#     - The seed generator already forbids stamp overlaps in produced samples, so the new O outline
#       appears cleanly around a T interior rather than being muddied by stacked copies.
#   color_reasoning:
#     - Still only 3 colors: bg, T, O.
#     - O keeps the same structural role as the lattice non-origin singleton color and the overlap/
#       origin-highlight color, so there is no arbitrary color mapping to infer.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _partition_template_cells(
    cells: list[tuple[int, int]] | np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split template-relative cells into border cells and interior cells."""
    cell_set = {(int(r), int(c)) for r, c in cells}
    border: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for r, c in sorted(cell_set):
        if any((r + dr, c + dc) not in cell_set for dr, dc in ORTHO_DIRS):
            border.append((r, c))
        else:
            interior.append((r, c))
    return border, interior


def _render_output(
    H: int,
    W: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    templ_rel: list[tuple[int, int]],
    border_rel: list[tuple[int, int]],
    deltas: list[tuple[int, int, bool]],
) -> np.ndarray:
    """Render the output grid from the parsed or constructed layout."""
    total = np.zeros((H, W), dtype=int)
    origin_border = np.zeros((H, W), dtype=bool)

    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
        if is_origin:
            for rr, cc in border_rel:
                R = int(r0 + off_r + rr)
                C = int(c0 + off_c + cc)
                if 0 <= R < H and 0 <= C < W:
                    origin_border[R, C] = True

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[origin_border] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by a singleton lattice.

    Input structure:
    - bg is the dominant background color.
    - One large 4-connected monochrome template uses color T.
    - Separate singleton pixels form a uniform-spacing lattice: exactly one singleton is T
      (the lattice origin), and the remaining singletons share color O.

    Rule:
    - Read each singleton as a lattice offset from the unique T-colored origin.
    - Convert those coarse offsets into template-sized translations and stamp a copy of the
      template at every translated location.
    - All copies start as T, but the origin copy is special: only its border pixels are
      recolored to O, while its interior pixels remain T.
    - Any pixel covered by two or more copies is also recolored to O.

    Key insight: The T singleton still marks which translated copy is the reference one, but in
    this variant you must further decompose that copy by shape geometry: border cells flip to O,
    while interior cells stay T.
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
    border_rel, _interior_rel = _partition_template_cells(templ_rel)

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

    deltas = [(int(r - orr), int(c - orc), int(col) == int(templ_col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    coarse_deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in deltas]

    return _render_output(
        H=H,
        W=W,
        bg=bg,
        templ_col=templ_col,
        overlap_col=overlap_col,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        templ_rel=templ_rel,
        border_rel=border_rel,
        deltas=coarse_deltas,
    )


# --------------- generator (D8) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]        # template bbox height options
TEMPLATE_W = 3                     # template bbox width (fixed)
TEMPLATE_AREA_RANGE = (6, 10)      # absolute cell-count bounds
TEMPLATE_FILL_RANGE = (0.65, 0.85)  # fill-ratio bounds within bbox
MIN_INTERIOR_CELLS = 1             # ensure the border-only recolor is visible
SINGLETON_COUNT_RANGE = (5, 6)     # number of lattice singletons
LATTICE_EXTENT_RANGE = (3, 4)      # rows/cols of lattice polyomino
GUNIT = 2                          # singleton spacing in pixels

# Retry budgets
SPANNING_RETRIES = 40              # attempts per spanning polyomino
TEMPLATE_RETRIES = 60              # attempts to build a valid template
SINGLETON_RETRIES = 50             # attempts to place singleton cluster
GENERATE_RETRIES = 300             # outer retry loop


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of *n* cells in a rows x cols grid."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in ORTHO_DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in ORTHO_DIRS:
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


def _gen_template(h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """4-connected template in h x w with at least one interior pixel."""
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
        border_cells, interior_cells = _partition_template_cells(cells)
        if len(interior_cells) < MIN_INTERIOR_CELLS:
            continue
        return cells, border_cells
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


def _delta_bounds(deltas: list[tuple[int, int]]):
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
    border_cells: list[tuple[int, int]],
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

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1
    rendered_deltas = [(dr, dc, dr == 0 and dc == 0) for dr, dc in deltas]

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

        expected = _render_output(
            H=GRID_SIZE,
            W=GRID_SIZE,
            bg=bg,
            templ_col=T,
            overlap_col=O,
            r0=r0,
            c0=c0,
            h=h,
            w=w,
            templ_rel=templ_cells,
            border_rel=border_cells,
            deltas=rendered_deltas,
        )
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: pick colors, build template + lattice, place stamps + singletons."""
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ = _gen_template(h, w)
    if templ is None:
        return None
    templ_cells, border_cells = templ

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
        deltas=deltas,
        min_dr=min_dr,
        max_dr=max_dr,
        min_dc=min_dc,
        max_dc=max_dc,
        templ_cells=templ_cells,
        border_cells=border_cells,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        bg=bg,
        T=T,
        O=O,
        occupied=occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
