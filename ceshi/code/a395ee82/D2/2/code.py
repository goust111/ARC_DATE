# Variant of a395ee82
# Dimension: D2 — 源对象结构 (src)
# Change: 单例像素晶格 — 5-6个单像素按均匀间距(gunit=2)排列成4-连通多联骨牌，其中1个T色为原点，4-5个O色为普通点 → 单像素但间距不均匀

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   grid: 22x22, 3 colors (bg / T / O)
#   template:
#     same as the original task: a single monochrome 4-connected T-shape
#     with bbox 3-4 x 3 and area 6-10
#   source_singletons:
#     still 5-6 isolated pixels, with exactly one T-colored origin and the
#     rest colored O
#     still derived from a 4-connected polyomino on a 3-4 by 3-4 coarse grid
#     but the coarse grid is now warped:
#       consecutive coarse rows/cols are separated by either 2 or 3 pixels
#       gap 2 encodes 1 stamp-step
#       gap 3 encodes 2 stamp-steps
#   output:
#     stamp the template at every decoded offset
#     the origin copy is recolored to O
#     any overlap would also become O, although generation keeps copies disjoint
#
# behavior_trace:
#   small_example:
#     coarse rows at source pixels [4, 6, 9] have gaps [2, 3]
#     decoded row steps are [0, 1, 3]
#     if the origin is the middle row (pixel 6), then a point at pixel 4 means
#     -1 template-height, while a point at pixel 9 means +2 template-heights
#   original_logic_here:
#     the old solver would look only for a single global gunit and would treat
#     the same raw row gaps as either [1, 2] after division by 2 or as [2, 3]
#     when the global gcd becomes 1; neither captures the intended
#     "short gap vs long gap" category mapping reliably
#   variant_logic_here:
#     each adjacent gap is decoded locally by category: 2->1 step, 3->2 steps
#     then cumulative decoded row/col positions determine stamp placement
#
# complexity_calibration:
#   the rule uses only two visible source gap categories (2 and 3), so the
#   change is easy to see but not ambiguous
#   the same categories repeat across row and column bands, giving human solvers
#   multiple examples per grid instead of a one-off hidden parameter
#
# implicit_assumption_audit:
#   original_main_lines_using_gcd:
#     they assume every non-zero delta is an integer multiple of one global
#     lattice unit and that dividing by gcd recovers the true coarse offsets
#   why_that_breaks:
#     this variant intentionally makes coarse offsets depend on the local gap
#     category, so a single gcd is no longer the right decoder
#   new_main_assumptions:
#     all singleton rows and all singleton cols come from consecutive warped
#     coarse levels
#     every adjacent row/col gap is either 2 or 3 pixels
#     the T-colored singleton is unique and marks the origin
#   generator_guarantees:
#     the polyomino spans its coarse bbox, so every intermediate coarse row/col
#     is occupied and the sorted unique source rows/cols expose the full warped
#     axis structure
#     both gap categories appear in every sample, preventing regression to the
#     uniform-spacing original task
#
# placement_legality_and_visuals:
#   output stamps are still placed on a clean h-by-w lattice in decoded space,
#   so copies remain non-overlapping and visually readable
#   the singleton cluster is kept disjoint from the template bbox with at least
#   one background cell of separation
#   source markers remain isolated single pixels, so the irregular spacing is
#   easy to see directly in the input


GAP_TO_STEP = {2: 1, 3: 2}
SOURCE_GAP_CHOICES = tuple(sorted(GAP_TO_STEP))

# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GAP_PATTERN_RETRIES = 80
GENERATE_RETRIES = 400


def _axis_step_map(values: list[int]) -> dict[int, int]:
    """Map visible singleton coordinates to decoded coarse steps."""
    vals = sorted({int(v) for v in values})
    assert len(vals) >= 2

    out = {vals[0]: 0}
    acc = 0
    for prev, cur in zip(vals, vals[1:]):
        gap = int(cur - prev)
        assert gap in GAP_TO_STEP
        acc += int(GAP_TO_STEP[gap])
        out[cur] = acc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template guided by an irregular singleton scaffold.

    Input structure:
    - bg = background. One large monochrome T-shape is the template.
    - Elsewhere are 5-6 isolated singleton pixels: exactly one has color T and marks the origin; the others share color O.
    - Singleton rows/cols are not uniformly spaced: adjacent gaps are either short or long, and those categories encode stamp-step counts.

    Rule:
    - Read the sorted singleton rows and cols as warped axes.
    - Decode each adjacent source gap by category (2 pixels -> 1 template step, 3 pixels -> 2 template steps).
    - Convert every singleton's decoded offset from the origin into template-height/template-width translations, then stamp the template there.
    - The origin copy is recolored to O, and any overlapped pixels are also recolored to O.

    Key insight: The singleton scaffold is no longer normalized by one global lattice unit; instead, each visible short/long gap contributes its own local number of stamp steps.
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
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    row_to_step = _axis_step_map([r for r, _c, _col in singletons])
    col_to_step = _axis_step_map([c for _r, c, _col in singletons])
    origin_sr = int(row_to_step[int(orr)])
    origin_sc = int(col_to_step[int(orc)])

    deltas = [
        (
            int(row_to_step[int(r)] - origin_sr),
            int(col_to_step[int(c)] - origin_sc),
            int(col),
        )
        for (r, c, col) in singletons
    ]

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
    """Polyomino of n cells that spans exactly rows x cols."""
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


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int], int, int] | None:
    """Lattice polyomino of n cells with a strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior), ext_r, ext_c


def _gen_gap_layout(ext_r: int, ext_c: int) -> tuple[list[int], list[int]] | None:
    """Generate visible source gaps with both categories present and bounded decoded span."""
    for _ in range(GAP_PATTERN_RETRIES):
        row_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_r - 1)]
        col_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_c - 1)]
        all_gaps = set(row_gaps + col_gaps)
        if all_gaps != set(SOURCE_GAP_CHOICES):
            continue
        if sum(GAP_TO_STEP[g] for g in row_gaps) > 4:
            continue
        if sum(GAP_TO_STEP[g] for g in col_gaps) > 4:
            continue
        return row_gaps, col_gaps
    return None


def _levels_from_gaps(gaps: list[int]) -> tuple[list[int], list[int]]:
    """Return visible pixel coordinates and decoded coarse-step coordinates."""
    src = [0]
    steps = [0]
    for gap in gaps:
        src.append(int(src[-1] + gap))
        steps.append(int(steps[-1] + GAP_TO_STEP[int(gap)]))
    return src, steps


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _place_singletons(
    raw_deltas: list[tuple[int, int]],
    raw_bounds: tuple[int, int, int, int],
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
    """Place the irregular singleton scaffold away from the template."""
    min_dr, max_dr, min_dc, max_dc = raw_bounds
    orr_lo, orr_hi = -min_dr, GRID_SIZE - 1 - max_dr
    orc_lo, orc_hi = -min_dc, GRID_SIZE - 1 - max_dc
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr
        sr1 = orr + max_dr
        sc0 = orc + min_dc
        sc1 = orc + max_dc
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        src_points = [(int(orr + dr), int(orc + dc)) for dr, dc in raw_deltas]
        if any(pt in occupied for pt in src_points):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for rr, cc in src_points:
            grid[int(rr), int(cc)] = int(O)
        grid[int(orr), int(orc)] = int(T)

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the irregular-gap variant."""
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
    lattice_cells, origin, ext_r, ext_c = lat

    gaps = _gen_gap_layout(ext_r, ext_c)
    if gaps is None:
        return None
    row_gaps, col_gaps = gaps
    row_src, row_steps = _levels_from_gaps(row_gaps)
    col_src, col_steps = _levels_from_gaps(col_gaps)

    raw_deltas: list[tuple[int, int]] = []
    step_deltas: list[tuple[int, int]] = []
    for lr, lc in lattice_cells:
        raw_deltas.append(
            (
                int(row_src[lr] - row_src[origin[0]]),
                int(col_src[lc] - col_src[origin[1]]),
            )
        )
        step_deltas.append(
            (
                int(row_steps[lr] - row_steps[origin[0]]),
                int(col_steps[lc] - col_steps[origin[1]]),
            )
        )

    step_bounds = _delta_bounds(step_deltas)
    min_sdr, max_sdr, min_sdc, max_sdc = step_bounds
    r0_lo, r0_hi = -min_sdr * h, GRID_SIZE - max_sdr * h - h
    c0_lo, c0_hi = -min_sdc * w, GRID_SIZE - max_sdc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied: set[tuple[int, int]] = set()
    for dr, dc in step_deltas:
        for tr, tc in templ_cells:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)

    return _place_singletons(
        raw_deltas=raw_deltas,
        raw_bounds=_delta_bounds(raw_deltas),
        templ_cells=templ_cells,
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
