# Variant of a395ee82
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: color_match — 单例像素中唯一与模板同色(T)的像素被识别为原点，其余O色像素为普通晶格点... → 位置规则确定原点

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D1 variant
# a) Concrete behavior:
#    Input still contains one large monochrome template and a separate lattice of singleton pixels.
#    The change is that every singleton now has the same color O. The "origin" singleton is no longer
#    color-marked; instead it is the unique singleton closest to the visible template pixels.
#    Small trace:
#    - Original: a lattice point colored T means delta (0,0), so the template already shown in the input
#      is recolored to O in the output.
#    - Variant: all lattice points are O, so the solver must inspect geometry. The singleton nearest to
#      the template is treated as delta (0,0); that still makes the input template position become the
#      special O-colored copy in the output.
#    - Difference: the key match step moves from "find singleton of color T" to "find unique nearest
#      singleton to the template shape".
# b) Complexity calibration:
#    Only one freedom changes: origin detection. Object types, colors, lattice spacing, stamp geometry,
#    and overlap recoloring stay the same. To keep the visual cue strong, the generator enforces a unique
#    nearest singleton rather than relying on hidden tie-breaks.
# c) Implicit assumptions audit:
#    - The original `origin = [pt for pt in singletons if col == templ_col]` no longer holds because no
#      singleton has color T.
#    - The original `is_origin = (col == templ_col)` also breaks for the same reason.
#    - Reusing the singleton color as overlap color is still valid, but now it must be read as the common
#      singleton color, not "the non-template singleton color(s) by majority vote".
#    - GCD-based lattice-unit recovery still works because the spacing pattern is unchanged.
# d) Placement legality:
#    The generator still ensures the template and singleton cluster are disjoint with at least one
#    background cell gap, all future stamps fit in-bounds, and no future stamp pixels overlap each other.
#    Additionally, singleton placement is constrained so the intended origin is uniquely nearest.
# e) Visual observability:
#    The nearest-singleton rule is visible from raw geometry. The generator places the singleton cluster on
#    one side of the template and aligns exactly one extremal singleton to the template edge, making the
#    origin visibly the closest point without using color ids or hidden ordering.
# f) Visual cleanliness:
#    Singletons remain isolated, uniformly spaced points; the template remains a single compact component.
#    No extra markers or mixed-color lattice decorations are introduced.
# g) Color inference:
#    Colors remain limited to bg / T / O. O has two roles visible in examples: all input singletons and
#    the special origin stamp / overlap pixels in the output. That color reuse is structural and local,
#    so it remains human-inferable.


def _manhattan_to_template(r: int, c: int, templ_coords: list[tuple[int, int]]) -> int:
    """Minimum Manhattan distance from a point to any filled template pixel."""
    return min(abs(int(r) - tr) + abs(int(c) - tc) for tr, tc in templ_coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template using a position-defined lattice origin.

    Input structure:
    - bg = background. A large monochrome shape (template, color T).
    - Singleton pixels elsewhere all share one color O; none is specially recolored.
    - The singleton points still lie on a uniform-spacing lattice, but the origin is the singleton closest to the template.

    Rule:
    - Find the unique singleton with minimum Manhattan distance to the template pixels and treat it as lattice origin.
    - Convert every singleton's offset from that origin into coarse lattice units.
    - Stamp the template at those offsets, scaled by template height/width; the origin's copy is recolored to O, and multi-covered pixels are also recolored to O.

    Key insight: Because singleton colors no longer distinguish the origin, the solver must infer the origin from geometry: the nearest singleton corresponds to the already-present template copy.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords_arr = template["coords"]
    templ_col = int(g[int(templ_coords_arr[0, 0]), int(templ_coords_arr[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_rel = [(int(r) - r0, int(c) - c0) for r, c in templ_coords_arr]
    templ_abs = [(int(r), int(c)) for r, c in templ_coords_arr]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(g[rr, cc])))

    assert len(singletons) >= 2
    singleton_cols = {int(col) for _r, _c, col in singletons}
    assert len(singleton_cols) == 1
    overlap_col = int(next(iter(singleton_cols)))
    assert overlap_col != templ_col

    dists = [_manhattan_to_template(r, c, templ_abs) for r, c, _col in singletons]
    min_dist = min(dists)
    origin_candidates = [pt for pt, dist in zip(singletons, dists) if dist == min_dist]
    assert len(origin_candidates) == 1
    orr, orc, _ = origin_candidates[0]

    raw_deltas = [(int(r - orr), int(c - orc), int(r == orr and c == orc)) for r, c, _col in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _is_origin in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in raw_deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, is_origin in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
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


# --------------- generator (D1 variant) ---------------

# ── Tunable configuration ──────────────────────────────────────────
GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
SINGLETON_GAP = 2
SIDE_CHOICES = ["left", "right", "above", "below"]

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
LATTICE_RETRIES = 180
TEMPLATE_POSITION_RETRIES = 60
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 600


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
    return sorted(cells) if len(cells) == n else None


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


def _select_side_origin(cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    """Choose a unique extremal lattice cell that faces the template from the given side."""
    if side == "left":
        target = max(c for _r, c in cells)
        frontier = [(r, c) for r, c in cells if c == target]
    elif side == "right":
        target = min(c for _r, c in cells)
        frontier = [(r, c) for r, c in cells if c == target]
    elif side == "above":
        target = max(r for r, _c in cells)
        frontier = [(r, c) for r, c in cells if r == target]
    else:
        target = min(r for r, _c in cells)
        frontier = [(r, c) for r, c in cells if r == target]
    return frontier[0] if len(frontier) == 1 else None


def _gen_lattice(n: int, side: str) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino with a unique extremal cell that can serve as the nearest-point origin."""
    for _ in range(LATTICE_RETRIES):
        ext_r = random.randint(*LATTICE_EXTENT_RANGE)
        ext_c = random.randint(*LATTICE_EXTENT_RANGE)
        cells = _spanning_polyomino(n, ext_r, ext_c)
        if cells is None:
            continue
        origin = _select_side_origin(cells, side)
        if origin is not None:
            return cells, origin
    return None


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_dr, max_dr, min_dc, max_dc) for a list of (dr, dc) offsets."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _template_edge_slots(
    templ_cells: list[tuple[int, int]], h: int, w: int, side: str
) -> list[int]:
    """Rows or columns on the template edge that can face the singleton origin."""
    if side == "left":
        return [tr for tr, tc in templ_cells if tc == 0]
    if side == "right":
        return [tr for tr, tc in templ_cells if tc == w - 1]
    if side == "above":
        return [tc for tr, tc in templ_cells if tr == 0]
    return [tc for tr, tc in templ_cells if tr == h - 1]


def _cluster_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in points]
    cs = [c for _r, c in points]
    return min(rs), max(rs), min(cs), max(cs)


def _place_singletons(
    side: str,
    lattice_cells: list[tuple[int, int]],
    origin: tuple[int, int],
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
    """Place a same-colored singleton cluster so the chosen origin is uniquely nearest to the template."""
    edge_slots = _template_edge_slots(templ_cells, h, w, side)
    template_abs = [(r0 + tr, c0 + tc) for tr, tc in templ_cells]
    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        slot = random.choice(edge_slots)
        if side == "left":
            origin_abs = (r0 + slot, c0 - SINGLETON_GAP)
        elif side == "right":
            origin_abs = (r0 + slot, c0 + w - 1 + SINGLETON_GAP)
        elif side == "above":
            origin_abs = (r0 - SINGLETON_GAP, c0 + slot)
        else:
            origin_abs = (r0 + h - 1 + SINGLETON_GAP, c0 + slot)

        base_r = int(origin_abs[0] - origin[0] * GUNIT)
        base_c = int(origin_abs[1] - origin[1] * GUNIT)
        abs_points = [(base_r + lr * GUNIT, base_c + lc * GUNIT) for lr, lc in lattice_cells]
        if any(not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE) for rr, cc in abs_points):
            continue

        sr0, sr1, sc0, sc1 = _cluster_bbox(abs_points)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue
        if any((rr, cc) in occupied for rr, cc in abs_points):
            continue

        dists = [_manhattan_to_template(rr, cc, template_abs) for rr, cc in abs_points]
        min_dist = min(dists)
        if dists.count(min_dist) != 1:
            continue
        origin_idx = lattice_cells.index(origin)
        if dists[origin_idx] != min_dist:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for rr, cc in abs_points:
            grid[rr, cc] = O

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build a template and a same-colored singleton lattice with nearest-point origin."""
    bg, T, O = random.sample(range(10), 3)
    side = random.choice(SIDE_CHOICES)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing, side)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    for _ in range(TEMPLATE_POSITION_RETRIES):
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)

        occupied: set[tuple[int, int]] = set()
        valid = True
        for dr, dc in deltas:
            for tr, tc in templ_cells:
                pt = (r0 + dr * h + tr, c0 + dc * w + tc)
                if pt in occupied:
                    valid = False
                    break
                occupied.add(pt)
            if not valid:
                break
        if not valid:
            continue

        grid = _place_singletons(
            side, lattice_cells, origin, templ_cells, r0, c0, h, w, bg, T, O, occupied
        )
        if grid is not None:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
