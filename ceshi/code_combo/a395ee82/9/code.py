# Combo variant of a395ee82 (index 9)
# combo_spec: D1/1+D8/1
# Name: 位置原点补集形态

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
# - Preserved backbone from the base task:
#   one large monochrome template in color T and a detached lattice of singleton pixels.
# - D1/1 removes the color-marked origin:
#   every singleton now uses the same color O, so the solver can no longer recover
#   delta (0,0) by matching singleton color to T.
# - D8/1 changes the special copy itself:
#   the origin copy is not a recolored positive stamp. It is the complement of the
#   template inside the template bbox, using O on former bg cells and bg on former
#   template cells.
# - Small trace:
#   suppose the template bbox is 3x3 and its filled mask misses exactly two cells.
#   Original task:
#     one T-colored singleton marks the origin, and the output paints the 7 filled
#     template cells at the origin in O.
#   This combo:
#     all singletons are O; the origin is instead the unique singleton with minimum
#     Manhattan distance to the visible template. At the origin bbox, the output
#     paints only the 2 complement cells in O and turns the former 7 template cells
#     into bg. Other lattice points still receive normal T-colored positive stamps.
#   The two edits therefore interact: geometry identifies which copy is special, and
#   that special copy is a negative image rather than a recolored duplicate.
#
# b) Complexity calibration:
# - The position rule is restricted to a single visibly supported criterion:
#   the unique nearest singleton to the template. No hidden tie-break by scan order
#   or color id is used.
# - The negative-image rule is visually strong because the template fill ratio stays
#   below 100%, so every origin bbox contains a non-empty complement.
# - The generator places the singleton cluster on one side of the template and picks
#   an extremal lattice cell as origin, ensuring the nearest-point cue is repeated
#   across samples without introducing extra marker types.
#
# c) Implicit-assumption audit against the original main():
# - Original `origin = [pt for pt in singletons if col == templ_col]` is invalid,
#   because no singleton has color T in this combo.
# - Original `is_origin = (col == templ_col)` inside the stamping loop is likewise
#   invalid and must be replaced by an explicit geometric-origin flag.
# - Original `out[(total == 1) & (center > 0)] = overlap_col` encoded "same shape,
#   different color" for the origin copy. That assumption breaks because the origin
#   now uses different support: the template complement, not the template cells.
# - Original majority-vote recovery of O from non-T singletons is unnecessary here;
#   instead, main() must assume all singleton colors are identical and distinct from T.
# - Original gcd recovery of lattice spacing still holds because only origin
#   identification changed, not the uniform-spacing lattice itself.
#
# d) Placement legality:
# - All translated copy anchors remain integer multiples of (h, w), so every copy
#   stays inside its own h x w bbox and all bboxes are pairwise disjoint.
# - The generator chooses the template anchor from lattice-delta bounds, so every
#   future copy fits inside the 22x22 canvas.
# - The singleton cluster is kept disjoint from the template bbox with at least one
#   background-cell gap, and singleton points are forbidden from landing on any
#   future stamped foreground pixel.
#
# e) Visual observability:
# - Because all singletons share color O, the solver must use visible geometry to
#   infer the origin; the generator enforces a unique nearest singleton each time.
# - The output contains exactly one anomalous bbox: it is a clear positive/negative
#   contrast against the ordinary T-colored copies.
# - The effect is visible in 2-3 examples because both the nearest-point cue and the
#   complement cue are local, repeated, and independent of absolute color ids.
#
# f) Visual cleanliness:
# - Input still has only two object families: one compact template and isolated
#   singletons. No extra guide marks are introduced.
# - Output copy bboxes are disjoint, so the negative origin copy never mixes with a
#   normal copy in the same bbox.
#
# g) Rule inferability / non-degeneration:
# - Colors remain exactly bg / T / O.
# - The generator prevents regression to the source task by making every singleton O.
# - It also prevents regression to the D1-only variant by requiring a non-full
#   template, so the complement-origin output always differs from a recolored origin.
#
# h) New main() assumptions and generator guarantees:
# - main() assumes there are at least two singleton pixels, all with one shared color
#   O distinct from T; generator enforces that.
# - main() assumes the nearest-template singleton is unique; generator checks the
#   intended origin is the unique distance minimizer before accepting a sample.
# - main() assumes the template bbox is not full so that the complement is non-empty;
#   generator only samples templates with area < h*w.


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

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
LATTICE_RETRIES = 180
TEMPLATE_POSITION_RETRIES = 60
SINGLETON_RETRIES = 80
GENERATE_RETRIES = 600


def _manhattan_to_template(r: int, c: int, templ_coords: list[tuple[int, int]]) -> int:
    """Minimum Manhattan distance from a point to any filled template pixel."""
    return min(abs(int(r) - tr) + abs(int(c) - tc) for tr, tc in templ_coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of one template, but infer the origin geometrically and render it as a negative image.

    Input structure:
    - One large 4-connected monochrome sprite in color T defines the template and its bbox.
    - Separate singleton pixels form a uniform-spacing lattice, and every singleton uses the same color O.
    - No singleton is color-marked as the origin; instead, the origin is the unique singleton closest to the template.

    Rule:
    - Find the singleton with minimum Manhattan distance to the visible template pixels and treat it as lattice origin.
    - Convert all singleton offsets from that origin into coarse lattice units, then scale them by the template bbox size.
    - Every non-origin offset stamps the positive template in color T, with multi-covered pixels recolored to O.
    - At the origin offset, draw the template's complement inside the bbox: former template cells become bg and former bbox-empty cells become O.

    Key insight:
    The special copy is identified by position rather than color, and that same geometrically inferred copy is rendered as the template's negative rather than as a recolored positive clone.
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

    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords_arr]
    templ_set = set(templ_rel)
    templ_abs = [(int(r), int(c)) for r, c in templ_coords_arr]
    assert len(templ_set) < h * w

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

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

    raw_deltas = [
        (int(r - orr), int(c - orc), bool(int(r) == int(orr) and int(c) == int(orc)))
        for r, c, _col in singletons
    ]
    gcd_vals = [abs(int(x)) for dr, dc, _is_origin in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in raw_deltas]

    non_origin_total = np.zeros((H, W), dtype=int)
    origin_delta = None
    for dr, dc, is_origin in deltas:
        if is_origin:
            origin_delta = (int(dr), int(dc))
            continue
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                non_origin_total[R, C] += 1

    assert origin_delta == (0, 0)

    out = np.full((H, W), int(bg), dtype=int)
    out[non_origin_total > 0] = int(templ_col)
    out[non_origin_total >= 2] = int(overlap_col)

    for rr in range(h):
        for cc in range(w):
            R = int(r0 + rr)
            C = int(c0 + cc)
            out[R, C] = int(bg) if (rr, cc) in templ_set else int(overlap_col)

    return out


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols box."""
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
    """Polyomino of n cells that touches all four edges of its rows x cols bbox."""
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
    """Generate a 4-connected spanning template whose bbox complement is non-empty."""
    total = int(h * w)
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        if area >= total:
            continue
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _select_side_origin(cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    """Choose a unique extremal cell that faces the template from the chosen side."""
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
    """Generate a lattice polyomino with a unique extremal cell for the nearest-point origin."""
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
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _template_edge_slots(templ_cells: list[tuple[int, int]], h: int, w: int, side: str) -> list[int]:
    """Rows or columns on the template edge that can face the singleton origin."""
    if side == "left":
        return [tr for tr, tc in templ_cells if tc == 0]
    if side == "right":
        return [tr for tr, tc in templ_cells if tc == w - 1]
    if side == "above":
        return [tc for tr, tc in templ_cells if tr == 0]
    return [tc for tr, tc in templ_cells if tr == h - 1]


def _cluster_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c in points]
    cs = [int(c) for _r, c in points]
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
    templ_col: int,
    overlap_col: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Place same-colored singletons so one chosen point is uniquely nearest to the template."""
    edge_slots = _template_edge_slots(templ_cells, h, w, side)
    template_abs = [(int(r0 + tr), int(c0 + tc)) for tr, tc in templ_cells]
    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)
    origin_idx = lattice_cells.index(origin)

    for _ in range(SINGLETON_RETRIES):
        slot = int(random.choice(edge_slots))
        if side == "left":
            origin_abs = (int(r0 + slot), int(c0 - SINGLETON_GAP))
        elif side == "right":
            origin_abs = (int(r0 + slot), int(c0 + w - 1 + SINGLETON_GAP))
        elif side == "above":
            origin_abs = (int(r0 - SINGLETON_GAP), int(c0 + slot))
        else:
            origin_abs = (int(r0 + h - 1 + SINGLETON_GAP), int(c0 + slot))

        base_r = int(origin_abs[0] - origin[0] * GUNIT)
        base_c = int(origin_abs[1] - origin[1] * GUNIT)
        abs_points = [(int(base_r + lr * GUNIT), int(base_c + lc * GUNIT)) for lr, lc in lattice_cells]
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
        if dists[origin_idx] != min_dist:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(templ_col)
        for rr, cc in abs_points:
            grid[int(rr), int(cc)] = int(overlap_col)

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Build one combo instance with a nearest-point origin and complement-rendered self copy."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)
    side = random.choice(SIDE_CHOICES)

    h = int(random.choice(TEMPLATE_H_CHOICES))
    w = int(TEMPLATE_W)
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing, side)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]
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
                pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
                if pt in occupied:
                    valid = False
                    break
                occupied.add(pt)
            if not valid:
                break
        if not valid:
            continue

        grid = _place_singletons(
            side,
            lattice_cells,
            origin,
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
        if grid is not None:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
