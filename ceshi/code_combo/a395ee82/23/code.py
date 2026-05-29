# Combo variant of a395ee82 (index 23)
# combo_spec: D1/1+D2/2
# Name: 位置规则不均匀间距

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    The input still has one large monochrome template plus a separate cluster of
#    5-6 singleton pixels. Two original cues are removed/changed together:
#    - the origin is no longer marked by template color; all singleton pixels are O
#    - singleton spacing is no longer one global lattice unit; adjacent visible
#      gaps are either 2 or 3 pixels, encoding 1 or 2 stamp steps respectively
#    Small trace:
#    - suppose visible singleton rows are [4, 6, 9], so row gaps are [2, 3] and
#      decoded row steps are [0, 1, 3]
#    - suppose visible singleton cols are [2, 5, 7], so col gaps are [3, 2] and
#      decoded col steps are [0, 2, 3]
#    - if the singleton at visible coordinate (6, 5) is the unique nearest point
#      to the template, its decoded offset is (1, 2); a singleton at (9, 7)
#      therefore means (+2, +1) template copies relative to the visible template
#    Original logic would have identified origin by color and normalized all
#    deltas by one gcd. Here both steps change together: origin comes from
#    geometry, and offsets come from cumulative short/long-gap decoding.
# b) Complexity calibration:
#    The fused rule uses exactly two visible gap categories, 2->1 and 3->2, which
#    keeps the irregular spacing readable. The generator forces both categories to
#    appear in every sample, so the task does not collapse back to uniform spacing.
#    The origin cue is also made strong: the intended origin is the unique side-
#    facing extremal singleton and is checked to be uniquely nearest.
# c) Implicit assumption audit against the original main():
#    - `origin = [pt for pt if col == templ_col]` is invalid now because no
#      singleton has template color.
#    - `overlap_col = majority(non-template singleton colors)` simplifies to the
#      common singleton color; the solver now asserts a single singleton color.
#    - `gunit = gcd(all non-zero deltas)` is invalid because local gap categories,
#      not one global unit, determine decoded offsets.
#    - `is_origin = (col == templ_col)` is also invalid; origin must instead be
#      tracked from the unique nearest singleton.
#    The new solver therefore assumes:
#    - there is exactly one large template component and the rest relevant
#      components are singleton pixels
#    - all singleton rows/cols lie on consecutive warped axis levels
#    - every adjacent visible singleton-row gap and singleton-col gap is 2 or 3
#    - exactly one singleton is uniquely closest to the template in Manhattan
#      distance
# d) Placement legality:
#    The generator remains output-first. It first chooses decoded stamp offsets,
#    places the template so every stamped copy fits in-bounds and copies do not
#    overlap, then places the warped singleton cluster away from the template with
#    at least one background cell of separation.
# e) Visual observability:
#    Human solvers can see both changed dimensions directly:
#    - all source singleton pixels have the same color, so color cannot mark the
#      origin
#    - visible singleton gaps clearly mix short and long spacings
#    The nearest-point origin is made observable by placing the cluster on one
#    side of the template and choosing a unique extremal lattice cell to face it.
# f) Visual cleanliness:
#    The scene still uses only three colors and only two object families: one
#    compact template component and several isolated singletons. No extra markers
#    or overlapping source decorations are introduced.
# g) Color inference:
#    O keeps a structural role that remains inferable from examples: every input
#    singleton is O, and the output recolors the origin copy and any overlap to O.
#    Because all source singletons share that same color, the solver must rely on
#    position and spacing rather than any hidden color identity.

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
SINGLETON_GAP = 2
SIDE_CHOICES = ["left", "right", "above", "below"]

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
LATTICE_RETRIES = 220
GAP_PATTERN_RETRIES = 80
TEMPLATE_POSITION_RETRIES = 80
SINGLETON_RETRIES = 100
GENERATE_RETRIES = 800


def _manhattan_to_template(r: int, c: int, templ_coords: list[tuple[int, int]]) -> int:
    """Minimum Manhattan distance from a point to any filled template pixel."""
    return min(abs(int(r) - tr) + abs(int(c) - tc) for tr, tc in templ_coords)


def _axis_step_map(values: list[int]) -> dict[int, int]:
    """Decode warped source coordinates into cumulative stamp-step coordinates."""
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
    Stamp translated copies of a monochrome template from a same-colored irregular singleton scaffold.

    Input structure:
    - bg is the background, and one large monochrome component is the template in color T.
    - Elsewhere are 5-6 isolated singleton pixels, all in one shared color O; none is color-marked as the origin.
    - The singleton scaffold is warped: adjacent visible row/column gaps are short or long rather than uniformly spaced.

    Rule:
    - Identify the unique singleton with minimum Manhattan distance to the template and treat it as the source origin.
    - Decode the scaffold axes locally: each visible gap of 2 pixels means 1 stamp step, and each gap of 3 pixels means 2 stamp steps.
    - Convert every singleton into a decoded offset from the origin, scaled by the template bbox height/width, and stamp the template at those offsets.
    - Recolor the origin copy to O, and recolor any multiply covered output pixels to O as well.

    Key insight: The two source cues must be combined. Color no longer reveals the origin, and one global spacing unit no longer exists; the solver has to infer origin from geometry and infer offsets from repeated short/long gap categories.
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

    row_to_step = _axis_step_map([r for r, _c, _col in singletons])
    col_to_step = _axis_step_map([c for _r, c, _col in singletons])
    origin_sr = int(row_to_step[int(orr)])
    origin_sc = int(col_to_step[int(orc)])

    deltas = [
        (
            int(row_to_step[int(r)] - origin_sr),
            int(col_to_step[int(c)] - origin_sc),
            bool(r == orr and c == orc),
        )
        for (r, c, _col) in singletons
    ]

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
    return sorted(cells) if len(cells) == n else None


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


def _select_side_origin(cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    """Choose a unique extremal lattice cell that faces the template from one side."""
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


def _gen_lattice(n: int, side: str) -> tuple[list[tuple[int, int]], tuple[int, int], int, int] | None:
    """Generate a coarse lattice with a unique side-facing origin cell."""
    for _ in range(LATTICE_RETRIES):
        ext_r = random.randint(*LATTICE_EXTENT_RANGE)
        ext_c = random.randint(*LATTICE_EXTENT_RANGE)
        cells = _spanning_polyomino(n, ext_r, ext_c)
        if cells is None:
            continue
        origin = _select_side_origin(cells, side)
        if origin is not None:
            return cells, origin, ext_r, ext_c
    return None


def _gen_gap_layout(ext_r: int, ext_c: int) -> tuple[list[int], list[int]] | None:
    """Generate warped source-axis gaps with both visible categories present."""
    for _ in range(GAP_PATTERN_RETRIES):
        row_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_r - 1)]
        col_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_c - 1)]
        if set(row_gaps + col_gaps) != set(SOURCE_GAP_CHOICES):
            continue
        if sum(GAP_TO_STEP[g] for g in row_gaps) > 4:
            continue
        if sum(GAP_TO_STEP[g] for g in col_gaps) > 4:
            continue
        return row_gaps, col_gaps
    return None


def _levels_from_gaps(gaps: list[int]) -> tuple[list[int], list[int]]:
    """Return visible pixel levels and decoded coarse-step levels."""
    src = [0]
    steps = [0]
    for gap in gaps:
        src.append(int(src[-1] + gap))
        steps.append(int(steps[-1] + GAP_TO_STEP[int(gap)]))
    return src, steps


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return min/max row and col deltas."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _template_edge_slots(
    templ_cells: list[tuple[int, int]], h: int, w: int, side: str
) -> list[int]:
    """Rows or columns on the template edge that can face the source origin."""
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
    row_src: list[int],
    col_src: list[int],
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
    """Place an irregular same-colored singleton cluster with a unique nearest origin."""
    edge_slots = _template_edge_slots(templ_cells, h, w, side)
    template_abs = [(int(r0 + tr), int(c0 + tc)) for tr, tc in templ_cells]
    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1
    origin_idx = lattice_cells.index(origin)

    for _ in range(SINGLETON_RETRIES):
        slot = random.choice(edge_slots)
        if side == "left":
            origin_abs = (int(r0 + slot), int(c0 - SINGLETON_GAP))
        elif side == "right":
            origin_abs = (int(r0 + slot), int(c0 + w - 1 + SINGLETON_GAP))
        elif side == "above":
            origin_abs = (int(r0 - SINGLETON_GAP), int(c0 + slot))
        else:
            origin_abs = (int(r0 + h - 1 + SINGLETON_GAP), int(c0 + slot))

        base_r = int(origin_abs[0] - row_src[origin[0]])
        base_c = int(origin_abs[1] - col_src[origin[1]])
        abs_points = [(int(base_r + row_src[lr]), int(base_c + col_src[lc])) for lr, lc in lattice_cells]
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

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for rr, cc in abs_points:
            grid[int(rr), int(cc)] = int(O)

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the fused nearest-origin irregular-gap variant."""
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
    lattice_cells, origin, ext_r, ext_c = lat

    gaps = _gen_gap_layout(ext_r, ext_c)
    if gaps is None:
        return None
    row_gaps, col_gaps = gaps
    row_src, row_steps = _levels_from_gaps(row_gaps)
    col_src, col_steps = _levels_from_gaps(col_gaps)

    step_deltas: list[tuple[int, int]] = []
    for lr, lc in lattice_cells:
        step_deltas.append(
            (
                int(row_steps[lr] - row_steps[origin[0]]),
                int(col_steps[lc] - col_steps[origin[1]]),
            )
        )

    min_sdr, max_sdr, min_sdc, max_sdc = _delta_bounds(step_deltas)
    r0_lo, r0_hi = -min_sdr * h, GRID_SIZE - max_sdr * h - h
    c0_lo, c0_hi = -min_sdc * w, GRID_SIZE - max_sdc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    for _ in range(TEMPLATE_POSITION_RETRIES):
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)

        occupied: set[tuple[int, int]] = set()
        valid = True
        for dr, dc in step_deltas:
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
            side=side,
            lattice_cells=lattice_cells,
            origin=origin,
            row_src=row_src,
            col_src=col_src,
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
        if grid is not None:
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
