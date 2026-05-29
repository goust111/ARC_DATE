# Combo variant of a395ee82 (index 3)
# combo_spec: D1/1+D4/4+D9/0
# Name: 位置擦除保留标记

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# a) Concrete behavior and trace:
#    - Base task: one large monochrome template T plus a disjoint singleton lattice. A specially colored
#      singleton marks the origin, every lattice point produces a template copy, and the origin copy is
#      recolored.
#    - Fused variant:
#      1. All singleton pixels share one marker color O, so color no longer identifies the origin.
#      2. The origin is instead the unique singleton with minimum Manhattan distance to the visible input
#         template. This is enforced geometrically by placing the lattice on one side of the template and
#         making exactly one extremal singleton face the template.
#      3. The origin copy is erased rather than recolored: after translating the template by all lattice
#         offsets, only non-origin offsets paint T. The input template position therefore becomes a blank
#         template-shaped hole in the stamp layer.
#      4. Finally, the original singleton markers are restored on top, so the output keeps the O-colored
#         lattice dots as visible source markers; some dots are forced to fall on future T-stamp pixels,
#         creating small O contrasts inside the replicated shapes.
#    - Small trace:
#      * Suppose the nearest singleton is two cells left of the template edge and the lattice extends
#        further left/down from that point.
#      * Original rule would color-match the origin singleton and recolor the template at delta (0,0).
#      * Fused rule instead infers that nearest point as origin, removes the delta (0,0) copy entirely,
#        stamps T at the other deltas, and then redraws every singleton O at its original coordinates.
#      * Visible difference: the input template disappears into a hole, while the singleton lattice remains
#        visible and at least one marker sits inside a stamped copy.
# b) Complexity calibration:
#    - Only three colors remain active: bg, template T, marker O. No hidden color-id branching is used.
#    - The singleton lattice still has 5-6 cells on a gunit=2 lattice, so the underlying task family stays
#      recognizable.
#    - The generator enforces a unique nearest singleton, a full erased origin slot, and at least one kept
#      marker that lands on a non-origin stamp. This keeps all three fused changes visually observable.
# c) Implicit-assumption audit:
#    - `origin = singleton with color T` is invalid after D1/1; origin detection must be geometric.
#    - `other singleton colors by majority vote` is unnecessary here because all singleton colors are the
#      same by construction; the solver asserts that fact explicitly.
#    - `recolor origin coverage to O` is invalid after D4/4; the stamp layer must ignore origin coverage
#      instead of painting it.
#    - `return stamp layer directly` is invalid after D9/0; singleton markers must be restored last so they
#      override any T stamp beneath them.
#    - The largest-component heuristic for finding the template still holds because the input contains one
#      area-6..10 template component and only area-1 singleton components.
# d) Placement legality:
#    - All translated template copies fit inside the 22x22 grid.
#    - Future template copies are kept pairwise non-overlapping for visual cleanliness.
#    - The singleton cluster bbox remains disjoint from the visible input template bbox with at least one
#      background cell gap, so the input still decomposes cleanly into template + singleton lattice.
#    - Unlike the base generator, singleton markers are allowed to land on future non-origin stamp pixels;
#      this is required for the "keep source" effect to be visible.
# e) Visual observability:
#    - From examples, a human can see that all singleton colors match, so the origin cannot be color-coded.
#    - The nearest marker to the visible template is singled out by geometry.
#    - The output removes the original template instead of recoloring it, which makes the erase rule
#      topological and easy to notice.
#    - Preserved O markers, especially the ones embedded inside T copies, reveal that the source dots remain.
# f) Visual cleanliness:
#    - The output uses one large family of T copies plus sparse O dots; non-overlapping copies avoid muddy
#      multi-layer interactions.
#    - The singleton cluster stays compact and separate from the input template, so object parsing remains
#      easy even though some future stamps pass through the cluster region in the output.
# g) Change inferability:
#    - Color changes are structural rather than arbitrary: T means template paint, O means retained marker.
#    - The hole always corresponds to the nearest singleton's delta, and marker preservation is repeated
#      across every example, so the fused behavior is recoverable from a few demonstrations.


def _manhattan_to_template(r: int, c: int, templ_abs: list[tuple[int, int]]) -> int:
    """Minimum Manhattan distance from (r,c) to any filled template pixel."""
    return min(abs(int(r) - tr) + abs(int(c) - tc) for tr, tc in templ_abs)


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    """Parse the template, same-colored singleton lattice, and geometric origin."""
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
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert len(singletons) >= 2
    singleton_cols = {int(col) for _r, _c, col in singletons}
    assert len(singleton_cols) == 1
    marker_col = int(next(iter(singleton_cols)))
    assert marker_col != templ_col

    dists = [_manhattan_to_template(r, c, templ_abs) for r, c, _col in singletons]
    min_dist = min(dists)
    assert dists.count(min_dist) == 1
    origin_idx = dists.index(min_dist)
    orr, orc, _ = singletons[origin_idx]

    raw_deltas = [
        (int(r - orr), int(c - orc), bool(idx == origin_idx))
        for idx, (r, c, _col) in enumerate(singletons)
    ]
    gcd_vals = [abs(int(x)) for dr, dc, _is_origin in raw_deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), bool(is_origin)) for dr, dc, is_origin in raw_deltas]

    return {
        "grid": g,
        "bg": int(bg),
        "H": int(H),
        "W": int(W),
        "templ_col": int(templ_col),
        "marker_col": int(marker_col),
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "templ_rel": templ_rel,
        "templ_abs": templ_abs,
        "singletons": singletons,
        "deltas": deltas,
        "origin_rc": (int(orr), int(orc)),
        "singleton_dists": dists,
    }


def _render_scene(scene: dict[str, object], *, keep_singletons: bool) -> np.ndarray:
    """Render the erase-stamp layer, then optionally restore the input singleton markers."""
    bg = int(scene["bg"])
    H = int(scene["H"])
    W = int(scene["W"])
    templ_col = int(scene["templ_col"])
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    h = int(scene["h"])
    w = int(scene["w"])
    templ_rel = list(scene["templ_rel"])
    deltas = list(scene["deltas"])
    singletons = list(scene["singletons"])

    out = np.full((H, W), int(bg), dtype=int)
    for dr, dc, is_origin in deltas:
        if is_origin:
            continue
        off_r = int(dr * h)
        off_c = int(dc * w)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                out[R, C] = int(templ_col)

    if keep_singletons:
        for rr, cc, col in singletons:
            out[int(rr), int(cc)] = int(col)

    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the generated input exhibits the intended fused behavior."""
    scene = _extract_scene(grid)
    stamp_only = _render_scene(scene, keep_singletons=False)
    final = _render_scene(scene, keep_singletons=True)

    bg = int(scene["bg"])
    templ_col = int(scene["templ_col"])
    r0 = int(scene["r0"])
    c0 = int(scene["c0"])
    templ_rel = list(scene["templ_rel"])
    singletons = list(scene["singletons"])
    dists = sorted(int(x) for x in scene["singleton_dists"])

    stamped_singletons = 0
    visible_keep_marks = 0
    for rr, cc, _col in singletons:
        rr = int(rr)
        cc = int(cc)
        if int(stamp_only[rr, cc]) == int(templ_col):
            stamped_singletons += 1
        if int(final[rr, cc]) != int(stamp_only[rr, cc]):
            visible_keep_marks += 1

    origin_gap_cells = 0
    for rr, cc in templ_rel:
        if int(stamp_only[int(r0 + rr), int(c0 + cc)]) == int(bg):
            origin_gap_cells += 1

    nearest_margin = int(dists[1] - dists[0])
    return {
        "stamped_singletons": int(stamped_singletons),
        "visible_keep_marks": int(visible_keep_marks),
        "origin_gap_cells": int(origin_gap_cells),
        "template_area": int(len(templ_rel)),
        "nearest_margin": int(nearest_margin),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a monochrome template from a singleton lattice, but infer the origin geometrically and erase that slot.

    Input structure:
    - bg is the background color.
    - One large monochrome component is the template sprite, colored T.
    - Elsewhere there is a uniform-spacing lattice of singleton pixels, and every singleton uses the same marker color O.
    - No singleton is specially recolored; the origin is instead the singleton closest to the visible template.

    Rule:
    - Find the unique nearest singleton to the template pixels and treat it as the lattice origin.
    - Convert every singleton's offset from that origin into coarse lattice units, then scale offsets by the template bbox height and width.
    - Stamp the template at every non-origin offset using color T, but do not draw the origin copy at all, leaving that translated slot as background.
    - After the stamp layer is built, restore all original singleton markers at their input coordinates so they remain visible and override any T pixel underneath.

    Key insight: The nearest marker identifies which translated template instance should be missing, and the surviving singleton dots are not consumed by stamping, so the output combines an erased origin slot with preserved O markers.
    """
    scene = _extract_scene(grid)
    return _render_scene(scene, keep_singletons=True)


# --------------- generator (combo D1/1 + D4/4 + D9/0) ---------------

# Tunable configuration
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

# Observability constraints
MIN_STAMPED_SINGLETONS = 1
MIN_VISIBLE_KEEP_MARKS = 1

# Retry budgets
SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
LATTICE_RETRIES = 180
TEMPLATE_POSITION_RETRIES = 80
SINGLETON_RETRIES = 160
GENERATE_RETRIES = 700


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
    """4-connected template in h x w, spanning its bbox and respecting area/fill bounds."""
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
    """Pick a unique extremal lattice cell that can be the nearest singleton from the chosen side."""
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
    """Generate a lattice polyomino whose side-facing extremal cell can be the geometric origin."""
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
    """Return min/max row and col offsets for a list of lattice deltas."""
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _template_edge_slots(templ_cells: list[tuple[int, int]], h: int, w: int, side: str) -> list[int]:
    """Rows or columns on the template edge that can face the nearest singleton."""
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
    templ_col: int,
    marker_col: int,
) -> np.ndarray | None:
    """Place same-colored singleton markers so the nearest-point, erase, and keep-source cues are all visible."""
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

        dists = [_manhattan_to_template(rr, cc, template_abs) for rr, cc in abs_points]
        min_dist = min(dists)
        origin_idx = lattice_cells.index(origin)
        if dists.count(min_dist) != 1 or dists[origin_idx] != min_dist:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = templ_col
        for rr, cc in abs_points:
            grid[int(rr), int(cc)] = marker_col

        try:
            scene = _extract_scene(grid)
            out = main(grid)
            metrics = _variant_metrics(grid)
        except Exception:
            continue

        if tuple(scene["origin_rc"]) != (int(origin_abs[0]), int(origin_abs[1])):
            continue
        if metrics["nearest_margin"] < 1:
            continue
        if metrics["origin_gap_cells"] != metrics["template_area"]:
            continue
        if metrics["stamped_singletons"] < MIN_STAMPED_SINGLETONS:
            continue
        if metrics["visible_keep_marks"] < MIN_VISIBLE_KEEP_MARKS:
            continue
        if not np.array_equal(out, _render_scene(scene, keep_singletons=True)):
            continue

        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Build one valid fused-variant input."""
    bg, templ_col, marker_col = random.sample(range(10), 3)
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
            marker_col,
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
