# Combo variant of b9e38dc0 (index 20)
# combo_spec: D2/0+D18/0
# Name: 距离决定步长+对角线移动

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the fused variant
# a) Concrete behavior:
#    Keep the original first stage: a single wall funnel encloses a fill seed,
#    the interior flood-fills, then the fill radiates outward through the open end.
#    Add one same-color singleton outside the opening. In the old family summary
#    that singleton would move straight outward and erase every traversed cell.
#    In this combo it instead moves diagonally: the outward component still matches
#    the funnel opening, while the lateral component is determined by which endpoint
#    side the singleton sits near. Its "speed" is no longer 1 cell per mark: if the
#    singleton starts N cells from its associated endpoint side of the wall, then only
#    diagonal positions at offsets 0, N, 2N, ... are turned back to bg, producing a
#    sparse slash through the filled wedge. The singleton itself ends on the last
#    in-bounds diagonal cell.
# b) Complexity calibration:
#    The generator forces step >= 2 and requires multiple visible erased marks inside
#    the radiated wedge, so the result is not visually confusable with either the
#    original straight ray or a dense step-1 diagonal. Only one wall + one singleton
#    are used because adding multiple sprites would mostly add association ambiguity
#    rather than helping the human infer the intended distance->spacing relation.
# c) Implicit assumptions audited:
#    - Wall detection still assumes the wall is the most frequent non-bg color.
#    - Fill-color detection still assumes the fill/isolated color is the non-wall
#      color nearest to the wall; the generator therefore uses no extra near-wall
#      colors.
#    - The fill source is no longer "all fill-colored cells": we must separate the
#      interior seed blob from the outside singleton. The solver now chooses the fill
#      component with the smallest projection opposite the radiate direction as the
#      interior source and treats later components as movable singletons.
#    - The diagonal path rule assumes the outside singleton is nearer one endpoint
#      than the other; the generator places it on an endpoint's outward cardinal
#      extension so that the side choice and the endpoint-distance step are both
#      unambiguous.
# d) Placement legality:
#    The generator first builds the wall and seed, runs the stage-1 fill, then places
#    the singleton only on a bg cell that lies inside the nominal radiated zone,
#    stays in-bounds, has the intended distance to its associated endpoint side,
#    and yields a long enough diagonal path to the border.
# e) Visual observability:
#    Each sample shows three coupled clues a human can count by eye: the singleton is
#    offset from one endpoint side, its endpoint distance is an integer N, and the
#    resulting bg holes on the diagonal repeat with the same gap N. That repeated
#    spatial support is stronger than a hidden numeric rule tied to color ids.
# f) Visual cleanliness:
#    The wall remains one color, the fill/isolated singleton share one color, and
#    there are no extra obstacle colors. The only subtraction in the output is the
#    sparse diagonal bg trail, which keeps the scene readable rather than cluttered.
# g) Inferability of the change:
#    No invisible ranking by color id is used. The diagonal direction comes from
#    endpoint side placement, and the spacing comes from a visible geometric distance.
#    The generator rejects cases where the combo output collapses to the dense-diagonal
#    or original straight-ray baselines.


# -- Geometry / generation constants -----------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)

OPEN_END_MARGIN = 5
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3
ISOLATED_GAP_RANGE = (2, 5)

MIN_NEW_FILL = 20
MIN_VISIBLE_BG_MARKS = 2
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _projection(p: tuple[int, int], rad: tuple[int, int]) -> int:
    return int(p[0]) * int(rad[0]) + int(p[1]) * int(rad[1])


def _component_sets(mask: np.ndarray, connectivity: int) -> list[set[tuple[int, int]]]:
    return [
        set((int(r), int(c)) for r, c in comp)
        for comp in components_mask(mask, connectivity=connectivity)
    ]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int) -> list[tuple[int, int]]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows: list[int] = []
    if n_side_steps:
        chosen_rows = [1]
        if n_side_steps > 1:
            chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
        chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        depth = random.randint(DEPTH_HORIZ[0], min(DEPTH_HORIZ[1], w - 6))
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    elif gap == 6:
        left_bottom, right_bottom = 2, 4
    else:
        left_bottom, right_bottom = random.choice([(2, 5), (3, 4)])

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    left = _make_arm(depth, 0, left_bottom, extra_l)
    right = _make_arm(depth, gap, right_bottom, extra_r)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < WALL_PIXELS_MIN:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            ) >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int]:
    def clamped_rand(lo: int, hi: int) -> int:
        if lo > hi:
            lo = hi
        return random.randint(lo, hi)

    if rad == (-1, 0):
        return clamped_rand(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return clamped_rand(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), clamped_rand(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), clamped_rand(depth + 2, w - 4)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior: list[tuple[int, int]] = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        p = frontier.pop(random.randrange(len(frontier)))
        u, v = p
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _outward_dir(comp_mask: np.ndarray, ep: tuple[int, int], rad_hint: tuple[int, int] | None = None):
    n1 = _comp_neighbors(comp_mask, *ep)[0]
    d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
    d_use = d1
    is_12 = False
    n2s = [p for p in _comp_neighbors(comp_mask, *n1) if p != ep]
    if n2s:
        n2 = n2s[0]
        d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
        if (d1[0] == 0) ^ (d1[1] == 0):
            if d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
    out = (-d_use[0], -d_use[1])
    if rad_hint is not None and out == (0, 0):
        out = rad_hint
    return out, is_12


def _endpoint_side(endpoints: list[tuple[int, int]], endpoint: tuple[int, int], rad: tuple[int, int]) -> int:
    if rad[0] != 0:
        mid = sum(c for _r, c in endpoints) / len(endpoints)
        return -1 if endpoint[1] < mid else 1
    mid = sum(r for r, _c in endpoints) / len(endpoints)
    return -1 if endpoint[0] < mid else 1


def _diagonal_dir(endpoints: list[tuple[int, int]], start: tuple[int, int], rad: tuple[int, int]) -> tuple[int, int]:
    endpoint = min(endpoints, key=lambda ep: (_manhattan(start, ep), ep))
    side = _endpoint_side(endpoints, endpoint, rad)
    if rad[0] != 0:
        return rad[0], side
    return side, rad[1]


def _associated_endpoint(endpoints: list[tuple[int, int]], start: tuple[int, int]) -> tuple[int, int]:
    return min(endpoints, key=lambda ep: (_manhattan(start, ep), ep))


def _inward_dir(endpoints: list[tuple[int, int]], endpoint: tuple[int, int], rad: tuple[int, int]) -> tuple[int, int]:
    side = _endpoint_side(endpoints, endpoint, rad)
    if rad[0] != 0:
        return 0, -side
    return -side, 0


def _pick_fill_color(g: np.ndarray, bg: int, wall: int, wall_points: list[tuple[int, int]]) -> int:
    other_colors = [int(v) for v in np.unique(g) if int(v) not in (bg, wall)]
    assert other_colors, "expected at least one non-wall color"

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        best_d = 10 ** 9
        for p in pts:
            for wp in wall_points:
                d = _manhattan(p, wp)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    return min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))


def _fill_stage(grid: np.ndarray):
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = int(get_bg(g))

    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"
    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_comps = _component_sets(g == wall, connectivity=8)
    assert wall_comps, "expected a wall component"

    wall_points = [p for comp in wall_comps for p in comp]
    fill = _pick_fill_color(g, bg, wall, wall_points)

    fill_components = _component_sets(g == fill, connectivity=4)
    assert fill_components, "expected at least one fill-colored component"

    all_fill_pts = [p for comp in fill_components for p in comp]

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        return min(_manhattan(p, q) for p in comp for q in all_fill_pts)

    wall_comp = min(wall_comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))

    comp_mask = np.zeros_like(g, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    endpoints = [p for p in wall_comp if len(_comp_neighbors(comp_mask, *p)) == 1]
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    out_info = [_outward_dir(comp_mask, ep) for ep in endpoints]
    out_dirs = [d for d, _is_12 in out_info]
    tip_12 = [flag for _d, flag in out_info]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad != (0, 0), "expected a non-zero radiate direction"

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int], skip: int = 0) -> None:
        rr, cc = start
        for _ in range(skip):
            rr += d[0]
            cc += d[1]
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += d[0]
                cc += d[1]
            else:
                if rad[0] != 0:
                    rr += d[0]
                else:
                    cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d, skip=0)

    seed_comp = min(
        fill_components,
        key=lambda comp: (
            sum(_projection(p, rad) for p in comp) / len(comp),
            -len(comp),
        ),
    )
    seed_points = list(seed_comp)
    seed_set = set(seed_points)

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for p in seed_points:
        if passable[p]:
            seen[p] = True
            q.append(p)
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)
    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    else:
        if dr == dc:
            for k in range(-(w - 1), h):
                coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                shadow_process(coords)
        else:
            for k in range(h + w - 1):
                coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                shadow_process(coords)

    isolated_components = []
    seed_proj = min(_projection(p, rad) for p in seed_set)
    for comp in fill_components:
        if comp == seed_comp:
            continue
        comp_proj = min(_projection(p, rad) for p in comp)
        if comp_proj > seed_proj:
            isolated_components.append(comp)

    ctx = {
        "bg": bg,
        "fill": fill,
        "wall": wall,
        "wall_comp": wall_comp,
        "wall_mask": comp_mask,
        "filled_mask": filled_mask.copy(),
        "endpoints": endpoints,
        "rad": rad,
        "seed_comp": seed_set,
        "isolated_components": isolated_components,
    }
    return out, ctx


def _trace_motion_path(
    start: tuple[int, int],
    move_dir: tuple[int, int],
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    coords = [start]
    cur = start
    while True:
        nr = int(cur[0] + move_dir[0])
        nc = int(cur[1] + move_dir[1])
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if occupied[nr, nc]:
            break
        coords.append((nr, nc))
        cur = (nr, nc)
    return coords


def _apply_isolated_motion(stage1: np.ndarray, ctx: dict, mode: str = "combo") -> np.ndarray:
    out = np.array(stage1, copy=True)
    bg = int(ctx["bg"])
    fill = int(ctx["fill"])
    endpoints = list(ctx["endpoints"])
    rad = tuple(ctx["rad"])

    occupied = (out != bg) & (out != fill)

    starts = []
    for comp in ctx["isolated_components"]:
        if len(comp) == 1:
            starts.append(next(iter(comp)))
        else:
            starts.extend(sorted(comp))

    for start in sorted(starts):
        move_dir = _diagonal_dir(endpoints, start, rad)
        step = _manhattan(start, _associated_endpoint(endpoints, start))
        if mode == "straight_step1":
            move_dir = rad
            step = 1
        elif mode == "diag_step1":
            step = 1
        elif mode == "combo":
            step = max(1, step)
        else:
            raise ValueError(f"unknown motion mode: {mode}")

        coords = _trace_motion_path(start, move_dir, occupied)
        if not coords:
            continue

        for idx, (r, c) in enumerate(coords):
            if idx % step == 0:
                out[r, c] = bg
        lr, lc = coords[-1]
        out[lr, lc] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill a wall-defined cavity, radiate outward, then carve a sparse diagonal trail.

    Input structure:
    - bg is the most common color. The wall is the dominant non-bg color and forms one
      open funnel with exactly two endpoints. A same-color seed blob sits inside near the
      closed end, and a same-color singleton sits outside the opening near one endpoint.

    Rule:
    - First, fill the cavity from the interior seed and continue that fill outward through
      the opening, bounded by endpoint rays just as in the base task.
    - Then move the outside singleton diagonally toward the grid edge: its outward component
      matches the funnel opening, its sideways component comes from the endpoint side it is
      nearest to, and if its initial distance to that associated endpoint is N, only every
      N-th diagonal position is turned back to bg. The singleton lands on the last reachable
      diagonal cell.

    Key insight:
    - The wall still determines the radiating wedge, but the outside singleton is no longer
      a dense straight eraser. Its endpoint-distance becomes the spacing parameter, and its
      endpoint-side placement determines which diagonal slash is cut through the fill.
    """
    stage1, ctx = _fill_stage(np.asarray(grid, dtype=int))
    return _apply_isolated_motion(stage1, ctx, mode="combo")


def _find_isolated_start(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    stage1, ctx = _fill_stage(g)
    bg = int(ctx["bg"])
    fill = int(ctx["fill"])
    starts = []
    for comp in ctx["isolated_components"]:
        if len(comp) == 1:
            starts.append(next(iter(comp)))
    if not starts:
        raise ValueError("no isolated singleton detected")
    start = starts[0]
    if g[start] != fill or stage1[start] != fill:
        raise ValueError("isolated singleton is not preserved through stage 1")
    return start


def _counterfactual_outputs(grid: np.ndarray):
    stage1, ctx = _fill_stage(np.asarray(grid, dtype=int))
    return {
        "combo": _apply_isolated_motion(stage1, ctx, mode="combo"),
        "diag_step1": _apply_isolated_motion(stage1, ctx, mode="diag_step1"),
        "straight_step1": _apply_isolated_motion(stage1, ctx, mode="straight_step1"),
        "stage1": stage1,
        "ctx": ctx,
    }


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), 3)
    bg, wall_color, fill_color = palette

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
    if not interior:
        return None
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_stage1, ctx = _fill_stage(grid)
    except Exception:
        return None

    nominal_fill_zone = (nominal_stage1 == fill_color) & (grid == bg)
    if int(nominal_fill_zone.sum()) < MIN_NEW_FILL or not _touches_border(nominal_fill_zone):
        return None

    local_endpoints = _degree1_pixels(set(local_wall))
    if len(local_endpoints) != 2:
        return None
    endpoint_grid = [_map_local(u, v, rad, anchor) for u, v in local_endpoints]

    random.shuffle(endpoint_grid)
    step_options = list(range(ISOLATED_GAP_RANGE[0], ISOLATED_GAP_RANGE[1] + 1))
    random.shuffle(step_options)

    placed = False
    for ep in endpoint_grid:
        move_dir = _diagonal_dir(endpoint_grid, (ep[0] + rad[0], ep[1] + rad[1]), rad)
        inward = _inward_dir(endpoint_grid, ep, rad)
        for step in step_options:
            forward_options = list(range(1, step + 1))
            random.shuffle(forward_options)
            for forward in forward_options:
                inward_steps = step - forward
                sr = ep[0] + forward * rad[0] + inward_steps * inward[0]
                sc = ep[1] + forward * rad[1] + inward_steps * inward[1]
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                if grid[sr, sc] != bg:
                    continue
                if not nominal_fill_zone[sr, sc]:
                    continue

                endpoint_dist = _manhattan((sr, sc), ep)
                if endpoint_dist != step or endpoint_dist < 2:
                    continue

                occupied = (nominal_stage1 != bg) & (nominal_stage1 != fill_color)
                path = _trace_motion_path((sr, sc), move_dir, occupied)
                if len(path) <= step + 1:
                    continue

                visible_marks = sum(
                    1
                    for idx, p in enumerate(path)
                    if idx % step == 0 and nominal_stage1[p] == fill_color
                )
                if visible_marks < MIN_VISIBLE_BG_MARKS:
                    continue

                lr, lc = path[-1]
                if lr not in (0, h - 1) and lc not in (0, w - 1):
                    continue

                grid[sr, sc] = fill_color
                placed = True
                break
            if placed:
                break
        if placed:
            break

    if not placed:
        return None

    try:
        outs = _counterfactual_outputs(grid)
    except Exception:
        return None

    combo = outs["combo"]
    diag_dense = outs["diag_step1"]
    straight_dense = outs["straight_step1"]
    stage1 = outs["stage1"]

    if np.array_equal(combo, diag_dense) or np.array_equal(combo, straight_dense):
        return None

    bg_holes = (combo == bg) & (stage1 == fill_color)
    if int(bg_holes.sum()) < MIN_VISIBLE_BG_MARKS:
        return None

    new_fill = (combo == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
