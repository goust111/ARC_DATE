# Combo variant of b9e38dc0 (index 18)
# combo_spec: D5/1+D16/2
# Name: 像素数编码射线长度+距离编码终止

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis for the fused variant
# a) Behavior trace on a small vertical example:
#    Original: a 3-pixel fill seed inside the wall floods the whole funnel interior and then keeps
#    radiating all the way to the top edge. Outside singletons are irrelevant unless they happen to
#    block the zone.
#    Fused: keep the same wall/opening detection, but first compute that same full funnel-to-edge
#    zone as a latent geometry. Let seed_count=3 and marker_count=4. The interior still fills fully.
#    Outside the opening, depth t=1,2,3 are left blank, depths t=4,5,6,7 are painted, and deeper
#    layers stay bg. So the output shows a visible gap caused by the seed blob size and a finite
#    painted band caused by the isolated-pixel count.
# b) Complexity calibration:
#    Two independent small counts are enough to create visible variation without adding ambiguous
#    object matching. seed_count varies in 1..4 and marker_count varies in 2..5, so examples can
#    repeat each count value across a few training pairs while still producing clearly different
#    external bands.
# c) Implicit-assumption audit:
#    The original code assumed rays can safely run to the border; that assumption is removed by
#    clipping the already-computed nominal fill zone by axial depth. It also assumed "other" pixels
#    could act as blockers; here the generator keeps marker pixels outside the nominal fill zone, so
#    non-fill foreground no longer needs to participate in propagation.
# d) Placement legality:
#    The generator chooses seed_count and marker_count only when the opening has enough free space
#    for skip + painted band + spare tail. Markers are placed on bg cells outside the nominal fill
#    zone with Chebyshev separation >= 2, so they stay isolated and never overlap the wall or seed.
# e) Visual observability:
#    A solver can count the fill blob and the isolated pixels directly from the input. In the output,
#    the exterior wedge shows two separate cues: an initial blank gap and then a finite-length band.
#    Because the tail beyond the band is forced to remain empty, the finite stopping rule is visible.
# f) Visual cleanliness:
#    Only three active structures occupy the scene: wall, interior seed blob, and isolated markers.
#    Markers are kept away from the funnel, so the output does not stack counting markers on top of
#    the radiated band.
# g) Change inferability:
#    No hidden ordering by color id is used. The fill color is still the wall-adjacent source color,
#    and the marker color is just "the other singleton foreground". The transform depends only on
#    countable geometry that appears in every example.

# -- Configuration constants --------------------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3

SEED_COUNT_RANGE = (1, 4)
MARKER_COUNT_RANGE = (2, 5)
EXTRA_TAIL = 2

MARKER_MIN_SEP = 2

MIN_NEW_FILL = 12
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points):
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
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


def _build_local_wall(rad, h, w):
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

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
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
            )
            >= 2
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


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth, gap):
    by_row = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed, interior_set, target_size):
    if target_size == 1:
        return {seed}
    for _ in range(50):
        blob = {seed}
        frontier = [seed]
        while frontier and len(blob) < target_size:
            p = frontier.pop(random.randrange(len(frontier)))
            u, v = p
            for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
                if q in interior_set and q not in blob:
                    blob.add(q)
                    frontier.append(q)
                    if len(blob) >= target_size:
                        break
        if len(blob) == target_size:
            return blob
    return None


def _comp_neighbors(comp_mask, r, c):
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _scene_meta(grid: np.ndarray):
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected non-background colors"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(wall_mask, connectivity=8)
    ]
    assert comps, "expected wall component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected fill color"
    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int):
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected fill seed"

    def comp_min_dist(comp):
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    endpoints = [(r, c) for r, c in wall_comp if len(_comp_neighbors(comp_mask, r, c)) == 1]
    assert endpoints, "expected endpoints"

    def outward_dir(ep):
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
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad != (0, 0), "expected radiate direction"

    return {
        "g": g,
        "h": h,
        "w": w,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
    }


def _add_full_rays(barrier, endpoints, out_dirs, tip_12, rad):
    h, w = barrier.shape

    def add_ray(start, d):
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start, d):
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
            add_ray(ep, d)


def _nominal_fill_mask(meta):
    g = meta["g"]
    barrier = meta["comp_mask"].copy()
    _add_full_rays(barrier, meta["endpoints"], meta["out_dirs"], meta["tip_12"], meta["rad"])

    passable = (~barrier) & ((g == meta["bg"]) | (g == meta["fill"]))
    q = meta["fill_pos"][:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < meta["h"] and 0 <= cc < meta["w"] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen & (g == meta["bg"])


def _marker_count(grid, bg, wall, fill):
    marker_mask = (grid != bg) & (grid != wall) & (grid != fill)
    count = 0
    for pts in components_mask(marker_mask, connectivity=8):
        if len(pts) == 1:
            count += 1
    return count


def _axial_depth(meta, r, c):
    endpoints = meta["endpoints"]
    rad = meta["rad"]
    if rad == (-1, 0):
        return min(ep[0] for ep in endpoints) - r
    if rad == (1, 0):
        return r - max(ep[0] for ep in endpoints)
    if rad == (0, -1):
        return min(ep[1] for ep in endpoints) - c
    return c - max(ep[1] for ep in endpoints)


def _external_depths(mask, meta):
    depths = sorted({_axial_depth(meta, int(r), int(c)) for r, c in zip(*np.where(mask)) if _axial_depth(meta, int(r), int(c)) > 0})
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the wall-bounded interior and paint only a counted slice of the outward funnel.

    Input structure:
    - The most common color is background; the most frequent non-background color is a partial wall
      with a single opening. A small fill blob sits inside near the closed end.
    - Remaining non-wall, non-fill foreground pixels are isolated singleton markers elsewhere on the
      canvas; their count is part of the instruction.

    Rule:
    - First recover the same interior + outward funnel that the original task would have filled.
    - Keep the interior fill, but outside the opening skip a number of depth layers equal to the
      fill-blob size, then paint only as many outward layers as there are isolated markers.

    Key insight: Two visible counts govern one ray system: source-pixel count sets the blank offset,
    and isolated-pixel count sets the length of the painted exterior band.
    """
    g = np.array(grid, copy=True)
    meta = _scene_meta(g)
    nominal_fill = _nominal_fill_mask(meta)

    seed_count = len(meta["fill_pos"])
    marker_count = _marker_count(g, meta["bg"], meta["wall"], meta["fill"])

    out = g.copy()
    for r, c in zip(*np.where(nominal_fill)):
        rr, cc = int(r), int(c)
        depth = _axial_depth(meta, rr, cc)
        if depth <= 0 or seed_count < depth <= seed_count + marker_count:
            out[rr, cc] = meta["fill"]
    return out


def _open_space(rad, endpoints, h, w):
    if rad == (-1, 0):
        return min(r for r, _ in endpoints)
    if rad == (1, 0):
        return h - 1 - max(r for r, _ in endpoints)
    if rad == (0, -1):
        return min(c for _, c in endpoints)
    return w - 1 - max(c for _, c in endpoints)


def _marker_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= MARKER_MIN_SEP for rr, cc in placed)


def _find_marker_candidates(grid, bg, wall_color, nominal_fill_zone):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (~nominal_fill_zone) & (dist > 1)
    coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    random.shuffle(coords)
    return coords


def _place_markers(grid, marker_count, marker_color, candidates):
    placed = []
    for p in candidates:
        if len(placed) >= marker_count:
            break
        if _marker_fits(p, placed):
            placed.append(p)
    if len(placed) != marker_count:
        return False
    for r, c in placed:
        grid[r, c] = marker_color
    return True


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), 4)
    bg, wall_color, fill_color, marker_color = palette

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    local_endpoints = _degree1_pixels(local_wall)
    endpoints = [_map_local(u, v, rad, anchor) for u, v in local_endpoints]
    open_space = _open_space(rad, endpoints, h, w)
    if open_space < SEED_COUNT_RANGE[0] + MARKER_COUNT_RANGE[0] + EXTRA_TAIL:
        return None

    interior = _local_interior(local_wall, depth, gap)
    interior_set = set(interior)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - 3)]
    if not seed_pool:
        return None

    pairs = [
        (seed_count, marker_count)
        for seed_count in range(SEED_COUNT_RANGE[0], SEED_COUNT_RANGE[1] + 1)
        for marker_count in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1)
        if seed_count + marker_count + EXTRA_TAIL <= open_space and seed_count <= len(interior)
    ]
    if not pairs:
        return None
    weights = [seed_count + marker_count for seed_count, marker_count in pairs]
    seed_count, marker_count = random.choices(pairs, weights=weights, k=1)[0]

    blob = _grow_seed_blob(random.choice(seed_pool), interior_set, seed_count)
    if blob is None:
        return None
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        meta = _scene_meta(grid)
    except Exception:
        return None
    nominal_fill_zone = _nominal_fill_mask(meta)
    full_ext_depths = _external_depths(nominal_fill_zone, meta)
    if not full_ext_depths or full_ext_depths[-1] < seed_count + marker_count + EXTRA_TAIL - 1:
        return None

    candidates = _find_marker_candidates(grid, bg, wall_color, nominal_fill_zone)
    if not _place_markers(grid, marker_count, marker_color, candidates):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    ext_depths = _external_depths(new_fill, _scene_meta(grid))
    expected = list(range(seed_count + 1, seed_count + marker_count + 1))
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if ext_depths != expected:
        return None
    if full_ext_depths[-1] <= ext_depths[-1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
