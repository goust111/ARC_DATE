# Combo variant of b9e38dc0 (index 6)
# combo_spec: D4/0+D13/0
# Name: 边缘控制像素+像素位置编码方向

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the combo variant
# a) Concrete behavior:
#    - The original task had one active funnel wall and an internal fill-colored blob. The wall's
#      endpoints revealed the opening, so direction came from the wall itself.
#    - This combo removes the internal seed entirely. The input instead contains one isolated border
#      control pixel plus two same-colored wall components. Only the wall nearest to the control
#      pixel is activated, so the solver must first establish a remote control -> region link.
#    - Direction is no longer read from the border side of that control pixel. Instead it is encoded
#      by the control pixel's offset from the chosen wall's center: larger vertical offset means up
#      or down; larger horizontal offset means left or right. Example: a control on the top edge but
#      far to the left of the chosen wall still means "radiate left", not "radiate up".
#    - Small trace:
#      original: internal blob near the closed end -> find nearest wall -> wall opening gives
#      direction -> flood + radiate.
#      combo: isolated border control pixel -> choose nearest wall component -> compare control
#      position to that wall's center to decode direction -> reconstruct hidden interior seed cells
#      from the chosen wall -> flood + radiate with the control color.
# b) Complexity calibration:
#    - Exactly two wall components are used. That is enough to make region selection real without
#      making the board cluttered or the solver branch over many ambiguous candidates.
#    - The control pixel is forced onto a border orthogonal to the true radiate direction. This keeps
#      the D13 change visually meaningful because "which border is touched" is no longer sufficient.
# c) Broken assumptions from the original solver:
#    - There is no interior fill-colored seed to locate, so "nearest non-wall color to the wall" is
#      invalid.
#    - The activated wall can no longer be inferred from those seed pixels; it must be selected by a
#      remote spatial correspondence to the control pixel.
#    - The final direction must not be derived from the wall opening alone, even though the generator
#      still builds a physically compatible opening in that same direction.
# d) Placement legality:
#    - The generator keeps each wall within the original size/margin ranges and ensures the two wall
#      components stay 8-disconnected.
#    - The distractor wall is placed outside the target wall's nominal fill zone so the target flood
#      cannot leak into the distractor interior.
#    - The control pixel is placed on an empty border cell, outside the fill zone, with a clear
#      nearest target wall and a non-tied dominant offset axis.
# e) Visual observability:
#    - Across examples, only one of the two similar wall objects changes in the output, and the one
#      that changes is always the wall spatially claimed by the remote border pixel.
#    - Because the control pixel often sits on a border orthogonal to the radiate direction, a human
#      must compare its position to the chosen wall's center rather than just reading the touched edge.
# f) Visual cleanliness:
#    - There is one control pixel, two wall objects, and sparse obstacles. No border clutter or extra
#      cue colors are introduced.
#    - Obstacles are kept off borders and out of both hidden interior seed pools, so the key cues stay
#      easy to parse.
# g) Color inference:
#    - The control pixel color is directly copied into the activated fill/radiate region. No color-id
#      ranking or invisible numeric ordering is used.
# h) Fusion-specific anti-degeneration checks:
#    - D4 is active because the input never contains an interior seed blob.
#    - D13 is active because the control border side is intentionally orthogonal to the real fill
#      direction, so a border-side heuristic produces a different output on most samples.
# i) New solver assumptions and generator guarantees:
#    - Assumption: there is exactly one non-wall border control pixel. Guaranteed by construction and
#      obstacles never touch the border.
#    - Assumption: the nearest wall center to that control pixel is unique. Guaranteed by candidate
#      filtering during control placement.
#    - Assumption: the dominant offset axis is unique. Guaranteed by a margin between |dr| and |dc|.
#    - Assumption: the chosen wall's true opening matches the decoded direction so the geometry is
#      physically realizable. Guaranteed when building the target wall.


# -- Configuration constants --------------------------------------------------
GRID_H_VERT = (13, 18)
GRID_H_HORIZ = (11, 18)
GRID_W = (17, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

DISTRACTOR_DEPTH = (3, 4)
DISTRACTOR_GAP = (4, 5)

OPEN_END_MARGIN = 3
SEED_DEPTH_LOOKBACK = 3

NUM_WALLS = 2
WALL_GAP = 1
CTRL_AXIS_MARGIN = 1
CTRL_TARGET_MARGIN = 2

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 500


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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _build_small_local_wall(rad, h, w):
    if rad[0]:
        max_depth = min(DISTRACTOR_DEPTH[1], h - 6)
        if max_depth < DISTRACTOR_DEPTH[0]:
            return None
        depth = random.randint(DISTRACTOR_DEPTH[0], max_depth)
        gap = DISTRACTOR_GAP[0] if w < 18 else random.randint(*DISTRACTOR_GAP)
    else:
        max_depth = min(DISTRACTOR_DEPTH[1], w - 6)
        if max_depth < DISTRACTOR_DEPTH[0]:
            return None
        depth = random.randint(DISTRACTOR_DEPTH[0], max_depth)
        gap = DISTRACTOR_GAP[0] if h < 14 else random.randint(*DISTRACTOR_GAP)

    if gap == 4:
        left_bottom, right_bottom = 1, 3
    else:
        left_bottom, right_bottom = 1, 4

    left = _make_arm(depth, 0, left_bottom, 0, tip_12=False)
    right = _make_arm(depth, gap, right_bottom, 0, tip_12=False)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

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
    return random.choice(_anchor_candidates(rad, h, w, depth, gap))


def _anchor_candidates(rad, h, w, depth, gap):
    if rad == (-1, 0):
        rows = range(OPEN_END_MARGIN, h - depth - 2)
        cols = range(0, w - gap)
    elif rad == (1, 0):
        rows = range(depth + 2, h - 3)
        cols = range(0, w - gap)
    elif rad == (0, -1):
        rows = range(0, h - gap)
        cols = range(OPEN_END_MARGIN, w - depth - 2)
    else:
        rows = range(0, h - gap)
        cols = range(depth + 2, w - 3)
    return [(r, c) for r in rows for c in cols]


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


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _comp_mask(shape, comp):
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _comp_neighbors(comp_mask, r, c):
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _component_geometry(comp, shape):
    comp = set((int(r), int(c)) for r, c in comp)
    comp_mask = _comp_mask(shape, comp)
    endpoints = []
    for r, c in comp:
        if len(_comp_neighbors(comp_mask, r, c)) == 1:
            endpoints.append((r, c))
    assert endpoints, "expected wall component endpoints"

    def outward_dir(ep):
        n1 = _comp_neighbors(comp_mask, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _comp_neighbors(comp_mask, *n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [f for _, f in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    barrier = comp_mask.copy()

    def add_ray(start, d):
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
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
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    center = ((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0)
    return {
        "comp": comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
        "center": center,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _anchor_from_comp(comp, rad):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if rad == (-1, 0):
        return min(rows), min(cols)
    if rad == (1, 0):
        return max(rows), min(cols)
    if rad == (0, -1):
        return min(rows), min(cols)
    return min(rows), max(cols)


def _to_local(comp, rad):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    local = set()
    for r, c in comp:
        if rad == (-1, 0):
            local.add((r - r_min, c - c_min))
        elif rad == (1, 0):
            local.add((r_max - r, c - c_min))
        elif rad == (0, -1):
            local.add((c - c_min, r - r_min))
        else:
            local.add((c_max - c, r - r_min))
    return local


def _infer_seed_positions(comp, rad):
    local_wall = _to_local(comp, rad)
    depth = max(u for u, _ in local_wall)
    gap = max(v for _, v in local_wall)
    interior = _local_interior(local_wall, depth, gap)
    assert interior, "expected wall interior"

    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not pool:
        pool = adjacent if adjacent else interior
    max_u = max(u for u, _ in pool)
    local_seed = [p for p in pool if p[0] == max_u]

    anchor = _anchor_from_comp(comp, rad)
    return [_map_local(u, v, rad, anchor) for u, v in local_seed]


def _flood_from_internal_seeds(g, bg, barrier, seed_positions):
    passable = (~barrier) & (g == bg)
    seen = np.zeros_like(passable, dtype=bool)
    q = []
    for r, c in seed_positions:
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and passable[r, c]:
            seen[r, c] = True
            q.append((r, c))
    assert q, "expected at least one background interior seed cell"
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _shadow_fill(out, filled_mask, g, bg, fill, wall, rad):
    obstacles = (g != bg) & (g != fill) & (g != wall)

    def shadow_process(coords):
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

    if rad == (0, 0):
        return

    dr, dc = rad
    h, w = g.shape
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    elif dr == dc:
        for k in range(-(w - 1), h):
            coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)
    else:
        for k in range(h + w - 1):
            coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)


def _border_control_points(g, bg, wall):
    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    pts = np.argwhere(border_mask & (g != bg) & (g != wall))
    return [(int(r), int(c)) for r, c in pts]


def _direction_from_control(control, center):
    dr = float(control[0]) - float(center[0])
    dc = float(control[1]) - float(center[1])
    if abs(dr) > abs(dc):
        return (_sgn(int(round(dr))) or _sgn(dr), 0)
    return (0, _sgn(int(round(dc))) or _sgn(dc))


def _component_distance_to_control(control, geom):
    cr, cc = control
    r0, c0, r1, c1 = geom["bbox"]
    # Distance to bbox is more visually stable than centroid distance for selecting the target wall.
    rr = min(max(cr, r0), r1)
    cc2 = min(max(cc, c0), c1)
    return abs(cr - rr) + abs(cc - cc2)


def _choose_target_geometry(geoms, control):
    scored = sorted(
        ((int(_component_distance_to_control(control, geom)), -len(geom["comp"]), idx, geom) for idx, geom in enumerate(geoms)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    return scored[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one funnel region using a remote border control pixel whose offset encodes direction.

    Input structure:
    - bg is the most common color. The wall color forms two separated partial enclosures of the same type.
    - Exactly one isolated non-wall border pixel is the control marker; its color is the future fill color.
    - Other non-bg interior pixels are fixed obstacles and never act as seeds.

    Rule:
    - Choose the wall component that the border control pixel remotely targets, then decode the radiate direction from the control pixel's position relative to that wall's center.
    - Reconstruct hidden interior seed cells for the chosen wall, flood-fill that enclosure, and continue the fill outward in the decoded direction; obstacles remain unchanged and cast background shadows behind them.

    Key insight: The border pixel carries two jobs at once. Its spatial correspondence selects which wall is active, and its offset from that wall's center determines the fill direction, so neither the border side alone nor the wall opening alone is the full rule.
    """
    g = np.array(grid, copy=True)
    bg = get_bg(g)

    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected non-background colors"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall component"

    ctrl_pts = _border_control_points(g, bg, wall)
    assert len(ctrl_pts) == 1, "expected exactly one border control pixel"
    control = ctrl_pts[0]
    fill = int(g[control[0], control[1]])

    geoms = [_component_geometry(comp, g.shape) for comp in comps]
    wall_geom = _choose_target_geometry(geoms, control)
    decoded_rad = _direction_from_control(control, wall_geom["center"])

    seed_positions = _infer_seed_positions(wall_geom["comp"], wall_geom["rad"])
    filled_mask = _flood_from_internal_seeds(g, bg, wall_geom["barrier"], seed_positions)

    out = g.copy()
    out[filled_mask] = fill
    _shadow_fill(out, filled_mask, g, bg, fill, wall, decoded_rad)
    return out


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone, reserved_mask):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    eligible = (grid == bg) & far_from_wall & (~border_mask) & (~reserved_mask)
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    placed = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _bundle_from_local(local_wall, depth, gap, rad, anchor, shape, bg):
    wall_points = {_map_local(u, v, rad, anchor) for u, v in local_wall}
    geom = _component_geometry(wall_points, shape)
    if geom["rad"] != rad:
        return None

    interior_local = _local_interior(local_wall, depth, gap)
    interior_global = {_map_local(u, v, rad, anchor) for u, v in interior_local}
    seed_positions = _infer_seed_positions(geom["comp"], geom["rad"])
    nominal_fill_zone = _flood_from_internal_seeds(np.full(shape, bg, dtype=int), bg, geom["barrier"], seed_positions)
    return {
        "wall_points": wall_points,
        "geom": geom,
        "interior": interior_global,
        "seed_positions": seed_positions,
        "nominal_fill_zone": nominal_fill_zone,
    }


def _bbox_with_margin(points, h, w, margin):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return (
        max(0, min(rows) - margin),
        max(0, min(cols) - margin),
        min(h - 1, max(rows) + margin),
        min(w - 1, max(cols) + margin),
    )


def _bbox_overlap(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _render_bundle_into_grid(grid, bundle, wall_color):
    for r, c in bundle["wall_points"]:
        grid[r, c] = wall_color


def _make_target_bundle(h, w, bg, rad):
    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchors = _anchor_candidates(rad, h, w, depth, gap)
    if rad == (-1, 0):
        anchors.sort(key=lambda p: (p[0], p[1]))
    elif rad == (1, 0):
        anchors.sort(key=lambda p: (-p[0], p[1]))
    elif rad == (0, -1):
        anchors.sort(key=lambda p: (p[1], p[0]))
    else:
        anchors.sort(key=lambda p: (-p[1], p[0]))
    keep = max(12, (2 * len(anchors)) // 3)
    anchors = anchors[:keep]
    random.shuffle(anchors)
    for anchor in anchors[:40]:
        bundle = _bundle_from_local(local_wall, depth, gap, rad, anchor, (h, w), bg)
        if bundle is not None:
            bundle["rad"] = rad
            bundle["local_wall"] = local_wall
            bundle["depth"] = depth
            bundle["gap"] = gap
            bundle["anchor"] = anchor
            return bundle
    return None


def _place_distractor_bundle(h, w, bg, target_bundle, control_pos=None):
    target_bbox = _bbox_with_margin(target_bundle["wall_points"], h, w, WALL_GAP)
    target_ctrl_dist = None
    if control_pos is not None:
        target_ctrl_dist = _component_distance_to_control(control_pos, target_bundle["geom"])
    for _ in range(120):
        rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        built = _build_small_local_wall(rad, h, w)
        if built is None:
            continue
        local_wall, depth, gap = built
        anchors = _anchor_candidates(rad, h, w, depth, gap)
        random.shuffle(anchors)
        for anchor in anchors:
            bundle = _bundle_from_local(local_wall, depth, gap, rad, anchor, (h, w), bg)
            if bundle is None:
                continue
            bbox = _bbox_with_margin(bundle["wall_points"], h, w, WALL_GAP)
            if _bbox_overlap(bbox, target_bbox):
                continue
            if any(target_bundle["nominal_fill_zone"][r, c] for r, c in bundle["wall_points"]):
                continue
            if any(target_bundle["nominal_fill_zone"][r, c] for r, c in bundle["interior"]):
                continue
            if control_pos is not None:
                if _component_distance_to_control(control_pos, bundle["geom"]) < target_ctrl_dist + CTRL_TARGET_MARGIN:
                    continue
            bundle["rad"] = rad
            return bundle
    return None


def _edge_positions(edge, h, w):
    if edge == "top":
        return [(0, c) for c in range(w)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(w)]
    if edge == "left":
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _control_candidates(grid, target_bundle, other_bundles):
    h, w = grid.shape
    target_geom = target_bundle["geom"]
    rad = target_geom["rad"]
    if rad == (-1, 0):
        allowed_edges = ["left", "right", "bottom"]
    elif rad == (1, 0):
        allowed_edges = ["left", "right", "top"]
    elif rad == (0, -1):
        allowed_edges = ["top", "bottom", "right"]
    else:
        allowed_edges = ["top", "bottom", "left"]
    target_center = target_geom["center"]
    geoms = [target_geom] + [bundle["geom"] for bundle in other_bundles]

    candidates = []
    for edge in allowed_edges:
        for p in _edge_positions(edge, h, w):
            r, c = p
            if grid[r, c] != get_bg(grid):
                continue
            if target_bundle["nominal_fill_zone"][r, c]:
                continue
            decoded = _direction_from_control(p, target_center)
            if decoded != rad:
                continue
            dr = abs(float(r) - target_center[0])
            dc = abs(float(c) - target_center[1])
            if abs(dr - dc) < CTRL_AXIS_MARGIN:
                continue
            dists = sorted(_component_distance_to_control(p, geom) for geom in geoms)
            if _component_distance_to_control(p, target_geom) != dists[0]:
                continue
            if len(dists) >= 2 and dists[1] - dists[0] < CTRL_TARGET_MARGIN:
                continue
            candidates.append(p)
    random.shuffle(candidates)
    return candidates


def _gen_attempt():
    target_rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if target_rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    target_bundle = _make_target_bundle(h, w, bg, target_rad)
    if target_bundle is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    _render_bundle_into_grid(grid, target_bundle, wall_color)
    control_choices = _control_candidates(grid, target_bundle, [])
    if not control_choices:
        return None

    distractor_bundle = None
    control_pos = None
    for cand in control_choices[:30]:
        bundle = _place_distractor_bundle(h, w, bg, target_bundle, cand)
        if bundle is not None:
            distractor_bundle = bundle
            control_pos = cand
            break
    if distractor_bundle is None or control_pos is None:
        return None

    _render_bundle_into_grid(grid, distractor_bundle, wall_color)
    grid[control_pos] = fill_color

    n_obstacles = random.randint(1, MAX_OBSTACLES)
    if not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    reserved = np.zeros_like(grid, dtype=bool)
    for bundle in [target_bundle, distractor_bundle]:
        for r, c in bundle["seed_positions"]:
            reserved[r, c] = True
        for r, c in bundle["interior"]:
            reserved[r, c] = True
    reserved[control_pos[0], control_pos[1]] = True
    candidates, fill_candidates = _find_obstacle_candidates(
        grid, bg, wall_color, target_bundle["nominal_fill_zone"], reserved
    )
    _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if any(new_fill[r, c] for r, c in distractor_bundle["interior"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
