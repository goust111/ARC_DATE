# Variant of b9e38dc0
# Dimension: D4 — 源对象——内部触发像素 (src)
# Change: 半包围区域内最内层的同色连通像素作为洪水填充和辐射的触发源... → 边缘控制像素作为触发源

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D4 variant
# a) Concrete behavior:
#    - The wall keeps the original U/funnel geometry and still determines the radiate direction.
#    - The input no longer contains an internal fill-colored seed blob. Instead there is exactly one
#      border control pixel whose color becomes the eventual fill color.
#    - The control pixel sits on the border that matches the opening direction, but outside the
#      radiating wedge. So it selects the region remotely instead of acting as a literal flood seed.
#    - Small trace example:
#      original: interior fill blob near the closed end -> BFS starts there -> flood escapes through
#      opening -> shadows remove cells behind obstacles.
#      variant: isolated border control pixel on the opening-side edge -> solver reads its edge side
#      and color -> chooses the wall pointing toward that edge -> reconstructs the hidden interior
#      source zone from wall geometry -> performs the same flood/radiate/shadow process with the
#      control color. The output therefore copies the border pixel color into the funnel region.
# b) Complexity calibration:
#    - Only one wall region is kept, so the task stays close to the original.
#    - The visible change is still strong because the input loses the internal seed completely and
#      replaces it with a remote border cue. The control pixel is isolated from the future fill zone,
#      so the old "seed position = source" heuristic no longer works.
# c) Old assumptions that break:
#    - Old code picked the fill color as the non-wall color nearest the wall and used those pixels as
#      BFS seeds. That is invalid now because the only fill-colored pixel is a remote control marker.
#    - Old code chose the closest wall component to those same pixels. The new solver instead keys off
#      the border side of the control marker, then infers interior start cells from wall geometry.
#    - Old passability included the fill-colored seed cells. Here the control pixel must not become a
#      real seed, otherwise the variant collapses back into the old behavior.
# d) Placement legality:
#    - The generator keeps the original wall size/margins.
#    - The control pixel is placed on the radiate-side border, strictly outside the nominal fill zone,
#      so it does not overlap the wall, the fill wedge, or obstacles.
#    - Obstacles are forbidden on the border and on the reconstructed interior seed cells, which are
#      now solver-critical hidden anchors.
# e) Visual observability:
#    - Across examples, the only border non-wall pixel always donates its color to the filled funnel.
#    - Its side matches the opening direction, so a human can relate the remote control marker to the
#      half-enclosed region even though it is not physically connected to the filled area.
# f) Visual cleanliness:
#    - There is only one isolated control pixel, no border clutter, and the wall remains the largest
#      non-background object. Obstacles stay sparse and away from the border.
# g) Color inference:
#    - No color-id ordering is used for semantic decisions. The control color is simply copied into
#      the wall-controlled fill/radiate region, which is directly visible from the examples.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one non-wall border color exists. Generator guarantees one control pixel and
#      keeps obstacles off the border.
#    - Assumption: at least one deepest interior cell remains background. Generator reserves those
#      inferred seed cells and never places obstacles there.
#    - Assumption: the chosen wall points toward the control pixel's border side. Generator always
#      places the control pixel on the opening-side border of the single wall component.


# -- Configuration constants --------------------------------------------------
# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3

# Hidden reconstructed seed zone
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


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
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
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

    return {
        "comp": comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
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


def _border_side(ctrl_pts, h, w):
    r, c = ctrl_pts[0]
    if r == 0:
        return (-1, 0)
    if r == h - 1:
        return (1, 0)
    if c == 0:
        return (0, -1)
    if c == w - 1:
        return (0, 1)
    raise AssertionError("control pixel must lie on border")


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the region enclosed by a wall sprite using a remote border control pixel.

    Input structure:
    - bg is the most common color. The wall is the largest non-bg structure and forms one partial enclosure.
    - Exactly one non-wall border color appears as an isolated control pixel; its border side matches the wall opening.
    - Other non-bg interior pixels are obstacles, not seeds.

    Rule:
    - Use the border control pixel to choose the fill color and the matching wall orientation, then reconstruct the hidden interior start cells from the wall geometry.
    - Flood-fill the enclosed region and radiate outward through the opening with the control color; obstacles keep their color and cast background shadows behind them along the radiate direction.

    Key insight: The border pixel is only a remote trigger. Its side identifies which way the wall opens, and its color is copied into the wall-governed flood/radiate region even though the control pixel itself is not the physical seed.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)

    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected non-background colors"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall component"

    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    ctrl_coords = np.argwhere(border_mask & (g != bg) & (g != wall))
    assert ctrl_coords.size > 0, "expected a border control pixel"
    ctrl_colors = {}
    for r, c in ctrl_coords:
        color = int(g[int(r), int(c)])
        ctrl_colors[color] = ctrl_colors.get(color, 0) + 1
    fill = max(ctrl_colors, key=ctrl_colors.get)
    ctrl_pts = [(int(r), int(c)) for r, c in ctrl_coords if int(g[int(r), int(c)]) == fill]
    ctrl_rad = _border_side(ctrl_pts, h, w)

    geoms = [_component_geometry(comp, g.shape) for comp in comps]
    matches = [geom for geom in geoms if geom["rad"] == ctrl_rad]
    wall_geom = max(matches if matches else geoms, key=lambda geom: len(geom["comp"]))

    seed_positions = _infer_seed_positions(wall_geom["comp"], wall_geom["rad"])
    filled_mask = _flood_from_internal_seeds(g, bg, wall_geom["barrier"], seed_positions)

    out = g.copy()
    out[filled_mask] = fill
    _shadow_fill(out, filled_mask, g, bg, fill, wall, wall_geom["rad"])
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


def _edge_positions(rad, h, w):
    if rad == (-1, 0):
        return [(0, c) for c in range(w)]
    if rad == (1, 0):
        return [(h - 1, c) for c in range(w)]
    if rad == (0, -1):
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _choose_control_pos(grid, nominal_fill_zone, barrier, rad):
    edge = _edge_positions(rad, *grid.shape)
    edge_zone = [p for p in edge if nominal_fill_zone[p[0], p[1]]]

    def edge_gap(p):
        if not edge_zone:
            return 99
        if rad[0] != 0:
            return min(abs(p[1] - q[1]) for q in edge_zone)
        return min(abs(p[0] - q[0]) for q in edge_zone)

    candidates = [p for p in edge if grid[p[0], p[1]] == get_bg(grid) and not barrier[p[0], p[1]] and not nominal_fill_zone[p[0], p[1]]]
    preferred = [p for p in candidates if edge_gap(p) >= 2]
    pool = preferred if preferred else candidates
    if not pool:
        return None
    return random.choice(pool)


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    wall_points = set()
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color
        wall_points.add((r, c))

    wall_geom = _component_geometry(wall_points, grid.shape)
    seed_positions = _infer_seed_positions(wall_geom["comp"], wall_geom["rad"])
    nominal_fill_zone = _flood_from_internal_seeds(grid, bg, wall_geom["barrier"], seed_positions)

    control_pos = _choose_control_pos(grid, nominal_fill_zone, wall_geom["barrier"], rad)
    if control_pos is None:
        return None
    grid[control_pos] = fill_color

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        reserved = np.zeros_like(grid, dtype=bool)
        for r, c in seed_positions:
            reserved[r, c] = True
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone, reserved)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(out == fill_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D4 input")
