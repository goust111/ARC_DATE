# Combo variant of b9e38dc0 (index 13)
# combo_spec: D2/3+D10/0
# Name: 重力下落+先移后填

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Base task logic fills the basin and radiates through the wall opening.
#    - This combo adds isolated obstacle pixels that first fall straight downward, leaving bg behind.
#    - Only after the fall do we flood-fill + radiate, so fallen pixels can now sit inside the later
#      radiate zone and block / shadow it. In the old order, that same pixel would hit already-filled
#      cells sooner and stop higher, producing a different final pattern.
# b) Complexity calibration:
#    - Gravity is fixed to one direction (down), while radiate direction is sampled from up/left/right.
#      This keeps the new rule visible instead of collapsing back into "move along radiate direction".
#    - Movers are limited to 1-3 isolated singleton pixels with repeated behavior across examples.
# c) Assumption audit:
#    - `wall` is still the most frequent non-bg color; generator keeps wall size 17-25 and obstacles sparse.
#    - `fill` is still the non-wall color closest to the wall; generator keeps falling pixels at taxicab
#      distance > 2 from the wall in the input so they do not steal the fill role.
#    - Movement uses only singleton components; generator avoids touching obstacle pixels so "isolated"
#      remains unambiguous.
# d) Placement legality:
#    - Wall placement reuses the original geometry bounds.
#    - Each falling pixel is placed above a bg run in one column, so it can descend without overlap.
#    - Chosen landing cells are bg in the pre-move grid and lie in the nominal future fill zone.
# e) Visual observability:
#    - Inputs show clear vertical drops for singleton pixels, while the wall still indicates a separate
#      opening/radiation direction. Across a few samples, humans can see gravity is always downward.
# f) Visual cleanliness:
#    - The generator uses few movers, distinct from the wall/fill colors, and keeps them separated.
#    - We avoid dense static clutter so the fallen blockers and the lost fill area remain legible.
# g) Inference support:
#    - The fused effect is validated against a legacy "fill first, then fall" baseline; accepted samples
#      must produce a measurable difference, so the generator does not degenerate into visually inert cases.


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

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

PALETTE_SIZE = (4, 5)
GEN_RETRIES = 400
MIN_NEW_FILL = 20
MIN_DROP = 3
MAX_MOVERS = 3
MIN_FILL_LOSS = 2
MIN_COMBO_DIFF = 2
START_WALL_DIST = 3

RADIATE_CHOICES = [(-1, 0), (0, -1), (0, 1)]


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


def _grow_seed_blob(seed, interior_set):
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


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _identify_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = get_bg(g)
    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert vals, "expected at least one non-background color"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(wall_mask, connectivity=8)
    ]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in vals if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(color))]
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
    fill_pos = [(int(r), int(c)) for r, c in np.argwhere(g == int(fill))]
    assert fill_pos, "expected at least one fill pixel"

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

    def comp_neighbors(r: int, c: int):
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall endpoints"

    def outward_dir(ep):
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
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
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    return {
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
    }


def _isolated_pixels(grid: np.ndarray, bg: int, wall: int, fill: int):
    movers = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, wall, fill):
            continue
        for comp in components_mask(grid == color, connectivity=8):
            pts = [(int(r), int(c)) for r, c in comp]
            if len(pts) == 1:
                movers.append(pts[0])
    return sorted(movers, reverse=True)


def _move_isolated_pixels_down(grid: np.ndarray, bg: int, wall: int, fill: int):
    g = np.array(grid, copy=True)
    h, _w = g.shape
    movers = _isolated_pixels(g, bg, wall, fill)
    colors = {(r, c): int(g[r, c]) for r, c in movers}
    for r, c in movers:
        g[r, c] = bg

    moves = []
    for r, c in movers:
        rr = int(r)
        while rr + 1 < h and int(g[rr + 1, c]) == int(bg):
            rr += 1
        g[rr, c] = colors[(r, c)]
        moves.append(((int(r), int(c)), (int(rr), int(c)), colors[(r, c)]))
    return g, moves


def _build_barrier(scene: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    comp_mask = scene["comp_mask"]
    rad = scene["rad"]
    barrier = comp_mask.copy()

    def add_ray(start, d, skip=0):
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

    for ep, d, is_12 in zip(scene["endpoints"], scene["out_dirs"], scene["tip_12"]):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d, skip=0)
    return barrier


def _apply_fill_and_shadow(grid: np.ndarray, scene: dict):
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = scene["bg"]
    wall = scene["wall"]
    fill = scene["fill"]
    rad = scene["rad"]

    fill_pos = [(int(r), int(c)) for r, c in np.argwhere(g == int(fill))]
    if not fill_pos:
        return g, np.zeros_like(g, dtype=bool)

    barrier = _build_barrier(scene, g.shape)
    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q = fill_pos[:]
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)
    out = g.copy()
    out[filled_mask] = fill

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

    if rad != (0, 0):
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

    return out, filled_mask


def _legacy_order_output(grid: np.ndarray) -> np.ndarray:
    scene = _identify_scene(grid)
    filled_first, _ = _apply_fill_and_shadow(grid, scene)
    moved_after, _ = _move_isolated_pixels_down(filled_first, scene["bg"], scene["wall"], scene["fill"])
    return moved_after


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move isolated obstacle pixels downward first, then fill the wall pocket and radiate through its opening.

    Input structure:
    - bg is the most common color. The wall is the most frequent non-bg color and forms a partial enclosure.
    - A fill-colored seed sits inside that enclosure. Other non-bg colors may appear as isolated singleton pixels.

    Rule:
    - First, every isolated non-wall, non-fill singleton falls straight down under gravity until the cell below is blocked or the grid bottom is reached, leaving bg behind.
    - After those falls, flood-fill from the seed inside the enclosure and continue the fill outward through the opening; the fallen pixels now remain in place and can block or shadow the radiating fill.

    Key insight: The opening direction still comes from the wall endpoints, but singleton motion no longer follows that direction; it is a separate downward preprocessing step whose new resting positions affect the later radiation.
    """
    g = np.array(grid, copy=True)
    scene = _identify_scene(g)
    moved, _moves = _move_isolated_pixels_down(g, scene["bg"], scene["wall"], scene["fill"])
    out, _filled_mask = _apply_fill_and_shadow(moved, scene)
    return out


def _pick_seed_blob(local_wall, depth, gap):
    interior = _local_interior(local_wall, depth, gap)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None
    return _grow_seed_blob(random.choice(seed_pool), set(interior))


def _candidate_fallers(base_grid, nominal_fill_zone, wall_color):
    h, w = base_grid.shape
    bg = get_bg(base_grid)
    wall_dist = distance_transform_cdt(base_grid != wall_color, metric="taxicab")
    candidates = []
    for c in range(w):
        for r in range(1, h):
            if int(base_grid[r, c]) != int(bg) or not nominal_fill_zone[r, c]:
                continue
            if r != h - 1 and int(base_grid[r + 1, c]) == int(bg):
                continue

            start = None
            for s in range(r - 1, -1, -1):
                if int(base_grid[s, c]) != int(bg):
                    break
                if wall_dist[s, c] >= START_WALL_DIST and not nominal_fill_zone[s, c]:
                    start = s
            if start is None:
                continue
            drop = r - start
            if drop < MIN_DROP:
                continue
            candidates.append(
                {
                    "start": (int(start), int(c)),
                    "dest": (int(r), int(c)),
                    "drop": int(drop),
                }
            )
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (-item["drop"], item["start"][1], item["start"][0]))
    return candidates


def _choose_fallers(candidates):
    chosen = []
    used_cols = set()
    target_count = random.randint(1, MAX_MOVERS)
    for cand in candidates:
        sr, sc = cand["start"]
        dr, dc = cand["dest"]
        if sc in used_cols:
            continue
        if any(max(abs(sr - r0), abs(sc - c0)) < 2 for r0, c0 in [x["start"] for x in chosen]):
            continue
        if any(max(abs(dr - r0), abs(dc - c0)) < 2 for r0, c0 in [x["dest"] for x in chosen]):
            continue
        chosen.append(cand)
        used_cols.add(sc)
        if len(chosen) >= target_count:
            break
    return chosen


def _gen_attempt():
    rad = random.choice(RADIATE_CHOICES)
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
    if not obstacle_colors:
        return None

    base_grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        base_grid[r, c] = wall_color

    blob = _pick_seed_blob(local_wall, depth, gap)
    if blob is None:
        return None
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if int(base_grid[r, c]) != int(bg):
            return None
        base_grid[r, c] = fill_color

    try:
        scene = _identify_scene(base_grid)
        nominal_out, _ = _apply_fill_and_shadow(base_grid, scene)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (base_grid == bg)
    if int(nominal_fill_zone.sum()) < MIN_NEW_FILL or not _touches_border(nominal_fill_zone):
        return None

    candidates = _candidate_fallers(base_grid, nominal_fill_zone, wall_color)
    chosen = _choose_fallers(candidates)
    if not chosen:
        return None

    grid = np.array(base_grid, copy=True)
    for idx, cand in enumerate(chosen):
        sr, sc = cand["start"]
        grid[sr, sc] = obstacle_colors[idx % len(obstacle_colors)]

    try:
        out = main(grid)
        legacy = _legacy_order_output(grid)
        _moved_grid, moves = _move_isolated_pixels_down(grid, scene["bg"], scene["wall"], scene["fill"])
    except Exception:
        return None

    moved_dests = {(rr, cc) for (_src, (rr, cc), _color) in moves}
    if not any(nominal_fill_zone[r, c] for r, c in moved_dests):
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    fill_loss = int(((nominal_out == fill_color) & (out != fill_color)).sum())
    combo_diff = int((out != legacy).sum())
    moved_any = any(src != dst for src, dst, _color in moves)
    if not moved_any or fill_loss < MIN_FILL_LOSS or combo_diff < MIN_COMBO_DIFF:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
