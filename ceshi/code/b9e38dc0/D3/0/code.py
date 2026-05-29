# Variant of b9e38dc0
# Dimension: D3 — 辐射轨迹涂色 (trajectory)
# Change: 辐射射线使用源像素颜色连续涂色；孤立像素移动后轨迹变为背景色... → 辐射轨迹使用周期性颜色图案

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior and trace:
#    The original task paints every reachable bg cell with a single fill color.
#    In this variant, only the outward-radiating part changes: the trapped
#    interior still becomes the seed color, but once the fill crosses the opening,
#    each depth band along the radiate direction uses a periodic AABB cycle:
#    fill, fill, wall, wall, fill, fill, ...
#    Example with downward radiation:
#      - Original: rows at depths 0..5 outside the opening all become FFFFFF.
#      - Variant: depths 0,1 -> fill color; depths 2,3 -> wall color;
#                 depths 4,5 -> fill color again.
#    Obstacles still cast shadows after reachability is computed, so some later
#    bands may be erased back to bg.
#
# b) Complexity calibration:
#    The pattern uses exactly two visible colors already present in the input:
#    wall color and seed color. The period is fixed globally (AABB), so solvers
#    only need one new regularity, not a family of unrelated cases.
#
# c) Hidden-assumption audit:
#    - The original `out[filled_mask] = fill` assumed every painted cell shared
#      one color. That is no longer valid; painting must be delayed until after
#      obstacle shadows are resolved, then assigned per depth band.
#    - The shadow code depends on a boolean reachability mask, not on colors.
#      That remains valid, so we preserve the scanline logic and only recolor
#      surviving painted cells afterward.
#    - The generator previously validated only `out == fill_color`. That would
#      wrongly discard wall-colored bands in the new outputs, so validation now
#      counts all newly painted non-bg cells.
#
# d) Placement legality:
#    The wall construction, endpoint geometry, and obstacle placement all remain
#    unchanged except for one added visibility guard: the opening must have
#    enough free depth to the border for at least one full AABB repeat.
#
# e) Visual observability:
#    The rule is visible from the output alone because the radiated wedge is the
#    only place where wall color reappears away from the original wall. The
#    interior stays solid fill color, making the transition at the opening easy
#    to spot across examples.
#
# f) Visual neatness:
#    No extra markers are introduced. The pattern is drawn as clean parallel
#    bands orthogonal to the radiate direction, so it does not compete with the
#    wall outline or obstacle singleton pixels.
#
# g) Color-inference quality:
#    The new colors are not derived from ids or sorting. They are the two
#    structurally meaningful colors already exposed by the puzzle: enclosure
#    border color and enclosed seed color.
# ---------------------------------------------------------------------------


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
PATTERN_REPEAT_DEPTH = 5  # need depths 0..5 to show fill-fill/wall-wall/fill-fill

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
MIN_RADIATE_PIXELS = 12
MIN_WALL_BAND_PIXELS = 6
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300

# Periodic trajectory pattern
PATTERN_BLOCK = 2  # AABB: two fill-color bands, then two wall-color bands


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _projection(r: int, c: int, rad: tuple[int, int]) -> int:
    return int(r) * int(rad[0]) + int(c) * int(rad[1])


def _degree1_pixels(points):
    """Return degree-1 pixels (exactly one 8-neighbor in the set)."""
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
    """Build one zigzag arm from (0, start_v) toward (depth, target_v)."""
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
    """Construct wall shape in local (u, v) coordinates. Returns (pts, depth, gap) or None."""
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
    """Map local (u, v) to grid (r, c) given radiate direction and anchor."""
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    """Choose anchor so the wall fits within margins and leaves visible pattern depth."""
    if rad == (-1, 0):
        row_lo = max(OPEN_END_MARGIN, PATTERN_REPEAT_DEPTH)
        row_hi = h - depth - 3
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        row_lo = depth + 2
        row_hi = h - 1 - PATTERN_REPEAT_DEPTH
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        col_lo = max(OPEN_END_MARGIN, PATTERN_REPEAT_DEPTH)
        col_hi = w - depth - 3
        if col_lo > col_hi:
            return None
        return random.randint(0, h - gap - 1), random.randint(col_lo, col_hi)
    col_lo = depth + 2
    col_hi = w - 1 - PATTERN_REPEAT_DEPTH
    if col_lo > col_hi:
        return None
    return random.randint(0, h - gap - 1), random.randint(col_lo, col_hi)


def _local_interior(wall, depth, gap):
    """Return bg cells strictly between the two wall arms in local coords."""
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
    """Grow a small 4-connected blob from seed within interior cells."""
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


def _obstacle_fits(pos, placed):
    """Check that pos is at least Chebyshev OBSTACLE_MIN_SEP from all placed obstacles."""
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
    """Find bg cells far enough from wall for obstacle placement, using distance transform."""
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    """Place up to n_obstacles on the grid, preferring some in the fill zone."""
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


def _touches_border(mask):
    """Check if any True cell in mask lies on the grid border."""
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _solve_core(grid: np.ndarray):
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)

    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall

    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color (fill color)"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
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
    if not fill_pos:
        meta = {
            "bg": int(bg),
            "wall": int(wall),
            "fill": int(fill),
            "rad": (0, 0),
            "pattern": "AABB",
            "paint_mask": np.zeros_like(g, dtype=bool),
            "radiated_mask": np.zeros_like(g, dtype=bool),
            "wall_band_mask": np.zeros_like(g, dtype=bool),
            "depth_map": np.full_like(g, -1, dtype=int),
        }
        return g, meta

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
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

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints (degree-1 pixels)"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
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

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
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

    for ep, (d, is_12) in zip(endpoints, out_info):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill))
    q = fill_pos[:]
    seen = np.zeros_like(passable, dtype=bool)
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

    out = g.copy()
    out[filled_mask] = fill

    depth_map = np.full((h, w), -1, dtype=int)
    radiated_mask = np.zeros_like(filled_mask, dtype=bool)
    wall_band_mask = np.zeros_like(filled_mask, dtype=bool)
    if rad != (0, 0):
        open_proj = max(_projection(r, c, rad) for r, c in endpoints)
        rows, cols = np.indices((h, w))
        depth_map = rows * rad[0] + cols * rad[1] - open_proj
        radiated_mask = filled_mask & (depth_map >= 0)
        wall_band_mask = radiated_mask & (((depth_map // PATTERN_BLOCK) % 2) == 1)
        out[wall_band_mask] = wall

    meta = {
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "rad": (int(rad[0]), int(rad[1])),
        "pattern": "AABB",
        "paint_mask": filled_mask.copy(),
        "radiated_mask": radiated_mask,
        "wall_band_mask": wall_band_mask,
        "depth_map": depth_map,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a funnel-shaped enclosure, then send a periodically colored trail out through its opening.

    Input structure:
    - bg is the most common color. The dominant non-bg color is a single wall component forming a partial enclosure.
    - A nearby non-wall color marks the seed blob inside the enclosure. Any other scattered non-bg pixels are obstacles.

    Rule:
    - Flood-fill every reachable bg cell connected to the seed inside the enclosure and through the open end.
    - Keep the enclosed interior in the seed color, but recolor the outward radiating part with a repeating AABB band pattern made from seed color and wall color. Obstacles still punch background shadows behind them along the radiate direction.

    Key insight:
    - The wall endpoints determine both the opening direction and where the periodic trajectory begins; once the fill crosses that opening, color depends on depth-band position rather than a single uniform paint color.
    """
    return _solve_core(grid)[0]


def _gen_attempt():
    """Single attempt to generate a valid input grid."""
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)
    if anchor is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

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

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        _, nominal_meta = _solve_core(grid)
    except Exception:
        return None
    nominal_fill_zone = nominal_meta["paint_mask"] & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out, meta = _solve_core(grid)
    except Exception:
        return None

    paint_mask = meta["paint_mask"] & (grid == bg)
    radiated_mask = meta["radiated_mask"] & paint_mask
    wall_band_mask = meta["wall_band_mask"] & paint_mask
    depth_map = meta["depth_map"]

    if int(paint_mask.sum()) < MIN_NEW_FILL or not _touches_border(paint_mask):
        return None
    if int(radiated_mask.sum()) < MIN_RADIATE_PIXELS:
        return None
    if int(wall_band_mask.sum()) < MIN_WALL_BAND_PIXELS:
        return None
    if not np.any(radiated_mask):
        return None
    if int(depth_map[radiated_mask].max()) < PATTERN_REPEAT_DEPTH:
        return None

    new_wall = (grid == bg) & (out == wall_color)
    new_fill = (grid == bg) & (out == fill_color)
    if not new_wall.any() or not new_fill.any():
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D3 input")
