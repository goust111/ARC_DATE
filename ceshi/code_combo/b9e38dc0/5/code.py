# Combo variant of b9e38dc0 (index 5)
# combo_spec: D3/2+D5/1
# Name: 射线长度等于填充面积+像素计数终止
#
# Design analysis
# - Concrete behavior: keep the original wall/opening detection and the wedge-shaped flood region,
#   but stop the outward propagation after L layers, where
#   L = min(newly filled interior bg cells, input seed-pixel count).
#   Example: if the enclosure contains 8 interior cells and 5 are already seed-colored, then the
#   solver first fills the remaining 3 interior bg cells, so the outward wedge extends exactly
#   min(3, 5) = 3 layers past the opening. If only 2 seed pixels were present instead, the same
#   geometry would extend 2 layers. This is the core fusion: both counts must be computed.
# - Complexity calibration: the generator restricts the enclosure interior to 5-11 cells and picks
#   seed sizes on either side of half-full, so samples alternate between "area-limited" and
#   "count-limited" cases. That makes the min-rule visible in a few examples instead of collapsing
#   to one always-dominant variable.
# - Assumption audit: the original solver assumed rays always run to the border. That is no longer
#   valid, so the barrier rays still define the corridor, but passability is capped by a forward
#   depth mask. The original fill-color detection, wall selection, endpoint detection, and obstacle
#   shadow logic remain valid under the new rule.
# - Placement legality: the generator chooses the opening margin after the target ray length is
#   known, ensuring enough empty space for the full intended outward run without crossing the grid
#   edge. Seeds are grown only on interior cells; obstacles are kept out of the interior so they do
#   not silently mutate the controlling counts.
# - Visual observability: humans can count the seed blob in the input and the newly added interior
#   cells in the output, then compare that smaller number to the visible outward depth. Both control
#   regimes are sampled, so the fused rule is observable across examples.
# - Visual cleanliness: obstacles are sparse and preferentially placed in the exterior radiate zone.
#   This preserves the original shadow effect without cluttering the enclosure where the two counts
#   must be read.
# - New solver assumptions guaranteed by the generator: exactly one relevant wall component with two
#   degree-1 endpoints, enough opening margin for the intended ray length, and a seed blob size that
#   matches the selected control regime.

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


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

SEED_DEPTH_LOOKBACK = 3
INTERIOR_SIZE_RANGE = (5, 11)
MIN_RAY_LEN = 2
MAX_SEED_SIZE = 6

MAX_OBSTACLES = 3
OBSTACLE_IN_FILL_PROB = 0.45
OBSTACLE_MIN_SEP = 2

PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows: list[int] = []
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


def _place_anchor(
    rad: tuple[int, int],
    h: int,
    w: int,
    depth: int,
    gap: int,
    min_open_run: int,
):
    if rad == (-1, 0):
        lo, hi = min_open_run, h - depth - 3
        if lo > hi:
            return None
        return random.randint(lo, hi), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        lo, hi = depth + 2, h - 1 - min_open_run
        if lo > hi:
            return None
        return random.randint(lo, hi), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        lo, hi = min_open_run, w - depth - 3
        if lo > hi:
            return None
        return random.randint(0, h - gap - 1), random.randint(lo, hi)
    lo, hi = depth + 2, w - 1 - min_open_run
    if lo > hi:
        return None
    return random.randint(0, h - gap - 1), random.randint(lo, hi)


def _local_interior(wall: set[tuple[int, int]], depth: int, gap: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
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


def _grow_seed_blob(
    seed: tuple[int, int],
    interior_set: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
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


def _bfs_mask(starts: list[tuple[int, int]], passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q = [(int(r), int(c)) for r, c in starts if passable[int(r), int(c)]]
    for r, c in q:
        seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _comp_neighbors(r: int, c: int, comp_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _outward_dir(ep: tuple[int, int], comp_mask: np.ndarray) -> tuple[tuple[int, int], bool]:
    n1 = _comp_neighbors(*ep, comp_mask)[0]
    d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
    d_use = d1
    is_12 = False
    n2s = [p for p in _comp_neighbors(*n1, comp_mask) if p != ep]
    if n2s:
        n2 = n2s[0]
        d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
        if (d1[0] == 0) ^ (d1[1] == 0):
            if d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
    return (-d_use[0], -d_use[1]), is_12


def _build_barrier(
    comp_mask: np.ndarray,
    endpoints: list[tuple[int, int]],
    out_dirs: list[tuple[int, int]],
    tip_12: list[bool],
    rad: tuple[int, int],
) -> np.ndarray:
    h, w = comp_mask.shape
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

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)
    return barrier


def _analyze_scene(grid: np.ndarray) -> dict:
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
    assert other_colors, "expected at least one non-wall color"

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
    assert fill_pos, "expected fill seed pixels"

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

    endpoints = [(r, c) for r, c in wall_comp if len(_comp_neighbors(r, c, comp_mask)) == 1]
    assert len(endpoints) >= 2, "expected at least two degree-1 wall pixels"

    out_info = [_outward_dir(ep, comp_mask) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [f for _, f in out_info]

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad != (0, 0), "expected a cardinal radiate direction"

    barrier = _build_barrier(comp_mask, endpoints, out_dirs, tip_12, rad)
    passable_base = (~barrier) & ((g == bg) | (g == fill))

    rows = np.arange(h, dtype=int)[:, None]
    cols = np.arange(w, dtype=int)[None, :]
    open_proj = max(rad[0] * r + rad[1] * c for r, c in endpoints)
    forward = rad[0] * rows + rad[1] * cols - open_proj

    return {
        "grid": g,
        "bg": bg,
        "fill": fill,
        "wall": wall,
        "fill_pos": fill_pos,
        "rad": rad,
        "endpoints": endpoints,
        "forward": forward,
        "passable_base": passable_base,
    }


def _combo_metrics(grid: np.ndarray) -> dict:
    scene = _analyze_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    fill = scene["fill"]
    forward = scene["forward"]

    interior_passable = scene["passable_base"] & (forward <= 0)
    interior_seen = _bfs_mask(scene["fill_pos"], interior_passable)
    interior_area = int((interior_seen & (g == bg) & (forward < 0)).sum())
    source_count = int(len(scene["fill_pos"]))
    ray_len = min(interior_area, source_count)

    out = main(grid)
    exterior_fill = (out == fill) & (g == bg) & (forward > 0)
    actual_max_depth = int(forward[exterior_fill].max()) if exterior_fill.any() else 0

    if interior_area < source_count:
        controller = "area"
    elif source_count < interior_area:
        controller = "count"
    else:
        controller = "tie"

    return {
        "interior_area": interior_area,
        "source_count": source_count,
        "ray_len": ray_len,
        "controller": controller,
        "actual_max_depth": actual_max_depth,
        "exterior_fill_cells": int(exterior_fill.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a partial enclosure, then radiate outward only for a numerically capped distance.

    Input structure:
    - The background is the dominant color. The most frequent non-background color is a wall sprite
      forming a single open enclosure. A nearby non-wall color marks the source blob inside it.
    - Any remaining non-background colors are obstacles; they stay fixed and can shadow the outward
      flood in the wall's radiate direction.

    Rule:
    - First flood-fill all reachable background cells inside the enclosure with the source color.
    - Let A be the number of newly filled background cells strictly inside the enclosure, and let S
      be the number of source-colored pixels already present in the input. The outward wedge through
      the opening extends for exactly min(A, S) forward layers, not all the way to the edge.
    - Obstacles still block already-started outward fill behind them along the same radiate axis.

    Key insight:
    - The wall geometry still reveals the opening direction, but the stopping distance is no longer
      geometric. It is the smaller of two interior counts: new enclosed fill area and source size.
    """
    scene = _analyze_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    fill = scene["fill"]
    wall = scene["wall"]
    rad = scene["rad"]
    forward = scene["forward"]

    interior_passable = scene["passable_base"] & (forward <= 0)
    interior_seen = _bfs_mask(scene["fill_pos"], interior_passable)
    interior_area = int((interior_seen & (g == bg) & (forward < 0)).sum())
    source_count = int(len(scene["fill_pos"]))
    ray_len = min(interior_area, source_count)

    limited_passable = scene["passable_base"] & (forward <= ray_len)
    seen = _bfs_mask(scene["fill_pos"], limited_passable)
    filled_mask = seen & (g == bg)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    visible_fill = filled_mask.copy()

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if visible_fill[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and visible_fill[r, c]:
                out[r, c] = bg
                visible_fill[r, c] = False

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(g.shape[1]):
            rows = range(g.shape[0]) if dr > 0 else range(g.shape[0] - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(g.shape[0]):
            cols = range(g.shape[1]) if dc > 0 else range(g.shape[1] - 1, -1, -1)
            shadow_process([(r, c) for c in cols])

    return out


def _choose_seed_plan(interior_size: int):
    modes = []

    count_hi = min(MAX_SEED_SIZE, (interior_size - 1) // 2)
    if count_hi >= MIN_RAY_LEN:
        modes.append(("count", random.randint(MIN_RAY_LEN, count_hi)))

    area_lo = max(interior_size - MAX_SEED_SIZE, interior_size // 2 + 1)
    area_hi = min(MAX_SEED_SIZE, interior_size - MIN_RAY_LEN)
    if area_lo <= area_hi:
        modes.append(("area", random.randint(area_lo, area_hi)))

    if not modes:
        return None
    return random.choice(modes)


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_zone: np.ndarray,
    exterior_zone: np.ndarray,
):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    exterior_coords = [
        (int(r), int(c))
        for r, c in zip(*np.where(eligible & nominal_fill_zone & exterior_zone))
    ]
    random.shuffle(all_coords)
    random.shuffle(exterior_coords)
    return all_coords, exterior_coords


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    exterior_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_fill = random.randint(0, min(n_obstacles, len(exterior_candidates)))
    if exterior_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)

    for p in exterior_candidates:
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


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built

    interior = _local_interior(local_wall, depth, gap)
    interior_size = len(interior)
    if not (INTERIOR_SIZE_RANGE[0] <= interior_size <= INTERIOR_SIZE_RANGE[1]):
        return None

    seed_plan = _choose_seed_plan(interior_size)
    if seed_plan is None:
        return None
    target_mode, target_seed_size = seed_plan
    target_ray_len = min(target_seed_size, interior_size - target_seed_size)
    if target_ray_len < MIN_RAY_LEN:
        return None

    anchor = _place_anchor(rad, h, w, depth, gap, target_ray_len + 1)
    if anchor is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = None
    interior_set = set(interior)
    for _ in range(40):
        seed = random.choice(seed_pool)
        blob = _grow_seed_blob(seed, interior_set, target_seed_size)
        if blob is not None:
            break
    if blob is None:
        return None

    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_metrics = _combo_metrics(grid)
        nominal_out = main(grid)
        nominal_scene = _analyze_scene(grid)
    except Exception:
        return None

    if nominal_metrics["controller"] != target_mode:
        return None
    if nominal_metrics["ray_len"] != target_ray_len:
        return None

    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)
    exterior_zone = nominal_scene["forward"] > 0

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, exterior_candidates = _find_obstacle_candidates(
            grid,
            bg,
            wall_color,
            nominal_fill_zone,
            exterior_zone,
        )
        _place_obstacles(n_obstacles, candidates, exterior_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    if metrics["controller"] != target_mode:
        return None
    if metrics["ray_len"] != target_ray_len:
        return None
    if metrics["actual_max_depth"] != metrics["ray_len"]:
        return None
    if metrics["exterior_fill_cells"] <= 0:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < interior_size:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo b9e38dc0 input")
