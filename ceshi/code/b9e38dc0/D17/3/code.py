# Variant of b9e38dc0
# Dimension: D17 — 精灵边框形态多样性 (obstacle)
# Change: 半包围边框为简单的U形/C形/倒U形等规则形状... → 多色边框标识不同功能面

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D17 variant
# a) Concrete behavior:
#    The wall is no longer a single-color partial enclosure. It is a single 8-connected sprite made
#    from three fixed colors: blue=absorbing side, red=reflective side, orange=closed bridge.
#    The seed still flood-fills the interior and exits through the opening, but the two endpoint rays
#    now depend on endpoint color. Red endpoints keep the original slanted boundary ray. Blue endpoints
#    drop the sideways component and project a straight ray along the overall radiate direction. This
#    makes the outside fill region visibly asymmetric while preserving the original "flood inside +
#    radiate outside + obstacle shadows" pipeline.
#    Small example: if the opening points upward, the original task emits two diagonal rays and creates
#    a symmetric funnel. In this variant, a red left arm still emits an up-left slanted ray, while a
#    blue right arm emits a straight upward ray from its endpoint. The filled output is clipped by one
#    diagonal side and one vertical side, so it differs from the original on the same geometry.
# b) Complexity calibration:
#    Only two arm functions are used, and every sample contains both. That is enough repetition for a
#    human to infer the color semantics without introducing too many branches. The bridge gets its own
#    neutral color so the multi-color wall is visually explicit.
# c) Old-code assumptions that break:
#    The original solver assumes the wall is "the most frequent non-bg color"; that fails once the wall
#    is split across colors. Here the solver instead treats {blue, red, orange} as the wall palette and
#    finds the wall component by connectivity. The original obstacle mask also excluded one wall color;
#    that would misclassify the other wall colors as obstacles, so obstacle detection now excludes the
#    full wall palette. The original endpoint-ray loop assumed identical behavior for both endpoints;
#    that is replaced with a per-endpoint branch keyed by endpoint color.
# d) Placement legality:
#    The generator still uses the original local-wall construction and anchor logic so the wall stays
#    in-bounds and preserves exactly two endpoints. Left/right/bridge labels are assigned before global
#    placement, and extra padded wall pixels inherit a nearby label so the component remains connected.
#    Obstacles are still placed on background cells away from the wall. Colors reserved for wall
#    functions are never reused by bg, fill, or obstacles, which keeps parsing unambiguous.
# e) Visual observability:
#    The generator rejects samples where the functional-wall output matches the legacy all-slanted-ray
#    output. This prevents silent regression to the original task and ensures the colored wall actually
#    changes the shape a human sees.
# f) Visual cleanliness:
#    The new behavior changes the outer boundary shape but keeps the number of object types small:
#    background, three wall colors, fill, and a few singleton obstacles. The filled region stays bounded
#    by two rays, so the scene does not devolve into whole-grid spill.
# g) Color-rule inferability:
#    Color ids are not used for sorting or hidden ranking. The mapping is structural and fixed:
#    blue-side -> straight boundary ray, red-side -> slanted boundary ray, orange -> closed bridge.
#    Generator samples swap which arm is blue/red so the color-function relation is visible directly.
#
# New solver assumptions introduced by this variant and enforced by the generator:
# - The three wall-function colors appear only in the wall.
# - The wall component closest to the seed is the intended enclosure.
# - Each endpoint belongs to an arm color (blue or red), never the bridge color.
# - The fill color is the non-wall color closest to the wall; obstacles are placed farther from the wall.


ABSORB_COLOR = 1   # blue side -> straight boundary ray along radiate_dir
REFLECT_COLOR = 2  # red side -> original slanted boundary ray
BRIDGE_COLOR = 7   # orange bridge -> neutral closed end
WALL_COLORS = (ABSORB_COLOR, REFLECT_COLOR, BRIDGE_COLOR)


def _solve(grid: np.ndarray, functional_walls: bool) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)

    wall_mask = np.isin(g, WALL_COLORS)
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one multicolor wall component"

    non_bg = [int(v) for v in vals if int(v) != bg and int(v) not in WALL_COLORS]
    assert non_bg, "expected at least one non-wall color"

    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)

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

    fill = min(non_bg, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return g

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
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (sgn(n2[0] - n1[0]), sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (sgn(dr_sum), sgn(dc_sum))
    assert rad != (0, 0), "opening direction must be cardinal"

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
        ep_color = int(g[ep])
        if not functional_walls or ep_color == REFLECT_COLOR:
            if is_12:
                add_ray_12(ep, d)
            else:
                add_ray(ep, d)
        elif ep_color == ABSORB_COLOR:
            add_ray(ep, rad)
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
    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (~wall_mask)

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

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill a multicolor half-enclosure and radiate outward using wall-function colors.

    Input structure:
    - Background is the most common color.
    - A single 8-connected wall sprite is built from three fixed wall colors: blue side, red side, and orange bridge.
    - A non-wall seed blob sits inside the enclosure; other non-wall non-seed pixels are singleton obstacles.

    Rule:
    - Fill spreads through background cells reachable from the seed, first inside the enclosure and then out through the opening.
    - The two wall endpoints project different outside boundary rays according to endpoint color:
      red keeps the original slanted ray, blue projects a straight ray along the opening direction.
    - Obstacles preserve their colors and erase fill behind them along the global radiate direction.

    Key insight:
    - The wall must be parsed as one multicolor object, then each endpoint's color determines which kind of
      boundary ray constrains the exterior fill region.
    """
    return _solve(grid, functional_walls=True)


# --------------- generator (o5x) ---------------

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

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 4]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
MIN_FUNCTIONAL_DELTA = 6
GEN_RETRIES = 400

FREE_COLORS = [0, 3, 4, 5, 6, 8, 9]


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


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


def _choose_added_label(
    p: tuple[int, int],
    labels: dict[tuple[int, int], str],
    depth: int,
    gap: int,
) -> str:
    u, v = p
    if u == depth:
        return "bridge"
    counts: dict[str, int] = {}
    for q in neighbors8(u, v):
        if q in labels:
            lab = labels[q]
            counts[lab] = counts.get(lab, 0) + 1
    if counts:
        best = max(counts.values())
        opts = [lab for lab, cnt in counts.items() if cnt == best]
        if len(opts) == 1:
            return opts[0]
        if "bridge" in opts and u >= depth - 1:
            return "bridge"
    return "left" if v * 2 < gap else "right"


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

    left = set(_make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12))
    right = set(_make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12))
    bridge = {(depth, v) for v in range(left_bottom, right_bottom + 1)}

    pts = set(left) | set(right) | set(bridge)
    labels: dict[tuple[int, int], str] = {}
    for p in left:
        labels[p] = "left"
    for p in right:
        labels[p] = "right"
    for p in bridge:
        labels[p] = "bridge"

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
                labels[p] = _choose_added_label(p, labels, depth, gap)
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap, labels


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
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
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


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid: np.ndarray, bg: int, nominal_fill_zone: np.ndarray):
    wall_mask = np.isin(grid, WALL_COLORS)
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _wall_color_for_label(label: str, reflect_side: str) -> int:
    if label == "bridge":
        return BRIDGE_COLOR
    if label == reflect_side:
        return REFLECT_COLOR
    return ABSORB_COLOR


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap, labels = built
    anchor = _place_anchor(rad, h, w, depth, gap)
    reflect_side = random.choice(["left", "right"])

    bg = random.choice(FREE_COLORS)
    remaining = [c for c in FREE_COLORS if c != bg]
    fill_color = random.choice(remaining)
    remaining = [c for c in remaining if c != fill_color]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = _wall_color_for_label(labels[(u, v)], reflect_side)

    interior = _local_interior(local_wall, depth)
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
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not remaining:
        return None
    if n_obstacles:
        n_obs_colors = random.randint(1, min(2, len(remaining)))
        obstacle_colors = random.sample(remaining, n_obs_colors)
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        legacy_out = _solve(grid, functional_walls=False)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    functional_delta = int(np.sum(out != legacy_out))
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if functional_delta < MIN_FUNCTIONAL_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid D17 b9e38dc0 input")
