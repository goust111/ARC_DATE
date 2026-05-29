# Combo variant of b9e38dc0 (index 34)
# combo_spec: D4/1+D15/2
# Name: 多源并行填充+渐变距离编码
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task had one interior fill color, so one BFS region became one flat color.
#    Here the enclosure contains 2-3 distinct interior source pixels. They all launch flood-fill
#    through the same opening in parallel. Each reachable background cell is colored by its
#    shortest-path distance to the nearest source: distance 1/2/3/4+ map to a fixed 4-color heatmap.
#    Example in a small U-shape: if the old task had one seed near the closed end, every reachable
#    pixel turned uniformly red; now two seeds on the left/right side create two sets of concentric
#    distance bands whose Voronoi boundary runs through the middle of the filled region.
# b) Complexity calibration:
#    The main visual signal must stay readable. The generator therefore keeps the wall family and
#    opening/radiation geometry from the original task, but uses only 2-3 singleton sources and a
#    4-band global gradient. This is richer than a flat fill yet still inferable from 2-4 examples.
# c) Broken assumptions from the original solver:
#    - "the closest non-wall color is the single fill color" is no longer valid.
#    - "newly filled pixels inherit the seed color" is no longer valid.
#    - "all non-wall non-bg pixels except one fill color are obstacles" is no longer valid.
#    The rewritten solver therefore classifies non-wall components by geometry: components touching
#    the interior wall boundary are sources; the others are obstacles.
# d) Placement legality:
#    Sources are chosen from interior cells adjacent to the wall near the closed end, with pairwise
#    separation so they do not overlap. Obstacles are only placed on background cells, away from the
#    wall and sources. Wall placement and ray margins remain inherited from the original generator.
# e) Visual observability:
#    Human-solvable cues come from repeated structure: multiple colored source markers always sit
#    inside the pocket, while the output always uses the same 4 heatmap colors in expanding bands.
#    No ordering depends on numeric color ids.
# f) Visual cleanliness:
#    Obstacles are kept sparse and the gradient palette is fixed, so the output reads as a coherent
#    heatmap rather than a collage of unrelated colors.
# g) Inferability of color change:
#    The gradient is global and distance-binned, not source-specific by arbitrary lookup. Across
#    examples the same near/mid/far heatmap colors recur, giving enough support to infer the rule.

from collections import deque
import itertools
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


GRADIENT_COLORS = [1, 2, 3, 4]  # distance 1, 2, 3, 4+
RESERVED_OUTPUT_COLORS = set(GRADIENT_COLORS)
ROLE_COLOR_POOL = [0, 5, 6, 7, 8, 9]


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _distance_to_color(dist: int) -> int:
    if dist <= 1:
        return GRADIENT_COLORS[0]
    if dist == 2:
        return GRADIENT_COLORS[1]
    if dist == 3:
        return GRADIENT_COLORS[2]
    return GRADIENT_COLORS[3]


def _wall_geometry(g: np.ndarray) -> dict:
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    comps = components_mask(g == wall, connectivity=8)
    assert len(comps) > 0, "expected a wall component"
    wall_comp = max(
        ([tuple(map(int, p)) for p in comp] for comp in comps),
        key=len,
    )
    wall_set = set(wall_comp)

    comp_mask = np.zeros_like(g, dtype=bool)
    for r, c in wall_set:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_set if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) >= 2, "expected at least two wall endpoints"

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
    tip_12 = [flag for _, flag in out_info]
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

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    return {
        "bg": bg,
        "wall": wall,
        "wall_set": wall_set,
        "comp_mask": comp_mask,
        "barrier": barrier,
        "rad": rad,
    }


def _classify_non_wall_components(
    g: np.ndarray, bg: int, wall: int, wall_set: set[tuple[int, int]]
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = g.shape
    source_components: list[list[tuple[int, int]]] = []
    obstacle_mask = np.zeros((h, w), dtype=bool)
    colors = [int(v) for v in np.unique(g) if int(v) not in (bg, wall)]
    for color in colors:
        for comp in components_mask(g == color, connectivity=4):
            pts = [tuple(map(int, p)) for p in comp]
            min_dist = min(
                abs(r - wr) + abs(c - wc)
                for r, c in pts
                for wr, wc in wall_set
            )
            touches_border = any(r in (0, h - 1) or c in (0, w - 1) for r, c in pts)
            if min_dist <= 1 and not touches_border:
                source_components.append(pts)
            else:
                for r, c in pts:
                    obstacle_mask[r, c] = True
    return source_components, obstacle_mask


def _solve_details(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape
    geo = _wall_geometry(g)
    bg = geo["bg"]
    wall = geo["wall"]
    wall_set = geo["wall_set"]
    barrier = geo["barrier"]
    rad = geo["rad"]

    source_components, obstacle_mask = _classify_non_wall_components(g, bg, wall, wall_set)
    assert source_components, "expected at least one interior source component"

    source_mask = np.zeros((h, w), dtype=bool)
    dist = np.full((h, w), -1, dtype=np.int16)
    owner = np.full((h, w), -1, dtype=np.int16)
    q = deque()

    for sid, comp in enumerate(source_components):
        for r, c in comp:
            source_mask[r, c] = True
            if dist[r, c] == -1:
                dist[r, c] = 0
                owner[r, c] = sid
                q.append((r, c))

    passable = (~barrier) & ((g == bg) | source_mask)
    seen = np.zeros((h, w), dtype=bool)
    for r, c in zip(*np.where(source_mask)):
        rr = int(r)
        cc = int(c)
        if passable[rr, cc]:
            seen[rr, cc] = True

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if not passable[rr, cc] or seen[rr, cc]:
                continue
            seen[rr, cc] = True
            dist[rr, cc] = dist[r, c] + 1
            owner[rr, cc] = owner[r, c]
            q.append((rr, cc))

    filled_mask = seen & (g == bg)
    final_mask = filled_mask.copy()

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if final_mask[r, c]:
                started = True
            if started and obstacle_mask[r, c]:
                blocked = True
                continue
            if blocked and final_mask[r, c]:
                final_mask[r, c] = False

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
    for r, c in zip(*np.where(final_mask)):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = _distance_to_color(int(dist[rr, cc]))

    return {
        "out": out,
        "bg": bg,
        "wall": wall,
        "rad": rad,
        "dist": dist,
        "owner": owner,
        "filled_mask": final_mask,
        "source_mask": source_mask,
        "obstacle_mask": obstacle_mask,
        "source_components": source_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a semi-enclosed region from several interior source pixels, then encode the reachable area as a distance heatmap.

    Input structure:
    - The background is the most common color; the most common non-background color is a partial wall with one opening.
    - Inside the wall pocket sit 2-3 small source markers of different colors, each touching the interior wall boundary.
    - Any other non-background, non-wall pixels are obstacles and remain fixed.

    Rule:
    - All interior source markers launch flood-fill in parallel through the pocket and out through the opening, using the same wall rays as the original task.
    - Every newly reached background cell is recolored by its shortest-path distance to the nearest source:
      distance 1 -> color 1, distance 2 -> color 2, distance 3 -> color 3, distance 4+ -> color 4.
    - Obstacles still cast background-colored shadows behind them along the outward radiation direction.

    Key insight:
    - The wall geometry still determines the only legal filled corridor, but the output color no longer comes from one seed.
      Instead, the visible bands are a multi-source distance field clipped by the opening rays and obstacle shadows.
    """
    return _solve_details(grid)["out"]


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

# Sources
SOURCE_COUNT_WEIGHTS = [2, 2, 2, 3, 3]
SOURCE_DEPTH_LOOKBACK = 3
SOURCE_MIN_SEP = 2
MIN_SOURCE_CONTRIB = 8

# Obstacles
MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.45
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
MIN_GRADIENT_COLORS = 3
GEN_RETRIES = 400


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


def _local_interior(wall, depth):
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


def _pick_source_positions(interior, wall_set, depth):
    adjacent = [
        p for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    pool = [p for p in adjacent if p[0] >= max(1, depth - SOURCE_DEPTH_LOOKBACK)]
    if len(pool) < 2:
        return None

    random.shuffle(pool)
    wanted = random.choice(SOURCE_COUNT_WEIGHTS)
    best_combo = None
    best_score = -1
    for k in [min(wanted, len(pool)), min(2, len(pool))]:
        if k < 2:
            continue
        for combo in itertools.combinations(pool, k):
            if any(
                abs(a[0] - b[0]) + abs(a[1] - b[1]) < SOURCE_MIN_SEP
                for a, b in itertools.combinations(combo, 2)
            ):
                continue
            score = sum(
                abs(a[0] - b[0]) + abs(a[1] - b[1])
                for a, b in itertools.combinations(combo, 2)
            )
            if score > best_score:
                best_score = score
                best_combo = combo
        if best_combo is not None and len(best_combo) == k:
            break
    if best_combo is None:
        return None
    return list(best_combo)


def _obstacle_fits(pos, placed, source_cells):
    r, c = pos
    if any(abs(r - sr) + abs(c - sc) <= 1 for sr, sc in source_cells):
        return False
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone, source_mask):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    src_coords = {(int(r), int(c)) for r, c in zip(*np.where(source_mask))}
    all_coords = [p for p in all_coords if p not in src_coords]
    fill_coords = [p for p in fill_coords if p not in src_coords]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(grid, candidates, fill_candidates, obstacle_colors, source_cells):
    if not obstacle_colors:
        return
    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles == 0:
        return

    placed = []
    need_fill = fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB
    if need_fill:
        for p in fill_candidates:
            if _obstacle_fits(p, placed, source_cells):
                placed.append(p)
                break
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if _obstacle_fits(p, placed, source_cells):
            placed.append(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _active_source_counts(details: dict) -> list[int]:
    owner = details["owner"]
    filled_mask = details["filled_mask"]
    counts = []
    for sid in range(len(details["source_components"])):
        counts.append(int(((owner == sid) & filled_mask).sum()))
    return counts


def _gradient_color_count(out: np.ndarray, filled_mask: np.ndarray) -> int:
    if not filled_mask.any():
        return 0
    return len({int(v) for v in out[filled_mask]})


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    role_colors = random.sample(ROLE_COLOR_POOL, len(ROLE_COLOR_POOL))
    bg = role_colors[0]
    wall_color = role_colors[1]
    source_colors = role_colors[2:5]
    obstacle_colors = role_colors[5:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
    wall_set = set(local_wall)
    source_positions = _pick_source_positions(interior, wall_set, depth)
    if source_positions is None:
        return None

    if len(source_positions) > len(source_colors):
        return None
    for color, (u, v) in zip(source_colors, source_positions):
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = color

    try:
        nominal = _solve_details(grid)
    except Exception:
        return None

    if len(nominal["source_components"]) < 2:
        return None
    nominal_counts = _active_source_counts(nominal)
    if sum(c >= MIN_SOURCE_CONTRIB for c in nominal_counts) < 2:
        return None

    source_cells = [tuple(map(int, p)) for p in zip(*np.where(nominal["source_mask"]))]
    candidates, fill_candidates = _find_obstacle_candidates(
        grid, bg, wall_color, nominal["filled_mask"], nominal["source_mask"]
    )
    _place_obstacles(grid, candidates, fill_candidates, obstacle_colors, source_cells)

    try:
        details = _solve_details(grid)
    except Exception:
        return None

    filled_mask = details["filled_mask"]
    out = details["out"]
    active_counts = _active_source_counts(details)
    if int(filled_mask.sum()) < MIN_NEW_FILL or not _touches_border(filled_mask):
        return None
    if _gradient_color_count(out, filled_mask) < MIN_GRADIENT_COLORS:
        return None
    if sum(c >= MIN_SOURCE_CONTRIB for c in active_counts) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
