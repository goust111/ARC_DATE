# Variant of b9e38dc0
# Dimension: D4 — 源对象——内部触发像素 (src)
# Change: 半包围区域内最内层的同色连通像素作为洪水填充和辐射的触发源
#   → 精灵边框上的特殊标记像素

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for this D4 variant
#
# variant_design:
#   src_change:
#     old: fill 色 blob 位于半包围区域内部，solver 直接从 blob 做 BFS
#     new: fill 色只出现在墙体闭口桥段上的 1 个 marker 像素，solver 必须先找到 marker，
#          再从 marker 朝开口方向的内侧背景格开始 BFS
#   visual_form:
#     wall: 仍是原题的单个 8 连通半包围墙体，主色不变
#     marker:
#       count: 1
#       placement: 只放在 closed bridge 上，并把该处墙像素改成 fill 色
#       observability: 输入里不再有内部 seed；唯一 fill 色像素贴在墙边，差异肉眼可见
#   concrete_trace:
#     old_flow:
#       - 找到内部 fill blob
#       - 依据 blob 距离选墙组件
#       - 从 blob 洪水填充并经开口向外辐射
#     new_flow:
#       - 找唯一贴着墙的非墙像素作为 marker
#       - 将 marker 视为墙体一部分，恢复完整半包围形状
#       - 用端点继续推 radiate_dir
#       - 选择 marker 沿 radiate_dir 方向的内侧格作为 flood 起点
#     difference:
#       - 旧规则依赖“触发色在区域内部”
#       - 新规则依赖“触发色嵌在边框上，且需要判断哪一侧是内部”
#   complexity_calibration:
#     - marker 只允许 1 个，避免多触点歧义
#     - marker 色直接就是 fill 色，不再额外引入颜色映射维度
#     - 辐射、阴影、障碍物规则维持原题
#   hidden_assumption_audit:
#     - 原 `fill = closest non-wall color to wall` 在这里会拿到 marker 色，但若继续把 marker
#       当 BFS 起点，就会错误地把墙像素当成可通行 seed
#     - 原 `comp_min_dist(fill_pos, comp)` 依赖 seed 不在墙上；变体中必须先把 marker 并回墙
#     - 原障碍物允许离墙 taxicab 距离 2 的对角位置；这会和“贴墙 marker”视觉上接近，
#       所以生成器改为要求障碍与墙的 chessboard 距离 > 1
#   legality:
#     - generator 仅从 bridge 上选择满足 `(u-1, v)` 是 interior 背景格的 wall 像素做 marker
#     - solver 从 marker 的 4 邻格中选与 radiate_dir 点积最大的 passable 格作为内侧起点
#     - 这样可保证不越界、不穿墙，也不会把 closed-end 外侧当作 seed
#   observability_and_cleanliness:
#     - 输入中 fill 色只出现 1 次，且附着在墙上；输出中同色区域会大面积充满内部并外射
#     - 障碍仍是稀疏单点，且不贴墙，marker 与 obstacle 易区分
#   new_solver_assumptions:
#     - 恰有 1 个与墙 8 邻接的非墙非背景像素作为 marker
#     - marker 朝 radiate_dir 方向存在唯一内侧可通行格
#     - generator 已显式保证这两点
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a half-enclosed region using a border marker and radiate the color outward through the opening.

    Input structure:
    - bg is the most common color. The wall is the dominant non-bg color and forms one partial enclosure.
    - A single special marker pixel of the fill color is embedded on the wall's closed border; other non-bg colors are obstacles.
    Rule:
    - Treat the marker as part of the wall to recover the enclosure, then start the flood from the marker's inner side.
    - Flood the interior with the marker color and continue radiating through the opening; obstacles keep their colors and cast shadows behind them.

    Key insight: The endpoints still reveal the opening direction, but the fill source is no longer inside the pocket; it is encoded by the lone wall-touching marker pixel.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)

    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    # Wall remains the most frequent non-background color.
    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall

    marker_candidates: list[tuple[int, int]] = []
    for r, c in zip(*np.where((g != bg) & (g != wall))):
        r = int(r)
        c = int(c)
        if any(0 <= rr < h and 0 <= cc < w and wall_mask[rr, cc] for rr, cc in neighbors8(r, c)):
            marker_candidates.append((r, c))

    assert len(marker_candidates) == 1, "expected exactly one wall-touching marker pixel"
    marker = marker_candidates[0]
    fill = int(g[marker])

    combined_wall = wall_mask.copy()
    combined_wall[marker] = True

    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(combined_wall, connectivity=8)
    ]
    assert comps, "expected at least one wall connected component"
    wall_comp = max(comps, key=len)

    comp_mask = np.zeros_like(combined_wall, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints: list[tuple[int, int]] = []
    for r, c in wall_comp:
        if len(comp_neighbors(r, c)) == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "expected wall component to have exactly two endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))  # inward along the arm
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (sgn(n2[0] - n1[0]), sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [f for _, f in out_info]

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (sgn(dr_sum), sgn(dc_sum))
    assert rad != (0, 0), "expected a non-zero radiate direction"

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

    passable = (~barrier) & ((g == bg) | (g == fill))

    seed_candidates: list[tuple[int, int, int]] = []
    for rr, cc in neighbors4(*marker):
        if 0 <= rr < h and 0 <= cc < w and passable[rr, cc]:
            score = (rr - marker[0]) * rad[0] + (cc - marker[1]) * rad[1]
            seed_candidates.append((score, int(rr), int(cc)))

    assert seed_candidates, "expected an interior start cell next to the marker"
    best_score, seed_r, seed_c = max(seed_candidates)
    assert best_score > 0, "marker must point toward the enclosure interior"
    q = [(seed_r, seed_c)]

    seen = np.zeros_like(passable, dtype=bool)
    seen[seed_r, seed_c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)

    out = g.copy()
    out[filled_mask] = fill

    # Shadow behind obstacles (non-bg, non-fill, non-wall) along radiate direction.
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

    return out


# --------------- generator (D4 variant) ---------------

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

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


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
    """Choose anchor so the wall fits within margins."""
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth):
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


def _obstacle_fits(pos, placed):
    """Check that pos is at least Chebyshev OBSTACLE_MIN_SEP from all placed obstacles."""
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
    """Find bg cells far enough from the wall for obstacle placement."""
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="chessboard")
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

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = [c for c in palette[3:] if c not in (bg, wall_color, fill_color)]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = set(_local_interior(local_wall, depth))
    if not interior:
        return None

    bridge_candidates = [
        (u, v)
        for u, v in local_wall
        if u == depth and (u - 1, v) in interior
    ]
    if not bridge_candidates:
        return None

    marker_u, marker_v = random.choice(bridge_candidates)
    marker_r, marker_c = _map_local(marker_u, marker_v, rad, anchor)
    if grid[marker_r, marker_c] != wall_color:
        return None
    grid[marker_r, marker_c] = fill_color

    if int((grid == fill_color).sum()) != 1:
        return None

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D4 variant input")
