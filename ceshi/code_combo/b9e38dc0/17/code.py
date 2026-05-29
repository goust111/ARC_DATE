# Combo variant of b9e38dc0 (index 17)
# combo_spec: D6/2+D14/1
# Name: 锥形辐射+瓶颈通道
#
# Design analysis
# ---
# variant_spec:
#   concrete_geometry:
#     - 墙仍是最常见的非背景色、单个 8 连通组件，但不再是直接敞开的 U/C 口，而是瓶身 + 3 格直颈。
#     - 颈部两侧是平行短墙，颈内通道宽度只取 1 或 2；两端点位于颈口外端，仍是整个墙组件唯一的两个 degree-1 像素。
#     - 外侧辐射不再由两条固定斜率直线界定，而是从颈口两端向外延伸成对称阶梯边界：
#       每隔固定 2 格向外推进时，边界再额外向侧面扩 1 格，于是形成明显的离散锥形。
#   behavior_trace:
#     - 原题: 种子填满腔室后，穿过整面开口，外侧区域宽度基本由两条直边界一次性决定。
#     - 仅 D6/2: 颜色必须先挤过窄颈，但出了颈口后仍是直边扇形。
#     - 本融合: 颜色先以瓶颈宽度穿过通道；离开颈口后的前几步仍保持窄束，随后边界按固定节奏逐层外扩。
#       因而输出上能直接看到“窄束 -> 逐层变宽的锥束”，而不是“颈口之后立刻全宽展开”。
#   complexity_calibration:
#     - 只保留两个主自由度: 通道宽度 {1,2} 与腔室张口宽度。
#     - 锥形扩展速率固定为每 2 步每侧扩 1 格，不再引入多档速率，避免样例过少时无法归纳。
#     - 颈长固定为 3，使“瓶颈约束”和“出口后扩散”在视觉上可分离。
#   implicit_assumption_audit:
#     - 原 solver 假设端点附近的墙走向就是外侧边界走向；这里失效，因为端点只标记通道出口。
#     - 新 solver 改为: 用两个端点仅推断主辐射方向，再用固定的阶梯扩展规则构造外侧边界。
#     - 这要求输入满足: 端点必须位于同一颈口，主方向必须是上下左右之一，颈口外侧确实留有可扩散空间。
#     - 障碍阴影仍沿主辐射轴逐列或逐行裁掉填充，因此障碍必须远离墙，避免与颈部几何混淆。
#   placement_legality:
#     - 生成器只构造一个 8 连通墙组件，固定两端点，确保通道和瓶身不自交。
#     - 锚点选择保证瓶身不越界，颈口朝向边界至少保留 5 格，让锥形有足够展开距离。
#     - 侧向也保留额外边距，减少锥形一出门就撞侧墙的退化样本。
#   visual_observability:
#     - 输入中可直接看见瓶颈；输出中可直接看见颈外束宽随距离分层增加。
#     - 如果去掉锥形边界，样本会退化成 D6/2；如果去掉瓶颈，样本又接近原题。因此两者都被强制出现。
#   cleanliness:
#     - 墙、填充、障碍分色，障碍仍为散点且不放入窄颈，避免和边界台阶混叠。
#     - 锥形由连续阶梯边界限定，不叠加额外标记，视觉上保持单一主结构。
#   generator_enforced_assumptions:
#     - 目标墙组件恰有两个 degree-1 端点，且它们就是颈口出口。
#     - 主辐射方向始终是基于两端点内侧邻居求和得到的四个正交方向之一。
#     - 锥形扩展固定为 period=2，因此 solver 无需从颜色或尺寸推断速率。
#     - 与原题一样，离墙最近的非墙色就是填充色，障碍不参与目标组件识别。

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


CORRIDOR_LEN = 3
CONE_EXPAND_PERIOD = 2


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a bottle-like chamber, squeeze through its neck, then widen outward as a cone.

    Input structure:
    - The background is the most common color. The wall is the most frequent non-background color and forms one 8-connected bottle or funnel with a short straight neck.
    - A fill-colored seed sits inside the deeper chamber near the closed end. Any remaining non-background colors are obstacle pixels.

    Rule:
    - Flood-fill every reachable background cell inside the chamber and through the neck.
    - Past the neck mouth, the fill keeps moving along the opening direction, but its allowed width grows stepwise: both side boundaries move sideways by one cell every two forward steps, producing a cone.
    - Obstacle pixels stay unchanged and cast shadows along the main radiate axis, turning filled cells behind them back to background.

    Key insight: The endpoints no longer define immediate full-width radiation; they identify a narrow outlet, and the exterior fill must respect a fixed bottleneck width first, then a gradually widening cone.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall component"

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

    inward_dirs = []
    for ep in endpoints:
        nb = comp_neighbors(*ep)[0]
        inward_dirs.append((_sgn(nb[0] - ep[0]), _sgn(nb[1] - ep[1])))

    dr_sum = sum(d[0] for d in inward_dirs)
    dc_sum = sum(d[1] for d in inward_dirs)
    rad = (-_sgn(dr_sum), -_sgn(dc_sum))
    assert rad in {(-1, 0), (1, 0), (0, -1), (0, 1)}, "expected cardinal radiate direction"

    lateral = (0, 1) if rad[0] != 0 else (1, 0)
    if rad[0] != 0:
        endpoints.sort(key=lambda p: (p[1], p[0]))
    else:
        endpoints.sort(key=lambda p: (p[0], p[1]))

    barrier = comp_mask.copy()

    def add_cone_boundary(start: tuple[int, int], side_sign: int) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            dr, dc = rad
            if step % CONE_EXPAND_PERIOD == 0:
                dr += side_sign * lateral[0]
                dc += side_sign * lateral[1]
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    add_cone_boundary(endpoints[0], side_sign=-1)
    add_cone_boundary(endpoints[1], side_sign=1)

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
    if dr != 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    else:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])

    return out


# --------------- generator ---------------

GRID_H_VERT = (12, 18)
GRID_H_HORIZ = (9, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (6, 8)
DEPTH_HORIZ = (6, 8)
BASE_SPAN = (6, 7)
CORRIDOR_WIDTHS = (1, 2)

OPEN_END_MARGIN = 5
SIDE_MARGIN = 2

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 500


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _make_arm(depth: int, start_v: int, target_v: int) -> list[tuple[int, int]]:
    sign = _sgn(target_v - start_v)
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        chosen_rows = [1]
        if n_side_steps > 1:
            chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
        chosen_rows.sort()

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in chosen_rows:
            v += sign
        pts.append((u, v))
    assert v == target_v
    return pts


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
    if rad[0] != 0:
        max_depth = min(DEPTH_VERT[1], h - OPEN_END_MARGIN - 4)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - OPEN_END_MARGIN - 4)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)

    body_depth = depth - CORRIDOR_LEN
    if body_depth < 3:
        return None

    corridor_width = random.choice(CORRIDOR_WIDTHS)
    candidates = []
    for base_span in range(BASE_SPAN[0], BASE_SPAN[1] + 1):
        widen_total = base_span - (corridor_width + 1)
        if widen_total < 2:
            continue
        for left_out in range(1, widen_total):
            right_out = widen_total - left_out
            if right_out < 1:
                continue
            if max(left_out, right_out) > body_depth:
                continue
            candidates.append((base_span, left_out, right_out))
    if not candidates:
        return None

    base_span, left_out, right_out = random.choice(candidates)
    left_bottom, right_bottom = 0, base_span
    neck_left = left_bottom + left_out
    neck_right = right_bottom - right_out

    left_corridor = [(u, neck_left) for u in range(CORRIDOR_LEN + 1)]
    right_corridor = [(u, neck_right) for u in range(CORRIDOR_LEN + 1)]
    left_body = [(u + CORRIDOR_LEN, v) for u, v in _make_arm(body_depth, neck_left, left_bottom)]
    right_body = [(u + CORRIDOR_LEN, v) for u, v in _make_arm(body_depth, neck_right, right_bottom)]

    pts = set(left_corridor) | set(right_corridor) | set(left_body) | set(right_body)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    if len(_degree1_pixels(pts)) != 2:
        return None
    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    return pts, depth, base_span


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, base_span: int):
    if rad == (-1, 0):
        lo_c = SIDE_MARGIN
        hi_c = w - base_span - 1 - SIDE_MARGIN
        if lo_c > hi_c:
            return None
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(lo_c, hi_c)
    if rad == (1, 0):
        lo_c = SIDE_MARGIN
        hi_c = w - base_span - 1 - SIDE_MARGIN
        if lo_c > hi_c:
            return None
        return random.randint(depth + 2, h - OPEN_END_MARGIN - 1), random.randint(lo_c, hi_c)
    if rad == (0, -1):
        lo_r = SIDE_MARGIN
        hi_r = h - base_span - 1 - SIDE_MARGIN
        if lo_r > hi_r:
            return None
        return random.randint(lo_r, hi_r), random.randint(OPEN_END_MARGIN, w - depth - 3)
    lo_r = SIDE_MARGIN
    hi_r = h - base_span - 1 - SIDE_MARGIN
    if lo_r > hi_r:
        return None
    return random.randint(lo_r, hi_r), random.randint(depth + 2, w - OPEN_END_MARGIN - 1)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        vals = by_row.get(u)
        if vals is None or len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
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


def _find_obstacle_candidates(grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles: int, candidates, fill_candidates, obstacle_colors, grid: np.ndarray) -> None:
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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, base_span = built
    anchor = _place_anchor(rad, h, w, depth, base_span)
    if anchor is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
    interior_set = set(interior)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), interior_set)
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
    raise RuntimeError("failed to generate valid cone+bottleneck b9e38dc0 input")
