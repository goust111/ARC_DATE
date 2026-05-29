# Variant of b9e38dc0
# Dimension: D6 — 半包围精灵的形状结构 (obstacle)
# Change: 半包围区域由U形或C形边框围成，有一个明确的开口方向（上/下/左/右）... → 包围区域带有瓶颈通道

# Design analysis
# ---
# variant_spec:
#   focus_change:
#     from: "宽开口的半包围墙体，填充直接从整条开口面向外辐射"
#     to: "内部腔室只通过固定长度的窄颈通道连接外部；填充必须先穿过通道，再在外侧扇形扩散"
#   concrete_geometry:
#     wall:
#       - 墙仍是最常见的非背景色、单个 8 连通组件。
#       - 墙体现在是 bottle/funnel 混合形：底部封闭，两侧上升到一个 1-2 像素宽的窄颈。
#       - 窄颈长度固定为 3 像素，这样 solver 可以把“通道长度”和“外扩模式”分开识别，避免几何歧义。
#       - 窄颈外端仍提供 2 个 degree-1 端点，便于推断朝外的主方向。
#     fill_seed:
#       - 仍放在腔室深处、靠近封闭端，与原任务保持一致。
#     radiate_zone:
#       - 原题: 开口一出去就按端点射线张开。
#       - 变体: 先通过窄颈，再从颈口的两个端点向外张开；张开斜率仍保留原题的两档:
#         1:1 (45 度) 或 1:2 (隔一步横向变化一次)。
#   behavior_trace:
#     - 原题中，solver 看到端点后，直接把端点当作扇形边界的起点。
#     - 变体里，端点先是颈部出口，不代表腔室立刻张开；solver 需要先承认“出口前面是一条窄通道”。
#     - 因为通道两侧是墙，所以 flood fill 会被强制挤过 1-2 像素的瓶颈。
#     - 一旦越过端点，外侧区域仍由两条边界射线限定，所以视觉上是“瓶颈 + 外扩”而不是“整面开口”。
#   complexity_calibration:
#     - 只增加一个核心自由度: 通道宽度 1 或 2。
#     - 通道长度固定为 3，避免样例过少时人类无法判断“哪些直行像素属于通道，哪些属于 1:2 斜率模式”。
#     - 外扩斜率只保留原题已有的两档，避免额外引入新规则。
#   implicit_assumption_audit:
#     - 原 solver 假设端点附近的第一段墙方向就等于外扩边界方向。
#       在瓶颈变体中这个假设失效，因为端点附近先是直通道，不是张开边。
#     - 原 solver 用端点近邻推断 1:2 模式；现在改为读取“通道后的第一步”，因为前 3 步固定属于通道。
#     - 原 flood-fill 逻辑、颜色识别逻辑、障碍阴影逻辑仍成立，所以只改几何边界构造。
#   placement_legality:
#     - 生成器保证墙体在局部坐标中只有 2 个 degree-1 端点，且不自交。
#     - 窄颈宽 1-2，腔室宽 5-7，确保“通道明显比腔室窄”。
#     - 墙 bbox、开放侧边距、封闭侧边距仍与原题接近，避免越界或贴边无法扩散。
#   visual_observability:
#     - 输入里可以肉眼看到一个明显的“瓶颈脖子”，输出里颜色必须通过脖子后才向边界扇形扩散。
#     - 若没有这条脖子，原题与变体在视觉上几乎一样；因此生成器强制所有样例都带颈部。
#   cleanliness:
#     - 墙、填充、障碍各用独立颜色；障碍仍是孤立像素，避免与窄颈混淆。
#     - 通道固定短而直，避免与主辐射区叠成杂乱的蛇形路径。
#   main_assumptions_to_enforce_in_generator:
#     - 墙最近的非墙色就是填充色。
#     - 目标墙组件只有 2 个 degree-1 端点，且这两个端点对应窄颈出口。
#     - 通道长度固定为 3，因此 solver 能从第 4 步判断 1:1 / 1:2 外扩模式。
#     - 端点在与 radiate_dir 垂直的轴上对齐，便于构造左右/上下两条边界射线。

import numpy as np
import random
from scipy.ndimage import distance_transform_cdt

from arc_utils import neighbors4, neighbors8, components_mask, get_bg


CORRIDOR_LEN = 3


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a bottle-shaped enclosure, forcing the color through a narrow neck before it fans outward.

    Input structure:
    - bg is the most common color. The wall is the most frequent non-bg color and forms a single partial enclosure with a short 1-2 cell neck.
    - A fill-color seed sits inside the deeper chamber near the closed end. Any other non-bg colors are obstacle pixels.

    Rule:
    - Flood-fill all bg cells reachable from the seed inside the chamber and through the neck.
    - After the fill exits the neck, it keeps expanding outward inside a ray-bounded wedge.
    - Obstacles cast shadows along the radiate axis, removing fill behind them while leaving the obstacle pixels intact.

    Key insight: The wall endpoints now mark a bottleneck exit, not a full open side; the fill must first squeeze through that neck, then radiate from the neck mouth.
    """
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

    inward_axis = (-rad[0], -rad[1])
    lateral = (0, 1) if rad[0] != 0 else (1, 0)
    if rad[0] != 0:
        endpoints.sort(key=lambda p: (p[1], p[0]))
    else:
        endpoints.sort(key=lambda p: (p[0], p[1]))

    def trace_side(ep: tuple[int, int], limit: int = CORRIDOR_LEN + 6) -> list[tuple[int, int]]:
        path = [ep]
        prev = None
        cur = ep
        while len(path) < limit:
            nbs = [p for p in comp_neighbors(*cur) if p != prev]
            if not nbs:
                break
            if len(nbs) == 1:
                nxt = nbs[0]
            else:
                nbs.sort(
                    key=lambda p: (
                        (p[0] - ep[0]) * inward_axis[0] + (p[1] - ep[1]) * inward_axis[1],
                        abs(p[0] - ep[0]) + abs(p[1] - ep[1]),
                    ),
                    reverse=True,
                )
                nxt = nbs[0]
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    def infer_tip_12(ep: tuple[int, int]) -> bool:
        path = trace_side(ep)
        steps = [
            (_sgn(path[i + 1][0] - path[i][0]), _sgn(path[i + 1][1] - path[i][1]))
            for i in range(len(path) - 1)
        ]
        assert len(steps) > CORRIDOR_LEN, "wall too short to decode corridor variant"
        return steps[CORRIDOR_LEN] == inward_axis

    tip_12 = sum(1 for ep in endpoints if infer_tip_12(ep)) >= 1

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], diag_d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += diag_d[0]
                cc += diag_d[1]
            else:
                rr += rad[0]
                cc += rad[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for side_sign, ep in [(-1, endpoints[0]), (1, endpoints[1])]:
        diag_d = (rad[0] + side_sign * lateral[0], rad[1] + side_sign * lateral[1])
        if tip_12:
            add_ray_12(ep, diag_d)
        else:
            add_ray(ep, diag_d)

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

# Canvas size bounds
GRID_H_VERT = (12, 18)
GRID_H_HORIZ = (9, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (6, 8)
DEPTH_HORIZ = (6, 8)
BASE_SPAN = (5, 7)              # distance between the two bottom side pixels
CORRIDOR_WIDTHS = (1, 2)        # passable width inside the neck
TIP_12_PROB = 0.45

# Placement margins
OPEN_END_MARGIN = 3

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _degree1_pixels(points):
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, tip_12: bool = False):
    sign = _sgn(target_v - start_v)
    n_side_steps = abs(target_v - start_v)
    if n_side_steps == 0:
        chosen_rows = []
    elif tip_12:
        chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
    else:
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


def _build_local_wall(rad, h, w):
    if rad[0]:
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
    want_12 = random.random() < TIP_12_PROB
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
            feasible_12 = body_depth >= 2 * max(left_out, right_out)
            if want_12 and feasible_12:
                candidates.append((base_span, left_out, right_out, True))
            elif not want_12:
                candidates.append((base_span, left_out, right_out, False))
            elif feasible_12:
                candidates.append((base_span, left_out, right_out, True))
    if not candidates:
        return None

    base_span, left_out, right_out, tip_12 = random.choice(candidates)
    left_bottom, right_bottom = 0, base_span
    neck_left = left_bottom + left_out
    neck_right = right_bottom - right_out

    left_corridor = [(u, neck_left) for u in range(CORRIDOR_LEN + 1)]
    right_corridor = [(u, neck_right) for u in range(CORRIDOR_LEN + 1)]
    left_body = [(u + CORRIDOR_LEN, v) for u, v in _make_arm(body_depth, neck_left, left_bottom, tip_12=tip_12)]
    right_body = [(u + CORRIDOR_LEN, v) for u, v in _make_arm(body_depth, neck_right, right_bottom, tip_12=tip_12)]

    pts = set(left_corridor) | set(right_corridor) | set(left_body) | set(right_body)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    if len(_degree1_pixels(pts)) != 2:
        return None
    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    return pts, depth, base_span, corridor_width, tip_12


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, base_span):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - base_span - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - base_span - 1)
    if rad == (0, -1):
        return random.randint(0, h - base_span - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - base_span - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth):
    by_row = {}
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


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
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
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, base_span, _corridor_width, _tip_12 = built
    anchor = _place_anchor(rad, h, w, depth, base_span)

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
    raise RuntimeError("failed to generate valid bottleneck b9e38dc0 input")
