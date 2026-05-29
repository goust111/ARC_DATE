# Variant of b9e38dc0
# Dimension: D13 — 开口方向的决定规则 (meta)
# Change: 开口方向由半包围边框的缺失面自然决定——边框哪一面不完整，射线就向那个方向辐射... → 开口方向由精灵边框颜色编码

# Design analysis
# ---------------
# spec:
#   variant_core:
#     original: |
#       原题的漏斗/半包围墙只有三面，solver 直接从两条臂的端点朝外方向求和，
#       因而“哪一面缺失”天然就是 radiate_dir。
#     variant: |
#       这里把漏斗补成闭合精灵：保留原来的两条臂和闭合端，再在原开口处补一条直线“门面”。
#       这条门面使用单独的 open_color；其余墙体统一用 closed_color。
#       solver 不再把几何缺口当成开口，而是把 open_color 这一整面视为“可透过面”。
#   concrete_geometry:
#     wall:
#       - 仍然使用原题的局部漏斗骨架与 1:1 / 1:2 两种臂角模式。
#       - 原开口两端之间补上一条水平或垂直线段，长度固定为 gap+1（即 6-8 格）。
#       - 这条线段就是颜色编码面；视觉上形成“蓝色外壳里夹着一条红色门面”的闭合边框。
#     fill_seed:
#       - 仍放在内部、靠近闭合端，保持 1-4 格的小 4 连通块。
#     obstacles:
#       - 仍是零散单像素，部分落在外放射区域中，继续制造阴影。
#   behavior_trace:
#     original_flow: |
#       旧逻辑看到漏斗端点后，直接沿端点朝外延长两条边界射线；内部填充自然从缺口泄出。
#     variant_flow: |
#       新输入里原来的缺口被 open_color 封成了一条线，几何上已经闭合。
#       solver 必须先识别那条直线色带，再把这条色带当作“门”而不是障碍：
#       flood-fill 可以穿过它，但输出仍保留它原色；向外辐射的方向取这条门面的外法向。
#     difference: |
#       如果仍按原任务把“缺口方向”当规则，同一输入不会得到正确输出：
#       旧 solver 要么找不到端点，要么把 open_color 误判成别的对象。
#   complexity_calibration:
#     - 只引入两种墙语义色：open_color 与 closed_color，避免颜色自由度过多。
#     - open_color 固定表现为一条长直线段；fill blob 刻意限制到 1-4 格，障碍只做单点，
#       这样“哪种颜色代表门面”在少量样例里能重复出现且不和别的对象混淆。
#     - 仍保留原题的 ray slope、阴影与障碍机制，因此视觉上还是同一家族任务。
#   implicit_assumptions_review:
#     - 原代码 `wall = most frequent non-bg color` 依赖墙体单色；现在只对 closed_color 成立，
#       因此 solver 需要额外从长直线段里识别 open_color。
#     - 原代码 `endpoints = degree-1 pixels` 依赖墙体有真实缺口；现在边框闭合，
#       所以要改为从 open_color 线段的两个端点恢复“门角”。
#     - 原代码 `passable = bg or fill` 默认墙全不可穿过；现在必须把 open_color 也放进可通行集合，
#       否则内部颜色永远出不去。
#     - 原代码阴影判断里 `wall` 只有一种颜色；现在要同时排除 closed_color 和 open_color。
#   placement_legality:
#     - 仍用原本的局部坐标和 anchor 约束，确保闭合端离边界至少 2 格、门面朝向的外侧至少 3 格。
#     - 门面只补在原开口连线处，因此不会越界，也不会与内部 seed 重叠。
#     - 障碍只放在背景格，并基于 solver 的名义填充区筛选，保证不覆盖墙、门面或种子。
#   visual_observability:
#     - 人类看到训练样例时，会稳定看到“同一种颜色的整面”总是对应外放射的出口。
#     - 因为门面被补上了，规则差异是肉眼可见的：不是从缺口漏出，而是穿过特定颜色的面漏出。
#   visual_cleanliness:
#     - 门面是一整条直线，障碍仍是点状，层次清晰；输出中门面保留原色，不会被填色覆盖。
#   color_inference:
#     - 颜色只编码二值语义：open vs closed。
#     - 不使用颜色 id 排序、数值大小规律或依赖“某个固定数字是特殊色”的不可见规则。

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill a closed funnel and let the color pass through its coded wall face.

    Input structure:
    - bg is the dominant color. The wall is now two-toned: a large closed-color shell plus one straight open-color face that seals the former gap.
    - A small fill-color blob sits inside the shell near the closed end. Other non-bg colors are sparse obstacles.

    Rule:
    - Treat the open-color face as the only permeable side of the wall: interior flood-fill may cross that face while the face itself keeps its own color.
    - After the fill exits, continue the same outward radiation and obstacle-shadow behavior as in the original task.

    Key insight: The opening is no longer read from missing geometry; the coded wall face identifies which side behaves like the outlet, and its two endpoints recover the radiating boundary rays.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)

    closed_color = max(non_bg, key=lambda c: int((g == c).sum()))

    def pick_open_face() -> tuple[int, set[tuple[int, int]]]:
        best: tuple[int, int, set[tuple[int, int]]] | None = None
        for color in non_bg:
            if color == closed_color:
                continue
            for cells in components_mask(g == color, connectivity=4):
                pts = {(int(r), int(c)) for r, c in cells}
                if len(pts) < 6:
                    continue
                rows = {r for r, _ in pts}
                cols = {c for _, c in pts}
                if len(rows) == 1:
                    row = next(iter(rows))
                    c0 = min(c for _, c in pts)
                    c1 = max(c for _, c in pts)
                    expected = {(row, cc) for cc in range(c0, c1 + 1)}
                elif len(cols) == 1:
                    col = next(iter(cols))
                    r0 = min(r for r, _ in pts)
                    r1 = max(r for r, _ in pts)
                    expected = {(rr, col) for rr in range(r0, r1 + 1)}
                else:
                    continue
                if pts != expected:
                    continue
                score = (len(pts), int(color))
                if best is None or score > best[:2]:
                    best = (score[0], score[1], pts)
        assert best is not None, "expected a straight open-face segment"
        length, color, pts = best
        assert length >= 6
        return color, pts

    open_color, open_face = pick_open_face()

    wall_mask = (g == closed_color) | (g == open_color)
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall component"

    other_colors = [c for c in non_bg if c not in (closed_color, open_color)]
    assert other_colors, "expected at least one non-wall color (fill color)"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color), strict=False)]
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
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill), strict=False)]
    if not fill_pos:
        return g

    fill_center = (
        sum(r for r, _ in fill_pos) / len(fill_pos),
        sum(c for _, c in fill_pos) / len(fill_pos),
    )

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
    open_face = open_face & wall_comp
    assert len(open_face) >= 6, "chosen wall component must contain the open face"

    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    closed_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True
        if g[r, c] == closed_color:
            closed_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    open_rows = {r for r, _ in open_face}
    open_cols = {c for _, c in open_face}
    if len(open_rows) == 1:
        face_r = next(iter(open_rows))
        endpoints = sorted(open_face, key=lambda p: p[1])
        interior_dr = 1 if fill_center[0] > face_r else -1
        rad = (-interior_dr, 0)
    else:
        face_c = next(iter(open_cols))
        endpoints = sorted(open_face, key=lambda p: p[0])
        interior_dc = 1 if fill_center[1] > face_c else -1
        rad = (0, -interior_dc)
    endpoints = [endpoints[0], endpoints[-1]]

    def dist_to_fill(pt: tuple[int, int]) -> float:
        return abs(pt[0] - fill_center[0]) + abs(pt[1] - fill_center[1])

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        arm_neighbors = [p for p in comp_neighbors(*ep) if p not in open_face]
        assert arm_neighbors, "open-face endpoint must attach to a wall arm"
        n1 = min(arm_neighbors, key=dist_to_fill)
        d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))  # inward along arm
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep and p not in open_face]
        if n2s:
            n2 = min(n2s, key=dist_to_fill)
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
    inferred_rad = (sgn(dr_sum), sgn(dc_sum))
    assert inferred_rad == rad, "wall color and arm geometry must agree on outward side"

    barrier = closed_mask.copy()

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

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12, strict=False):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill) | (g == open_color))
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

    obstacles = (g != bg) & (g != fill) & (g != closed_color) & (g != open_color)

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


# --------------- generator (o5x) ---------------

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

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 3, 4]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 6)
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
    """Construct the original open funnel in local (u, v) coordinates."""
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


def _find_obstacle_candidates(grid, bg, wall_mask, nominal_fill_zone):
    """Find bg cells far enough from wall for obstacle placement, using distance transform."""
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible), strict=False))
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
    bg, closed_color, open_color, fill_color = palette[:4]
    obstacle_colors = palette[4:]

    open_face_local = {(0, v) for v in range(gap + 1)}
    closed_wall_local = set(local_wall) - {(0, 0), (0, gap)}

    grid = np.full((h, w), bg, dtype=int)
    for u, v in closed_wall_local:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = closed_color
    for u, v in open_face_local:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg and grid[r, c] != closed_color:
            return None
        grid[r, c] = open_color

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
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, closed_color, open_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        wall_mask = (grid == closed_color) | (grid == open_color)
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_mask, nominal_fill_zone)
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
    raise RuntimeError("failed to generate valid b9e38dc0 input")
