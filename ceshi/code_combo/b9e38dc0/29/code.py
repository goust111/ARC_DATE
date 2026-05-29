# Combo variant of b9e38dc0 (index 29)
# combo_spec: D12/0+D13/1
# Name: 统一开口方向+颜色编码方向

# Design analysis
# ---------------
# spec:
#   fused_variant:
#     summary: |
#       以 D13/1 的“闭合门面颜色编码开口”作为基础，再叠加 D12/0 的“同图多个精灵且方向统一”。
#       每个精灵仍是原题家族里的漏斗/半包围墙，但原本的几何缺口被一条直线门面补上；
#       门面颜色不再只是“这是门”，而是从固定色表里编码全局辐射方向。
#       同一张图至少有两个精灵，所有精灵的门面颜色相同，因此它们共享同一 outward direction。
#   concrete_geometry:
#     sprite:
#       - 保留原题的两条锯齿臂 + 闭合端 + 1:1 / 1:2 两种臂角模式。
#       - 在原开口处补一条整齐的水平/垂直门面，长度仍为 gap+1。
#       - 非门面的墙统一使用 closed_color；门面使用 direction_color。
#     multiplicity:
#       - 每张图放置 2 个精灵，彼此独立、互不重叠，并沿与辐射垂直的轴分开摆放。
#       - 因为两个精灵共享相同门面色，所以会重复支撑“这个颜色代表这个方向”。
#     palette:
#       - 方向编码色固定为 4 个显眼颜色：1/2/3/4 分别对应 上/右/下/左。
#       - 背景、闭合墙、填充种子、障碍全部避开这 4 个方向色，避免视觉混淆。
#   behavior_trace:
#     example: |
#       例：若门面颜色为 2（红），则两只闭合漏斗都朝右辐射。
#       原题流程：从真实缺口的两个端点推出开口方向，再向外放射。
#       现在流程：solver 先识别所有长直线门面，读出它们的颜色 2 -> 右；
#       然后对每个精灵，把这条线视作“可穿透门”，从门两端恢复边界射线，
#       内部填充穿门而出，继续向右扩散；障碍仍沿同方向制造阴影。
#     difference: |
#       如果还按原题把“缺失的一面”当作开口，则这里根本没有缺口；
#       如果只做单个 D13/1 精灵，也看不出 D12/0 的“统一全局方向重复出现”。
#   complexity_calibration:
#     - 保留原题的锯齿臂、两类射线斜率、阴影机制，所以仍属于同一视觉家族。
#     - 只增加一个低自由度的符号层：4 个固定方向色；同图内只激活其中 1 个，便于少样例归纳。
#     - 精灵数固定为 2，既能体现“多个且统一”，又不会把辐射区挤得过于混乱。
#   implicit_assumptions_review:
#     - 原代码假设 wall 是单色且唯一主组件；这里改为 closed_color 最大、open_color 为多段直线门面。
#     - 原代码依赖 degree-1 端点来找开口；这里墙体闭合，必须改为读取门面两端。
#     - 原代码默认墙完全不可穿过；这里 open_color 必须放入可通行集合，但输出保留原色。
#     - 原代码按单组件处理；这里要遍历多个 wall component，并给每个组件单独加外放射边界。
#     - 新增假设：所有门面都用同一个方向色，且每个门面是长度 >= 6 的单一直线段。
#       generator 通过固定门面构造、统一色表和障碍配色限制来保证这些前提成立。
#   placement_legality:
#     - 组件按与辐射垂直的轴分槽放置，保证输入时 bbox 不重叠。
#     - 同时要求门面投影区间留足空隙，降低多组件辐射区在输出中相互缠绕的概率。
#     - 障碍只放在背景格、远离墙体，并基于无障碍时的 nominal fill zone 采样。
#   observability_and_cleanliness:
#     - 人类会看到：同一种门面色在同图重复出现，两个精灵都朝同一方向出流。
#     - 方向色不复用于背景/种子/障碍，因此“颜色是语义标签”肉眼可见。
#     - 两个精灵分开摆放，门面是整齐直线，障碍仍是稀疏点状，层次不会打架。

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


DIR_TO_COLOR = {
    (-1, 0): 1,  # blue -> up
    (0, 1): 2,   # red -> right
    (1, 0): 3,   # green -> down
    (0, -1): 4,  # yellow -> left
}
COLOR_TO_DIR = {color: direction for direction, color in DIR_TO_COLOR.items()}
NON_DIRECTION_COLORS = [0, 5, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several closed funnels and let a shared door color dictate their common exit direction.

    Input structure:
    - The background is the dominant color. Each sprite is a closed funnel: most wall pixels use one closed-wall color, while one straight wall face uses a direction-coded color.
    - There are multiple such sprites in one grid, and all coded faces share the same color, so every sprite uses the same outward direction. Small fill blobs sit inside the funnels; other sparse colors are obstacles.

    Rule:
    - Decode the common outward direction from the door-face color, then treat each coded face as permeable while the rest of the wall stays solid.
    - Flood-fill from the interior seeds through each coded face, extend the fill outward within the two recovered boundary rays, and apply obstacle shadows along that shared direction.

    Key insight: Geometry now tells you where each door segment is, but the door color tells you which global direction all sprites radiate toward; once that color is decoded, the same direction applies to every funnel in the grid.
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

    def straight_segments_for_color(color: int) -> list[set[tuple[int, int]]]:
        segments: list[set[tuple[int, int]]] = []
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
                endpoints = [(row, c0), (row, c1)]
            elif len(cols) == 1:
                col = next(iter(cols))
                r0 = min(r for r, _ in pts)
                r1 = max(r for r, _ in pts)
                expected = {(rr, col) for rr in range(r0, r1 + 1)}
                endpoints = [(r0, col), (r1, col)]
            else:
                continue
            if pts != expected:
                continue
            ok = True
            for ep in endpoints:
                touches_closed = any(
                    0 <= rr < h and 0 <= cc < w and g[rr, cc] == closed_color
                    for rr, cc in neighbors8(*ep)
                )
                if not touches_closed:
                    ok = False
                    break
            if ok:
                segments.append(pts)
        return segments

    open_color = None
    open_segments: list[set[tuple[int, int]]] = []
    best_score = None
    for color in non_bg:
        if color == closed_color or color not in COLOR_TO_DIR:
            continue
        segments = straight_segments_for_color(color)
        if len(segments) < 2:
            continue
        score = (sum(len(seg) for seg in segments), len(segments))
        if best_score is None or score > best_score:
            best_score = score
            open_color = color
            open_segments = segments

    assert open_color is not None, "expected repeated straight door faces in a direction color"
    rad = COLOR_TO_DIR[open_color]

    wall_mask = (g == closed_color) | (g == open_color)
    wall_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert len(wall_comps) >= 2, "expected multiple wall components"

    other_colors = [c for c in non_bg if c not in (closed_color, open_color)]
    assert other_colors, "expected at least one non-wall color"
    wall_points = [p for comp in wall_comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill_color = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in np.argwhere(g == fill_color)]
    if not fill_pos:
        return g

    barrier = g == closed_color

    def comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

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

    assigned_segments = 0
    for wall_comp in wall_comps:
        comp_open_segments = [seg for seg in open_segments if seg <= wall_comp]
        assert len(comp_open_segments) == 1, "each wall component must contain exactly one coded face"
        assigned_segments += 1
        open_face = comp_open_segments[0]

        comp_mask = np.zeros_like(wall_mask, dtype=bool)
        for r, c in wall_comp:
            comp_mask[r, c] = True

        if len({r for r, _ in open_face}) == 1:
            endpoints = sorted(open_face, key=lambda p: p[1])
        else:
            endpoints = sorted(open_face, key=lambda p: p[0])
        endpoints = [endpoints[0], endpoints[-1]]

        def dist_to_fill(pt: tuple[int, int]) -> int:
            return min(abs(pt[0] - fr) + abs(pt[1] - fc) for fr, fc in fill_pos)

        def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
            arm_neighbors = [p for p in comp_neighbors(comp_mask, *ep) if p not in open_face]
            assert arm_neighbors, "door endpoint must attach to a wall arm"
            n1 = min(arm_neighbors, key=dist_to_fill)
            d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))
            d_use = d1
            is_12 = False
            n2s = [p for p in comp_neighbors(comp_mask, *n1) if p != ep and p not in open_face]
            if n2s:
                n2 = min(n2s, key=dist_to_fill)
                d2 = (sgn(n2[0] - n1[0]), sgn(n2[1] - n1[1]))
                if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
            return (-d_use[0], -d_use[1]), is_12

        out_info = [outward_dir(ep) for ep in endpoints]
        out_dirs = [d for d, _ in out_info]
        tip_12 = [flag for _, flag in out_info]
        inferred = (sgn(sum(d[0] for d in out_dirs)), sgn(sum(d[1] for d in out_dirs)))
        assert inferred == rad, "door color and wall geometry must agree"

        for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
            if is_12:
                add_ray_12(ep, d)
            else:
                add_ray(ep, d)

    assert assigned_segments == len(open_segments)

    passable = (~barrier) & ((g == bg) | (g == fill_color) | (g == open_color))
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
    out[filled_mask] = fill_color

    obstacles = (g != bg) & (g != fill_color) & (g != closed_color) & (g != open_color)

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

GRID_H_VERT = (13, 20)
GRID_W_VERT = (20, 26)
GRID_H_HORIZ = (18, 24)
GRID_W_HORIZ = (13, 20)

SPRITE_COUNT = 2
SPRITE_GAP = 4
OPEN_END_MARGIN = 3

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 9)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 3, 4]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.7
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 28
GEN_RETRIES = 400


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
        max_depth = min(DEPTH_HORIZ[1], w - 6)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
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
        u, v = frontier.pop(random.randrange(len(frontier)))
        neighbors = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(neighbors)
        for q in neighbors:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _bbox_dims(rad: tuple[int, int], depth: int, gap: int) -> tuple[int, int]:
    if rad[0]:
        return depth + 1, gap + 1
    return gap + 1, depth + 1


def _anchors_for_two(rad: tuple[int, int], h: int, w: int, specs: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if rad[0]:
        widths = [gap + 1 for _, gap in specs]
        required = sum(widths) + SPRITE_GAP
        if required > w:
            return None
        slack = w - required
        left_pad = random.randint(0, slack)
        between_extra = random.randint(0, slack - left_pad)
        x0 = left_pad
        x1 = x0 + widths[0] + SPRITE_GAP + between_extra
        anchors = []
        for idx, (depth, gap) in enumerate(specs):
            c0 = x0 if idx == 0 else x1
            if rad == (-1, 0):
                r0 = random.randint(OPEN_END_MARGIN, h - depth - 3)
            else:
                r0 = random.randint(depth + 2, h - 4)
            anchors.append((r0, c0))
        return anchors

    heights = [gap + 1 for _, gap in specs]
    required = sum(heights) + SPRITE_GAP
    if required > h:
        return None
    slack = h - required
    top_pad = random.randint(0, slack)
    between_extra = random.randint(0, slack - top_pad)
    y0 = top_pad
    y1 = y0 + heights[0] + SPRITE_GAP + between_extra
    anchors = []
    for idx, (depth, gap) in enumerate(specs):
        r0 = y0 if idx == 0 else y1
        if rad == (0, -1):
            c0 = random.randint(OPEN_END_MARGIN, w - depth - 3)
        else:
            c0 = random.randint(depth + 2, w - 4)
        anchors.append((r0, c0))
    return anchors


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid: np.ndarray, bg: int, wall_mask: np.ndarray, nominal_fill_zone: np.ndarray):
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in np.argwhere(eligible)]
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


def _door_projection_disjoint(rad: tuple[int, int], door_segments: list[set[tuple[int, int]]]) -> bool:
    intervals = []
    for seg in door_segments:
        if rad[0]:
            cols = sorted(c for _, c in seg)
            intervals.append((cols[0], cols[-1]))
        else:
            rows = sorted(r for r, _ in seg)
            intervals.append((rows[0], rows[-1]))
    intervals.sort()
    for (_, hi), (lo2, _) in zip(intervals, intervals[1:]):
        if lo2 - hi <= 2:
            return False
    return True


def _gen_attempt():
    rad = random.choice(list(DIR_TO_COLOR))
    open_color = DIR_TO_COLOR[rad]

    if rad[0]:
        h = random.randint(*GRID_H_VERT)
        w = random.randint(*GRID_W_VERT)
    else:
        h = random.randint(*GRID_H_HORIZ)
        w = random.randint(*GRID_W_HORIZ)

    built = []
    for _ in range(SPRITE_COUNT):
        item = _build_local_wall(rad, h, w)
        if item is None:
            return None
        built.append(item)

    specs = [(depth, gap) for _, depth, gap in built]
    anchors = _anchors_for_two(rad, h, w, specs)
    if anchors is None:
        return None

    palette = NON_DIRECTION_COLORS[:]
    random.shuffle(palette)
    bg = palette[0]
    closed_color = palette[1]
    fill_color = palette[2]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    door_segments = []

    for (local_wall, depth, gap), anchor in zip(built, anchors):
        open_face_local = {(0, v) for v in range(gap + 1)}
        closed_wall_local = set(local_wall) - {(0, 0), (0, gap)}

        door_global = set()
        occupied = set()
        for u, v in closed_wall_local:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return None
            grid[r, c] = closed_color
            occupied.add((r, c))
        for u, v in open_face_local:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] not in (bg, closed_color):
                return None
            grid[r, c] = open_color
            occupied.add((r, c))
            door_global.add((r, c))
        door_segments.append(door_global)

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

    if not _door_projection_disjoint(rad, door_segments):
        return None

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        return None
    if n_obstacles:
        wall_mask = (grid == closed_color) | (grid == open_color)
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_mask, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if not _touches_border(new_fill):
        return None
    if len(door_segments) != SPRITE_COUNT:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo b9e38dc0 input")
