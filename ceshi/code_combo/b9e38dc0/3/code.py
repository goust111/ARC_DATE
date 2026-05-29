# Combo variant of b9e38dc0 (index 3)
# combo_spec: D2/0+D7/1
# Name: 对角线移动+障碍反射

import sys
import random
import numpy as np
from scipy.ndimage import distance_transform_cdt

for _p in (
    "/home/lixinyu/snap/new/solvers",
    "/home/lixinyu/snap/new/arc2_solver_select",
):
    if _p not in sys.path:
        sys.path.append(_p)

from arc_utils import neighbors4, neighbors8, components_mask, get_bg


# Design analysis
# a) 具体化与行为追踪:
#    - 仍保留原题第一阶段: 最大的非背景色是 wall, 它形成一个半包围漏斗; fill seed 在内部,
#      solver 先按开口方向把内部与外辐射区填满, 得到 wall+fill 的单个 stage1 sprite.
#    - 融合后的第二阶段不再做“沿 radiate_dir 的直线位移”或“阴影扫描”. 其余非背景单像素都视为 mover:
#      每个 mover 先根据 wall 开口确定一个对角线方向:
#        * 沿 radiate_dir 保留“向外”的那个轴分量;
#        * 垂直于 radiate_dir 的那个分量, 由 mover 位于开口中线哪一侧决定, 取远离中线的一侧.
#      因此 rad=(1,0) 时 mover 只可能走 down-left 或 down-right; rad=(0,1) 时只可能走 up-right 或 down-right.
#    - mover 每一步都尝试走 45 度对角. 若斜前方或斜前方邻接的 sprite 边界会被碰到, 它像光线一样反射:
#        * 竖向分量撞到 sprite -> 翻转 dr;
#        * 横向分量撞到 sprite -> 翻转 dc;
#        * 两个分量同时撞到/正撞角点 -> 两轴同时翻转.
#      反射后继续沿新的对角方向前进, 直到最后一个格子停在网格边界附近.
#    - 具体小例子: 若 rad=(0,1), 开口朝右, 某 mover 在 sprite 上方, 则初始方向是 up-right.
#      原题/D7 直线版会只向右走并在命中 sprite 后改成上下之一; 本融合版会先斜向右上, 若右上的路径擦到
#      sprite 的上边缘, 只翻转 dr 变成 down-right, 继续弹走. 输出里表现为“起点清空、终点移到另一条对角边界”.
# b) 复杂度校准:
#    - 只引入两个离散自由度: 初始对角方向的侧向符号、以及碰撞时翻转哪一个轴. 不引入变色、速度变化、
#      多类障碍等额外机制.
#    - 第一阶段 wall 仍保留 1:1 / 1:2 两种开口几何, 第二阶段则至少保证一个 mover 发生反射, 并尽量再配一个
#      不反射或反射次数不同的 mover, 使训练样例里能同时看出“先对角移动, 再由 sprite 几何决定弹射”.
# c) 隐式假设审查:
#    - 原 main 在 `filled_mask` 之后假设 obstacle 是静止遮挡物, 通过 `shadow_process` 擦掉填充. 该假设失效,
#      必须整段删除, 否则 mover 会被当作阴影源而不是位移对象.
#    - 原 `passable = (~barrier) & ((g == bg) | (g == fill))` 依赖“其余非背景像素不可被 stage1 填充覆盖”.
#      融合后依然成立, 所以生成器必须把 mover 放在 nominal stage1 sprite 外侧.
#    - 原 fill 颜色选择依赖“离 wall 最近的非 wall 颜色就是 seed”. 因此 mover 不能靠 wall 太近, 否则 solver
#      会把 mover 颜色误认成 fill. 生成器继续强制 mover 到 wall 的 taxicab 距离至少为 3.
#    - 新的对角规则隐含要求: mover 不应恰好落在开口中线上, 否则侧向符号不明确. solver 有回退规则, 但生成器会
#      过滤这类候选, 让样例中的方向判据始终清晰.
# d) 放置合法性:
#    - generator 先运行 stage1 solver, 只在 `sprite_mask == False` 且 `nominal_out` 仍为背景的位置挑 mover.
#    - 每个候选都会经过与正式 solver 相同的对角追踪器, 只接受能到边界、不会循环、且 footprint 与已选 mover
#      不重叠的路径.
#    - 为了避免起点或反射点紧贴 wall 造成 fill 颜色误判, 再额外要求 mover 距 wall >= 3 且路径长度 >= 3.
# e) 视觉可观察性:
#    - 人类可见线索只有几何: wall 开口给出 outward 轴, mover 相对开口中线的左右/上下位置给出另一轴, 反射点由
#      sprite 边缘决定. 这些都能在 2-4 个样例中直接看出来, 不依赖颜色编号或隐藏排序.
#    - 至少一个 mover 会发生反射; 大多数样例还会带一个不同路径的第二 mover, 帮助解题者区分“对角+反射”与
#      “直走+反射”或“静态阴影”.
# f) 可视化整洁度:
#    - mover 保持单像素, 轨迹不着色, 输出只保留终点像素, 避免在 fill 区域上叠出杂乱彩线.
#    - 生成器禁止 mover 之间的 source/path/end 重叠, 也禁止占用 stage1 sprite, 视觉层次清楚.
# g) 变化的可推断性:
#    - mover 颜色不变, 只搬移位置. 颜色只区分不同 mover 身份, 规则本身完全由空间关系驱动.
# 9) main() 的新隐式假设:
#    - `_initial_diag_dir` 假设开口中线可由两个 wall endpoints 的中点近似表示; 生成器总构造恰有两个 endpoints
#      的单个 wall 组件, 因而该判据稳定.
#    - `_collision_options` 假设 stage1 sprite 是轴对齐像素集合, 可以用“竖向相邻格/横向相邻格/斜前方格”区分
#      撞击轴. 对 ARC 像素网格该假设天然成立.
#    - `_trace_mover` 通过 `(r,c,dr,dc)` 去重来防止异常循环; generator 只接受 `ok=True` 的候选, 从构造上保证
#      最终训练样例不会出现 solver 依赖的退化输入.


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

MAX_MOVERS = 4
MOVER_MIN_TRAVEL = 3
MOVER_MIN_WALL_DIST = 3
REFLECT_REQUIRED = 1
STRAIGHT_BONUS_PROB = 0.55

MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 500


def _sgn(x: float) -> int:
    return int((x > 0) - (x < 0))


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


def _compute_stage1(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)

    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(wall_mask, connectivity=8)
    ]
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
        return {
            "out": g,
            "filled_mask": np.zeros_like(g, dtype=bool),
            "sprite_mask": wall_mask.copy(),
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "rad": (0, 0),
            "wall_comp_mask": wall_mask.copy(),
            "endpoints": [],
        }

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
        return [
            (rr, cc)
            for rr, cc in neighbors8(r, c)
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]
        ]

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints"

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
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

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
    sprite_mask = comp_mask | (out == fill)

    return {
        "out": out,
        "filled_mask": filled_mask,
        "sprite_mask": sprite_mask,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "rad": rad,
        "wall_comp_mask": comp_mask,
        "endpoints": endpoints,
    }


def _axis_sign_from_opening(
    start: tuple[int, int],
    rad: tuple[int, int],
    endpoints: list[tuple[int, int]],
    sprite_mask: np.ndarray,
) -> int:
    rows, cols = np.where(sprite_mask)
    if rad[0] != 0:
        center2 = sum(ep[1] for ep in endpoints[:2]) if len(endpoints) >= 2 else int(cols.min() + cols.max())
        side = _sgn(2 * start[1] - center2)
        if side == 0:
            side = _sgn(2 * start[1] - int(cols.min() + cols.max()))
        if side == 0:
            side = -1 if start[1] < sprite_mask.shape[1] / 2 else 1
        return int(side)

    center2 = sum(ep[0] for ep in endpoints[:2]) if len(endpoints) >= 2 else int(rows.min() + rows.max())
    side = _sgn(2 * start[0] - center2)
    if side == 0:
        side = _sgn(2 * start[0] - int(rows.min() + rows.max()))
    if side == 0:
        side = -1 if start[0] < sprite_mask.shape[0] / 2 else 1
    return int(side)


def _initial_diag_dir(
    start: tuple[int, int],
    rad: tuple[int, int],
    endpoints: list[tuple[int, int]],
    sprite_mask: np.ndarray,
) -> tuple[int, int]:
    if rad[0] != 0:
        return int(rad[0]), _axis_sign_from_opening(start, rad, endpoints, sprite_mask)
    return _axis_sign_from_opening(start, rad, endpoints, sprite_mask), int(rad[1])


def _collision_options(
    cur: tuple[int, int],
    d: tuple[int, int],
    sprite_mask: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = sprite_mask.shape
    dr, dc = int(d[0]), int(d[1])
    r, c = int(cur[0]), int(cur[1])

    def occupied(rr: int, cc: int) -> bool:
        return 0 <= rr < h and 0 <= cc < w and bool(sprite_mask[rr, cc])

    vhit = occupied(r + dr, c)
    hhit = occupied(r, c + dc)
    diag_hit = occupied(r + dr, c + dc)

    if not (vhit or hhit or diag_hit):
        return []
    if vhit and not hhit:
        return [(-dr, dc), (dr, -dc), (-dr, -dc)]
    if hhit and not vhit:
        return [(dr, -dc), (-dr, dc), (-dr, -dc)]
    return [(-dr, -dc), (-dr, dc), (dr, -dc)]


def _trace_mover(
    start: tuple[int, int],
    d: tuple[int, int],
    sprite_mask: np.ndarray,
) -> dict:
    h, w = sprite_mask.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(d[0]), int(d[1])
    path: list[tuple[int, int]] = []
    reflections = 0
    seen = set()

    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return {"ok": False, "end": cur, "path": path, "reflections": reflections, "dir": d}
        seen.add(state)

        turned = False
        for nd in _collision_options(cur, (dr, dc), sprite_mask):
            nr, nc = cur[0] + nd[0], cur[1] + nd[1]
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if sprite_mask[nr, nc]:
                continue
            dr, dc = int(nd[0]), int(nd[1])
            reflections += 1
            turned = True
            break
        if turned:
            continue

        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return {"ok": True, "end": cur, "path": path, "reflections": reflections, "dir": d}
        if sprite_mask[nr, nc]:
            return {"ok": False, "end": cur, "path": path, "reflections": reflections, "dir": d}

        cur = (int(nr), int(nc))
        path.append(cur)


def _turn_cardinal_reflection(
    cur: tuple[int, int],
    hit: tuple[int, int],
    d: tuple[int, int],
    sprite_mask: np.ndarray,
) -> tuple[int, int] | None:
    h, w = sprite_mask.shape

    if d[0] != 0:
        cols = np.where(sprite_mask[hit[0]])[0]
        if len(cols) == 0:
            return None
        mid = (int(cols[0]) + int(cols[-1])) / 2.0
        pref = -1 if hit[1] <= mid else 1
        for dc in (pref, -pref):
            nr, nc = cur[0], cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not sprite_mask[nr, nc]:
                return 0, int(dc)
        return None

    rows = np.where(sprite_mask[:, hit[1]])[0]
    if len(rows) == 0:
        return None
    mid = (int(rows[0]) + int(rows[-1])) / 2.0
    pref = -1 if hit[0] <= mid else 1
    for dr in (pref, -pref):
        nr, nc = cur[0] + dr, cur[1]
        if 0 <= nr < h and 0 <= nc < w and not sprite_mask[nr, nc]:
            return int(dr), 0
    return None


def _trace_cardinal_mover(
    start: tuple[int, int],
    rad: tuple[int, int],
    sprite_mask: np.ndarray,
) -> dict:
    h, w = sprite_mask.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(rad[0]), int(rad[1])
    reflections = 0
    seen = set()

    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return {"ok": False, "end": cur, "reflections": reflections}
        seen.add(state)
        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return {"ok": True, "end": cur, "reflections": reflections}
        if sprite_mask[nr, nc]:
            turned = _turn_cardinal_reflection(cur, (nr, nc), (dr, dc), sprite_mask)
            if turned is None:
                return {"ok": False, "end": cur, "reflections": reflections}
            dr, dc = turned
            reflections += 1
            continue
        cur = (int(nr), int(nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the funnel interior, then let isolated pixels leave along diagonals and bounce off the filled sprite.

    Input structure:
    - Background is the most common color. The most frequent non-background color is a wall component forming a partial enclosure.
    - One non-wall color sits inside that enclosure as the fill seed. Any remaining non-background singleton pixels are movers placed outside the eventual wall+fill sprite.

    Rule:
    - First flood-fill the reachable interior and outward radiate zone with the seed color, using the wall endpoints to define the opening direction exactly as in the base task.
    - Then move each isolated pixel along a 45-degree diagonal: one axis matches the outward radiate direction, while the perpendicular axis points away from the opening midpoint. Whenever that diagonal ray meets the wall+fill sprite, it reflects by flipping the blocked axis and continues until it reaches the grid edge.

    Key insight: The wall still defines the stage1 filled sprite, but now that sprite also determines both the movers' initial diagonal slant and the later reflection geometry, so the second stage depends on relative position and obstacle-facing axis together.
    """
    info = _compute_stage1(np.array(grid, copy=True))
    out = np.array(info["out"], copy=True)
    bg = int(info["bg"])
    wall = int(info["wall"])
    fill = int(info["fill"])
    rad = tuple(info["rad"])
    sprite_mask = np.array(info["sprite_mask"], copy=True)
    endpoints = list(info["endpoints"])

    if rad == (0, 0):
        return out

    g = np.array(grid, copy=True)
    mover_cells = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != bg) & (g != wall) & (g != fill))
    ]
    for r, c, _col in mover_cells:
        out[r, c] = bg

    for r, c, col in sorted(mover_cells):
        d = _initial_diag_dir((r, c), rad, endpoints, sprite_mask)
        traced = _trace_mover((r, c), d, sprite_mask)
        end_r, end_c = traced["end"]
        out[end_r, end_c] = col

    return out


def _find_mover_candidates(grid, stage1_info):
    bg = int(stage1_info["bg"])
    wall = int(stage1_info["wall"])
    rad = tuple(stage1_info["rad"])
    sprite_mask = np.array(stage1_info["sprite_mask"], copy=True)
    endpoints = list(stage1_info["endpoints"])

    wall_mask = grid == wall
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (~sprite_mask) & (dist >= MOVER_MIN_WALL_DIST)

    reflect = []
    straight = []
    for r, c in zip(*np.where(eligible)):
        start = (int(r), int(c))
        d = _initial_diag_dir(start, rad, endpoints, sprite_mask)
        if 0 in d:
            continue
        traced = _trace_mover(start, d, sprite_mask)
        if not traced["ok"]:
            continue
        if len(traced["path"]) < MOVER_MIN_TRAVEL:
            continue
        end = traced["end"]
        if end == start:
            continue
        if not (end[0] in (0, grid.shape[0] - 1) or end[1] in (0, grid.shape[1] - 1)):
            continue
        if len({p[0] for p in traced["path"] + [start]}) < 2:
            continue
        if len({p[1] for p in traced["path"] + [start]}) < 2:
            continue

        cardinal = _trace_cardinal_mover(start, rad, sprite_mask)
        item = {
            "start": start,
            "end": (int(end[0]), int(end[1])),
            "path": list(traced["path"]),
            "footprint": {start, *traced["path"]},
            "reflections": int(traced["reflections"]),
            "dir": d,
            "cardinal_end": tuple(cardinal["end"]),
            "cardinal_diff": tuple(cardinal["end"]) != tuple(end),
        }
        if traced["reflections"] >= 1:
            reflect.append(item)
        else:
            straight.append(item)

    random.shuffle(reflect)
    random.shuffle(straight)
    reflect.sort(key=lambda x: (x["cardinal_diff"], x["reflections"], len(x["path"])), reverse=True)
    straight.sort(key=lambda x: (x["cardinal_diff"], len(x["path"])), reverse=True)
    return reflect, straight


def _choose_movers(reflect, straight):
    if len(reflect) < REFLECT_REQUIRED:
        return None

    chosen = []
    occupied = set()

    def can_take(item):
        return not (item["footprint"] & occupied)

    def take(item):
        chosen.append(item)
        occupied.update(item["footprint"])

    mixed = False
    if straight and random.random() < STRAIGHT_BONUS_PROB:
        for r_item in reflect:
            if not can_take(r_item):
                continue
            for s_item in straight:
                if not can_take(s_item):
                    continue
                if r_item["footprint"] & s_item["footprint"]:
                    continue
                take(r_item)
                take(s_item)
                mixed = True
                break
            if mixed:
                break

    if not chosen:
        for item in reflect:
            if can_take(item):
                take(item)
                break
    if not chosen:
        return None

    pools = [reflect, straight, reflect]
    for pool in pools:
        if len(chosen) >= MAX_MOVERS:
            break
        for item in pool:
            if len(chosen) >= MAX_MOVERS:
                break
            if item in chosen:
                continue
            if can_take(item):
                take(item)
                if len(chosen) >= 2 and random.random() < 0.45:
                    break

    if sum(item["reflections"] >= 1 for item in chosen) < REFLECT_REQUIRED:
        return None
    if not any(item["cardinal_diff"] for item in chosen):
        return None
    return chosen[:MAX_MOVERS]


def _gen_attempt():
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
    mover_colors = palette[3:]
    if not mover_colors:
        mover_colors = random.sample([c for c in range(10) if c not in (bg, wall_color, fill_color)], 1)

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
        stage1_info = _compute_stage1(grid)
    except Exception:
        return None

    nominal_out = np.array(stage1_info["out"], copy=True)
    reflect, straight = _find_mover_candidates(grid, stage1_info)
    chosen = _choose_movers(reflect, straight)
    if chosen is None:
        return None

    for i, item in enumerate(chosen):
        r, c = item["start"]
        grid[r, c] = mover_colors[i % len(mover_colors)]

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    actual_mover_mask = (grid != bg) & (grid != wall_color) & (grid != fill_color)
    if int(actual_mover_mask.sum()) != len(chosen):
        return None
    if not np.array_equal((nominal_out == fill_color), (out == fill_color)):
        return None

    reflect_hits = 0
    cardinal_diff_hits = 0
    for item in chosen:
        color = int(grid[item["start"]])
        endpoints = [(int(r), int(c)) for r, c in zip(*np.where(out == color))]
        if item["end"] not in endpoints:
            return None
        if item["reflections"] >= 1:
            reflect_hits += 1
        if item["cardinal_diff"]:
            cardinal_diff_hits += 1

    if reflect_hits < REFLECT_REQUIRED or cardinal_diff_hits < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
