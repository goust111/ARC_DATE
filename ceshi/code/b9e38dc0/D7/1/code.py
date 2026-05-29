# Variant of b9e38dc0
# Dimension: D7 — 障碍物阻挡机制 (obstacle)
# Change: obstacle_blocking——第二步变换中，第一步的精灵（含填充后的区域）作为障碍物阻挡孤立像素的直线路径... → 障碍物导致反射

import os
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
#    - 第一步保持原题骨架: 一个由墙色构成的半包围漏斗, 内部有 fill seed, 沿开口方向形成一块
#      连通的 fill sprite. 这一步仍由 wall 端点推导 radiate_dir, 并用两条射线限定可达区域.
#    - 第二步不再做“障碍像素对填充投下阴影”的扫描. 相反, 输入中的其余非背景单像素视为
#      独立 mover. 它们都沿 radiate_dir 运动, 起点清空为背景色, 末端停在边界.
#    - 若 mover 的下一步将进入第一步得到的 sprite(墙 + seed + 新填充区), 它不会穿入 sprite;
#      而是像光线碰到挡板一样改为垂直方向继续走:
#        * 纵向撞击(原本 dr != 0) -> 转成水平移动
#        * 横向撞击(原本 dc != 0) -> 转成垂直移动
#      为了消除“向左还是向右/向上还是向下”的二义性, 偏转方向取“远离被撞切片中点”的那一侧,
#      也就是撞到 sprite 某一行时向该行占据区间外侧偏, 撞到某一列时向该列占据区间外侧偏.
#    - 小例子: rad=(0,1) 时, 原逻辑只会向右做阴影扫描; 若某个孤立像素位于 sprite 左边,
#      直走会撞到 sprite. 变体里它先向右走, 到 sprite 左侧一格时因下一步会进入 sprite,
#      改为向上或向下(取远离该列中点的方向)继续, 最后停在上/下边界. 原逻辑与该路径完全不同.
# b) 复杂度校准:
#    - 只保留一种核心新机制: mover 对 sprite 的 90 度反射. 不引入颜色继承、可变速率、
#      多种障碍类别等额外自由度.
#    - 仍允许 1:1 / 1:2 两种墙臂模式, 因而 sprite 轮廓变化足够大; mover 既可能直走也可能
#      反射, 但每题至少保证一个反射样例, 使 D7 的差异肉眼可见.
# c) 隐式假设审查:
#    - 原 main 中 `filled_mask` 之后的 `shadow_process` 假设 obstacle 本身静止, 且影响的是 fill.
#      该假设在变体中失效, 必须整体删除, 否则 mover 会被误当成静态遮挡物.
#    - 原 flood-fill 的 `passable = (~barrier) & ((g == bg) | (g == fill))` 依赖 obstacle 不可穿越.
#      这一点在变体中仍成立, 因为 mover 不属于第一步 sprite, 不应被填充覆盖.
#    - 选 fill 色时依赖“离墙最近的非墙色就是 seed”. 变体新增 mover 后仍必须保证 mover 远离墙,
#      否则 solver 会把 mover 颜色误判成 fill. generator 会强制 mover 的墙距离 > 2.
# d) 放置合法性:
#    - mover 只放在 nominal stage1 输出仍为背景色的位置, 避免它们占据未来 fill sprite 内部,
#      也避免第二步移动后在 sprite 上留下背景洞.
#    - generator 在放置前对每个候选起点执行同一个反射追踪器, 只接受会到达边界、不会陷入循环、
#      且 footprint 与已选 mover 不重叠的路径.
# e) 视觉可观察性:
#    - 至少放置一个发生反射的 mover, 并常配一个直走 mover 形成对照. 反射路径由 sprite 几何决定,
#      不依赖颜色 id、排序或不可见标签.
# f) 可视化整洁度:
#    - mover 保持单像素; 轨迹不着色, 输出只显示“起点消失 + 边界出现终点像素”, 避免与 fill 区域重叠成杂色.
#    - 不允许多个 mover 的 source/path/end 重叠, 降低视觉噪声.
# g) 颜色变化可推断性:
#    - mover 颜色不变化, 只是位置改变. 颜色只承担对象身份, 不参与任何映射.
# 9) main() 的新隐式假设:
#    - 反射方向通过“被撞切片中点”确定, 因此 sprite 在对应行/列上必须有可定义的占据区间.
#      对于由 stage1 生成的单个连通 sprite 该条件恒成立; generator 也只使用 solver 自身生成的 sprite.
#    - 若优先偏转侧立即越界或仍被 sprite 占据, solver 会尝试反侧; 若两侧都不可行则原地停住.
#      generator 会过滤掉这种退化候选, 确保训练样本里反射路径清晰可见.


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
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Movers (formerly static obstacles)
MAX_MOVERS = 4
MOVER_MIN_TRAVEL = 3
MOVER_MIN_WALL_DIST = 3
REFLECT_REQUIRED = 1
STRAIGHT_BONUS_PROB = 0.65

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
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
    }


def _turn_on_reflection(
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


def _trace_mover(start: tuple[int, int], d: tuple[int, int], sprite_mask: np.ndarray) -> dict:
    h, w = sprite_mask.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(d[0]), int(d[1])
    path: list[tuple[int, int]] = []
    reflections = 0
    seen = set()

    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return {"ok": False, "end": cur, "path": path, "reflections": reflections}
        seen.add(state)

        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return {"ok": True, "end": cur, "path": path, "reflections": reflections}

        if sprite_mask[nr, nc]:
            turned = _turn_on_reflection(cur, (nr, nc), (dr, dc), sprite_mask)
            if turned is None:
                return {"ok": False, "end": cur, "path": path, "reflections": reflections}
            dr, dc = turned
            reflections += 1
            continue

        cur = (int(nr), int(nc))
        path.append(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a funnel interior, then let isolated pixels travel outward and reflect off that filled sprite.

    Input structure:
    - Background is the most common color. The most frequent non-background color is a wall component forming a partial enclosure.
    - One non-wall color sits inside that enclosure as the fill seed. Any remaining non-background pixels are isolated movers outside the eventual filled sprite.

    Rule:
    - First flood-fill the reachable interior and outward radiate zone with the seed color, using the wall endpoints to define the opening direction exactly as in the base task.
    - Then move each isolated pixel in that same radiate direction, clearing its start cell. If its next step would enter the wall+fill sprite, it reflects into the perpendicular axis and continues until it reaches a grid edge.

    Key insight: The wall still determines a single outward direction for the fill, but afterwards that whole generated sprite becomes a reflective obstacle that bends the movers' straight paths.
    """
    info = _compute_stage1(np.array(grid, copy=True))
    out = np.array(info["out"], copy=True)
    bg = int(info["bg"])
    wall = int(info["wall"])
    fill = int(info["fill"])
    rad = tuple(info["rad"])
    sprite_mask = np.array(info["sprite_mask"], copy=True)

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
        traced = _trace_mover((r, c), rad, sprite_mask)
        end_r, end_c = traced["end"]
        out[end_r, end_c] = col

    return out


def _find_mover_candidates(grid, stage1_info):
    bg = int(stage1_info["bg"])
    wall = int(stage1_info["wall"])
    rad = tuple(stage1_info["rad"])
    sprite_mask = np.array(stage1_info["sprite_mask"], copy=True)

    wall_mask = grid == wall
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (~sprite_mask) & (dist >= MOVER_MIN_WALL_DIST)

    reflect = []
    straight = []
    for r, c in zip(*np.where(eligible)):
        traced = _trace_mover((int(r), int(c)), rad, sprite_mask)
        if not traced["ok"]:
            continue
        if len(traced["path"]) < MOVER_MIN_TRAVEL:
            continue
        end = traced["end"]
        if not (end[0] in (0, grid.shape[0] - 1) or end[1] in (0, grid.shape[1] - 1)):
            continue
        footprint = {(int(r), int(c)), *traced["path"]}
        item = {
            "start": (int(r), int(c)),
            "end": (int(end[0]), int(end[1])),
            "path": list(traced["path"]),
            "footprint": footprint,
            "reflections": int(traced["reflections"]),
        }
        if traced["reflections"] >= 1:
            reflect.append(item)
        else:
            straight.append(item)

    random.shuffle(reflect)
    random.shuffle(straight)
    reflect.sort(key=lambda x: (x["reflections"], len(x["path"])), reverse=True)
    straight.sort(key=lambda x: len(x["path"]), reverse=True)
    return reflect, straight


def _choose_movers(reflect, straight):
    if len(reflect) < REFLECT_REQUIRED:
        return None

    chosen = []
    occupied = set()

    def take_item(item):
        chosen.append(item)
        occupied.update(item["footprint"])

    def try_take(pool, limit):
        taken = 0
        for item in pool:
            if taken >= limit:
                break
            if item["footprint"] & occupied:
                continue
            take_item(item)
            taken += 1
        return taken

    # Prefer a mixed pair when possible so the reflection rule has an in-grid contrast case.
    if straight and random.random() < STRAIGHT_BONUS_PROB:
        paired = False
        for r_item in reflect:
            if paired:
                break
            for s_item in straight:
                if r_item["footprint"] & s_item["footprint"]:
                    continue
                take_item(r_item)
                take_item(s_item)
                paired = True
                break

    if not chosen:
        n_reflect_target = 1 + (1 if len(reflect) > 1 and random.random() < 0.45 else 0)
        if try_take(reflect, n_reflect_target) < REFLECT_REQUIRED:
            return None
    else:
        try_take(reflect, 1)

    if len(chosen) < 2:
        try_take(straight, 1)
    if len(chosen) < 2:
        try_take(reflect, 1)

    if not chosen:
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

    # Verify that the first-stage sprite is preserved and at least one mover actually reflected.
    if not np.array_equal((nominal_out == fill_color), (out == fill_color)):
        return None

    reflect_hits = 0
    for item in chosen:
        color = int(grid[item["start"]])
        endpoints = [(int(r), int(c)) for r, c in zip(*np.where(out == color))]
        if item["end"] not in endpoints:
            return None
        if item["reflections"] >= 1:
            reflect_hits += 1

    if reflect_hits < REFLECT_REQUIRED:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D7 reflection input")
