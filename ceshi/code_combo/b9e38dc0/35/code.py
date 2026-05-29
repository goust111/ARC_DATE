# Combo variant of b9e38dc0 (index 35)
# combo_spec: D7/2+D10/0
# Name: 穿透+先移后填

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import neighbors4, neighbors8, components_mask, get_bg


# Design analysis for combo D7/2 + D10/0
# ---------------------------------------
# a) 具体化与行为追踪
#    - 本家族的真正两步骨架是:
#        1. 由 wall 端点推导 `radiate_dir`，fill seed 把半包围内部与开口外辐射区染成 fill。
#        2. 其余孤立像素沿同一 `radiate_dir` 运动，起点清空为背景色。
#    - D7/2 把第二步的“被第一步生成的 sprite 挡住”改成“直接穿透，不受任何精灵阻挡，直到边界”。
#    - D10/0 再把顺序反过来: 先移动孤立像素，再做 fill/radiate。
#    - 小例子: `rad=(0,1)` 向右时，某个 mover 位于未来填充楔形左侧。
#      原 blocking+先填后移: 先得到整块 fill plume；mover 向右前进，遇到 plume 就停在其左边一格。
#      D7/2 only: mover 仍在第二步运动，但会穿过 plume 到达右边界，并在 plume 上留下完整背景裂缝。
#      本融合: mover 更早移动，先留下背景轨迹，再做 fill，最后把 mover 终点画回去。
#      因而最终样例里会同时出现:
#        * mover 明显到达边界，暴露“穿透”。
#        * 同一条路径若穿过未来 plume，则 plume 内侧那段裂缝会被后续 fill 修补，只剩外侧裂缝，
#          与“先填后移”的完整裂缝不同，暴露“先移后填”。
# b) 复杂度校准
#    - 不引入多类 mover 规则；所有额外非墙非填色单像素都遵守同一条直线穿透规则。
#    - 仍保留原题的四个辐射方向与 1:1 / 1:2 两种墙臂模式，保证 fill plume 的几何多样性。
#    - 生成器固定至少 2 个 mover，且每个被选中的 mover 都满足:
#        * 若按 blocking 基线，它不会到达边界，而是会在 sprite 前停住。
#        * 若按 penetrating 路径，它会穿过至少 1 个未来 fill 单元，所以“先填后移”和“先移后填”的裂缝长度一定不同。
# c) 隐式假设审查
#    - 原单步 fill 代码把“其他非背景色”当作静态障碍或无关像素。融合后不能这样做: mover 会先离开，
#      所以 fill 计算必须基于“清空 mover 起点后的网格”，否则会错误缩小可填区域。
#    - 同时，wall 几何与 fill seed 身份仍必须从原输入解析，因为 mover 只是额外像素，不能反客为主。
#      因此 solver 先从原输入识别 wall/fill，再在“移走 mover 的底图”上做 flood fill。
#    - D7/2 取消“stage1 sprite 阻挡 mover”的前提，所以第二步不再读取 `sprite_mask` 做停靠判定；
#      mover 只看 `rad` 和边界。
# d) 放置合法性
#    - 生成器沿用原题墙体构造器，保证单个 8 连通 wall、恰好两个度 1 端点、开口朝向明确。
#    - mover 只放在背景格上，不与 wall/seed 重叠；起点彼此分离，边界终点也分离。
#    - 候选 mover 只从“直达边界时会穿过未来 fill plume”的背景格里选；这样顺序反转一定会留下可见差异。
#    - 同时要求它们在 blocking 基线下的停点早于边界终点，保证 D7/2 也一定有效。
# e) 视觉可观察性
#    - 解题者能看到: mover 到达边界，而它们在 plume 内留下的裂缝却只保留外半段。
#      边界终点对应 D7/2，裂缝被局部修补对应 D10/0。
#    - 所有判定只依赖可见几何关系: 墙的开口方向、mover 相对辐射区的位置、以及路径与 plume 的重叠；
#      不使用颜色 id 大小、黑色优先级等人类不可视因素。
# f) 可视化整洁度
#    - mover 仍保持单像素；生成器限制数量，避免整张图出现大面积杂色。
#    - 终点总是在最后一层画回，因此不会出现“输入里的 mover 全部消失”这种难读场景。
#    - 可见差异集中在几条整齐的水平/垂直背景裂缝上，而不是多层杂乱覆盖。
# g) 颜色变化可推断性
#    - 不引入新颜色映射。wall 仍是最多的非背景色，fill 仍是离墙最近的另一色，mover 颜色只表示对象身份并原样保留。
# 9) main() 的新隐式假设
#    - solver 假设 mover 不会冒充 fill: 也就是离墙最近的非墙色仍是真 seed。生成器把 fill 固定为 1 像素，
#      而所有 mover 共享同一颜色且数量 >= 2，于是即便与 seed 同为距离 1，fill 仍会因像素数更少而被正确选中。
#    - solver 把 mover 终点作为最后一层重绘；顺序反转真正改变的是 plume 内轨迹会被后来 fill 部分补回。
#    - solver 假设 mover 起点清空后不会破坏 wall/fill 结构，因为 mover 本就只放在背景格上；生成器构造保证这一点。


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

MIN_MOVERS = 2
MAX_MOVERS = 4
MOVER_MIN_TRAVEL = 4
MOVER_MIN_WALL_DIST = 1
MOVER_START_SEP = 2
MOVER_END_SEP = 2
EXTRA_MOVER_PROB = 0.35

MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 500


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


def _cheb_far_enough(pos, placed, min_sep):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in placed)


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
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
            "g": g,
            "h": h,
            "w": w,
            "bg": int(bg),
            "wall": int(wall),
            "fill": int(fill),
            "fill_pos": [],
            "comp_mask": wall_mask.copy(),
            "barrier": wall_mask.copy(),
            "rad": (0, 0),
            "wall_points": wall_points,
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

    return {
        "g": g,
        "h": h,
        "w": w,
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "fill_pos": fill_pos,
        "wall_points": wall_points,
        "comp_mask": comp_mask,
        "barrier": barrier,
        "rad": (int(rad[0]), int(rad[1])),
        "endpoints": endpoints,
    }


def _clear_movers(grid: np.ndarray, bg: int, wall: int, fill: int):
    g = np.array(grid, copy=True)
    movers = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != bg) & (g != wall) & (g != fill))
    ]
    for r, c, _col in movers:
        g[r, c] = bg
    return g, movers


def _compute_fill_mask(scene: dict, cleared: np.ndarray) -> np.ndarray:
    if not scene["fill_pos"]:
        return np.zeros_like(cleared, dtype=bool)

    bg = int(scene["bg"])
    fill = int(scene["fill"])
    barrier = np.array(scene["barrier"], copy=False)
    h, w = cleared.shape

    passable = (~barrier) & ((cleared == bg) | (cleared == fill))
    q = list(scene["fill_pos"])
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

    return seen & (cleared == bg)


def _stage_fill_state(grid: np.ndarray) -> dict:
    scene = _analyze_scene(grid)
    cleared, movers = _clear_movers(grid, scene["bg"], scene["wall"], scene["fill"])
    filled_mask = _compute_fill_mask(scene, cleared)
    stage = np.array(cleared, copy=True)
    stage[filled_mask] = scene["fill"]
    sprite_mask = np.array(scene["comp_mask"], copy=True) | (stage == scene["fill"])
    return {
        "scene": scene,
        "cleared": cleared,
        "movers": movers,
        "filled_mask": filled_mask,
        "stage": stage,
        "sprite_mask": sprite_mask,
    }


def _trace_to_border(start: tuple[int, int], rad: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = rad
    path = []
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return path
        r, c = nr, nc
        path.append((r, c))


def _first_hit(start: tuple[int, int], rad: tuple[int, int], mask: np.ndarray):
    h, w = mask.shape
    r, c = start
    dr, dc = rad
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if mask[nr, nc]:
            return int(nr), int(nc)
        r, c = nr, nc


def _end_before_hit_or_border(start: tuple[int, int], rad: tuple[int, int], mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    r, c = start
    dr, dc = rad
    last = (int(r), int(c))
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return last
        if mask[nr, nc]:
            return last
        r, c = nr, nc
        last = (int(r), int(c))


def _solve_combo(grid: np.ndarray, *, return_meta: bool = False):
    state = _stage_fill_state(grid)
    scene = state["scene"]
    out = np.array(state["cleared"], copy=True)
    rad = tuple(scene["rad"])
    mover_info = []
    endpoints = []

    for r, c, col in sorted(state["movers"]):
        path = _trace_to_border((r, c), rad, out.shape)
        end = (r, c) if not path else path[-1]
        for rr, cc in path:
            if out[rr, cc] != scene["wall"]:
                out[rr, cc] = scene["bg"]
        endpoints.append((end, col))
        mover_info.append(
            {
                "start": (int(r), int(c)),
                "end": (int(end[0]), int(end[1])),
                "path_len": int(len(path)),
                "fill_overlap": int(sum(state["filled_mask"][rr, cc] for rr, cc in path)),
            }
        )

    out[state["filled_mask"]] = scene["fill"]
    for (er, ec), col in endpoints:
        out[er, ec] = col

    if return_meta:
        return out, {
            "state": state,
            "movers": mover_info,
        }
    return out


def _solve_after_fill(grid: np.ndarray, *, penetrate: bool, return_meta: bool = False):
    state = _stage_fill_state(grid)
    scene = state["scene"]
    out = np.array(state["stage"], copy=True)
    rad = tuple(scene["rad"])
    mover_info = []

    for r, c, col in sorted(state["movers"]):
        if penetrate:
            path = _trace_to_border((r, c), rad, out.shape)
            end = (r, c) if not path else path[-1]
        else:
            end = _end_before_hit_or_border((r, c), rad, state["sprite_mask"])
            path = []
            rr, cc = r, c
            while (rr, cc) != end:
                rr += rad[0]
                cc += rad[1]
                path.append((rr, cc))
        for rr, cc in path:
            if out[rr, cc] != scene["wall"]:
                out[rr, cc] = scene["bg"]
        out[end] = col
        mover_info.append(
            {
                "start": (int(r), int(c)),
                "end": (int(end[0]), int(end[1])),
                "path_len": int(len(path)),
                "fill_overlap": int(sum(state["filled_mask"][rr, cc] for rr, cc in path)),
            }
        )

    if return_meta:
        return out, {"state": state, "movers": mover_info}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the isolated pixels first, letting them pass straight through every future obstacle, then fill and radiate the enclosure afterward.

    Input structure:
    - The background is the most common color. The most frequent non-background color is one wall component that forms a partial enclosure with a single opening direction.
    - A nearby non-wall color is the fill seed inside that enclosure. Any other non-background singleton pixels are movers that will travel in the same opening direction.

    Rule:
    - First remove each mover from its start cell and send it straight along the opening direction all the way to the grid edge without stopping for the wall or the future filled plume, clearing each travelled non-wall cell to background.
    - Then, on the mover-cleared grid, flood-fill the enclosure from the seed and radiate that fill outward through the opening just as in the base geometry, and finally redraw the moved endpoints on the border. Any trail segment inside the future plume is therefore repainted by the later fill.

    Key insight: The wall still determines one global radiate direction, but the operation order is reversed: movers carve their trails before the plume exists, so the later fill repairs only the plume-interior part of those straight trails while leaving the border endpoints in place.
    """
    return _solve_combo(np.asarray(grid, dtype=int))


def _candidate_movers(base_grid: np.ndarray):
    state = _stage_fill_state(base_grid)
    scene = state["scene"]
    bg = int(scene["bg"])
    wall = int(scene["wall"])
    rad = tuple(scene["rad"])
    if rad == (0, 0):
        return [], state

    wall_dist = distance_transform_cdt(~(base_grid == wall), metric="taxicab")
    eligible = (
        (base_grid == bg)
        & (~state["filled_mask"])
        & (wall_dist >= MOVER_MIN_WALL_DIST)
    )

    valid = []
    for r, c in zip(*np.where(eligible)):
        start = (int(r), int(c))
        path = _trace_to_border(start, rad, base_grid.shape)
        if len(path) < MOVER_MIN_TRAVEL:
            continue
        end = path[-1]
        block_end = _end_before_hit_or_border(start, rad, state["sprite_mask"])
        fill_overlap = int(sum(state["filled_mask"][rr, cc] for rr, cc in path))
        if end == block_end or fill_overlap <= 0:
            continue
        item = {
            "start": start,
            "end": (int(end[0]), int(end[1])),
            "path_len": int(len(path)),
            "fill_overlap": fill_overlap,
            "block_end": block_end,
        }
        valid.append(item)

    random.shuffle(valid)
    valid.sort(key=lambda item: (item["fill_overlap"], item["path_len"]), reverse=True)
    return valid, state


def _choose_movers(valid):
    if len(valid) < MIN_MOVERS:
        return None

    chosen = []
    starts = []
    ends = []

    def can_take(item):
        if item["start"] in starts or item["end"] in ends:
            return False
        if not _cheb_far_enough(item["start"], starts, MOVER_START_SEP):
            return False
        if not _cheb_far_enough(item["end"], ends, MOVER_END_SEP):
            return False
        return True

    for item in valid:
        if can_take(item):
            chosen.append(item)
            starts.append(item["start"])
            ends.append(item["end"])
        if len(chosen) >= MIN_MOVERS:
            break

    if len(chosen) < MIN_MOVERS:
        return None

    for item in valid:
        if len(chosen) >= MAX_MOVERS:
            break
        if item in chosen or random.random() > EXTRA_MOVER_PROB:
            continue
        if can_take(item):
            chosen.append(item)
            starts.append(item["start"])
            ends.append(item["end"])

    return chosen


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
    mover_color = palette[3] if len(palette) > 3 else random.choice([c for c in range(10) if c not in (bg, wall_color, fill_color)])

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

    u, v = random.choice(seed_pool)
    r, c = _map_local(u, v, rad, anchor)
    if grid[r, c] != bg:
        return None
    grid[r, c] = fill_color

    valid, state = _candidate_movers(grid)
    chosen = _choose_movers(valid)
    if chosen is None:
        return None

    for item in chosen:
        r, c = item["start"]
        grid[r, c] = mover_color

    try:
        combo_out, combo_meta = _solve_combo(grid, return_meta=True)
        block_out, block_meta = _solve_after_fill(grid, penetrate=False, return_meta=True)
        pen_out, pen_meta = _solve_after_fill(grid, penetrate=True, return_meta=True)
    except Exception:
        return None

    new_fill = (combo_out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    combo_map = {item["start"]: item["end"] for item in combo_meta["movers"]}
    block_map = {item["start"]: item["end"] for item in block_meta["movers"]}
    block_diff = sum(combo_map[start] != block_map[start] for start in combo_map)
    overlap_count = sum(item["fill_overlap"] > 0 for item in combo_meta["movers"])
    if block_diff < 1 or overlap_count < 1:
        return None

    if np.array_equal(combo_out, block_out):
        return None
    if np.array_equal(combo_out, pen_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo D7/2+D10/0 input")
