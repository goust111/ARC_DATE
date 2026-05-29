# Combo variant of b9e38dc0 (index 23)
# combo_spec: D5/2+D10/3+D12/2
# Name: 射线交汇终止+链式反应+开口相对配对

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis
# ---------------
# notes:
#   concretization_and_trace:
#     - 保留原题的半包围漏斗墙体与“由端点外推两条边界射线”这一几何核心，但场景改成
#       只有一对同色精灵，二者开口相对而立：例如左精灵向右开口，右精灵向左开口。
#     - 输入中只有先手精灵带有内部 fill 种子；后手精灵开口可见但内部为空。
#     - 小例子：左精灵先被激活。按原题，它会先填满自身内腔，再沿开口向右辐射，直到
#       右精灵的开口段被触及。此时不是“两个精灵各自独立到边界”，而是触发右精灵。
#     - 右精灵被触发后，以被击中的开口格为种子，向左反向辐射。它的两条边界射线与左
#       精灵早先射出的边界射线在中间走廊内相遇，于是后手射线在交点停下，先手射线也
#       被截断到交点；但先手已经铺开的填充不回退，因此链式因果在最终输出里仍可见。
#     - 对比原始任务：原题只处理一个墙体，且射线总到边界；本融合版必须同时追踪
#       “谁先启动、谁触发谁、第二次反向辐射在哪里撞上已有射线”。
#   complexity_calibration:
#     - 只生成 2 个精灵，避免多对配对和多重碰撞把画面搅乱，但两者共享 wall/fill 颜色，
#       足以让“相对开口 + 连锁触发 + 交汇终止”三条规则同时出现。
#     - 两个精灵的开口长度、纵向/横向对齐和 arm-angle mode 在一例中保持一致，使配对
#       关系可以由几何而非颜色值推断；同时随机化水平/垂直摆放、起手方向与 1:1/1:2 射线，
#       防止规则退化成固定模板。
#     - 生成器强制验证：必须真的触发后手、必须真的发生至少一个碰撞格、且“有碰撞”
#       与“射线到边界”的同输入输出必须不同，避免样例退化成原题或纯 D10。
#   implicit_assumptions_review:
#     - 原 main 默认“只有一个相关墙体”，因此用最近 fill 选单个 wall_comp；这里改成对
#       所有 wall component 分别建模，并选与唯一种子最近者为起始精灵。
#     - 原 main 默认“所有后续填充都能一次性用最终 barrier 求出”；这里不成立，因为
#       右精灵是否启动取决于左精灵先前是否到达其开口，所以必须按激活顺序顺推。
#     - 原 D10 参考实现允许父精灵的传播穿入未激活子精灵的内腔；面对“开口相对”时会让
#       父精灵直接灌满对子，掩盖链式触发。这里显式把未激活精灵的 interior 当作封闭区，
#       只允许其 opening 被击中。
#     - 原 D5 参考实现会在碰撞后回算最终 barrier 再整体 flood-fill；那会抹掉“先触发后
#       截断”的时序线索。此组合改为：射线状态会被截断，但已落下的早期填充不回滚。
#   placement_legality:
#     - 生成器用同一局部墙体模板镜像出一对精灵，保证二者开口长度一致、方向相反、走廊
#       宽度充足、整体不越界且墙体互不 8 邻接。
#     - 先手种子只放在其自身内腔、靠近闭口端；后手内部保持背景色，确保链起点可见。
#     - 通过求解器自检筛掉无触发、无碰撞、奇偶错位导致“对角穿叉但不同格”之类无效布局。
#   observability_and_cleanliness:
#     - 输入里最显眼的结构就是一对面对面的漏斗墙，只有一侧有种子；输出里则额外出现
#       反向回来的第二片填充区，且其远端明显比“到边界版”更早收口。
#     - 不引入额外标记色，也默认不放障碍物，避免和链式几何关系抢注意力。
#     - 颜色值不参与排序、配对或先后判断；所有关系都由“唯一内种子 + 开口朝向 + 空间位置”
#       决定，人眼可从 2-3 组样例直接归纳。
#   new_main_assumptions:
#     - 输入中相关 wall component 恰好构成一个面对面的配对；求解器支持一般互选配对，但
#       生成器固定为 2 个精灵，保证不存在歧义链路。
#     - 每个墙体仍是单个 8 连通部件且恰有两个 degree-1 端点；opening 由这两个端点确定。
#     - later sprite 的 opening 会被 earlier sprite 击中，但 earlier sprite 不会在激活前穿透
#       later sprite 的 interior；生成器与 `_reachable_from_seeds` 一起保证这一点。


GRID_H = (18, 24)
GRID_W = (18, 24)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 9)
DEPTH_HORIZ = (5, 9)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 4
PAIR_GAP_RANGE = (6, 12)
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3

GEN_RETRIES = 500
PAIR_RETRIES = 120
MIN_TOTAL_NEW_FILL = 28
MIN_CHILD_NEW_FILL = 10
MIN_COLLISION_DIFF = 8


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _coords_of_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    pts = list(zip(*np.where(grid == color)))
    return [(int(r), int(c)) for r, c in pts]


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, *, tip_12: bool = False) -> list[tuple[int, int]]:
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


def _build_local_wall(rad: tuple[int, int], h: int, w: int, *, force_tip_12: bool | None = None):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 7)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 7)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 2))

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
    if force_tip_12 is None:
        tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    elif force_tip_12:
        if depth < 2 * max_n_side:
            return None
        tip_12 = True
    else:
        tip_12 = False

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
    return pts, depth, gap, tip_12


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _map_points(points, rad: tuple[int, int], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    return [_map_local(u, v, rad, anchor) for u, v in points]


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior: list[tuple[int, int]] = []
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
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _opening_cells_from_endpoints(endpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(endpoints) == 2
    (r1, c1), (r2, c2) = sorted(endpoints)
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return [(r1, c) for c in range(lo + 1, hi)]
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return [(r, c1) for r in range(lo + 1, hi)]
    dr = _sgn(r2 - r1)
    dc = _sgn(c2 - c1)
    out = []
    rr, cc = r1 + dr, c1 + dc
    while (rr, cc) != (r2, c2):
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _interior_mask_from_opening(g: np.ndarray, wall_mask: np.ndarray, opening: list[tuple[int, int]], rad: tuple[int, int], bg: int, fill: int) -> np.ndarray:
    h, w = g.shape
    blocked = wall_mask.copy()
    for r, c in opening:
        blocked[r, c] = True
    passable = ((g == bg) | (g == fill)) & (~blocked)
    seen = np.zeros_like(blocked, dtype=bool)
    q: list[tuple[int, int]] = []
    for r, c in opening:
        seed = (r - rad[0], c - rad[1])
        if 0 <= seed[0] < h and 0 <= seed[1] < w and passable[seed[0], seed[1]] and not seen[seed[0], seed[1]]:
            seen[seed[0], seed[1]] = True
            q.append(seed)
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _component_info(comp: set[tuple[int, int]], g: np.ndarray, *, bg: int, fill: int, wall: int) -> dict:
    h, w = g.shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = sorted((r, c) for r, c in comp if len(comp_neighbors(r, c)) == 1)
    assert len(endpoints) == 2, "expected exactly two endpoints"

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
    assert rad in {(-1, 0), (1, 0), (0, -1), (0, 1)}, "expected cardinal radiate direction"

    opening = _opening_cells_from_endpoints(endpoints)
    interior = _interior_mask_from_opening(g, comp_mask, opening, rad, bg, fill)

    ray_specs = []
    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        ray_specs.append({"start": ep, "dir": d, "is_12": is_12})

    return {
        "points": frozenset(comp),
        "mask": comp_mask,
        "endpoints": tuple(endpoints),
        "opening": tuple(opening),
        "interior": interior,
        "rad": rad,
        "ray_specs": ray_specs,
    }


def _facing_distance(a: dict, b: dict) -> int | None:
    ar, ac = zip(*a["opening"])
    br, bc = zip(*b["opening"])
    rad = a["rad"]
    if rad == (0, 1):
        if b["rad"] != (0, -1):
            return None
        overlap = min(max(ar), max(br)) - max(min(ar), min(br)) + 1
        if overlap <= 0 or max(ac) >= min(bc):
            return None
        return min(bc) - max(ac)
    if rad == (0, -1):
        if b["rad"] != (0, 1):
            return None
        overlap = min(max(ar), max(br)) - max(min(ar), min(br)) + 1
        if overlap <= 0 or max(bc) >= min(ac):
            return None
        return min(ac) - max(bc)
    if rad == (1, 0):
        if b["rad"] != (-1, 0):
            return None
        overlap = min(max(ac), max(bc)) - max(min(ac), min(bc)) + 1
        if overlap <= 0 or max(ar) >= min(br):
            return None
        return min(br) - max(ar)
    if b["rad"] != (1, 0):
        return None
    overlap = min(max(ac), max(bc)) - max(min(ac), min(bc)) + 1
    if overlap <= 0 or max(br) >= min(ar):
        return None
    return min(ar) - max(br)


def _pair_components(comp_infos: list[dict]) -> dict[int, int]:
    best: dict[int, tuple[int, int]] = {}
    for i, info in enumerate(comp_infos):
        for j, other in enumerate(comp_infos):
            if i == j:
                continue
            dist = _facing_distance(info, other)
            if dist is None:
                continue
            cur = best.get(i)
            if cur is None or dist < cur[1]:
                best[i] = (j, dist)
    pair_map: dict[int, int] = {}
    for i, val in best.items():
        j, _dist = val
        other = best.get(j)
        if other is not None and other[0] == i:
            pair_map[i] = j
    return pair_map


def _extract_scene(g: np.ndarray):
    bg = int(get_bg(g))
    vals = [int(v) for v in np.unique(g)]
    non_bg = [v for v in vals if v != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected a fill color"

    wall_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == wall, connectivity=8)
    ]
    assert wall_comps, "expected wall components"

    wall_points = [p for comp in wall_comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = _coords_of_color(g, color)
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
    fill_pos = _coords_of_color(g, fill)
    if not fill_pos:
        return None

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

    comp_infos = [_component_info(comp, g, bg=bg, fill=fill, wall=wall) for comp in wall_comps]
    pair_map = _pair_components(comp_infos)
    start_idx = min(range(len(wall_comps)), key=lambda i: (comp_min_dist(wall_comps[i]), -len(wall_comps[i])))

    wall_mask = np.zeros_like(g, dtype=bool)
    for comp in wall_comps:
        for r, c in comp:
            wall_mask[r, c] = True

    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "components": comp_infos,
        "pair_map": pair_map,
        "start_idx": start_idx,
        "wall_mask": wall_mask,
    }


def _next_step(ray: dict, rad: tuple[int, int]) -> tuple[int, int]:
    if not ray["is_12"]:
        return ray["dir"]
    if ray["phase"] % 2 == 0:
        return ray["dir"]
    if rad[0] != 0:
        return (ray["dir"][0], 0)
    return (0, ray["dir"][1])


def _advance_phase(ray: dict) -> None:
    if ray["is_12"]:
        ray["phase"] = (ray["phase"] + 1) % 2


def _append_cell(state: dict, ray: dict, cell: tuple[int, int]) -> None:
    if cell in ray["index"]:
        return
    ray["index"][cell] = len(ray["path"])
    ray["path"].append(cell)
    state["cell_rays"].setdefault(cell, set()).add(ray["rid"])


def _truncate_at_cell(state: dict, rid: int, cell: tuple[int, int]) -> None:
    ray = state["rays"][rid]
    if cell not in ray["index"]:
        return
    idx = ray["index"][cell]
    tail = ray["path"][idx + 1 :]
    for old in tail:
        owners = state["cell_rays"].get(old)
        if owners is not None:
            owners.discard(rid)
            if not owners:
                state["cell_rays"].pop(old, None)
        ray["index"].pop(old, None)
    ray["path"] = ray["path"][: idx + 1]
    ray["pos"] = cell
    ray["active"] = False


def _init_ray_state() -> dict:
    return {
        "rays": {},
        "cell_rays": {},
        "comp_rays": {},
        "next_id": 0,
        "collision_cells": set(),
    }


def _register_ray(state: dict, comp_idx: int, spec: dict) -> dict:
    rid = state["next_id"]
    state["next_id"] += 1
    ray = {
        "rid": rid,
        "comp_id": comp_idx,
        "pos": spec["start"],
        "dir": spec["dir"],
        "is_12": spec["is_12"],
        "phase": 0,
        "path": [],
        "index": {},
        "active": True,
    }
    state["rays"][rid] = ray
    state["comp_rays"].setdefault(comp_idx, []).append(rid)
    return ray


def _add_rays_to_edge(scene: dict, state: dict, comp_idx: int) -> None:
    comp = scene["components"][comp_idx]
    for spec in comp["ray_specs"]:
        ray = _register_ray(state, comp_idx, spec)
        while True:
            step = _next_step(ray, comp["rad"])
            nxt = (ray["pos"][0] + step[0], ray["pos"][1] + step[1])
            if not _in_bounds(scene["wall_mask"].shape, nxt):
                ray["active"] = False
                break
            if scene["wall_mask"][nxt[0], nxt[1]]:
                ray["active"] = False
                break
            _append_cell(state, ray, nxt)
            ray["pos"] = nxt
            _advance_phase(ray)


def _add_rays_with_collisions(scene: dict, state: dict, comp_idx: int) -> None:
    comp = scene["components"][comp_idx]
    new_rays = [_register_ray(state, comp_idx, spec) for spec in comp["ray_specs"]]
    max_iters = scene["wall_mask"].shape[0] * scene["wall_mask"].shape[1] * 4
    for _ in range(max_iters):
        active = [ray for ray in new_rays if ray["active"]]
        if not active:
            return

        proposals: dict[int, tuple[int, int]] = {}
        by_cell: dict[tuple[int, int], list[int]] = {}
        for ray in active:
            step = _next_step(ray, comp["rad"])
            nxt = (ray["pos"][0] + step[0], ray["pos"][1] + step[1])
            if not _in_bounds(scene["wall_mask"].shape, nxt):
                ray["active"] = False
                continue
            if scene["wall_mask"][nxt[0], nxt[1]]:
                ray["active"] = False
                continue
            proposals[ray["rid"]] = nxt
            by_cell.setdefault(nxt, []).append(ray["rid"])

        if not proposals:
            return

        handled: set[int] = set()
        for cell, bidders in sorted(by_cell.items()):
            owners = state["cell_rays"].get(cell, set())
            owner_diff = [oid for oid in owners if state["rays"][oid]["comp_id"] != comp_idx]
            if owner_diff:
                state["collision_cells"].add(cell)
                for oid in owner_diff:
                    _truncate_at_cell(state, oid, cell)
                for rid in bidders:
                    ray = state["rays"][rid]
                    _append_cell(state, ray, cell)
                    ray["pos"] = cell
                    ray["active"] = False
                    handled.add(rid)
                continue
            if len(bidders) > 1:
                for rid in bidders:
                    ray = state["rays"][rid]
                    _append_cell(state, ray, cell)
                    ray["pos"] = cell
                    ray["active"] = False
                    handled.add(rid)

        for rid, cell in sorted(proposals.items()):
            if rid in handled:
                continue
            ray = state["rays"][rid]
            if not ray["active"]:
                continue
            owners = state["cell_rays"].get(cell)
            if owners:
                owner_diff = [oid for oid in owners if state["rays"][oid]["comp_id"] != comp_idx]
                if owner_diff:
                    state["collision_cells"].add(cell)
                    for oid in owner_diff:
                        _truncate_at_cell(state, oid, cell)
                    _append_cell(state, ray, cell)
                    ray["pos"] = cell
                    ray["active"] = False
                    continue
                ray["active"] = False
                continue
            _append_cell(state, ray, cell)
            ray["pos"] = cell
            _advance_phase(ray)
    raise AssertionError("ray simulation did not converge")


def _build_component_barrier(scene: dict, state: dict, comp_idx: int) -> np.ndarray:
    barrier = scene["components"][comp_idx]["mask"].copy()
    for rid in state["comp_rays"].get(comp_idx, []):
        for r, c in state["rays"][rid]["path"]:
            barrier[r, c] = True
    return barrier


def _reachable_from_seeds(out: np.ndarray, barrier: np.ndarray, seeds: list[tuple[int, int]], inactive_masks: list[np.ndarray], bg: int, fill: int) -> np.ndarray:
    blocked = barrier.copy()
    for mask in inactive_masks:
        blocked |= mask
    passable = (~blocked) & ((out == bg) | (out == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q: list[tuple[int, int]] = []
    for r, c in seeds:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and passable[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _apply_shadow(orig: np.ndarray, reach: np.ndarray, *, bg: int, fill: int, wall: int, rad: tuple[int, int]) -> np.ndarray:
    out = reach.copy()
    obstacles = (orig != bg) & (orig != fill) & (orig != wall)
    h, w = orig.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if out[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and out[r, c]:
                out[r, c] = False

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


def _simulate_combo(grid: np.ndarray, *, stop_on_collision: bool, want_details: bool = False):
    g = np.array(grid, copy=True)
    scene = _extract_scene(g)
    if scene is None:
        return g if not want_details else (g, {})

    pair_map = scene["pair_map"]
    start_idx = scene["start_idx"]
    assert start_idx in pair_map, "expected seeded component to have a facing partner"

    out = g.copy()
    state = _init_ray_state()
    activated: set[int] = {start_idx}
    activation_order = [start_idx]
    parent: dict[int, int | None] = {start_idx: None}
    queue: list[tuple[int, list[tuple[int, int]]]] = [(start_idx, scene["fill_pos"][:])]
    propagation_masks = [np.zeros_like(g, dtype=bool) for _ in scene["components"]]
    new_fill_counts = [0 for _ in scene["components"]]
    trigger_hits: dict[tuple[int, int], int] = {}

    while queue:
        idx, seeds = queue.pop(0)
        if stop_on_collision:
            _add_rays_with_collisions(scene, state, idx)
        else:
            _add_rays_to_edge(scene, state, idx)

        barrier = _build_component_barrier(scene, state, idx)
        inactive_masks = [
            scene["components"][j]["interior"]
            for j in range(len(scene["components"]))
            if j != idx and j not in activated
        ]
        reach = _reachable_from_seeds(
            out,
            barrier,
            seeds,
            inactive_masks,
            scene["bg"],
            scene["fill"],
        )
        prop = _apply_shadow(
            g,
            reach,
            bg=scene["bg"],
            fill=scene["fill"],
            wall=scene["wall"],
            rad=scene["components"][idx]["rad"],
        )
        propagation_masks[idx] = prop
        new_fill = prop & (out == scene["bg"])
        new_fill_counts[idx] = int(new_fill.sum())
        out[new_fill] = scene["fill"]

        partner = pair_map.get(idx)
        if partner is not None and partner not in activated:
            hits = [(r, c) for r, c in scene["components"][partner]["opening"] if prop[r, c]]
            if hits:
                activated.add(partner)
                activation_order.append(partner)
                parent[partner] = idx
                queue.append((partner, hits))
                trigger_hits[(idx, partner)] = len(hits)

    if not want_details:
        return out
    return out, {
        "scene": scene,
        "activated": activated,
        "activation_order": activation_order,
        "parent": parent,
        "propagation_masks": propagation_masks,
        "new_fill_counts": new_fill_counts,
        "collision_cells": set(state["collision_cells"]),
        "trigger_hits": trigger_hits,
        "ray_state": state,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let a seeded funnel trigger its opposite-facing partner, then stop the return rays at their first meeting point.

    Input structure:
    - The most common color is the background, and the dominant non-background color forms a facing pair of partial-wall sprites.
    - Exactly one sprite contains the fill-color seed blob near its closed end; its partner shares the same wall geometry but starts empty, with its opening directly facing the first sprite.
    - Any extra non-background colors behave as fixed obstacles, though the generated combo examples keep the scene minimal.

    Rule:
    - The seeded sprite fills its own pocket and radiates outward through its opening using the usual endpoint rays as boundaries.
    - When that first radiation reaches the partner's opening cells, it activates the partner: those touched opening cells become the partner's seeds, and the partner radiates back in the opposite direction.
    - A returning ray stops at the first cell already occupied by a ray from the other sprite; the older ray is truncated to that meeting cell too. Obstacles still cast background shadows along each sprite's own radiate direction.

    Key insight:
    - The output depends on temporal interaction, not just static geometry: identify the only seeded sprite, follow which opposite-facing opening it ignites, and then cap the second sprite's reverse wedge where its rays collide with the first sprite's rays.
    """
    return _simulate_combo(grid, stop_on_collision=True, want_details=False)


def _make_component(rad: tuple[int, int], h: int, w: int, *, force_tip_12: bool | None = None):
    built = _build_local_wall(rad, h, w, force_tip_12=force_tip_12)
    if built is None:
        return None
    local_wall, depth, gap, tip_12 = built
    interior = _local_interior(local_wall, depth)
    return {
        "rad": rad,
        "wall": local_wall,
        "depth": depth,
        "gap": gap,
        "tip_12": tip_12,
        "interior": interior,
        "opening_local": [(0, v) for v in range(1, gap)],
    }


def _mapped_component_cells(comp_def: dict, anchor: tuple[int, int]) -> dict:
    wall = _map_points(comp_def["wall"], comp_def["rad"], anchor)
    interior = _map_points(comp_def["interior"], comp_def["rad"], anchor)
    opening = _map_points(comp_def["opening_local"], comp_def["rad"], anchor)
    return {"wall": wall, "interior": interior, "opening": opening}


def _all_in_bounds(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in points)


def _wall_is_separate(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    aset = set(a)
    for r, c in b:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in aset:
                    return False
    return True


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _paint_points(grid: np.ndarray, pts: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in pts:
        grid[r, c] = color


def _anchor_pair(horizontal: bool, start_left_or_top: bool, depth: int, gap: int, h: int, w: int):
    corridor = random.randint(*PAIR_GAP_RANGE)
    if horizontal:
        row0 = random.randint(1, h - gap - 2)
        if start_left_or_top:
            min_left = depth + 2
            max_left = w - depth - corridor - 3
            if min_left > max_left:
                return None
            c_left = random.randint(min_left, max_left)
            c_right = c_left + corridor
            if c_right < OPEN_END_MARGIN or c_right + depth > w - 3:
                return None
            return (row0, c_left), (row0, c_right)
        min_right = depth + 2
        max_right = w - depth - corridor - 3
        if min_right > max_right:
            return None
        c_right = random.randint(min_right, max_right)
        c_left = c_right + corridor
        if c_left < OPEN_END_MARGIN or c_left + depth > w - 3:
            return None
        return (row0, c_left), (row0, c_right)

    col0 = random.randint(1, w - gap - 2)
    if start_left_or_top:
        min_top = depth + 2
        max_top = h - depth - corridor - 3
        if min_top > max_top:
            return None
        r_top = random.randint(min_top, max_top)
        r_bottom = r_top + corridor
        if r_bottom < OPEN_END_MARGIN or r_bottom + depth > h - 3:
            return None
        return (r_top, col0), (r_bottom, col0)
    min_bottom = depth + 2
    max_bottom = h - depth - corridor - 3
    if min_bottom > max_bottom:
        return None
    r_bottom = random.randint(min_bottom, max_bottom)
    r_top = r_bottom + corridor
    if r_top < OPEN_END_MARGIN or r_top + depth > h - 3:
        return None
    return (r_top, col0), (r_bottom, col0)


def _build_pair_grid():
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    bg, wall_color, fill_color = random.sample(range(10), 3)
    horizontal = random.random() < 0.5
    start_left_or_top = random.random() < 0.5

    if horizontal:
        start_rad = (0, 1) if start_left_or_top else (0, -1)
        partner_rad = (0, -1) if start_left_or_top else (0, 1)
    else:
        start_rad = (1, 0) if start_left_or_top else (-1, 0)
        partner_rad = (-1, 0) if start_left_or_top else (1, 0)

    start_def = _make_component(start_rad, h, w, force_tip_12=None)
    if start_def is None:
        return None
    partner_def = {
        "rad": partner_rad,
        "wall": set(start_def["wall"]),
        "depth": start_def["depth"],
        "gap": start_def["gap"],
        "tip_12": start_def["tip_12"],
        "interior": list(start_def["interior"]),
        "opening_local": list(start_def["opening_local"]),
    }

    anchors = _anchor_pair(horizontal, start_left_or_top, start_def["depth"], start_def["gap"], h, w)
    if anchors is None:
        return None
    start_anchor, partner_anchor = anchors

    mapped_start = _mapped_component_cells(start_def, start_anchor)
    mapped_partner = _mapped_component_cells(partner_def, partner_anchor)
    footprint = mapped_start["wall"] + mapped_start["interior"] + mapped_start["opening"]
    footprint += mapped_partner["wall"] + mapped_partner["interior"] + mapped_partner["opening"]
    if not _all_in_bounds(footprint, h, w):
        return None
    if not _wall_is_separate(mapped_start["wall"], mapped_partner["wall"]):
        return None

    grid = np.full((h, w), bg, dtype=int)
    _paint_points(grid, mapped_start["wall"], wall_color)
    _paint_points(grid, mapped_partner["wall"], wall_color)

    wall_set = set(start_def["wall"])
    adjacent = [
        p
        for p in start_def["interior"]
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, start_def["depth"] - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None
    blob_local = _grow_seed_blob(random.choice(seed_pool), set(start_def["interior"]))
    blob = _map_points(blob_local, start_def["rad"], start_anchor)
    if any(grid[r, c] != bg for r, c in blob):
        return None
    _paint_points(grid, blob, fill_color)
    return grid


def _gen_attempt():
    for _ in range(PAIR_RETRIES):
        grid = _build_pair_grid()
        if grid is None:
            continue
        try:
            out_collision, details_collision = _simulate_combo(grid, stop_on_collision=True, want_details=True)
            out_edge, details_edge = _simulate_combo(grid, stop_on_collision=False, want_details=True)
        except Exception:
            continue

        if len(details_collision["activated"]) != 2:
            continue
        if len(details_collision["scene"]["components"]) != 2:
            continue
        if len(details_collision["collision_cells"]) < 1:
            continue

        start_idx = details_collision["scene"]["start_idx"]
        partner_idx = details_collision["scene"]["pair_map"].get(start_idx)
        if partner_idx is None:
            continue
        if details_collision["parent"].get(partner_idx) != start_idx:
            continue
        if details_collision["trigger_hits"].get((start_idx, partner_idx), 0) < 1:
            continue

        diff_pixels = int(np.count_nonzero(out_collision != out_edge))
        if diff_pixels < MIN_COLLISION_DIFF:
            continue
        if details_collision["new_fill_counts"][partner_idx] < MIN_CHILD_NEW_FILL:
            continue

        new_fill = (out_collision == details_collision["scene"]["fill"]) & (grid == details_collision["scene"]["bg"])
        if int(new_fill.sum()) < MIN_TOTAL_NEW_FILL:
            continue
        if not _touches_border(new_fill):
            continue

        if len(details_edge["collision_cells"]) != 0:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input")
