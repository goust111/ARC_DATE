# Variant of b9e38dc0
# Dimension: D10 — 多步处理顺序 (multi_step)
# Change: 两步处理：第一步填充半包围区域并辐射，第二步孤立像素直线移动（第一步结果作为障碍物影响第二步）... → 逐个精灵串行处理
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     - 输入中不再只有一个半包围 wall sprite，而是 2 个同色、同向的 wall sprite。
#     - 每个 wall sprite 内部各有一个同色 fill seed blob。
#     - solver 不做“同时扩张”，而是把这些 sprite 按开口到其对应网格边界的距离从近到远串行处理。
#     - 先处理出的新填充会留在网格上，后处理 sprite 只能穿过当前仍是背景色的格子，因此会被前者截断。
#   behavior_trace:
#     - 原逻辑若直接扩展到多个 sprite，会把每个 sprite 当成独立洪水填充问题；它们只受原始输入障碍物影响。
#     - 新逻辑中，假设两个都向右开口的 sprite 共享同一条辐射通道：右侧 sprite 离右边界更近，先填满自己的扇区；
#       左侧 sprite 再扩张时，右侧已经变成非背景色，所以左侧只能在剩余背景里停止，输出明显更短。
#     - 因此排序键必须是可见的几何量，不能用颜色 id；这里使用“开口端点到对应边界的距离”。
#   complexity_calibration:
#     - 仍保留原题的 wall 几何、开口方向推断、端点射线和 obstacle shadow。
#     - 只把“单 sprite 求解”提升为“多 sprite 串行求解”，避免引入额外颜色语义。
#     - generator 固定每例 2 个同向 sprite，保证 2-3 组样例里人眼能看到“离边界更近者优先占地”的规律。
#   implicit_assumptions_review:
#     - 原代码默认 fill 颜色只有一个连通块并只服务一个 wall component；变体后必须先提取全部 fill 连通块，再按最近 wall 分配。
#     - 原代码把 fill 色整体视为可通行；变体后若继续这么做，先处理 sprite 的结果会被后处理 sprite 穿过，串行效果消失。
#       新实现只允许“当前 sprite 自己的初始 seed”通行，之前生成的 fill 会被视为已有占用。
#     - 原代码的 obstacle shadow 只看原始障碍物；变体后应把“之前 sprite 的填充结果”和“其他未处理 sprite 的实体像素”
#       一并当成当前 sprite 的障碍物，否则后处理 sprite 会错误穿过已有实体。
#   placement_legality:
#     - 两个 wall component 颜色相同但必须 8-邻域分离，seed blob 也分离。
#     - 两者始终朝同一方向开口，并沿该方向设置不同的边界距离；同时在垂直方向保持部分重叠，确保后者会与前者争夺同一填充通道。
#     - generator 用串行/并行结果差异做拒绝采样，避免退化为“虽然有两个 sprite，但互不影响”。
#   observability_and_cleanliness:
#     - 排序完全由可见几何位置决定：谁更靠近开口方向的边界，谁先处理。
#     - 颜色角色维持低复杂度：bg / wall / fill / 0-2 个 obstacle 颜色。
#     - obstacle 数量压低，避免把“sprite 间串行截断”淹没在额外噪声里。

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _manhattan_min(a_pts: list[tuple[int, int]], b_pts: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a_pts:
        for br, bc in b_pts:
            d = abs(ar - br) + abs(ac - bc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _component_neighbors(comp: set[tuple[int, int]], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for rr, cc in neighbors8(r, c):
        if (rr, cc) in comp:
            out.append((rr, cc))
    return out


def _sprite_meta(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    endpoints = []
    for r, c in comp:
        if len(_component_neighbors(comp, r, c)) == 1:
            endpoints.append((r, c))
    assert len(endpoints) >= 2, "expected wall component with at least two endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _component_neighbors(comp, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _component_neighbors(comp, *n1) if p != ep]
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

    if rad == (-1, 0):
        edge_dist = min(r for r, _ in endpoints)
    elif rad == (1, 0):
        edge_dist = (h - 1) - max(r for r, _ in endpoints)
    elif rad == (0, -1):
        edge_dist = min(c for _, c in endpoints)
    else:
        edge_dist = (w - 1) - max(c for _, c in endpoints)

    r_mean = sum(r for r, _ in comp) / len(comp)
    c_mean = sum(c for _, c in comp) / len(comp)
    return {
        "comp": comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
        "edge_dist": int(edge_dist),
        "center": (r_mean, c_mean),
    }


def _apply_shadows(
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    rad: tuple[int, int],
) -> np.ndarray:
    out_mask = filled_mask.copy()
    h, w = out_mask.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if out_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and out_mask[r, c]:
                out_mask[r, c] = False

    if rad == (0, 0):
        return out_mask

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
    return out_mask


def _infer_roles(g: np.ndarray) -> tuple[int, int, int] | None:
    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    if len(non_bg) < 2:
        return None

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    other_colors = [c for c in non_bg if c != wall]
    wall_points = [(int(r), int(c)) for r, c in zip(*np.where(g == wall))]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        return _manhattan_min(pts, wall_points), len(pts)

    fill = min(other_colors, key=lambda c: min_dist_to_wall(c))
    return bg, wall, fill


def _collect_sprites(g: np.ndarray, wall: int, fill: int) -> list[dict]:
    h, w = g.shape
    wall_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == wall, connectivity=8)
    ]
    fill_comps = [
        [(int(r), int(c)) for r, c in pts]
        for pts in components_mask(g == fill, connectivity=4)
    ]
    assert wall_comps, "expected at least one wall component"
    assert fill_comps, "expected at least one fill component"

    metas = [_sprite_meta(comp, (h, w)) for comp in wall_comps]
    assigned: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(metas))}
    for seed_comp in fill_comps:
        idx = min(
            range(len(metas)),
            key=lambda i: (
                _manhattan_min(seed_comp, list(metas[i]["comp"])),
                metas[i]["edge_dist"],
            ),
        )
        assigned[idx].extend(seed_comp)

    active = []
    for i, meta in enumerate(metas):
        if not assigned[i]:
            continue
        seed_mask = np.zeros_like(g, dtype=bool)
        for r, c in assigned[i]:
            seed_mask[r, c] = True
        active.append(
            {
                **meta,
                "seed_points": assigned[i],
                "seed_mask": seed_mask,
            }
        )

    active.sort(
        key=lambda m: (
            m["edge_dist"],
            round(m["center"][0], 3),
            round(m["center"][1], 3),
        )
    )
    return active


def _fill_for_sprite(
    occupancy: np.ndarray,
    bg: int,
    sprite: dict,
) -> np.ndarray:
    passable = (~sprite["barrier"]) & ((occupancy == bg) | sprite["seed_mask"])
    q = list(sprite["seed_points"])
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < occupancy.shape[0] and 0 <= cc < occupancy.shape[1]:
                if passable[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))

    filled_mask = seen & (occupancy == bg)
    obstacles = (occupancy != bg) & (~sprite["comp_mask"]) & (~sprite["seed_mask"])
    return _apply_shadows(filled_mask, obstacles, sprite["rad"])


def _solve_serial_core(g: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[dict], int] | None:
    roles = _infer_roles(g)
    if roles is None:
        return None
    bg, wall, fill = roles
    sprites = _collect_sprites(g, wall, fill)
    out = np.array(g, copy=True)
    per_sprite_masks = []
    for sprite in sprites:
        mask = _fill_for_sprite(out, bg, sprite)
        per_sprite_masks.append(mask)
        out[mask] = fill
    return out, per_sprite_masks, sprites, fill


def _solve_parallel_core(g: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[dict], int] | None:
    roles = _infer_roles(g)
    if roles is None:
        return None
    bg, wall, fill = roles
    sprites = _collect_sprites(g, wall, fill)
    all_fill = np.zeros_like(g, dtype=bool)
    per_sprite_masks = []
    for sprite in sprites:
        mask = _fill_for_sprite(g, bg, sprite)
        per_sprite_masks.append(mask)
        all_fill |= mask
    out = np.array(g, copy=True)
    out[all_fill] = fill
    return out, per_sprite_masks, sprites, fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill multiple partial enclosures one sprite at a time, letting earlier sprites block later ones.

    Input structure:
    - bg is the most common color. The most frequent non-bg color forms 2 same-color wall components, each a U/V-like partial enclosure.
    - A same-color fill seed blob sits inside each enclosure. Any additional non-bg colors are fixed obstacles.
    Rule:
    - Infer each wall component's opening direction from its endpoints, then sort the sprites by how close that opening is to the corresponding grid edge.
    - Process the sprites serially in that visible near-edge order: each sprite flood-fills and radiates through its own opening, but only through cells that are still background at that moment; previously created fill and existing solids block later sprites and also cast shadows along the radiate direction.

    Key insight: The puzzle is no longer a single simultaneous fill. The crucial signal is the geometric processing order: the sprite closer to its outward edge claims territory first, and that claimed territory becomes a real obstacle for the sprites behind it.
    """
    g = np.array(grid, copy=True)
    solved = _solve_serial_core(g)
    if solved is None:
        return g
    out, _masks, _sprites, _fill = solved
    return out


# --------------- generator ---------------

# Canvas size bounds.
GRID_H_VERT = (16, 22)
GRID_H_HORIZ = (12, 18)
GRID_W = (20, 24)

# Wall geometry.
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 24
DEPTH_VERT = (4, 6)
DEPTH_HORIZ = (5, 7)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement / sprite count.
SPRITE_COUNT = 2
OPEN_END_MARGIN = 3
AXIS_DISTANCE_DELTA = (6, 8)
CROSS_OFFSET_CHOICES = [-2, -1, 1, 2]

# Fill seed.
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3

# Obstacles.
MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.5
OBSTACLE_MIN_SEP = 2

# Validation.
MIN_NEW_FILL = 20
MIN_SERIAL_BLOCKED = 8
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 400


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
        max_depth = min(DEPTH_VERT[1], h - 8)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 8)
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
            )
            >= 2
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


def _grow_seed_blob(seed, interior_set):
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


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _in_bounds_all(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _axis_limit(rad, h, w):
    return h if rad[0] else w


def _cross_limit(rad, h, w):
    return w if rad[0] else h


def _anchor_from_dist(rad, h, w, depth, gap, edge_dist, cross_start):
    if rad == (-1, 0):
        return edge_dist, cross_start
    if rad == (1, 0):
        return (h - 1) - edge_dist, cross_start
    if rad == (0, -1):
        return cross_start, edge_dist
    return cross_start, (w - 1) - edge_dist


def _render_wall(local_wall, rad, anchor):
    return [_map_local(u, v, rad, anchor) for u, v in local_wall]


def _place_sprites(rad, h, w, sprites_local):
    axis_cap = _axis_limit(rad, h, w)
    cross_cap = _cross_limit(rad, h, w)
    depths = [item["depth"] for item in sprites_local]
    max_far = axis_cap - max(depths) - 3
    if max_far < OPEN_END_MARGIN + AXIS_DISTANCE_DELTA[0]:
        return None
    for _ in range(120):
        near_dist = random.randint(OPEN_END_MARGIN, OPEN_END_MARGIN + 1)
        far_min = near_dist + AXIS_DISTANCE_DELTA[0]
        far_max = min(max_far, near_dist + AXIS_DISTANCE_DELTA[1])
        if far_min > far_max:
            continue
        edge_dists = [near_dist, random.randint(far_min, far_max)]

        occupied: set[tuple[int, int]] = set()
        placed = []
        common_center = random.randint(3, cross_cap - 4)
        offsets = [0, random.choice(CROSS_OFFSET_CHOICES)]

        ok = True
        for idx, item in enumerate(sprites_local):
            depth = item["depth"]
            gap = item["gap"]
            wall = item["wall"]

            candidates = [offsets[idx]]
            around = CROSS_OFFSET_CHOICES[:]
            random.shuffle(around)
            candidates += around

            accepted = None
            for offset in candidates:
                desired_center = common_center + offset
                cross_start = desired_center - (gap // 2)
                cross_start = max(0, min(cross_start, cross_cap - gap - 1))
                anchor = _anchor_from_dist(rad, h, w, depth, gap, edge_dists[idx], cross_start)
                cells = _render_wall(wall, rad, anchor)
                if not _in_bounds_all(cells, h, w):
                    continue
                if _touches_existing(cells, occupied):
                    continue
                accepted = (anchor, cells, cross_start)
                break
            if accepted is None:
                ok = False
                break

            anchor, cells, cross_start = accepted
            occupied.update(cells)
            placed.append(
                {
                    **item,
                    "anchor": anchor,
                    "cells": cells,
                    "edge_dist": edge_dists[idx],
                    "cross_start": cross_start,
                }
            )

        if ok:
            return placed
    return None


def _choose_seed_cells(sprite):
    wall = sprite["wall"]
    depth = sprite["depth"]
    interior = _local_interior(wall, depth)
    if not interior:
        return None
    wall_set = set(wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None
    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    return [_map_local(u, v, sprite["rad"], sprite["anchor"]) for u, v in blob]


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, fill_zone):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    placed = []
    need_fill = 0
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        need_fill = 1
    for p in fill_candidates:
        if len(placed) >= need_fill:
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


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    sprites_local = []
    for _ in range(SPRITE_COUNT):
        built = _build_local_wall(rad, h, w)
        if built is None:
            return None
        wall, depth, gap = built
        sprites_local.append({"wall": wall, "depth": depth, "gap": gap, "rad": rad})

    placed = _place_sprites(rad, h, w, sprites_local)
    if placed is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for sprite in placed:
        for r, c in sprite["cells"]:
            if grid[r, c] != bg:
                return None
            grid[r, c] = wall_color

    all_seed_cells = []
    for sprite in placed:
        seed_cells = _choose_seed_cells(sprite)
        if seed_cells is None:
            return None
        for r, c in seed_cells:
            if grid[r, c] != bg:
                return None
        all_seed_cells.append(seed_cells)

    for seed_cells in all_seed_cells:
        for r, c in seed_cells:
            grid[r, c] = fill_color

    serial = _solve_serial_core(grid)
    parallel = _solve_parallel_core(grid)
    if serial is None or parallel is None:
        return None
    serial_out, serial_masks, sprites, _fill = serial
    parallel_out, _parallel_masks, _parallel_sprites, _fill2 = parallel

    if len(sprites) != SPRITE_COUNT:
        return None

    blocked_by_serial = int(((parallel_out == fill_color) & (serial_out != fill_color) & (grid == bg)).sum())
    if blocked_by_serial < MIN_SERIAL_BLOCKED:
        return None
    if any(int(mask.sum()) == 0 for mask in serial_masks):
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles:
        if not obstacle_colors:
            pool = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
            obstacle_colors = random.sample(pool, 1)
        fill_zone = (parallel_out == fill_color) & (grid == bg)
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    serial = _solve_serial_core(grid)
    parallel = _solve_parallel_core(grid)
    if serial is None or parallel is None:
        return None
    serial_out, serial_masks, sprites, _fill = serial
    parallel_out, _parallel_masks, _parallel_sprites, _fill2 = parallel

    new_fill = (serial_out == fill_color) & (grid == bg)
    blocked_by_serial = int(((parallel_out == fill_color) & (serial_out != fill_color) & (grid == bg)).sum())
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if not _touches_border(new_fill):
        return None
    if blocked_by_serial < MIN_SERIAL_BLOCKED:
        return None
    if any(int(mask.sum()) == 0 for mask in serial_masks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D10 variant input")
