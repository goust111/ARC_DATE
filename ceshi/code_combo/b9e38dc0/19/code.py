# Combo variant of b9e38dc0 (index 19)
# combo_spec: D1/2+D17/0
# Name: 不规则边框+边界追踪替代填充

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis for the combo variant
# combo_variant_spec:
#   concretization:
#   - 场景角色仍是: 一个主墙体、一个贴墙填色种子、若干障碍点。变化在于墙体不再接近规则 U/C 形，而是同向开口的锯齿半包围单路径。
#   - 原题在墙内先做整块洪水填充，再从开口向外喷射。这里内部改成“只给可达内轮廓描边”，外部仍保留沿开口方向的整片辐射。
#   - 不规则性的落实方式不是换成多个对象，而是在两条臂的中段加入若干内外侧锯齿，使同一深度行经常出现 3 个以上墙像素；
#     这样描边路径会贴着凹凸边界走，明显不同于规则漏斗里的光滑一圈。
#   - 具体追踪例子:
#     设开口朝上，种子在下方封闭口袋里。原逻辑会把口袋内所有背景格都染满，再穿过顶部开口把上方楔形区也染满。
#     融合后先用同一组端点射线与 BFS 找到所有可达背景格，但位于开口内侧的一部分只保留“4 邻接墙体”的那些格。
#     若左臂第 3、5 行各向内凸出一个像素，描边就会在这两行向右绕开，内部留下更多未染核心格；外部辐射区仍是实心喷射。
#   complexity_calibration:
#   - 保留四个基向开口方向以及 1:1 / 1:2 两种端点射线模式，避免变体退化成单朝向模板题。
#   - 不规则性只用少量单像素锯齿表达，并强制它们分布在多个不同行，避免自由度过高导致识别混乱。
#   - 生成器同时要求: 有足够的内部未描边核心格、足够的外辐射面积、以及多个锯齿行。这样 2-3 组样例就能看出“锯齿墙 + 空心描边”的组合规律。
#   implicit_assumptions_review:
#   - 原 `filled_mask = seen & (g == bg)` 默认内外全部实心染色；这里失效，必须拆成 `interior_boundary` 和 `radiated_mask`。
#   - 原阴影逻辑若直接作用于全部新填色，会把内部描边也擦掉；因此这里只让障碍遮挡外辐射区。
#   - 原几何里默认墙体近似规则漏斗。融合后不能再依赖 bbox 或左右对称，只能继续使用“端点 + 局部出射方向”恢复开口方向。
#   - `opening_side_masks` 仍假定两个端点位于同一开口边界线上；生成器继续用共享起点深度构造两臂来保证这一点。
#   placement_legality:
#   - 锯齿只允许加在中间深度，且每行最多加一个，这样不会封死开口，也不容易制造额外端点。
#   - 每次加锯齿后都检查: 单一 8 连通、仍恰有两个 degree-1 端点、总像素数不超过上限。
#   - 内部候选格按“骨架双臂之间”的走廊定义，再减去真实墙像素；这允许内凹锯齿侵入口袋，但不会把外侧区域误当内部。
#   visual_observability:
#   - 输出里必须同时看到: 锯齿墙、贴墙描边、未填核心区、开口外实心辐射。少任一项都会让人误判成原题或普通描边题。
#   - 生成器拒绝 `interior_gap` 太小的样例，确保人眼能看出“并不是整块填满”。
#   cleanliness_and_color:
#   - 颜色角色仍只由频率和空间关系决定，不依赖颜色 id。
#   - 障碍优先放在外辐射区或远离墙体的位置，避免墙锯齿、描边、障碍三者堆叠在同一小块区域里。
#   new_main_assumptions:
#   - 端点导出的 `rad` 是单一基方向；生成器保持两端点位于同一开口边界。
#   - 内部一定存在可达描边格和未描边核心格；生成器用 `trace` 与 `interior_gap` 下限保证这一点。
#   - 墙体虽不规则，但仍是一条单路径带局部附着像素，而不是多个独立围栏；否则端点法会失真。


GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (6, 10)
DEPTH_HORIZ = (6, 9)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 5
OBSTACLE_IN_RADIATE_PROB = 0.75
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
MIN_INTERIOR_TRACE = 8
MIN_INTERIOR_GAP = 7
MIN_RADIATED_FILL = 12
MIN_JAG_ROWS = 3
MIN_JAGGED_PIXELS = 3
PALETTE_SIZE = (3, 5)
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


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)
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
    assert other_colors, "expected at least one non-wall color (fill color)"

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
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "fill_pos": fill_pos,
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
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints (degree-1 pixels)"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
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
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "comp_mask": comp_mask,
        "wall_comp": wall_comp,
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
        "tip_12": tip_12,
    }


def _reachable_mask(
    g: np.ndarray,
    bg: int,
    fill: int,
    barrier: np.ndarray,
    fill_pos: list[tuple[int, int]],
) -> np.ndarray:
    passable = (~barrier) & ((g == bg) | (g == fill))
    q = fill_pos[:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _opening_side_masks(
    h: int,
    w: int,
    endpoints: list[tuple[int, int]],
    rad: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    ep_rows = [r for r, _ in endpoints]
    ep_cols = [c for _, c in endpoints]

    if rad == (-1, 0):
        interior = rows > max(ep_rows)
    elif rad == (1, 0):
        interior = rows < min(ep_rows)
    elif rad == (0, -1):
        interior = cols > max(ep_cols)
    elif rad == (0, 1):
        interior = cols < min(ep_cols)
    else:
        raise AssertionError("expected a cardinal radiate direction")

    interior = np.broadcast_to(interior, (h, w)).copy()
    exterior = ~interior
    return interior, exterior


def _solution_masks(grid: np.ndarray) -> dict:
    scene = _analyze_scene(grid)
    g = scene["g"]
    h, w = scene["h"], scene["w"]

    if not scene["fill_pos"]:
        zero = np.zeros((h, w), dtype=bool)
        scene.update(
            {
                "seen": zero,
                "interior_reachable": zero,
                "radiated_reachable": zero,
                "interior_boundary": zero,
                "radiated_mask": zero,
                "flood_mask": zero,
            }
        )
        return scene

    seen = _reachable_mask(g, scene["bg"], scene["fill"], scene["barrier"], scene["fill_pos"])
    interior_side, exterior_side = _opening_side_masks(h, w, scene["endpoints"], scene["rad"])
    bg_mask = g == scene["bg"]

    interior_reachable = seen & interior_side
    radiated_reachable = seen & exterior_side

    interior_boundary = np.zeros((h, w), dtype=bool)
    for r, c in zip(*np.where(interior_reachable & bg_mask)):
        if any(
            0 <= rr < h and 0 <= cc < w and scene["comp_mask"][rr, cc]
            for rr, cc in neighbors4(int(r), int(c))
        ):
            interior_boundary[r, c] = True

    scene.update(
        {
            "seen": seen,
            "interior_reachable": interior_reachable,
            "radiated_reachable": radiated_reachable,
            "interior_boundary": interior_boundary,
            "radiated_mask": radiated_reachable & bg_mask,
            "flood_mask": seen & bg_mask,
        }
    )
    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the inside contour of an irregular half-enclosure and still radiate color through its missing side.

    Input structure:
    - The background is the most common color. The dominant non-background color is a single jagged wall component whose two endpoints mark an open side.
    - A second non-background color is a seed tucked inside the pocket near the closed end. Any remaining non-background pixels are obstacles.
    Rule:
    - Reachability is still determined by the wall and its endpoint rays, but the enclosed pocket is no longer flood-filled. Only reachable background cells on the wall's inner 4-neighbor contour are painted.
    - Past the opening, the same ray-bounded region is painted solidly outward, and obstacles only erase shadows in that outward radiation zone.

    Key insight: Even with a serrated border, the two degree-1 tips still expose the opening direction; the crucial change is to separate "reachable interior" from "cells that actually touch the wall" so the pocket becomes an outline instead of a filled blob.
    """
    g = np.array(grid, copy=True)
    info = _solution_masks(g)
    if not info["fill_pos"]:
        return g

    bg = info["bg"]
    fill = info["fill"]
    wall = info["wall"]
    rad = info["rad"]

    traced_mask = info["interior_boundary"].copy()
    radiated_mask = info["radiated_mask"].copy()

    out = g.copy()
    out[traced_mask | radiated_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if radiated_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and radiated_mask[r, c]:
                out[r, c] = bg
                radiated_mask[r, c] = False

    if rad != (0, 0):
        dr, dc = rad
        if dr != 0 and dc == 0:
            for c in range(g.shape[1]):
                rows = range(g.shape[0]) if dr > 0 else range(g.shape[0] - 1, -1, -1)
                shadow_process([(r, c) for r in rows])
        elif dc != 0 and dr == 0:
            for r in range(g.shape[0]):
                cols = range(g.shape[1]) if dc > 0 else range(g.shape[1] - 1, -1, -1)
                shadow_process([(r, c) for c in cols])
        else:
            if dr == dc:
                for k in range(-(g.shape[1] - 1), g.shape[0]):
                    coords = [(r, r - k) for r in range(g.shape[0]) if 0 <= r - k < g.shape[1]]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)
            else:
                for k in range(g.shape[0] + g.shape[1] - 1):
                    coords = [(r, k - r) for r in range(g.shape[0]) if 0 <= k - r < g.shape[1]]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)

    return out


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
    profile = [start_v]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
        profile.append(v)
    return pts, profile


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
    left_pts, left_profile = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right_pts, right_profile = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)

    pts = set(left_pts) | set(right_pts)
    for v in range(left_profile[-1], right_profile[-1] + 1):
        pts.add((depth, v))
    base_wall = set(pts)

    inner_candidates = []
    outer_candidates = []
    for u in range(1, depth):
        width = right_profile[u] - left_profile[u] - 1
        if width >= 4:
            inner_candidates.append(("inner_left", (u, left_profile[u] + 1)))
            inner_candidates.append(("inner_right", (u, right_profile[u] - 1)))
        if left_profile[u] - 1 >= 0:
            outer_candidates.append(("outer_left", (u, left_profile[u] - 1)))
        if right_profile[u] + 1 <= gap:
            outer_candidates.append(("outer_right", (u, right_profile[u] + 1)))

    accepted = []
    used_rows = set()

    inner_pool = inner_candidates[:]
    random.shuffle(inner_pool)
    for kind, p in inner_pool:
        if p in pts or p[0] in used_rows:
            continue
        trial = pts | {p}
        if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
            pts = trial
            accepted.append((kind, p))
            used_rows.add(p[0])
            break

    remaining_capacity = WALL_PIXELS_MAX - len(pts)
    if remaining_capacity < MIN_JAGGED_PIXELS:
        return None

    all_pool = inner_candidates + outer_candidates
    random.shuffle(all_pool)
    target_jags = random.randint(MIN_JAGGED_PIXELS, min(5, remaining_capacity))
    for kind, p in all_pool:
        if len(accepted) >= target_jags:
            break
        if p in pts or p[0] in used_rows:
            continue
        trial = pts | {p}
        if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
            pts = trial
            accepted.append((kind, p))
            used_rows.add(p[0])

    row_counts = {}
    for u, v in pts:
        row_counts.setdefault(u, 0)
        row_counts[u] += 1
    jag_rows = sum(1 for u in range(1, depth) if row_counts.get(u, 0) >= 3)
    inner_teeth = sum(1 for kind, _ in accepted if kind.startswith("inner"))

    if len(_degree1_pixels(pts)) != 2:
        return None
    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if jag_rows < MIN_JAG_ROWS or inner_teeth < 1 or len(pts - base_wall) < MIN_JAGGED_PIXELS:
        return None

    return {
        "wall": pts,
        "depth": depth,
        "gap": gap,
        "left_profile": left_profile,
        "right_profile": right_profile,
        "tip_12": tip_12,
        "jag_rows": jag_rows,
        "jagged_pixels": len(pts - base_wall),
        "inner_teeth": inner_teeth,
    }


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


def _local_interior(left_profile, right_profile, wall, depth):
    wall_set = set(wall)
    interior = []
    for u in range(1, depth):
        lo, hi = left_profile[u], right_profile[u]
        for v in range(lo + 1, hi):
            if (u, v) not in wall_set:
                interior.append((u, v))
    return interior


def _local_outline_and_core(wall, interior):
    wall_set = set(wall)
    outline = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    outline_set = set(outline)
    core = [p for p in interior if p not in outline_set]
    return outline, core


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


def _find_obstacle_candidates(grid, bg, wall_color, radiated_zone):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    radiate_coords = [p for p in all_coords if radiated_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(radiate_coords)
    return all_coords, radiate_coords


def _place_obstacles(n_obstacles, candidates, radiate_candidates, obstacle_colors, grid):
    placed = []
    n_radiate = random.randint(0, min(n_obstacles, len(radiate_candidates)))
    if radiate_candidates and random.random() < OBSTACLE_IN_RADIATE_PROB:
        n_radiate = max(1, n_radiate)
    for p in radiate_candidates:
        if len(placed) >= n_radiate:
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


def _wall_local_rows(info: dict) -> dict[int, list[int]]:
    wall_comp = info["wall_comp"]
    endpoints = info["endpoints"]
    rad = info["rad"]

    if rad == (-1, 0):
        r0 = min(r for r, _ in endpoints)
        c0 = min(c for _, c in wall_comp)
        coords = [(r - r0, c - c0) for r, c in wall_comp]
    elif rad == (1, 0):
        r0 = max(r for r, _ in endpoints)
        c0 = min(c for _, c in wall_comp)
        coords = [(r0 - r, c - c0) for r, c in wall_comp]
    elif rad == (0, -1):
        r0 = min(r for r, _ in wall_comp)
        c0 = min(c for _, c in endpoints)
        coords = [(c - c0, r - r0) for r, c in wall_comp]
    else:
        r0 = min(r for r, _ in wall_comp)
        c0 = max(c for _, c in endpoints)
        coords = [(c0 - c, r - r0) for r, c in wall_comp]

    rows = {}
    for u, v in coords:
        rows.setdefault(int(u), []).append(int(v))
    return rows


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _solution_masks(grid)
    interior_bg = info["interior_reachable"] & (info["g"] == info["bg"])
    wall_rows = _wall_local_rows(info)
    jag_rows = sum(1 for u, vals in wall_rows.items() if 0 < u < max(wall_rows) and len(vals) >= 3)
    return {
        "trace": int(info["interior_boundary"].sum()),
        "interior_gap": int(interior_bg.sum() - info["interior_boundary"].sum()),
        "radiated": int(info["radiated_mask"].sum()),
        "flood_delta": int(info["flood_mask"].sum() - (info["interior_boundary"] | info["radiated_mask"]).sum()),
        "jag_rows": int(jag_rows),
        "tip12": bool(any(info.get("tip_12", []))),
        "rad": info["rad"],
    }


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None

    local_wall = built["wall"]
    depth = built["depth"]
    gap = built["gap"]
    interior = _local_interior(built["left_profile"], built["right_profile"], local_wall, depth)
    outline, core = _local_outline_and_core(local_wall, interior)
    if len(outline) < MIN_INTERIOR_TRACE or len(core) < MIN_INTERIOR_GAP:
        return None

    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

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
        metrics = _variant_metrics(grid)
    except Exception:
        return None
    if (
        metrics["trace"] < MIN_INTERIOR_TRACE
        or metrics["interior_gap"] < MIN_INTERIOR_GAP
        or metrics["radiated"] < MIN_RADIATED_FILL
        or metrics["jag_rows"] < MIN_JAG_ROWS
        or metrics["flood_delta"] < MIN_INTERIOR_GAP
    ):
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        info = _solution_masks(grid)
        candidates, radiate_candidates = _find_obstacle_candidates(grid, bg, wall_color, info["radiated_mask"])
        _place_obstacles(n_obstacles, candidates, radiate_candidates, obstacle_colors, grid)

    try:
        metrics = _variant_metrics(grid)
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if (
        metrics["trace"] < MIN_INTERIOR_TRACE
        or metrics["interior_gap"] < MIN_INTERIOR_GAP
        or metrics["radiated"] < MIN_RADIATED_FILL
        or metrics["jag_rows"] < MIN_JAG_ROWS
        or metrics["flood_delta"] < MIN_INTERIOR_GAP
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
