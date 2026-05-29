# Variant of b9e38dc0
# Dimension: D19 — 颜色使用规则与数量 (meta)
# Change: 每个精灵的边框一种颜色，内部源像素可能同色或不同色；孤立像素与源像素同色... → 边框颜色决定辐射行为
#
# Design analysis
# variant:
#   core_change:
#     original_rule: 一个墙色控制唯一的半包围精灵，所有端点射线默认都是 1 像素厚。
#     new_rule: 固定两种可见墙色都是真墙；蓝色边框发出 1 像素厚边界射线，红色边框发出 2 像素厚边界射线。
#     mapping_visibility: 颜色到参数的映射在整题内恒定，且每个样例同时出现蓝墙和红墙，让解题者能直接对比。
#   concretization:
#     objects:
#       - 每个输入含 2 个 U/V 形半包围墙组件，形状沿用原题的锯齿臂和 1:1 / 1:2 端点模式。
#       - 两个组件共享同一种填充色，确保差异只能从边框颜色解释。
#       - 其余非背景单点仍是障碍物，保留原题“投影阴影”行为。
#     visual_effect:
#       - 蓝墙输出与原题近似：填充从内部穿过开口，向外在两条细边界射线之间扩散。
#       - 红墙的每条边界射线向填充区域内侧再复制一条平行射线，形成 2 像素厚边界带，因此外放区域明显更窄。
#   behavior_trace:
#     small_example:
#       - 原题：若开口宽约 6 格，则两端各发一条细射线，远处横截面大致保留原本张角。
#       - 变体：同样几何下，红墙两端各多一条紧贴内侧的平行射线，远处横截面会少掉两侧各 1 格。
#       - difference: 不是换颜色，不是换方向，而是“同样开口方向下，输出辐射束宽度由墙色决定”。
#   complexity_calibration:
#     - 只保留两个档位 {1, 2}，避免颜色到连续参数的自由度过大。
#     - 每个样例只用两种真墙色 {蓝, 红}，颜色种类少，视觉对比强。
#     - 两个墙组件共用同一 fill 色，避免解题者误把差异归因于 fill 色。
#   implicit_assumption_review:
#     - 原代码 `wall = max(non_bg, key=count)` 依赖“真墙只有一种颜色”；变体中会把另一面真墙误判成 fill 或障碍，因此必须改成“按已知墙色集合逐个取大组件”。
#     - 原代码 `other_colors = [c for c in non_bg if c != wall]` 默认只有一个墙色；变体里要同时排除蓝墙和红墙。
#     - 原代码 `passable = (~barrier) & ((g == bg) | (g == fill))` 会把另一组件的 fill 种子也当作当前组件可通行区域；变体里改成“背景 or 当前 seed 组件”。
#     - 原代码 `obstacles = ... & (g != wall)` 只排除一个墙色；变体里要只排除当前墙色，否则当前墙会在阴影扫描中把自己的辐射全部挡掉。
#   placement_legality:
#     - 生成器固定每个样例放两个墙，但把它们分配到不同半区：竖向模板用左右分区，横向模板用上下分区，避免填充束互相覆盖。
#     - 两个墙复用同一个局部几何，减少“形状差异掩盖颜色差异”的风险，同时依然保留原题的随机 arm 细节。
#     - 所有 seed 仍放在各自墙内、靠近闭口端，并继续用 solver 反查验证。
#   observability:
#     - 解题者在单个样例里就能看到蓝墙和红墙输出束宽不同；跨样例再看到颜色与宽度关系保持稳定。
#     - 生成器额外拒绝“变体输出与所有墙都按 1 像素宽处理时完全相同”的样本，防止退化回原题。
#   cleanliness:
#     - 障碍数降低到少量单点，保留原题难点但不让画面变脏。
#     - 两个辐射束被放在不同空间象限，避免覆盖层次过多。

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


RAY_WIDTH_BY_WALL = {
    1: 1,  # blue wall -> thin boundary rays
    2: 2,  # red wall -> thick boundary rays
}
WALL_COLORS = tuple(RAY_WIDTH_BY_WALL)
WALL_COMPONENT_MIN = 14


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _wall_components(g: np.ndarray) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in WALL_COLORS:
        for pts in components_mask(g == color, connectivity=8):
            comp = {(int(r), int(c)) for r, c in pts}
            if len(comp) < WALL_COMPONENT_MIN:
                continue
            if len(_degree1_pixels(comp)) != 2:
                continue
            comps.append({"color": color, "points": comp})
    return comps


def _min_pair_dist(a: list[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in a:
        for rr, cc in b:
            d = abs(r - rr) + abs(c - cc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _fill_color(g: np.ndarray, bg: int, wall_comps: list[dict[str, object]]) -> int:
    wall_points = [p for comp in wall_comps for p in comp["points"]]
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    other_colors = [c for c in non_bg if c not in WALL_COLORS]
    assert other_colors, "expected at least one non-wall color"

    def score(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    return min(other_colors, key=score)


def _comp_mask(shape: tuple[int, int], comp: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _comp_neighbors(r: int, c: int, comp_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _wall_geometry(comp: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[bool], tuple[int, int]]:
    comp_mask = _comp_mask(shape, comp)
    endpoints = [p for p in comp if len(_comp_neighbors(*p, comp_mask)) == 1]
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _comp_neighbors(*ep, comp_mask)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _comp_neighbors(*n1, comp_mask) if p != ep]
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
    return endpoints, out_dirs, tip_12, rad


def _inward_offset(ep: tuple[int, int], endpoints: list[tuple[int, int]], rad: tuple[int, int]) -> tuple[int, int]:
    if rad[0] != 0:
        center = sum(c for _, c in endpoints) / len(endpoints)
        return (0, 1 if ep[1] < center else -1)
    center = sum(r for r, _ in endpoints) / len(endpoints)
    return (1 if ep[0] < center else -1, 0)


def _ray_points(start: tuple[int, int], d: tuple[int, int], rad: tuple[int, int], is_12: bool, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    pts = []
    step = 0
    while True:
        step += 1
        if not is_12:
            rr += d[0]
            cc += d[1]
        elif step % 2 == 1:
            rr += d[0]
            cc += d[1]
        elif rad[0] != 0:
            rr += d[0]
        else:
            cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        pts.append((rr, cc))
    return pts


def _apply_shadows(
    out: np.ndarray,
    g: np.ndarray,
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    rad: tuple[int, int],
    bg: int,
) -> None:
    h, w = g.shape

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

    if rad == (0, 0):
        return
    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    elif dr == dc:
        for k in range(-(w - 1), h):
            coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)
    else:
        for k in range(h + w - 1):
            coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)


def _solve(grid: np.ndarray, ray_width_by_wall: dict[int, int]) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)

    wall_comps = _wall_components(g)
    assert wall_comps, "expected at least one wall component"

    fill = _fill_color(g, bg, wall_comps)
    fill_components = []
    for pts in components_mask(g == fill, connectivity=4):
        seed = [(int(r), int(c)) for r, c in pts]
        if seed:
            fill_components.append(seed)
    if not fill_components:
        return g

    out = g.copy()
    available = list(range(len(wall_comps)))

    for seed_pts in fill_components:
        candidates = available or list(range(len(wall_comps)))
        wall_idx = min(
            candidates,
            key=lambda idx: (_min_pair_dist(seed_pts, wall_comps[idx]["points"]), -len(wall_comps[idx]["points"])),
        )
        if wall_idx in available:
            available.remove(wall_idx)
        wall_info = wall_comps[wall_idx]
        wall_color = int(wall_info["color"])
        wall_comp = wall_info["points"]
        endpoints, out_dirs, tip_12, rad = _wall_geometry(wall_comp, (h, w))
        barrier = _comp_mask((h, w), wall_comp)
        width = ray_width_by_wall.get(wall_color, 1)

        for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
            inward = _inward_offset(ep, endpoints, rad)
            for band in range(width):
                start = (ep[0] + inward[0] * band, ep[1] + inward[1] * band)
                if not (0 <= start[0] < h and 0 <= start[1] < w):
                    continue
                if band > 0:
                    barrier[start] = True
                for rr, cc in _ray_points(start, d, rad, is_12, (h, w)):
                    barrier[rr, cc] = True

        seed_mask = np.zeros_like(barrier, dtype=bool)
        for r, c in seed_pts:
            seed_mask[r, c] = True
        passable = (~barrier) & ((g == bg) | seed_mask)
        seen = np.zeros_like(passable, dtype=bool)
        q = seed_pts[:]
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
        out[filled_mask] = fill

        obstacles = (g != bg) & (g != fill) & (g != wall_color)
        _apply_shadows(out, g, filled_mask, obstacles, rad, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each half-enclosed wall sprite and radiate it outward, with ray thickness determined by wall color.

    Input structure:
    - bg is the most common color. Blue and red 8-connected wall components are the true enclosure sprites.
    - One shared fill-color seed blob sits inside each wall near its closed end; any other non-bg singletons are obstacles.

    Rule:
    - For every wall sprite, flood-fill the reachable interior and continue through the opening toward the outward direction inferred from the two endpoints.
    - Blue walls emit the original 1-pixel boundary rays, while red walls emit 2-pixel-wide boundary bands by adding one inward parallel ray per side.
    - Obstacles still cast background shadows behind themselves along that sprite's radiate direction.

    Key insight: Wall color is no longer decorative; it selects the geometric parameter of the outward radiation, so the same wall shape produces a narrower beam when its border is red.
    """
    return _solve(np.asarray(grid, dtype=int), RAY_WIDTH_BY_WALL)


# --------------- generator ---------------

GRID_H = (17, 18)
GRID_W = (19, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5
OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 3
OBSTACLE_IN_FILL_PROB = 0.5
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 24
MIN_VARIANT_DELTA = 6
GEN_RETRIES = 400


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
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


def _build_local_wall(rad: tuple[int, int], h: int, w: int) -> tuple[set[tuple[int, int]], int, int] | None:
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
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _sample_anchor(rad: tuple[int, int], depth: int, gap: int, h: int, w: int, orientation: str) -> tuple[int, int] | None:
    if orientation == "vertical":
        if rad == (-1, 0):
            row_lo, row_hi = OPEN_END_MARGIN, h - depth - 3
            col_lo, col_hi = 0, w // 2 - gap - 2
        else:
            row_lo, row_hi = depth + 2, h - OPEN_END_MARGIN - 1
            col_lo, col_hi = w // 2 + 1, w - gap - 1
    else:
        if rad == (0, -1):
            row_lo, row_hi = 0, h // 2 - gap - 1
            col_lo, col_hi = OPEN_END_MARGIN, w - depth - 3
        else:
            row_lo, row_hi = h // 2 + 1, h - gap - 1
            col_lo, col_hi = depth + 2, w - OPEN_END_MARGIN - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _render_wall(local_wall: set[tuple[int, int]], rad: tuple[int, int], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return {_map_local(u, v, rad, anchor) for u, v in local_wall}


def _choose_obstacle_colors(bg: int, fill_color: int, n_obstacles: int) -> list[int]:
    if n_obstacles <= 0:
        return []
    pool = [c for c in range(10) if c not in (bg, fill_color, *WALL_COLORS)]
    n_colors = min(max(1, n_obstacles), len(pool))
    return random.sample(pool, n_colors)


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid: np.ndarray, bg: int, wall_mask: np.ndarray, nominal_fill_zone: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
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


def _variant_delta(grid: np.ndarray) -> int:
    base = _solve(grid, {1: 1, 2: 1})
    actual = main(grid)
    return int((base != actual).sum())


def _build_two_sprite_grid() -> np.ndarray | None:
    orientation = random.choice(["vertical", "horizontal"])
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)

    proto_rad = (-1, 0) if orientation == "vertical" else (0, -1)
    built = _build_local_wall(proto_rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built

    if orientation == "vertical":
        slot_rads = [(-1, 0), (1, 0)]
    else:
        slot_rads = [(0, -1), (0, 1)]

    anchors = []
    mapped_walls = []
    for rad in slot_rads:
        anchor = _sample_anchor(rad, depth, gap, h, w, orientation)
        if anchor is None:
            return None
        anchors.append(anchor)
        mapped_walls.append(_render_wall(local_wall, rad, anchor))

    if mapped_walls[0] & mapped_walls[1]:
        return None

    bg_choices = [c for c in range(10) if c not in WALL_COLORS]
    bg = random.choice(bg_choices)
    fill_choices = [c for c in bg_choices if c != bg]
    fill_color = random.choice(fill_choices)

    grid = np.full((h, w), bg, dtype=int)
    wall_colors = list(WALL_COLORS)
    random.shuffle(wall_colors)
    for wall_pts, wall_color in zip(mapped_walls, wall_colors):
        for r, c in wall_pts:
            grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
    if not interior:
        return None
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    for rad, anchor in zip(slot_rads, anchors):
        blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
        for u, v in blob:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return None
            grid[r, c] = fill_color

    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _build_two_sprite_grid()
    if grid is None:
        return None

    bg = get_bg(grid)
    fill_color = _fill_color(grid, bg, _wall_components(grid))

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    obstacle_colors = _choose_obstacle_colors(bg, fill_color, n_obstacles)
    if n_obstacles:
        wall_mask = np.isin(grid, WALL_COLORS)
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_mask, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None
    if _variant_delta(grid) < MIN_VARIANT_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D19 input")
