# Variant of b9e38dc0
# Dimension: D1 — 主路径机制——填充与辐射 (path)
# Change: flood_fill + ray_casting——先洪水填充半包围区域的内部，再沿包围边框的开口方向向外辐射延伸... → 同心收缩填充

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import neighbors4, neighbors8, components_mask, get_bg

# Design analysis for the D1 variant:
#   concrete_behavior:
#     original_region: |
#       原题先求出一个“可达区域”: 种子在半包围墙体内启动，墙体本身和两端点向外延伸的射线
#       共同构成边界；随后该区域被单色填满，并且障碍像素会沿辐射方向在其后方切掉一整条阴影。
#     variant_region: |
#       这次不改“区域几何”，只改“区域着色”。
#       仍然先求出同一个最终可达区域，再把这个区域看成一个整体，按“距离该区域边界的层数”
#       逐层向内收缩：第 0 层涂成种子色，第 1 层保留背景色，第 2 层再涂种子色……
#       于是半包围内部和向外辐射的楔形区域都会出现同心条纹。
#     traced_example: |
#       例如某一列的最终可达区域从上到下是 6 个格子。
#       原题会把这 6 个格子全部变成 fill。
#       变体先看整个区域的边界距离；若这 6 格在该列中的层号是 0,1,2,2,1,0，
#       则输出颜色变成 fill,bg,fill,fill,bg,fill。
#       如果中间有障碍，原题是在单色填完后把障碍后方抹回背景；
#       变体若沿用该顺序会把“本来就该是 bg 的条纹”误当成未填充，阴影逻辑会失真。
#       因此必须先在几何层面删掉阴影，再对剩余区域做分层着色。
#   complexity_calibration:
#     - 只使用两种可见颜色状态: fill 与 bg，避免引入额外颜色映射。
#     - 保留原来的四个辐射方向与 1:1 / 1:2 两种墙臂模式，这样样例之间仍有重复支撑。
#     - 通过验收条件强制 max_depth >= 2 且条纹中的 bg 层数量足够，确保肉眼能看出不是纯色填充。
#   hidden_assumption_audit:
#     - 原 main() 中 `filled_mask = (seen & (g == bg))` 默认“所有可达 bg 都应变成 fill”，该前提已失效。
#     - 原 shadow_process 依赖 `filled_mask[r, c]` 判断扫描线是否进入填充区；若先条纹上色，bg 条纹会让该判断失真。
#       所以现在用“最终可达区域 region_mask”而不是“被涂成 fill 的像素”来投射阴影。
#     - 原 fill 颜色通过“离墙最近的非墙色”识别。新 generator 仍保证障碍远离墙、种子单像素贴墙，前提继续成立。
#     - 原逻辑只改 bg 像素，天然保留种子。新逻辑仍显式恢复输入中的种子色，避免小种子被偶数/奇数层覆盖。
#   placement_legality:
#     - 墙体、开口、射线边界的构造与原题一致，因此不会破坏端点识别与辐射方向求解。
#     - 种子固定为单像素，并从“紧贴墙的内部格”中选取，既保证是可达区域的一部分，也保证它处于边界层。
#     - 障碍候选继续要求与墙有距离，避免它们被误判为 fill 颜色来源，也避免直接堵死种子。
#   visual_observability:
#     - 人类可见的关键变化是: 输出中的辐射区域不再是实心，而是 fill/bg 交替的壳层。
#     - 样例会同时覆盖不同辐射方向，并保留少量障碍阴影，使“条纹区域整体被截断”这一规律也可见。
#   visual_cleanliness:
#     - 只在原本应被填充的区域内部做条纹；墙和障碍保持原色，层次清晰。
#     - generator 会筛掉条纹太浅、bg 条纹太少的样本，避免看上去像噪声。
#   color_inference:
#     - 颜色规则不依赖颜色 id；唯一变化是“可达区域内按距离边界奇偶在 fill / bg 间交替”。
#   generator_guarantees_for_new_main:
#     - 保证存在且仅需一个贴墙种子，支撑最近颜色识别。
#     - 保证最终区域触边、深度至少 2、且有可见数量的 bg 条纹，支撑新规则学习。


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neighbor distance from each True cell to the boundary of its component.
    False cells receive -1.
    """
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            boundary = False
            for rr, cc in neighbors8(r, c):
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc]:
                    boundary = True
                    break
            if boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _scene_info(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=False)
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
    assert other_colors, "expected at least one non-wall color (fill color)"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = list(zip(*np.where(g == color)))
        pts = [(int(r), int(c)) for r, c in pts]
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

    fill_pos = list(zip(*np.where(g == fill)))
    fill_pos = [(int(r), int(c)) for r, c in fill_pos]
    assert fill_pos, "expected at least one fill seed pixel"

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

    obstacles = (g != bg) & (g != fill) & (g != wall)
    region_mask = seen.copy()

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if seen[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and seen[r, c]:
                region_mask[r, c] = False

    if rad != (0, 0):
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

    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_seed_mask": g == fill,
        "region_mask": region_mask,
        "rad": rad,
        "tip_12_mode": any(tip_12),
        "obstacles": obstacles,
    }


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _scene_info(grid)
    region_mask = info["region_mask"]
    dist = _boundary_distance(region_mask)
    bg_region = region_mask & (grid == info["bg"])
    fill_stripes = bg_region & ((dist % 2) == 0)
    bg_stripes = bg_region & ((dist % 2) == 1)
    max_depth = int(dist[region_mask].max()) if region_mask.any() else -1
    return {
        "region_bg_cells": int(bg_region.sum()),
        "fill_stripe_cells": int(fill_stripes.sum()),
        "bg_stripe_cells": int(bg_stripes.sum()),
        "max_depth": max_depth,
        "touches_border": _touches_border(region_mask),
        "rad": info["rad"],
        "tip_12_mode": info["tip_12_mode"],
        "obstacle_count": int(info["obstacles"].sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the region enclosed by a wall sprite using concentric alternating bands, then radiate that banded region through the opening.

    Input structure:
    - bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure with one open end.
    - A small fill-color seed sits inside the enclosure. Other non-bg pixels are fixed obstacles and remain unchanged.

    Rule:
    - Compute the same reachable region as the original task: the wall and endpoint rays define the corridor, and obstacles cast shadows along the radiate direction.
    - Instead of making every reachable cell solid fill-color, color the final reachable region by boundary depth: boundary layer = fill color, next layer = bg, then fill again, alternating inward to create concentric stripes.

    Key insight: The wall still determines a radiating reachable shape, but the output color is controlled by each cell's distance from that shape's boundary, not just by reachability alone.
    """
    g = np.array(grid, copy=True)
    info = _scene_info(g)
    region_mask = info["region_mask"]
    dist = _boundary_distance(region_mask)

    out = g.copy()
    out[region_mask] = info["bg"]
    out[region_mask & ((dist % 2) == 0)] = info["fill"]
    out[info["fill_seed_mask"]] = info["fill"]
    return out


# --------------- generator (o5x, D1 concentric variant) ---------------

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
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_REGION_BG_CELLS = 24
MIN_FILL_STRIPE_CELLS = 12
MIN_BG_STRIPE_CELLS = 6
MIN_STRIPE_DEPTH = 2
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


def _degree1_pixels(points):
    """Return degree-1 pixels (exactly one 8-neighbor in the set)."""
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


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
    """Construct wall shape in local (u, v) coordinates. Returns (pts, depth, gap) or None."""
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


def _local_interior(wall, depth, gap):
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


def _obstacle_fits(pos, placed):
    """Check that pos is at least Chebyshev OBSTACLE_MIN_SEP from all placed obstacles."""
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_region):
    """Find obstacle candidates, preferring bg cells that participate in the patterned region."""
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_region[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    """Place up to n_obstacles on the grid, preferring some in the patterned region."""
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


def _gen_attempt():
    """Single attempt to generate a valid input grid for the concentric-fill variant."""
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
    obstacle_colors = palette[3:]

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

    seed = random.choice(seed_pool)
    r, c = _map_local(seed[0], seed[1], rad, anchor)
    if grid[r, c] != bg:
        return None
    grid[r, c] = fill_color

    try:
        nominal_region = _scene_info(grid)["region_mask"] & (grid == bg)
    except Exception:
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_region)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if not metrics["touches_border"]:
        return None
    if metrics["region_bg_cells"] < MIN_REGION_BG_CELLS:
        return None
    if metrics["fill_stripe_cells"] < MIN_FILL_STRIPE_CELLS:
        return None
    if metrics["bg_stripe_cells"] < MIN_BG_STRIPE_CELLS:
        return None
    if metrics["max_depth"] < MIN_STRIPE_DEPTH:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D1 variant input")
