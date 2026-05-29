# Combo variant of b9e38dc0 (index 22)
# combo_spec: D1/0+D3/1+D14/2
# Name: 仅辐射+间歇涂色+粗射线

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors8

# Design analysis
# spec:
#   concrete_behavior:
#     - 保留原题的对象识别方式：背景色仍是出现次数最多的颜色，墙色仍是最多的非背景色，
#       seed/fill 仍取“最靠近墙的另一种颜色”。
#     - 墙的几何结构和开口方向推断方法保持不变，但不再做 BFS 洪水填充，也不再把半包围腔体涂满。
#     - 融合后的输出只在开口外出现一束直带：以 seed 的横向/纵向对齐位置为中心，沿开口方向发射固定 3 像素宽的射线。
#     - 射线采用固定节律的虚线：深度 1、3、5... 的层保留填色，深度 2、4、6... 留成背景色。
#     - 障碍物保留原色，并按“通道阻断”处理：粗射线的 3 条并行通道分别独立前进，某一通道碰到障碍或墙后，
#       该通道更远处不再着色，其他通道继续。
#   behavior_trace:
#     - 例：若开口朝下(rad=(1,0))，seed 位于腔体中部一列。原题会先把腔体内部整块染满，再把开口外的扇区连续染色。
#     - 现在不再染内部；输出里，腔体内除了原 seed 外保持背景。开口外第 1 行会出现 3 格宽色带，第 2 行留空，
#       第 3 行再出现 3 格宽色带，如此交替直到边界。
#     - 若某列通道在第 5 层撞到障碍，则该列的第 5 层本身保留障碍色，且第 6 层以后这一路都停止；旁边两列仍可继续。
#   complexity_calibration:
#     - 虚线周期固定为 2，只引入一种明确可见的节律，避免像 D3 单变体那样再依赖开口宽度分档。
#     - 射线宽度固定为 3，且 generator 会把 seed 放在至少 3 格宽的开口走廊上，使“粗射线”在多个样例中稳定可见。
#     - seed 约束为单像素，避免多个发射源叠加成大块区域，防止“仅辐射”退化回近似填充。
#   implicit_assumptions_review:
#     - 原 solver 的 `barrier + passable + BFS` 假设“目标区域靠封闭边界定义”；这里不再成立，因此整段区域可达性逻辑必须移除。
#     - 原 solver 的阴影扫描基于连续 filled_mask；这里改成逐通道直射，因此阻挡要在射线追踪时在线处理，而不能事后扫线。
#     - “粗射线宽度为 3” 隐含要求：seed 对齐的 3 条通道必须都能穿过开口。generator 通过选择具有 3 格净空走廊的 seed 保证这一点。
#     - “区域保持空心” 隐含要求：输出不能在腔体内部新增 fill。solver 只从开口外第一层开始着色；generator 额外验证内部新增像素数为 0。
#   placement_legality:
#     - 继续沿用原题墙体构造，保证单个 8 连通墙组件、恰好两个端点、开口方向明确。
#     - seed 仅放在局部 interior 中，且其所在列/行以及相邻两条并行通道从 seed 到开口全程无墙，保证 3 宽射线物理可通过。
#     - 障碍只放在背景上，彼此保持最小 Chebyshev 间距，并在生成后再次跑 solver 校验输出有效。
#   visual_observability:
#     - 解题者从少量样例里能同时看到三个稳定现象：腔体没有被填满、开口外是隔一行/列出现的虚线、且每次出现时宽度恒为 3。
#     - 阻挡效应仍可见，但只作为次级线索；主规则不依赖颜色编号、像素遍历顺序等不可视信息。
#   visual_cleanliness:
#     - 只有一束直带参与变换，不会和原题的整块扇形填区叠在一起，视觉上比“宽扇区虚线化”更整洁。
#     - generator 会要求输出新增像素触边且数量达到阈值，避免只剩下很短的一截宽虚线而难以辨认。
#   color_inference:
#     - 不引入新颜色映射。扩张颜色始终等于 seed 颜色，墙和障碍颜色保持不变，变化完全由几何结构表达。


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points):
    """Return degree-1 pixels (exactly one 8-neighbor in the set)."""
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _scene_signature(grid: np.ndarray) -> dict[str, object]:
    """Recover the wall component, seed color, endpoints, and radiate axis."""
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = int(get_bg(g))
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
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
    assert fill_pos, "expected at least one fill pixel"

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
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[int, int]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))  # inward from endpoint
        d_use = d1
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
        return -d_use[0], -d_use[1]

    out_dirs = [outward_dir(ep) for ep in endpoints]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad in {(-1, 0), (1, 0), (0, -1), (0, 1)}, "expected cardinal radiate direction"

    dr, dc = rad
    open_proj = max(r * dr + c * dc for r, c in endpoints)
    perp_dr, perp_dc = -dc, dr
    endpoint_perps = [r * perp_dr + c * perp_dc for r, c in endpoints]
    open_min = min(endpoint_perps) + 1
    open_max = max(endpoint_perps) - 1
    assert open_min <= open_max, "expected non-empty opening interior"

    return {
        "bg": bg,
        "fill": int(fill),
        "wall": int(wall),
        "fill_pos": fill_pos,
        "rad": rad,
        "open_proj": int(open_proj),
        "perp": (int(perp_dr), int(perp_dc)),
        "open_min": int(open_min),
        "open_max": int(open_max),
        "endpoints": endpoints,
    }


def _lane_cell(rad: tuple[int, int], proj: int, perp: int) -> tuple[int, int]:
    """Map (projection along rad, perpendicular coordinate) back to grid coordinates."""
    if rad == (1, 0):
        return proj, perp
    if rad == (-1, 0):
        return -proj, perp
    if rad == (0, 1):
        return perp, proj
    return perp, -proj


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _solve(grid: np.ndarray, *, dashed: bool, width: int) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sig = _scene_signature(g)
    bg = int(sig["bg"])
    fill = int(sig["fill"])
    wall = int(sig["wall"])
    rad = sig["rad"]
    open_proj = int(sig["open_proj"])
    perp_dr, perp_dc = sig["perp"]
    open_min = int(sig["open_min"])
    open_max = int(sig["open_max"])
    fill_pos = sig["fill_pos"]

    out = g.copy()
    lane_half = width // 2
    lane_perps: set[int] = set()
    for r, c in fill_pos:
        center_perp = r * perp_dr + c * perp_dc
        for off in range(-lane_half, lane_half + 1):
            p = int(center_perp + off)
            if open_min <= p <= open_max:
                lane_perps.add(p)

    assert lane_perps, "expected at least one ray lane through the opening"

    for p in sorted(lane_perps):
        depth = 0
        blocked = False
        while True:
            depth += 1
            proj = open_proj + depth
            r, c = _lane_cell(rad, proj, p)
            if not (0 <= r < h and 0 <= c < w):
                break
            color = int(g[r, c])
            if color == wall or (color != bg and color != fill):
                blocked = True
                break
            if blocked:
                break
            if color == bg:
                if (not dashed) or (depth % 2 == 1):
                    out[r, c] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the chamber hollow and project the seed color outward as a thick dashed ray.

    Input structure:
    - The background is the most common color, and the dominant non-background color forms a single half-enclosing wall with one opening.
    - A nearby non-wall color marks the seed inside that chamber; any additional non-background pixels are fixed obstacles.

    Rule:
    - Do not flood-fill the chamber interior. Instead, use the seed as the ray source and emit a straight outward trail through the opening.
    - The trail is always 3 pixels wide and only paints every other depth layer beyond the opening; obstacles and stray wall pixels stop the affected lane while remaining unchanged.

    Key insight:
    - The wall endpoints still reveal the outward direction, but the transformed region is no longer an area fill: only a 3-wide dashed strip appears outside the opening, aligned with the seed's cross-axis position.
    """
    return _solve(grid, dashed=True, width=3)


# --------------- generator (combo D1/0 + D3/1 + D14/2) ---------------

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
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 4
OBSTACLE_MIN_SEP = 2
FORCE_BAND_OBSTACLE_PROB = 0.55

MIN_NEW_FILL = 8
MIN_HOLLOW_INTERIOR = 6
MIN_DASH_REMOVALS = 5
MIN_WIDTH_GAIN = 5
MIN_VISIBLE_BAND = 8
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


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
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _seed_candidates(local_wall, depth, gap):
    """Pick interior cells near the closed end whose 3-lane corridor to the opening is clear."""
    wall_set = set(local_wall)
    candidates = []
    for u, v in _local_interior(local_wall, depth, gap):
        if u < max(2, depth - SEED_DEPTH_LOOKBACK):
            continue
        if not (2 <= v <= gap - 2):
            continue
        ok = True
        for off in (-1, 0, 1):
            vv = v + off
            if not (1 <= vv <= gap - 1):
                ok = False
                break
            for uu in range(0, u + 1):
                if (uu, vv) in wall_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((u, v))
    return candidates


def _project_mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in zip(*np.where(mask))]


def _combo_metrics(grid: np.ndarray, interior_mask: np.ndarray):
    combo = _solve(grid, dashed=True, width=3)
    solid = _solve(grid, dashed=False, width=3)
    thin = _solve(grid, dashed=True, width=1)
    sig = _scene_signature(grid)
    fill = int(sig["fill"])

    combo_new = (combo == fill) & (grid == sig["bg"])
    solid_new = (solid == fill) & (grid == sig["bg"])
    thin_new = (thin == fill) & (grid == sig["bg"])

    removed_by_dash = solid_new & ~combo_new
    width_gain = combo_new & ~thin_new
    hollow_interior = interior_mask & (grid == sig["bg"]) & ~combo_new

    return combo, {
        "variant_new_fill": int(combo_new.sum()),
        "dash_removed": int(removed_by_dash.sum()),
        "width_gain": int(width_gain.sum()),
        "hollow_interior": int(hollow_interior.sum()),
        "interior_leak": int((combo_new & interior_mask).sum()),
        "border_touch": int(_touches_border(combo_new)),
        "visible_band": int(combo_new.sum()),
    }


def _place_obstacles(grid, obstacle_colors, band_mask):
    """Place a few singleton obstacles, preferring some inside the prospective ray band."""
    placed = []
    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles == 0:
        return

    bg = int(get_bg(grid))
    band_coords = _project_mask_coords((grid == bg) & band_mask)
    other_coords = _project_mask_coords((grid == bg) & ~band_mask)
    random.shuffle(band_coords)
    random.shuffle(other_coords)

    want_band = 0
    if band_coords and random.random() < FORCE_BAND_OBSTACLE_PROB:
        want_band = random.randint(1, min(n_obstacles, max(1, min(2, len(band_coords)))))

    for p in band_coords:
        if len(placed) >= want_band:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)

    for p in other_coords:
        if len(placed) >= n_obstacles:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)

    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


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
    obstacle_colors = palette[3:]
    if not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    candidates = _seed_candidates(local_wall, depth, gap)
    if not candidates:
        return None
    seed_local = random.choice(candidates)
    seed_r, seed_c = _map_local(seed_local[0], seed_local[1], rad, anchor)
    if grid[seed_r, seed_c] != bg:
        return None
    grid[seed_r, seed_c] = fill_color

    interior_mask = np.zeros_like(grid, dtype=bool)
    for u, v in _local_interior(local_wall, depth, gap):
        r, c = _map_local(u, v, rad, anchor)
        interior_mask[r, c] = True

    try:
        nominal = _solve(grid, dashed=True, width=3)
    except Exception:
        return None
    band_mask = (nominal == fill_color) & (grid == bg)
    _place_obstacles(grid, obstacle_colors, band_mask)

    try:
        out, metrics = _combo_metrics(grid, interior_mask)
    except Exception:
        return None
    if metrics["variant_new_fill"] < MIN_NEW_FILL:
        return None
    if metrics["visible_band"] < MIN_VISIBLE_BAND:
        return None
    if metrics["dash_removed"] < MIN_DASH_REMOVALS:
        return None
    if metrics["width_gain"] < MIN_WIDTH_GAIN:
        return None
    if metrics["interior_leak"] != 0:
        return None
    if metrics["hollow_interior"] < MIN_HOLLOW_INTERIOR:
        return None
    if not metrics["border_touch"]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
