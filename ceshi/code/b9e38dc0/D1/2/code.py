# Variant of b9e38dc0
# Dimension: D1 — 主路径机制——填充与辐射 (path)
# Change: flood_fill + ray_casting——先洪水填充半包围区域的内部，再沿包围边框的开口方向向外辐射延伸... → 边界追踪替代洪水填充

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis for the D1 variant
# variant_spec:
#   concretization:
#   - 仍然只有一个主墙体组件、一个靠墙的填色种子、以及若干障碍点；开口方向与两条端点射线的几何关系不变。
#   - 原题把“墙体 + 射线”围出的整块可达区域全部染满。这个变体只改半包围内部:
#     先在开口内侧的封闭口袋里找出所有可达背景格，再只保留其中与墙体 4 邻接的那一圈，形成描边。
#   - 开口外的辐射区仍然保持整片填充，这样样例中会稳定出现“里面是空心描边，外面仍是实心喷射”的对比。
#   - 具体追踪例子:
#     原逻辑: 种子在 U 形内部，BFS 会把 U 内所有背景格都记成 filled，然后继续穿过开口把外侧楔形区也填满。
#     新逻辑: 同一批可达格先按端点所在行/列切成“开口内侧”和“开口外侧”两半；内侧只给贴墙的一圈上色，外侧仍整片上色。
#   complexity_calibration:
#   - 保留原题四个辐射方向与 1:1 / 1:2 两种端点射线模式，避免把任务简化成单一朝向。
#   - 生成器额外要求内部存在一批“非描边核心格”，保证输出不会退化成原题那种整块填充，也不会因为口袋过窄而看不出差异。
#   implicit_assumptions_review:
#   - 原 `filled_mask = seen & (g == bg)` 依赖“内部与外部都应该整块染色”。变体中这个假设失效，必须把 seen 分成内外两部分再分别处理。
#   - 原 shadow 直接作用于所有新填色格，在描边模式下会无端擦掉内部轮廓，所以这里把阴影限定在“开口外的实心辐射区”。
#   - 原 solver 里“距离墙最近的非墙色就是填色”这一点仍成立；因此生成器必须继续把障碍放得比种子更远，避免颜色角色混淆。
#   placement_legality:
#   - 生成器只接受满足以下条件的墙体: 有至少一圈可描边内轮廓，并且轮廓内还剩若干未上色核心格。
#   - 障碍优先落在外辐射区，这样既能触发原有阴影逻辑，又不会把内部描边压得过乱。
#   visual_observability:
#   - 拒绝“内部剩余未填格太少”的样例；人眼必须能从 2-3 个例子里看出，变体并不是把口袋整个涂满。
#   cleanliness_and_color:
#   - 规则完全由空间关系决定，不依赖颜色 id 大小。
#   - 颜色角色仍由频率与距离关系决定: 墙体最多，填色种子离墙最近，障碍是较远的零散像素。
#   new_main_assumptions:
#   - 端点导出的开口方向是单一基向量，且内部位于端点行/列的闭口一侧；生成器保持这一几何。
#   - 内侧描边格总能通过“与墙体 4 邻接”被识别；生成器保证这种格非空且不会被种子完全占满。


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

MAX_OBSTACLES = 5
OBSTACLE_IN_RADIATE_PROB = 0.75
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 20
MIN_INTERIOR_TRACE = 8
MIN_INTERIOR_GAP = 6
MIN_RADIATED_FILL = 12
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 300


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
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
        "tip_12": tip_12,
    }


def _reachable_mask(g: np.ndarray, bg: int, fill: int, barrier: np.ndarray, fill_pos: list[tuple[int, int]]) -> np.ndarray:
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


def _opening_side_masks(h: int, w: int, endpoints: list[tuple[int, int]], rad: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    interior = np.zeros((h, w), dtype=bool)
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
    Trace the inner contour of a partial enclosure and still spray color outward through its opening.

    Input structure:
    - The most common color is background. The most common non-background color is a wall sprite that forms a U/funnel-like partial enclosure.
    - A second non-background color appears as a seed inside the pocket near the closed end. Any further non-background pixels are fixed obstacles.
    Rule:
    - Do not flood the whole pocket. Instead, color only the reachable background cells on the inside contour that touch the wall, making an outline of the enclosed region.
    - The opening still emits the same outward ray-bounded spray as before, and obstacles only carve shadows in that outward spray.

    Key insight: The wall endpoints still reveal the opening direction, but the interior action changes from area filling to boundary tracing; only the exterior side remains a solid radiation zone.
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


def _variant_metrics(grid: np.ndarray) -> dict:
    info = _solution_masks(grid)
    interior_bg = info["interior_reachable"] & (info["g"] == info["bg"])
    return {
        "trace": int(info["interior_boundary"].sum()),
        "interior_gap": int(interior_bg.sum() - info["interior_boundary"].sum()),
        "radiated": int(info["radiated_mask"].sum()),
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
    local_wall, depth, gap = built
    interior = _local_interior(local_wall, depth)
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
    if metrics["trace"] < MIN_INTERIOR_TRACE or metrics["interior_gap"] < MIN_INTERIOR_GAP or metrics["radiated"] < MIN_RADIATED_FILL:
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
    if metrics["trace"] < MIN_INTERIOR_TRACE or metrics["interior_gap"] < MIN_INTERIOR_GAP or metrics["radiated"] < MIN_RADIATED_FILL:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D1 variant input")
