# Combo variant of b9e38dc0 (index 38)
# combo_spec: D6/0+D17/2
# Name: 圆弧形半包围+多开口围栏

import math
import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis
# a) 具体化与行为追踪:
#    这里把原题的“直线 U/C 形单开口围栏”改成“向右张开的半圆环围栏”。墙由内外两条近似同心的
#    阶梯圆弧和上下两条短连接桥组成，视觉上像一个圆弧形的细围栏通道。填充种子放在两条弧之间的
#    通道里。原题是从唯一开口整体向一个方向辐射；融合后固定保留两个开口: 一个开在外弧上，一个开在
#    上桥或下桥上。外弧缺口会根据所在带区朝右、右下或右上辐射，桥缺口则朝上或朝下辐射。输出先灌满
#    弧形通道，再从两个开口分别漏出成两条方向不同的条带。与原题相比，开口识别从“找两个端点”变成
#    “按半圆理论外轮廓找缺失弧段，再额外检查上下桥上的缺口”，辐射也从单一方向变成多方向并集。
# b) 复杂度校准:
#    如果允许任意朝向的任意曲线围栏，solver 需要做完整曲线重建，生成成功率会明显下降。这里收敛到
#    固定朝右的半圆环，并固定成“1 个外弧缺口 + 1 个桥缺口”的双开口结构，但保留缺口所在带区、
#    上下桥位置、颜色和障碍的变化。这样视觉上仍能明显看出“圆弧 + 多开口”，同时样例里反复出现
#    “弧向辐射”和“竖向辐射”的组合，方便人类从少量样例归纳。
# c) 隐式假设审查:
#    原 main() 依赖的关键前提都失效了:
#    - degree-1 端点不存在: 半圆环有桥接，外弧缺口不会产生原题那种两个端点。
#    - 单一 radiate_dir 失效: 外弧缺口和桥缺口的方向不同，不能合并成一个方向。
#    - 由端点射出两条斜边的几何失效: 现在应由每个缺口两侧的墙尖端分别发射平行射线，形成多条独立带区。
#    因此 solver 改为依赖生成器保证的“外框=理论半圆 bbox”，先按理论外弧逐行检查缺失像素，再检查
#    上下桥上的缺口，然后对每个缺口分别建立边界和阴影扫描带。
# d) 放置合法性:
#    生成器固定使用输出优先构造。先在局部坐标里画外弧、内弧和上下桥，确保是单个 8 连通墙组件；再只
#    在内弧跨度范围内挖外弧缺口，避免桥被破坏。对象放到全局网格时保证右侧有至少 3 格外部空间，缺口
#    射线必定能打到边界。障碍只放在背景上，且与墙保持距离，不与缺口粘连。
# e) 视觉可观察性:
#    人类能直接看见两件新现象: 第一，围栏不再是直角 U，而是近似圆弧通道；第二，输出里会同时出现
#    一条沿桥缺口竖直泄漏的条带，以及一条沿外弧缺口向右侧泄漏的条带。缺口位置完全由形状决定，与
#    颜色值大小无关。样例间重复出现“桥缺口 + 弧缺口”的组合，足以让解题者从少量例子中总结“每个
#    缺口都按自己的朝向独立发射条带”。
# f) 可视化整洁度:
#    开口总数固定为 2，且桥缺口与弧缺口分属不同部位，能避免大量射线互相交叉。外部填充被两条平行
#    边界射线约束成干净的窄带，不会演化成整片外部背景被全部染色。障碍仍是少量孤立像素，视觉层次
#    保持清楚。
# g) 颜色变化可推断性:
#    颜色逻辑沿用原题: 墙色是最显眼的非背景色，离墙最近的另一种颜色是填充色，其他颜色是障碍并保持
#    不变。唯一的颜色变化仍是“背景 -> 填充色”，而变化区域明显由弧形通道与多条缺口带共同决定。
# h) 新增 solver 假设审查:
#    新 solver 的额外假设包括:
#    - 目标墙组件的 bbox 正好包住完整外半圆，且朝右张开。
#    - 外弧逐行理论列坐标可由 bbox 半径公式重建；一个缺口只发生在外弧上，另一个只发生在上桥或下桥。
#    - 桥缺口不碰到桥两端，外弧缺口不贴近桥行，因而 solver 能把两类开口分开识别。
#    - 障碍是非墙非填充的少量散点。
#    生成器逐条保证这些条件，并在验证阶段拒绝不满足的样本。


# -- Configuration constants --------------------------------------------------
GRID_H = (15, 18)
GRID_W = (16, 19)

R_OUT = (7, 7)
RING_THICKNESS = (4, 4)
RIGHT_MARGIN = 3
LEFT_MARGIN = 2
VERT_MARGIN = 0

OPEN_LEN = (2, 2)
MIN_OPENINGS = 2
MAX_OPENINGS = 2
BRIDGE_OPEN_LEN = (1, 1)

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
MAX_OBSTACLES = 4
OBSTACLE_IN_FILL_PROB = 0.55
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 18
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 500
MIN_MULTI_DELTA = 8


def _component_mask(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _flood_from_seeds(passable: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros_like(passable, dtype=bool)
    q = seeds[:]
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _select_wall_and_fill(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid)
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
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

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill-color seed pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    return bg, wall, fill, fill_pos, wall_comp


def _outer_col_from_bbox(row: int, r0: int, c0: int, radius: int) -> int:
    dy = row - (r0 + radius)
    inside = max(0, radius * radius - dy * dy)
    return c0 + int(round(math.sqrt(inside)))


def _opening_dir(mid_row: int, center_row: int) -> tuple[int, int]:
    if mid_row <= center_row - 2:
        return (-1, 1)
    if mid_row >= center_row + 2:
        return (1, 1)
    return (0, 1)


def _find_arc_openings(comp_mask: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int], int]:
    coords = np.argwhere(comp_mask)
    assert coords.size, "expected non-empty wall component"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    radius = (r1 - r0) // 2
    assert radius >= 4 and r1 - r0 == 2 * radius, "expected odd-height outer semicircle bbox"
    center_row = r0 + radius
    inner_radius = radius - RING_THICKNESS[0]

    missing_rows: list[int] = []
    for r in range(r0 + 1, r1):
        outer_c = _outer_col_from_bbox(r, r0, c0, radius)
        if outer_c <= c0 + 1:
            continue
        if not comp_mask[r, outer_c]:
            missing_rows.append(r)

    runs: list[tuple[int, int]] = []
    for r in missing_rows:
        if not runs or r != runs[-1][1] + 1:
            runs.append((r, r))
        else:
            runs[-1] = (runs[-1][0], r)

    openings: list[dict] = []
    for a, b in runs:
        if a <= r0 + 1 or b >= r1 - 1:
            continue
        tip_top = (a - 1, _outer_col_from_bbox(a - 1, r0, c0, radius))
        tip_bot = (b + 1, _outer_col_from_bbox(b + 1, r0, c0, radius))
        if not comp_mask[tip_top] or not comp_mask[tip_bot]:
            continue
        mid = (a + b) // 2
        rad = _opening_dir(mid, center_row)
        openings.append(
            {
                "kind": "arc",
                "rows": (a, b),
                "tips": (tip_top, tip_bot),
                "rad": rad,
            }
        )

    for bridge_row, rad in ((center_row - inner_radius, (-1, 0)), (center_row + inner_radius, (1, 0))):
        outer_c = _outer_col_from_bbox(bridge_row, r0, c0, radius)
        missing_cols: list[int] = []
        for c in range(c0 + 1, outer_c):
            if not comp_mask[bridge_row, c]:
                missing_cols.append(c)
        if not missing_cols:
            continue
        col_runs: list[tuple[int, int]] = []
        for c in missing_cols:
            if not col_runs or c != col_runs[-1][1] + 1:
                col_runs.append((c, c))
            else:
                col_runs[-1] = (col_runs[-1][0], c)
        for a, b in col_runs:
            if a <= c0 or b >= outer_c:
                continue
            left_tip = (bridge_row, a - 1)
            right_tip = (bridge_row, b + 1)
            if not comp_mask[left_tip] or not comp_mask[right_tip]:
                continue
            openings.append(
                {
                    "kind": "bridge",
                    "cols": (a, b),
                    "tips": (left_tip, right_tip),
                    "rad": rad,
                }
            )

    return openings, (r0, c0, r1, c1), center_row


def _add_ray(barrier: np.ndarray, start: tuple[int, int], d: tuple[int, int]) -> None:
    h, w = barrier.shape
    rr, cc = start
    while True:
        rr += d[0]
        cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        barrier[rr, cc] = True


def _shadow_lines_for_opening(opening: dict, h: int, w: int) -> list[list[tuple[int, int]]]:
    tip_top, tip_bot = opening["tips"]
    d = opening["rad"]
    lines: list[list[tuple[int, int]]] = []

    if d == (-1, 0):
        for c in range(tip_top[1], tip_bot[1] + 1):
            lines.append([(r, c) for r in range(h - 1, -1, -1)])
        return lines

    if d == (1, 0):
        for c in range(tip_top[1], tip_bot[1] + 1):
            lines.append([(r, c) for r in range(h)])
        return lines

    if d == (0, 1):
        for r in range(tip_top[0], tip_bot[0] + 1):
            lines.append([(r, c) for c in range(w)])
        return lines

    if d == (-1, 1):
        lo = min(tip_top[0] + tip_top[1], tip_bot[0] + tip_bot[1])
        hi = max(tip_top[0] + tip_top[1], tip_bot[0] + tip_bot[1])
        for s in range(lo, hi + 1):
            coords = [(r, s - r) for r in range(h) if 0 <= s - r < w]
            coords.sort(key=lambda rc: rc[1])
            if coords:
                lines.append(coords)
        return lines

    lo = min(tip_top[0] - tip_top[1], tip_bot[0] - tip_bot[1])
    hi = max(tip_top[0] - tip_top[1], tip_bot[0] - tip_bot[1])
    for diff in range(lo, hi + 1):
        coords = [(r, r - diff) for r in range(h) if 0 <= r - diff < w]
        coords.sort(key=lambda rc: rc[1])
        if coords:
            lines.append(coords)
    return lines


def _apply_shadows(
    out: np.ndarray,
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    openings: list[dict],
    bg: int,
) -> None:
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

    h, w = out.shape
    for opening in openings:
        for coords in _shadow_lines_for_opening(opening, h, w):
            shadow_process(coords)


def _solve_core(grid: np.ndarray, keep_all_openings: bool) -> np.ndarray:
    g = np.array(grid, copy=True)
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(g)

    comp_mask = _component_mask(wall_comp, g.shape)
    openings, _bbox, _center_row = _find_arc_openings(comp_mask)
    assert len(openings) >= 2, "expected two or more arc openings"

    if not keep_all_openings:
        openings = [sorted(openings, key=lambda op: (op["tips"][0][0], op["tips"][0][1]))[0]]

    barrier = comp_mask.copy()
    for opening in openings:
        for tip in opening["tips"]:
            _add_ray(barrier, tip, opening["rad"])

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = _flood_from_seeds(passable, fill_pos)
    filled_mask = seen & (g == bg)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    _apply_shadows(out, filled_mask, obstacles, openings, bg)
    return out


def analyze_primary_arc(grid: np.ndarray) -> dict:
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(grid)
    comp_mask = _component_mask(wall_comp, grid.shape)
    openings, bbox, center_row = _find_arc_openings(comp_mask)
    dirs = tuple(sorted({op["rad"] for op in openings}))
    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pixels": len(fill_pos),
        "opening_count": len(openings),
        "directions": dirs,
        "bbox": bbox,
        "center_row": center_row,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a curved half-ring enclosure and leak the color through each gap in the fence.

    Input structure:
    - The most common color is background. The wall is the dominant non-background color and forms a thin right-facing half-ring made of two stair-step arcs plus short top/bottom bridges.
    - A small fill-color seed lies inside the curved corridor between the arcs. Any remaining non-bg non-wall pixels are obstacles.
    Rule:
    - Flood-fill the corridor reachable from the seed, then continue through every detected fence gap.
    - An outer-arc gap emits a narrow band toward the right side of the grid, while a bridge gap emits a vertical band toward the nearest border; each band is bounded by rays launched from the two wall tips around that gap.
    - Obstacles keep their colors and erase fill behind them independently along the band direction that reaches them.

    Key insight: The wall must be parsed as one semicircular half-ring with two kinds of missing wall pixels: absent outer-arc rows and a missing bridge cell. Each detected gap defines its own outward direction and its own shadow scan.
    """
    return _solve_core(grid, keep_all_openings=True)


def _arc_depths(r_out: int, r_in: int) -> tuple[dict[int, int], dict[int, int], int]:
    center = r_out
    outer: dict[int, int] = {}
    inner: dict[int, int] = {}
    for row in range(0, 2 * r_out + 1):
        dy = row - center
        outer[row] = int(round(math.sqrt(max(0, r_out * r_out - dy * dy))))
    for row in range(center - r_in, center + r_in + 1):
        dy = row - center
        inner[row] = int(round(math.sqrt(max(0, r_in * r_in - dy * dy))))
    return outer, inner, center


def _connect_points_along_rows(depths: dict[int, int]) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    prev = None
    for row in sorted(depths):
        col = depths[row]
        pts.add((row, col))
        if prev is not None:
            prow, pcol = prev
            step = 1 if row > prow else -1
            for rr in range(prow + step, row, step):
                pts.add((rr, pcol))
            if abs(col - pcol) > 1:
                step_c = 1 if col > pcol else -1
                for cc in range(pcol, col + step_c, step_c):
                    pts.add((row, cc))
        prev = (row, col)
    return pts


def _build_local_wall() -> tuple[set[tuple[int, int]], dict] | None:
    r_out = random.randint(*R_OUT)
    thickness = random.randint(*RING_THICKNESS)
    r_in = r_out - thickness
    if r_in < 2:
        return None

    outer, inner, center = _arc_depths(r_out, r_in)
    wall = _connect_points_along_rows(outer) | _connect_points_along_rows(inner)

    top_bridge = center - r_in
    bottom_bridge = center + r_in
    for c in range(inner[top_bridge], outer[top_bridge] + 1):
        wall.add((top_bridge, c))
    for c in range(inner[bottom_bridge], outer[bottom_bridge] + 1):
        wall.add((bottom_bridge, c))

    band_rows = {
        "upper": [r for r in (center - 3, center - 2) if r in inner and outer[r] - inner[r] >= 3],
        "middle": [r for r in (center,) if r in inner and outer[r] - inner[r] >= 3],
        "lower": [r for r in (center + 2, center + 3) if r in inner and outer[r] - inner[r] >= 3],
    }
    usable_bands = [name for name, rows in band_rows.items() if rows]
    if not usable_bands:
        return None

    name = random.choice(usable_bands)
    rows = band_rows[name]
    run_len = min(random.randint(*OPEN_LEN), len(rows))
    start_idx = random.randint(0, len(rows) - run_len)
    selected = rows[start_idx : start_idx + run_len]
    if selected[-1] - selected[0] != run_len - 1:
        selected = [selected[len(selected) // 2]]
    arc_open = (selected[0], selected[-1])
    for row in range(arc_open[0], arc_open[1] + 1):
        wall.discard((row, outer[row]))

    bridge_row = random.choice([top_bridge, bottom_bridge])
    bridge_outer = outer[bridge_row]
    bridge_cols = list(range(inner[bridge_row] + 1, bridge_outer))
    if len(bridge_cols) < BRIDGE_OPEN_LEN[0] + 1:
        return None
    bridge_len = min(random.randint(*BRIDGE_OPEN_LEN), len(bridge_cols))
    bridge_start = random.randint(bridge_cols[0], bridge_cols[-1] - bridge_len + 1)
    bridge_open = (bridge_row, bridge_start, bridge_start + bridge_len - 1)
    for col in range(bridge_open[1], bridge_open[2] + 1):
        wall.discard((bridge_row, col))

    info = {
        "r_out": r_out,
        "r_in": r_in,
        "center": center,
        "outer": outer,
        "inner": inner,
        "top_bridge": top_bridge,
        "bottom_bridge": bottom_bridge,
        "openings": (("arc", arc_open), ("bridge", bridge_open)),
        "height": 2 * r_out + 1,
        "width": max(outer.values()) + 1,
    }
    mask = _component_mask(wall, (info["height"], info["width"]))
    if len(components_mask(mask, connectivity=8)) != 1:
        return None
    try:
        openings, _bbox, _center = _find_arc_openings(mask)
    except Exception:
        return None
    if len(openings) < 2:
        return None
    return wall, info


def _local_interior(info: dict, wall: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for row in range(info["top_bridge"] + 1, info["bottom_bridge"]):
        left = info["inner"][row]
        right = info["outer"][row]
        for col in range(left + 1, right):
            if (row, col) not in wall:
                cells.append((row, col))
    return cells


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target:
                    break
    return blob


def _seed_pool(interior: list[tuple[int, int]], info: dict, wall: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pool: list[tuple[int, int]] = []
    for row, col in interior:
        if col > info["inner"][row] + 2:
            continue
        if any((rr, cc) in wall for rr, cc in neighbors8(row, col)):
            pool.append((row, col))
    return pool


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(
    grid: np.ndarray, bg: int, wall_color: int, nominal_fill_zone: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
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


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H)
    grid_w = random.randint(*GRID_W)

    built = _build_local_wall()
    if built is None:
        return None
    local_wall, info = built

    if info["height"] + 2 * VERT_MARGIN > grid_h or info["width"] + LEFT_MARGIN + RIGHT_MARGIN > grid_w:
        return None

    r0 = random.randint(VERT_MARGIN, grid_h - info["height"] - VERT_MARGIN)
    c0 = random.randint(LEFT_MARGIN, grid_w - info["width"] - RIGHT_MARGIN)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for row, col in local_wall:
        grid[r0 + row, c0 + col] = wall_color

    interior = _local_interior(info, local_wall)
    if not interior:
        return None
    pool = _seed_pool(interior, info, local_wall)
    if not pool:
        return None

    blob = _grow_seed_blob(random.choice(pool), set(interior))
    for row, col in blob:
        gr, gc = r0 + row, c0 + col
        if grid[gr, gc] != bg:
            return None
        grid[gr, gc] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        collapsed = _solve_core(grid, keep_all_openings=False)
    except Exception:
        return None

    metrics = analyze_primary_arc(grid)
    if metrics["opening_count"] < 2 or len(metrics["directions"]) < 2:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    multi_delta = int(np.sum(out != collapsed))
    if multi_delta < MIN_MULTI_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
