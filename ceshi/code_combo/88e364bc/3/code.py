# Combo variant of 88e364bc (index 3)
# combo_spec: D1/3+D4/0
# Name: 数量编码步长+固定步数停止
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题舞台: 20x20 网格、角落实心 legend、彩色墙围出的封闭空腔、空腔中的黄色像素。
#    - 原题链路是“墙色匹配 legend -> legend 解出方向 -> 黄色一直滑到墙”。融合后链路改成
#      “墙色匹配 legend -> legend 同时解出方向和步长 -> 黄色只移动恰好最多 N 次合法单步后停止”。
#    - 具体例子: 若某个右指的 5x5 legend 中红色 indicator 有 3 个，则该墙色对应的所有区域都向右移动，
#      且每个黄色最多只前进 3 步。若第三步走完前方还有空格，也必须停在中途；原题则会继续贴到最右墙。
#    - 两个维度不是并列叠加而是共用同一信息源: 红点相对中心的位置给方向，红点数量给固定步长。
#      因此解题者需要先把 legend 当作“统一动作参数表”来解码，再对每个区域执行固定步数移动。
# b) 复杂度校准
#    - 步长只取 2/3/4 三档，既能肉眼数清，又足以让结果明显偏离原题的“贴墙”效果。
#    - legend 数量仍为 1-2 个，每个 legend 仍可控制 1-3 个区域，让同一个 `(方向, 步长)` 在单图中重复出现，
#      便于从少量样例归纳“同色墙共享同一动作参数”。
#    - 不新增新颜色类别，也不改墙/空腔/黄色的识别方式，避免把难点转移到无关维度。
# c) 隐式假设审查
#    - 原 `_legend_directions()` 假设非填充色里“数量更少者”必是 indicator；融合后 step=4 会与 center=4 打平，
#      所以必须保留 tie-break: 更靠近矩形中心的是 center，更外侧的是 indicator。
#    - 原 `_slide_points()` 假设停止条件只能是撞墙或撞已停黄色；融合后这个前提失效，必须增加步数上限。
#    - 原 region 识别依赖“legend 填充色 = 对应墙色”“黄色只位于封闭空腔内”，这些前提在融合后仍成立。
#    - 新 `main()` 额外假设 indicator 数量只会是 2/3/4 且 center 固定为 4 个像素；generator 需要显式保证。
# d) 放置合法性
#    - legend 仍然贴角放置，且必须是完整实心矩形，这样不会和墙体或空腔组件混淆。
#    - 区域生成仍先长 interior blob，再以其 4 邻域形成一圈墙，并给 legend 与墙留缓冲，因此不会重叠或越界。
#    - 黄色只放在 region interior 内；移动时仍受 region 边界、对角不切角、已停黄色占位约束，所以输出也合法。
# e) 视觉可观察性
#    - 人类可以直接从 legend 看出两层信息: 红点相对蓝色中心团的方位给方向，红点数量给步长。
#    - 若 generator 允许所有黄色刚好撞墙，样本会退化成与原题同输出，因此必须强制至少一个黄色满足
#      “走满 N 步后前方仍可继续”，让固定步数停止这一行为在样例里稳定可见。
# f) 可视化整洁度
#    - 变化只体现在 legend 内红点个数，以及黄色最终没有贴墙而停在中途；不引入额外标记层。
#    - 5x5/6x7 模板都保持三色、中心团固定、indicator 始终更外侧，保证方向与数量都容易辨认。
# g) 变化的可推断性
#    - 颜色没有新增映射任务；填充色仍只承担“这个 legend 控制哪种墙色”的索引作用。
#    - 推理负担集中在结构特征与计数特征上，且这两者都能在 2-4 个样例中直接观察。
# h) 生成策略调整
#    - 采用“静态结构先行 + 在 interior 内采样输入黄色”的策略，用新滑动规则前向模拟输出。
#    - generator 至少选一个存在 `step_limit` 终止候选的 region，并要求该 region 的至少一个黄色真实触发该终止，
#      从构造上防止退化回原题。
#    - oracle 不只检查 solver 不报错，而是比较 `main(input)` 与按 region 元数据算出的期望输出是否完全一致。
# i) 新 main() 假设与生成器保证
#    - `main()` 假设 border legend 可以唯一解出 `(dr, dc, steps)`；generator 用固定模板保证三色、矩形、可区分。
#    - `main()` 假设 region 的主边框色在 4 邻域计数上占优；generator 让每个 interior 只对应一圈单色墙来保证。
#    - `main()` 的黄色处理顺序是 far-to-near；generator 在采样黄色后用同一顺序模拟并做 oracle，保证顺序依赖可满足。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

# Legend configuration
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1
STEP_CHOICES = [2, 3, 4]

# Region configuration
REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (15, 85),
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

# Yellow placement
YELLOWS_PER_REGION = [1, 2]
YELLOW_PLACEMENT_RETRIES = 250

# Outer loop
GEN_RETRIES = 300


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * int(p[0]) + dc * int(p[1]))


def _legend_specs(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int]]:
    """
    Detect border legends and decode (dr, dc, steps) per fill color.

    Direction is still sign(centroid(indicator) - centroid(center)).
    The indicator pixel count encodes the fixed step budget in {2, 3, 4}.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int], int]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()

        if others[0][0] == others[1][0]:
            rect_center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_info = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                dist2 = float(np.sum((pts_col.mean(axis=0) - rect_center) ** 2))
                tie_info.append((dist2, col))
            tie_info.sort()
            center_col = tie_info[0][1]
            indicator_col = tie_info[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == center_col]
        indicator = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == indicator_col]
        if len(center) != 4 or len(indicator) not in {2, 3, 4}:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc), len(indicator))

    return {k: (vec[0], vec[1], steps) for k, (_, vec, steps) in best.items()}


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _slide_points_meta(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
) -> list[dict[str, object]]:
    if not points:
        return []

    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_meta: list[dict[str, object]] = []
    for r0, c0 in points_sorted:
        r, c = int(r0), int(c0)
        steps_taken = 0
        stop_reason = "wall"
        for _ in range(int(steps)):
            if not _slide_step_valid(region_set, r, c, dr, dc):
                stop_reason = "wall"
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                stop_reason = "occupied"
                break
            r, c = nr, nc
            steps_taken += 1
        else:
            if _slide_step_valid(region_set, r, c, dr, dc) and (r + dr, c + dc) not in occupied:
                stop_reason = "step_limit"
            elif _slide_step_valid(region_set, r, c, dr, dc):
                stop_reason = "occupied"
            else:
                stop_reason = "wall"

        occupied.add((r, c))
        out_meta.append(
            {
                "start": (int(r0), int(c0)),
                "end": (r, c),
                "steps": steps_taken,
                "stop": stop_reason,
            }
        )
    return out_meta


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
) -> list[tuple[int, int]]:
    return [item["end"] for item in _slide_points_meta(region_set, points, dr, dc, steps)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels inside enclosed regions using corner legends that encode both direction and step count.

    Input structure:
    - The background is the most common color; yellow pixels lie in enclosed passable regions bounded by colored walls.
    - Each relevant wall color has a matching filled legend rectangle on the outer border.
    - Inside a legend, the center cluster and the outer indicator cluster still define a direction by centroid offset, and the number of indicator pixels is the fixed step count.

    Rule:
    - For each enclosed region, determine its controlling wall color and read the matching legend.
    - Decode the movement vector from center-to-indicator position and decode the step budget from the indicator count.
    - Every yellow in that region advances at most that many legal single-cell steps, stopping early only if the region boundary, diagonal corner-cutting rule, or a previously stopped yellow blocks the next step.

    Key insight:
    - The legend now acts as a compact motion program: wall color chooses the legend, indicator placement gives direction, and indicator quantity gives the exact travel distance, so yellows often stop mid-corridor instead of piling against the far wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    legend_specs = _legend_specs(g, bg)
    assert legend_specs, "legend_specs must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == MOVER:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in legend_specs:
            continue
        dr, dc, steps = legend_specs[border_col]

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc, steps):
            out[r, c] = MOVER

    return out


def _legend_5x5(fill: int, dr: int, dc: int, step_count: int) -> np.ndarray:
    """5x5 cardinal legend. Canonical direction: (0, +1)."""
    g = np.full((5, 5), fill, dtype=int)
    center = [(1, 1), (1, 2), (3, 1), (3, 2)]
    indicator_by_step = {
        2: [(1, 3), (3, 3)],
        3: [(1, 3), (3, 3), (2, 4)],
        4: [(1, 3), (2, 3), (3, 3), (2, 4)],
    }
    for r, c in center:
        g[r, c] = 1
    for r, c in indicator_by_step[int(step_count)]:
        g[r, c] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int, step_count: int) -> np.ndarray:
    """6x7 diagonal legend. Canonical direction: (+1, +1)."""
    g = np.full((6, 7), fill, dtype=int)
    center = [(1, 3), (2, 1), (2, 4), (3, 2)]
    indicator_by_step = {
        2: [(3, 5), (4, 3)],
        3: [(3, 5), (4, 3), (4, 4)],
        4: [(3, 5), (4, 3), (4, 4), (5, 4)],
    }
    for r, c in center:
        g[r, c] = 1
    for r, c in indicator_by_step[int(step_count)]:
        g[r, c] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(int(sr), int(sc))}
    frontier: list[tuple[int, int]] = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((int(nr), int(nc)))

    push_neighbors(int(sr), int(sc))
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((int(nr), int(nc)))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((int(r), int(c)))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((int(nr), int(nc)))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int], int]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[tuple[int, tuple[int, int], int]] = []

    for corner, wall_color in zip(corners, wall_colors):
        step_count = rng.choice(STEP_CHOICES)
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wall_color, dr, dc, step_count)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wall_color, dr, dc, step_count)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((int(wall_color), (int(dr), int(dc)), int(step_count)))

    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int], int]],
    legend_occupied: np.ndarray,
) -> list[dict[str, object]] | None:
    h, w = legend_occupied.shape
    color_map = {wall_color: (direction, steps) for wall_color, direction, steps in legends}
    all_colors = [wall_color for wall_color, _, _ in legends]

    region_colors: list[int] = []
    for color in all_colors:
        region_colors.extend([int(color)] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

    for region_color in region_colors:
        (dr, dc), step_count = color_map[int(region_color)]
        target_size = rng.randint(size_min, size_max)

        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": int(region_color),
                "dir": (int(dr), int(dc)),
                "step": int(step_count),
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((int(nr), int(nc)))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {int(reg["color"]) for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _region_has_step_limited_candidate(
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
) -> bool:
    for r0, c0 in blob:
        r, c = int(r0), int(c0)
        moved = 0
        while moved < steps and _slide_step_valid(blob, r, c, dr, dc):
            r, c = r + dr, c + dc
            moved += 1
        if moved == steps and _slide_step_valid(blob, r, c, dr, dc):
            return True
    return False


def _place_region_yellows(
    rng: random.Random,
    reg: dict[str, object],
    grid: np.ndarray,
    n_yellows: int,
    require_step_limit: bool,
) -> bool:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    steps = reg["step"]
    cells = list(blob)
    if len(cells) < n_yellows:
        return False

    for _ in range(YELLOW_PLACEMENT_RETRIES):
        inputs = rng.sample(cells, n_yellows)
        meta = _slide_points_meta(blob, inputs, dr, dc, steps)
        if any(int(item["steps"]) <= 0 for item in meta):
            continue
        if require_step_limit and not any(item["stop"] == "step_limit" for item in meta):
            continue
        for r, c in inputs:
            grid[r, c] = MOVER
        return True
    return False


def _place_yellows(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    visible_regions = [
        idx
        for idx, reg in enumerate(regions)
        if _region_has_step_limited_candidate(
            reg["interior"],
            int(reg["dir"][0]),
            int(reg["dir"][1]),
            int(reg["step"]),
        )
    ]
    if not visible_regions:
        return False
    visible_idx = rng.choice(visible_regions)

    for idx, reg in enumerate(regions):
        n_yellows = 1 if idx == visible_idx else rng.choice(YELLOWS_PER_REGION)
        if not _place_region_yellows(rng, reg, grid, n_yellows, require_step_limit=(idx == visible_idx)):
            return False
    return True


def _expected_output_from_regions(input_grid: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    out = input_grid.copy()
    out[input_grid == MOVER] = 0
    for reg in regions:
        movers = [(r, c) for r, c in reg["interior"] if int(input_grid[r, c]) == MOVER]
        if not movers:
            continue
        dr, dc = reg["dir"]
        steps = reg["step"]
        for r, c in _slide_points(reg["interior"], movers, dr, dc, steps):
            out[r, c] = MOVER
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = _expected_output_from_regions(input_grid, regions)
    return bool(np.array_equal(solver_out, expected))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["color"])

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
