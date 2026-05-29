# Variant of 88e364bc
# Dimension: D1 — 图例方向编码方式 (symbol_instructions)
# Change: 图例为5x5或6x7实心矩形，包含三种颜色：填充色C（最多）、中心色（4像素，靠近中心）、指示色/红色（2像素，更外围）... → 图例中指示色像素数量编码步长
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的总体舞台：20x20 网格，角落放 5x5 或 6x7 的实心图例块，内部空腔由彩色墙围住，黄色像素在空腔内移动。
#    - 原题里图例只编码方向。现在图例仍然用 `sign(centroid(red) - centroid(center))` 编码方向，
#      但红色像素数量从固定 2 个扩展为 2/3/4 个，对应“每个黄色最多移动 2/3/4 步”。
#    - 具体例子：若某个 5x5 图例的红色在中心块右边，且红点数为 3，则这个颜色对应区域内的黄色都向右移动，
#      但最多只走 3 格；如果第 4 格仍然是空腔，也必须停住。这与原题“滑到墙为止”不同。
#    - 影响环节：
#      1. legend 解析：除了方向，还要读出 indicator 的像素数。
#      2. 移动执行：旧 `_slide_points()` 会一直滑到墙，新规则要改成“重复尝试至多 step_count 次”。
#      3. 输入构造：若 generator 仍把黄色放成“最后贴墙”的输出态，很多样本会退化成原题，所以必须显式生成
#         至少一个“走满步长后在中途停下”的黄色。
# b) 复杂度校准
#    - 步长只取 3 档：2、3、4。档位少，便于从少量样例归纳；同时又足够让输出与原题产生肉眼可见差异。
#    - 同一 legend 颜色仍可控制 1-3 个区域，因此“同一个红点数对应相同步长”会在单个样例里重复出现。
#    - 不引入新的颜色类别，也不改变墙/空腔/黄色的识别方式，避免把难度转移到无关维度。
# c) 隐式假设审查（旧 main 中与本变体相关的假设）
#    - `_legend_directions()`：旧代码默认更少的非填充色就是 indicator，这在 step=4 时不成立，因为 center=4、
#      indicator=4 会打平。这里保留原 tie-break 思路，但显式返回 indicator 数量。
#    - `indicator_col = others[0][1]`：依赖“红点总比中心点少”；变体下只对 step=2/3 成立，step=4 需要按距离中心更远
#      的那种颜色识别为 indicator。
#    - `_slide_points()`：旧版 while 循环没有步数上限，默认“方向一旦确定就滑到极限”。这是本变体最关键的不成立前提。
#    - `main()` 中其它区域识别逻辑依赖的只是“图例填充色=墙色”“黄色只在封闭空腔内”，这些前提不变。
# d) 放置合法性
#    - 图例仍然贴角放置，仍然是完整实心矩形，避免被墙体误识别。
#    - region 生成逻辑保持原样：先长出 interior blob，再以其 4 邻域作为墙环，并为 legend 区域和墙环留缓冲带，
#      因此 legend / 墙 / 空腔不会重叠。
#    - 黄色只放在 region interior 内部；移动时的新位置也只由同一个 interior 上的受限滑动产生，所以不会越界或穿墙。
# e) 视觉可观察性
#    - 单看 input 的角落图例，人类能看到同方向图例中红点数量不同；对照 output，会发现黄色不是总贴墙，而是常常在中途停下。
#    - generator 强制每个样本至少出现一个 `stop_reason == step_limit` 的黄色，即“如果按原题规则会继续前进，但这里停住”。
#      这样新维度一定在样例里可见，不会只体现在代码层面。
# f) 可视化整洁度
#    - 不增加新的对象种类，也不在同一区域叠加额外标记；变化只体现在 legend 内红点个数，以及 yellow 的终点位置。
#    - 5x5 / 6x7 图例内部仍然只用三种颜色，且红点始终位于中心块的外侧，便于肉眼看出“方向”和“数量”两层信息。
# g) 颜色变化可推断性
#    - 本变体不引入新的颜色映射；颜色仍只承担“墙色匹配图例填充色”的作用，推理重点落在形状/计数上。
# h) 生成策略调整
#    - 原 generator 是 output-first，为了让变体稳定可见，这里改为“先生成静态结构，再在 interior 中直接采样黄色输入位置”，
#      用受限滑动模拟输出，并要求至少一个黄色因步长上限而停住。
#    - 随后用新 `main()` 做 oracle，对比 solver 输出与基于 region 元数据得到的期望输出，确保 solver/generator 一致。
# i) 新 main() 的输入假设审查
#    - 假设每个 legend 恰好有三种颜色，且 fill 色最多；generator 用固定模板保证。
#    - 假设 indicator 像素数只会是 2/3/4；generator 只从这三档模板中取。
#    - 假设至少有一个 region 中存在可走满步长且仍能继续前进的起点；generator 在选定“可见 region”前先检测这一点。
#    - 假设黄色输入位置互不重叠且在各自 interior 内；generator 通过随机采样 + 受限滑动模拟过滤保证。

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
LEGEND_STEP_CHOICES = [2, 3, 4]

# Regions
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


def _legend_specs(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int]]:
    """
    Detect border legends and decode (dr, dc, max_steps) per fill color.

    The movement direction is still:
      sign(centroid(indicator) - centroid(center))
    The new step cap is:
      len(indicator) in {2, 3, 4}
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
    max_steps: int,
) -> list[dict]:
    if not points:
        return []

    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_meta = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        steps_taken = 0
        stop_reason = "wall"
        for _ in range(max_steps):
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
                "start": (r0, c0),
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
    max_steps: int,
) -> list[tuple[int, int]]:
    return [item["end"] for item in _slide_points_meta(region_set, points, dr, dc, max_steps)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions using border legends.

    The legend fill color still identifies which wall color it controls.
    The center->indicator centroid offset gives direction.
    The number of indicator pixels (2/3/4) gives the per-pixel maximum slide distance.
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
        dr, dc, max_steps = legend_specs[border_col]

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc, max_steps):
            out[r, c] = MOVER

    return out


def _legend_5x5(fill: int, dr: int, dc: int, step_count: int) -> np.ndarray:
    """5x5 cardinal legend. Canonical direction: (0, +1) right."""
    g = np.full((5, 5), fill, dtype=int)
    center = [(1, 1), (1, 2), (3, 1), (3, 2)]
    indicator_by_step = {
        2: [(1, 3), (3, 3)],
        3: [(1, 3), (3, 3), (2, 4)],
        4: [(1, 3), (2, 3), (3, 3), (2, 4)],
    }
    for r, c in center:
        g[r, c] = 1
    for r, c in indicator_by_step[step_count]:
        g[r, c] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int, step_count: int) -> np.ndarray:
    """6x7 diagonal legend. Canonical direction: (+1, +1) down-right."""
    g = np.full((6, 7), fill, dtype=int)
    center = [(1, 3), (2, 1), (2, 4), (3, 2)]
    indicator_by_step = {
        2: [(3, 5), (4, 3)],
        3: [(3, 5), (4, 3), (4, 4)],
        4: [(3, 5), (4, 3), (4, 4), (5, 4)],
    }
    for r, c in center:
        g[r, c] = 1
    for r, c in indicator_by_step[step_count]:
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
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
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
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

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
    legends = []

    for corner, wc in zip(corners, wall_colors):
        step_count = rng.choice(LEGEND_STEP_CHOICES)
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc, step_count)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc, step_count)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc), step_count))

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
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_map = {wc: (direction, steps) for wc, direction, steps in legends}
    all_colors = [wc for wc, _, _ in legends]

    region_colors = []
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        (dr, dc), step_count = color_map[rc]
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
                "color": rc,
                "dir": (dr, dc),
                "step": step_count,
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
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
    max_steps: int,
) -> bool:
    for r0, c0 in blob:
        r, c = r0, c0
        steps = 0
        while steps < max_steps and _slide_step_valid(blob, r, c, dr, dc):
            r, c = r + dr, c + dc
            steps += 1
        if steps == max_steps and _slide_step_valid(blob, r, c, dr, dc):
            return True
    return False


def _place_region_yellows(
    rng: random.Random,
    reg: dict,
    grid: np.ndarray,
    n_yellows: int,
    require_step_limit: bool,
) -> bool:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    max_steps = reg["step"]
    cells = list(blob)
    if len(cells) < n_yellows:
        return False

    for _ in range(YELLOW_PLACEMENT_RETRIES):
        inputs = rng.sample(cells, n_yellows)
        meta = _slide_points_meta(blob, inputs, dr, dc, max_steps)
        if any(item["steps"] <= 0 for item in meta):
            continue
        if require_step_limit and not any(item["stop"] == "step_limit" for item in meta):
            continue
        for r, c in inputs:
            grid[r, c] = MOVER
        return True
    return False


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    visible_regions = [
        idx
        for idx, reg in enumerate(regions)
        if _region_has_step_limited_candidate(reg["interior"], reg["dir"][0], reg["dir"][1], reg["step"])
    ]
    if not visible_regions:
        return False
    visible_idx = rng.choice(visible_regions)

    for idx, reg in enumerate(regions):
        n_yellows = 1 if idx == visible_idx else rng.choice(YELLOWS_PER_REGION)
        if not _place_region_yellows(rng, reg, grid, n_yellows, require_step_limit=(idx == visible_idx)):
            return False
    return True


def _expected_output_from_regions(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    out[input_grid == MOVER] = 0
    for reg in regions:
        movers = [(r, c) for r, c in reg["interior"] if int(input_grid[r, c]) == MOVER]
        if not movers:
            continue
        dr, dc = reg["dir"]
        max_steps = reg["step"]
        for r, c in _slide_points(reg["interior"], movers, dr, dc, max_steps):
            out[r, c] = MOVER
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = _expected_output_from_regions(input_grid, regions)
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

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
