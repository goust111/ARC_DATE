# Combo variant of 88e364bc (index 36)
# combo_spec: D1/3+D22/2
# Name: 红色移动像素+数量编码步长

"""
design_analysis:
  a_concretization_and_trace:
    - 原题里 corner legend 的红色只负责指方向，封闭区域里的 mover 是黄色并且会一直滑到墙。
    - 本融合版里，封闭区域里的 mover 也改成红色，因此同一个颜色 2 同时出现在 legend 和 region interior。
    - 具体例子：如果右上角 legend 的填充色是 5，中心色是 1，外围红点朝左，则颜色 5 的区域都向左移动；
      若该 legend 里红点数量为 3，则每个 interior 红像素最多只走 3 步，即使第 4 步仍能继续，也必须停下。
    - 因而 solver 流程变成：先识别 border legend 矩形并做 mask；再从 legend 内红点质心解方向、从红点数量解步长；
      最后只对“非 legend 的红色”执行受限滑动。
  b_complexity_calibration:
    - 步长只允许 2/3/4 三档，足够可见且便于从少量样例归纳。
    - 仍保留 1-2 个 legend、每个 legend 控制 1-3 个 region，这样同一步长/方向会在同一题内重复出现。
    - 不增加新颜色；歧义只来自“红色双角色”，视觉负担集中但不过载。
  c_implicit_assumptions_review:
    - 原 `passable = bg or mover` 在 mover=2 后会错误吞入 legend 红点，因此必须先构造 `legend_mask`。
    - 原 legend 解析默认 indicator 比 center 更少；本变体在 step=4 时两者都可能是 4 个，需要用“谁更靠近矩形中心”
      来识别 center，较外围者识别为 indicator。
    - 原无限滑动 while 循环默认一直移动到墙；现在必须显式加 `max_steps` 上限并保留 stacking 顺序。
    - 原 oracle 把所有 mover 色都当作非结构色；本变体不成立，因为 legend indicator 也是红色。
  d_placement_legality:
    - legend 仍先贴在四角并保留缓冲区，region interior 只在剩余自由空间生长。
    - 红色 mover 只会被放进 region interior，墙色集合排除了红色，因此 legend 外出现的红色必定是 mover。
    - generator 采样 mover 输入位置时要求至少走 1 步，并通过 solver 同款物理模拟保证不会穿墙或重叠。
  e_visual_observability:
    - 人类可从位置上下文区分两类红色：角落实心矩形内部的红色属于 legend，封闭腔体中的红色会移动。
    - 多个样例里同墙色区域共享同一方向/步长，足以支持“红点朝向=方向、红点数量=步长”的归纳。
    - generator 额外强制至少一个 mover 因 `step_limit` 停下，否则样例会退化成原题的贴墙行为。
  f_cleanliness:
    - legend 仍是完整矩形，region 与 legend 之间有 buffer；不把额外记号叠到 mover 或墙上。
    - 颜色 2 的双角色仅靠空间区分，不会在同一局部区域内混在一起。
  g_change_inferability:
    - 颜色规则很少：fill 色匹配墙色，蓝色中心仅是 legend 内辅助点，红色数量决定步长且 interior 红色会移动。
    - 由于步长只取 3 档，且 generator 强制出现中途停住案例，解题者能从 2-4 组样例看出“不是一直贴墙”。
  h_generator_strategy:
    - 为了稳定制造步长效应，generator 采用“先造静态结构，再在 interior 中直接采样输入 mover”而非原始 output-first。
    - 之后用新 solver 做 oracle，确保生成输入与融合规则一致。
  i_new_solver_assumptions_and_generator_guarantees:
    - solver 假设 legend 是 border-touching 的三色实心矩形，且 indicator 像素数只会是 2/3/4。
    - solver 假设所有可移动红点都在非 border-touching 的 passable component 内。
    - generator 通过固定 legend 模板、非红墙色、封闭 region 生长和 mover 采样过滤来保证这些前提成立。
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 2

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

# Mover placement
MOVERS_PER_REGION = [1, 2]
MOVER_PLACEMENT_RETRIES = 250

# Outer loop
GEN_RETRIES = 300


def _detect_legends(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int, int]], np.ndarray]:
    """
    Detect border legends and decode (dr, dc, max_steps) by fill color.

    Direction is still decoded from centroid(indicator) - centroid(center).
    Step cap is len(indicator) in {2, 3, 4}.
    Also returns a legend mask so red indicator pixels are not mistaken for movers.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        coords = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(coords):
            continue

        counts: dict[int, int] = {}
        for r, c in coords:
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
            tie_info: list[tuple[float, int]] = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in coords if int(g[r, c]) == col], dtype=float)
                dist2 = float(np.sum((pts_col.mean(axis=0) - rect_center) ** 2))
                tie_info.append((dist2, col))
            tie_info.sort()
            center_col = tie_info[0][1]
            indicator_col = tie_info[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in coords if int(g[r, c]) == center_col]
        indicator = [(r, c) for r, c in coords if int(g[r, c]) == indicator_col]
        if len(center) != 4 or len(indicator) not in {2, 3, 4}:
            continue

        center_centroid = np.mean(np.array(center, dtype=float), axis=0)
        indicator_centroid = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (indicator_centroid - center_centroid).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(coords) > prev[0]:
            best[fill_col] = (len(coords), (dr, dc), len(indicator), coords)

    legend_mask = np.zeros_like(g, dtype=bool)
    for _, (_, _, _, coords) in best.items():
        for r, c in coords:
            legend_mask[r, c] = True

    specs = {fill_col: (payload[1][0], payload[1][1], payload[2]) for fill_col, payload in best.items()}
    return specs, legend_mask


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
        out_meta.append({"start": (r0, c0), "end": (r, c), "steps": steps_taken, "stop": stop_reason})
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
    Slide red pixels inside walled regions according to border legends that encode both direction and step cap.

    Input structure:
    - The grid background is the most common color.
    - Corner legends are border-touching filled rectangles whose majority fill color matches the wall color of one or more enclosed regions.
    - Each legend contains four center pixels and 2/3/4 red indicator pixels; enclosed movers are also red, but they sit inside non-legend cavities.

    Rule:
    - For each enclosed passable region, find its dominant adjacent wall color and match it to the legend with the same fill color.
    - Decode direction as sign(centroid(indicator) - centroid(center)), and decode the maximum travel distance from the number of indicator pixels.
    - Slide only non-legend red pixels in that region up to the decoded number of steps, stopping early at walls, diagonal corner cuts, or previously stacked movers.

    Key insight:
    - Color alone no longer separates clue from object: the same red pixels appear in legends and as movers.
    - The solver must first isolate legend rectangles spatially, then treat legend reds as instructions and interior reds as the pieces that move.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    legend_specs, legend_mask = _detect_legends(g, bg)
    assert legend_specs, "legend detection must find at least one border legend"

    passable = (g == bg) | ((g == MOVER) & (~legend_mask))
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [
            (int(r), int(c))
            for r, c in pts
            if int(g[r, c]) == MOVER and not bool(legend_mask[int(r), int(c)])
        ]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
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
    """5x5 cardinal legend. Canonical direction: right."""
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
    """6x7 diagonal legend. Canonical direction: down-right."""
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
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for region_color in region_colors:
        (dr, dc), step_count = color_map[region_color]
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
                "color": region_color,
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


def _place_region_movers(
    rng: random.Random,
    reg: dict,
    grid: np.ndarray,
    n_movers: int,
    require_step_limit: bool,
) -> bool:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    max_steps = reg["step"]
    cells = list(blob)
    if len(cells) < n_movers:
        return False

    for _ in range(MOVER_PLACEMENT_RETRIES):
        inputs = rng.sample(cells, n_movers)
        meta = _slide_points_meta(blob, inputs, dr, dc, max_steps)
        if any(item["steps"] <= 0 for item in meta):
            continue
        if require_step_limit and not any(item["stop"] == "step_limit" for item in meta):
            continue
        for r, c in inputs:
            grid[r, c] = MOVER
        return True
    return False


def _place_movers(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    visible_regions = [
        idx
        for idx, reg in enumerate(regions)
        if _region_has_step_limited_candidate(reg["interior"], reg["dir"][0], reg["dir"][1], reg["step"])
    ]
    if not visible_regions:
        return False
    visible_idx = rng.choice(visible_regions)

    for idx, reg in enumerate(regions):
        n_movers = 1 if idx == visible_idx else rng.choice(MOVERS_PER_REGION)
        if not _place_region_movers(rng, reg, grid, n_movers, require_step_limit=(idx == visible_idx)):
            return False
    return True


def _expected_output_from_regions(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    region_mask = np.zeros_like(input_grid, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
    out[region_mask & (input_grid == MOVER)] = 0
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
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

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
    if not _place_movers(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
