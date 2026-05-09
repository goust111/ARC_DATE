# Combo variant of 88e364bc (index 1)
# combo_spec: D1/0+D9/0
# Name: 箭头图例+仅基数方向
#
# Design analysis
# ===============
# fusion_spec:
#   original_core:
#     - 每个封闭区域的墙色 C 会在边角图例中重复出现，solver 先用墙色把区域和图例配对。
#     - 原题图例用三色矩形里的 "center / indicator" 质心偏移得到 8 个方向。
#     - 区域内部的黄色像素沿该方向直线滑动，并在远端墙边依次堆叠。
#   fused_change:
#     - D1/0: 方向不再来自双色质心，而是来自 5x5 图例里由墙色 C 画出的粗箭头轮廓。
#     - D9/0: 方向空间缩减为 4 个基数方向，因此所有图例统一为 5x5，路径也只允许上下左右。
#   behavior_trace:
#     - 原逻辑示例：若右上角图例里 4 个中心点更靠中、2 个红点偏右，则区域内黄色向右滑。
#     - 融合后示例：若右上角图例是 5x5 两色矩形，墙色 C 在其中形成一个朝右的箭头，则 solver
#       直接把该区域方向解读为 (0,+1)，黄色一路滑到该封闭区的最右内壁并堆叠。
#     - 差异：解题者不再做质心比较，也不会遇到对角线；方向读取变成稳定的形状识别 + 基数直线滑动。
#   complexity_calibration:
#     - 仍保留 1-2 个图例、1-2 种墙色、每色 1-3 个区域，让同一颜色-方向关系在单图中重复出现。
#     - 箭头只保留 4 个取值，既能明显看出与原题不同，又避免 20x20 上出现难辨认的小对角模板。
#     - 箭头模板采用粗短且不对称的 5x5 轮廓，降低与普通色块混淆的风险。
#   solver_assumption_audit:
#     - 原 `_legend_directions` 依赖 `len(counts)==3`、center 色和 indicator 色都存在；融合后这些前提全部失效。
#     - 新 solver 改为要求图例是边界上的 5x5 实心矩形、恰含两色，且主色掩码精确匹配四个箭头模板之一。
#     - 原 `_slide_points` / `_slide_step_valid` 中对角防切角分支在融合后不再需要；保留基数直线推进即可。
#     - `main()` 中按 4 连通寻找封闭通行区、统计相邻墙色、按远端优先顺序堆叠黄色，这些前提仍然成立。
#   placement_legality:
#     - generator 仍将图例贴在四角，并保留 1 格缓冲，确保图例不会与墙体 4 连通或重叠。
#     - 区域生长仍要求墙环不碰边界、不碰其他保留区；因此每个封闭区都能被唯一墙色包围。
#     - 每种图例墙色至少生成一个区域，保证 solver 的颜色匹配总能落地。
#   visual_observability:
#     - 人类可直接看到：角落小块里的墙色本身画成箭头，且黄色总是沿箭头所指的上下左右方向靠墙。
#     - 没有对角运动，也没有依赖颜色编号的隐式排序；方向信息完全来自肉眼可见的箭头朝向。
#   cleanliness:
#     - 图例只有 "墙色箭头 + 统一底板" 两层信息，比原题三色点标更紧凑。
#     - 区域之间保持墙环和缓冲间距，减少对象粘连，输入输出都能清楚区分图例、墙、空腔、黄色。
#   color_inference:
#     - 唯一需要学习的颜色关系仍是 "图例主色 = 对应区域墙色"；颜色 1 只是图例底板，不参与方向语义。
#   fusion_behavior_check:
#     - 量化验证时应统计：所有 legend 是否都是两色 5x5 箭头、方向是否覆盖 4 个基数档位，
#       以及这些输入交给原始三色质心 solver 时是否无法按原规则解析。
#   new_solver_assumptions_to_check_later:
#     - 假设每个 legend 的主色掩码恰好等于四个 5x5 箭头模板之一；generator 会严格按模板构造。
#     - 假设所有滑动方向满足 `abs(dr) + abs(dc) == 1`；generator 与 `_legend_directions` 共同保证这一点.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (15, 85),
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5

GEN_RETRIES = 300

LEGEND_BG = 1
CARDINAL_ARROW_RIGHT = np.array(
    [
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
    ],
    dtype=bool,
)


def _cardinal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(CARDINAL_ARROW_RIGHT, rot[(dr, dc)])


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((5, 5), LEGEND_BG, dtype=int)
    patch[_cardinal_arrow_mask(dr, dc)] = fill
    return patch


CARDINAL_TEMPLATE_MASKS = {(dr, dc): _cardinal_arrow_mask(dr, dc) for dr, dc in CARDINAL_DIRS}


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


def _match_arrow_direction(fill_mask: np.ndarray) -> tuple[int, int] | None:
    if fill_mask.shape != (5, 5):
        return None
    for direction, template in CARDINAL_TEMPLATE_MASKS.items():
        if np.array_equal(fill_mask, template):
            return direction
    return None


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode direction from a dominant-color 5x5 arrow.

    Each legend is a border-touching filled rectangle whose dominant color is the wall color it
    controls. The dominant-color cells form one of four cardinal arrow templates.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        vals, counts = np.unique(patch, return_counts=True)
        if len(vals) != 2:
            continue
        fill = int(vals[np.argmax(counts)])
        direction = _match_arrow_direction(patch == fill)
        if direction is None:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), direction)

    return {color: direction for color, (_, direction) in best.items()}


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set or (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions using border legends drawn as cardinal arrows.

    Input structure:
    - The background is the most common color, and yellow pixels lie inside closed cavities bounded
      by colored wall rings.
    - On the outer border sit 5x5 legend rectangles. Each legend uses two colors: a wall color C
      that forms a visible arrow shape, plus a uniform backing color. The wall color C matches the
      wall color of one or more enclosed regions.

    Rule:
    - For each enclosed passable region, determine its wall color C from the surrounding ring and
      find the legend whose dominant arrow color is also C.
    - Read the arrow orientation directly as one of the four cardinal directions.
    - Move all yellow pixels in that region straight along that direction until they hit the far
      wall or previously stacked yellows.

    Key insight: The color match still links each region to its legend, but the direction now comes
    from the legend's arrow silhouette itself, and every valid movement is purely horizontal or
    vertical.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                value = int(g[nr, nc])
                if value == bg or value == mover:
                    continue
                border_counts[value] = border_counts.get(value, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover

    return out


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    return (r + dr, c + dc) in region_set


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)

        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

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
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wall_color: direction for wall_color, direction in legends}
    all_colors = [wall_color for wall_color, _ in legends]

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
                "dir": color_dir[region_color],
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


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_yellows(
    rng: random.Random,
    regions: list[dict],
    grid: np.ndarray,
) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]
        if n_yellows == 2:
            prev_cell = (first[0] - dr, first[1] - dc)
            candidates = ([(prev_cell)] if prev_cell in blob else []) + [p for p in tier1 if p != first]
            if candidates:
                second = candidates[rng.randrange(len(candidates))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []
        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


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
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
