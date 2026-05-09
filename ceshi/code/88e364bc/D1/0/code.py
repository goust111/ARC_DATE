# Variant of 88e364bc
# Dimension: D1 — 图例方向编码方式 (symbol_instructions)
# Change: 图例为5x5或6x7实心矩形，包含三种颜色：填充色C（最多）、中心色（4像素，靠近中心）、指示色/红色（2像素，更外围）... → 箭头形状编码方向

# Design analysis
# ===============
# variant_spec:
#   original_core:
#     - 墙色 C 既是封闭区域的边框色，也是角落图例里数量最多的颜色。
#     - 原题图例方向来自两种少数颜色的质心偏移；solver 先按颜色频次分出 center / indicator，再取 sign(centroid(indicator)-centroid(center))。
#     - 黄色像素在各自封闭区域内沿该方向滑动，并在远端墙边堆叠。
#   concrete_variant:
#     - 保留 20x20 网格、1-2 个角落图例、1-2 种墙色、同样的封闭区域与黄色滑动规则。
#     - 仍保留 5x5（四向）和 6x7（对角）两类边角矩形图例；图例改成双色：
#       - 填充/箭头色 C：数量最多，并继续与对应区域的墙色相同。
#       - 辅助底色 1：作为矩形底板，不参与方向编码。
#     - 真正编码方向的是颜色 C 在矩形内部形成的箭头轮廓：
#       - 5x5: 一个粗短的十字箭头，旋转后表示上/下/左/右。
#       - 6x7: 一个斜向楔形箭头，翻转/旋转 180 度后表示四个对角方向。
#   behavior_trace:
#     - 原逻辑示例：若 5x5 图例里 4 个蓝点在中间、2 个红点偏右，则 solver 用质心得到 (0,+1)，区域内黄色向右滑。
#     - 变体逻辑示例：若 5x5 图例里颜色 C 本身画成“向右箭头”，另一种颜色只是统一底板，则 solver 直接识别 C 的形状朝向为 (0,+1)。
#     - 差异：原题依赖“少数颜色的相对位置 + 质心计算”，变体依赖“主色轮廓模板匹配”。黄色滑动阶段完全不变。
#   complexity_calibration:
#     - 方向取值仍只有 8 个离散档位，避免自由度过大。
#     - 每个样例有 1-2 个图例，且墙色会在 1-3 个区域上重复使用，使“墙色 ↔ 图例”对应关系在单图中可重复观察。
#     - 箭头模板保持粗而不对称，避免与普通斑块或矩形装饰混淆；对角方向继续专用 6x7 形状，避免 5x5 小图里对角箭头难读。
#   solver_assumption_audit:
#     - 旧 `_legend_directions` 的 `len(counts)==3` 前提已不成立；新图例只有“箭头色 C + 底色 1”两种颜色。
#     - 旧代码 `bg_col = max(counts, key=counts.get)` 仍成立，因为颜色 C 仍被设计成图例中最多的颜色。
#     - 旧代码依赖少数颜色分工为 center / indicator，这一前提彻底失效，因为现在只有单一底色，不再存在第二个可比较的少数颜色。
#     - 因此方向解码必须改为“抽取颜色 C 的布尔掩码，并和 5x5 / 6x7 箭头模板匹配”。
#     - `main()` 中其余部分（封闭区域检测、边框主色统计、黄色滑动顺序、对角防切角）都不依赖旧图例语义，可原样保留。
#   placement_legality:
#     - 图例仍贴四角放置，并保留 1 格缓冲区，防止与墙体 4 连通。
#     - 箭头图例仍是完整非背景矩形，因此连通域 + bbox=area 的“矩形图例”检测仍有效。
#     - 颜色 C 与墙色唯一对应；generator 继续保证每种图例墙色至少有一个封闭区域。
#   visual_observability:
#     - 人类看到输入输出时，能直接看出角落矩形里的墙色 C 画成了某个箭头，而黄色沿该箭头方向滑向区域远端。
#     - 图例里除 C 以外只有统一底色，因此方向信息只能来自 C 的形状本身。
#   cleanliness:
#     - 图例内部只有“粗箭头 + 简单底板 + 四角点”，不会叠加太多细碎标记。
#     - 区域生成逻辑不变，仍用不重叠闭合墙与 1 格缓冲保证画面整洁。
#   color_inference:
#     - 不引入新的颜色映射规则；颜色 C 仍通过“图例主色 = 区域墙色”被推断。
#     - 颜色 1 只承担稳定背景作用，不需要解题者学习额外颜色语义。
#   new_solver_assumptions_to_check_later:
#     - 假设所有图例的颜色 C 掩码恰好等于 8 个模板之一；generator 会严格按模板生成。
#     - 假设 5x5 只对应四向、6x7 只对应对角；generator 维持这一结构。

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
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

DIAGONAL_ARROW_DR = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=bool,
)


def _cardinal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(CARDINAL_ARROW_RIGHT, rot[(dr, dc)])


def _diagonal_arrow_mask(dr: int, dc: int) -> np.ndarray:
    if (dr, dc) == (1, 1):
        return DIAGONAL_ARROW_DR.copy()
    if (dr, dc) == (-1, 1):
        return np.flipud(DIAGONAL_ARROW_DR)
    if (dr, dc) == (1, -1):
        return np.fliplr(DIAGONAL_ARROW_DR)
    if (dr, dc) == (-1, -1):
        return np.rot90(DIAGONAL_ARROW_DR, 2)
    raise ValueError((dr, dc))


def _legend_base(h: int, w: int) -> np.ndarray:
    return np.full((h, w), LEGEND_BG, dtype=int)


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = _legend_base(5, 5)
    g[_cardinal_arrow_mask(dr, dc)] = fill
    return g


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = _legend_base(6, 7)
    g[_diagonal_arrow_mask(dr, dc)] = fill
    return g


CARDINAL_TEMPLATE_MASKS = {(dr, dc): _cardinal_arrow_mask(dr, dc) for dr, dc in CARDINAL_DIRS}
DIAGONAL_TEMPLATE_MASKS = {(dr, dc): _diagonal_arrow_mask(dr, dc) for dr, dc in DIAGONAL_DIRS}


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
    if fill_mask.shape == (5, 5):
        for d, templ in CARDINAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, templ):
                return d
    elif fill_mask.shape == (6, 7):
        for d, templ in DIAGONAL_TEMPLATE_MASKS.items():
            if np.array_equal(fill_mask, templ):
                return d
    return None


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode direction from the dominant-color arrow silhouette.

    The dominant legend color remains the controlled wall color C. Unlike the original task,
    the minority colors are not semantically ordered; direction is read by matching the mask
    of C-cells inside the rectangle against a small set of arrow templates.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) not in ((5, 5), (6, 7)):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        vals, counts = np.unique(patch, return_counts=True)
        if len(vals) < 2:
            continue
        fill = int(vals[np.argmax(counts)])
        fill_mask = patch == fill
        direction = _match_arrow_direction(fill_mask)
        if direction is None:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), direction)

    return {k: v for k, (_, v) in best.items()}


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
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction dictated by border legends.

    Input structure:
    - bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
    - Border legends: small rectangles on the grid edge whose dominant color C matches a region's
      border color. The C-cells form an arrow silhouette that directly indicates direction.

    Rule:
    - For each enclosed region, identify its border color C and look up the legend with dominant
      color C.
    - The orientation of the C-colored arrow inside that legend gives the slide direction.
    - All yellow pixels in that region slide in that direction, stacking against the far wall.

    Key insight: Legend dominant color still links legend ↔ wall; the directional cue is now the
    shape of the dominant-color cells, not the centroid offset of two minority colors.
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
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points(pset, movers, dr, dc)
        for r, c in new_pts:
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


def _slide_step_valid(region_set: set, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set,
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set,
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

    for corner, wc in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))

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
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

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
        dr, dc = color_dir[rc]
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
        regions.append({"color": rc, "dir": (dr, dc), "interior": blob, "walls": walls})

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
    blob: set,
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set,
    existing_inputs: list,
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
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
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
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
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
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
