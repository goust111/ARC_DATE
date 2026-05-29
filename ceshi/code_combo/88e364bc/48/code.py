# Combo variant of 88e364bc (index 48)
# combo_spec: D4/0+D11/1
# Name: 固定步数停止+起点标记保留
#
# Design analysis
# - a) 具体化与行为追踪:
#   输入仍是 20x20，角落 legend 的主填充色继续绑定同色墙区域。不同的是，同一个 legend 现在同时给出两个参数:
#   红色指示像素的方位给出方向，红色指示像素的数量给出固定步数 k。输出不再把起点擦成背景，而是在起点留下蓝色标记(1)，
#   终点仍是黄色(4)。例如某区域 legend 指向右且有 3 个红点，则黄色从 `(r,c)` 只向右走 3 格，在 `(r,c)` 留下蓝点，
#   在 `(r,c+3)` 留下黄点；若前方还有空位也必须停。原题则会继续滑到右墙。
# - b) 复杂度校准:
#   步数档位限制为 2 或 3，方向仍覆盖正交/对角 8 个方向。这样人类可以肉眼数起点到终点的间隔，
#   又不会引入过多离散值；同色 legend 仍可控制多个区域，给出重复支撑。
# - c) 隐式假设审查:
#   原 `_slide_points` 默认“停止原因只能是撞墙/撞已停黄色”，融合后必须改成“最多走 k 次合法单步”。
#   原 solver 只返回终点即可，但 D11/1 需要保留起点，因此输出阶段必须先清掉输入黄点，再把起点重涂为蓝色、终点重涂为黄色。
#   这还引入了新假设: 终点不能落回任何起点，否则蓝色标记会被黄色覆盖；generator 会显式规避。
# - d) 放置合法性:
#   generator 继续先放 legend，再生长封闭区域，再选黄色。候选起点必须满足:
#   1) 能恰好走满 k 步；
#   2) 固定步终点后方仍至少还有 1 个合法空位，保证不是“刚好撞墙”；
#   3) 若按原题一直滑到边界，终点会更远，保证 D4 改动真实生效；
#   4) 固定步终点不与任何起点重合，保证 D11/1 的蓝色起点全部可见。
# - e) 视觉可观察性:
#   解题者可以先从 legend 看出方向，再数红点得步数，再从输出里直接观察“蓝起点 -> 黄终点”的距离。
#   因为步数只取 2/3，且同图中常有多个同色区域共享同一 legend，2-4 个样例足以支撑归纳。
# - f) 可视化整洁度:
#   区域内部只新增一种标记色: 蓝色起点标记。它不与墙色集合冲突，且只出现在区域内部；
#   终点仍是黄色，没有连续轨迹，因此画面比“整条路径着色”更干净。
# - g) 变化的可推断性:
#   没有新增按墙色变化的隐藏映射。蓝色始终表示“原起点”，黄色始终表示“固定步后的终点”。
#   颜色的角色靠位置和结构区分，不依赖颜色编号本身的不可视排序。
# - h) 反退化策略:
#   oracle 使用完整 `output_grid` 精确校验，而不只是检查黄色数量。另加构造约束让每个已放入黄色都满足
#   “固定步终点 != 原题边界终点”，从而 D4 不会退化；并要求输出里的蓝色起点数量等于输入黄色数量，从而 D11/1 不会退化。
# - i) main() 的新假设与生成器保证:
#   `main()` 假设 legend 的红点数就是步数且至少为 1，这由 generator 保证；假设输入区域内部没有蓝色起点标记，
#   这也由 generator 保证，因为蓝色只出现在边框 legend。`main()` 还假设按 far-to-near 顺序处理时不会出现
#   “终点覆盖另一黄色起点导致蓝点消失”，generator 通过候选筛选确保这一点。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER_COLOR = 4
START_MARK_COLOR = 1

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1
STEP_CHOICES = [2, 3]

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
GEN_RETRIES = 300


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * int(p[0]) + dc * int(p[1]))


def _legend_specs(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int]]:
    """Return wall-color -> (dr, dc, fixed_steps) decoded from border legends."""
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int, int]]] = {}
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
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cents = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        spec = (dr, dc, len(indic))
        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), spec)

    return {k: v for k, (_, v) in best.items()}


def _step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = int(r + dr), int(c + dc)
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (int(r + dr), int(c)) not in region_set or (int(r), int(c + dc)) not in region_set:
            return False
    return True


def _slide_points_fixed(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = int(r0), int(c0)
        for _ in range(int(steps)):
            if not _step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = int(r + dr), int(c + dc)
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _slide_points_boundary(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = int(r0), int(c0)
        while True:
            if not _step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = int(r + dr), int(c + dc)
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each yellow a legend-decoded fixed number of steps and preserve its launch cell as a marker.

    Input structure:
    - The grid is mostly background, with enclosed regions bounded by colored walls and yellow movers inside.
    - Border legends are solid rectangles whose dominant fill color matches a wall color. Blue center pixels and red indicator pixels still encode direction by centroid offset, and the number of red pixels encodes the shared step count.

    Rule:
    - For each enclosed region, find the legend whose fill color matches that region's surrounding wall color.
    - Every yellow in that region attempts to move exactly the legend's fixed number of steps in the decoded direction, stopping early only if a wall, diagonal corner-cut restriction, or an already-settled yellow blocks the next step.
    - In the output, the original launch cell becomes a blue start marker and the resting cell is yellow.

    Key insight:
    - The legend now controls both direction and distance, and the preserved blue start marker makes the step count visible directly as the gap from start to finish.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    specs = _legend_specs(g, bg)
    assert specs, "legend_specs must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER_COLOR)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER_COLOR]
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
                if v == bg or v == MOVER_COLOR:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in specs:
            continue
        dr, dc, steps = specs[border_col]

        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points_fixed(pset, movers, dr, dc, steps)
        for r, c in movers:
            out[r, c] = START_MARK_COLOR
        for r, c in new_pts:
            out[r, c] = MOVER_COLOR

    return out


def _legend_5x5(fill: int, dr: int, dc: int, steps: int) -> np.ndarray:
    """5x5 cardinal legend. Direction is the centroid offset; red-count is the fixed step count."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[3, 1] = 1
    g[3, 2] = 1
    reds = {
        2: [(1, 3), (3, 3)],
        3: [(1, 3), (2, 4), (3, 3)],
    }[int(steps)]
    for r, c in reds:
        g[r, c] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int, steps: int) -> np.ndarray:
    """6x7 diagonal legend. Canonical direction is (+1, +1)."""
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    reds = {
        2: [(3, 5), (4, 3)],
        3: [(3, 5), (4, 3), (4, 4)],
    }[int(steps)]
    for r, c in reds:
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


def _forward_fixed_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    steps: int,
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    r, c = int(start[0]), int(start[1])
    moved = 0
    for _ in range(int(steps)):
        if not _step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = int(r + dr), int(c + dc)
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
        moved += 1
    return (r, c), moved


def _forward_boundary_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = int(start[0]), int(start[1])
    while True:
        if not _step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = int(r + dr), int(c + dc)
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    step_values = rng.sample(STEP_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[dict[str, object]] = []

    for corner, wc, steps in zip(corners, wall_colors, step_values):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc, int(steps))
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc, int(steps))
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"color": int(wc), "dir": (int(dr), int(dc)), "steps": int(steps)})

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
    legends: list[dict[str, object]],
    legend_occupied: np.ndarray,
) -> list[dict[str, object]] | None:
    h, w = legend_occupied.shape
    color_to_legend = {int(leg["color"]): leg for leg in legends}
    all_colors = [int(leg["color"]) for leg in legends]

    region_colors: list[int] = []
    for c in all_colors:
        region_colors.extend([int(c)] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

    for rc in region_colors:
        leg = color_to_legend[int(rc)]
        dr, dc = leg["dir"]  # type: ignore[misc]
        steps = int(leg["steps"])
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
                "color": int(rc),
                "dir": (int(dr), int(dc)),
                "steps": int(steps),
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


def _enumerate_visible_fixed_moves(
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
    blocked_outputs: set[tuple[int, int]],
    used_inputs: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for start in blob:
        if start in used_inputs or start in blocked_outputs:
            continue
        end, moved = _forward_fixed_slide(blob, start, dr, dc, steps, blocked_outputs)
        if moved != steps:
            continue
        if end in used_inputs:
            continue
        if not _step_valid(blob, end[0], end[1], dr, dc):
            continue
        ahead = (int(end[0] + dr), int(end[1] + dc))
        if ahead in blocked_outputs:
            continue
        boundary_end = _forward_boundary_slide(blob, start, dr, dc, blocked_outputs)
        if boundary_end == end:
            continue
        candidates.append((start, end))
    candidates.sort(key=lambda pair: (_projection(pair[1], dr, dc), pair[1][0], pair[1][1]), reverse=True)
    return candidates


def _pick_region_moves(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    steps: int,
    n_yellows: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    blocked_outputs: set[tuple[int, int]] = set()
    used_inputs: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for _ in range(int(n_yellows)):
        cands = _enumerate_visible_fixed_moves(blob, dr, dc, steps, blocked_outputs, used_inputs)
        if not cands:
            return None
        top_band = cands[: min(len(cands), 8)]
        start, end = rng.choice(top_band)
        placements.append((start, end))
        used_inputs.add(start)
        blocked_outputs.add(end)

    ordered = sorted(placements, key=lambda pair: (_projection(pair[0], dr, dc), pair[0][0], pair[0][1]), reverse=True)
    ordered_inputs = [start for start, _ in ordered]
    actual_outputs = _slide_points_fixed(blob, ordered_inputs, dr, dc, steps)
    expected_outputs = [end for _, end in ordered]
    if actual_outputs != expected_outputs:
        return None
    if any(end in set(ordered_inputs) for end in expected_outputs):
        return None
    boundary_outputs = _slide_points_boundary(blob, ordered_inputs, dr, dc)
    if not any(fixed != boundary for fixed, boundary in zip(actual_outputs, boundary_outputs)):
        return None
    return ordered


def _place_yellows(
    rng: random.Random,
    regions: list[dict[str, object]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]  # type: ignore[misc]
        steps = int(reg["steps"])

        desired = rng.choice(YELLOWS_PER_REGION)
        counts_to_try = [int(desired)] if desired == 1 else [2, 1]
        placements = None
        for n_yellows in counts_to_try:
            placements = _pick_region_moves(rng, blob, int(dr), int(dc), steps, int(n_yellows))
            if placements is not None:
                break
        if placements is None:
            return False

        for start, end in placements:
            input_grid[start] = MOVER_COLOR
            output_grid[start] = START_MARK_COLOR
            output_grid[end] = MOVER_COLOR
    return True


def _oracle_check(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    return bool(np.array_equal(solver_out, output_grid))


def _behavior_metrics(input_grid: np.ndarray) -> dict[str, int]:
    """Quantify whether both fixed-distance motion and start-marker retention are visible."""
    g = np.asarray(input_grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    specs = _legend_specs(g, bg)
    out = main(g)

    passable = (g == bg) | (g == MOVER_COLOR)
    mover_count = int((g == MOVER_COLOR).sum())
    interior_marker_count = 0
    fixed_vs_boundary_diff = 0
    max_gap = 0

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER_COLOR]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v in (bg, MOVER_COLOR):
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        dr, dc, steps = specs[border_col]
        ordered_inputs = sorted(movers, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
        fixed = _slide_points_fixed(pset, ordered_inputs, dr, dc, steps)
        boundary = _slide_points_boundary(pset, ordered_inputs, dr, dc)
        fixed_vs_boundary_diff += sum(int(a != b) for a, b in zip(fixed, boundary))
        for start, end in zip(ordered_inputs, fixed):
            if int(out[start]) == START_MARK_COLOR:
                interior_marker_count += 1
            max_gap = max(max_gap, max(abs(end[0] - start[0]), abs(end[1] - start[1])))

    return {
        "mover_count": mover_count,
        "marker_count": interior_marker_count,
        "fixed_vs_boundary_diff": fixed_vs_boundary_diff,
        "max_gap": max_gap,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

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
    output_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid, output_grid):
        return None
    if not _oracle_check(input_grid, output_grid):
        return None

    metrics = _behavior_metrics(input_grid)
    if metrics["mover_count"] == 0:
        return None
    if metrics["marker_count"] != metrics["mover_count"]:
        return None
    if metrics["fixed_vs_boundary_diff"] <= 0:
        return None
    if metrics["max_gap"] < 2:
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
