# Variant of 88e364bc
# Dimension: D4 — 滑动终止条件 (dst)
# Change: boundary_collision——黄色像素滑到区域边界（墙壁或区域外）时停止；多个黄色像素在同区域内会堆叠... → 滑动固定步数后停止

# Design analysis
# - a) 具体化与行为追踪:
#   输入仍然是 20x20、角落 legend + 彩色墙围住的区域 + 黄色像素。保留原题最重要的可见结构:
#   legend 的主填充色仍然等于对应区域的墙色，蓝色中心团和红色指示团仍然通过相对方位编码方向。
#   变体只新增一个可见参数: 红色指示像素的数量 = 固定滑动步数 k，取值 1..3。
#   因此同一墙色控制的所有区域共享 (direction, steps)。
#   例子: 若某个右指 legend 里有 3 个红点，则该墙色区域内每个黄色都向右尝试移动 3 步；
#   如果走满 3 步后前面还有空格，也必须停下。原题则会继续滑到最右侧墙或前面黄色后面。
# - b) 复杂度校准:
#   距离档位限制为 1..3，避免自由度过大；1-2 个 legend、每个 legend 可对应 1-3 个区域，
#   让同一 distance 在单图内通常有重复支撑。方向仍可为 4 个正交或 4 个对角，和原题同级。
# - c) 隐式假设审查:
#   原 `_slide_points` 的 while 循环隐含“停止原因一定是撞墙/撞已停黄色”，这在变体中不成立，
#   必须改成“最多执行 k 次合法单步”。区域识别、边框颜色统计、legend 的矩形检测仍然成立，
#   但 legend 解析现在还依赖“红点数量 < 蓝点数量”，所以生成器必须保持蓝点固定为 4、红点为 1..3。
# - d) 放置合法性:
#   generator 仍先放 legend，再长不接边的封闭区域，再放黄色。为了让新规则真正可见，
#   黄色输入位置只从那些“能恰好前进 k 步且终点前方仍有至少 1 格合法空间”的起点中选择。
#   这样输出不会退化为“刚好也贴墙”。
# - e) 视觉可观察性:
#   人类先能从蓝/红相对位置看出方向，再从红点个数看出步数。因为红点直接可数，且同色 legend
#   会在多个区域上重复生效，2-4 个样例足以支撑归纳“同色区域按图例给出的方向和固定距离移动”。
# - f) 可视化整洁度:
#   不新增第四种 legend 标记色，也不在区域内叠加额外数字/纹理；只改变红点个数，视觉干净且和原题连续。
# - g) 颜色变化可推断性:
#   没有新增颜色映射。颜色只承担原题中的“墙色 ↔ legend 填充色”链接作用，步数完全由可数的红点数量提供。
# - h) 生成器反退化策略:
#   oracle 由“结构不变 + 黄色仍在区域内”强化为“`main(input)` 必须完全等于预构造 output_grid`”。
#   同时每个黄色都要求走满 k 步且停下后前方还有空间，从构造上排除大部分与原题同输出的退化样本。
# - i) main() 的新假设与生成器保证:
#   `main()` 现在假设 legend 的红点数量就是步数，且至少为 1；生成器保证所有 legend 都符合这一点。
#   `main()` 仍保留黄色之间的先后阻挡（far-to-near），生成器选择输入时保证所选黄色在该顺序下仍能得到预期终点。

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * int(p[0]) + dc * int(p[1]))


def _legend_specs(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int]]:
    """
    Detect legend blocks on the border.

    As in the original task, the dominant color of each border-touching filled rectangle is the wall
    color it controls. The direction is still sign(centroid(indicator) - centroid(center)).
    The new D4 variant adds: distance = number of indicator pixels.
    """
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
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
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

        prev = best.get(fill_col)
        spec = (dr, dc, len(indic))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction and distance dictated by border legends.

    Input structure:
    - bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
    - Border legends: small filled rectangles on the border whose fill color C matches a region's border.
      The blue center cluster + red indicator cluster still encode direction by relative position.
      The number of red indicator pixels encodes a fixed step count k.

    Rule:
    - For each enclosed region, identify its border color C and find the matching legend.
    - Every yellow in that region attempts to move exactly k steps in the legend direction.
    - Movement is still blocked by region limits, diagonal corner-cutting, and earlier stopped yellows,
      but unlike the original task it does not keep sliding after k legal steps.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    specs = _legend_specs(g, bg)
    assert specs, "legend_specs must find at least one legend; grid should always have border legends"

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
        if border_col not in specs:
            continue
        dr, dc, steps = specs[border_col]

        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points_fixed(pset, movers, dr, dc, steps)
        for r, c in new_pts:
            out[r, c] = mover

    return out


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1
STEP_CHOICES = [1, 2, 3]

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


def _legend_5x5(fill: int, dr: int, dc: int, steps: int) -> np.ndarray:
    """5x5 cardinal legend. Direction is the centroid offset; red-count is the step count."""
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[3, 1] = 1
    g[3, 2] = 1
    reds = {
        1: [(2, 4)],
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
        1: [(4, 4)],
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


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    step_values = rng.sample(STEP_CHOICES, n_legends) if n_legends > 1 else [rng.choice(STEP_CHOICES)]

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
        top_band = cands[: min(len(cands), 6)]
        start, end = rng.choice(top_band)
        placements.append((start, end))
        used_inputs.add(start)
        blocked_outputs.add(end)

    ordered_inputs = [p[0] for p in sorted(placements, key=lambda pair: _projection(pair[0], dr, dc), reverse=True)]
    actual_outputs = _slide_points_fixed(blob, ordered_inputs, dr, dc, steps)
    expected_outputs = [p[1] for p in sorted(placements, key=lambda pair: _projection(pair[0], dr, dc), reverse=True)]
    if actual_outputs != expected_outputs:
        return None
    return placements


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
            input_grid[start] = 4
            output_grid[end] = 4
    return True


def _oracle_check(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    return bool(np.array_equal(solver_out, output_grid))


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
