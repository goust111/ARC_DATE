# Variant of 88e364bc
# Dimension: D4 — 滑动终止条件 (dst)
# Change: boundary_collision——黄色像素滑到区域边界（墙壁或区域外）时停止；多个黄色像素在同区域内会堆叠... → 滑到区域中心最近位置停止
#
# Design analysis
# ===============
# variant_spec:
#   a_concretization_and_trace:
#     - 图例、墙壁、封闭区域、颜色匹配机制全部保留；仍然是“图例底色 C 对应墙色 C，图例中的红色指示像素给出方向”。
#     - 唯一核心变化是黄色像素的终点：不再沿方向一直滑到远端墙边，而是在“该黄色像素沿该方向可达的离散轨迹”上，
#       选择距离该封闭区域几何中心最近的格子作为终点。
#     - 几何中心定义为该封闭区域所有可通行格（背景格和黄色格）坐标的算术平均值；比较时使用平方欧氏距离。
#     - 若两步同样接近中心，则继续向前走，等价于“在所有同样最近的格子里，取更靠前的那个”，这样 tie-break 仍然可由方向解释。
#     - 具体追踪例子：若方向为右，某一区域中心列约为 9.5，某黄点从列 4 出发，沿右侧轨迹的到中心距离可能是
#       5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 0.5, 1.5...；原题会继续滑到最右墙边，本变体会停在第二个 0.5 对应的更靠前格。
#       若同一区域有第二个黄点，则第一个先占据中心附近格，第二个会在其后方一格堆叠。
#   b_complexity_calibration:
#     - 不增加新图元，也不引入额外颜色编码，仍保持 1-2 个图例、1-4 个区域、每区 1-2 个黄点。
#     - 为了让 D4 变化肉眼可见，generator 会拒绝“中心停点恰好也是远端边界停点”的退化样本，确保至少一个黄点的新旧规则输出不同。
#     - 方向档位仍只有 8 个（4 个正交 + 4 个对角），不会引入额外自由度。
#   c_implicit_assumptions_review:
#     - `_legend_directions` 只负责读方向，与终点定义无关，可以原样保留。
#     - 原 `_slide_points` 假设“只要前方未出界/未撞黄点，就一直前进直到边界”，这正是旧 D4 的核心，必须替换。
#     - `main()` 中的封闭区域识别、相邻墙色计数、图例颜色匹配都不依赖“终点是墙”，仍然成立。
#     - 原处理顺序“按方向投影从前到后处理黄色”仍需要保留；否则多个黄色在中心附近的堆叠次序会改变。
#   d_placement_legality:
#     - 图例仍固定在角落并带 buffer；区域仍通过 blob + wall ring 生成，保证不越界、不相互压住。
#     - 黄色输入位置只能从“反向走 1..N 步后，重新按新规则正向滑回目标中心停点”的候选中选择。
#     - 对角移动仍沿用原任务的防穿角约束：若 `(dr,dc)` 为对角，则两个正交中间格都必须在区域内。
#   e_visual_observability:
#     - 训练样例中，人能看到黄色不是贴着远端墙，而是停在区域中部偏中心的位置；多个黄色会在中心附近堆叠。
#     - 新旧差异不依赖颜色编号大小、数组索引等不可见量，只依赖空间位置与中心关系。
#   f_visual_cleanliness:
#     - 不增加任何新标记，输出仍只改变黄色位置；墙壁、图例完全不动。
#     - 维持原任务的区域数与留白策略，避免中心附近过度拥挤。
#   g_color_inference:
#     - 颜色语义保持原样：4 是黄色 mover，2 是红色指示像素，1 是图例中心像素，墙色由图例填充色直接匹配。
#   solver_assumptions_after_change:
#     - 新 solver 假设封闭区域是用 passable cells 精确表示的，generator 通过闭合墙和拓扑校验保证这一点。
#     - 新 solver 假设至少存在一个边界图例；generator 仍保证 1-2 个角落图例。
#     - 新 solver 允许个别黄点已经在中心最优格而不动，但 generator 会优先放置需要实际滑动的样本。

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
MAX_REVERSE_STEPS = 6
OUTPUT_CANDIDATE_POOL = 6
START_CANDIDATE_POOL = 6

GEN_RETRIES = 400
EPS = 1e-9
MOVER = 4


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * p[0] + dc * p[1])


def _region_center(region_set: set[tuple[int, int]]) -> tuple[float, float]:
    pts = np.array(sorted(region_set), dtype=float)
    ctr = pts.mean(axis=0)
    return float(ctr[0]), float(ctr[1])


def _center_dist_sq(pos: tuple[int, int], center: tuple[float, float]) -> float:
    dr = float(pos[0]) - center[0]
    dc = float(pos[1]) - center[1]
    return dr * dr + dc * dc


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend rectangles on the border and decode their direction.
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

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_info = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_info.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            tie_info.sort()
            center_col = tie_info[0][1]
            indicator_col = tie_info[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
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
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _center_slide_single(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    center: tuple[float, float] | None = None,
) -> tuple[int, int]:
    if center is None:
        center = _region_center(region_set)
    r, c = start
    cur_d = _center_dist_sq((r, c), center)
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        next_d = _center_dist_sq((nr, nc), center)
        if next_d > cur_d + EPS:
            break
        r, c = nr, nc
        cur_d = next_d
    return (r, c)


def _slide_points_to_center(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    center = _region_center(region_set)
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for pt in points_sorted:
        stop = _center_slide_single(region_set, pt, dr, dc, occupied, center)
        occupied.add(stop)
        out_pts.append(stop)
    return out_pts


def _boundary_slide_single(
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


def _slide_points_to_boundary(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts = []
    for pt in points_sorted:
        stop = _boundary_slide_single(region_set, pt, dr, dc, occupied)
        occupied.add(stop)
        out_pts.append(stop)
    return out_pts


def _border_color_for_region(
    g: np.ndarray, region_pts: np.ndarray, bg: int, mover: int
) -> tuple[int | None, set[tuple[int, int]]]:
    h, w = g.shape
    pset = {(int(r), int(c)) for r, c in region_pts}
    border_counts: dict[int, int] = {}
    for r, c in region_pts:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in pset:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None, pset
    return max(border_counts, key=border_counts.get), pset


def _solve_with_slider(grid: np.ndarray, *, center_mode: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue
        border_col, pset = _border_color_for_region(g, pts, bg, MOVER)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        for r, c in movers:
            out[r, c] = bg
        new_pts = (
            _slide_points_to_center(pset, movers, dr, dc)
            if center_mode
            else _slide_points_to_boundary(pset, movers, dr, dc)
        )
        for r, c in new_pts:
            out[r, c] = MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside each enclosed region according to its legend direction.

    Variant rule:
    - Keep moving only while the next step stays inside the region, does not cut a diagonal corner,
      does not collide with an already-stopped yellow, and is no farther from the region center
      than the current cell.
    - Therefore each yellow stops at the center-nearest reachable point on its forward trajectory,
      with ties resolved toward the farther-along cell.
    """
    return _solve_with_slider(grid, center_mode=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_slider(grid, center_mode=False)


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
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
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
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
    rng: random.Random, grid: np.ndarray
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
    for col in all_colors:
        region_colors.extend([col] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
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
        regions.append({"color": rc, "dir": color_dir[rc], "interior": blob, "walls": walls})

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


def _available_center_outputs(
    blob: set[tuple[int, int]], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> list[dict]:
    center = _region_center(blob)
    output_to_starts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start in blob:
        if start in blocked:
            continue
        out_pos = _center_slide_single(blob, start, dr, dc, blocked, center)
        if out_pos == start:
            continue
        output_to_starts.setdefault(out_pos, []).append(start)

    items = []
    for out_pos, starts in output_to_starts.items():
        starts_sorted = sorted(
            set(starts),
            key=lambda p: (_projection(p, dr, dc), p[0], p[1]),
            reverse=True,
        )
        items.append(
            {
                "out": out_pos,
                "starts": starts_sorted,
                "dist2": _center_dist_sq(out_pos, center),
                "proj": _projection(out_pos, dr, dc),
                "distinctive": _boundary_slide_single(blob, out_pos, dr, dc, blocked) != out_pos,
            }
        )
    items.sort(key=lambda d: (d["dist2"], -d["proj"], d["out"][0], d["out"][1]))
    return items


def _reverse_start_candidates(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    blocked_outputs: set[tuple[int, int]],
    existing_inputs: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    center = _region_center(blob)
    candidates = []
    for steps in range(1, MAX_REVERSE_STEPS + 1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in blocked_outputs or (pr, pc) in existing_inputs:
                ok = False
                break
            if not _slide_step_valid(blob, pr, pc, dr, dc):
                ok = False
                break
            r, c = pr, pc
        if not ok:
            continue
        start = (r, c)
        if _center_slide_single(blob, start, dr, dc, blocked_outputs, center) == out_pos:
            key = (_projection(start, dr, dc), start[0], start[1])
            candidates.append((steps, key, start))
    candidates.sort(key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]), reverse=True)
    return candidates


def _plan_region_yellows(
    rng: random.Random, blob: set[tuple[int, int]], dr: int, dc: int, n_yellows: int
) -> list[tuple[int, int]] | None:
    outputs: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for idx in range(n_yellows):
        candidates = _available_center_outputs(blob, dr, dc, blocked)
        if not candidates:
            return None
        if idx == 0:
            pool = [c for c in candidates if c["distinctive"]] or candidates
        else:
            pool = candidates
        pool = pool[: min(len(pool), OUTPUT_CANDIDATE_POOL)]
        choice = pool[rng.randrange(len(pool))]
        outputs.append(choice["out"])
        blocked.add(choice["out"])

    inputs = []
    existing_inputs: set[tuple[int, int]] = set()
    blocked_outputs: set[tuple[int, int]] = set()
    prev_key: tuple[int, int, int] | None = None
    for out_pos in outputs:
        starts = _reverse_start_candidates(blob, out_pos, dr, dc, blocked_outputs, existing_inputs)
        if prev_key is not None:
            starts = [item for item in starts if item[1] <= prev_key]
        if not starts:
            return None
        starts = starts[: min(len(starts), START_CANDIDATE_POOL)]
        steps, key, start = starts[rng.randrange(len(starts))]
        if steps < 1:
            return None
        inputs.append(start)
        existing_inputs.add(start)
        blocked_outputs.add(out_pos)
        prev_key = key

    if set(_slide_points_to_center(blob, inputs, dr, dc)) != set(outputs):
        return None
    return inputs


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        preferred = rng.choice(YELLOWS_PER_REGION)
        count_order = [preferred] + [n for n in YELLOWS_PER_REGION if n != preferred]
        planned = None
        for n_yellows in count_order:
            planned = _plan_region_yellows(rng, blob, dr, dc, n_yellows)
            if planned is not None:
                break
        if planned is None:
            return False
        for r, c in planned:
            grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
        legacy_out = _legacy_main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == MOVER).sum() != (input_grid == MOVER).sum():
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not bool(np.all(regions_mask[solver_out == MOVER])):
        return False

    if np.array_equal(solver_out, legacy_out):
        return False
    return True


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
