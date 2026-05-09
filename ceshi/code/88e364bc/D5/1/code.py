# Variant of 88e364bc
# Dimension: D5 — 移动对象（源）类型 (src)
# Change: control_pixels——黄色单像素（颜色4），每个区域1-2个，作为被移动的对象... → 不同颜色的移动像素
#
# Design analysis
# ===============
# variant_spec:
#   unchanged_backbone:
#     - grid 仍是 20x20，背景色仍是 0。
#     - 墙围成封闭区域；边框上的矩形 legend 仍通过“填充色 = 墙色”与某类区域关联。
#     - 运动仍是沿 8 个单位方向之一滑动到远端墙边；对角移动仍禁止 corner cutting。
#   core_D5_change:
#     - 原题每个区域只有黄色 mover（4），整个区域共用一个方向。
#     - 现在每块 legend 内同时出现 3 个 mover 颜色的“中心点”：
#       yellow=4, pink=6, orange=7。
#     - 每个中心点旁边有一个红点（2），红点相对该颜色点的 8 邻域偏移就是该颜色在该墙色区域中的滑动方向。
#     - 同一区域内会放 2-3 个不同颜色的 mover；它们分别读取自己颜色对应的方向，所以同一区域里会朝不同墙面分流。
#   concrete_visual_form:
#     - legend 固定为 9x9 实心矩形，填充色是墙色（只从 3/5/8/9 中选，避免与 mover 颜色混淆）。
#     - 三个 mover 颜色点固定放在 legend 内三个彼此分离的位置；每个点各配一个红点。
#     - 人眼可直接看到：“同一块 legend 里，黄/粉/橙分别对应不同红点方向”。
#   behavior_trace:
#     - 原逻辑：区域里若有两枚黄色，先找该区域墙色，再从对应 legend 读出一个统一方向，所有黄色沿同一方向堆叠。
#     - 新逻辑：例如某个绿色墙区域内有 yellow/pink/orange 三点；绿色 legend 里
#       yellow 的红点在右侧、pink 的红点在左上、orange 的红点在下方。
#       则输出里 yellow 去右壁，pink 去左上角附近，orange 去下壁。
#     - 差异的关键不是“墙色变了”，而是“同一区域内 mover 的颜色分组决定方向”。
#   complexity_calibration:
#     - 只用 3 个 mover 颜色，避免颜色类别过多导致 2-4 个样例无法归纳。
#     - 每个区域强制放 2-3 种 mover 颜色，保证变体在每个样本中肉眼可见。
#     - 墙色仍只有 1-2 种，保持原题结构，不把难度转移到别的维度。
#   main_hidden_assumptions_review:
#     - 原 `mover = 4` 假设失效；必须改成识别 `MOVER_COLORS`。
#     - 原 `passable = (g == bg) | (g == mover)` 假设失效；否则粉/橙会把区域割裂。
#     - 原 `if v == bg or v == mover: continue` 只排除了黄色；现在必须排除所有 mover 颜色。
#     - 原 “整个区域只有一个方向” 假设失效；现在方向查找粒度变为 `(wall_color, mover_color)`。
#     - 原 “清空旧 mover 后统一重画黄色” 也失效；若仍无条件写回 4，会覆盖颜色信息。
#   placement_legality:
#     - legend 使用专门的 9x9 子布局，三种 mover 色与三枚红点互不重叠。
#     - 生成器只使用 3/5/8/9 作为墙色，避免与 mover 颜色、红色指示点混淆。
#     - 每个 mover 颜色在一个区域内只放 1 枚，简化逆推并避免跨颜色落点冲突。
#     - 生成器只接受“各颜色最终落点互异”的区域，避免 solver 出现颜色覆盖。
#   visual_observability:
#     - 即使不读代码，样例中也能看到：同一墙色图例里有多个彩色点，每个彩色点都各自指向不同方向；
#       区域内部的同色像素在输出里向与图例同色点一致的方向移动。
#   cleanliness:
#     - 一个区域里最多 3 枚 mover，且都是单像素，和原题视觉密度接近，不会因为叠层过多而杂乱。
#   color_inference:
#     - 颜色不参与任何“不可见排序”；颜色只是可见类别标签。
#     - legend 内彩色点与区域内 mover 用同色直接绑定，不需要学习隐藏的颜色映射表。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


MOVER_COLORS = (4, 6, 7)
WALL_COLOR_CHOICES = (3, 5, 8, 9)
INDICATOR_COLOR = 2
GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]

LEGEND_H, LEGEND_W = 9, 9
LEGEND_BUFFER = 1
LEGEND_CENTERS = {
    4: (2, 2),
    6: (2, 6),
    7: (6, 4),
}

MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (15, 52),
    3: (12, 30),
}
BLOB_GROW_RETRIES = 25
GEN_RETRIES = 300
ACTIVE_COLORS_PER_REGION = (2, 3)
MAX_SLIDE_STEPS = 6


def _legend_direction_map(
    g: np.ndarray, bg: int
) -> tuple[dict[int, dict[int, tuple[int, int]]], set[tuple[int, int]]]:
    """Map wall color -> mover color -> direction, and return all legend cells."""
    h, w = g.shape
    best: dict[int, tuple[int, dict[int, tuple[int, int]]]] = {}
    legend_cells: set[tuple[int, int]] = set()
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_H, LEGEND_W):
            continue

        counts: dict[int, int] = {}
        coords_by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            col = int(g[r, c])
            counts[col] = counts.get(col, 0) + 1
            coords_by_color.setdefault(col, []).append((int(r), int(c)))

        fill = max(counts, key=counts.get)
        mover_cols = [col for col in MOVER_COLORS if counts.get(col, 0) == 1]
        red_pts = coords_by_color.get(INDICATOR_COLOR, [])
        if fill in MOVER_COLORS or fill == INDICATOR_COLOR:
            continue
        if len(mover_cols) < 2 or len(red_pts) != len(mover_cols):
            continue

        remaining = set(red_pts)
        dir_map: dict[int, tuple[int, int]] = {}
        ok = True
        for color in mover_cols:
            cr, cc = coords_by_color[color][0]
            nearest = None
            for rr, rc in remaining:
                cand = (abs(rr - cr) + abs(rc - cc), rr, rc)
                if nearest is None or cand < nearest:
                    nearest = cand
            if nearest is None:
                ok = False
                break
            _, rr, rc = nearest
            dr = rr - cr
            dc = rc - cc
            if dr == 0 and dc == 0:
                ok = False
                break
            if abs(dr) > 1 or abs(dc) > 1:
                ok = False
                break
            dir_map[color] = (1 if dr > 0 else -1 if dr < 0 else 0, 1 if dc > 0 else -1 if dc < 0 else 0)
            remaining.remove((rr, rc))
        if not ok:
            continue

        prev = best.get(fill)
        if prev is None or len(pts) > prev[0]:
            best[fill] = (len(pts), dir_map)
            legend_cells.update((int(r), int(c)) for r, c in pts)
    return {wall_color: dir_map for wall_color, (_, dir_map) in best.items()}, legend_cells


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
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
    Slide multi-colored single-pixel movers inside walled regions.

    The legend still keys on wall color, but each legend now carries three sub-instructions:
    yellow/pink/orange points each have their own red indicator, so movement direction depends on
    both the region border color and the mover color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs, legend_cells = _legend_direction_map(g, bg)
    assert dirs, "legend parsing failed; generated inputs always include at least one border legend"

    mover_mask = np.isin(g, MOVER_COLORS)
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in legend_cells:
        legend_mask[r, c] = True
    passable = (g == bg) | (mover_mask & ~legend_mask)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers_by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            col = int(g[r, c])
            if col in MOVER_COLORS:
                movers_by_color.setdefault(col, []).append((int(r), int(c)))
        if not movers_by_color:
            continue

        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v in MOVER_COLORS:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        region_dirs = dirs.get(border_col, {})

        for pts_color in movers_by_color.values():
            for r, c in pts_color:
                out[r, c] = bg

        placements: list[tuple[int, int, int]] = []
        for color, pts_color in movers_by_color.items():
            if color not in region_dirs:
                for r, c in pts_color:
                    placements.append((r, c, color))
                continue
            dr, dc = region_dirs[color]
            for r, c in _slide_points(pset, pts_color, dr, dc):
                placements.append((r, c, color))

        for r, c, color in placements:
            out[r, c] = color

    return out


def _legend_patch(fill: int, dir_map: dict[int, tuple[int, int]]) -> np.ndarray:
    g = np.full((LEGEND_H, LEGEND_W), fill, dtype=int)
    for color, (cr, cc) in LEGEND_CENTERS.items():
        dr, dc = dir_map[color]
        ir, ic = cr + dr, cc + dc
        g[cr, cc] = color
        g[ir, ic] = INDICATOR_COLOR
    return g


def _sample_legend_dirs(rng: random.Random) -> dict[int, tuple[int, int]]:
    for _ in range(200):
        picked = rng.sample(ALL_DIRS, len(MOVER_COLORS))
        dir_map = dict(zip(MOVER_COLORS, picked))
        occupied = set(LEGEND_CENTERS.values())
        red_cells = set()
        ok = True
        for color, (dr, dc) in dir_map.items():
            cr, cc = LEGEND_CENTERS[color]
            ir, ic = cr + dr, cc + dc
            if not (0 <= ir < LEGEND_H and 0 <= ic < LEGEND_W):
                ok = False
                break
            if (ir, ic) in occupied or (ir, ic) in red_cells:
                ok = False
                break
            red_cells.add((ir, ic))
        if ok:
            return dir_map
    raise RuntimeError("_sample_legend_dirs: unable to place non-overlapping indicator pixels")


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int
) -> tuple[int, int]:
    r, c = start
    while _slide_step_valid(region_set, r, c, dr, dc):
        r, c = r + dr, c + dc
    return r, c


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(list(WALL_COLOR_CHOICES), n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[dict[str, object]] = []
    for corner, wc in zip(corners, wall_colors):
        dir_map = _sample_legend_dirs(rng)
        patch = _legend_patch(wc, dir_map)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"wall_color": wc, "dir_map": dir_map})
    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random, canvas: np.ndarray, occupied: np.ndarray, target_size: int, h: int, w: int
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
    rng: random.Random, legends: list[dict[str, object]], legend_occupied: np.ndarray
) -> list[dict[str, object]] | None:
    h, w = legend_occupied.shape
    all_colors = [int(legend["wall_color"]) for legend in legends]

    region_colors = all_colors.copy()
    while len(region_colors) < MAX_TOTAL_REGIONS and rng.random() < 0.6:
        region_colors.append(rng.choice(all_colors))
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

    for wall_color in region_colors:
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
        legend = next(legend for legend in legends if int(legend["wall_color"]) == wall_color)
        regions.append(
            {
                "color": wall_color,
                "dir_map": dict(legend["dir_map"]),
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
    region_sets = [set(reg["interior"]) for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _find_reverse_slide_single(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    used_inputs: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for steps in range(MAX_SLIDE_STEPS, 0, -1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in used_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    ok = False
                    break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc) == out_pos:
            return r, c
    return None


def _choose_output_and_input(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    used_outputs: set[tuple[int, int]],
    used_inputs: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    stop_cells = [p for p in blob if not _slide_step_valid(blob, p[0], p[1], dr, dc)]
    if not stop_cells:
        return None
    stop_cells.sort(key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)

    best_key = dr * stop_cells[0][0] + dc * stop_cells[0][1]
    shortlist = [p for p in stop_cells if dr * p[0] + dc * p[1] >= best_key - 2]
    rng.shuffle(shortlist)
    ordered = shortlist + [p for p in stop_cells if p not in shortlist]

    for out_pos in ordered:
        if out_pos in used_outputs:
            continue
        start = _find_reverse_slide_single(blob, out_pos, dr, dc, used_inputs)
        if start is not None:
            return out_pos, start
    return None


def _place_movers(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = set(reg["interior"])
        dir_map = dict(reg["dir_map"])
        n_colors = rng.choice(ACTIVE_COLORS_PER_REGION)
        active_colors = rng.sample(list(MOVER_COLORS), n_colors)

        used_outputs: set[tuple[int, int]] = set()
        used_inputs: set[tuple[int, int]] = set()
        expected_outputs: dict[int, tuple[int, int]] = {}
        expected_inputs: dict[int, tuple[int, int]] = {}

        for color in active_colors:
            picked = _choose_output_and_input(
                rng, blob, dir_map[color][0], dir_map[color][1], used_outputs, used_inputs
            )
            if picked is None:
                return False
            out_pos, in_pos = picked
            expected_outputs[color] = out_pos
            expected_inputs[color] = in_pos
            used_outputs.add(out_pos)
            used_inputs.add(in_pos)

        for color, (r, c) in expected_inputs.items():
            grid[r, c] = color

        reg["active_colors"] = active_colors
        reg["expected_inputs"] = expected_inputs
        reg["expected_outputs"] = expected_outputs
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = input_grid.copy()
    for reg in regions:
        for r, c in dict(reg["expected_inputs"]).values():
            expected[r, c] = 0
        for color, (r, c) in dict(reg["expected_outputs"]).items():
            expected[r, c] = int(color)
    return np.array_equal(solver_out, expected)


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
    if not _place_movers(rng, regions, input_grid):
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
