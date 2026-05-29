# Combo variant of 88e364bc (index 22)
# combo_spec: D2/0+D8/0
# Name: 面积排序匹配+单像素无堆叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# spec_variant:
#   summary:
#     融合后的任务保留“边角图例给方向、封闭区域内黄点沿该方向滑动”的外观骨架，
#     但把控制关系改成“图例按阅读顺序、区域按面积顺序”一一配对，同时每个区域固定只有 1 个黄点。
#   a_concretization_and_trace:
#     example:
#       legends_in_order:
#         - L1 at top-left, fill=green, dir=right
#         - L2 at bottom-right, fill=gray, dir=up-left
#       regions_by_area:
#         - Rbig area=46, wall=gray, yellow_count=1
#         - Rsmall area=18, wall=green, yellow_count=1
#       original_logic:
#         Rbig matches gray legend L2 by wall color, so its yellow goes up-left.
#         Rsmall matches green legend L1 by wall color, so its yellow goes right.
#       combo_logic:
#         Rbig is the largest region, so it must use L1 regardless of wall color, hence moves right.
#         Rsmall is the second region by area, so it must use L2, hence moves up-left.
#         Because each region has exactly one yellow, there is no yellow-yellow blocking or stacking term.
#       visible_difference:
#         图例颜色仍然存在，但只剩“分辨不同图例”的视觉作用；真正的配对线索变成“区域面积大小”。
#         同时每个区域只有一个黄点，输出里不会出现相邻黄点挤压成堆的现象。
#   b_complexity_calibration:
#     choices:
#       - 固定 2 个图例和 2 个区域。若只有 1 个区域，面积排序退化；若更多区域则需要定义额外配对规则。
#       - 两个区域面积差至少 AREA_GAP_MIN，使“大/小”在少量样例中稳定可见。
#       - 两个图例方向必须不同，否则即使配对交换，人眼也看不出差异。
#       - 每区恰好 1 个黄点，既实现 D8/0，又让 D2 的面积排序成为主要难点，不被堆叠细节淹没。
#   c_old_assumption_audit:
#     solver_lines:
#       - 原 `dirs = _legend_directions(...): dict[fill_color -> dir]` 假设“颜色就是匹配键”。
#         融合后失效，必须改成保留图例顺序的列表。
#       - 原 `border_col = max(border_counts, ...)` 的用途是查字典。
#         融合后墙色不再决定方向，只保留给 generator 反转颜色匹配并做行为检验。
#       - 原循环只在“当前区域含黄点”时才需要处理。
#         融合后区域排序是全局操作，必须先收集全部封闭区域并按面积排序，然后再看各区黄点。
#       - 原 `_slide_points` 支持多黄点堆叠。
#         合法输入里每区只有 1 个黄点，因此 solver 在合法样本上不会进入堆叠交互；generator 负责保证这一点。
#     combined_conflicts:
#       - 若面积出现平局，面积排序会变得不可观察，因此 generator 强制显著面积差。
#       - 若仍按颜色匹配就能得到同样输出，融合会退化，因此 generator 强制 wall color 与 area rank 配对反转，
#         并量化检查 `old_color_main` 与 `main` 的输出差异。
#   d_placement_legality:
#     legends:
#       图例固定放四角中的两个角，并加 1 格缓冲，避免与墙体连成同一联通块。
#     regions:
#       继续用 grow-blob + 4 邻接 wall ring 生成封闭腔体，并验证 bounded background components 与 region interiors 完全一致。
#       两个区域都远离边界和图例缓冲区，且各自墙环不重叠。
#     yellows:
#       先在每个区域的“远端”选输出黄点，再沿反方向回溯 1..MAX_SLIDE_STEPS 步得到输入位置，
#       并用 forward slide 验证确实会停回目标位置。
#   e_visual_observability:
#     - 面积差显著，解题者能肉眼判断谁是“大区域”、谁是“小区域”。
#     - 颜色匹配被故意做反向，因此单看墙色会在样例间失败，迫使解题者转向面积排序规律。
#     - 单黄点避免了“输出变了是因为堆叠顺序还是因为匹配规则”的歧义。
#   f_visual_cleanliness:
#     - 图例仍是规整矩形，区域仍是单层彩色墙围成的黑色空腔，黄色只出现为孤立单像素。
#     - 每格中最多承担一种语义，不出现多层覆盖。
#   g_inferability_of_change:
#     - 颜色种类保持很少：2 个图例填充色 / 2 个墙色 / 1 个黄色 / 固定蓝-红图例标记。
#     - 颜色本身不参与不可见排序；排序只基于人眼可见的区域面积和图例位置。
#   new_solver_assumptions_and_generator_guards:
#     - legend_count == ranked_region_count == 2; `_place_legends` 与 `_grow_ranked_regions` 固定并验证。
#     - regions are strictly area-ordered with no tie ambiguity; `_grow_ranked_regions` enforces AREA_GAP_MIN.
#     - each controlled region has exactly 1 yellow; `_place_single_yellows` enforces this and metrics re-check it.
#     - old color matching would disagree; `_behavior_metrics` requires `swapped_assignments == 2` and `old_new_diff_cells > 0`.


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

LARGE_REGION_BOUNDS = (38, 62)
SMALL_REGION_BOUNDS = (12, 24)
AREA_GAP_MIN = 12
BLOB_GROW_RETRIES = 30
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 500


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


def _bbox_key(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _extract_legends(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    legends = []
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            candidates = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            candidates.sort()
            center_col = candidates[0][1]
            indicator_col = candidates[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue
        vr, vc = (indicator.mean(axis=0) - center.mean(axis=0)).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        legends.append(
            {
                "fill": int(fill),
                "dir": (dr, dc),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(pts),
            }
        )

    legends.sort(key=lambda item: _bbox_key(item["bbox"]))
    return legends


def _collect_regions(g: np.ndarray, bg: int, mover: int = MOVER) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    regions = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        border_counts: dict[int, int] = {}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1

        r0, c0, r1, c1 = bbox_coords(pts)
        regions.append(
            {
                "cells": pset,
                "movers": movers,
                "area": len(pset),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "border_color": max(border_counts, key=border_counts.get) if border_counts else None,
            }
        )

    regions.sort(key=lambda item: (-item["area"],) + _bbox_key(item["bbox"]))
    return regions


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
    Slide each region's lone yellow pixel according to border legends matched by area rank.

    Input structure:
    - The background is the most frequent color; border legends sit in grid corners as solid rectangles with a fill color plus center/indicator marks that encode a direction.
    - Inside the grid are exactly two enclosed regions bounded by colored walls. Each region contains exactly one yellow pixel, and the two regions have clearly different areas.

    Rule:
    - Read all legends in rectangle reading order using their top-left positions.
    - Read all enclosed regions by descending interior area, largest first.
    - Pair the first legend with the largest region and the second legend with the second-largest region, then slide that region's yellow pixel in the legend's direction until it hits the far wall.

    Key insight:
    - Wall color no longer tells you which legend controls a region; the controlling correspondence comes from a global size comparison across all regions.
    - Because each region has only one yellow pixel, motion is a pure wall-limited slide with no yellow-yellow stacking interaction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends = _extract_legends(g, bg)
    assert legends, "expected at least one border legend"

    regions = _collect_regions(g, bg, mover=MOVER)
    out = g.copy()

    for region, legend in zip(regions, legends):
        movers = region["movers"]
        if not movers:
            continue
        for r, c in movers:
            out[r, c] = bg
        dr, dc = legend["dir"]
        for r, c in _slide_points(region["cells"], movers, dr, dc):
            out[r, c] = MOVER

    return out


def _old_color_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends = _extract_legends(g, bg)
    dirs = {legend["fill"]: legend["dir"] for legend in legends}
    regions = _collect_regions(g, bg, mover=MOVER)
    out = g.copy()

    for region in regions:
        movers = region["movers"]
        border_color = region["border_color"]
        if not movers or border_color not in dirs:
            continue
        for r, c in movers:
            out[r, c] = bg
        dr, dc = dirs[border_color]
        for r, c in _slide_points(region["cells"], movers, dr, dc):
            out[r, c] = MOVER

    return out


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
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    corners = rng.sample(CORNERS, 2)
    fills = rng.sample(WALL_COLOR_CHOICES, 2)
    directions = rng.sample(ALL_DIRS, 2)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, fill, direction in zip(corners, fills, directions):
        dr, dc = direction
        patch = _legend_5x5(fill, dr, dc) if direction in CARDINAL_DIRS else _legend_6x7(fill, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"fill": fill, "dir": direction, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1)})

    legends.sort(key=lambda item: _bbox_key(item["bbox"]))
    if legends[0]["dir"] == legends[1]["dir"]:
        return legend_occupied, []
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


def _grow_ranked_regions(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    h, w = legend_occupied.shape
    occupied = legend_occupied.copy()
    ranked_specs = [
        {"wall_color": legends[1]["fill"], "dir": legends[0]["dir"], "bounds": LARGE_REGION_BOUNDS},
        {"wall_color": legends[0]["fill"], "dir": legends[1]["dir"], "bounds": SMALL_REGION_BOUNDS},
    ]
    regions = []

    for spec in ranked_specs:
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            return None

        target_size = rng.randint(*spec["bounds"])
        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None
        walls = _blob_wall_cells(blob, h, w)
        bbox = bbox_coords(np.array(sorted(blob), dtype=int))
        regions.append(
            {
                "color": spec["wall_color"],
                "dir": spec["dir"],
                "interior": blob,
                "walls": walls,
                "bbox": bbox,
                "area": len(blob),
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

    regions.sort(key=lambda item: (-item["area"],) + _bbox_key(item["bbox"]))
    if len(regions) != 2:
        return None
    if regions[0]["area"] - regions[1]["area"] < AREA_GAP_MIN:
        return None
    if regions[0]["dir"] != legends[0]["dir"] or regions[1]["dir"] != legends[1]["dir"]:
        return None
    if regions[0]["color"] != legends[1]["fill"] or regions[1]["color"] != legends[0]["fill"]:
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


def _pick_output_position(rng: random.Random, blob: set[tuple[int, int]], dr: int, dc: int) -> tuple[int, int]:
    cells_sorted = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
    tier = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]
    return tier[rng.randrange(len(tier))]


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, set()) == out_pos:
            return r, c
    return None


def _place_single_yellows(
    rng: random.Random, regions: list[dict], input_grid: np.ndarray, expected_output: np.ndarray
) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        out_pos = _pick_output_position(rng, blob, dr, dc)
        in_pos = _find_reverse_slide(blob, out_pos, dr, dc, rng)
        if in_pos is None:
            return False
        input_grid[in_pos] = MOVER
        expected_output[out_pos] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    return np.array_equal(solver_out, expected_output)


def _behavior_metrics(input_grid: np.ndarray) -> dict:
    bg = get_bg(input_grid)
    legends = _extract_legends(input_grid, bg)
    regions = _collect_regions(input_grid, bg, mover=MOVER)
    new_out = main(input_grid)
    old_out = _old_color_main(input_grid)
    return {
        "num_legends": len(legends),
        "num_regions": len(regions),
        "area_gap": regions[0]["area"] - regions[1]["area"] if len(regions) == 2 else 0,
        "input_yellows_per_region": [len(region["movers"]) for region in regions],
        "swapped_assignments": sum(
            int(region["border_color"] != legends[idx]["fill"]) for idx, region in enumerate(regions[: len(legends)])
        ),
        "old_new_diff_cells": int(np.sum(new_out != old_out)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    if len(legends) != 2:
        return None

    regions = _grow_ranked_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    expected_output = grid.copy()
    if not _place_single_yellows(rng, regions, input_grid, expected_output):
        return None
    if not _oracle_check(input_grid, expected_output):
        return None

    metrics = _behavior_metrics(input_grid)
    if metrics["num_legends"] != 2 or metrics["num_regions"] != 2:
        return None
    if metrics["area_gap"] < AREA_GAP_MIN:
        return None
    if metrics["input_yellows_per_region"] != [1, 1]:
        return None
    if metrics["swapped_assignments"] != 2:
        return None
    if metrics["old_new_diff_cells"] <= 0:
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
