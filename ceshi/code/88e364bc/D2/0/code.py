# Variant of 88e364bc
# Dimension: D2 — 图例与区域的匹配机制 (symbol_instructions)
# Change: legend_lookup——图例填充色C与区域边界墙壁的多数颜色匹配，每种墙壁颜色恰好对应一个图例... → 按区域面积排序匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# spec_variant:
#   core_change:
#     original: 每个封闭区域先统计邻接墙壁的多数颜色, 再去匹配同填充色的边缘图例, 读出方向.
#     variant: 先把所有图例按其左上角 bbox 坐标做阅读顺序排序, 再把所有封闭区域按面积从大到小排序,
#       第 k 大区域使用第 k 个图例的方向, 与墙壁颜色无关.
#   concrete_visualization:
#     legends:
#       count: 固定 2 个, 仍放在四角中的两个角, 保持原题 5x5/6x7 箭头图例样式.
#       order: 按 bbox 左上角 `(r0, c0)` 升序, 即“从上到下, 同行从左到右”.
#     regions:
#       count: 固定 2 个封闭区域, 都包含黄色像素.
#       area: 一个显著更大, 一个显著更小, 面积差至少 10, 避免人眼无法稳定排序.
#       walls: 仍为单色不规则闭墙, 两个区域墙色不同.
#     visibility_support:
#       为了让 D2 变化真正可见, 生成器故意把“区域面积排序”和“颜色匹配”做反向:
#       排名第 1 的大区域使用第 1 个图例方向, 但它的墙色等于第 2 个图例的填充色;
#       排名第 2 的小区域使用第 2 个图例方向, 但它的墙色等于第 1 个图例的填充色.
#       因此同一输入若按原题颜色规则求解, 两个区域会交换方向.
#   behavior_trace:
#     example:
#       legends_in_order:
#         - L1 at top row, fill=green, dir=right
#         - L2 at bottom row, fill=gray, dir=up-left
#       regions_by_area:
#         - Rbig area=48, wall=gray
#         - Rsmall area=18, wall=green
#       original_logic:
#         Rbig sees gray wall -> matches L2 -> slides up-left
#         Rsmall sees green wall -> matches L1 -> slides right
#       variant_logic:
#         Rbig is largest -> matches L1 -> slides right
#         Rsmall is second -> matches L2 -> slides up-left
#       difference:
#         同样的墙色与图例颜色仍然存在, 但不再承担匹配语义; 语义来自“图例位置顺序”和“区域面积顺序”.
#   complexity_calibration:
#     why_fixed_two_legends_two_regions:
#       - 若只有 1 个图例, 面积排序退化, 新旧规则几乎不可区分.
#       - 若区域数多于图例数, 需要额外定义未匹配区域如何处理, 会引入与 D2 无关的新维度.
#       - 两个区域已经足以体现“最大/次大”的数值比较, 同时保证 2-4 个样例内可归纳.
#     why_distinct_directions:
#       若两个图例方向相同, 即使匹配关系改变, 输出也可能不变, 因而要求两个方向必须不同.
#   old-assumption-audit:
#     relevant_main_lines:
#       - `dirs = _legend_directions(g, bg)`: 旧代码把图例组织成 `{fill_color: direction}` 字典.
#         依赖前提: “填充色就是控制该区域的键”. 变体中失效, 必须改成保留位置顺序的列表.
#       - `border_col = max(border_counts, key=border_counts.get)`: 旧代码计算墙色只是为了查图例.
#         变体中墙色不再控制方向, solver 不应再据此分派方向.
#       - `if border_col not in dirs: continue`: 旧题中无害, 新题若保留这行会让大部分样本完全不滑动.
#       - `for pts in components_mask(passable...)`: 旧题只处理含黄区域也能工作.
#         新题中面积排序必须基于所有封闭区域, 所以要先收集全部区域再排序, 之后再看哪些区域有黄点.
#   placement_legality:
#     legends:
#       图例与墙体之间保留至少 1 格缓冲, 避免 border-touching 非背景联通块把图例与墙连成一个组件.
#     regions:
#       仍通过 grow blob + 1-cell wall ring 生成封闭区域, 并验证 bounded bg components 与 region interiors 一致.
#     yellows:
#       输出位置先按正确方向选在远端, 再反推输入位置 1..5 步, 并用 solver 同步顺序验证.
#   visual_cleanliness:
#     - 图例仍是规整实心矩形, 区域仍是单层墙+空腔, 黄色仅出现在区域内部.
#     - 只使用 2 个图例和 2 个区域, 降低网格杂乱度.
#   color_reasoning:
#     颜色仍然提供“这是两个不同区域/两个不同图例”的分组线索, 但不能再单独解释控制关系.
#     解题者必须结合面积大小与图例位置顺序, 才能统一解释所有样例.
#   new_solver_assumptions_to_enforce:
#     - 图例数量 == 需要被控制的区域数量. 生成器固定为 2==2.
#     - 区域面积无平局且差距明显. 生成器强制面积差至少 10.
#     - 图例方向不同. 生成器强制两个方向不同.
#     - 图例之间、图例与墙体不相连. 生成器用角落放置和 buffer 保证.


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

LARGE_REGION_BOUNDS = (38, 62)
SMALL_REGION_BOUNDS = (12, 24)
AREA_GAP_MIN = 10
BLOB_GROW_RETRIES = 30
YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 400


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
            centers = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                centers.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            centers.sort()
            center_col = centers[0][1]
            indicator_col = centers[1][1]
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
    Variant rule:
    - Detect all border legends and sort them by reading order of their rectangles.
    - Detect all enclosed regions and sort them by descending area.
    - The largest region uses the first legend, the second largest uses the second legend, etc.
    - Yellow pixels inside each matched region slide in that legend's direction.
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

    occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, fill, direction in zip(corners, fills, directions):
        dr, dc = direction
        patch = _legend_5x5(fill, dr, dc) if direction in CARDINAL_DIRS else _legend_6x7(fill, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"fill": fill, "dir": direction, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1)})

    legends.sort(key=lambda item: _bbox_key(item["bbox"]))
    if legends[0]["dir"] == legends[1]["dir"]:
        return occupied, []
    return occupied, legends


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
        pts = np.array(sorted(blob), dtype=int)
        bbox = bbox_coords(pts)
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
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return r, c
    return None


def _place_yellows(
    rng: random.Random, regions: list[dict], input_grid: np.ndarray, expected_output: np.ndarray
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
            input_grid[r, c] = MOVER
        for r, c in output_positions:
            expected_output[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    return np.array_equal(solver_out, expected_output)


def _variant_signal(input_grid: np.ndarray) -> dict:
    bg = get_bg(input_grid)
    legends = _extract_legends(input_grid, bg)
    regions = _collect_regions(input_grid, bg, mover=MOVER)
    new_out = main(input_grid)
    old_out = _old_color_main(input_grid)
    return {
        "legend_fills_in_order": [legend["fill"] for legend in legends],
        "region_areas": [region["area"] for region in regions],
        "region_colors_by_area": [region["border_color"] for region in regions],
        "area_gap": regions[0]["area"] - regions[1]["area"] if len(regions) >= 2 else 0,
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
    if not _place_yellows(rng, regions, input_grid, expected_output):
        return None
    if not _oracle_check(input_grid, expected_output):
        return None

    signal = _variant_signal(input_grid)
    if signal["area_gap"] < AREA_GAP_MIN:
        return None
    if signal["swapped_assignments"] != 2:
        return None
    if signal["old_new_diff_cells"] <= 0:
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
