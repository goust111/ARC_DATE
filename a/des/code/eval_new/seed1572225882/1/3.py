"""
按种子与封闭区域边界的距离决定生长方向，并沿该方向逐层展开种子颜色。

Input structure:
- 背景色为黑色（bg），灰色边界线将网格划分为多个不相连的封闭区域。
- 每个区域内部是 bg，内部可能分布 1 到 3 个单像素种子。
- 种子颜色来自固定可见色表（red、blue、green、yellow、orange、magenta、azure、maroon）。
- 区域外侧可能还有大范围 bg 区域。

Rule:
- 识别灰色边界并找出所有不与边界接触的封闭区域。
- 每个种子所属区域中分别计算到上、下、左、右边界的最近距离，距离最小者决定生长方向；若平局按 上 > 右 > 下 > 左。
- 对于每个种子，取其所在行（向上/下生长）或所在列（向左/右生长）在其所属区域中的连续跨度作为每层线段长度。
- 从第 1 步起，同步逐层沿生长方向推进：每一步用该种子颜色填充对应一整条线段。
- 如果某一层遇到已有其他种子颜色、边界外或越界，则该种子停止后续扩展。
- 同一步中多个种子争夺同一像素时，离该像素更近者优先；若距离相同则按颜色顺序（red,blue,green,yellow,orange,magenta,azure,maroon）裁决。

Key insight:
- 种子并非固定方向生长，而是由其在封闭区域内到四条方向边界的相对位置编码方向；多种子时按层同步扩展即可让“最近、同层平局、颜色次序”自然决定领地边界。
"""

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

BG_COLOR = 0
BOUNDARY_COLOR = 1
PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
COLOR_RANK = {c: i for i, c in enumerate(PALETTE)}
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ["up", "right", "down", "left"]
DIR_INDEX = {name: idx for idx, name in enumerate(DIR_ORDER)}
DIR_NAMES = DIR_ORDER


def _color_rank(color: int) -> int:
    return COLOR_RANK.get(int(color), len(PALETTE) + int(color))


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _collect_components(grid: np.ndarray, blocked: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    h, w = grid.shape
    comp_id = np.full((h, w), -1, dtype=np.int32)
    comps: List[Dict[str, object]] = []
    cid = 0

    for r in range(h):
        for c in range(w):
            if blocked[r, c] or comp_id[r, c] != -1:
                continue
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells = [(r, c)]
            touches = r in (0, h - 1) or c in (0, w - 1)

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and comp_id[nr, nc] == -1:
                        comp_id[nr, nc] = cid
                        touches = touches or nr in (0, h - 1) or nc in (0, w - 1)
                        q.append((nr, nc))
                        cells.append((nr, nc))

            comps.append({"id": cid, "cells": cells, "touches_border": touches})
            cid += 1

    return comp_id, comps


def _direction_from_seed(r: int, c: int, comp_id: np.ndarray, cid: int, h: int, w: int) -> str:
    dists = []

    # 上
    d = 0
    rr = r - 1
    while _inside(h, w, rr, c) and comp_id[rr, c] == cid:
        d += 1
        rr -= 1
    dists.append(d)

    # 右
    d = 0
    cc = c + 1
    while _inside(h, w, r, cc) and comp_id[r, cc] == cid:
        d += 1
        cc += 1
    dists.append(d)

    # 下
    d = 0
    rr = r + 1
    while _inside(h, w, rr, c) and comp_id[rr, c] == cid:
        d += 1
        rr += 1
    dists.append(d)

    # 左
    d = 0
    cc = c - 1
    while _inside(h, w, r, cc) and comp_id[r, cc] == cid:
        d += 1
        cc -= 1
    dists.append(d)

    best_idx = min(range(4), key=lambda i: (dists[i], [0, 1, 2, 3][i]))
    return DIR_NAMES[best_idx]


def _row_span(r: int, c: int, comp_id: np.ndarray, cid: int, cmin: int, cmax: int) -> Tuple[int, int]:
    cc1 = c
    while cc1 - 1 >= cmin and comp_id[r, cc1 - 1] == cid:
        cc1 -= 1
    cc2 = c
    while cc2 + 1 <= cmax and comp_id[r, cc2 + 1] == cid:
        cc2 += 1
    return cc1, cc2


def _col_span(r: int, c: int, comp_id: np.ndarray, cid: int, rmin: int, rmax: int) -> Tuple[int, int]:
    rr1 = r
    while rr1 - 1 >= rmin and comp_id[rr1 - 1, c] == cid:
        rr1 -= 1
    rr2 = r
    while rr2 + 1 <= rmax and comp_id[rr2 + 1, c] == cid:
        rr2 += 1
    return rr1, rr2


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid)
    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    output = grid.copy()

    # Identify background and boundary color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    non_bg_mask = grid != bg
    if not np.any(non_bg_mask):
        return output

    non_bg_colors, non_bg_counts = np.unique(grid[non_bg_mask], return_counts=True)
    boundary_color = int(non_bg_colors[np.argmax(non_bg_counts)])

    comp_id, comps = _collect_components(grid, grid == boundary_color)

    inside_regions = {comp["id"] for comp in comps if not comp["touches_border"]}

    seed_cells = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col != bg and col != boundary_color:
                rid = comp_id[r, c]
                if rid in inside_regions:
                    seed_cells.append((r, c, col, rid))

    if not seed_cells:
        return output

    # Map region id -> seeds
    by_region: Dict[int, List[Tuple[int, int, int]]] = {}
    for r, c, col, rid in seed_cells:
        by_region.setdefault(rid, []).append((r, c, col))

    # Precompute bounds per component for span extension.
    comp_bounds: Dict[int, Tuple[int, int, int, int]] = {}
    for comp in comps:
        if comp["touches_border"]:
            continue
        rid = comp["id"]
        rs = [p[0] for p in comp["cells"]]
        cs = [p[1] for p in comp["cells"]]
        comp_bounds[rid] = (min(rs), max(rs), min(cs), max(cs))

    seed_info = []
    for rid, seeds in by_region.items():
        rmin, rmax, cmin, cmax = comp_bounds[rid]
        for r, c, col in seeds:
            direction = _direction_from_seed(r, c, comp_id, rid, h, w)
            if direction in ("up", "down"):
                c1, c2 = _row_span(r, c, comp_id, rid, cmin, cmax)
                if c1 <= c2:
                    seed_info.append({"r": r, "c": c, "color": col, "rid": rid, "dir": direction, "span": (c1, c2)})
            else:
                r1, r2 = _col_span(r, c, comp_id, rid, rmin, rmax)
                if r1 <= r2:
                    seed_info.append({"r": r, "c": c, "color": col, "rid": rid, "dir": direction, "span": (r1, r2)})

    if not seed_info:
        return output

    active = set(range(len(seed_info)))

    for step in range(1, max(h, w) + 1):
        if not active:
            break

        proposals: Dict[int, List[Tuple[int, int]]] = {}
        blocked_this_round = set()

        for idx in list(active):
            info = seed_info[idx]
            rid = info["rid"]
            col = info["color"]
            direction = info["dir"]
            r0 = info["r"]
            c0 = info["c"]

            cells: List[Tuple[int, int]] = []
            valid = True

            if direction == "up":
                dr, dc = DIRS["up"]
                rr = r0 + dr * step
                c_start, c_end = info["span"]
                for cc in range(c_start, c_end + 1):
                    if not (0 <= rr < h and 0 <= cc < w):
                        valid = False
                        break
                    if comp_id[rr, cc] != rid:
                        valid = False
                        break
                    cells.append((rr, cc))
            elif direction == "down":
                dr, dc = DIRS["down"]
                rr = r0 + dr * step
                c_start, c_end = info["span"]
                for cc in range(c_start, c_end + 1):
                    if not (0 <= rr < h and 0 <= cc < w):
                        valid = False
                        break
                    if comp_id[rr, cc] != rid:
                        valid = False
                        break
                    cells.append((rr, cc))
            elif direction == "left":
                dr, dc = DIRS["left"]
                cc = c0 + dc * step
                r_start, r_end = info["span"]
                for rr in range(r_start, r_end + 1):
                    if not (0 <= rr < h and 0 <= cc < w):
                        valid = False
                        break
                    if comp_id[rr, cc] != rid:
                        valid = False
                        break
                    cells.append((rr, cc))
            else:
                dr, dc = DIRS["right"]
                cc = c0 + dc * step
                r_start, r_end = info["span"]
                for rr in range(r_start, r_end + 1):
                    if not (0 <= rr < h and 0 <= cc < w):
                        valid = False
                        break
                    if comp_id[rr, cc] != rid:
                        valid = False
                        break
                    cells.append((rr, cc))

            if not valid:
                active.discard(idx)
                continue

            # 已占用（边界或其他已生长种子）阻挡该种子继续生长。
            for rr, cc in cells:
                if output[rr, cc] != bg:
                    valid = False
                    break

            if not valid:
                active.discard(idx)
                continue

            proposals[idx] = cells
            for cell in cells:
                blocked_this_round.add(cell)

        if not proposals:
            break

        cell_to_seed: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for idx, cells in proposals.items():
            for cell in cells:
                cell_to_seed.setdefault(cell, []).append(idx)

        winner_cells: Dict[int, List[Tuple[int, int]]] = {idx: [] for idx in proposals}
        losers = set()

        for cell, idxs in cell_to_seed.items():
            if len(idxs) == 1:
                winner_cells[idxs[0]].append(cell)
            else:
                # 同一步内距离相同，按色序裁决。
                rr = sorted(idxs, key=lambda k: (_color_rank(seed_info[k]["color"])))
                winner = rr[0]
                winner_cells[winner].append(cell)
                for loser in idxs:
                    if loser != winner:
                        losers.add(loser)

        # 应用胜出像素。
        for idx in list(proposals):
            if idx in losers:
                active.discard(idx)
                continue
            col = seed_info[idx]["color"]
            for rr, cc in winner_cells[idx]:
                output[rr, cc] = col

    return output


def _find_regions(grid: np.ndarray, boundary_color: int) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    return _collect_components(grid, grid == boundary_color)


def _draw_region_box(grid: np.ndarray, reserved: np.ndarray, rng: np.random.Generator, h: int, w: int) -> bool:
    max_inner_h = min(10, h - 6)
    max_inner_w = min(10, w - 6)
    if max_inner_h < 4 or max_inner_w < 4:
        return False

    inner_h = int(rng.integers(4, max_inner_h + 1))
    inner_w = int(rng.integers(4, max_inner_w + 1))
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if h - outer_h - 1 <= 1 or w - outer_w - 1 <= 1:
        return False

    r0 = int(rng.integers(1, h - outer_h - 1))
    c0 = int(rng.integers(1, w - outer_w - 1))
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
        return False

    grid[r0, c0 : c1 + 1] = BOUNDARY_COLOR
    grid[r1, c0 : c1 + 1] = BOUNDARY_COLOR
    grid[r0:r1 + 1, c0] = BOUNDARY_COLOR
    grid[r0:r1 + 1, c1] = BOUNDARY_COLOR

    # 简单的不规则边界：随机向内延伸一两个“短齿”使外形更有变化。
    for _ in range(int(rng.integers(0, 3))):
        side = int(rng.integers(0, 4))
        depth = int(rng.integers(1, 3))
        if side in (0, 2):
            # top / bottom
            y = r0 + (1 if side == 0 else -1)
            if side == 2:
                y = r1 - 1
            x0 = int(rng.integers(c0 + 1, c1 - 1))
            span = int(rng.integers(1, min(4, max(2, outer_w // 3)) + 1))
            for x in range(x0, min(x0 + span, c1)):
                for d in range(1, depth + 1):
                    rr = (r0 + d) if side == 0 else (r1 - d)
                    if 0 <= rr < h:
                        grid[rr, x] = BOUNDARY_COLOR
        else:
            # left / right
            x = c0 + (1 if side == 3 else -1)
            if side == 1:
                x = c1 - 1
            y0 = int(rng.integers(r0 + 1, r1 - 1))
            span = int(rng.integers(1, min(4, max(2, outer_h // 3)) + 1))
            for y in range(y0, min(y0 + span, r1)):
                for d in range(1, depth + 1):
                    cc = (c0 + d) if side == 3 else (c1 - d)
                    if 0 <= cc < w:
                        grid[y, cc] = BOUNDARY_COLOR

    reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
    return True


def _determine_target_direction(rng: np.random.Generator) -> str:
    return DIR_NAMES[int(rng.integers(0, 4))]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(1000):
        h = int(rng.integers(16, 30))
        w = int(rng.integers(16, 30))
        grid = np.full((h, w), BG_COLOR, dtype=np.int64)
        reserved = np.zeros((h, w), dtype=bool)

        target_regions = int(rng.integers(2, 6))
        count = 0
        for _ in range(200):
            if count >= target_regions:
                break
            if _draw_region_box(grid, reserved, rng, h, w):
                count += 1

        if count != target_regions:
            continue

        _, comps = _find_regions(grid, BOUNDARY_COLOR)
        inner = [c for c in comps if not c["touches_border"]]
        if len(inner) != target_regions:
            continue

        inner = sorted(inner, key=lambda x: len(x["cells"]))

        # 需要给每个封闭区域至少一个种子，且每个区域种子数在 1~3。
        ok = True
        comp_id, comps = _collect_components(grid, grid == BOUNDARY_COLOR)
        comp_ids = {comp["id"] for comp in comps if not comp["touches_border"]}

        if len(comp_ids) != target_regions:
            continue

        for comp in comps:
            if comp["touches_border"]:
                continue
            if len(comp["cells"]) < 9:
                ok = False
                break

        if not ok:
            continue

        for rid in comp_ids:
            cells = next(comp["cells"] for comp in comps if comp["id"] == rid)
            cells_list = cells
            if not cells_list:
                ok = False
                break

            num_seeds = int(rng.choice([1, 2, 3]))
            used_colors = set()
            target_dirs = [_determine_target_direction(rng) for _ in range(num_seeds)]

            for target_dir in target_dirs:
                placed = False
                for _ in range(80):
                    r, c = cells_list[int(rng.integers(0, len(cells_list)))]
                    if grid[r, c] != BG_COLOR:
                        continue
                    rr, cc = int(r), int(c)
                    d = _direction_from_seed(rr, cc, comp_id, rid, h, w)
                    if d != target_dir:
                        continue
                    available = [c_ for c_ in PALETTE if c_ not in used_colors]
                    if not available:
                        available = list(PALETTE)
                    color = int(rng.choice(available))
                    used_colors.add(color)
                    grid[r, c] = color
                    placed = True
                    break
                if not placed:
                    # 降级到任意可用位置，确保样例仍可生成。
                    attempts = [p for p in cells_list if grid[p[0], p[1]] == BG_COLOR]
                    if not attempts:
                        ok = False
                        break
                    rr, cc = attempts[0]
                    available = [c_ for c_ in PALETTE if c_ not in used_colors]
                    if not available:
                        available = list(PALETTE)
                    color = int(rng.choice(available))
                    used_colors.add(color)
                    grid[rr, cc] = color
            if not ok:
                break

        if not ok:
            continue

        # 快速检查：每个区域都有1~3个种子且颜色不重复。
        for comp in comps:
            if comp["touches_border"]:
                continue
            rid = comp["id"]
            colors = []
            for r, c in comp["cells"]:
                v = int(grid[r, c])
                if v not in (BG_COLOR, BOUNDARY_COLOR):
                    colors.append(v)
            if not (1 <= len(colors) <= 3):
                ok = False
                break
            if len(colors) != len(set(colors)):
                ok = False
                break

        if not ok:
            continue

        return grid

    # 兜底：返回一个最小可运行样例。
    h = 18
    w = 18
    grid = np.full((h, w), BG_COLOR, dtype=np.int64)
    grid[2, 2 : w - 2] = BOUNDARY_COLOR
    grid[h - 3, 2 : w - 2] = BOUNDARY_COLOR
    grid[2 : h - 2, 2] = BOUNDARY_COLOR
    grid[2 : h - 2, w - 3] = BOUNDARY_COLOR
    grid[8, 8] = PALETTE[0]
    grid[9, 13] = PALETTE[1]
    return grid
