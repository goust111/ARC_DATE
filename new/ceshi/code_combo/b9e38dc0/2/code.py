# Combo variant of b9e38dc0 (index 2)
# combo_spec: D1/1+D6/0
# Name: 多开口辐射+多开口围栏

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


DESIGN_ANALYSIS = """
Combo design analysis
=====================

a) 具体化与行为追踪
- 原题依赖单个半包围墙体的两个端点来推出唯一开口方向, 然后把内部填满, 再沿同一方向向外延展。
- 融合后把墙体改成 1 像素厚的矩形围栏, bbox 四条边上会出现 2 到 3 个连续缺口。种子仍在围栏内部。
- 一个具体例子: 如果围栏顶部有 2 格缺口、左侧有 1 格缺口, 原题式的“合成一个主方向”会失败,
  因为顶部和左侧分别对应向上、向左两条辐射。新逻辑先只在 bbox 内部做 4 邻域洪水填充, 再让
  每个缺口各自发出一条轴对齐的带状射线; 因此输出会同时在上边界和左边界看到填充色抵达。

b) 复杂度校准
- 如果允许任意厚度、任意数量开口, 视觉会过乱。这里把围栏固定为薄矩形, 开口数限制为 2 或 3,
  开口长度限制为 1 或 2, 既能明显看出“多开口”, 又不会让外部辐射区互相覆盖成一团。
- 生成器要求至少两个不同朝向的边参与开口, 并要求融合版输出相比“只处理第一个开口”的基线
  多出一块可见填充区, 防止样本退化成单开口等价情况。

c) 隐式假设审查
- 原 solver 的 degree-1 端点检测失效: 矩形缺口不会产生原题那种唯一端点对。
- 原 solver 的单一 `rad` 失效: 多个缺口可能指向不同边, 不能再合并成一个全局辐射方向。
- 原 solver 把“围栏内部 + 外部楔形区”放在一次 BFS 里处理, 依赖的是单口楔形边界; 这里必须改成
  “先识别 bbox 内部, 再逐个缺口处理外辐射与阴影”。
- 新 solver 的缺口检测依赖两个前提: 主墙组件近似薄矩形边框, 且真正的开口表现为 bbox 边上的连续
  背景段。生成器只构造这种围栏, 所以这些前提是被显式保证的。

d) 放置合法性
- 围栏整体放在网格内部, 开口边朝向的外部至少留 3 格空间, 保证辐射可见。
- 开口只允许出现在非角点位置。为了让墙色仍是单个 8 连通组件, 生成器会在部分缺口后方补一条
  1 格深的内桥; 这不影响 bbox 边缺口的可见性, 也不影响 solver 的缺口检测。
- 种子严格放在围栏 bbox 内部且不与墙重叠。障碍只放在背景格上并与墙保持至少 1 格距离。

e) 视觉可观察性
- 人类在样例里能直接看到: 输入有一个带多个缺口的围栏, 输出先灌满内部, 再从每个缺口各自向外
  延伸。因为开口分布在不同边上, “每个缺口都要辐射”会比“只有一个主开口”更容易被看出来。
- 规则完全由空间结构触发, 不依赖颜色编号大小、扫描顺序等人类不可视因素。

f) 可视化整洁度
- 开口数量少, 且平行边上的开口投影被强制错开, 所以外部带状辐射通常不会大面积重叠。
- 障碍数量保持稀疏, 只作为局部阴影证据, 不会淹没围栏和辐射结构。

g) 颜色变化可推断性
- 颜色机制不变: 墙色保持不变, 种子色扩张成填充色, 障碍色保持原样。真正变化的只有
  “背景 -> 填充色”, 且变化区域由“围栏内部 + 多个缺口对应的辐射带”共同决定。

h) 融合行为量化与反退化
- 量化指标采用 `multi_bonus`: 融合 solver 输出中为填充色、但“仅处理第一个缺口”的基线 solver
  不会填到的背景格数量。生成器要求该值达到阈值, 并要求填充至少触达两个边界。
- 因此通过测试的不只是“solver 不报错”, 还保证样本里确实存在多开口带来的额外辐射效果。

i) 新 solver 假设与 generator 保证逐项对照
- `_find_openings()` 假设主墙组件的 bbox 边上能读出缺口: generator 只画矩形边框并只在边框挖洞。
- `_flood_room()` 假设种子在 bbox 内部: generator 从内部种子池中选点并扩成小 blob。
- `_apply_opening_fill()` 假设每个缺口都对应一条轴对齐外射线: generator 只使用 top/bottom/left/right 缺口。
- `_apply_opening_shadows()` 假设障碍是稀疏的非墙非填充像素: generator 只在背景上放少量障碍并复用原题
  “障碍后方变回背景”的机制。
"""


GRID_H = (12, 18)
GRID_W = (15, 19)

FENCE_H = (7, 10)
FENCE_W = (8, 12)
OPEN_LEN = (1, 2)
OPEN_OUTER_MARGIN = 3
SIDE_MARGIN = 1

WALL_PIXELS_MIN = 20
WALL_PIXELS_MAX = 36

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]

MAX_OBSTACLES = 4
OBSTACLE_IN_FILL_PROB = 0.45
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 18
MIN_MULTI_BONUS = 5
MIN_BORDER_HITS = 2
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 500

SIDE_ORDER = {"top": 0, "bottom": 1, "left": 2, "right": 3}


def _component_mask(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _sorted_openings(openings: list[dict]) -> list[dict]:
    return sorted(openings, key=lambda op: (SIDE_ORDER[op["side"]], op["span"][0], op["span"][1]))


def _find_fill_color(g: np.ndarray, bg: int, wall: int, wall_points: list[tuple[int, int]]) -> int:
    vals = [int(v) for v in np.unique(g) if int(v) not in (bg, wall)]
    assert vals, "expected at least one non-wall color"

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    return min(vals, key=lambda color: (min_dist_to_wall(color)[0], min_dist_to_wall(color)[1]))


def _select_wall_and_fill(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid)
    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda color: int((g == color).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    wall_points = [p for comp in comps for p in comp]
    fill = _find_fill_color(g, bg, wall, wall_points)
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill-color seed pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    return bg, wall, fill, fill_pos, wall_comp


def _false_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(values):
        if values[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(values) and not values[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _find_openings(comp_mask: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    coords = np.argwhere(comp_mask)
    assert coords.size, "expected non-empty wall component"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)

    openings: list[dict] = []

    top = [bool(comp_mask[r0, c]) for c in range(c0 + 1, c1)]
    for a, b in _false_runs(top):
        lo = c0 + 1 + a
        hi = c0 + 1 + b
        openings.append(
            {
                "side": "top",
                "dir": (-1, 0),
                "span": (lo, hi),
                "cells": [(r0, c) for c in range(lo, hi + 1)],
                "inner": [(r0 + 1, c) for c in range(lo, hi + 1)],
            }
        )

    bottom = [bool(comp_mask[r1, c]) for c in range(c0 + 1, c1)]
    for a, b in _false_runs(bottom):
        lo = c0 + 1 + a
        hi = c0 + 1 + b
        openings.append(
            {
                "side": "bottom",
                "dir": (1, 0),
                "span": (lo, hi),
                "cells": [(r1, c) for c in range(lo, hi + 1)],
                "inner": [(r1 - 1, c) for c in range(lo, hi + 1)],
            }
        )

    left = [bool(comp_mask[r, c0]) for r in range(r0 + 1, r1)]
    for a, b in _false_runs(left):
        lo = r0 + 1 + a
        hi = r0 + 1 + b
        openings.append(
            {
                "side": "left",
                "dir": (0, -1),
                "span": (lo, hi),
                "cells": [(r, c0) for r in range(lo, hi + 1)],
                "inner": [(r, c0 + 1) for r in range(lo, hi + 1)],
            }
        )

    right = [bool(comp_mask[r, c1]) for r in range(r0 + 1, r1)]
    for a, b in _false_runs(right):
        lo = r0 + 1 + a
        hi = r0 + 1 + b
        openings.append(
            {
                "side": "right",
                "dir": (0, 1),
                "span": (lo, hi),
                "cells": [(r, c1) for r in range(lo, hi + 1)],
                "inner": [(r, c1 - 1) for r in range(lo, hi + 1)],
            }
        )

    return _sorted_openings(openings), (r0, c0, r1, c1)


def _flood_room(
    g: np.ndarray,
    comp_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    fill: int,
    fill_pos: list[tuple[int, int]],
) -> np.ndarray:
    h, w = g.shape
    r0, c0, r1, c1 = bbox
    room_mask = np.zeros_like(comp_mask, dtype=bool)
    if r1 - r0 > 1 and c1 - c0 > 1:
        room_mask[r0 + 1:r1, c0 + 1:c1] = True
    passable = room_mask & (~comp_mask) & ((g == bg) | (g == fill))

    seen = np.zeros_like(passable, dtype=bool)
    q = [p for p in fill_pos if passable[p[0], p[1]]]
    for r, c in q:
        seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _apply_opening_fill(
    g: np.ndarray,
    filled_mask: np.ndarray,
    openings: list[dict],
    bbox: tuple[int, int, int, int],
    bg: int,
) -> None:
    h, w = g.shape
    r0, c0, r1, c1 = bbox
    for opening in openings:
        lo, hi = opening["span"]
        side = opening["side"]
        if side == "top":
            for c in range(lo, hi + 1):
                for r in range(0, r0 + 1):
                    if g[r, c] == bg:
                        filled_mask[r, c] = True
        elif side == "bottom":
            for c in range(lo, hi + 1):
                for r in range(r1, h):
                    if g[r, c] == bg:
                        filled_mask[r, c] = True
        elif side == "left":
            for r in range(lo, hi + 1):
                for c in range(0, c0 + 1):
                    if g[r, c] == bg:
                        filled_mask[r, c] = True
        else:
            for r in range(lo, hi + 1):
                for c in range(c1, w):
                    if g[r, c] == bg:
                        filled_mask[r, c] = True


def _shadow_lines_for_opening(opening: dict, h: int, w: int) -> list[list[tuple[int, int]]]:
    lo, hi = opening["span"]
    direction = opening["dir"]
    lines: list[list[tuple[int, int]]] = []
    if direction == (-1, 0):
        rows = range(h - 1, -1, -1)
        for c in range(lo, hi + 1):
            lines.append([(r, c) for r in rows])
    elif direction == (1, 0):
        rows = range(h)
        for c in range(lo, hi + 1):
            lines.append([(r, c) for r in rows])
    elif direction == (0, -1):
        cols = range(w - 1, -1, -1)
        for r in range(lo, hi + 1):
            lines.append([(r, c) for c in cols])
    else:
        cols = range(w)
        for r in range(lo, hi + 1):
            lines.append([(r, c) for c in cols])
    return lines


def _apply_opening_shadows(
    out: np.ndarray,
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    openings: list[dict],
    bg: int,
) -> None:
    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    h, w = out.shape
    for opening in openings:
        for coords in _shadow_lines_for_opening(opening, h, w):
            shadow_process(coords)


def _solve_core(grid: np.ndarray, opening_limit: int | None = None) -> np.ndarray:
    g = np.array(grid, copy=True)
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(g)

    comp_mask = _component_mask(wall_comp, g.shape)
    openings, bbox = _find_openings(comp_mask)
    assert len(openings) >= 2, "expected a fence with two or more openings"
    if opening_limit is not None:
        openings = openings[:opening_limit]

    room_seen = _flood_room(g, comp_mask, bbox, bg, fill, fill_pos)
    filled_mask = room_seen & (g == bg)
    _apply_opening_fill(g, filled_mask, openings, bbox, bg)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    _apply_opening_shadows(out, filled_mask, obstacles, openings, bg)
    return out


def analyze_primary_fence(grid: np.ndarray) -> dict:
    bg, wall, fill, fill_pos, wall_comp = _select_wall_and_fill(grid)
    comp_mask = _component_mask(wall_comp, grid.shape)
    openings, bbox = _find_openings(comp_mask)
    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pixels": len(fill_pos),
        "opening_count": len(openings),
        "opening_sides": tuple(sorted({op["side"] for op in openings})),
        "bbox": bbox,
        "spans": tuple((op["side"], op["span"]) for op in openings),
    }


def behavior_metrics(grid: np.ndarray) -> dict:
    info = analyze_primary_fence(grid)
    out = main(grid)
    base = _single_opening_baseline(grid)
    fill = info["fill"]
    new_fill = (out == fill) & (grid == info["bg"])
    multi_bonus = (out == fill) & (base != fill)
    return {
        "opening_count": info["opening_count"],
        "unique_sides": len(info["opening_sides"]),
        "border_hits": _border_hit_count(new_fill),
        "new_fill": int(new_fill.sum()),
        "multi_bonus": int(multi_bonus.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a fenced interior and let the color leak outward through every visible gap.

    Input structure:
    - The background is the most common color. The wall is the dominant non-background color and
      forms a thin rectangular fence whose bbox edges contain two or more missing segments.
    - A non-wall seed blob lies inside that fence. Any other non-background pixels are obstacles
      sitting on background cells.
    Rule:
    - First flood-fill the reachable interior of the fence from the seed.
    - Then extend the fill outward through every detected edge gap, producing one axis-aligned
      radiating strip per opening. Obstacles stay fixed and erase filled cells behind them along
      each opening's own outward direction.

    Key insight: The openings are continuous gaps on the fence border, not a single endpoint-defined
    mouth; each gap contributes an independent outward ray and an independent shadow direction.
    """
    return _solve_core(grid)


def _single_opening_baseline(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, opening_limit=1)


def _rect_border_points(height: int, width: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(width):
        pts.add((0, c))
        pts.add((height - 1, c))
    for r in range(height):
        pts.add((r, 0))
        pts.add((r, width - 1))
    return pts


def _interval_choices(length: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for size in range(OPEN_LEN[0], OPEN_LEN[1] + 1):
        start_lo = 2
        start_hi = length - size - 2
        if start_lo > start_hi:
            continue
        for start in range(start_lo, start_hi + 1):
            choices.append((start, start + size - 1))
    return choices


def _gap_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _choose_openings(frame_h: int, frame_w: int) -> list[dict]:
    patterns = [
        ["top", "left"],
        ["top", "right"],
        ["bottom", "left"],
        ["bottom", "right"],
        ["top", "bottom"],
        ["left", "right"],
        ["top", "left", "right"],
        ["bottom", "left", "right"],
    ]
    weights = [4, 4, 4, 4, 3, 3, 1, 1]

    for _ in range(160):
        sides = random.choices(patterns, weights=weights, k=1)[0]
        openings: list[dict] = []
        valid = True
        used_parallel: dict[str, tuple[int, int]] = {}
        for side in sides:
            length = frame_w if side in ("top", "bottom") else frame_h
            choices = _interval_choices(length)
            random.shuffle(choices)
            picked = None
            axis_key = "vertical" if side in ("top", "bottom") else "horizontal"
            peer = used_parallel.get(axis_key)
            for span in choices:
                if peer is not None and _gap_overlap(span, peer):
                    continue
                if peer is not None and abs((span[0] + span[1]) - (peer[0] + peer[1])) < 3:
                    continue
                picked = span
                break
            if picked is None:
                valid = False
                break
            openings.append({"side": side, "span": picked})
            used_parallel[axis_key] = picked
        if valid and len({op["side"] for op in openings}) >= 2:
            return openings
    return []


def _build_local_fence(frame_h: int, frame_w: int, openings: list[dict]) -> set[tuple[int, int]]:
    wall = _rect_border_points(frame_h, frame_w)
    for op in openings:
        a, b = op["span"]
        if op["side"] == "top":
            for c in range(a, b + 1):
                wall.discard((0, c))
        elif op["side"] == "bottom":
            for c in range(a, b + 1):
                wall.discard((frame_h - 1, c))
        elif op["side"] == "left":
            for r in range(a, b + 1):
                wall.discard((r, 0))
        else:
            for r in range(a, b + 1):
                wall.discard((r, frame_w - 1))

    bridge_indices = list(range(len(openings) - 1))
    random.shuffle(bridge_indices)
    for idx in bridge_indices:
        side = openings[idx]["side"]
        a, b = openings[idx]["span"]
        if side == "top":
            for c in range(a, b + 1):
                wall.add((1, c))
        elif side == "bottom":
            for c in range(a, b + 1):
                wall.add((frame_h - 2, c))
        elif side == "left":
            for r in range(a, b + 1):
                wall.add((r, 1))
        else:
            for r in range(a, b + 1):
                wall.add((r, frame_w - 2))
    return wall


def _place_anchor(frame_h: int, frame_w: int, grid_h: int, grid_w: int, openings: list[dict]) -> tuple[int, int] | None:
    top_margin = OPEN_OUTER_MARGIN if any(op["side"] == "top" for op in openings) else SIDE_MARGIN
    bottom_margin = OPEN_OUTER_MARGIN if any(op["side"] == "bottom" for op in openings) else SIDE_MARGIN
    left_margin = OPEN_OUTER_MARGIN if any(op["side"] == "left" for op in openings) else SIDE_MARGIN
    right_margin = OPEN_OUTER_MARGIN if any(op["side"] == "right" for op in openings) else SIDE_MARGIN

    r_lo = top_margin
    r_hi = grid_h - frame_h - bottom_margin
    c_lo = left_margin
    c_hi = grid_w - frame_w - right_margin
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _local_interior(frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, frame_h - 1) for c in range(1, frame_w - 1)]


def _seed_pool(frame_h: int, frame_w: int, wall: set[tuple[int, int]], openings: list[dict]) -> list[tuple[int, int]]:
    interior = _local_interior(frame_h, frame_w)
    opening_cells = []
    for op in openings:
        a, b = op["span"]
        if op["side"] == "top":
            opening_cells.extend((0, c) for c in range(a, b + 1))
        elif op["side"] == "bottom":
            opening_cells.extend((frame_h - 1, c) for c in range(a, b + 1))
        elif op["side"] == "left":
            opening_cells.extend((r, 0) for r in range(a, b + 1))
        else:
            opening_cells.extend((r, frame_w - 1) for r in range(a, b + 1))

    candidates: list[tuple[int, int, int]] = []
    for r, c in interior:
        if not any((rr, cc) in wall for rr, cc in neighbors8(r, c)):
            continue
        if any((r, c) == inner for op in openings for inner in op.get("inner", [])):
            continue
        dist = min(abs(r - orow) + abs(c - ocol) for orow, ocol in opening_cells)
        candidates.append((dist, r, c))

    if not candidates:
        return []
    best = max(dist for dist, _, _ in candidates)
    threshold = max(1, best - 1)
    return [(r, c) for dist, r, c in candidates if dist >= threshold]


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target:
                    break
    return blob


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_zone: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (dist > 1)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _border_hit_count(mask: np.ndarray) -> int:
    hits = 0
    if mask[0, :].any():
        hits += 1
    if mask[-1, :].any():
        hits += 1
    if mask[:, 0].any():
        hits += 1
    if mask[:, -1].any():
        hits += 1
    return hits


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H)
    grid_w = random.randint(*GRID_W)
    frame_h = random.randint(*FENCE_H)
    frame_w = random.randint(*FENCE_W)

    openings = _choose_openings(frame_h, frame_w)
    if len(openings) < 2:
        return None

    local_wall = _build_local_fence(frame_h, frame_w, openings)
    if not (WALL_PIXELS_MIN <= len(local_wall) <= WALL_PIXELS_MAX):
        return None
    local_mask = _component_mask(local_wall, (frame_h, frame_w))
    if len(components_mask(local_mask, connectivity=8)) != 1:
        return None

    anchor = _place_anchor(frame_h, frame_w, grid_h, grid_w, openings)
    if anchor is None:
        return None
    r0, c0 = anchor

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for r, c in local_wall:
        grid[r0 + r, c0 + c] = wall_color

    local_openings = []
    for op in openings:
        a, b = op["span"]
        side = op["side"]
        if side == "top":
            inner = [(1, c) for c in range(a, b + 1)]
        elif side == "bottom":
            inner = [(frame_h - 2, c) for c in range(a, b + 1)]
        elif side == "left":
            inner = [(r, 1) for r in range(a, b + 1)]
        else:
            inner = [(r, frame_w - 2) for r in range(a, b + 1)]
        local_openings.append({"side": side, "span": (a, b), "inner": inner})

    pool = _seed_pool(frame_h, frame_w, local_wall, local_openings)
    if not pool:
        return None

    interior = _local_interior(frame_h, frame_w)
    blob = _grow_seed_blob(random.choice(pool), set(interior))
    for r, c in blob:
        gr, gc = r0 + r, c0 + c
        if grid[gr, gc] != bg:
            return None
        grid[gr, gc] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)
    if int(nominal_fill_zone.sum()) < MIN_NEW_FILL:
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        base = _single_opening_baseline(grid)
    except Exception:
        return None

    info = analyze_primary_fence(grid)
    if info["opening_count"] < 2 or len(info["opening_sides"]) < 2:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    multi_bonus = (out == fill_color) & (base != fill_color)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if _border_hit_count(new_fill) < MIN_BORDER_HITS:
        return None
    if int(multi_bonus.sum()) < MIN_MULTI_BONUS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
