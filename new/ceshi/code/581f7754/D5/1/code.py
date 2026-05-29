# Variant of 581f7754
# Dimension: D5 — 变换：裁剪规则 (transform)
# Change: 滑动后超出网格边界的像素直接丢弃（裁剪）；目标位置的背景像素被覆盖，非背景像素不被覆盖... → 碰撞停止

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4

# Design analysis
# variant:
#   concrete_behavior:
#     - 保留原题的主体结构：内部多色精灵、每个精灵里有一个少数色 marker、边界上有同色孤立锚点。
#     - 保留原题的轴规则：上/下边锚点决定水平滑动，左/右边锚点决定垂直滑动。
#     - 新增静态障碍物：它们是内部单色组件，颜色不属于任何锚点色，因此不会被 solver 当成可移动精灵。
#     - 精灵不再一次性平移到锚点列/行再做裁剪；而是沿着唯一轴逐格试走，下一步只要会撞到非背景像素
#       或离开网格，就停在前一格。于是输出里会出现“有些精灵到锚点，有些精灵被挡在障碍物前”的混合样例。
#   behavior_trace:
#     - 小例子：顶边锚点在列 8，某精灵 marker 初始在 (5,2)，应向右移动 6 格。
#     - 原逻辑：直接把整个组件平移到 marker=(5,8)，若某些像素越界则裁剪，若目标格已有非背景则该像素不写入。
#     - 新逻辑：先试走到 marker=(5,3)，再到 (5,4)... 若从 (5,5) 走到 (5,6) 时组件前缘会撞上障碍，
#       则最终停在 (5,5)，不会继续接近锚点，也不会发生“末端覆盖/裁剪”。
#     - 差异集中在原 solver 的 STEP 5：旧版是“算出总位移后一次性 paste”，新版必须改成“逐步检测是否还能再走一步”。
#   complexity_calibration:
#     - 障碍物只引入少量、重复的类别：1-2 个颜色、1-3 个简单短条/小块，避免规则分支过多。
#     - 每题至少有 1 个被挡住的精灵；其余精灵通常仍能对齐锚点，这样解题者能从同一题里同时看到
#       “无遮挡时到锚点”和“有障碍时提前停下”两个分支。
#     - 障碍物颜色固定为非锚点色，避免把“颜色匹配”规则和“碰撞”规则混在一起。
#   legacy_assumptions_to_break:
#     - 原版把所有 mover 清空后，直接用 (dr, dc) 目标位移 paste；这隐含“路径中间发生什么都不重要”。
#       变体下该前提失效，必须在 paste 之前模拟中间每一步。
#     - 原版允许越界裁剪；变体下若下一步会越界，就应提前停止，所以 generator 也不再依赖越界裁剪制造样例。
#     - 原版目标格的非背景冲突只在最终 paste 时处理；变体下冲突需要在路径上提前阻断，所以必须用静态障碍图
#       在移动前就决定最大可达步数。
#   generator_legality:
#     - 输入中的 mover 彼此保持 8 邻域间隔，确保组件检测稳定。
#     - 障碍物颜色不属于 anchor 色；单像素障碍也不会被误判为 mover，因为只有“anchor 色单像素”才会移动。
#     - mover 的运动判定只依赖 anchors + 障碍物这张静态图；generator 同时保证 mover 的最终位置两两不重叠，
#       从而避免 solver 顺序 paste 时产生额外遮挡。
#   observability:
#     - 输出中常见模式是“marker 与同色锚点在同一列/行，但中间隔着一个短条障碍，精灵刚好贴在障碍前一格”。
#     - 若路径畅通，精灵仍会像原题那样与锚点对齐；两类样例并存，人类可以从 2-3 个样例归纳出‘障碍会截停’。
#   cleanliness:
#     - 障碍物数量受控，颜色少，形状简单，且输入中与 mover/anchor 留有空隙，避免组件粘连。
#     - 输出里允许 mover 与自己的障碍物正交相邻，这是碰撞结果本身；但 generator 仍避免与别的对象贴得过乱。
#   color_reasoning:
#     - 颜色只承担两种可见角色：anchor/marker 匹配色，和“任何非背景都能挡住”的障碍/身体色。
#     - 不使用颜色 id 排序驱动任何不可见逻辑；body/blocker 颜色只用于区分物体，不编码额外状态。
#   new_solver_assumptions:
#     - 每个可移动组件都能由最少色 marker 唯一配到某个 anchor 色。
#     - 静态障碍物是非 mover 组件；generator 通过颜色和形状保证这一点。
#     - 不同 mover 的最终落点互不重叠；generator 显式检查，避免 solver 的顺序写入影响语义。


_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_PLACEMENT_TRY_LIMIT = 160

_BLOCKED_SPRITE_CAP = 2

_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 7)
_FRAME_HEIGHT_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (3, 7)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.28
_WIDE_BODY_THRESHOLD = 0.55
_SYMMETRIC_BODY_THRESHOLD = 0.82
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.45
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.append((rr, cc))
    return frontier


def _grow_body(area):
    choices = [
        (h, w)
        for h in _inclusive_range(_BODY_HEIGHT_RANGE)
        for w in _inclusive_range(_BODY_WIDTH_RANGE)
        if h * w >= area
    ]
    h, w = random.choice(choices)
    cells = set(random.choice(_GROW_BODY_SEEDS))
    while len(cells) < area:
        frontier = _frontier_cells(cells, h, w)
        if not frontier:
            return _grow_body(area)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _symmetric_body(area):
    choices = []
    for height in _inclusive_range(_BODY_HEIGHT_RANGE):
        extra = area - height
        if extra >= 2 and extra % 2 == 0 and extra // 2 <= height:
            choices.append((height, extra // 2))
    if not choices:
        return _grow_body(area)

    height, pair_count = random.choice(choices)
    cells = {(r, 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((r, 0))
        cells.add((r, 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(c, r) for r, c in cells}
    return _flip(cells)


def _wide_body(area):
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, c) for c in range(width)}
    cells.add((1, random.randrange(width)))
    while len(cells) < area:
        frontier = _frontier_cells(cells, height, width)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _frame_body(area):
    choices = []
    for h in _inclusive_range(_FRAME_HEIGHT_RANGE):
        for w in _inclusive_range(_FRAME_WIDTH_RANGE):
            perim = 2 * h + 2 * w - 4
            removable = 2 * max(0, h - 2) + 2 * max(0, w - 2)
            if perim >= area and perim - area <= min(_FRAME_MAX_GAP, removable):
                choices.append((h, w))
    if not choices:
        return _grow_body(area)

    h, w = random.choice(choices)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    gap = len(cells) - area
    if gap:
        removable = [
            (r, c)
            for r, c in cells
            if not (r in (0, h - 1) and c in (0, w - 1))
        ]
        for cell in random.sample(removable, gap):
            cells.remove(cell)
    return _flip(cells)


def _make_body(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _marker_cell(body):
    r0, c0, r1, c1 = _bbox(body)
    interior, exterior = set(), set()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            if r0 <= p[0] <= r1 and c0 <= p[1] <= c1:
                interior.add(p)
            else:
                exterior.add(p)
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg):
    body = _make_body(area)
    marker = _marker_cell(body)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    cells = []
    for r, c in body:
        rr, cc = r - r0, c - c0
        patch[rr, cc] = int(body_color)
        cells.append((int(rr), int(cc), int(body_color)))
    mr, mc = marker[0] - r0, marker[1] - c0
    patch[mr, mc] = int(marker_color)
    cells.append((int(mr), int(mc), int(marker_color)))
    return {"patch": patch, "offset": (int(mr), int(mc)), "cells": cells}


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    S = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _detect_anchors_and_movers(g: np.ndarray, bg: int):
    H, W = g.shape
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), int(H), int(W)):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color
    anchor_cells = set(anchors_by_color.values())

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    movers = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        movers.append(
            {
                "coords": coords.astype(int),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "anchor": (int(ar), int(ac)),
                "horizontal": bool(horizontal),
                "desired_dr": int(dr),
                "desired_dc": int(dc),
            }
        )
    return anchors_by_color, movers


def _collision_stop_shift(coords: np.ndarray, desired_dr: int, desired_dc: int, static: np.ndarray, bg: int):
    H, W = static.shape
    step_r = 0 if int(desired_dr) == 0 else (1 if int(desired_dr) > 0 else -1)
    step_c = 0 if int(desired_dc) == 0 else (1 if int(desired_dc) > 0 else -1)
    steps = abs(int(desired_dr)) + abs(int(desired_dc))
    moved = 0
    current = [(int(r), int(c)) for r, c in coords]
    for _ in range(steps):
        nxt = []
        for r, c in current:
            rr = int(r + step_r)
            cc = int(c + step_c)
            if not (0 <= rr < H and 0 <= cc < W):
                return int(step_r * moved), int(step_c * moved)
            if int(static[rr, cc]) != int(bg):
                return int(step_r * moved), int(step_c * moved)
            nxt.append((int(rr), int(cc)))
        current = nxt
        moved += 1
    return int(step_r * moved), int(step_c * moved)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, but stop as soon as the next step is blocked.

    Input structure:
    - Background is the most common color.
    - Interior multi-color sprites each contain a least-frequent marker pixel whose color matches a border anchor.
    - Border anchors are isolated single pixels, and extra interior single-color obstacle pieces may sit in the slide lanes.

    Rule:
    - Match each movable sprite to the anchor with the same marker color.
    - Top/bottom-edge anchors make the sprite slide horizontally; left/right-edge anchors make it slide vertically.
    - Move one grid step at a time toward the anchor, clearing the sprite from its start first; if the next step would hit any non-background cell or leave the grid, the sprite stops immediately.

    Key insight: The anchor still chooses the axis, but the final resting place is no longer the anchor line by default; it is the farthest reachable position before the first collision.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _anchors, movers = _detect_anchors_and_movers(g, int(bg))

    static = g.copy()
    for mover in movers:
        coords = mover["coords"]
        static[coords[:, 0], coords[:, 1]] = int(bg)

    out = static.copy()
    H, W = out.shape
    for mover in movers:
        dr, dc = _collision_stop_shift(
            mover["coords"],
            int(mover["desired_dr"]),
            int(mover["desired_dc"]),
            static,
            int(bg),
        )
        for r, c in mover["coords"]:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])
    return out


def _sprite_abs_cells(sprite, marker_r: int, marker_c: int, bg: int):
    off_r, off_c = sprite["offset"]
    top = int(marker_r) - int(off_r)
    left = int(marker_c) - int(off_c)
    out = []
    for r, c, v in sprite["cells"]:
        if int(v) != int(bg):
            out.append((int(top + r), int(left + c), int(v)))
    return out


def _sprite_abs_coords(sprite, marker_r: int, marker_c: int, bg: int):
    return [(int(r), int(c)) for r, c, _v in _sprite_abs_cells(sprite, marker_r, marker_c, bg)]


def _cells_ok(canvas: np.ndarray, coords, bg: int, *, require_gap: bool, allowed_touch=frozenset()):
    H, W = canvas.shape
    allowed_touch = {(int(r), int(c)) for r, c in allowed_touch}
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if not (_INNER_BORDER_GAP <= int(r) < H - _INNER_BORDER_GAP):
            return False
        if not (_INNER_BORDER_GAP <= int(c) < W - _INNER_BORDER_GAP):
            return False
        if int(canvas[int(r), int(c)]) != int(bg):
            return False
    if not require_gap:
        return True
    for r, c in coord_set:
        for dr, dc in _NEIGHBORS8:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (int(rr), int(cc)) in coord_set or (int(rr), int(cc)) in allowed_touch:
                continue
            if int(canvas[rr, cc]) != int(bg):
                return False
    return True


def _can_place_sprite(canvas: np.ndarray, sprite, marker_r: int, marker_c: int, bg: int, *, require_gap: bool, allowed_touch=frozenset()):
    coords = _sprite_abs_coords(sprite, int(marker_r), int(marker_c), int(bg))
    return _cells_ok(canvas, coords, int(bg), require_gap=require_gap, allowed_touch=allowed_touch)


def _place_sprite(canvas: np.ndarray, sprite, marker_r: int, marker_c: int, bg: int):
    for r, c, v in _sprite_abs_cells(sprite, int(marker_r), int(marker_c), int(bg)):
        canvas[int(r), int(c)] = int(v)


def _paint_cells(canvas: np.ndarray, cells, color: int):
    for r, c in cells:
        canvas[int(r), int(c)] = int(color)


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _aligned_marker(anchor, marker_r: int, marker_c: int):
    if anchor["axis"] == "h":
        return int(marker_r), int(anchor["c"])
    return int(anchor["r"]), int(marker_c)


def _step_unit(anchor, marker_r: int, marker_c: int):
    target_r, target_c = _aligned_marker(anchor, int(marker_r), int(marker_c))
    dr = 0 if target_r == int(marker_r) else (1 if target_r > int(marker_r) else -1)
    dc = 0 if target_c == int(marker_c) else (1 if target_c > int(marker_c) else -1)
    return int(dr), int(dc)


def _simulate_marker_stop(sprite, marker_r: int, marker_c: int, anchor, static_canvas: np.ndarray, bg: int):
    H, W = static_canvas.shape
    target_r, target_c = _aligned_marker(anchor, int(marker_r), int(marker_c))
    step_r, step_c = _step_unit(anchor, int(marker_r), int(marker_c))
    total_steps = abs(int(target_r) - int(marker_r)) + abs(int(target_c) - int(marker_c))
    current = _sprite_abs_cells(sprite, int(marker_r), int(marker_c), int(bg))
    moved = 0
    for _ in range(total_steps):
        nxt = []
        for r, c, v in current:
            rr = int(r + step_r)
            cc = int(c + step_c)
            if not (0 <= rr < H and 0 <= cc < W):
                return {"marker": (int(marker_r + step_r * moved), int(marker_c + step_c * moved)), "steps": int(moved), "target_steps": int(total_steps), "blocked": True}
            if int(static_canvas[rr, cc]) != int(bg):
                return {"marker": (int(marker_r + step_r * moved), int(marker_c + step_c * moved)), "steps": int(moved), "target_steps": int(total_steps), "blocked": True}
            nxt.append((int(rr), int(cc), int(v)))
        current = nxt
        moved += 1
    return {"marker": (int(target_r), int(target_c)), "steps": int(moved), "target_steps": int(total_steps), "blocked": False}


def _marker_candidates(canvas: np.ndarray, anchor):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        out = [
            (r, c)
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != int(anchor["c"])
        ]
    else:
        out = [
            (r, c)
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if r != int(anchor["r"])
        ]
    random.shuffle(out)
    return out


def _blocker_shape_candidates(axis: str, impact: tuple[int, int]):
    r, c = impact
    if axis == "h":
        shapes = [
            {(r - 1, c), (r, c), (r + 1, c)},
            {(r - 1, c), (r, c)},
            {(r, c), (r + 1, c)},
            {(r, c)},
        ]
    else:
        shapes = [
            {(r, c - 1), (r, c), (r, c + 1)},
            {(r, c - 1), (r, c)},
            {(r, c), (r, c + 1)},
            {(r, c)},
        ]
    random.shuffle(shapes[:-1])
    return shapes


def _choose_blocker_cells(axis: str, impact: tuple[int, int], inp: np.ndarray, out: np.ndarray, bg: int):
    for cells in _blocker_shape_candidates(axis, impact):
        coords = [(int(r), int(c)) for r, c in cells]
        if _cells_ok(inp, coords, int(bg), require_gap=True) and _cells_ok(out, coords, int(bg), require_gap=True):
            return coords
    return None


def _body_color(bg, marker, blocker_color, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker), int(blocker_color)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker), int(blocker_color)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _place_clear_sprite(static_canvas, inp, out, sprite, anchor, bg):
    tried = 0
    for mr, mc in _marker_candidates(inp, anchor):
        if tried >= _PLACEMENT_TRY_LIMIT:
            break
        tried += 1
        if not _can_place_sprite(inp, sprite, mr, mc, bg, require_gap=True):
            continue
        sim = _simulate_marker_stop(sprite, mr, mc, anchor, static_canvas, bg)
        if sim["blocked"]:
            continue
        fr, fc = sim["marker"]
        if not _can_place_sprite(out, sprite, fr, fc, bg, require_gap=True):
            continue
        _place_sprite(inp, sprite, mr, mc, bg)
        _place_sprite(out, sprite, fr, fc, bg)
        return {"blocked": False, "input_marker": (mr, mc), "output_marker": (fr, fc)}
    return None


def _place_blocked_sprite(static_canvas, inp, out, sprite, anchor, blocker_color, bg):
    tried = 0
    for mr, mc in _marker_candidates(inp, anchor):
        if tried >= _PLACEMENT_TRY_LIMIT:
            break
        tried += 1
        sim_clear = _simulate_marker_stop(sprite, mr, mc, anchor, static_canvas, bg)
        if sim_clear["blocked"] or sim_clear["steps"] < 2:
            continue
        step_r, step_c = _step_unit(anchor, mr, mc)
        stop_steps = list(range(1, int(sim_clear["steps"])))
        random.shuffle(stop_steps)
        for steps in stop_steps:
            fr = int(mr + step_r * steps)
            fc = int(mc + step_c * steps)
            final_cells = _sprite_abs_cells(sprite, fr, fc, bg)
            next_cells = [(int(r + step_r), int(c + step_c)) for r, c, _v in final_cells]
            random.shuffle(next_cells)
            for impact in next_cells:
                blocker_cells = _choose_blocker_cells(anchor["axis"], impact, inp, out, bg)
                if blocker_cells is None:
                    continue
                temp_static = static_canvas.copy()
                _paint_cells(temp_static, blocker_cells, blocker_color)
                sim_blocked = _simulate_marker_stop(sprite, mr, mc, anchor, temp_static, bg)
                if not sim_blocked["blocked"] or tuple(sim_blocked["marker"]) != (fr, fc):
                    continue
                temp_inp = inp.copy()
                _paint_cells(temp_inp, blocker_cells, blocker_color)
                if not _can_place_sprite(temp_inp, sprite, mr, mc, bg, require_gap=True):
                    continue
                temp_out = out.copy()
                _paint_cells(temp_out, blocker_cells, blocker_color)
                if not _can_place_sprite(temp_out, sprite, fr, fc, bg, require_gap=True, allowed_touch=blocker_cells):
                    continue
                _paint_cells(static_canvas, blocker_cells, blocker_color)
                _paint_cells(inp, blocker_cells, blocker_color)
                _paint_cells(out, blocker_cells, blocker_color)
                _place_sprite(inp, sprite, mr, mc, bg)
                _place_sprite(out, sprite, fr, fc, bg)
                return {"blocked": True, "input_marker": (mr, mc), "output_marker": (fr, fc), "blocker_cells": blocker_cells}
    return None


def _new_canvases():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    static = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)
    return bg, static, inp, out


def _place_anchors(static, inp, out, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(static, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        static[int(r), int(c)] = int(color)
        inp[int(r), int(c)] = int(color)
        out[int(r), int(c)] = int(color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, static, inp, out = _new_canvases()
    placed = _place_anchors(static, inp, out, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    marker_colors = _marker_color_sequence(anchor_colors)
    blocker_pool = [c for c in _COLORS if c not in set(anchor_colors) | {int(bg)}]
    blocker_color = int(random.choice(blocker_pool))
    blocked_target = min(len(marker_colors), _BLOCKED_SPRITE_CAP)
    blocked_target = random.randint(1, blocked_target) if blocked_target > 0 else 0
    blocked_flags = [True] * blocked_target + [False] * (len(marker_colors) - blocked_target)
    random.shuffle(blocked_flags)

    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    placed_meta = []
    for marker, want_blocked in zip(marker_colors, blocked_flags):
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, blocker_color, used_body)
        used_body.add(int(body))
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        meta = None
        if want_blocked:
            meta = _place_blocked_sprite(static, inp, out, sprite, anchor, blocker_color, bg)
        if meta is None:
            meta = _place_clear_sprite(static, inp, out, sprite, anchor, bg)
        if meta is None:
            return None
        placed_meta.append(meta)

    if not any(m.get("blocked", False) for m in placed_meta):
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def _analyze_variant_behavior(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _anchors, movers = _detect_anchors_and_movers(g, bg)
    static = g.copy()
    for mover in movers:
        coords = mover["coords"]
        static[coords[:, 0], coords[:, 1]] = int(bg)
    blocked = 0
    for mover in movers:
        dr, dc = _collision_stop_shift(
            mover["coords"],
            int(mover["desired_dr"]),
            int(mover["desired_dc"]),
            static,
            int(bg),
        )
        if (dr, dc) != (int(mover["desired_dr"]), int(mover["desired_dc"])):
            blocked += 1
    return {"movers": len(movers), "blocked": int(blocked)}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D5 collision-stop input")
