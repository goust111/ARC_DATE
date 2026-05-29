# Variant of 581f7754
# Dimension: D16 — 非背景像素碰撞处理 (transform)
# Change: 滑动目标位置的非背景像素不被覆盖（精灵像素跳过这些位置）... → 重叠区域变为特殊标记色

# Design analysis
# variant_spec:
#   core_change:
#     original: "多个精灵滑到同一落点时，先落下的非背景像素会阻止后来的像素写入。"
#     new: "多个已移动精灵像素若落在同一目标格，该格统一改成预定义冲突色 9。"
#   concrete_behavior:
#     input:
#       - 仍然沿用原题结构：边界锚点 + 内部多色精灵 + 可选的单像素迷惑项。
#       - 输入阶段不允许精灵互相接触或重叠，这样 solver 仍可稳定用连通域拆分对象。
#       - 冲突色 9 在输入里被保留，不作为背景、锚点色、主体色或 marker 色使用。
#     output:
#       - 每个精灵仍沿由锚点边决定的单轴滑动。
#       - 若两个或更多移动像素投影到同一格，该格不保留任何原色，而改成固定的冲突色 9。
#       - 锚点本身仍保留；本变体只强调“移动像素之间”的重叠，生成器也避免精灵撞到锚点。
#   trace_example:
#     - "原逻辑：A 的蓝像素先落在 (5,7)，B 的绿像素也要落在 (5,7)，因为该格已非背景，B 被跳过，输出保留蓝色。"
#     - "新逻辑：A 与 B 都声明自己要落在 (5,7)，合成阶段统计到该格有 2 个移动像素，因此 (5,7) 直接着成 9。"
#     - "差异不在位移量，而在所有位移完成后的落点归并。"
#   complexity_calibration:
#     - "每个样例强制至少出现 1 组碰撞，且碰撞面积为 2-5 格；否则视觉上会退化成原题。"
#     - "碰撞只安排为成对重叠，不制造三重覆盖，避免输出过乱。"
#     - "碰撞对共享同一个锚点颜色，这样人类能从样例中看到：同一目标轴线上有两块精灵滑入并留下 9 色冲突区。"
#   implicit_assumptions_review:
#     - "anchor 检测逻辑依赖：边界锚点仍是孤立单像素。生成器保持这一点。"
#     - "marker 识别逻辑依赖：输入连通域仍是原始精灵，不含碰撞色。生成器因此把碰撞只放在输出，不让输入重叠。"
#     - "原 main 的顺序写入依赖旧规则：先写入的非背景像素天然胜出。D16 下这会产生错误的顺序依赖，因此改成先收集所有目标像素、再按计数统一着色。"
#     - "单像素迷惑项仍按原题处理：如果它们会移动，它们也是“移动像素”，碰撞后同样会变成 9。"
#   placement_legality:
#     - "输入放置继续使用无接触、无重叠的约束，保证 component parsing 稳定。"
#     - "输出规划单独维护每个精灵的目标坐标集合；非碰撞精灵之间仍保持 8 邻域隔离。"
#     - "碰撞对允许彼此重叠，但禁止碰到其他精灵或锚点的 8 邻域。"
#   observability:
#     - "冲突色 9 为样例专用色，且每个生成样例都至少出现一次；因此解题者能直接看到新增现象。"
#     - "主体色和 marker 色仍保持少量、重复出现的颜色集合，不引入额外颜色映射规则。"
#   cleanliness:
#     - "控制为单对碰撞，且保留每个参与精灵至少 2 个未重叠像素，避免输出只剩一团无法辨认的块。"
#   new_main_assumptions:
#     - "main 假设需要移动的对象在输入中互不接触。generator 保证。"
#     - "main 允许任意数量像素落到同一格并统一转成 9；generator 只生成成对冲突，但 solver 不依赖这个限制。"

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


_CONFLICT_COLOR = 9


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, marking landing conflicts with color 9.

    Input structure:
    - The background is the modal color.
    - Interior connected sprites use one body color plus a rarer marker color; optional singleton sprites still exist.
    - Isolated non-background border pixels serve as anchors keyed by color.

    Rule:
    - Match each movable sprite to the anchor with the same marker color.
    - A top/bottom anchor makes the sprite slide horizontally to the anchor column; a left/right anchor makes it slide vertically to the anchor row.
    - When multiple moved pixels land on the same destination cell, that cell becomes conflict color 9 instead of keeping either original pixel.

    Key insight:
    - The anchor chooses the slide axis, but the final picture must be composed from all moved landing pixels at once, because overlap is now a counted event rather than a "first writer wins" overwrite rule.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def apply_landing_pixels(base: np.ndarray, landed: list[tuple[int, int, int]]) -> np.ndarray:
        out = base.copy()
        blocked = out != int(bg)
        counts: dict[tuple[int, int], list[int]] = {}
        for rr, cc, color in landed:
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if bool(blocked[rr, cc]):
                continue
            counts.setdefault((int(rr), int(cc)), []).append(int(color))
        for (rr, cc), vals in counts.items():
            out[rr, cc] = int(vals[0]) if len(vals) == 1 else int(_CONFLICT_COLOR)
        return out

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
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

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
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

    moves: list[tuple[np.ndarray, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
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

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    base = g.copy()
    for coords, _dr, _dc in moves:
        base[coords[:, 0], coords[:, 1]] = int(bg)

    landed: list[tuple[int, int, int]] = []
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            landed.append((int(rr), int(cc), int(g[int(r), int(c)])))

    return apply_landing_pixels(base, landed)


# --------------- generator ---------------

_COLORS = tuple(range(10))
_USABLE_COLORS = tuple(c for c in _COLORS if c != _CONFLICT_COLOR)
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 300

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

_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

_COLLISION_MIN_OVERLAP = 2
_COLLISION_MAX_OVERLAP = 5
_VISIBLE_REMAINDER_MIN = 2
_PAIR_BUILD_ATTEMPTS = 50

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
        removable = [(r, c) for r, c in cells if not (r in (0, h - 1) and c in (0, w - 1))]
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
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _sprite_pixels(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    coords = np.argwhere(patch != int(bg))
    return [
        (int(top + r), int(left + c), int(patch[int(r), int(c)]))
        for r, c in coords
    ]


def _sprite_cells(sprite, marker_r, marker_c, bg):
    return {(r, c) for r, c, _color in _sprite_pixels(sprite, marker_r, marker_c, bg)}


def _expand_cells(cells, shape):
    H, W = shape
    out = set()
    for r, c in cells:
        for dr, dc in _NEIGHBORS8:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


def _can_place(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
        return False
    if top + h > H - _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _touches_foreground(canvas, mask, top, left, bg):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


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


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions_free(shape, sprite, anchor):
    H, W = shape
    h, w = sprite["patch"].shape
    off_r, off_c = sprite["offset"]
    positions = []
    if anchor["axis"] == "h":
        col = int(anchor["c"])
        left = col - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            return []
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            top = int(r) - int(off_r)
            if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
                continue
            positions.append((int(r), int(col)))
        return positions

    row = int(anchor["r"])
    top = row - int(off_r)
    if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
        return []
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        left = int(c) - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            continue
        positions.append((int(row), int(c)))
    return positions


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg)
    ]


def _body_color(bg, marker, used):
    pool = [c for c in _USABLE_COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _USABLE_COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    palette = [c for c in _USABLE_COLORS if c != int(bg)]
    anchor_colors = random.sample(palette, random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    collision_color = random.choice(anchor_colors)
    rest = [random.choice(anchor_colors) for _ in range(max(0, n_sprites - 2))]
    random.shuffle(rest)
    return [int(collision_color), int(collision_color), *rest], int(collision_color)


def _anchor_keepout(anchors, shape):
    out = set()
    H, W = shape
    for anchor in anchors:
        for dr, dc in _NEIGHBORS8:
            rr = int(anchor["r"]) + int(dr)
            cc = int(anchor["c"]) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


def _make_item(sprite, anchor, marker_pos, bg):
    cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
    return {
        "sprite": sprite,
        "anchor": anchor,
        "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
        "cells": cells,
    }


def _occupied_cells(items):
    cells = set()
    for item in items:
        cells |= set(item["cells"])
    return cells


def _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg):
    occupied = _occupied_cells(items)
    positions = []
    for marker_pos in _aligned_positions_free(shape, sprite, anchor):
        cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
        if cells & anchor_keepout:
            continue
        if _expand_cells(cells, shape) & occupied:
            continue
        positions.append((int(marker_pos[0]), int(marker_pos[1])))
    return positions


def _colliding_positions(shape, sprite, anchor, target_item, other_items, anchor_keepout, bg):
    other_occupied = _occupied_cells(other_items)
    target_cells = set(target_item["cells"])
    positions = []
    for marker_pos in _aligned_positions_free(shape, sprite, anchor):
        cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
        if cells & anchor_keepout:
            continue
        overlap = cells & target_cells
        if not (_COLLISION_MIN_OVERLAP <= len(overlap) <= _COLLISION_MAX_OVERLAP):
            continue
        if len(cells - overlap) < _VISIBLE_REMAINDER_MIN:
            continue
        if len(target_cells - overlap) < _VISIBLE_REMAINDER_MIN:
            continue
        if _expand_cells(cells, shape) & other_occupied:
            continue
        positions.append((int(marker_pos[0]), int(marker_pos[1]), len(overlap)))
    return positions


def _render_output(shape, anchors, items, bg):
    H, W = shape
    base = np.full((H, W), int(bg), dtype=int)
    for anchor in anchors:
        base[int(anchor["r"]), int(anchor["c"])] = int(anchor["color"])

    blocked = base != int(bg)
    landed: dict[tuple[int, int], list[int]] = {}
    for item in items:
        for rr, cc, color in _sprite_pixels(item["sprite"], item["marker_pos"][0], item["marker_pos"][1], bg):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if bool(blocked[rr, cc]):
                continue
            landed.setdefault((int(rr), int(cc)), []).append(int(color))

    out = base.copy()
    for (rr, cc), vals in landed.items():
        out[rr, cc] = int(vals[0]) if len(vals) == 1 else int(_CONFLICT_COLOR)
    return out


def _place_collision_pair(shape, anchors, collision_anchor, bg, used_body):
    anchor_keepout = _anchor_keepout(anchors, shape)
    for _ in range(_PAIR_BUILD_ATTEMPTS):
        body_a = _body_color(bg, collision_anchor["color"], used_body)
        sprite_a = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_a, collision_anchor["color"], bg)
        positions_a = _noncolliding_positions(shape, sprite_a, collision_anchor, [], anchor_keepout, bg)
        if not positions_a:
            continue
        random.shuffle(positions_a)
        for marker_a in positions_a[: min(len(positions_a), 8)]:
            item_a = _make_item(sprite_a, collision_anchor, marker_a, bg)
            for _ in range(_PAIR_BUILD_ATTEMPTS):
                body_b = _body_color(bg, collision_anchor["color"], used_body | {body_a})
                sprite_b = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_b, collision_anchor["color"], bg)
                positions_b = _colliding_positions(shape, sprite_b, collision_anchor, item_a, [], anchor_keepout, bg)
                if not positions_b:
                    continue
                positions_b.sort(key=lambda x: (abs(x[2] - 3), random.random()))
                marker_b = random.choice(positions_b[: min(len(positions_b), 6)])
                item_b = _make_item(sprite_b, collision_anchor, (marker_b[0], marker_b[1]), bg)
                used_body.add(int(body_a))
                used_body.add(int(body_b))
                return [item_a, item_b]
    return None


def _place_remaining_sprites(shape, anchors, marker_colors, bg, used_body, items):
    anchor_keepout = _anchor_keepout(anchors, shape)
    by_color = {a["color"]: a for a in anchors}
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        placed = False
        for _ in range(_PAIR_BUILD_ATTEMPTS):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            positions = _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            items.append(_make_item(sprite, anchor, marker_pos, bg))
            used_body.add(int(body))
            placed = True
            break
        if not placed:
            return False
    return True


def _add_distractors(inp, anchors, bg, items, shape):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return True

    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    anchor_keepout = _anchor_keepout(anchors, shape)
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        out_positions = _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg)
        if not out_positions:
            return True
        marker_pos = random.choice(out_positions)
        candidates = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not candidates:
            return False
        in_pos = random.choice(candidates)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
        items.append(_make_item(sprite, anchor, marker_pos, bg))
    return True


def _place_aligned_sprites(shape, anchors, marker_colors, collision_color, bg):
    by_color = {a["color"]: a for a in anchors}
    collision_anchor = by_color[int(collision_color)]
    used_body = set()
    items = _place_collision_pair(shape, anchors, collision_anchor, bg, used_body)
    if items is None:
        return None
    remaining = marker_colors[2:]
    if not _place_remaining_sprites(shape, anchors, remaining, bg, used_body, items):
        return None
    return items


def _scatter_sprites(inp, items, bg):
    shuffled = list(items)
    random.shuffle(shuffled)
    for item in shuffled:
        positions = _scatter_positions(inp, item["sprite"], item["anchor"], item["marker_pos"], bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, item["sprite"], in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool((out == int(_CONFLICT_COLOR)).any()):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors, collision_color = _marker_color_sequence(anchor_colors)
    items = _place_aligned_sprites(out.shape, anchors, marker_colors, collision_color, bg)
    if items is None:
        return None

    if not _scatter_sprites(inp, items, bg):
        return None
    if not _add_distractors(inp, anchors, bg, items, out.shape):
        return None

    out = _render_output(out.shape, anchors, items, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D16 input")
