# Variant of 581f7754
# Dimension: D4 — 目标位置：锚点类型与位置 (dst)
# Change: 边界上的孤立单像素（非角落），4-邻域均为背景色，1-3个不同颜色锚点... → 角落锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 变体后的锚点只放在四个角上，仍然是单像素、颜色互异、与周围背景隔开。
#    - 精灵仍是“主体色 + 1 个标记色”的 8 连通小图形；输入里全部放在内部，输出里会沿单轴滑动。
#    - 原题对角落的未显式情况会落入“左/右边 => 竖直滑动”的默认分支；例如左上角锚点 (0,0) 与标记
#      在 (6,3) 的精灵，旧逻辑只会把它移到第 0 行同一列，得到 (0,3)。
#    - 本变体中，角落锚点先比较输入标记到该角落的列距和行距：若列距更小，就水平滑到角落所在列；
#      否则竖直滑到角落所在行。上例里列距 3 < 行距 6，所以新逻辑应把标记移到 (6,0)。
#    - 这只改动“目标轴如何由锚点决定”这一环；组件检测、按颜色配对、清空原位、裁剪和不覆盖非背景等
#      后续流程保持原样。
# b) 复杂度校准
#    - 若仍大量生成非角落锚点，样本可能退化回原题，所以这里将锚点全集收缩为角落，确保 D4 可见。
#    - 角落规则只有两档：水平或竖直；并且每个样本尽量同时出现两档，避免观察者把题误判成“永远贴左/右”
#      或“永远贴上/下”。
# c) 隐式假设审查
#    - 原 main() 中 `horizontal = on_top_bottom and not on_left_right` 依赖“角落不存在”；一旦角落出现，
#      它会强制落入竖直分支，和新规则冲突。
#    - 原生成器 `_anchor_ok()` 明确拒绝角落，`_anchor_candidates()` 只枚举非角落边界，二者都必须改。
#    - 原 `_aligned_positions()` 默认标记最终仍在内部列/行；角落锚点会把标记推到边界列/行，因此输出放置
#      需要允许贴边与裁剪，但输入散布仍必须保持完整内置。
# d) 放置合法性
#    - 输入端仍要求所有精灵完整位于内部，并与其它对象保持 8 邻域隔离。
#    - 输出端允许裁剪；但为了画面整洁，生成器要求至少保留可见标记和若干主体像素，并禁止与其它精灵/
#      锚点发生 8 邻域接触。
#    - 角落规则使用严格不等式比较距离，因此生成器避免制造“行距 = 列距”的平局样本。
# e) 视觉可观察性
#    - 每个样本只使用 1-3 个角落锚点，颜色直接匹配标记色，不依赖颜色 id 排序。
#    - 当某个角落颜色对应多个精灵时，优先让它们分摊到水平/竖直两种结果，便于从少量样本中看出
#      “同一个角落会按当前位置选轴”。
# f) 可视化整洁度
#    - 输出端虽然允许贴边，但仍保持对象之间的空隙；裁剪量被限制，避免只剩零散碎片。
#    - 角落附近只保留锚点本身；精灵不会直接压到锚点相邻的 8 邻域里。
# g) 颜色变化可推断性
#    - 颜色只承担“标记色匹配角落锚点”的作用；没有颜色替换或依赖颜色数值大小的逻辑。
# 新增假设审查
#    - solver 对角落分支新增了“比较输入标记到角落的行距和列距”的假设；generator 用严格不等式构造
#      输入，保证该比较总是唯一决定滑动轴。
#    - solver 仍假设每种锚点颜色至多取第一个孤立边界像素；generator 因而维持锚点颜色互异。


def _solve(grid: np.ndarray, *, corner_mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def slide_axis(anchor_r: int, anchor_c: int, marker_r: int, marker_c: int) -> str:
        on_top_bottom = int(anchor_r) in (0, int(H - 1))
        on_left_right = int(anchor_c) in (0, int(W - 1))
        if on_top_bottom and on_left_right:
            if corner_mode == "legacy":
                return "v"
            row_gap = abs(int(anchor_r) - int(marker_r))
            col_gap = abs(int(anchor_c) - int(marker_c))
            return "h" if int(col_gap) < int(row_gap) else "v"
        if on_top_bottom:
            return "h"
        return "v"

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

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
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
        cell_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cell_set and int(g[int(rr), int(cc)]) != int(marker):
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
        axis = slide_axis(int(ar), int(ac), int(mp[0]), int(mp[1]))
        dr = 0
        dc = 0
        if axis == "h":
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move interior sprites toward matching border anchors, including corner anchors.

    Input structure:
    - The background is the dominant color on the grid.
    - Interior foreground components are small sprites whose rarest color is a one-pixel marker.
    - Matching anchor colors appear as isolated border singletons; in this variant the anchors are allowed at corners.

    Rule:
    - Pair every sprite or eligible singleton with the border anchor of the same color.
    - Non-corner anchors keep the old behavior: top/bottom anchors force a horizontal slide, left/right anchors force a vertical slide.
    - Corner anchors choose per sprite: compare the marker's column distance and row distance to that corner in the input; smaller column distance means slide horizontally to the corner column, otherwise slide vertically to the corner row. Pixels that leave the grid are clipped.

    Key insight:
    - A corner anchor no longer hard-codes one axis; the sprite's current position relative to that corner decides whether matching the corner's row or matching its column is the shorter move.
    """
    return _solve(np.array(grid, dtype=int), corner_mode="proximity")


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
_INNER_BORDER_GAP = 1
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400

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
_MARKER_INTERIOR_PROB = 0.2
_FLIP_PROB = 0.5

_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

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
    if exterior and random.random() >= _MARKER_INTERIOR_PROB:
        return random.choice(list(exterior))
    pool = list(interior if interior else (exterior | interior))
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
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "body_cells": int(len(body)),
    }


def _singleton_sprite(color):
    return {
        "patch": np.array([[int(color)]], dtype=int),
        "offset": (0, 0),
        "body_cells": 0,
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


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


def _visible_cells(sprite, marker_r, marker_c, shape, bg):
    H, W = shape
    patch = sprite["patch"]
    off_r, off_c = sprite["offset"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            v = int(patch[pr, pc])
            if v == int(bg):
                continue
            rr = int(top + pr)
            cc = int(left + pc)
            if 0 <= rr < H and 0 <= cc < W:
                cells.append((rr, cc, v, pr == int(off_r) and pc == int(off_c)))
    return cells


def _can_place_clipped(canvas, sprite, marker_r, marker_c, bg, *, min_body_visible):
    cells = _visible_cells(sprite, marker_r, marker_c, canvas.shape, bg)
    if not cells:
        return False
    coords = {(int(r), int(c)) for r, c, _v, _m in cells}
    if any(int(canvas[r, c]) != int(bg) for r, c in coords):
        return False
    body_visible = sum(0 if is_marker else 1 for _r, _c, _v, is_marker in cells)
    if int(body_visible) < int(min_body_visible):
        return False
    for r, c in coords:
        for dr, dc in _NEIGHBORS8:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
                if (rr, cc) not in coords and int(canvas[rr, cc]) != int(bg):
                    return False
    return True


def _place_clipped(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), clip=True)


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) and c in (0, W - 1)):
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
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    out = []
    for r, c in corners:
        if _anchor_ok(canvas, r, c, bg):
            out.append(("corner", int(r), int(c)))
    random.shuffle(out)
    return out


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, slide_axis, bg, *, min_body_visible):
    H, W = canvas.shape
    out = []
    if slide_axis == "h":
        rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
        random.shuffle(rows)
        for r in rows:
            if abs(int(anchor["r"]) - int(r)) < 2:
                continue
            if _can_place_clipped(canvas, sprite, r, int(anchor["c"]), bg, min_body_visible=min_body_visible):
                out.append((int(r), int(anchor["c"])))
        return out

    cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    random.shuffle(cols)
    for c in cols:
        if abs(int(anchor["c"]) - int(c)) < 2:
            continue
        if _can_place_clipped(canvas, sprite, int(anchor["r"]), c, bg, min_body_visible=min_body_visible):
            out.append((int(anchor["r"]), int(c)))
    return out


def _scatter_positions(canvas, sprite, anchor, marker, slide_axis, bg):
    H, W = canvas.shape
    mr, mc = map(int, marker)
    out = []
    if slide_axis == "h":
        row_gap = abs(int(anchor["r"]) - int(mr))
        cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
        random.shuffle(cols)
        for c in cols:
            if abs(int(anchor["c"]) - int(c)) >= int(row_gap):
                continue
            if _can_place(canvas, sprite, mr, c, bg):
                out.append((int(mr), int(c)))
        return out

    col_gap = abs(int(anchor["c"]) - int(mc))
    rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    random.shuffle(rows)
    for r in rows:
        if abs(int(anchor["r"]) - int(r)) >= int(col_gap):
            continue
        if _can_place(canvas, sprite, r, mc, bg):
            out.append((int(r), int(mc)))
    return out


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
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
    candidates = _anchor_candidates(out, bg)
    n = random.randint(*_ANCHOR_COUNT_RANGE)
    if len(candidates) < n:
        return None
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], n)
    anchors = []
    for color, (_kind, r, c) in zip(anchor_colors, candidates[:n]):
        anchor = {"color": int(color), "r": int(r), "c": int(c)}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _sprite_plans(anchor_colors, anchors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    counts = Counter(map(int, marker_colors))
    by_color = {int(anchor["color"]): anchor for anchor in anchors}
    plans = []
    for color in anchor_colors:
        axes = []
        if counts[int(color)] >= 2:
            axes.extend(["h", "v"])
        while len(axes) < counts[int(color)]:
            axes.append(random.choice(("h", "v")))
        random.shuffle(axes)
        for axis in axes:
            plans.append({"marker": int(color), "anchor": by_color[int(color)], "slide_axis": axis})
    random.shuffle(plans)
    if len(plans) >= 2 and len({plan["slide_axis"] for plan in plans}) == 1:
        plans[0]["slide_axis"] = "h"
        plans[1]["slide_axis"] = "v"
    return plans


def _place_aligned_sprites(out, plans, bg):
    used_body = set()
    placed = []
    order = plans[:]
    random.shuffle(order)
    for plan in order:
        marker = int(plan["marker"])
        anchor = plan["anchor"]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(
            out,
            sprite,
            anchor,
            plan["slide_axis"],
            bg,
            min_body_visible=min(3, int(sprite["body_cells"])),
        )
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place_clipped(out, sprite, marker_pos[0], marker_pos[1], bg)
        placed.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "slide_axis": plan["slide_axis"],
                "marker_pos": marker_pos,
            }
        )
    return placed


def _scatter_sprites(inp, placed, bg):
    order = placed[:]
    random.shuffle(order)
    for item in order:
        positions = _scatter_positions(
            inp,
            item["sprite"],
            item["anchor"],
            item["marker_pos"],
            item["slide_axis"],
            bg,
        )
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, item["sprite"], in_pos[0], in_pos[1], bg)
    return True


def _add_distractors(out, inp, anchors, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        anchor = random.choice(anchors)
        slide_axis = random.choice(("h", "v"))
        sprite = _singleton_sprite(anchor["color"])
        out_positions = _aligned_positions(out, sprite, anchor, slide_axis, bg, min_body_visible=0)
        if not out_positions:
            return
        marker_pos = random.choice(out_positions)
        in_positions = _scatter_positions(inp, sprite, anchor, marker_pos, slide_axis, bg)
        if not in_positions:
            return
        in_pos = random.choice(in_positions)
        _place_clipped(out, sprite, marker_pos[0], marker_pos[1], bg)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    return not bool(np.array_equal(_solve(inp, corner_mode="legacy"), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    plans = _sprite_plans(anchor_colors, anchors)
    sprites = _place_aligned_sprites(out, plans, bg)
    if sprites is None:
        return None
    if not _scatter_sprites(inp, sprites, bg):
        return None
    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D4 input")
