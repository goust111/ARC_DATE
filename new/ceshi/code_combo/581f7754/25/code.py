# Combo variant of 581f7754 (index 25)
# combo_spec: D4/0+D11/0+D15/1
# Name: 角落锚点+静态诱饵+非孤立锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题把“孤立边界单像素”直接当作锚点，再按其所在边决定滑动轴；同色 interior 单像素也可能被当成可移动
#      的迷你精灵。
#    - 融合后，真正锚点固定放在角落，且角落旁边一定再放至少一个静态边界装饰像素，所以角落锚点不再满足
#      “4 邻域全背景”的旧判据。
#    - 输入里还会出现若干 interior 同色孤立单像素诱饵。它们颜色与某个锚点/marker 相同，但从 input 到 output
#      完全不动；真正移动的对象只有内部多色连通精灵。
#    - 具体小例子：左上角有颜色 4 锚点，(0,1) 放一个颜色 7 装饰。某个 marker=4 的精灵在输入里 marker 位于
#      (6,2)。列距 2 小于行距 6，因此新规则水平滑动，把 marker 移到 (6,0)。若仍按旧逻辑：
#        1) “孤立锚点检测”会漏掉角落 4，因为它与 (0,1) 相邻；
#        2) 即便强行找到角落，旧角落默认竖直滑动会把 marker 移到 (0,2)；
#        3) 同色 interior 单像素诱饵还会被误移动。
#    - 所以三项改动在推理链中是耦合的：solver 必须先忽略边界孤立性、再为角落按相对距离选轴、同时排除静态
#      singleton 诱饵。
# b) 复杂度校准
#    - 为了让 D4 可见，锚点全集收缩到四个角，而不是“偶尔允许角落”。这样每个样本都迫使观察者处理角落轴规则。
#    - 为了让 D15 可见，每个角落锚点都强制带一个正交相邻装饰像素，保证“非孤立”是稳定现象，不会退化回原题。
#    - 为了让 D11 可见，每个样本至少放 1 个静态同色 singleton；数量维持在 1-4，既有重复支撑，又不把画面塞满。
#    - 精灵数量仍为 3-5，颜色仍由 1-3 个锚点颜色重复分配，并强制样本里同时出现水平/竖直两种角落决策结果。
# c) 隐式假设审查
#    - 原 main() 第 1 步依赖“锚点 = 孤立边界像素”；融合后失效，因此新 solver 改为：
#      先从内部多色精灵提取 marker 颜色集合，再去边界上找这些颜色对应的唯一边界像素。
#    - 原逻辑允许单色 singleton 作为可移动对象；D11/0 使该分支失效，因此新 solver 彻底忽略 interior 单色组件。
#    - 原逻辑默认角落走竖直分支；D4/0 使该默认值失效，因此新 solver 对角落改为比较 marker 到角落的列距/行距。
#    - 原逻辑没有必要跳过触边组件，因为旧边界只有孤立锚点；D15/1 会产生多像素边界簇，所以新 solver 必须显式
#      跳过所有触边组件，避免把边界装饰误当作 sprite。
#    - 新增假设：每种 marker 颜色在边界上只有一个真正锚点像素。generator 通过“装饰颜色永不等于锚点颜色”来保证。
# d) 放置合法性
#    - 生成器仍采用 output-first：先在 output 上放角落锚点和静态边界装饰，再放已对齐精灵，随后沿反轴打散到 input。
#    - 输入端精灵必须完整位于内部并保持 8 邻域隔离；输出端允许因贴角而裁剪，但至少保留 marker 和若干 body 像素。
#    - 静态 singleton 诱饵在 input/output 同位放置，且与其他前景保持 8 邻域隔离，因此不会与可移动精灵连通。
# e) 视觉可观察性
#    - 人类可直接看到：真正决定运动的是角落上的同色边界像素，哪怕它旁边挤着别的颜色；而 interior 的同色小点不动。
#    - 同一批样本里会出现“有的角落精灵水平贴角，有的角落精灵竖直贴角”，从少量例子就能归纳“角落按更近方向选轴”。
#    - 整个规则只依赖可见结构：连通性、是否触边、相对行列距离；不依赖颜色编号大小或不可见排序。
# f) 可视化整洁度
#    - 每个角落只加 1 个必需装饰像素，必要时不再叠加更多噪声，避免边角区域变成难以辨认的大块团。
#    - 静态 singleton 控制在少量且彼此隔离，和多色精灵在形态上可区分，不会在同一区域产生多层覆盖。
# g) 变化的可推断性
#    - 没有颜色映射，颜色只表达“marker 对应哪个锚点”。
#    - 装饰色刻意避开锚点色，避免出现“同色多个边界候选点”导致规则不可归纳。
#    - 变化来自位置与连通性：多色精灵按角落轴规则移动，孤立同色单像素不动，边界装饰也不动。


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
_GEN_RETRY_LIMIT = 600

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

_STATIC_DISTRACTOR_COUNT_RANGE = (1, 4)

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _is_border(r: int, c: int, shape: tuple[int, int]) -> bool:
    H, W = shape
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _is_corner(r: int, c: int, shape: tuple[int, int]) -> bool:
    H, W = shape
    return int(r) in (0, int(H - 1)) and int(c) in (0, int(W - 1))


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return any(_is_border(int(r), int(c), shape) for r, c in coords)


def _component_marker_color(grid: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(grid[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(grid: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    cell_set = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in cell_set and int(grid[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _slide_axis(anchor_r: int, anchor_c: int, marker_r: int, marker_c: int, shape: tuple[int, int], *, corner_mode: str) -> str:
    H, W = shape
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


def _detect_anchors_from_border_colors(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = grid.shape
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    candidate_colors: set[int] = set()
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or _touches_border(coords, grid.shape):
            continue
        marker = _component_marker_color(grid, coords)
        if marker is not None:
            candidate_colors.add(int(marker))

    border_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(r, c, grid.shape):
                continue
            v = int(grid[r, c])
            if v == int(bg):
                continue
            border_by_color.setdefault(int(v), []).append((int(r), int(c)))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for color in sorted(candidate_colors):
        pts = border_by_color.get(int(color), [])
        if pts:
            anchors_by_color[int(color)] = pts[0]
    return anchors_by_color


def _solve_variant(grid: np.ndarray, *, corner_mode: str, move_singletons: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors_from_border_colors(g, bg)
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or _touches_border(coords, g.shape):
            continue

        marker = _component_marker_color(g, coords)
        if marker is None:
            if not move_singletons or int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            color = int(g[r, c])
            if color == int(bg) or color not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker = int(color)

        if int(marker) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker)]
        axis = _slide_axis(int(ar), int(ac), int(mp[0]), int(mp[1]), (H, W), corner_mode=corner_mode)
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


def _solve_originalish(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(r, c, g.shape) or _is_corner(r, c, g.shape):
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

    if not anchors_by_color:
        return g.copy()

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker = _component_marker_color(g, coords)
        if marker is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            color = int(g[r, c])
            if color == int(bg) or color not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker = int(color)

        if int(marker) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
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
    Move only the interior multi-color sprites so their marker pixels align with matching corner anchors hidden inside border clutter, while same-color singleton lures stay fixed.

    Input structure:
    - The background is the most common color.
    - True anchors are border pixels whose colors match sprite markers; in this combo they are placed at corners and may touch adjacent non-background border clutter.
    - The interior contains scattered multi-color sprites plus isolated same-color singleton decoys that do not participate in the motion.

    Rule:
    - Read each movable sprite's least-frequent color as its marker and pair it with the border pixel of the same color.
    - Non-corner border anchors would still use the usual edge-based axis, but a corner anchor chooses per sprite: if the marker is closer to the corner by column than by row, slide horizontally to the corner column; otherwise slide vertically to the corner row.
    - Ignore interior singleton lures and all border clutter; after moving sprites, clip pixels that leave the grid and never overwrite existing non-background cells.

    Key insight:
    - Anchor detection now depends on border position plus marker-color matching, not on isolation, and corner anchors are resolved by comparing row-vs-column distance from each sprite's marker.
    """
    return _solve_variant(np.array(grid, dtype=int), corner_mode="proximity", move_singletons=False)


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
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
                if (rr, cc) not in coords and int(canvas[rr, cc]) != int(bg):
                    return False
    return True


def _place_clipped(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), clip=True)


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    out = []
    for r, c in corners:
        if int(canvas[r, c]) == int(bg):
            out.append((int(r), int(c)))
    random.shuffle(out)
    return out


def _put_pixel(canvas, r, c, color):
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
    for color, (r, c) in zip(anchor_colors, candidates[:n]):
        anchor = {"color": int(color), "r": int(r), "c": int(c)}
        anchors.append(anchor)
        _put_pixel(out, r, c, color)
        _put_pixel(inp, r, c, color)
    return anchor_colors, anchors


def _corner_neighbor_choices(anchor, shape):
    H, W = shape
    r = int(anchor["r"])
    c = int(anchor["c"])
    choices = []
    if r == 0:
        choices.append((1, c))
    else:
        choices.append((H - 2, c))
    if c == 0:
        choices.append((r, 1))
    else:
        choices.append((r, W - 2))
    return choices


def _decorate_anchors(out, inp, anchors, bg):
    used_colors = {int(a["color"]) for a in anchors}
    decor_pool = [c for c in _COLORS if c != int(bg) and c not in used_colors]
    if len(decor_pool) < len(anchors):
        return False

    random.shuffle(decor_pool)
    for anchor, color in zip(anchors, decor_pool):
        choices = [p for p in _corner_neighbor_choices(anchor, out.shape) if int(out[p]) == int(bg)]
        if not choices:
            return False
        rr, cc = random.choice(choices)
        _put_pixel(out, rr, cc, color)
        _put_pixel(inp, rr, cc, color)
    return True


def _sprite_plans(anchor_colors, anchors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)

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


def _add_static_distractors(out, inp, anchor_colors, bg):
    count = random.randint(*_STATIC_DISTRACTOR_COUNT_RANGE)
    placed = []
    for _ in range(count):
        sprite = _singleton_sprite(random.choice(anchor_colors))
        candidates = []
        for r in range(_INNER_BORDER_GAP, out.shape[0] - _INNER_BORDER_GAP):
            for c in range(_INNER_BORDER_GAP, out.shape[1] - _INNER_BORDER_GAP):
                if _can_place(out, sprite, r, c, bg) and _can_place(inp, sprite, r, c, bg):
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = random.choice(candidates)
        _place(out, sprite, r, c, bg)
        _place(inp, sprite, r, c, bg)
        placed.append((int(r), int(c), int(sprite["patch"][0, 0])))
    return placed


def _count_static_anchor_singletons(inp, out, anchor_colors, bg):
    count = 0
    for comp in find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict"):
        coords = comp["coords"]
        if int(coords.shape[0]) != 1:
            continue
        r, c = map(int, coords[0])
        if _touches_border(coords, inp.shape):
            continue
        color = int(inp[r, c])
        if color in anchor_colors and int(out[r, c]) == color:
            count += 1
    return count


def _nonisolated_corner_anchor_count(grid, anchor_colors):
    H, W = grid.shape
    found = 0
    for color in anchor_colors:
        pts = [(int(r), int(c)) for r in range(H) for c in range(W) if int(grid[r, c]) == int(color) and _is_border(r, c, grid.shape)]
        if len(pts) != 1:
            return -1
        r, c = pts[0]
        if not _is_corner(r, c, grid.shape):
            return -1
        if any(0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(get_bg(grid)) for rr, cc in neighbors4(r, c)):
            found += 1
    return found


def _generated_grid_is_valid(inp, out, bg, anchor_colors):
    if not _bg_is_mode(inp, bg):
        return False
    if _count_static_anchor_singletons(inp, out, set(map(int, anchor_colors)), bg) < 1:
        return False
    if _nonisolated_corner_anchor_count(inp, anchor_colors) != len(anchor_colors):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    if bool(np.array_equal(_solve_variant(inp, corner_mode="legacy", move_singletons=False), out)):
        return False
    if bool(np.array_equal(_solve_variant(inp, corner_mode="proximity", move_singletons=True), out)):
        return False
    if bool(np.array_equal(_solve_originalish(inp), out)):
        return False
    return True


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    if not _decorate_anchors(out, inp, anchors, bg):
        return None

    plans = _sprite_plans(anchor_colors, anchors)
    placed_sprites = _place_aligned_sprites(out, plans, bg)
    if placed_sprites is None:
        return None
    if not _scatter_sprites(inp, placed_sprites, bg):
        return None

    distractors = _add_static_distractors(out, inp, anchor_colors, bg)
    if not distractors:
        return None
    if not _generated_grid_is_valid(inp, out, bg, anchor_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
