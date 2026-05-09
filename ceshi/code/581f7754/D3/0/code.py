# Variant of 581f7754
# Dimension: D3 — 源对象：精灵结构 (src)
# Change: 多色精灵（8-连通分量），包含1个主体色（4-13像素）和1个标记色（1像素，最少数色），标记与主体4-相邻... → 多标记像素精灵

# Design analysis
# yaml:
#   source_task: none
#   baseline_summary:
#     solver:
#       - 原题把每个内部多色精灵视为一个 8 连通分量。
#       - 最少数颜色是 marker 色；旧代码默认真正参与对齐的是一个 marker 像素。
#       - 锚点是边界上的孤立单像素；顶/底锚点决定水平滑动，左/右锚点决定垂直滑动。
#     generator:
#       - output-first：先把 sprite 放到输出中的“已对齐”位置，再沿反向轴散开得到输入。
#       - sprite 的几何锚点由单个 marker 像素定义，`offset` 和 `_top_left` 都依赖这个前提。
#   variant_goal:
#     - 每个 sprite 现在有 2-3 个 marker 像素，marker 色仍然是组件中的少数色。
#     - 同一 sprite 的 marker 必须共线：要么同一行，要么同一列。
#     - 真正用于对齐的不是任意一个 marker，而是整条 marker 线的有效位置。
#     - 这里选用“中位位置”作为有效位置；为了保证平移仍是整数，生成器只产生中位位置为整数的线型。
#   concrete_pixel_behavior:
#     - 顶/底锚点对应水平平移，因此 generator 让 marker 排成同一行、列坐标发生变化。
#       例如 marker 在 (r, c-1) 和 (r, c+1)，有效列是 c；或者 marker 在 (r, c-1),(r, c),(r, c+1)，有效列也是 c。
#     - 左/右锚点对应垂直平移，因此 generator 让 marker 排成同一列、行坐标发生变化。
#       例如 marker 在 (r-1, c) 和 (r+1, c)，有效行是 r；或者 marker 在 (r-1, c),(r, c),(r+1, c)，有效行也是 r。
#   behavior_trace:
#     original_small_case:
#       input:
#         - 锚点在底边第 8 列。
#         - sprite 内部 marker 只有一个，位于第 4 列。
#       old_solver:
#         - 直接令 dc = 8 - 4，把整块水平平移 4。
#     variant_small_case:
#       input:
#         - 锚点仍在底边第 8 列。
#         - sprite 有两个 marker，位于第 4 列和第 6 列，同一行。
#       new_solver:
#         - 先识别两个 marker 的中位列是 5，再令 dc = 8 - 5 = 3。
#       difference:
#         - 旧逻辑若错误地取左侧 marker，会平移 4；取右侧 marker，会平移 2。
#         - 新逻辑只允许按整条 marker 线的中心定义位移。
#   complexity_calibration:
#     - 保留原题的 1-3 个锚点、3-5 个 sprite、4-13 个主体像素，避免把任务改成完全不同的布局题。
#     - 每个样例里 marker 数量只在 2 和 3 之间变化，且多个 sprite 会重复这些情况，便于从少量例子中归纳。
#     - 为了让 D3 变化可观察，marker 线总是沿“被对齐的轴”展开，而不是沿垂直轴展开；否则多 marker 会退化得像单 marker。
#   implicit_assumption_audit:
#     - `component_marker_color(coords)` 依赖“marker 色是最少数色”，这个前提仍成立；generator 保证 body 面积 >= 4 且 marker 数 <= 3。
#     - 旧版 `marker_pixel(coords, marker)` 依赖“有效对齐点 = 一个真实 marker 像素”，这个前提失效，必须替换成整组 marker 的聚合坐标。
#     - 旧版 `_make_sprite(...)->offset` 依赖 patch 内存在唯一 marker anchor；新版本改为 `align_offset`，它可以落在 marker 中点上而不一定是着色像素。
#     - 旧版 `_aligned_positions` / `_scatter_positions` 只要把“marker 坐标”理解成“有效对齐点坐标”就仍然正确，因此逻辑结构可以保留。
#     - 旧版可选 singleton distractor 会重新引入“单像素对象也会移动”的旧观感，削弱 D3 的可观察性；为保持视觉聚焦，这里移除 distractor 分支。
#   placement_legality:
#     - generator 只在 body 的外露边上放 marker 线，保证每个 marker 都与 body 4 邻接，从而整个 sprite 仍是一个 8 连通分量。
#     - 二点 marker 只取对称间隔 2 的模式，三点 marker 只取连续三点模式，因此中位位置始终是整数。
#     - sprite 的有效对齐点用于 `_top_left`、`_can_place`、`_place`，因此 output-first 的逆向散开流程不需要额外几何特判。
#   visual_observability:
#     - 二点模式会在两个 marker 中间留下一个背景格，三点模式会有明显的“中间 marker”；两者都让“不是按某一个 marker 对齐”肉眼可见。
#     - 锚点颜色直接等于 marker 颜色，没有隐藏颜色映射，也不依赖颜色 id 排序。
#   cleanliness:
#     - 保持对象之间 8 邻域隔离和离边界至少 1 格的要求，避免多条 marker 线互相贴住造成读图混乱。
#     - 不增加第三种功能色，也不引入区域覆盖关系，输出仍是清晰的平移结果。
#   post_change_main_assumptions:
#     - solver 假设每个真正参与移动的 multicolor sprite 的少数色像素都在同一行或同一列；generator 明确保证。
#     - solver 假设该少数色确实对应某个锚点颜色；generator 仍按原题方式保证。
#     - solver 允许输入里偶然出现单 marker 或奇怪布局时退化到坐标中位数近似，但正式生成样本不会依赖这种 fallback。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-marker sprites so that each marker line centers on its matching isolated border anchor.

    Input structure:
    - The background is the most frequent color.
    - Interior sprites are 8-connected multi-color components with one body color and 2-3 minority-color marker pixels.
    - The marker pixels of a sprite lie on one shared row or one shared column; isolated border pixels of the same color are anchors.

    Rule:
    - Match each sprite to the border anchor sharing its minority color.
    - A top/bottom-edge anchor means horizontal motion, using the median marker column as the alignment column.
    - A left/right-edge anchor means vertical motion, using the median marker row as the alignment row.

    Key insight: The moved feature is no longer a single special pixel; you must identify the whole marker run, collapse it to one effective coordinate, and translate along the axis chosen by the anchor edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def median_int(values: list[int]) -> int:
        values = sorted(int(v) for v in values)
        n = len(values)
        if n == 0:
            raise ValueError("empty median")
        if n % 2 == 1:
            return int(values[n // 2])
        total = int(values[n // 2 - 1] + values[n // 2])
        if total % 2 == 0:
            return int(total // 2)
        return int((total + 1) // 2)

    # --- STEP 1) Detect anchors: isolated border pixels (non-bg) ---
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

    # --- STEP 2) Extract all foreground connected components as candidate sprites. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_alignment_point(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if not pts:
            return None
        rows = sorted({int(r) for r, _c in pts})
        cols = sorted({int(c) for _r, c in pts})
        if len(rows) == 1:
            return int(rows[0]), int(median_int([c for _r, c in pts]))
        if len(cols) == 1:
            return int(median_int([r for r, _c in pts])), int(cols[0])
        return int(median_int([r for r, _c in pts])), int(median_int([c for _r, c in pts]))

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        # --- STEP 3) Decide marker color for this component. ---
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

        mp = marker_alignment_point(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]

        # --- STEP 4) Compute translation (dr,dc). ---
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

    out = g.copy()

    # --- STEP 5) Apply all moves. ---
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (o5x) ---------------

# CFG: palette and canvas bounds.
_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

# CFG: placement isolation and anchor location.
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500
_SPRITE_RETRY_LIMIT = 50

# CFG: body geometry and shape mix.
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
_FLIP_PROB = 0.5

# CFG: marker-line generation.
_PAIR_MARKER_PROB = 0.4

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


def _marker_line_options(body, axis):
    groups = {}
    if axis == "h":
        for r, c in body:
            for rr in (r - 1, r + 1):
                if (rr, c) not in body:
                    groups.setdefault(int(rr), set()).add(int(c))
    else:
        for r, c in body:
            for cc in (c - 1, c + 1):
                if (r, cc) not in body:
                    groups.setdefault(int(cc), set()).add(int(r))

    options = []
    seen = set()
    for fixed, vars_ in groups.items():
        vals = sorted(int(v) for v in vars_)
        valset = set(vals)
        for center in vals:
            if {center - 1, center, center + 1}.issubset(valset):
                if axis == "h":
                    cells = ((fixed, center - 1), (fixed, center), (fixed, center + 1))
                    align = (fixed, center)
                else:
                    cells = ((center - 1, fixed), (center, fixed), (center + 1, fixed))
                    align = (center, fixed)
                if cells not in seen:
                    seen.add(cells)
                    options.append({"cells": cells, "align": align, "count": 3})
            if {center - 1, center + 1}.issubset(valset):
                if axis == "h":
                    if (fixed, center) in body:
                        continue
                    cells = ((fixed, center - 1), (fixed, center + 1))
                    align = (fixed, center)
                else:
                    if (center, fixed) in body:
                        continue
                    cells = ((center - 1, fixed), (center + 1, fixed))
                    align = (center, fixed)
                if cells not in seen:
                    seen.add(cells)
                    options.append({"cells": cells, "align": align, "count": 2})
    return options


def _choose_marker_option(options):
    by_count = {
        2: [opt for opt in options if int(opt["count"]) == 2],
        3: [opt for opt in options if int(opt["count"]) == 3],
    }
    prefer_pair = random.random() < _PAIR_MARKER_PROB
    order = [2, 3] if prefer_pair else [3, 2]
    for count in order:
        if by_count[count]:
            return random.choice(by_count[count])
    return None


def _make_sprite(area, body_color, marker_color, axis, bg):
    for _ in range(_SPRITE_RETRY_LIMIT):
        body = _make_body(area)
        option = _choose_marker_option(_marker_line_options(body, axis))
        if option is None:
            continue
        markers = set(option["cells"])
        all_cells = set(body) | markers
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        for r, c in markers:
            patch[r - r0, c - c0] = int(marker_color)
        ar, ac = option["align"]
        return {
            "patch": patch,
            "align_offset": (int(ar - r0), int(ac - c0)),
            "marker_count": int(option["count"]),
            "axis": str(axis),
        }
    return None


def _top_left(sprite, align_r, align_c):
    off_r, off_c = sprite["align_offset"]
    return int(align_r) - int(off_r), int(align_c) - int(off_c)


def _can_place(canvas, sprite, align_r, align_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, align_r, align_c)
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


def _place(canvas, sprite, align_r, align_c, bg):
    top, left = _top_left(sprite, align_r, align_c)
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


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg)
    ]


def _scatter_positions(canvas, sprite, anchor, align_pos, bg):
    H, W = canvas.shape
    ar, ac = align_pos
    if anchor["axis"] == "h":
        return [
            (ar, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != ac and _can_place(canvas, sprite, ar, c, bg)
        ]
    return [
        (r, ac)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != ar and _can_place(canvas, sprite, r, ac, bg)
    ]


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
        return None

    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": str(axis)}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, anchor["axis"], bg)
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        align_pos = random.choice(positions)
        _place(out, sprite, align_pos[0], align_pos[1], bg)
        sprites.append((sprite, anchor, align_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, align_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, align_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D3 input")
