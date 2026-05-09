# Variant of 581f7754
# Dimension: D8 — 多步处理：精灵移动顺序 (multi_step)
# Change: 所有精灵同时移动（先统一清除原位置，再统一粘贴到新位置），无先后顺序依赖... → 按颜色分组依次处理

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# -----------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# a) 具体化与行为追踪
# - 保留原题的可见对象：边框上的孤立锚点，内部的双色精灵，少数色像素仍是 marker。
# - 仍按锚点所在边决定平移轴：上/下边锚点对应水平平移，左/右边锚点对应垂直平移。
# - 新变化不是“往哪里移动”，而是“什么时候移动”：同 marker 色的一组同时移动，不同组按锚点沿边框顺时针顺序依次结算。
# - 关键的小例子：
#   * 组 A 的锚点在上边，组 B 的锚点在右边，且 A 的目标落点里有几个格子恰好被 B 在输入中的原位置占着。
#   * 原题做法：先把 A/B 全部原位置一起清空，再一起粘贴，所以 A 能完整落到那些格子上。
#   * 变体做法：A 先动时，B 还没清空，因此 A 只能落下未被 B 源位置占据的那部分；之后 B 再移走，A 也不会补贴回来。
#   * 差异体现在输出中：A 对齐到了锚点，但形状会在被后组“临时挡住”的位置缺像素。
#
# b) 复杂度校准
# - 若只改 solver 顺序、不改 generator，许多样本仍会与“同时移动”等价，D8 不可见。
# - 因此 generator 必须主动制造跨组目标/源位置冲突；同时保持冲突数量有限，避免输出过乱。
# - 锚点数固定为 2-3，且每个样例至少有一个重复 marker 色，保证“分组”而不是“单个精灵顺序”有样例支撑。
#
# c) 隐式假设审查（原 main 中与 D8 相关的前提）
# - 原 `moves.append(...)` 本身没问题，只是收集平移信息。
# - 原 “先清空所有 moves 再粘贴所有 moves” 隐含前提：所有可移动物体在结算时互不阻挡，顺序无关。
# - 该前提在变体里失效；必须改成“只清空当前组，再立刻粘贴当前组”。
# - 原 generator 把 `out` 当成所有精灵同时到位后的静态目标图；变体里这会退化，因此改成：
#   * `plan` 仅用于规划各精灵本来想对齐到哪里；
#   * 真正的 `output` 由新 `main(input)` 计算得出。
#
# d) 放置合法性
# - 规划态 `plan` 中，精灵仍按原题方式彼此分离、与边框留空，保证“本来想去哪里”清楚。
# - 输入态 `inp` 中，精灵之间也仍保持不重叠、不接触；只是某些后组源位置会落到前组规划目标区域上。
# - 这样输入可被稳定分割成独立组件，solver 能可靠识别 marker 与锚点。
#
# e) 视觉可观察性
# - 组顺序不用颜色编号排序，改用锚点在边框上的顺时针顺序；这是像素上可见的规则。
# - generator 强制检查新 solver 与“原始同时移动 solver”输出不同，避免退化成原题。
#
# f) 可视化整洁度
# - 冲突只通过“后组源位置暂时挡住前组目标位置”产生，不引入额外覆盖层或新对象类型。
# - 规划态精灵仍保持分离，输入态精灵也保持分离，因此画面主元素仍然清楚。
#
# g) 颜色变化可推断性
# - 颜色不参与隐藏排序；颜色仍只承担“marker 与锚点配对”的可见角色。
# - 多个精灵共享同一 marker 色，支持解题者看出“同色一起动”的分组。
#
# h) 生成器验证策略
# - 除了 `main(inp)` 不报错，还额外要求：
#   * 至少存在一个“前组目标格子被后组源位置占据”的冲突单元；
#   * 新 solver 输出与旧的同时移动输出存在差异像素。
#
# i) 新 solver 的输入假设与生成器对应保证
# - 假设 1：每个可移动组件都有可判定的 marker 色。生成器始终生成双色精灵；可选单像素干扰物只使用锚点色。
# - 假设 2：每个 marker 色恰有一个锚点。生成器维持每种锚点色唯一。
# - 假设 3：按组件原始坐标清空当前组不会误删别组像素。生成器保证输入态组件互不重叠。
# -----------------------------------------------------------------------------


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _clockwise_key(r: int, c: int, H: int, W: int) -> int:
    """Visible step order: top L->R, right T->B, bottom R->L, left B->T."""
    r = int(r)
    c = int(c)
    H = int(H)
    W = int(W)
    if r == 0:
        return c
    if c == W - 1:
        return (W - 1) + r
    if r == H - 1:
        return (W - 1) + (H - 1) + (W - 1 - c)
    return (W - 1) + (H - 1) + (W - 1) + (H - 1 - r)


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


def _collect_scene(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(r, c, H, W):
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
    group_order = sorted(
        anchors_by_color,
        key=lambda color: _clockwise_key(
            anchors_by_color[int(color)][0],
            anchors_by_color[int(color)][1],
            H,
            W,
        ),
    )

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves_by_color: dict[int, list[dict]] = {int(color): [] for color in group_order}

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

        moves_by_color[int(marker_col)].append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker": int(marker_col),
            }
        )

    return g, bg, anchors_by_color, group_order, moves_by_color


def _apply_grouped_moves(
    g: np.ndarray,
    bg: int,
    group_order: list[int],
    moves_by_color: dict[int, list[dict]],
) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    for color in group_order:
        group = moves_by_color.get(int(color), [])
        if not group:
            continue
        for move in group:
            coords = move["coords"]
            out[coords[:, 0], coords[:, 1]] = int(bg)
        for move in group:
            coords = move["coords"]
            dr = int(move["dr"])
            dc = int(move["dc"])
            for r, c in coords:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])
    return out


def _apply_simultaneous_moves(
    g: np.ndarray,
    bg: int,
    group_order: list[int],
    moves_by_color: dict[int, list[dict]],
) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    moves = [move for color in group_order for move in moves_by_color.get(int(color), [])]
    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for move in moves:
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        for r, c in coords:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift marker-bearing sprites toward matching border anchors, but resolve the shifts one color-group at a time.

    Input structure:
    - The modal color is background.
    - Interior foreground components are mostly two-color sprites whose minority color is the marker; isolated interior singletons of an anchor color may also move.
    - Isolated border pixels serve as anchors, and the side of the border tells whether matching sprites slide horizontally or vertically.

    Rule:
    - Pair each movable component with the unique border anchor of the same marker color.
    - Sprites sharing one marker color move together, while different color groups are processed in clockwise anchor order around the border.
    - For each group step, clear only that group's source cells, then paste its translated pixels with clipping and without overwriting any non-background cells already present.

    Key insight:
    - The anchor now determines both the slide axis and the processing order, so earlier groups can lose pixels to later groups that are still sitting in the way.
    """
    g, bg, _anchors_by_color, group_order, moves_by_color = _collect_scene(np.array(grid, dtype=int))
    return _apply_grouped_moves(g, bg, group_order, moves_by_color)


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
_EDGE_ORDER = ("top", "right", "bottom", "left")

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_MIN_DEPENDENCY_CELLS = 1

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
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

# CFG: optional singleton distractors.
_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 3)

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
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _sprite_cells(sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    patch = sprite["patch"]
    cells = set()
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if int(patch[r, c]) != int(bg):
                cells.add((int(top + r), int(left + c)))
    return cells


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


def _anchor_candidates_for_edge(canvas, bg, edge):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))

    if edge == "top":
        raw = [("h", 0, c) for c in cols]
    elif edge == "bottom":
        raw = [("h", H - 1, c) for c in cols]
    elif edge == "left":
        raw = [("v", r, 0) for r in rows]
    else:
        raw = [("v", r, W - 1) for r in rows]

    random.shuffle(raw)
    return [(axis, r, c) for axis, r, c in raw if _anchor_ok(canvas, r, c, bg)]


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
    plan = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, plan, inp


def _place_anchors(plan, inp, bg):
    anchor_count = random.randint(*_ANCHOR_COUNT_RANGE)
    chosen_edges = random.sample(_EDGE_ORDER, anchor_count)
    colors = random.sample([c for c in _COLORS if c != bg], anchor_count)
    anchors = []
    for edge, color in zip(chosen_edges, colors):
        candidates = _anchor_candidates_for_edge(plan, bg, edge)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis, "edge": edge}
        anchors.append(anchor)
        _put_anchor(plan, r, c, color)
        _put_anchor(inp, r, c, color)
    anchors.sort(key=lambda a: _clockwise_key(a["r"], a["c"], plan.shape[0], plan.shape[1]))
    return anchors


def _marker_color_sequence(anchor_colors):
    min_sprites = max(_SPRITE_COUNT_RANGE[0], len(anchor_colors) + 1)
    n_sprites = random.randint(min_sprites, _SPRITE_COUNT_RANGE[1])
    marker_colors = list(anchor_colors)
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _plan_sprite_records(plan, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    records = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(plan, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(plan, sprite, marker_pos[0], marker_pos[1], bg)
        records.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "marker_color": int(marker),
                "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                "planned_cells": _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg),
            }
        )
    return records


def _maybe_add_distractors(plan, anchors, records, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return True
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    count = random.randint(*_DISTRACTOR_COUNT_RANGE)
    for _ in range(count):
        positions = _aligned_positions(plan, sprite, anchor, bg)
        if not positions:
            return False
        marker_pos = random.choice(positions)
        _place(plan, sprite, marker_pos[0], marker_pos[1], bg)
        records.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "marker_color": int(anchor["color"]),
                "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                "planned_cells": _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg),
            }
        )
    return True


def _scatter_records(inp, anchors, records, bg):
    ordered_colors = [int(a["color"]) for a in anchors]
    earlier_cells: dict[int, set[tuple[int, int]]] = {}
    seen = set()
    running = set()
    for color in ordered_colors:
        earlier_cells[int(color)] = set(running)
        if int(color) not in seen:
            running |= set().union(*(rec["planned_cells"] for rec in records if int(rec["marker_color"]) == int(color)))
            seen.add(int(color))

    dependency_cells = 0
    by_color: dict[int, list[dict]] = {}
    for rec in records:
        by_color.setdefault(int(rec["marker_color"]), []).append(rec)

    for color in reversed(ordered_colors):
        group = by_color.get(int(color), [])
        random.shuffle(group)
        target_cells = earlier_cells[int(color)]
        for rec in group:
            positions = _scatter_positions(inp, rec["sprite"], rec["anchor"], rec["marker_pos"], bg)
            if not positions:
                return None

            scored = []
            for pos in positions:
                cells = _sprite_cells(rec["sprite"], pos[0], pos[1], bg)
                overlap = len(cells & target_cells)
                scored.append((int(overlap), (int(pos[0]), int(pos[1])), cells))

            positive = [item for item in scored if item[0] > 0]
            if positive and (dependency_cells == 0 or random.random() < 0.8):
                scored = sorted(positive, key=lambda item: item[0], reverse=True)
                window = scored[: min(4, len(scored))]
                overlap, chosen, cells = random.choice(window)
            else:
                overlap, chosen, cells = random.choice(scored)

            _place(inp, rec["sprite"], chosen[0], chosen[1], bg)
            rec["source_pos"] = chosen
            rec["source_cells"] = cells
            rec["dependency_overlap"] = int(overlap)
            dependency_cells += int(overlap)
    return dependency_cells


def _simultaneous_main(grid: np.ndarray) -> np.ndarray:
    g, bg, _anchors_by_color, group_order, moves_by_color = _collect_scene(np.array(grid, dtype=int))
    return _apply_simultaneous_moves(g, bg, group_order, moves_by_color)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g, _bg, _anchors_by_color, group_order, moves_by_color = _collect_scene(np.array(grid, dtype=int))
    H, W = g.shape

    later_sources: dict[int, set[tuple[int, int]]] = {}
    running = set()
    for color in reversed(group_order):
        later_sources[int(color)] = set(running)
        for move in moves_by_color.get(int(color), []):
            running |= {(int(r), int(c)) for r, c in move["coords"]}

    blocked_cells = 0
    repeated_groups = 0
    for color in group_order:
        group = moves_by_color.get(int(color), [])
        if len(group) >= 2:
            repeated_groups += 1
        blockers = later_sources[int(color)]
        for move in group:
            for r, c in move["coords"]:
                rr = int(r) + int(move["dr"])
                cc = int(c) + int(move["dc"])
                if 0 <= rr < H and 0 <= cc < W and (int(rr), int(cc)) in blockers:
                    blocked_cells += 1

    grouped = main(g)
    simultaneous = _simultaneous_main(g)
    diff_pixels = int(np.count_nonzero(grouped != simultaneous))
    return {
        "groups": int(len(group_order)),
        "repeated_groups": int(repeated_groups),
        "blocked_cells": int(blocked_cells),
        "diff_pixels": int(diff_pixels),
    }


def _generated_grid_is_valid(inp, bg):
    if not _bg_is_mode(inp, bg):
        return False
    metrics = _variant_metrics(inp)
    if metrics["blocked_cells"] < _MIN_DEPENDENCY_CELLS:
        return False
    if metrics["diff_pixels"] <= 0:
        return False
    return True


def _gen_attempt():
    bg, plan, inp = _new_canvas_pair()
    anchors = _place_anchors(plan, inp, bg)
    if anchors is None:
        return None

    marker_colors = _marker_color_sequence([a["color"] for a in anchors])
    records = _plan_sprite_records(plan, anchors, marker_colors, bg)
    if records is None:
        return None

    if not _maybe_add_distractors(plan, anchors, records, bg):
        return None

    dependency_cells = _scatter_records(inp, anchors, records, bg)
    if dependency_cells is None or dependency_cells < _MIN_DEPENDENCY_CELLS:
        return None

    if not _generated_grid_is_valid(inp, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D8 variant input")
