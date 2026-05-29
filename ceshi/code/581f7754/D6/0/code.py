# Variant of 581f7754
# Dimension: D6 — 变换：原始位置处理 (trajectory)
# Change: 精灵的原始位置像素全部变为背景色（清除），锚点保持不变... → 保留源精灵（复制而非移动）

from __future__ import annotations

# Design analysis for the D6 keep_source variant
# ------------------------------------------------
# a) 具体化与行为追踪
#    - 画面元素仍和原题一致：边界上有孤立锚点，内部有 2 色精灵，少数色像素是 marker。
#    - 原题流程：
#         1. 找到 marker 颜色对应的锚点。
#         2. 按锚点所在边决定滑动轴。
#         3. 计算平移量。
#         4. 先把原位置清空成背景，再把精灵粘到对齐位置。
#    - 本变体只改第 4 步：平移量完全不变，但不再清空源位置。输出是“输入原样保留 + 新增一个对齐副本”。
#    - 小例子：
#         * 顶边锚点颜色为 4，列号为 8。
#         * 某个精灵内部唯一的 4 在 (6,3)，其余主体是颜色 7。
#         * 原题会水平平移 +5，把整块从原处移到 marker 列=8，并清掉旧块。
#         * 本题同样水平平移 +5，但旧块不消失，所以输出里同时看到列 3 的源块和列 8 的新块。
#
# b) 复杂度校准
#    - 如果允许源块和目标副本重叠，输出可能只像“局部加粗”，人类不容易看出是复制。
#    - 因此生成器增加一个关键约束：输入中的源精灵必须既能放进 input，也必须能“放进 solved output 的背景区”，
#      也就是它与未来的对齐副本、锚点、其他已对齐精灵都不重叠且不 8 邻接。
#    - 这样每个移动精灵在输出里都形成两个清晰分离的副本，D6 变化在 2-3 个样例里就足够可见。
#    - 其余自由度保持原题级别：1-3 个锚点、3-5 个精灵、marker 颜色可复用，避免变成另一道题。
#
# c) 隐式假设审查（对应 main 的关键代码段）
#    - 锚点检测依赖“边界单像素且 4 邻域为背景”；D6 不改变这一点，原逻辑可保留。
#    - 组件提取与 marker 颜色判定依赖“marker 是组件内少数色”；D6 不改变对象内部结构，原逻辑可保留。
#    - 平移量计算只依赖锚点位置和 marker 像素；D6 不改变对齐规则，原逻辑可保留。
#    - 原题唯一依赖“移动”语义的代码是 `out[coords]=bg` 这一清除步骤。对 keep_source 来说该前提失效，必须删掉。
#    - 贴图仍然使用“只写入背景格”的保护逻辑；这要求 generator 保证目标区域在 input 中确实是背景，
#      否则副本会被现有前景挡住。新的 generator 会显式构造这一点。
#
# d) 放置合法性
#    - output-first 框架保留：先在 solved canvas 上放好锚点和对齐副本，再把对应源精灵散射到 input。
#    - 散射时同时检查：
#         * 对 input 来说不重叠、不越界、不接触其他输入前景；
#         * 对 solved output 来说也不重叠、不越界、不接触现有前景。
#    - 于是 `main(input)` 时新增副本一定能完整贴入，且不会和源块/别的副本打架。
#
# e) 视觉可观察性
#    - 复制规则不能依赖颜色 id 排序或隐藏状态；这里完全由可见的空间现象体现：
#      输入中的每个精灵在输出里出现“原位保留 + 沿特定轴再出现一次”。
#    - 多个精灵共享同一锚点颜色仍然保留，这帮助解题者看到“同色 marker 对应同一轴/同一锚点”的稳定规律。
#
# f) 可视化整洁度
#    - 新增的“源位置也保留”会天然增加密度，所以生成器比原题更严格地把 source 与 solved 区域隔开，
#      避免两层图形纠缠成一团。
#    - 锚点仍是边界孤立像素，和内部双副本结构保持明显分层。
#
# g) 颜色变化可推断性
#    - 本变体不引入新的颜色映射；颜色作用与原题一致：
#      锚点颜色 == marker 颜色，主体颜色只是区分形状。
#    - 因为 D6 只改轨迹处理方式，所以保持颜色规则不变最有利于人类隔离出“复制而非移动”这一核心变化。
#
# Reference-task calibration
#    - 参考任务 247ef758 的核心机制是“先保留源识别结构，再把同一精灵复制到多个目标位置”。
#    - 对本题的 keep_source 变体，复杂度应保持在同一量级：不发明新的匹配规则，只把“擦除后粘贴”
#      改成“保留原图并额外粘贴”，并用生成约束确保复制效果稳定可见。

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy multi-color sprites along the anchor-determined axis so their marker pixels line up with matching border anchors.

    Input structure:
    - The background is the most frequent color.
    - Interior sprites are 8-connected multi-color components whose least-frequent color serves as a marker.
    - Isolated non-background border pixels provide anchor colors and determine the slide axis.

    Rule:
    - Match each sprite to the anchor sharing its marker color.
    - Top/bottom anchors make the sprite copy slide horizontally; left/right anchors make it copy vertically.
    - Keep the original sprite in place and add a translated copy, clipping any out-of-bounds pixels and never overwriting existing non-background cells.

    Key insight: The color match chooses the anchor, but the anchor's border side chooses the axis, and the output keeps both the source sprite and its aligned copy.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

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

    assert not (not anchors_by_color)

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

    out = g.copy()

    # Keep the source pixels and stamp translated copies only into empty cells.
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (D6 keep_source) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500

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


def _scatter_positions(inp, solved, sprite, anchor, marker, bg):
    H, W = inp.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        positions = []
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc:
                continue
            if not _can_place(inp, sprite, mr, c, bg):
                continue
            if not _can_place(solved, sprite, mr, c, bg):
                continue
            positions.append((mr, c))
        return positions

    positions = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        if r == mr:
            continue
        if not _can_place(inp, sprite, r, mc, bg):
            continue
        if not _can_place(solved, sprite, r, mc, bg):
            continue
        positions.append((r, mc))
    return positions


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _merge_layers(base, overlay, bg):
    out = base.copy()
    overlay_mask = overlay != int(bg)
    conflict = overlay_mask & (out != int(bg)) & (overlay != out)
    if bool(conflict.any()):
        return None
    out[overlay_mask & (out == int(bg))] = overlay[overlay_mask & (out == int(bg))]
    return out


def _add_distractors(solved, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return True
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in range(_INNER_BORDER_GAP, solved.shape[1] - _INNER_BORDER_GAP):
            if not _can_place(solved, sprite, anchor["r"], c, bg):
                continue
            for r in range(_INNER_BORDER_GAP, solved.shape[0] - _INNER_BORDER_GAP):
                if r == anchor["r"]:
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                if not _can_place(solved, sprite, r, c, bg):
                    continue
                pairs.append((r, c))
        if not pairs:
            return False
        in_r, col = random.choice(pairs)
        _place(solved, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)
    return True


def _gen_attempt():
    bg, solved, inp = _new_canvas_pair()
    placed = _place_anchors(solved, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(solved, anchors, marker_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, solved, sprites, bg):
        return None
    if not _add_distractors(solved, inp, anchors, bg):
        return None

    expected = _merge_layers(solved, inp, bg)
    if expected is None:
        return None
    if not _generated_grid_is_valid(inp, expected, bg):
        return None
    return inp


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    solved = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, solved, inp


def _place_anchors(solved, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(solved, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(solved, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(solved, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(solved, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(solved, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, solved, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, solved, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, expected, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if int(np.count_nonzero(expected != inp)) == 0:
        return False
    return bool(np.array_equal(main(inp), expected))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D6 input")
