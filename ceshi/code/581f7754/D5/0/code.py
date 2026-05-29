# Variant of 581f7754
# Dimension: D5 — 变换：裁剪规则 (transform)
# Change: 滑动后超出网格边界的像素直接丢弃（裁剪）；目标位置的背景像素被覆盖，非背景像素不被覆盖... → 环绕而非裁剪
#
# Design analysis
# spec:
#   variant_core:
#     original_behavior: 每个精灵按锚点决定的单轴平移后，落到网格外的像素直接消失。
#     new_behavior: 同样的单轴平移仍然发生，但越界像素改为按行/列取模回到另一侧；
#       也就是右侧滑出从左侧出现、上侧滑出从下侧出现，形成环面拓扑。
#   concrete_effect:
#     input: 仍然是内部多色精灵 + 边界孤立锚点。输入中的精灵继续完全放在边界内侧，避免把“环绕”
#       混进对象识别阶段；变化只体现在滑动后的输出。
#     output: 至少一个精灵在对齐后会被切成两块，分别出现在网格两侧或上下两端，但两块颜色和形状
#       仍然能看出来自同一精灵，因为它们与同色锚点的对齐关系没有变。
#   behavior_trace:
#     example:
#       grid: 宽 10。某个顶边锚点位于列 2。一个水平滑动精灵的 marker 当前在列 7，所以平移量 dc = -5。
#       original: 该精灵中原本位于列 1 的身体像素移动到列 -4，直接被裁掉。
#       variant: 同一像素移动到列 -4 后改为列 6（-4 mod 10），因此输出会同时在左边和右边出现该精灵。
#       delta: 原题输出少一截；变体输出保留整只精灵，只是被边界“缝合”到另一侧。
#   complexity_calibration:
#     keep:
#       - 仍按 marker 颜色匹配锚点。
#       - 仍按锚点所在边决定水平或垂直滑动。
#       - 仍保留 1-3 个锚点和 3-5 个精灵，允许同色 marker 重复。
#     adapt_for_visibility:
#       - 生成器强制至少一个精灵真的发生环绕，而且至少 2 个像素越界回卷，避免退化成原题可解释样本。
#       - 为了更容易观察环绕，至少一个锚点优先放在可导致回卷的边缘坐标附近，并给对应精灵更偏大的面积。
#   implicit_assumption_audit:
#     main_lines:
#       - 旧版 anchor 检测、组件提取、marker 颜色推断都不依赖裁剪，可直接保留。
#       - 旧版 `out[coords] = bg` 的“先清空后粘贴”仍然成立，因为环绕只改变目的地坐标，不改变同时移动的语义。
#       - 真正失效的是 `if 0 <= rr < H and 0 <= cc < W` 这一剪裁判断；变体必须改为 `rr % H, cc % W`。
#       - 非背景不覆盖的假设仍需保留，否则多个精灵环绕后可能互相吞掉，改变题目语义。
#   placement_legality:
#     input_generation:
#       - 输入精灵继续使用原来的非环绕放置规则：完全位于内部、互不重叠、8 邻域不接触。
#     output_generation:
#       - 对齐后的输出精灵改用环绕放置检查：取模后的目标格必须为空，且与已有前景保持可见间隔。
#       - 同一个环绕精灵如果取模后自撞，判为非法；虽然当前尺寸范围几乎不会发生，生成器仍显式防御。
#   observability:
#     - 解题者在 2-4 组样例里会看到：同样是把 marker 对齐到同色边界锚点，但某些精灵不会缺角，
#       而是从另一侧“冒出来”。这和原题的裁剪结果是稳定可见的差异。
#     - 变化完全由几何位置触发，不涉及颜色编号排序或隐藏映射。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def _collect_moves(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, object]]]:
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
        cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[dict[str, object]] = []
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

        moves.append({"coords": coords.astype(int), "dr": int(dr), "dc": int(dc)})

    return g, bg, moves


def _apply_moves(g: np.ndarray, bg: int, moves: list[dict[str, object]], *, wrap: bool) -> np.ndarray:
    H, W = g.shape
    out = g.copy()

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
            if wrap:
                rr %= H
                cc %= W
                if int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])
            else:
                if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])

    return out


def variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g, bg, moves = _collect_moves(grid)
    H, W = g.shape
    wrapped_pixels = 0
    wrapped_components = 0
    for move in moves:
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        comp_wrap = 0
        for r, c in coords:
            rr = int(r) + dr
            cc = int(c) + dc
            if not (0 <= rr < H and 0 <= cc < W):
                comp_wrap += 1
        wrapped_pixels += comp_wrap
        if comp_wrap:
            wrapped_components += 1

    clipped = _apply_moves(g, bg, moves, wrap=False)
    wrapped = _apply_moves(g, bg, moves, wrap=True)
    return {
        "move_count": int(len(moves)),
        "wrapped_pixels": int(wrapped_pixels),
        "wrapped_components": int(wrapped_components),
        "changes_output": bool(not np.array_equal(clipped, wrapped)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior marker-bearing sprites until each marker lines up with the border anchor of the same color.

    Input structure:
    - The background is the most frequent color.
    - Interior 8-connected sprites are mostly one body color plus a rarer marker color.
    - Border singleton pixels with 4-neighbors equal to background serve as anchors.

    Rule:
    - Pair each sprite with the anchor that matches its marker color.
    - A top/bottom anchor makes the sprite slide horizontally to the anchor's column; a left/right anchor makes it slide vertically to the anchor's row.
    - After the slide, any pixel that would leave the grid wraps around to the opposite side; background may be overwritten, existing non-background cells may not.

    Key insight:
    - The anchor decides both which sprite moves and which axis is active; the only new twist is that the grid behaves like a torus during the final paste.
    """
    g, bg, moves = _collect_moves(grid)
    return _apply_moves(g, bg, moves, wrap=True)


# --------------- generator (o5x, D5 toroidal variant) ---------------

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
_WRAP_EDGE_BAND = 3

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_WRAP_BODY_AREA_MIN = 7
_GEN_RETRY_LIMIT = 300
_SPRITE_BUILD_TRIES = 60
_MIN_WRAP_PIXELS = 2

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


def _wrapped_sprite_cells(canvas_shape, sprite, marker_r, marker_c, bg):
    H, W = canvas_shape
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    wrapped_pixels = 0
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            color = int(patch[pr, pc])
            if color == int(bg):
                continue
            raw_r = top + pr
            raw_c = left + pc
            if not (0 <= raw_r < H and 0 <= raw_c < W):
                wrapped_pixels += 1
            cells.append((int(raw_r % H), int(raw_c % W), color))
    return cells, int(wrapped_pixels)


def _touches_foreground_cells(canvas, occupied_cells, bg):
    H, W = canvas.shape
    occupied = set(occupied_cells)
    for r, c in occupied:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (int(rr), int(cc)) in occupied:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return True
    return False


def _can_place_wrapped(canvas, sprite, marker_r, marker_c, bg, *, min_wrap_pixels=0):
    cells, wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, marker_r, marker_c, bg)
    if wrapped_pixels < int(min_wrap_pixels):
        return False

    occupied = set()
    for rr, cc, _color in cells:
        key = (int(rr), int(cc))
        if key in occupied:
            return False
        occupied.add(key)
        if int(canvas[rr, cc]) != int(bg):
            return False

    return not _touches_foreground_cells(canvas, occupied, bg)


def _place_wrapped(canvas, sprite, marker_r, marker_c, bg):
    cells, _wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, marker_r, marker_c, bg)
    for rr, cc, color in cells:
        canvas[rr, cc] = int(color)


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


def _anchor_is_wrap_friendly(axis, r, c, shape):
    H, W = shape
    if axis == "h":
        return int(c) <= _WRAP_EDGE_BAND or int(c) >= int(W - 1 - _WRAP_EDGE_BAND)
    return int(r) <= _WRAP_EDGE_BAND or int(r) >= int(H - 1 - _WRAP_EDGE_BAND)


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, bg, *, min_wrap_pixels=0):
    H, W = canvas.shape
    positions = []
    if anchor["axis"] == "h":
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if _can_place_wrapped(canvas, sprite, r, anchor["c"], bg, min_wrap_pixels=min_wrap_pixels):
                positions.append((int(r), int(anchor["c"])))
        return positions
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        if _can_place_wrapped(canvas, sprite, anchor["r"], c, bg, min_wrap_pixels=min_wrap_pixels):
            positions.append((int(anchor["r"]), int(c)))
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
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _add_distractors(out, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in range(_INNER_BORDER_GAP, out.shape[1] - _INNER_BORDER_GAP):
            if not _can_place(out, sprite, anchor["r"], c, bg):
                continue
            for r in range(_INNER_BORDER_GAP, inp.shape[0] - _INNER_BORDER_GAP):
                if r != anchor["r"] and _can_place(inp, sprite, r, c, bg):
                    pairs.append((r, c))
        if not pairs:
            return
        in_r, col = random.choice(pairs)
        _place(out, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    wrap_color = int(anchor_colors[0])
    anchors = []
    for idx, color in enumerate(anchor_colors):
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        if idx == 0:
            preferred = [cand for cand in candidates if _anchor_is_wrap_friendly(*cand, out.shape)]
            if preferred:
                candidates = preferred
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors, wrap_color


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _build_aligned_sprite(out, anchor, marker, bg, used_body, *, require_wrap):
    for _ in range(_SPRITE_BUILD_TRIES):
        area_lo = max(_WRAP_BODY_AREA_MIN, _BODY_AREA_RANGE[0]) if require_wrap else _BODY_AREA_RANGE[0]
        area = random.randint(area_lo, _BODY_AREA_RANGE[1])
        body = _body_color(bg, marker, used_body)
        sprite = _make_sprite(area, body, marker, bg)
        positions = _aligned_positions(
            out,
            sprite,
            anchor,
            bg,
            min_wrap_pixels=_MIN_WRAP_PIXELS if require_wrap else 0,
        )
        if not positions:
            continue
        marker_pos = random.choice(positions)
        _place_wrapped(out, sprite, marker_pos[0], marker_pos[1], bg)
        used_body.add(body)
        wrapped_pixels = _wrapped_sprite_cells(out.shape, sprite, marker_pos[0], marker_pos[1], bg)[1]
        return sprite, marker_pos, int(wrapped_pixels)
    return None


def _place_aligned_sprites(out, anchors, marker_colors, bg, wrap_color):
    by_color = {a["color"]: a for a in anchors}
    wrap_index = next((i for i, color in enumerate(marker_colors) if int(color) == int(wrap_color)), None)
    used_body = set()
    sprites = []
    for idx, marker in enumerate(marker_colors):
        anchor = by_color[int(marker)]
        placed = _build_aligned_sprite(
            out,
            anchor,
            int(marker),
            bg,
            used_body,
            require_wrap=idx == wrap_index,
        )
        if placed is None:
            return None
        sprite, marker_pos, wrapped_pixels = placed
        sprites.append((sprite, anchor, marker_pos, int(wrapped_pixels)))
    if max((wrapped_pixels for _sprite, _anchor, _marker_pos, wrapped_pixels in sprites), default=0) < _MIN_WRAP_PIXELS:
        return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos, _wrapped_pixels in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    metrics = variant_metrics(inp)
    if int(metrics["wrapped_pixels"]) < _MIN_WRAP_PIXELS:
        return False
    if not bool(metrics["changes_output"]):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors, wrap_color = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg, wrap_color)
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
    raise RuntimeError("failed to generate a valid 581f7754 D5 input")
