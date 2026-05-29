# Variant of 581f7754
# Dimension: D7 — 网格形状：输出裁切方式 (grid_shape)
# Change: 输出网格与输入网格尺寸完全一致，不进行裁切或扩展... → 扩展网格以容纳完整精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# DESIGN ANALYSIS
# variant_analysis:
#   core_change:
#     - 原题在精灵平移后直接把越出边界的像素裁掉；D7 变体改为保留这些像素。
#     - 输出画布不是 tight crop，也不是固定原尺寸，而是“至少包含整个原输入矩形，
#       再向发生越界的方向扩张到足以容纳所有平移后的像素”。
#   concrete_behavior:
#     - 例：宽 4 的精灵，其标记像素距左边 2 格；若它匹配到顶边锚点 c=1，
#       原题水平对齐后会让左侧 1 格被裁掉；本变体会在输出左侧补 1 列背景，
#       把整只精灵完整保留。输入网格其余区域整体平移到新画布中。
#     - 同理，左/右边锚点触发的是垂直滑动；如果标记对齐到 r=1 而精灵上方还
#       有 2 行，则输出会在顶部补 1 行。
#   affected_pipeline_steps:
#     - solver:
#       1) 锚点检测与组件匹配逻辑不变；
#       2) 位移向量计算不变；
#       3) 唯一核心变化是渲染阶段：先统计所有目标坐标的全局最小/最大行列，
#          再把原输入嵌入扩张后的画布并执行移动。
#     - generator:
#       1) 原生成器把“对齐后的解态”限制在原画布内，因此不会触发扩张；
#       2) 现在显式要求至少一个精灵在对齐后横向或纵向越出原边界，
#          但输入中的散开位置仍须完整位于原网格内。
#   complexity_calibration:
#     - 仍保留 1-3 个边界锚点、3-5 个双色精灵、少量不同 body 形状；
#       这样人类仍能靠“标记色匹配锚点 + 看哪一侧扩张”推断规则。
#     - 不引入新的排序、颜色映射或多阶段变换，避免把 D7 之外的维度混进来。
#   implicit_assumption_review:
#     - 原 solver 在粘贴时假设输出尺寸固定为 HxW，因而用 in-bounds 判断来裁剪；
#       这个前提在变体中失效，必须先重算输出 bounds 再渲染。
#     - 原 generator 的 _aligned_positions 只返回完全在框内的位置；这会使 D7 退化。
#       现在改成枚举“允许解态越界但输入态必须合法”的位置。
#     - 原始“先清空源组件再粘贴目标组件”的顺序仍然需要，否则会被自身旧位置挡住。
#   placement_legality:
#     - 输入态沿用原任务的隔离要求：所有精灵完整在网格内，且与其他前景保持 8 邻域间隔。
#     - 解态虽然允许越出原框，但不同精灵与锚点仍保持分离，不制造碰撞或贴边混淆。
#   visual_observability:
#     - 生成器拒绝所有“不发生扩张”的样本，确保训练对中能直接看到输出比输入更大。
#     - 扩张方向由精灵越界方向决定，可出现在上/下/左/右，不依赖颜色 id。
#   neatness:
#     - 只允许少量扩张；精灵越界幅度受其小尺寸限制，输出不会出现大面积无意义空白。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker-tagged sprites toward matching border anchors and enlarge the canvas when needed.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are sprites; each multi-color sprite has a rare marker color.
    - Isolated non-background border pixels serve as anchors keyed by color.

    Rule:
    - Pair each sprite with the anchor that shares its marker color.
    - A top/bottom anchor fixes the marker column, so the sprite slides horizontally.
    - A left/right anchor fixes the marker row, so the sprite slides vertically.
    - If the translated sprite would cross the old frame, expand the output grid on that side instead of clipping it.

    Key insight:
    - The anchor decides the slide axis, but the new part is geometric: compute the full destination footprint first, then build a larger canvas that contains both the original grid and every translated pixel.
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
        comp_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []
    min_r = 0
    min_c = 0
    max_r = int(H - 1)
    max_c = int(W - 1)

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
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        rr = coords[:, 0] + int(dr)
        cc = coords[:, 1] + int(dc)
        min_r = min(int(min_r), int(rr.min()))
        min_c = min(int(min_c), int(cc.min()))
        max_r = max(int(max_r), int(rr.max()))
        max_c = max(int(max_c), int(cc.max()))
        moves.append((coords, int(dr), int(dc)))

    out_h = int(max_r - min_r + 1)
    out_w = int(max_c - min_c + 1)
    row_off = int(-min_r)
    col_off = int(-min_c)

    out = np.full((out_h, out_w), int(bg), dtype=int)
    paste(out, g, row_off, col_off)

    for coords, _dr, _dc in moves:
        out[row_off + coords[:, 0], col_off + coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = row_off + int(r) + int(dr)
            cc = col_off + int(c) + int(dc)
            if int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 1
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
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_OVERFLOW_SPRITE_PROB = 0.6

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
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    random.shuffle(rows)
    random.shuffle(cols)

    horiz = [("h", 0, c) for c in cols] + [("h", H - 1, c) for c in cols]
    vert = [("v", r, 0) for r in rows] + [("v", r, W - 1) for r in rows]

    def bias_key(item):
        axis, r, c = item
        if axis == "h":
            return min(int(c), int(W - 1 - c))
        return min(int(r), int(H - 1 - r))

    all_candidates = horiz + vert
    all_candidates.sort(key=bias_key)
    head = all_candidates[: max(6, len(all_candidates) // 2)]
    random.shuffle(head)
    tail = all_candidates[max(6, len(all_candidates) // 2) :]
    random.shuffle(tail)
    ordered = head + tail
    return [(axis, r, c) for axis, r, c in ordered if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _scatter_positions(canvas, sprite, anchor, aligned_marker, bg):
    H, W = canvas.shape
    mr, mc = aligned_marker
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


def _sprite_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v != int(bg):
                cells.append((int(top + r), int(left + c), int(v)))
    return cells


def _candidate_clear(existing, candidate_cells):
    occupied = existing
    for r, c, _v in candidate_cells:
        if (int(r), int(c)) in occupied:
            return False
        for dr, dc in _NEIGHBORS8:
            if (int(r + dr), int(c + dc)) in occupied:
                return False
    return True


def _overflow_amount(sprite, marker_r, marker_c, H, W):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    return {
        "top": max(0, -int(top)),
        "left": max(0, -int(left)),
        "bottom": max(0, int(top + h - H)),
        "right": max(0, int(left + w - W)),
    }


def _place_anchors(canvas, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(canvas, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(canvas, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _aligned_marker_candidates(sprite, anchor, canvas_shape, solved_occupied, bg, require_overflow):
    H, W = canvas_shape
    patch = sprite["patch"]
    h, w = patch.shape
    off_r, off_c = sprite["offset"]
    out = []
    if anchor["axis"] == "h":
        for marker_r in range(_INNER_BORDER_GAP + off_r, H - _INNER_BORDER_GAP - (h - off_r) + 1):
            marker_c = int(anchor["c"])
            cells = _sprite_cells(sprite, marker_r, marker_c, bg)
            over = _overflow_amount(sprite, marker_r, marker_c, H, W)
            has_overflow = any(over.values())
            if require_overflow is not None and has_overflow != bool(require_overflow):
                continue
            if _candidate_clear(solved_occupied, cells):
                out.append(((int(marker_r), int(marker_c)), over))
    else:
        for marker_c in range(_INNER_BORDER_GAP + off_c, W - _INNER_BORDER_GAP - (w - off_c) + 1):
            marker_r = int(anchor["r"])
            cells = _sprite_cells(sprite, marker_r, marker_c, bg)
            over = _overflow_amount(sprite, marker_r, marker_c, H, W)
            has_overflow = any(over.values())
            if require_overflow is not None and has_overflow != bool(require_overflow):
                continue
            if _candidate_clear(solved_occupied, cells):
                out.append(((int(marker_r), int(marker_c)), over))
    random.shuffle(out)
    return out


def _gen_attempt():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    inp = np.full((H, W), int(bg), dtype=int)

    placed = _place_anchors(inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    marker_colors = _marker_color_sequence(anchor_colors)

    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    solved_occupied = {(int(a["r"]), int(a["c"])) for a in anchors}
    has_overflow = False

    for i, marker in enumerate(marker_colors):
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)

        wants_overflow = (not has_overflow) or (random.random() < _OVERFLOW_SPRITE_PROB)
        aligned_options = _aligned_marker_candidates(
            sprite,
            anchor,
            inp.shape,
            solved_occupied,
            bg,
            require_overflow=wants_overflow,
        )
        if not aligned_options:
            aligned_options = _aligned_marker_candidates(
                sprite,
                anchor,
                inp.shape,
                solved_occupied,
                bg,
                require_overflow=None,
            )
        if not aligned_options:
            return None

        chosen = None
        for aligned_marker, over in aligned_options:
            scatter_positions = _scatter_positions(inp, sprite, anchor, aligned_marker, bg)
            if not scatter_positions:
                continue
            chosen = (aligned_marker, over, random.choice(scatter_positions))
            break
        if chosen is None:
            return None

        aligned_marker, over, input_marker = chosen
        _place(inp, sprite, input_marker[0], input_marker[1], bg)
        for r, c, _v in _sprite_cells(sprite, aligned_marker[0], aligned_marker[1], bg):
            solved_occupied.add((int(r), int(c)))
        has_overflow = has_overflow or any(over.values())

    if not has_overflow:
        return None
    if not _bg_is_mode(inp, bg):
        return None

    out = main(inp)
    if out.shape == inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D7 input")
