# Combo variant of 581f7754 (index 23)
# combo_spec: D7/1+D14/0
# Name: 扩展网格+反向滑动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# DESIGN ANALYSIS
# combo_analysis:
#   concrete_trace:
#     - 原题里，顶/底边锚点只决定“水平对齐到锚点列”，左/右边锚点只决定“垂直对齐到锚点行”，
#       并且移动方向朝向锚点；若越界就裁掉。
#     - 本融合版保留“哪条边决定哪根轴”的部分，但把位移取反。例：顶边锚点在 c=2，
#       某精灵 marker 在 c=6，则原题 dc=-4，输出 marker 去 c=2；这里改成 dc=+4，
#       输出 marker 去 c=10，也就是和锚点关于输入 marker 镜像对称地远离它。
#     - 同时取消裁剪。若上例中精灵右半部分超出原宽度，就不是丢失，而是把整张输出向右扩展，
#       并保留完整精灵。
#   generator_strategy:
#     - 不能沿用原题/纯 D7 的 output-first，因为“解态”不再由锚点列/行直接给出。
#     - 更稳妥的是 input-first：先枚举输入中合法的 marker 位置，再由反向公式推出输出 marker：
#         horizontal axis -> (r, 2*c-anchor_c)
#         vertical axis   -> (2*r-anchor_r, c)
#       输出允许越出原框，但候选之间仍需彼此隔离。
#   complexity_calibration:
#     - 仍保留 1-3 个锚点、3-5 个精灵、两色主体+marker、小概率单像素干扰物。
#     - 要求至少一个对象真实扩张，且所有移动距离至少为 2，避免样本退化成原题或仅 D14。
#     - 锚点候选偏向边缘，增加“远离锚点后发生扩张”的可见性，但限制最大滑动距离和最大扩张，
#       防止输出出现大片无意义留白。
#   implicit_assumption_review:
#     - anchors_by_color 仍假设每种锚点颜色唯一；生成器继续保证这一点。
#     - component_marker_color 仍依赖 marker 是组件内最少颜色；生成器继续构造单 marker 的双色精灵。
#     - marker_pixel 仍优先选贴主体的少数色像素；生成器继续保证 marker 与主体 4 邻接。
#     - 原 solver 假设输出尺寸固定，因此直接做 in-bounds 裁剪；这里该前提失效，必须先统计所有目标像素
#       的全局 min/max 行列，再把原输入整体嵌入扩展后的画布。
#     - 原 clear-then-paste 顺序仍然需要；否则目的地若穿过源位置，会被自身旧像素阻挡。
#     - 新增前提是：反向平移后的多个对象在绝对坐标系里不能碰撞，也不能贴到锚点旁边。生成器通过
#       输出态坐标集合上的 8 邻域排斥来保证这一点。
#   placement_legality:
#     - 输入态所有对象完整位于原网格内部，且彼此及锚点保持至少 1 格背景间隔。
#     - 输出态允许越界，但越界只发生在“相对原框”的意义上；对象之间仍不得重叠或 8 邻接。
#     - 构造 expected output 时显式保留整个原输入矩形，再在发生越界的方向扩张。
#   visual_observability:
#     - 人类仍能先用“marker 颜色 = 锚点颜色”配对对象。
#     - 输出中对象不会对齐到锚点，而是出现在锚点反侧，并且至少一例会把画布撑大。
#     - 这两个信号会在同一网格里重复出现：同色多精灵共享同一方向规则，扩张方向也直接可见。
#   neatness:
#     - 扩张幅度被限制在小范围内，避免画布过度稀疏。
#     - 可选干扰物只用单像素，且和主精灵一样遵守同轴反向滑动，不引入额外规则层。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker-tagged sprites away from matching border anchors and enlarge the canvas to keep them intact.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are sprites; multi-color ones contain a rare marker pixel, and isolated interior singletons may also act as tiny sprites.
    - Isolated non-background border pixels are anchors; each anchor color matches the marker color of the sprites it controls.

    Rule:
    - Match each movable component to the anchor with the same marker color.
    - A top/bottom anchor still means horizontal motion and a left/right anchor still means vertical motion.
    - The move has the same magnitude as the original task but the opposite sign, so the sprite slides away from the anchor instead of onto it.
    - Any translated pixels that would leave the old frame are preserved by expanding the output grid on the needed side.

    Key insight:
    - The anchor still chooses the axis, but now it is only a distance reference; after mirroring that offset away from the anchor, the solver must render on a dynamically enlarged canvas rather than clipping to the old bounds.
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
            dc = int(mp[1] - ac)
        else:
            dr = int(mp[0] - ar)

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
_MIN_SLIDE_DISTANCE = 2
_MAX_SLIDE_DISTANCE = 8
_MAX_OVERFLOW_PER_SIDE = 5
_CANDIDATE_POOL_LIMIT = 18

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

_OVERFLOW_SPRITE_PROB = 0.6
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


def _variant_output_marker(anchor, marker_r, marker_c):
    if anchor["axis"] == "h":
        return int(marker_r), int(2 * int(marker_c) - int(anchor["c"]))
    return int(2 * int(marker_r) - int(anchor["r"])), int(marker_c)


def _overflow_amount(cells, H, W):
    rows = [int(r) for r, _c, _v in cells]
    cols = [int(c) for _r, c, _v in cells]
    return {
        "top": max(0, -min(rows)),
        "left": max(0, -min(cols)),
        "bottom": max(0, max(rows) - (int(H) - 1)),
        "right": max(0, max(cols) - (int(W) - 1)),
    }


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


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _paired_positions(inp, sprite, anchor, solved_occupied, bg, require_overflow):
    H, W = inp.shape
    rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    random.shuffle(rows)
    random.shuffle(cols)
    pairs = []

    for r in rows:
        for c in cols:
            if not _can_place(inp, sprite, r, c, bg):
                continue
            dist = abs(int(c) - int(anchor["c"])) if anchor["axis"] == "h" else abs(int(r) - int(anchor["r"]))
            if dist < _MIN_SLIDE_DISTANCE or dist > _MAX_SLIDE_DISTANCE:
                continue
            out_r, out_c = _variant_output_marker(anchor, r, c)
            out_cells = _sprite_cells(sprite, out_r, out_c, bg)
            overflow = _overflow_amount(out_cells, H, W)
            has_overflow = any(overflow.values())
            if require_overflow is not None and has_overflow != bool(require_overflow):
                continue
            if max(overflow.values()) > _MAX_OVERFLOW_PER_SIDE:
                continue
            if not _candidate_clear(solved_occupied, out_cells):
                continue
            score = (
                int(has_overflow),
                max(overflow.values()),
                int(dist),
            )
            pairs.append(
                {
                    "input_marker": (int(r), int(c)),
                    "output_marker": (int(out_r), int(out_c)),
                    "output_cells": out_cells,
                    "overflow": overflow,
                    "distance": int(dist),
                    "score": score,
                }
            )

    if not pairs:
        return None

    pairs.sort(key=lambda item: item["score"], reverse=True)
    trimmed = pairs[: min(len(pairs), _CANDIDATE_POOL_LIMIT)]
    return random.choice(trimmed)


def _place_anchors(inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(inp, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_sprite_pairs(inp, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    solved_occupied = {(int(a["r"]), int(a["c"])) for a in anchors}
    placed = []
    has_overflow = False

    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)

        wants_overflow = (not has_overflow) or (random.random() < _OVERFLOW_SPRITE_PROB)
        pair = _paired_positions(inp, sprite, anchor, solved_occupied, bg, require_overflow=wants_overflow)
        if pair is None:
            pair = _paired_positions(inp, sprite, anchor, solved_occupied, bg, require_overflow=None)
        if pair is None:
            return None

        _place(inp, sprite, pair["input_marker"][0], pair["input_marker"][1], bg)
        for r, c, _v in pair["output_cells"]:
            solved_occupied.add((int(r), int(c)))
        pair["anchor"] = anchor
        pair["sprite"] = sprite
        pair["kind"] = "sprite"
        placed.append(pair)
        has_overflow = has_overflow or any(pair["overflow"].values())

    return placed if has_overflow else None


def _add_distractors(inp, anchors, placed, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return True

    solved_occupied = {(int(a["r"]), int(a["c"])) for a in anchors}
    for item in placed:
        for r, c, _v in item["output_cells"]:
            solved_occupied.add((int(r), int(c)))

    movable_anchors = anchors[:]
    random.shuffle(movable_anchors)
    if not movable_anchors:
        return True

    count = random.randint(*_DISTRACTOR_COUNT_RANGE)
    added = 0
    for _ in range(count):
        anchor = random.choice(movable_anchors)
        sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
        pair = _paired_positions(inp, sprite, anchor, solved_occupied, bg, require_overflow=None)
        if pair is None:
            break
        _place(inp, sprite, pair["input_marker"][0], pair["input_marker"][1], bg)
        for r, c, _v in pair["output_cells"]:
            solved_occupied.add((int(r), int(c)))
        pair["anchor"] = anchor
        pair["sprite"] = sprite
        pair["kind"] = "singleton"
        placed.append(pair)
        added += 1
    return added > 0 or True


def _render_expected_output(inp_shape, anchors, placed, bg):
    H, W = inp_shape
    min_r, min_c = 0, 0
    max_r, max_c = int(H - 1), int(W - 1)

    for item in placed:
        rows = [int(r) for r, _c, _v in item["output_cells"]]
        cols = [int(c) for _r, c, _v in item["output_cells"]]
        min_r = min(int(min_r), min(rows))
        min_c = min(int(min_c), min(cols))
        max_r = max(int(max_r), max(rows))
        max_c = max(int(max_c), max(cols))

    row_off = int(-min_r)
    col_off = int(-min_c)
    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)

    for anchor in anchors:
        out[row_off + int(anchor["r"]), col_off + int(anchor["c"])] = int(anchor["color"])
    for item in placed:
        for r, c, v in item["output_cells"]:
            out[row_off + int(r), col_off + int(c)] = int(v)
    return out


def _generate_case():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    inp = np.full((H, W), int(bg), dtype=int)

    placed_anchors = _place_anchors(inp, bg)
    if placed_anchors is None:
        return None
    anchor_colors, anchors = placed_anchors

    marker_colors = _marker_color_sequence(anchor_colors)
    placed = _place_sprite_pairs(inp, anchors, marker_colors, bg)
    if placed is None:
        return None

    _add_distractors(inp, anchors, placed, bg)
    out = _render_expected_output(inp.shape, anchors, placed, bg)

    if out.shape == inp.shape:
        return None
    if not _bg_is_mode(inp, bg):
        return None
    if not np.array_equal(main(inp), out):
        return None

    overflow_count = sum(1 for item in placed if any(item["overflow"].values()))
    max_expand = max(max(item["overflow"].values()) for item in placed)
    return {
        "input": inp,
        "output": out,
        "anchors": anchors,
        "placed": placed,
        "bg": int(bg),
        "overflow_count": int(overflow_count),
        "max_expand": int(max_expand),
    }


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid combo variant for 581f7754")
