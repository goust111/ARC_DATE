# Variant of 581f7754
# Dimension: D15 — 锚点边缘约束 (meta)
# Change: 锚点必须位于边界行/列上，且不在角落；4-邻域全部为背景色（孤立）... → 锚点不需要孤立（可相邻其他非背景像素）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# -----------------------------------------------------------------------------
# Design analysis for D15
#
# a) 具体化与行为追踪
# - 原题把锚点做成“孤立的边界单像素”，solver 因而直接扫描边界并要求该像素的 4 邻域全是背景。
# - 这个变体保留“锚点仍然是单个、非角落的边界像素，颜色仍与某个 marker 对应”，但在其旁边
#   放入静态非背景装饰像素。装饰可在同一条边上，也可在边界内侧一格。这样 input/output 中都会
#   出现“边界簇”，其中只有一个像素是实际锚点，其余只是静态陪衬。
# - 小例子：
#   原逻辑：顶边 `... 4 ...` 且其左右/下方都是背景，颜色 4 被认作锚点，marker=4 的精灵水平对齐到该列。
#   新逻辑：顶边变成 `... 7 4 ...` 或 `... 4 ...` 且 `(1,c)` 还有一个装饰色 7。颜色 4 依旧是边界锚点，
#   但已经不满足“孤立”。solver 必须改成：先从内部精灵读出 marker 颜色集合，再去边界上找到这些颜色
#   的边界像素，用其所在边决定滑动轴。
#
# b) 复杂度校准
# - 不把“整段边框”都做成候选锚点，否则颜色和结构都会过乱；仍坚持每种锚点颜色只对应一个边界像素。
# - 非孤立性通过每个锚点旁边 1 个静态装饰像素体现，复杂度只增加在“检测标准”，不改变匹配关系本身。
# - 锚点颜色数仍是 1-3，sprite 数仍是 3-5，确保同一颜色会重复支持，多样例可归纳。
#
# c) 隐式假设审查
# - 原 `main()` 第 1 步假设：锚点 = 边界上的非背景像素且 4 邻居全背景。这个前提在 D15 下失效，必须删除。
# - 原 `main()` 在组件循环里没有跳过“触边组件”，因为原题触边组件只有孤立锚点，不会被误移动。
#   现在边界会出现多像素装饰簇，若不显式跳过触边组件，就可能把边框装饰当成多色 sprite 去移动。
# - 原单色 singleton 逻辑依赖 `anchor_cells` 只包含孤立锚点；现在仍可保留，但 `anchor_cells` 要来自
#   “颜色匹配到的边界像素”，而不是孤立性检测。
#
# d) 放置合法性
# - generator 仍采用 output-first：先放锚点和边界装饰，再放已对齐的 sprite，最后沿反轴打散到 input。
# - 所有 sprite 仍保持在内部，不接触边界，因此边界装饰永远不会与可移动组件连成一体。
# - 装饰像素同时复制到 input/output，属于静态元素；solver 通过“跳过触边组件”保证不会移动它们。
#
# e) 视觉可观察性
# - 每个样本都强制至少一个锚点非孤立；这里更进一步，默认给每个锚点都加一个邻接装饰像素。
# - 人类解题者会直接看到：边界上不再只有孤零零的彩点，而是出现小型边界簇，但真正决定移动的仍是
#   与 sprite marker 同色的那一个边界像素。
#
# f) 可视化整洁度
# - 每个锚点只配一个装饰像素，避免边界噪声过多。
# - 装饰颜色不使用锚点颜色，避免同色多边界像素造成“哪个才是锚点”的额外歧义。
#
# g) 颜色变化可推断性
# - 本变体不引入颜色映射。颜色仍只承担“marker 与 anchor 对应”的可见配对关系。
# - 静态装饰颜色没有任何变换语义，只是为了破坏孤立性，因此不能与锚点颜色重合。
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their marker pixels line up with matching border anchors, even when the anchors sit inside border clutter.

    Input structure:
    - The background is the most common color.
    - The border contains one non-corner anchor pixel for each marker color; those anchor pixels may touch other non-background border decorations.
    - The interior contains scattered multi-color sprites and optional singleton movers whose color matches an anchor.

    Rule:
    - For each movable interior component, use its marker color to find the border pixel of the same color.
    - A top/bottom-edge anchor makes the sprite slide horizontally to match the anchor column; a left/right-edge anchor makes it slide vertically to match the anchor row.
    - Border decorations stay fixed; moved pixels clip at the frame and never overwrite existing non-background cells.

    Key insight:
    - Isolation is no longer the signal for anchors. The stable cue is that marker colors point to unique border pixels, and the edge containing that border pixel determines the slide axis.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def is_corner(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) and int(c) in (0, int(W - 1))

    def touches_border(coords: np.ndarray) -> bool:
        return any(is_border(int(r), int(c)) for r, c in coords)

    # --- STEP 1) Read candidate marker colors from interior multi-color sprites. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    candidate_anchor_colors: set[int] = set()
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or touches_border(coords):
            continue
        marker_col = component_marker_color(coords)
        if marker_col is not None:
            candidate_anchor_colors.add(int(marker_col))

    # --- STEP 2) Locate anchors by border position, not by isolation. ---
    border_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c) or is_corner(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            border_by_color.setdefault(int(v), []).append((int(r), int(c)))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for color in sorted(candidate_anchor_colors):
        pts = border_by_color.get(int(color), [])
        if pts:
            anchors_by_color[int(color)] = pts[0]

    assert anchors_by_color

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}

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

    # --- STEP 3) Keep only interior movable components and assign each to its anchor color. ---
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or touches_border(coords):
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

        # --- STEP 4) Compute translation from the anchor's edge. ---
        dr = 0
        dc = 0
        if int(ar) in (0, int(H - 1)):
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
_BORDER_DECOR_SIDE_PROB = 0.75

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 300

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


def _anchor_slot_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return int(canvas[r, c]) == int(bg)


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
    return [(axis, r, c) for axis, r, c in out if _anchor_slot_ok(canvas, r, c, bg)]


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
            for r in range(_INNER_BORDER_GAP, out.shape[0] - _INNER_BORDER_GAP):
                if r != anchor["r"] and _can_place(inp, sprite, r, c, bg):
                    pairs.append((r, c))
        if not pairs:
            return
        in_r, col = random.choice(pairs)
        _place(out, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)


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

    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg, anchor_colors):
        return None
    return inp


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _anchor_neighbor_choices(anchor, shape):
    H, W = shape
    r, c = int(anchor["r"]), int(anchor["c"])
    side = []
    inward = []
    if anchor["axis"] == "h":
        for dc in (-1, 1):
            cc = c + dc
            if 0 < cc < W - 1:
                side.append((r, cc))
        inward.append((1 if r == 0 else H - 2, c))
    else:
        for dr in (-1, 1):
            rr = r + dr
            if 0 < rr < H - 1:
                side.append((rr, c))
        inward.append((r, 1 if c == 0 else W - 2))
    return side, inward


def _decorate_anchors(out, inp, anchors, bg):
    anchor_colors = {int(a["color"]) for a in anchors}
    decor_pool = [c for c in _COLORS if c != int(bg) and c not in anchor_colors]
    if not decor_pool:
        return False

    for anchor in anchors:
        side, inward = _anchor_neighbor_choices(anchor, out.shape)
        side = [p for p in side if int(out[p]) == int(bg)]
        inward = [p for p in inward if int(out[p]) == int(bg)]
        if side and random.random() < _BORDER_DECOR_SIDE_PROB:
            pos = random.choice(side)
        elif side or inward:
            pos = random.choice(side + inward)
        else:
            return False
        color = random.choice(decor_pool)
        _put_anchor(out, pos[0], pos[1], color)
        _put_anchor(inp, pos[0], pos[1], color)
    return True


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
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
    if not _decorate_anchors(out, inp, anchors, bg):
        return None
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
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _count_nonisolated_anchors(grid, anchor_colors, bg):
    H, W = grid.shape
    counts = 0
    for color in anchor_colors:
        pts = []
        for r in range(H):
            for c in range(W):
                if int(grid[r, c]) != int(color):
                    continue
                if r in (0, H - 1) or c in (0, W - 1):
                    if not (r in (0, H - 1) and c in (0, W - 1)):
                        pts.append((int(r), int(c)))
        if len(pts) != 1:
            return -1
        r, c = pts[0]
        if any(0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg) for rr, cc in neighbors4(r, c)):
            counts += 1
    return counts


def _generated_grid_is_valid(inp, out, bg, anchor_colors):
    if not _bg_is_mode(inp, bg):
        return False
    if _count_nonisolated_anchors(inp, anchor_colors, bg) != len(anchor_colors):
        return False
    return bool(np.array_equal(main(inp), out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D15 input")
