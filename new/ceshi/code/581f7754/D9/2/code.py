# Variant of 581f7754
# Dimension: D9 — 精灵内部标记识别规则 (meta)
# Change: 标记色为精灵内部出现次数最少的颜色（少数色），仅1个像素... → 标记为唯一不与其他精灵共享的颜色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis
# variant:
#   core_change:
#     original_marker_rule: "在每个精灵内部找出现次数最少的颜色。"
#     new_marker_rule: "先收集所有精灵的颜色集合；每个精灵里恰有一种颜色不出现在其他任何精灵中，这个跨精灵唯一色就是 marker。"
#   concrete_visualization:
#     - 保持原题的整体骨架：边界上有孤立锚点，内部有若干多色精灵，输出仍是沿单轴平移使 marker 对齐同色锚点。
#     - 为了让 D9 真正可见，每个精灵都共享两种内部颜色：
#       * shared_major: 大部分 body 像素。
#       * shared_accent: 恰好 1 个 body 像素。
#     - 第三种颜色 unique_marker 只出现在某一个精灵中，并且也只占 1 个像素；它与一个边界锚点同色。
#     - 因此每个精灵的颜色集合形如 {shared_major, shared_accent, unique_marker_i}。
#       只有 unique_marker_i 需要跨精灵比较后才能认出。
#   behavior_trace:
#     - 原逻辑在一个三色精灵里会统计频次；若 shared_accent 和 unique_marker 都只出现 1 次，
#       它只能靠旧的 tie-break 规则随意选一个，已经不再可靠。
#     - 新逻辑先看所有精灵：
#       sprite A colors = {2, 4, 7}
#       sprite B colors = {2, 4, 8}
#       sprite C colors = {2, 4, 9}
#       于是 7/8/9 各自只属于一个精灵，而 2/4 被所有精灵共享。
#       marker 分别是 7/8/9，再与同色锚点匹配并平移。
#   complexity_calibration:
#     - 若仍允许 1-3 个锚点、3-5 个精灵，则“marker 唯一且与锚点同色”会迫使多个精灵共用 marker 色，和新规则冲突。
#     - 因此改成 3-4 个锚点且精灵数 = 锚点数，保持一一对应；这样跨精灵比较既必要，又不会产生歧义。
#     - 保留原来的多种 body 几何，以免所有样例都像模板复制。
#   implicit_assumptions_review:
#     - 原 main 中 component_marker_color() 依赖 “marker = least frequent color”；该前提失效，必须改成跨组件比较。
#     - 原 main 中对单色 singleton distractor 的处理依赖旧任务允许“同锚点色的小点也一起移动”；
#       新变体若保留它们，会引入不参与颜色集合比较的组件，反而稀释 D9，因此去掉。
#     - 原 main 的 marker_pixel() 能处理多 marker 像素；新生成器保证 unique_marker 恰好 1 像素，但 solver 仍保留稳健回退。
#     - 原 anchors_by_color 只保存每种颜色的一个锚点；新生成器保证 marker 色与锚点一一对应，所以该假设仍成立。
#   placement_legality:
#     - 所有精灵仍是 8 连通、与其他对象 8 邻域隔离、离边界至少 1 格。
#     - 输出先放对齐态，再沿反轴散开得到输入；每一步都用 _can_place() 防重叠、防接触、防越界。
#   observability:
#     - 人类看 2-4 组样例时，会发现所有精灵都共享两种内部颜色，而每个精灵额外多出一种不同颜色；
#       这个“额外颜色”恰好与某个边界锚点同色，并且对齐时是拿这个像素去对准锚点。
#     - 规则不依赖颜色 id 的可见外属性；颜色 id 只被生成器拿来挑选一组稳定的共享色/锚点色，便于反制旧 solver 的退化。
#   cleanliness:
#     - 取消 distractor，避免单像素组件与真正 marker 混杂。
#     - 共享 accent 固定为 body 中的单点，视觉上能看出“每个精灵都有一个共同的小彩点 + 一个各自特有的小彩点”。
#   generator_guarantees_needed_by_main:
#     - 每个非锚点前景连通块都是多色精灵。
#     - 每个精灵恰有 1 种颜色只出现在自己身上，且该颜色恰有 1 个像素，并且有同色锚点。
#     - 除 marker 色外，其余 sprite 颜色至少被另一个精灵共享。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior multi-color sprites so that each sprite's globally unique color lines up with a same-color border anchor.

    Input structure:
    - The background is the most frequent color.
    - Border anchors are isolated non-background pixels on the outer frame.
    - Interior sprites are multi-color connected components; each sprite shares some colors with the other sprites and also contains one color that appears in no other sprite.

    Rule:
    - For each sprite, identify the color that is unique across the full set of sprites and match that sprite to the border anchor of the same color.
    - If the anchor sits on the top or bottom edge, slide the sprite horizontally until the unique-color pixel reaches the anchor's column.
    - If the anchor sits on the left or right edge, slide the sprite vertically until the unique-color pixel reaches the anchor's row.

    Key insight:
    - The marked color is not determined inside one sprite by frequency; it only becomes identifiable after comparing color sets across sprites and finding the one color owned by exactly one sprite.
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

    sprites: list[dict[str, object]] = []
    color_to_sprite_ids: dict[int, list[int]] = {}
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) == 1 and pts[0] in anchor_cells:
            continue
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) <= 1:
            continue
        idx = len(sprites)
        sprites.append({"coords": coords, "pts": pts, "colors": colors})
        for color in colors:
            color_to_sprite_ids.setdefault(int(color), []).append(idx)

    def marker_pixel(pts: list[tuple[int, int]], marker: int) -> tuple[int, int] | None:
        marker_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(marker)]
        if not marker_pts:
            return None
        if len(marker_pts) == 1:
            return marker_pts[0]
        pt_set = set(pts)
        for r, c in marker_pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in pt_set and int(g[rr, cc]) != int(marker):
                    return (int(r), int(c))
        return marker_pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []
    for sprite in sprites:
        coords = sprite["coords"]
        pts = sprite["pts"]
        colors = sprite["colors"]
        marker_candidates = [
            int(color)
            for color in colors
            if len(color_to_sprite_ids.get(int(color), [])) == 1 and int(color) in anchors_by_color
        ]
        if len(marker_candidates) != 1:
            continue

        marker_col = int(marker_candidates[0])
        mp = marker_pixel(pts, marker_col)
        if mp is None:
            continue

        ar, ac = anchors_by_color[marker_col]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))
        dr, dc = 0, 0
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


# --------------- generator (D9 variant) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (10, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (3, 4)
_BODY_AREA_RANGE = (4, 10)
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


def _accent_cell(body):
    borderish = []
    for r, c in body:
        if any((r + dr, c + dc) not in body for dr, dc in _NEIGHBORS4):
            borderish.append((r, c))
    return random.choice(borderish or list(body))


def _make_sprite(area, major_color, accent_color, marker_color, bg):
    body = _make_body(area)
    accent = _accent_cell(body)
    marker = _marker_cell(body)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(major_color)
    patch[accent[0] - r0, accent[1] - c0] = int(accent_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "marker_color": int(marker_color),
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


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


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


def _choose_palette(bg):
    available = [c for c in sorted(_COLORS) if c != int(bg)]
    shared_accent = int(available[0])
    shared_major = int(available[1])
    marker_pool = [int(c) for c in available[2:]]
    n_sprites = random.randint(*_ANCHOR_COUNT_RANGE)
    marker_colors = random.sample(marker_pool, n_sprites)
    return shared_major, shared_accent, marker_colors


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


def _place_anchors(out, inp, marker_colors, bg):
    anchors = []
    for color in marker_colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchors


def _place_aligned_sprites(out, anchors, shared_major, shared_accent, bg):
    sprites = []
    order = anchors[:]
    random.shuffle(order)
    for anchor in order:
        sprite = _make_sprite(
            random.randint(*_BODY_AREA_RANGE),
            shared_major,
            shared_accent,
            anchor["color"],
            bg,
        )
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


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    shared_major, shared_accent, marker_colors = _choose_palette(bg)
    anchors = _place_anchors(out, inp, marker_colors, bg)
    if anchors is None:
        return None

    sprites = _place_aligned_sprites(out, anchors, shared_major, shared_accent, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
        return None

    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D9 input")
