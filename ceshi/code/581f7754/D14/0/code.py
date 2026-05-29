# Variant of 581f7754
# Dimension: D14 — 滑动方向：单向vs双向 (meta)
# Change: 精灵只能沿一个轴的一个方向滑动，方向由标记到锚点的相对位置决定... → 精灵向锚点的反方向滑动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D14 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     original: 对于匹配到的锚点，精灵沿锚点规定的轴朝锚点滑动，平移量是 anchor_coord - marker_coord。
#     variant:  保持“轴由锚点所在边决定”的机制不变，但把平移方向取反，平移长度保持不变。
#               因此新的平移量是 marker_coord - anchor_coord。若锚点在上/下边，就只改列；若锚点
#               在左/右边，就只改行。
#   concrete_pixel_behavior:
#     - 顶/底边锚点仍然只控制水平移动。若锚点列为 3、某精灵标记列为 6，则原题输出会去 3；
#       本变体输出改为去 9，也就是同轴、同距离，但远离锚点。
#     - 左/右边锚点仍然只控制垂直移动。若锚点行为 4、标记行为 7，则原题输出会去 4；
#       本变体输出改为去 10。
#     - 这会让锚点始终留在精灵移动方向的反侧；锚点不再是落点，而只是“方向与距离参考”。
#   flow_impact_trace:
#     - 原始 solver 在 STEP 4 里把 dc 设为 ac - marker_c，或把 dr 设为 ar - marker_r。
#     - 本变体只在这里翻转符号：dc = marker_c - ac / dr = marker_r - ar。
#     - 原始 generator 先构造“对齐锚点”的输出，再把精灵散回输入；这隐含了“输出一定落在锚点列/行上”。
#     - 本变体里输出不再对齐锚点，所以 generator 改成 input-first：先放输入中的精灵，再检查它们按新规
#       则移动后的输出位置是否合法，并同步写入输出画布。
#   complexity_calibration:
#     - 保留 1-3 个边界锚点、3-5 个内部双色精灵、偶发单像素干扰物，避免任务被简化成单个对象演示。
#     - 仍允许多个精灵共享同一个锚点颜色，这样样例里会重复出现同一规则，利于人眼归纳“方向反了”。
#     - 生成器要求滑动距离至少为 2，避免零位移或几乎不动而看不出与原题差异。
#   implicit_assumption_audit:
#     - anchors_by_color: 仍假设每种锚点颜色唯一；生成器继续保证这一点。
#     - component_marker_color: 仍依赖“标记色是组件内最少颜色”；生成器继续只做单标记双色精灵。
#     - marker_pixel: 仍优先选贴着主体的少数色像素；生成器继续保证标记 4 邻接主体。
#     - horizontal 判定: 仍只由锚点是否位于顶/底边决定；本变体不改轴规则，所以假设仍成立。
#     - clear-then-paste: 仍假设移动后的目标位置若撞到已有前景就不覆盖；生成器在输出态避免冲突，
#       因而 solver 不依赖处理顺序。
#   placement_legality:
#     - 输入态和输出态都要求精灵完整落在内框里，且与其他前景保持至少 1 格背景隔离。
#     - 对每个候选输入位置，都显式计算按新规则得到的输出标记位置；只有输入可放、输出也可放时才接受。
#     - 这样保证 main(input) 不会因为重叠、越界或锚点贴邻而退化。
#   visual_observability:
#     - 锚点依旧落在边界，颜色与精灵标记一一对应，因此人仍能先发现“谁对应谁”。
#     - 输出中精灵不会停在锚点那一列/行，而是比输入更远离该锚点；多个同色精灵会重复展示这一反常方向。
#     - 不使用颜色 id 排序或任何不可见元信息；所有规则都来自可见的颜色匹配与空间距离。
#   cleanliness:
#     - 保留原题的小型不规则实体和稀疏布局，不引入遮挡层。
#     - 通过候选位置筛选维持对象分离，避免输出网格因为“远离锚点”而挤成一团。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites away from their matching isolated border anchors.

    Input structure:
    - bg is the most frequent color in the grid.
    - Interior multi-color sprites each contain a minority-color marker pixel.
    - Isolated non-bg border pixels serve as anchors, with anchor color matching the marker color of the sprite to move.

    Rule:
    - Pair each sprite with the border anchor of the same marker color.
    - A top/bottom anchor still selects horizontal motion and a left/right anchor still selects vertical motion.
    - However, the sprite moves the same distance in the opposite direction: instead of shifting the marker onto the anchor, it shifts away from the anchor by that offset.

    Key insight: The anchor still chooses the motion axis, but its coordinate is now a signed reference that must be mirrored into an equal-magnitude move away from the border cue.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

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

        # --- STEP 3) Decide marker color for this component. ---
        marker_col = component_marker_color(coords)
        if marker_col is None:
            # Single-color component: move only if it's a singleton of an anchor color, not the anchor itself.
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

        # --- STEP 4) Compute translation (dr,dc). ---
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(mp[1] - ac)
        else:
            dr = int(mp[0] - ar)

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


# --------------- generator ---------------

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
_MIN_SLIDE_DISTANCE = 2
_CANDIDATE_POOL_LIMIT = 18

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500

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


def _variant_output_marker(anchor, marker_r, marker_c):
    if anchor["axis"] == "h":
        return int(marker_r), int(2 * int(marker_c) - int(anchor["c"]))
    return int(2 * int(marker_r) - int(anchor["r"])), int(marker_c)


def _paired_positions(inp, out, sprite, anchor, bg):
    H, W = inp.shape
    pairs = []

    if anchor["axis"] == "h":
        rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
        cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
        random.shuffle(rows)
        random.shuffle(cols)
        for r in rows:
            for c in cols:
                dist = abs(int(c) - int(anchor["c"]))
                if dist < _MIN_SLIDE_DISTANCE:
                    continue
                out_r, out_c = _variant_output_marker(anchor, r, c)
                if not (_INNER_BORDER_GAP <= out_c < W - _INNER_BORDER_GAP):
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                if not _can_place(out, sprite, out_r, out_c, bg):
                    continue
                pairs.append(((int(r), int(c)), (int(out_r), int(out_c)), int(dist)))
    else:
        rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
        cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
        random.shuffle(rows)
        random.shuffle(cols)
        for c in cols:
            for r in rows:
                dist = abs(int(r) - int(anchor["r"]))
                if dist < _MIN_SLIDE_DISTANCE:
                    continue
                out_r, out_c = _variant_output_marker(anchor, r, c)
                if not (_INNER_BORDER_GAP <= out_r < H - _INNER_BORDER_GAP):
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                if not _can_place(out, sprite, out_r, out_c, bg):
                    continue
                pairs.append(((int(r), int(c)), (int(out_r), int(out_c)), int(dist)))

    if not pairs:
        return None
    pairs.sort(key=lambda item: item[2], reverse=True)
    trimmed = pairs[: min(len(pairs), _CANDIDATE_POOL_LIMIT)]
    return random.choice(trimmed)


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
        return []

    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    moved = []
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pair = _paired_positions(inp, out, sprite, anchor, bg)
        if pair is None:
            break
        inp_pos, out_pos, dist = pair
        _place(inp, sprite, inp_pos[0], inp_pos[1], bg)
        _place(out, sprite, out_pos[0], out_pos[1], bg)
        moved.append({"axis": anchor["axis"], "distance": int(dist), "color": int(anchor["color"])})
    return moved


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
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
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


def _place_sprite_pairs(out, inp, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        pair = _paired_positions(inp, out, sprite, anchor, bg)
        if pair is None:
            return None
        inp_pos, out_pos, dist = pair
        _place(inp, sprite, inp_pos[0], inp_pos[1], bg)
        _place(out, sprite, out_pos[0], out_pos[1], bg)
        sprites.append({"axis": anchor["axis"], "distance": int(dist), "color": int(marker)})
    return sprites


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _generate_case():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_sprite_pairs(out, inp, anchors, marker_colors, bg)
    if sprites is None:
        return None

    distractors = _add_distractors(out, inp, anchors, bg)
    if not sprites and not distractors:
        return None

    motions = sprites + distractors
    if not motions or max(item["distance"] for item in motions) < _MIN_SLIDE_DISTANCE:
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None

    return {"input": inp, "output": out, "anchors": anchors, "motions": motions, "bg": int(bg)}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid D14 variant for 581f7754")
