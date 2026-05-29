# Variant of 581f7754
# Dimension: D2 — 匹配条件：滑动轴决定规则 (match_cond)
# Change: 锚点在上/下边缘（非角落）→水平滑动对齐列；锚点在左/右边缘（或角落）→垂直滑动对齐行
# ... → 颜色编码决定轴

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D2 variant
# ----------------------------------
# spec_variant:
#   preserved_core:
#     - 背景色仍是全图众数。
#     - 边界上仍有若干孤立锚点；内部仍有 3-5 个双色精灵；marker 仍是组件里最少数的颜色。
#     - 精灵仍通过平移与同色锚点配对，原位置清空，越界像素裁剪，已有非背景目的地不覆盖。
#   changed_core:
#     - 原题用“锚点在哪条边”决定滑动轴。
#     - 本变体改成“锚点颜色属于暖色/冷色”决定滑动轴：
#       warm = {red=2, yellow=4, orange=7} -> horizontal slide -> 对齐锚点列
#       cool = {blue=1, green=3, purple-ish=6} -> vertical slide -> 对齐锚点行
#   concrete_behavior_trace:
#     - 原逻辑例子：左边界一个红色锚点，红色 marker 精灵会竖直滑动，让 marker 行号等于锚点行号。
#     - 新逻辑例子：同样的左边界红色锚点，红色属于暖色，因此精灵改为水平滑动，让 marker 列号等于锚点列号。
#       因为该列在边框上，generator 会把 marker 放在靠左侧，平移后 marker 正好贴到边界锚点位置，主体留在网格内。
#     - 差异：同一个“左边界锚点”，旧题看边，变体看颜色；输入外观相似，但输出位移方向改变。
#   complexity_calibration:
#     - 只使用 6 种可见的锚点颜色，且明确分成 3 暖 / 3 冷两组，避免隐藏的 id 规则。
#     - 每个样例至少同时出现一枚暖色锚点和一枚冷色锚点，并强制让它们落在与旧规则相反的边上，
#       这样人类能直接看到“边不再决定方向”。
#     - 保留重复 marker 颜色与多精灵配置，避免规则只靠单个对象支撑。
#   implicit_assumptions_audit:
#     - 原 main() 中唯一与 D2 直接耦合的是“根据锚点坐标判断 horizontal/vertical”的分支；现改成颜色查表。
#     - component_marker_color() 仍依赖 marker 是组件内少数色；generator 继续保证每个精灵只有 1 个 marker。
#     - marker_pixel() 仍假设 marker 4 邻接主体；generator 保持该结构，并在需要贴边时让 marker 位于朝向边界的一侧。
#     - moves 仍假设每种颜色至多一个锚点；generator 继续保证锚点颜色互异。
#   placement_legality:
#     - 输入态精灵仍完全位于内部，彼此及锚点间保留 1 格背景隔离。
#     - 输出态若锚点与新轴“冲突”，允许 marker 与自身锚点重合在边框位置，但不允许与其他前景重叠。
#     - 为避免冲突边例导致主体大量裁剪，generator 定向选择 marker 朝向边界，使主体尽量留在网格内。
#   visual_observability:
#     - 暖色锚点故意放到左/右边，冷色锚点故意放到上/下边，直接反驳旧规则。
#     - 颜色分组只依赖人眼可见的暖/冷色差异，不依赖数值排序。
#   cleanliness:
#     - 关闭原本的 singleton distractor，避免在“贴边隐藏 marker”的样例中出现完全消失的小点，降低可读性。


_WARM_COLORS = frozenset({2, 4, 7})
_COOL_COLORS = frozenset({1, 3, 6})
_AXIS_BY_COLOR = {color: "h" for color in _WARM_COLORS} | {color: "v" for color in _COOL_COLORS}


def _axis_from_color(color: int) -> str | None:
    return _AXIS_BY_COLOR.get(int(color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift interior sprites until each marker lines up with its same-colored border anchor.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are mostly two-color sprites: a larger body plus a rarer marker pixel.
    - Matching anchors are isolated single pixels on the border and reuse the marker colors.

    Rule:
    - Pair every sprite with the border anchor of the same marker color.
    - Warm anchor colors (red / yellow / orange) mean the sprite slides horizontally, so the marker must share the anchor's column.
    - Cool anchor colors (blue / green / purple-like) mean the sprite slides vertically, so the marker must share the anchor's row.

    Key insight: The anchor still chooses the sprite by color, but the slide axis now comes from the anchor color family rather than from which border edge the anchor sits on.
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
        coord_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in coord_set and int(g[int(rr), int(cc)]) != int(marker):
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

        axis = _axis_from_color(int(marker_col))
        if axis is None:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
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


# --------------- generator (o5x) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (4, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_SPRITE_RETRY_LIMIT = 40

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

_DISTRACTOR_PROB = 0.0
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


def _marker_cell(body, desired_side=None):
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

    if desired_side is not None:
        if desired_side == "left":
            side_pool = [p for p in exterior if p[1] < c0]
        elif desired_side == "right":
            side_pool = [p for p in exterior if p[1] > c1]
        elif desired_side == "top":
            side_pool = [p for p in exterior if p[0] < r0]
        else:
            side_pool = [p for p in exterior if p[0] > r1]
        if side_pool:
            return random.choice(side_pool)

    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg, desired_side=None):
    body = _make_body(area)
    marker = _marker_cell(body, desired_side=desired_side)
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


def _patch_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for pr, pc in np.argwhere(patch != int(bg)):
        cells.append((top + int(pr), left + int(pc), int(patch[int(pr), int(pc)])))
    return cells


def _neighbor_coords(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r) + dr, int(c) + dc


def _can_place(canvas, sprite, marker_r, marker_c, bg, border_gap=1, allow_contact=()):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < border_gap or left < border_gap:
        return False
    if top + h > H - border_gap or left + w > W - border_gap:
        return False

    allowed = {(int(r), int(c)) for r, c in allow_contact}
    placed = {(r, c): v for r, c, v in _patch_cells(sprite, marker_r, marker_c, bg)}

    for (r, c), v in placed.items():
        cell = int(canvas[r, c])
        if cell != int(bg):
            if (r, c) not in allowed or cell != int(v):
                return False

    for r, c in placed:
        for rr, cc in _neighbor_coords(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if int(canvas[rr, cc]) == int(bg):
                continue
            if (rr, cc) in allowed:
                continue
            return False
    return True


def _place(canvas, sprite, marker_r, marker_c, bg, overwrite=True):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), overwrite=overwrite)


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


def _edge_family(edge):
    return "tb" if edge in ("top", "bottom") else "lr"


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append({"edge": "top", "r": 0, "c": int(c)})
        out.append({"edge": "bottom", "r": H - 1, "c": int(c)})
    for r in rows:
        out.append({"edge": "left", "r": int(r), "c": 0})
        out.append({"edge": "right", "r": int(r), "c": W - 1})
    random.shuffle(out)
    return [cand for cand in out if _anchor_ok(canvas, cand["r"], cand["c"], bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _desired_marker_side(anchor):
    if anchor["axis"] == "h" and anchor["edge"] in ("left", "right"):
        return anchor["edge"]
    if anchor["axis"] == "v" and anchor["edge"] in ("top", "bottom"):
        return anchor["edge"]
    return None


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    allow_contact = {(anchor["r"], anchor["c"])} if _desired_marker_side(anchor) else set()
    border_gap = 0 if allow_contact else _INNER_BORDER_GAP
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg, border_gap=border_gap, allow_contact=allow_contact)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg, border_gap=border_gap, allow_contact=allow_contact)
    ]


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg, border_gap=_INNER_BORDER_GAP)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg, border_gap=_INNER_BORDER_GAP)
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
    del out, inp, anchors, bg
    return


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


def _pick_anchor_colors(bg):
    warm_pool = [c for c in _WARM_COLORS if c != int(bg)]
    cool_pool = [c for c in _COOL_COLORS if c != int(bg)]
    if not warm_pool or not cool_pool:
        return None

    n = random.randint(*_ANCHOR_COUNT_RANGE)
    picked = [random.choice(warm_pool), random.choice(cool_pool)]
    remaining = [c for c in (_WARM_COLORS | _COOL_COLORS) if c != int(bg) and c not in picked]
    if len(picked) < n:
        if len(remaining) < n - len(picked):
            return None
        picked.extend(random.sample(remaining, n - len(picked)))
    random.shuffle(picked)
    return picked


def _preferred_family(color, forced_mismatch):
    axis = _axis_from_color(color)
    if axis == "h":
        return "lr" if forced_mismatch else "tb"
    return "tb" if forced_mismatch else "lr"


def _place_anchors(out, inp, bg):
    anchor_colors = _pick_anchor_colors(bg)
    if anchor_colors is None:
        return None

    anchors = []
    warm_mismatch_used = False
    cool_mismatch_used = False
    for idx, color in enumerate(anchor_colors):
        axis = _axis_from_color(color)
        forced_mismatch = (color in _WARM_COLORS and not warm_mismatch_used) or (
            color in _COOL_COLORS and not cool_mismatch_used
        )
        preferred_family = _preferred_family(color, forced_mismatch=forced_mismatch)
        fallbacks = [preferred_family, "tb" if preferred_family == "lr" else "lr"]

        anchor = None
        for family in fallbacks:
            candidates = [cand for cand in _anchor_candidates(out, bg) if _edge_family(cand["edge"]) == family]
            if not candidates:
                continue
            cand = random.choice(candidates)
            anchor = {
                "color": int(color),
                "r": int(cand["r"]),
                "c": int(cand["c"]),
                "edge": cand["edge"],
                "axis": axis,
            }
            break
        if anchor is None:
            return None

        if color in _WARM_COLORS and _edge_family(anchor["edge"]) == "lr":
            warm_mismatch_used = True
        if color in _COOL_COLORS and _edge_family(anchor["edge"]) == "tb":
            cool_mismatch_used = True

        anchors.append(anchor)
        _put_anchor(out, anchor["r"], anchor["c"], color)
        _put_anchor(inp, anchor["r"], anchor["c"], color)

    if not warm_mismatch_used or not cool_mismatch_used:
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
        desired_side = _desired_marker_side(anchor)
        body = _body_color(bg, marker, used_body)
        used_body.add(body)

        placed = None
        for _ in range(_SPRITE_RETRY_LIMIT):
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg, desired_side=desired_side)
            positions = _aligned_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg, overwrite=False)
            placed = (sprite, anchor, marker_pos)
            break
        if placed is None:
            return None
        sprites.append(placed)
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


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D2 input")
