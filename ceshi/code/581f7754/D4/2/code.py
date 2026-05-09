# Variant of 581f7754
# Dimension: D4 — 目标位置：锚点类型与位置 (dst)
# Change: 边界上的孤立单像素（非角落），4-邻域均为背景色，1-3个不同颜色锚点... → 成对锚点（同色双锚点）

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D4 paired-anchor variant
# a) Concrete behavior trace:
#    - Each anchor color now appears twice, once on each side of one opposite-edge pair:
#      top+bottom for a horizontal-slide color, or left+right for a vertical-slide color.
#    - The two same-color anchors are intentionally offset (different columns for top/bottom,
#      different rows for left/right), so selecting one anchor changes the destination.
#    - Example: red anchors at (0,2) and (11,8), red marker pixel at (4,7).
#      Old logic kept only the first red anchor found in the border scan, so it always slid
#      to column 2. New logic compares Manhattan distances:
#      d_top = |4-0| + |7-2| = 9, d_bottom = |4-11| + |7-8| = 8, so the sprite must slide
#      horizontally to column 8 instead. The changed branch is visible in the output.
# b) Complexity calibration:
#    - The canvas size, sprite sizes, and body construction stay close to the source task.
#    - To make the variant legible from a few examples, the generator forces at least one
#      repeated marker color per grid and splits that color across both anchors of its pair.
#    - Opposite anchors of the same color are separated by at least 2 cells on the varying
#      coordinate, so "first anchor wins" cannot accidentally match the paired-anchor result.
# c) Implicit assumptions audited in the original main():
#    - `anchors_by_color[v] = (r,c)` assumed one anchor per color. This fails immediately for
#      paired anchors and is replaced with `anchors_by_color[v] = [(r,c), ...]`.
#    - `anchor_cells` previously stored one cell per color; it now tracks every anchor cell so
#      border anchors are never mistaken for movable singleton sprites.
#    - Translation used the only anchor position for a color. The new solver must choose the
#      nearest same-color anchor before deriving the slide axis from that chosen anchor.
# d) Placement legality:
#    - The generator still uses output-first construction. It places solved sprites first,
#      then scatters them along the inverse axis.
#    - Scatter candidates are filtered so the intended target anchor is the unique nearest
#      same-color anchor from the input marker position. This guarantees solver/generator
#      agreement and prevents accidental ties.
#    - Isolation rules from the source task are preserved: sprites stay inside the inner area,
#      anchors remain isolated border singletons, and objects do not touch.
# e) Visual observability:
#    - A human can see two same-color border anchors and infer that identical marker colors may
#      land at different output columns/rows depending on which anchor is closer.
#    - Because at least one color is reused and split across both anchors, the "choose nearer"
#      branch is usually demonstrated within a single grid instead of relying only on cross-grid
#      comparison.
# f) Visual cleanliness:
#    - The generator keeps the original object scale and spacing, avoids overlapping solved
#      destinations, and does not add extra layered effects beyond occasional singleton distractors.
#    - Same-color anchor pairs are few (1-3 colors), keeping the border readable.
# g) Color inference:
#    - Colors are only used as visible equality cues between marker and anchors.
#    - No branch depends on color-id ordering. The only geometric decision is nearest-anchor
#      choice by border location, which is visible from the grids themselves.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites toward the nearer member of each same-color border-anchor pair.

    Input structure:
    - bg = most common color.
    - Interior sprites are multi-color connected components with a minority-color marker pixel.
    - Each marker color appears as two isolated border anchors on opposite edges.

    Rule:
    - Match each sprite to the two border anchors sharing its marker color.
    - Compare the marker pixel's Manhattan distance to those same-color anchors and choose the nearer one.
    - If the chosen anchor is on the top/bottom edge, slide horizontally so the marker column matches that anchor column; if it is on the left/right edge, slide vertically so the marker row matches that anchor row.

    Key insight:
    - Color no longer determines a unique destination by itself; the sprite's current marker position decides which same-color border anchor controls the translation.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def anchor_priority(pos: tuple[int, int]) -> tuple[int, int]:
        r, c = map(int, pos)
        if r == 0:
            return (0, c)
        if c == 0:
            return (1, r)
        if c == W - 1:
            return (2, r)
        return (3, c)

    # --- STEP 1) Detect anchors: isolated border pixels (non-bg). ---
    anchors_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
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
            if ok:
                anchors_by_color[int(v)].append((int(r), int(c)))

    assert anchors_by_color
    for color in list(anchors_by_color):
        anchors_by_color[color].sort(key=anchor_priority)

    anchor_cells = {
        (int(r), int(c))
        for positions in anchors_by_color.values()
        for (r, c) in positions
    }

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
        cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    def nearest_anchor(marker_pos: tuple[int, int], color: int) -> tuple[int, int]:
        mr, mc = map(int, marker_pos)
        return min(
            anchors_by_color[int(color)],
            key=lambda pos: (
                abs(mr - int(pos[0])) + abs(mc - int(pos[1])),
                anchor_priority(pos),
            ),
        )

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

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = nearest_anchor(mp, int(marker_col))

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


# --------------- generator (paired-anchor D4 variant) ---------------

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
_GEN_RETRY_LIMIT = 400
_PAIR_OFFSET_MIN = 2
_SCATTER_MOVE_MIN = 2
_NEAREST_MARGIN_MIN = 1

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


def _anchor_span(axis, anchor_a, anchor_b):
    if axis == "h":
        return abs(int(anchor_a[1]) - int(anchor_b[1]))
    return abs(int(anchor_a[0]) - int(anchor_b[0]))


def _edge_anchor_candidates(canvas, bg, axis, side):
    H, W = canvas.shape
    if axis == "h":
        r = 0 if side == "top" else H - 1
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
        cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
        random.shuffle(cols)
        return [(r, c) for c in cols if _anchor_ok(canvas, r, c, bg)]

    c = 0 if side == "left" else W - 1
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    random.shuffle(rows)
    return [(r, c) for r in rows if _anchor_ok(canvas, r, c, bg)]


def _pair_for_axis(canvas, bg, axis):
    side_a, side_b = ("top", "bottom") if axis == "h" else ("left", "right")
    cands_a = _edge_anchor_candidates(canvas, bg, axis, side_a)
    cands_b = _edge_anchor_candidates(canvas, bg, axis, side_b)
    if not cands_a or not cands_b:
        return None

    good = []
    weak = []
    for pos_a in cands_a:
        for pos_b in cands_b:
            span = _anchor_span(axis, pos_a, pos_b)
            if span >= _PAIR_OFFSET_MIN:
                good.append((pos_a, pos_b))
            elif span > 0:
                weak.append((pos_a, pos_b))
    pairs = good or weak
    if not pairs:
        return None
    pos_a, pos_b = random.choice(pairs)
    if axis == "h":
        return [
            {"r": int(pos_a[0]), "c": int(pos_a[1]), "axis": "h", "side": "top"},
            {"r": int(pos_b[0]), "c": int(pos_b[1]), "axis": "h", "side": "bottom"},
        ]
    return [
        {"r": int(pos_a[0]), "c": int(pos_a[1]), "axis": "v", "side": "left"},
        {"r": int(pos_b[0]), "c": int(pos_b[1]), "axis": "v", "side": "right"},
    ]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _anchors_by_color(anchors):
    grouped = defaultdict(list)
    side_order = {"top": 0, "left": 0, "bottom": 1, "right": 1}
    for anchor in anchors:
        grouped[int(anchor["color"])].append(anchor)
    for color in grouped:
        grouped[color].sort(key=lambda a: (side_order[a["side"]], int(a["r"]), int(a["c"])))
    return grouped


def _anchor_distance(marker, anchor):
    return abs(int(marker[0]) - int(anchor["r"])) + abs(int(marker[1]) - int(anchor["c"]))


def _nearest_anchor_info(marker, anchors):
    ranked = sorted(
        anchors,
        key=lambda a: (
            _anchor_distance(marker, a),
            0 if a["side"] in {"top", "left"} else 1,
            int(a["r"]),
            int(a["c"]),
        ),
    )
    best = ranked[0]
    best_d = _anchor_distance(marker, best)
    second_d = _anchor_distance(marker, ranked[1]) if len(ranked) > 1 else 10**9
    return best, int(second_d - best_d)


def _same_anchor(a, b):
    return int(a["r"]) == int(b["r"]) and int(a["c"]) == int(b["c"])


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


def _scatter_positions(canvas, sprite, anchor_pair, target_anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = map(int, marker)
    strong = []
    weak = []
    if target_anchor["axis"] == "h":
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc or not _can_place(canvas, sprite, mr, c, bg):
                continue
            best, margin = _nearest_anchor_info((mr, c), anchor_pair)
            if not _same_anchor(best, target_anchor) or margin < _NEAREST_MARGIN_MIN:
                continue
            item = (mr, c)
            if abs(c - mc) >= _SCATTER_MOVE_MIN:
                strong.append(item)
            else:
                weak.append(item)
        return strong or weak

    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        if r == mr or not _can_place(canvas, sprite, r, mc, bg):
            continue
        best, margin = _nearest_anchor_info((r, mc), anchor_pair)
        if not _same_anchor(best, target_anchor) or margin < _NEAREST_MARGIN_MIN:
            continue
        item = (r, mc)
        if abs(r - mr) >= _SCATTER_MOVE_MIN:
            strong.append(item)
        else:
            weak.append(item)
    return strong or weak


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
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchor_pairs(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        axes = ["h", "v"]
        random.shuffle(axes)
        pair = None
        for axis in axes:
            pair = _pair_for_axis(out, bg, axis)
            if pair is not None:
                break
        if pair is None:
            return None
        for anchor in pair:
            anchor["color"] = int(color)
            anchors.append(anchor)
            _put_anchor(out, anchor["r"], anchor["c"], color)
            _put_anchor(inp, anchor["r"], anchor["c"], color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    min_sprites = max(_SPRITE_COUNT_RANGE[0], len(anchor_colors) + 1)
    max_sprites = _SPRITE_COUNT_RANGE[1]
    if min_sprites > max_sprites:
        min_sprites = max_sprites
    n_sprites = random.randint(min_sprites, max_sprites)
    marker_colors = list(anchor_colors)
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _sprite_plans(anchors, marker_colors):
    grouped_anchors = _anchors_by_color(anchors)
    color_counts = Counter(map(int, marker_colors))
    plans = []
    split_colors = 0
    for color, count in color_counts.items():
        pair = grouped_anchors[int(color)]
        if len(pair) < 2:
            return None, 0
        targets = []
        if count == 1:
            targets = [random.choice(pair)]
        else:
            split_colors += 1
            targets = [pair[0], pair[1]]
            while len(targets) < count:
                targets.append(random.choice(pair))
            random.shuffle(targets)
        for target in targets:
            plans.append({"marker": int(color), "pair": pair, "target": target})
    random.shuffle(plans)
    return plans, split_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    plans, split_colors = _sprite_plans(anchors, marker_colors)
    if plans is None or split_colors == 0:
        return None

    used_body = set()
    sprites = []
    used_second_anchor = False
    for plan in plans:
        marker = int(plan["marker"])
        pair = plan["pair"]
        target = plan["target"]
        if _same_anchor(target, pair[1]):
            used_second_anchor = True
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, target, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, pair, target, marker_pos))

    if not used_second_anchor:
        return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, pair, target, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, pair, target, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _add_distractors(out, inp, anchors, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return
    grouped = _anchors_by_color(anchors)
    color = random.choice(list(grouped))
    pair = grouped[int(color)]
    sprite = {"patch": np.array([[int(color)]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        target = random.choice(pair)
        solved_positions = _aligned_positions(out, sprite, target, bg)
        if not solved_positions:
            return
        marker_pos = random.choice(solved_positions)
        scatter = _scatter_positions(inp, sprite, pair, target, marker_pos, bg)
        if not scatter:
            return
        in_pos = random.choice(scatter)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchor_pairs(out, inp, bg)
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


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-anchor 581f7754 input")
