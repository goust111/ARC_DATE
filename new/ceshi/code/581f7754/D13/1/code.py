# Variant of 581f7754
# Dimension: D13 — 背景色选择规则 (meta)
# Change: 背景色为网格中出现最多的颜色，生成时从1-8中随机选择... → 双背景色（分区背景）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, neighbors4, paste


# Design analysis for the D13 variant
# a) Concrete behavior:
#    The canvas is now split into two rectangles: either left/right or top/bottom.
#    Each rectangle has its own background color, and all sprites plus anchors stay
#    fully inside one rectangle. The motion rule itself stays the same as 581f7754:
#    a top/bottom anchor means horizontal sliding, and a left/right anchor means
#    vertical sliding. The crucial new twist is that anchor colors can repeat across
#    the two rectangles, so color alone no longer selects the destination anchor.
#    Small example: if both the left and right halves each contain a border anchor
#    of color 4, a sprite with marker 4 in the left half must align to the left-half
#    anchor, while another sprite with marker 4 in the right half must align to the
#    right-half anchor. The original logic would collapse both into one global
#    color→anchor map; the variant requires region→(color→anchor).
# b) Complexity calibration:
#    The generator guarantees both partitions are used and that at least one marker
#    color is shared across them. This keeps the new rule visible in 2-3 examples
#    without adding many unrelated cases. Anchor count stays close to the original
#    (2-3 instead of 1-3) and sprite count stays 3-5.
# c) Implicit assumptions from the original solver that break here:
#    - `get_bg(...)` returning one background no longer works.
#    - `find_components(..., bg=bg)` would wrongly treat the second background as
#      foreground.
#    - Anchor detection cannot compare neighbors to one global bg.
#    - Clearing moved sprites cannot write one color everywhere; each vacated cell
#      must restore the background of its own partition.
#    - Matching by `anchors_by_color[color]` is no longer valid because the same
#      anchor color may appear once per partition.
# d) Placement legality:
#    The generator builds the partition first, then places anchors and sprites inside
#    region-specific bounds with a one-cell gap to borders and to the partition cut.
#    Sprites are aligned in the output, scattered along the inverse axis in the
#    input, and every sample is accepted only if `main(input) == output`.
# e) Visual observability:
#    The partition is a large rectangular color block split, not a hidden metadata
#    flag. The same marker color appears in both partitions, so a human solver can
#    see that “same color does not imply same destination unless the region matches”.
# f) Visual cleanliness:
#    Objects never cross the partition, anchors remain isolated singletons, and
#    sprites keep the original spacing constraints. This keeps the two zones legible.
# g) Color inference:
#    No invisible ordering by color id is used. Color reuse across partitions is
#    supported by visible spatial structure only: region membership plus edge side.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward the border anchor that matches both their marker color and their background partition.

    Input structure:
    - The board is divided into two rectangular background regions, either left/right or top/bottom, each with its own dominant color.
    - Multi-color sprites live entirely inside one region; each sprite has a least-frequent marker pixel.
    - Border anchors are isolated non-background singletons, and the same anchor color may appear once in each region.

    Rule:
    - First detect the partition, then assign every sprite to its own region.
    - Match a sprite to the anchor of the same marker color inside that same region.
    - Top/bottom anchors align the marker's column by horizontal sliding; left/right anchors align the marker's row by vertical sliding.

    Key insight: marker color is no longer a global key because colors can repeat across the two background regions; region membership plus anchor side determines the move.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    layout = _detect_partition(g)

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors_by_key: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v in layout["bgs"]:
                continue
            region = _region_id(layout, r, c)
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != _bg_at(layout, rr, cc):
                    ok = False
                    break
            if ok and (region, v) not in anchors_by_key:
                anchors_by_key[(int(region), int(v))] = (int(r), int(c))

    assert anchors_by_key

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_key.values()}
    fg_mask = ~np.isin(g, list(layout["bgs"]))
    comps = [{"coords": coords} for coords in components_mask(fg_mask, connectivity=8)]

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
            region = _region_id(layout, r, c)
            if v in layout["bgs"] or (region, v) not in anchors_by_key or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        region = _region_id(layout, mp[0], mp[1])
        key = (int(region), int(marker_col))
        if key not in anchors_by_key:
            continue

        ar, ac = anchors_by_key[key]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        for r, c in coords:
            out[int(r), int(c)] = _bg_at(layout, int(r), int(c))

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == _bg_at(layout, rr, cc):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_MIN_REGION_SPAN = 4

_ANCHOR_COUNT_RANGE = (2, 3)
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


def _segment_coords(start: int, end: int) -> list[int]:
    if start > end:
        return []
    lo = int(start) + int(_ANCHOR_EDGE_MARGIN) - 1
    hi = int(end) - int(_ANCHOR_EDGE_MARGIN) + 1
    if lo <= hi:
        return list(range(lo, hi + 1))
    return list(range(int(start), int(end) + 1))


def _region_id(layout, r, c):
    if layout["orientation"] == "v":
        return 0 if int(c) < int(layout["split"]) else 1
    return 0 if int(r) < int(layout["split"]) else 1


def _region_bounds(shape, layout, region):
    H, W = shape
    split = int(layout["split"])
    if layout["orientation"] == "v":
        if int(region) == 0:
            return 0, 0, int(H - 1), int(split - 1)
        return 0, int(split), int(H - 1), int(W - 1)
    if int(region) == 0:
        return 0, 0, int(split - 1), int(W - 1)
    return int(split), 0, int(H - 1), int(W - 1)


def _bg_at(layout, r, c):
    return int(layout["bgs"][_region_id(layout, int(r), int(c))])


def _foreground_mask(canvas, layout):
    return ~np.isin(canvas, list(layout["bgs"]))


def _touches_foreground(canvas, mask, top, left, layout):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    fg = _foreground_mask(canvas, layout)
    occupied = fg[top - radius : top + h + radius, left - radius : left + w + radius]
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas, sprite, marker_r, marker_c, layout, region):
    patch = sprite["patch"]
    bg = int(layout["bgs"][int(region)])
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    r0, c0, r1, c1 = _region_bounds(canvas.shape, layout, region)
    if top < r0 + _INNER_BORDER_GAP or left < c0 + _INNER_BORDER_GAP:
        return False
    if top + h - 1 > r1 - _INNER_BORDER_GAP or left + w - 1 > c1 - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, layout)


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _isolated_bg_cell(canvas, r, c, layout):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if int(canvas[rr, cc]) != _bg_at(layout, rr, cc):
                return False
    return True


def _anchor_ok(canvas, r, c, layout, region):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    if _region_id(layout, r, c) != int(region):
        return False
    return _isolated_bg_cell(canvas, r, c, layout)


def _anchor_candidates(canvas, layout, region):
    H, W = canvas.shape
    r0, c0, r1, c1 = _region_bounds(canvas.shape, layout, region)
    rows = _segment_coords(r0 + 1, r1 - 1)
    cols = _segment_coords(c0 + 1, c1 - 1)
    out = []
    if layout["orientation"] == "v":
        for c in cols:
            out.append(("h", 0, c))
            out.append(("h", H - 1, c))
        edge_col = 0 if int(region) == 0 else int(W - 1)
        for r in rows:
            out.append(("v", r, edge_col))
    else:
        for r in rows:
            out.append(("v", r, 0))
            out.append(("v", r, W - 1))
        edge_row = 0 if int(region) == 0 else int(H - 1)
        for c in cols:
            out.append(("h", edge_row, c))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, layout, region)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _region_inner_rows(shape, layout, region):
    r0, _c0, r1, _c1 = _region_bounds(shape, layout, region)
    return range(int(r0 + _INNER_BORDER_GAP), int(r1 - _INNER_BORDER_GAP + 1))


def _region_inner_cols(shape, layout, region):
    _r0, c0, _r1, c1 = _region_bounds(shape, layout, region)
    return range(int(c0 + _INNER_BORDER_GAP), int(c1 - _INNER_BORDER_GAP + 1))


def _aligned_positions(canvas, sprite, anchor, layout, region):
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in _region_inner_rows(canvas.shape, layout, region)
            if _can_place(canvas, sprite, r, anchor["c"], layout, region)
        ]
    return [
        (anchor["r"], c)
        for c in _region_inner_cols(canvas.shape, layout, region)
        if _can_place(canvas, sprite, anchor["r"], c, layout, region)
    ]


def _scatter_positions(canvas, sprite, anchor, marker, layout, region):
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in _region_inner_cols(canvas.shape, layout, region)
            if c != mc and _can_place(canvas, sprite, mr, c, layout, region)
        ]
    return [
        (r, mc)
        for r in _region_inner_rows(canvas.shape, layout, region)
        if r != mr and _can_place(canvas, sprite, r, mc, layout, region)
    ]


def _body_color(bgs, marker, used):
    banned = {int(marker), *map(int, bgs)}
    pool = [c for c in _COLORS if c not in banned and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in banned]
    return random.choice(pool)


def _partition_is_visible(grid, layout):
    top_two = [c for c, _n in Counter(map(int, grid.ravel())).most_common(2)]
    return len(top_two) == 2 and set(top_two) == set(map(int, layout["bgs"]))


def _fill_partition_background(canvas, layout):
    bg0, bg1 = map(int, layout["bgs"])
    split = int(layout["split"])
    if layout["orientation"] == "v":
        canvas[:, :split] = bg0
        canvas[:, split:] = bg1
    else:
        canvas[:split, :] = bg0
        canvas[split:, :] = bg1


def _detect_partition(grid):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    colors = [c for c, _n in Counter(map(int, g.ravel())).most_common(2)]
    assert len(colors) == 2
    best = None
    for orientation in ("v", "h"):
        limit = W if orientation == "v" else H
        for split in range(3, limit - 2):
            for b0, b1 in (colors, colors[::-1]):
                if orientation == "v":
                    left = g[:, :split]
                    right = g[:, split:]
                    wrong = int((left == int(b1)).sum() + (right == int(b0)).sum())
                    support = int((left == int(b0)).sum() + (right == int(b1)).sum())
                else:
                    top = g[:split, :]
                    bottom = g[split:, :]
                    wrong = int((top == int(b1)).sum() + (bottom == int(b0)).sum())
                    support = int((top == int(b0)).sum() + (bottom == int(b1)).sum())
                score = (wrong, -support)
                if best is None or score < best[0]:
                    best = (score, {"orientation": orientation, "split": int(split), "bgs": (int(b0), int(b1))})
    assert best is not None
    return best[1]


def _choose_layout():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    orientation = random.choice(["v", "h"])
    limit = W if orientation == "v" else H
    split = random.randint(_MIN_REGION_SPAN, limit - _MIN_REGION_SPAN)
    bgs = tuple(random.sample(list(range(_BG_COLOR_RANGE[0], _BG_COLOR_RANGE[1] + 1)), 2))
    return H, W, {"orientation": orientation, "split": int(split), "bgs": tuple(map(int, bgs))}


def _choose_anchor_colors(layout):
    pool = [c for c in _COLORS if c not in set(map(int, layout["bgs"]))]
    shared = random.choice(pool)
    extras = [c for c in pool if c != int(shared)]
    random.shuffle(extras)
    total = random.randint(*_ANCHOR_COUNT_RANGE)
    region_colors = {0: [int(shared)], 1: [int(shared)]}
    for _ in range(total - 2):
        region = random.choice([0, 1])
        color = extras.pop()
        while color in region_colors[region]:
            color = extras.pop()
        region_colors[region].append(int(color))
    return region_colors


def _new_canvas_pair():
    H, W, layout = _choose_layout()
    out = np.zeros((H, W), dtype=int)
    inp = np.zeros((H, W), dtype=int)
    _fill_partition_background(out, layout)
    _fill_partition_background(inp, layout)
    return layout, out, inp


def _place_anchors(out, inp, layout):
    region_colors = _choose_anchor_colors(layout)
    anchors = []
    order = [(region, color) for region in (0, 1) for color in region_colors[region]]
    random.shuffle(order)
    for region, color in order:
        candidates = _anchor_candidates(out, layout, region)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis, "region": int(region)}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return region_colors, anchors


def _marker_plan(region_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    shared = random.choice(sorted(set(region_colors[0]) & set(region_colors[1])))
    plan = [(0, int(shared)), (1, int(shared))]
    while len(plan) < n_sprites:
        region = random.choice([0, 1])
        plan.append((int(region), int(random.choice(region_colors[region]))))
    random.shuffle(plan)
    return plan


def _place_aligned_sprites(out, anchors, marker_plan, layout):
    by_key = {(a["region"], a["color"]): a for a in anchors}
    used_body = set()
    sprites = []
    for region, marker in marker_plan:
        anchor = by_key[(int(region), int(marker))]
        bg = int(layout["bgs"][int(region)])
        body = _body_color(layout["bgs"], marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, anchor, layout, region)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos, int(region)))
    return sprites


def _scatter_sprites(inp, sprites, layout):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos, region in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, layout, region)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], int(layout["bgs"][int(region)]))
    return True


def _add_distractors(out, inp, anchors, layout):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    bg = int(layout["bgs"][int(anchor["region"])])
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in _region_inner_cols(out.shape, layout, anchor["region"]):
            if not _can_place(out, sprite, anchor["r"], c, layout, anchor["region"]):
                continue
            for r in _region_inner_rows(inp.shape, layout, anchor["region"]):
                if r != anchor["r"] and _can_place(inp, sprite, r, c, layout, anchor["region"]):
                    pairs.append((r, c))
        if not pairs:
            return
        in_r, col = random.choice(pairs)
        _place(out, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)


def _generated_grid_is_valid(inp, out, layout):
    if not _partition_is_visible(inp, layout):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    layout, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, layout)
    if placed is None:
        return None
    region_colors, anchors = placed

    marker_plan = _marker_plan(region_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_plan, layout)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, layout):
        return None

    _add_distractors(out, inp, anchors, layout)
    if not _generated_grid_is_valid(inp, out, layout):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D13 input")
