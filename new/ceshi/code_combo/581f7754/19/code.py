# Combo variant of 581f7754 (index 19)
# combo_spec: D13/1+D17/2
# Name: 双背景分区+图例定义颜色配对

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The base task used one global background and direct color equality:
#    a sprite marker color 4 matched the unique border anchor color 4.
#    This combo changes two links in that chain.
#    First, the board is split into left/right background regions with different colors.
#    Second, a corner legend encodes anchor->marker pairs, so sprites carry the legend's
#    marker colors rather than the anchor colors themselves.
#    Example: left bg=1, right bg=8, legend rows are [2,7] and [5,2]. Then a left-half
#    sprite whose minority pixel is 7 must move to the left-half anchor colored 2, while
#    a right-half sprite with the same marker 7 must move to the right-half anchor colored 2.
#    Relative to the original task:
#    - original solver would wrongly try 7->7 directly;
#    - legend-only logic would decode 7->2 but could still confuse left/right copies of 2;
#    - partition-only logic would stay within the correct half but would still try 7->7.
#    The fused rule is therefore "decode marker through legend, then choose the same-region
#    anchor of that decoded color, then use edge side to choose slide axis".
# b) Complexity calibration:
#    I keep the legend small and regular (two mapping rows in a 4-column framed block) and
#    force exactly two globally shared anchor colors. Those same two anchor colors appear once
#    in each region, so the repeated evidence is strong enough for a human to see that both
#    halves reuse the same symbolic mapping. Sprite count is 4-5, with one mandatory sprite
#    for each (region, anchor-color) pair, so neither dimension can disappear in a sample.
# c) Hidden assumptions broken by the combo:
#    - One global background no longer exists: every "is this bg?" check must become region-aware.
#    - Foreground extraction cannot treat the second background as sprite pixels.
#    - Color equality no longer identifies the anchor; marker colors must be translated via legend.
#    - A repeated anchor color is no longer globally unique; matching needs (region, anchor color).
#    - The corner legend is a non-moving multicolor foreground component and must be excluded from
#      sprite processing and fake-anchor detection.
#    - Clearing moved sprites must restore the local region background, not one global color.
# d) Placement legality in the generator:
#    Generation is output-first. I lay down the two backgrounds, then the corner legend, then
#    four anchors (two colors x two regions), then aligned sprites inside their own regions,
#    and finally scatter those sprites along the inverse axis to form the input. Region-aware
#    placement rejects overlap, region crossing, border touching, and contact with the legend.
# e) Visual observability:
#    The split is a large rectangular left/right color partition; the legend is always visible
#    in one corner; the same anchor colors appear in both halves; and the sprite marker colors
#    come from the legend's right column rather than the anchors themselves. Across 2-4 samples,
#    a human can see both that "same marker color can live in both halves" and that the border
#    anchor color is obtained indirectly from the legend.
# f) Visual cleanliness:
#    The legend is compact, anchored to a corner, and isolated by the existing non-touch rule.
#    Body colors avoid both backgrounds, the legend frame, and all semantic anchor/marker colors,
#    which keeps the minority marker pixel visually salient. Objects stay within one half, so the
#    partition remains easy to parse.
# g) Inferability of the color rule:
#    The legend uses only two rows, each repeated through multiple sprites, and the mapping is a
#    derangement (never identity). No rule depends on numeric color order or invisible metadata:
#    region membership is visible from the background split, and color translation is visible from
#    the legend rows themselves.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior sprite to the same-region border anchor named by the corner legend.

    Input structure:
    - The board is divided into left and right background regions with different colors.
    - One corner contains a framed legend whose inner rows encode [anchor color, marker color].
    - Outside the legend, isolated border pixels are anchors and interior sprites carry the marker colors from the legend.

    Rule:
    - Detect the left/right partition and decode the legend into marker -> anchor color pairs.
    - For each sprite, use its own region plus its marker color to identify the matching border anchor in that same region.
    - Top/bottom anchors cause horizontal sliding to align columns; left/right anchors cause vertical sliding to align rows.

    Key insight: Neither color cue is sufficient alone: the legend translates marker colors into anchor colors, and the repeated anchor colors are disambiguated by the sprite's background partition.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    layout = _detect_partition(g)

    fg_mask = _foreground_mask(g, layout)
    comps = components_mask(fg_mask, connectivity=8)
    legend_cells, marker_to_anchor = _decode_legend(g, comps, layout)

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors_by_key: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c) or (int(r), int(c)) in legend_cells:
                continue
            v = int(g[r, c])
            if v == _bg_at(layout, r, c):
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
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_key.values()}

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

    moves: list[tuple[np.ndarray, int, int]] = []

    for coords in comps:
        if coords.size == 0:
            continue
        if all((int(r), int(c)) in legend_cells for r, c in coords):
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            region = _region_id(layout, r, c)
            if v == _bg_at(layout, r, c) or v not in marker_to_anchor or (r, c) in anchor_cells:
                continue
            if (region, int(marker_to_anchor[int(v)])) not in anchors_by_key:
                continue
            marker_col = int(v)

        if int(marker_col) not in marker_to_anchor:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        region = _region_id(layout, mp[0], mp[1])
        anchor_col = int(marker_to_anchor[int(marker_col)])
        key = (int(region), int(anchor_col))
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
_ACTIVE_COLORS = tuple(range(1, 10))
_LEGEND_FRAME_COLOR = 0
_LEGEND_WIDTH = 4

_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (14, 22)
_MIN_REGION_WIDTH = 6

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COLOR_COUNT = 2
_SPRITE_COUNT_RANGE = (4, 5)
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
    return 0 if int(c) < int(layout["split"]) else 1


def _region_bounds(shape, layout, region):
    H, W = shape
    split = int(layout["split"])
    if int(region) == 0:
        return 0, 0, int(H - 1), int(split - 1)
    return 0, int(split), int(H - 1), int(W - 1)


def _bg_at(layout, r, c):
    return int(layout["bgs"][_region_id(layout, int(r), int(c))])


def _fill_partition_background(canvas, layout):
    bg0, bg1 = map(int, layout["bgs"])
    split = int(layout["split"])
    canvas[:, :split] = bg0
    canvas[:, split:] = bg1


def _bg_grid(shape, layout):
    H, W = shape
    bg = np.empty((H, W), dtype=int)
    _fill_partition_background(bg, layout)
    return bg


def _foreground_mask(canvas, layout):
    return canvas != _bg_grid(canvas.shape, layout)


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
    H, _W = canvas.shape
    _r0, c0, _r1, c1 = _region_bounds(canvas.shape, layout, region)
    rows = _segment_coords(1, H - 2)
    cols = _segment_coords(c0 + 1, c1 - 1)
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    edge_col = 0 if int(region) == 0 else int(canvas.shape[1] - 1)
    for r in rows:
        out.append(("v", r, edge_col))
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


def _scatter_positions(canvas, sprite, anchor, marker_pos, layout, region):
    mr, mc = marker_pos
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


def _body_color(layout, used):
    reserved = {int(_LEGEND_FRAME_COLOR), *map(int, layout["bgs"]), *map(int, layout["anchor_colors"]), *map(int, layout["marker_colors"])}
    pool = [c for c in _ACTIVE_COLORS if c not in reserved and c not in used]
    if not pool:
        pool = [c for c in _ACTIVE_COLORS if c not in reserved]
    return random.choice(pool)


def _partition_is_visible(grid, layout):
    top_two = [c for c, _n in Counter(map(int, grid.ravel())).most_common(2)]
    return len(top_two) == 2 and set(top_two) == set(map(int, layout["bgs"]))


def _decode_legend(g, comps, layout):
    H, W = g.shape
    corners = {(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))}
    legend_coords = None
    for coords in comps:
        cells = {(int(r), int(c)) for r, c in coords}
        if any(cell in corners for cell in cells):
            if legend_coords is None or int(coords.shape[0]) > int(legend_coords.shape[0]):
                legend_coords = coords
    assert legend_coords is not None

    legend_cells = {(int(r), int(c)) for r, c in legend_coords}
    r0 = int(legend_coords[:, 0].min())
    c0 = int(legend_coords[:, 1].min())
    r1 = int(legend_coords[:, 0].max())
    c1 = int(legend_coords[:, 1].max())
    legend = g[r0 : r1 + 1, c0 : c1 + 1]
    frame = int(legend[0, 0])

    anchor_to_marker: dict[int, int] = {}
    for rr in range(1, int(legend.shape[0]) - 1):
        if int(legend.shape[1]) < 3:
            continue
        anchor_col = int(legend[int(rr), 1])
        marker_col = int(legend[int(rr), 2])
        if anchor_col in {int(frame), *map(int, layout["bgs"])}:
            continue
        if marker_col in {int(frame), *map(int, layout["bgs"])}:
            continue
        anchor_to_marker[int(anchor_col)] = int(marker_col)

    marker_to_anchor = {int(marker): int(anchor) for anchor, marker in anchor_to_marker.items()}
    assert marker_to_anchor
    return legend_cells, marker_to_anchor


def _detect_partition(grid):
    g = np.array(grid, dtype=int)
    _H, W = g.shape
    colors = sorted(set(map(int, g.ravel())))
    best = None
    for split in range(4, int(W - 3)):
        left = g[:, :split]
        right = g[:, split:]
        for b0 in colors:
            for b1 in colors:
                if int(b0) == int(b1):
                    continue
                wrong = int((left == int(b1)).sum() + (right == int(b0)).sum())
                support = int((left == int(b0)).sum() + (right == int(b1)).sum())
                score = (wrong, -support)
                cand = {"split": int(split), "bgs": (int(b0), int(b1))}
                if best is None or score < best[0]:
                    best = (score, cand)
    assert best is not None
    return best[1]


def _swap_mapping(anchor_colors):
    assert len(anchor_colors) == 2
    return {int(anchor_colors[0]): int(anchor_colors[1]), int(anchor_colors[1]): int(anchor_colors[0])}


def _build_legend(anchor_to_marker):
    legend = np.full((len(anchor_to_marker) + 2, _LEGEND_WIDTH), int(_LEGEND_FRAME_COLOR), dtype=int)
    for i, (anchor_col, marker_col) in enumerate(anchor_to_marker.items(), start=1):
        legend[int(i), 1] = int(anchor_col)
        legend[int(i), 2] = int(marker_col)
    return legend


def _choose_layout():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    split = random.randint(_MIN_REGION_WIDTH, W - _MIN_REGION_WIDTH)
    bgs = tuple(random.sample(list(range(_BG_COLOR_RANGE[0], _BG_COLOR_RANGE[1] + 1)), 2))
    return H, W, {"split": int(split), "bgs": tuple(map(int, bgs))}


def _legend_origins(H, W, legend, layout):
    h, w = legend.shape
    origins = []
    if int(layout["split"]) >= int(w):
        origins.append((0, 0))
        origins.append((H - h, 0))
    if int(W - layout["split"]) >= int(w):
        origins.append((0, W - w))
        origins.append((H - h, W - w))
    return origins


def _choose_semantic_colors(layout):
    pool = [c for c in _ACTIVE_COLORS if c not in set(map(int, layout["bgs"])) and c != int(_LEGEND_FRAME_COLOR)]
    anchor_colors = random.sample(pool, _ANCHOR_COLOR_COUNT)
    anchor_to_marker = _swap_mapping(anchor_colors)
    layout["anchor_colors"] = tuple(map(int, anchor_colors))
    layout["marker_colors"] = tuple(int(anchor_to_marker[int(c)]) for c in anchor_colors)
    return anchor_to_marker


def _place_anchors(out, inp, layout):
    anchors = []
    for region in (0, 1):
        colors = list(layout["anchor_colors"])
        random.shuffle(colors)
        assignments = list(zip(colors, ("h", "v")))
        random.shuffle(assignments)
        for color, required_axis in assignments:
            candidates = [
                (axis, r, c)
                for axis, r, c in _anchor_candidates(out, layout, region)
                if axis == required_axis
            ]
            if not candidates:
                return None
            axis, r, c = random.choice(candidates)
            anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis, "region": int(region)}
            anchors.append(anchor)
            _put_anchor(out, r, c, color)
            _put_anchor(inp, r, c, color)
    return anchors


def _sprite_plan(layout):
    total = random.randint(*_SPRITE_COUNT_RANGE)
    plan = [(region, int(color)) for region in (0, 1) for color in layout["anchor_colors"]]
    while len(plan) < total:
        plan.append((random.choice([0, 1]), int(random.choice(layout["anchor_colors"]))))
    random.shuffle(plan)
    return plan


def _place_aligned_sprites(out, anchors, anchor_to_marker, layout):
    by_key = {(a["region"], a["color"]): a for a in anchors}
    used_body = set()
    sprites = []
    for region, anchor_color in _sprite_plan(layout):
        anchor = by_key[(int(region), int(anchor_color))]
        bg = int(layout["bgs"][int(region)])
        marker = int(anchor_to_marker[int(anchor_color)])
        body = _body_color(layout, used_body)
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


def _generated_grid_is_valid(inp, out, layout):
    if not _partition_is_visible(inp, layout):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    H, W, seed_layout = _choose_layout()
    anchor_to_marker = _choose_semantic_colors(seed_layout)
    legend = _build_legend(anchor_to_marker)
    if int(seed_layout["split"]) < legend.shape[1] and int(W - seed_layout["split"]) < legend.shape[1]:
        return None

    layout = {"split": int(seed_layout["split"]), "bgs": seed_layout["bgs"], "anchor_colors": seed_layout["anchor_colors"], "marker_colors": seed_layout["marker_colors"]}
    out = np.zeros((H, W), dtype=int)
    inp = np.zeros((H, W), dtype=int)
    _fill_partition_background(out, layout)
    _fill_partition_background(inp, layout)

    origins = _legend_origins(H, W, legend, layout)
    if not origins:
        return None
    r0, c0 = random.choice(origins)
    paste(out, legend, r0, c0)
    paste(inp, legend, r0, c0)
    layout["legend_origin"] = (int(r0), int(c0))

    anchors = _place_anchors(out, inp, layout)
    if anchors is None:
        return None

    sprites = _place_aligned_sprites(out, anchors, anchor_to_marker, layout)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, layout):
        return None

    if not _generated_grid_is_valid(inp, out, layout):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
