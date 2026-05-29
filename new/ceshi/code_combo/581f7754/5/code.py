# Combo variant of 581f7754 (index 5)
# combo_spec: D3/1+D9/0
# Name: 三色精灵+多数色标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
#    - Original 581f7754: a 2-color sprite had one tiny minority-color marker pixel. That same pixel both chose the
#      anchor by color and supplied the row/column that had to align after sliding.
#    - D3/1 adds a role split: the sprite now has 3 colors, with one color deciding which anchor to use and another
#      color deciding the alignment point.
#    - D9/0 inverts marker identification: the anchor-matching color is no longer the tiny odd pixel; it is the largest
#      visible region in the sprite.
#    - Fused realization used here:
#      * large majority-color region = match marker, and its color must equal a border anchor color
#      * one tucked singleton pixel of a third color = align marker, and its position is what must line up
#      * one small support tail of a second minority color keeps the sprite genuinely 3-colored and visibly asymmetric
#    - Example trace: suppose a sprite contains 7 red cells, 2 blue tail cells, and 1 yellow tucked cell. A top-edge
#      anchor is also red at column 9. The solver ignores the tiny blue/yellow colors for pairing, matches by the large
#      red region, then slides horizontally until the yellow singleton reaches column 9.
#
# b) Complexity calibration:
#    - Majority-vs-minority is visually strong because the anchor color occupies 6-12 cells while the two auxiliary
#      colors occupy only 2-3 cells and 1 cell.
#    - The align marker is always a singleton with at least two 4-neighbors in the majority region, so it looks tucked
#      into a notch instead of like another dangling nub.
#    - The support tail is attached through exactly one majority-contact point, which keeps the sprite 3-colored without
#      creating ambiguity about which minority color is the alignment cue.
#    - Several sprites share anchor colors, so the "majority color chooses anchor" rule repeats within a grid.
#
# c) Hidden-assumption audit for main():
#    - Original `component_marker_color(...)` assumed the least frequent color was the marker. That is invalid now; the
#      solver must require a unique most-frequent color instead.
#    - Original `marker_pixel(...)` assumed the same color used for matching also supplied the alignment pixel. That is
#      invalid now because matching uses the large region while alignment uses a different singleton color.
#    - The new `_sprite_roles(...)` assumes:
#      * exactly 3 non-bg colors per movable sprite
#      * one unique majority color
#      * exactly one singleton non-majority color whose pixel has at least two 4-neighbors of the majority color
#      * the remaining color is just support structure
#      The generator explicitly constructs only sprites satisfying those conditions.
#    - Singleton distractors from the original task are dropped; every movable object is a 3-color sprite so the fused
#      behavior is always present.
#
# d) Placement legality:
#    - Generation stays output-first: place solved sprites first with the align marker already aligned to the anchor,
#      then scatter each sprite along the inverse axis.
#    - `_can_place(...)` enforces in-bounds placement, no overlap, and an 8-neighborhood moat between anchors/sprites.
#    - The support tail grows so only its first cell touches the majority region; later tail cells do not touch the
#      majority region or the align marker, preventing merged role confusion.
#
# e) Visual observability:
#    - Across a few samples, a human can see that border-anchor colors match the biggest region of each sprite, not one
#      of the odd little colors.
#    - The aligned point is also visible because the tucked singleton, not the bulk region, ends up sharing the anchor's
#      row or column in the output.
#    - If one applied either the original "minority is the marker" rule or a pure D9 "majority connector pixel aligns"
#      rule, many generated examples would move differently.
#
# f) Visual cleanliness:
#    - Each sprite uses one dominant patch plus two small attachments, which is easy to parse by eye.
#    - Anchors stay isolated on the border and sprites keep at least one background cell of spacing.
#    - Auxiliary colors avoid anchor colors when possible, so the border-equality cue remains clean.
#
# g) Change inferability:
#    - There is no hidden arithmetic or color-id ordering. Equality of visible colors drives pairing.
#    - The fused rule is inferable from examples because the same majority anchor-color relation repeats, while the
#      output consistently lines up the tucked singleton rather than the centroid or the support tail.


def _find_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = g.shape

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
    return anchors_by_color


def _sprite_roles(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict[str, object] | None:
    pts = [(int(r), int(c)) for r, c in coords]
    vals = [int(grid[r, c]) for r, c in pts]
    cnt = Counter(vals)
    fg_counts = {int(color): int(n) for color, n in cnt.items() if int(color) != int(bg)}
    if len(fg_counts) != 3:
        return None

    top = max(int(n) for n in fg_counts.values())
    match_colors = [int(color) for color, n in fg_counts.items() if int(n) == int(top)]
    if len(match_colors) != 1:
        return None
    match_color = int(match_colors[0])
    match_cells = {(int(r), int(c)) for r, c in pts if int(grid[r, c]) == int(match_color)}

    align_candidates = []
    support_candidates = []
    for color, n in fg_counts.items():
        if int(color) == int(match_color):
            continue
        cells = [(int(r), int(c)) for r, c in pts if int(grid[r, c]) == int(color)]
        if int(n) == 1:
            r, c = cells[0]
            match_contacts = sum((int(rr), int(cc)) in match_cells for rr, cc in neighbors4(int(r), int(c)))
            if int(match_contacts) >= 2:
                align_candidates.append((int(color), (int(r), int(c)), int(match_contacts)))
            else:
                support_candidates.append((int(color), cells))
        else:
            support_candidates.append((int(color), cells))

    if len(align_candidates) != 1 or len(support_candidates) != 1:
        return None

    align_color, align_pixel, align_contacts = align_candidates[0]
    support_color, support_cells = support_candidates[0]
    if int(fg_counts[int(support_color)]) >= int(fg_counts[int(match_color)]):
        return None

    return {
        "match_color": int(match_color),
        "align_color": int(align_color),
        "support_color": int(support_color),
        "align_pixel": tuple(map(int, align_pixel)),
        "align_contacts": int(align_contacts),
        "match_count": int(fg_counts[int(match_color)]),
        "support_count": int(fg_counts[int(support_color)]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide three-color sprites by matching border anchors with their dominant color while aligning a different tucked singleton.

    Input structure:
    - The background is the most common color.
    - Each movable interior object is an 8-connected 3-color sprite: one dominant majority-color region, one small support tail, and one tucked singleton alignment pixel.
    - Isolated non-background border singletons act as anchors.

    Rule:
    - Pair each sprite with the anchor whose color matches the sprite's largest color region.
    - For top/bottom anchors, slide the whole sprite horizontally until the tucked singleton shares the anchor column.
    - For left/right anchors, slide the whole sprite vertically until the tucked singleton shares the anchor row.

    Key insight:
    - The color that chooses the anchor is the big obvious region, but the pixel that must line up is a different tiny singleton, so solving requires combining majority-color matching with a separate alignment cue.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color = _find_anchors(g, int(bg))
    assert anchors_by_color

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue

        roles = _sprite_roles(g, coords, int(bg))
        if roles is None:
            continue

        match_color = int(roles["match_color"])
        if int(match_color) not in anchors_by_color:
            continue

        align_r, align_c = map(int, roles["align_pixel"])
        ar, ac = anchors_by_color[int(match_color)]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - align_c)
        else:
            dr = int(ar - align_r)
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


# --------------- generator ---------------

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
_MATCH_AREA_RANGE = (6, 12)
_SUPPORT_AREA_RANGE = (2, 3)
_GEN_RETRY_LIMIT = 500
_SPRITE_BUILD_RETRY_LIMIT = 150

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


def _make_match_region(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _align_candidates(match_cells):
    r0, c0, r1, c1 = _bbox(match_cells)
    touch_counts = Counter()
    for r, c in match_cells:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p not in match_cells:
                touch_counts[p] += 1

    inner = []
    outer = []
    for cell, touches in touch_counts.items():
        if int(touches) < 2:
            continue
        if r0 <= cell[0] <= r1 and c0 <= cell[1] <= c1:
            inner.append(cell)
        else:
            outer.append(cell)
    random.shuffle(inner)
    random.shuffle(outer)
    return inner + outer


def _match_neighbor_count(cell, match_cells):
    r, c = cell
    return sum((r + dr, c + dc) in match_cells for dr, dc in _NEIGHBORS4)


def _support_touches_blocked(cell, blocked):
    r, c = cell
    return any((r + dr, c + dc) in blocked for dr, dc in _NEIGHBORS4)


def _tail_start_candidates(match_cells, blocked):
    connectors = list(match_cells)
    random.shuffle(connectors)
    out = []
    for cr, cc in connectors:
        for dr, dc in _NEIGHBORS4:
            cell = (cr + dr, cc + dc)
            if cell in match_cells or cell in blocked:
                continue
            if _match_neighbor_count(cell, match_cells) != 1:
                continue
            if _support_touches_blocked(cell, blocked):
                continue
            out.append(cell)
    random.shuffle(out)
    return out


def _tail_frontier(tail, match_cells, blocked):
    out = set()
    for r, c in tail:
        for dr, dc in _NEIGHBORS4:
            cell = (r + dr, c + dc)
            if cell in tail or cell in match_cells or cell in blocked:
                continue
            if _match_neighbor_count(cell, match_cells) != 0:
                continue
            if _support_touches_blocked(cell, blocked):
                continue
            out.add(cell)
    return list(out)


def _grow_support_tail(match_cells, blocked, area):
    for start in _tail_start_candidates(match_cells, blocked):
        tail = {start}
        while len(tail) < area:
            frontier = _tail_frontier(tail, match_cells, blocked)
            if not frontier:
                break
            tail.add(random.choice(frontier))
        if len(tail) == area:
            return tail
    return None


def _make_sprite(match_area, support_area, match_color, support_color, align_color, bg):
    for _ in range(_SPRITE_BUILD_RETRY_LIMIT):
        match_cells = _make_match_region(match_area)
        align_sites = _align_candidates(match_cells)
        if not align_sites:
            continue
        for align in align_sites:
            blocked = {align}
            for rr, cc in neighbors4(int(align[0]), int(align[1])):
                blocked.add((int(rr), int(cc)))
            support = _grow_support_tail(match_cells, blocked, support_area)
            if support is None:
                continue
            all_cells = set(match_cells) | set(support) | {align}
            r0, c0, r1, c1 = _bbox(all_cells)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
            for r, c in match_cells:
                patch[r - r0, c - c0] = int(match_color)
            for r, c in support:
                patch[r - r0, c - c0] = int(support_color)
            patch[align[0] - r0, align[1] - c0] = int(align_color)
            sprite = {
                "patch": patch,
                "align_offset": (int(align[0] - r0), int(align[1] - c0)),
            }
            roles = _sprite_roles(patch, np.argwhere(patch != int(bg)), int(bg))
            if roles is None:
                continue
            if int(roles["match_color"]) != int(match_color):
                continue
            if tuple(map(int, roles["align_pixel"])) != tuple(map(int, sprite["align_offset"])):
                continue
            return sprite
    return None


def _top_left(sprite, align_r, align_c):
    off_r, off_c = sprite["align_offset"]
    return int(align_r) - int(off_r), int(align_c) - int(off_c)


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


def _can_place(canvas, sprite, align_r, align_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, align_r, align_c)
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


def _place(canvas, sprite, align_r, align_c, bg):
    top, left = _top_left(sprite, align_r, align_c)
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


def _scatter_positions(canvas, sprite, anchor, align_marker, bg):
    H, W = canvas.shape
    ar, ac = align_marker
    if anchor["axis"] == "h":
        return [
            (ar, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != ac and _can_place(canvas, sprite, ar, c, bg)
        ]
    return [
        (r, ac)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != ar and _can_place(canvas, sprite, r, ac, bg)
    ]


def _support_color(bg, anchor_colors, used):
    forbidden = {int(bg), *map(int, anchor_colors), *map(int, used)}
    pool = [c for c in _COLORS if c not in forbidden]
    if not pool:
        pool = [c for c in _COLORS if c != int(bg)]
    return random.choice(pool)


def _align_color(bg, match_color, support_color, anchor_colors):
    forbidden = {int(bg), int(match_color), int(support_color), *map(int, anchor_colors)}
    pool = [c for c in _COLORS if c not in forbidden]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(match_color), int(support_color)}]
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


def _match_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    match_colors = list(anchor_colors)
    while len(match_colors) < n_sprites:
        match_colors.append(random.choice(anchor_colors))
    random.shuffle(match_colors)
    return match_colors


def _place_aligned_sprites(out, anchors, match_colors, anchor_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_support = set()
    sprites = []
    for match_color in match_colors:
        anchor = by_color[int(match_color)]
        support_color = _support_color(bg, anchor_colors, used_support)
        used_support.add(int(support_color))
        align_color = _align_color(bg, match_color, support_color, anchor_colors)
        sprite = _make_sprite(
            random.randint(*_MATCH_AREA_RANGE),
            random.randint(*_SUPPORT_AREA_RANGE),
            match_color,
            support_color,
            align_color,
            bg,
        )
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        align_marker_pos = random.choice(positions)
        _place(out, sprite, align_marker_pos[0], align_marker_pos[1], bg)
        sprites.append((sprite, anchor, align_marker_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, align_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, align_pos, bg)
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
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    match_colors = _match_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, match_colors, anchor_colors, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
