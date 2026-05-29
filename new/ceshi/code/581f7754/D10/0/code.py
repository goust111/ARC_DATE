# Variant of 581f7754
# Dimension: D10 — 锚点-精灵基数关系 (meta)
# Change: 多对多：1-3个锚点颜色，3-5个精灵，多个精灵可共享同一标记色（因此共享同一锚点）... → 严格一对一：每个锚点恰好一个精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D10 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     old: 1-3 anchor colors could control 3-5 sprites, so several sprites could carry the same
#       minority marker color and compete for the same border anchor.
#     new: choose N in [3,5] once, place exactly N isolated border anchors with pairwise-distinct
#       colors, and build exactly N interior sprites. Each sprite has one unique marker color and
#       that color matches exactly one anchor. The pairing is a bijection.
#   concrete_behavior:
#     input now visibly contains many anchor colors rather than a small reused set. Every interior
#     multicolor component has a unique minority pixel color; after solving, each such marker lands
#     on the row or column dictated by its same-colored anchor. No anchor color appears on a second
#     sprite and no singleton distractor mimics an extra sprite.
#   flow_delta_example:
#     old local pattern:
#       anchors = {2 on top, 4 on left}
#       sprite markers = [2,4,4]
#       solver moved two different sprites toward the same color-4 anchor.
#     new local pattern:
#       anchors = {2 on top, 4 on left, 7 on bottom}
#       sprite markers = [2,4,7]
#       each anchor attracts exactly one sprite; there is no color-sharing branch to reason about.
#   complexity_calibration:
#     the variant should remain visually rich, so the bijection is not collapsed to a single pair.
#     Instead N is lifted to 3-5 anchors/sprites. This keeps several simultaneous moves on each grid
#     while making the matching relation simpler and more observable.
#   implicit_solver_assumptions_review:
#     - original anchor scan stored the first anchor per color and silently ignored duplicates.
#       That was harmless when repeated colors were allowed. Here duplicate anchor colors would break
#       the bijection, so valid inputs must keep anchor colors unique.
#     - original component handling accepted single-cell distractors whose color matched an anchor.
#       Those were meaningful in the many-to-one task, but would create extra claimants in this
#       variant. The new generator removes them, and main only treats multicolor sprites as movers.
#     - original main moved every component whose least-frequent color matched an anchor. That still
#       works, but now valid inputs guarantee at most one such component per anchor color.
#   placement_legality:
#     anchors remain isolated non-corner border pixels. Sprites remain fully interior, 8-separated
#     from other foreground, and are first built in the solved output state, then scattered along the
#     inverse axis to form the input. To fit 3-5 pairs without clutter, larger pair counts bias the
#     grid size upward and shrink maximum sprite area slightly.
#   visual_observability:
#     the distinctive evidence for D10 is "many different anchor colors, one matching sprite each".
#     A human can verify this from 2-3 examples because every marker color appears exactly twice in
#     the whole puzzle: once as an isolated border anchor and once as a minority pixel inside one
#     sprite. There is no hidden ordering by color id or color-dependent transform.
#   visual_cleanliness:
#     no distractor singletons are generated; objects keep a one-cell moat; anchors stay isolated.
#     This avoids ambiguous stacks of same-colored noise after increasing the number of anchors.
#   color_inference:
#     colors are only used as equality cues. They are never sorted to decide motion. Axis still comes
#     from anchor edge location, which is directly visible.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide one interior sprite per anchor so each unique marker color matches its unique border anchor.

    Input structure:
    - The background is the most frequent color.
    - There are 3-5 interior multicolor sprites, each with exactly one minority-color marker pixel.
    - There are equally many isolated border anchors, and each anchor color appears on exactly one sprite marker.

    Rule:
    - Pair every sprite with the isolated border anchor of the same color.
    - If that anchor is on the top or bottom edge, slide the sprite horizontally until the marker shares the anchor's column.
    - If that anchor is on the left or right edge, slide the sprite vertically until the marker shares the anchor's row.

    Key insight:
    - Colors establish a one-to-one sprite-anchor match, while the anchor's border side decides which coordinate stays fixed during the translation.
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
            if any(
                0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg)
                for rr, cc in neighbors4(int(r), int(c))
            ):
                continue
            if v in anchors_by_color:
                raise ValueError("duplicate anchor color violates D10 bijection")
            anchors_by_color[v] = (int(r), int(c))

    assert anchors_by_color

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if len(pts) != 1:
            return None
        return pts[0]

    sprites_by_color: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        marker_col = component_marker_color(coords)
        if marker_col is None or int(marker_col) not in anchors_by_color:
            continue
        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue
        if int(marker_col) in sprites_by_color:
            raise ValueError("multiple sprites share one anchor color")
        sprites_by_color[int(marker_col)] = (coords, mp)

    if len(sprites_by_color) != len(anchors_by_color):
        raise ValueError("expected a one-to-one anchor/sprite matching")

    moves: list[tuple[np.ndarray, int, int]] = []
    for color, (coords, mp) in sprites_by_color.items():
        ar, ac = anchors_by_color[int(color)]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))
        dr = 0
        dc = 0
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


# --------------- generator (o5x, D10 variant) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_PAIR_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 300

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

_PAIR_MIN_SIDE = {
    3: (10, 8),
    4: (12, 10),
    5: (14, 12),
}

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
        removable = [(r, c) for r, c in cells if not (r in (0, h - 1) and c in (0, w - 1))]
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


def _body_color(bg, marker, used, anchor_colors):
    preferred = [
        c
        for c in _COLORS
        if c not in {int(bg), int(marker)}
        and c not in used
        and c not in anchor_colors
    ]
    if preferred:
        return random.choice(preferred)
    secondary = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if secondary:
        return random.choice(secondary)
    return random.choice([c for c in _COLORS if c not in {int(bg), int(marker)}])


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _body_area(pair_count):
    lo, hi = _BODY_AREA_RANGE
    hi = max(lo, hi - 2 * max(0, int(pair_count) - 3))
    return random.randint(int(lo), int(hi))


def _gen_attempt():
    pair_count = random.randint(*_PAIR_COUNT_RANGE)
    bg, out, inp = _new_canvas_pair(pair_count)
    anchors = _place_anchors(out, inp, bg, pair_count)
    if anchors is None:
        return None

    sprites = _place_aligned_sprites(out, anchors, bg)
    if sprites is None:
        return None
    if not _scatter_sprites(inp, sprites, bg):
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def _new_canvas_pair(pair_count):
    min_h, min_w = _PAIR_MIN_SIDE[int(pair_count)]
    H = random.randint(int(min_h), int(_GRID_HEIGHT_RANGE[1]))
    W = random.randint(int(min_w), int(_GRID_WIDTH_RANGE[1]))
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg, pair_count):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], int(pair_count))
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
    return anchors


def _place_aligned_sprites(out, anchors, bg):
    used_body = set()
    anchor_colors = {int(a["color"]) for a in anchors}
    shuffled = anchors[:]
    random.shuffle(shuffled)
    sprites = []
    for anchor in shuffled:
        marker = int(anchor["color"])
        body = _body_color(bg, marker, used_body, anchor_colors)
        used_body.add(int(body))
        sprite = _make_sprite(_body_area(len(anchors)), body, marker, bg)
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


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D10 input")
