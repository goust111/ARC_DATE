# Variant of 581f7754
# Dimension: D14 — 滑动方向：单向vs双向 (meta)
# Change: 精灵只能沿一个轴的一个方向滑动，方向由标记到锚点的相对位置决定... → 精灵滑到边界停止（不看锚点列/行）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D14 variant
# - Concrete behavior:
#   The input still contains isolated border anchors and interior multicolor sprites.
#   The marker color still selects the matching anchor color, and the anchor edge still
#   chooses the motion axis: top/bottom anchors mean horizontal motion, left/right
#   anchors mean vertical motion. The changed part is the stopping rule: the sprite no
#   longer aligns its marker with the anchor's column/row. Instead, the marker-anchor
#   relative order only picks a direction along that axis, and the whole component slides
#   until one of its cells reaches the corresponding grid boundary.
# - Small traced example:
#   Original logic with a top anchor at column 2 and a marker at column 6 moved the
#   sprite left by 4 so the marker landed exactly at column 2. The variant uses the same
#   "anchor is left of marker" fact only as a leftward direction cue; if the component's
#   leftmost cell was at column 4, it now moves left by 4 so that leftmost cell reaches
#   column 0, regardless of the anchor column.
# - Complexity calibration:
#   The generator keeps 1-3 anchors and 3-5 sprites so repeated marker colors still
#   appear. That repetition is important: the task should still look like a color-based
#   anchor-matching problem, but the output difference is now visually obvious because
#   solved sprites are flush against borders instead of merely sharing a row/column.
# - Hidden assumptions checked against the old solver:
#   `horizontal = top/bottom` still holds, so anchor edge classification can stay.
#   The old `dc = ac - mp[1]` / `dr = ar - mp[0]` assumed anchor coordinates were the
#   destination. That is no longer valid. The new code replaces those lines with
#   boundary-distance shifts computed from component extrema, while still using the same
#   marker extraction and anchor-color matching.
#   The old generator assumed solved sprites stayed one cell away from every border.
#   That assumption breaks here because the solved state must touch exactly one boundary.
#   Placement checks were rewritten to allow contact on the chosen stopping side only.
# - Placement legality:
#   Input sprites still stay strictly inside the grid with an empty 8-neighborhood gap.
#   Output sprites are also isolated from anchors and from each other, but may touch one
#   boundary on the stopping side. Scatter positions are restricted so the input marker
#   is on the correct side of its anchor; this guarantees the solver's direction branch.
# - Visual observability:
#   Across a few examples, the difference from the original task is easy to see: solved
#   sprites stack against left/right/top/bottom borders, while anchors remain detached.
#   Because anchors no longer serve as exact row/column targets, examples naturally show
#   anchors whose columns/rows do not match the final marker positions.
# - Neatness:
#   Only one contact mode is allowed per solved sprite, and anchors remain isolated border
#   singletons. This keeps the output legible even though more objects end up on borders.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites along the anchor-selected axis until they hit a boundary.

    Input structure:
    - bg = most common color.
    - Interior multi-color sprites each contain a minority-color marker pixel.
    - Isolated non-bg border singletons act as anchors for matching marker colors.

    Rule:
    - Match each sprite to the anchor sharing its marker color.
    - Top/bottom anchors still mean horizontal motion; left/right anchors still mean vertical motion.
    - The anchor no longer gives the destination column/row: it only tells which way to move along that axis, and the sprite keeps sliding until it touches the grid edge.

    Key insight: Separate the anchor's two roles — its edge chooses the motion axis, while its relative position to the marker chooses left/right or up/down; the stopping point is always the boundary, not the anchor line.
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
        body = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in body and int(g[int(rr), int(cc)]) != int(marker):
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

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        rows = [int(r) for r, _c in coords]
        cols = [int(c) for _r, c in coords]

        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            if int(ac) < int(mp[1]):
                dc = -min(cols)
            elif int(ac) > int(mp[1]):
                dc = int(W - 1 - max(cols))
        else:
            if int(ar) < int(mp[0]):
                dr = -min(rows)
            elif int(ar) > int(mp[0]):
                dr = int(H - 1 - max(rows))

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


# --------------- generator (D14 gravity variant) ---------------

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
_GEN_RETRY_LIMIT = 300
_SPRITE_RETRY_LIMIT = 60

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
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (interior | exterior))
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
    occupied = {
        (int(top) + int(r), int(left) + int(c))
        for r, c in np.argwhere(mask)
    }
    H, W = canvas.shape
    for r, c in occupied:
        for rr in range(r - _ISOLATION_RADIUS, r + _ISOLATION_RADIUS + 1):
            for cc in range(c - _ISOLATION_RADIUS, c + _ISOLATION_RADIUS + 1):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in occupied:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return True
    return False


def _can_place(
    canvas,
    sprite,
    marker_r,
    marker_c,
    bg,
    *,
    allow_top=False,
    allow_bottom=False,
    allow_left=False,
    allow_right=False,
):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    if not allow_top and top < _INNER_BORDER_GAP:
        return False
    if not allow_left and left < _INNER_BORDER_GAP:
        return False
    if not allow_bottom and top + h > H - _INNER_BORDER_GAP:
        return False
    if not allow_right and left + w > W - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


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
    r0, r1 = max(0, r - _ISOLATION_RADIUS), min(H, r + _ISOLATION_RADIUS + 1)
    c0, c1 = max(0, c - _ISOLATION_RADIUS), min(W, c + _ISOLATION_RADIUS + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
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


def _settled_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    patch = sprite["patch"]
    off_r, off_c = sprite["offset"]
    h, w = patch.shape
    out = []
    if anchor["axis"] == "h":
        left_marker_c = int(off_c)
        if int(anchor["c"]) < left_marker_c:
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
                if _can_place(canvas, sprite, r, left_marker_c, bg, allow_left=True):
                    out.append({"marker": (int(r), left_marker_c), "direction": "left"})
        right_marker_c = int(W - (w - off_c))
        if int(anchor["c"]) > right_marker_c:
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
                if _can_place(canvas, sprite, r, right_marker_c, bg, allow_right=True):
                    out.append({"marker": (int(r), right_marker_c), "direction": "right"})
        return out
    top_marker_r = int(off_r)
    if int(anchor["r"]) < top_marker_r:
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if _can_place(canvas, sprite, top_marker_r, c, bg, allow_top=True):
                out.append({"marker": (top_marker_r, int(c)), "direction": "up"})
    bottom_marker_r = int(H - (h - off_r))
    if int(anchor["r"]) > bottom_marker_r:
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if _can_place(canvas, sprite, bottom_marker_r, c, bg, allow_bottom=True):
                out.append({"marker": (bottom_marker_r, int(c)), "direction": "down"})
    return out


def _scatter_positions(canvas, sprite, anchor, settled, bg):
    H, W = canvas.shape
    mr, mc = settled["marker"]
    direction = settled["direction"]
    out = []
    if anchor["axis"] == "h":
        if direction == "left":
            start = max(_INNER_BORDER_GAP, int(anchor["c"]) + 1, int(mc) + 1)
            cols = range(start, W - _INNER_BORDER_GAP)
        else:
            stop = min(W - _INNER_BORDER_GAP, int(anchor["c"]), int(mc))
            cols = range(_INNER_BORDER_GAP, stop)
        for c in cols:
            if _can_place(canvas, sprite, mr, c, bg):
                out.append((int(mr), int(c)))
        return out
    if direction == "up":
        start = max(_INNER_BORDER_GAP, int(anchor["r"]) + 1, int(mr) + 1)
        rows = range(start, H - _INNER_BORDER_GAP)
    else:
        stop = min(H - _INNER_BORDER_GAP, int(anchor["r"]), int(mr))
        rows = range(_INNER_BORDER_GAP, stop)
    for r in rows:
        if _can_place(canvas, sprite, r, mc, bg):
            out.append((int(r), int(mc)))
    return out


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


def _place_settled_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        chosen = None
        for _ in range(_SPRITE_RETRY_LIMIT):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            positions = _settled_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            chosen = (sprite, body, random.choice(positions))
            break
        if chosen is None:
            return None
        sprite, body, settled = chosen
        used_body.add(body)
        _place(out, sprite, settled["marker"][0], settled["marker"][1], bg)
        sprites.append((sprite, anchor, settled))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, settled in sprites:
        positions = _scatter_positions(inp, sprite, anchor, settled, bg)
        if not positions:
            return False
        marker_r, marker_c = random.choice(positions)
        _place(inp, sprite, marker_r, marker_c, bg)
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
    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_settled_sprites(out, anchors, marker_colors, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D14 input")
