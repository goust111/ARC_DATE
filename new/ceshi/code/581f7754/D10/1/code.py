# Variant of 581f7754
# Dimension: D10 — 锚点-精灵基数关系 (meta)
# Change: 多对多：1-3个锚点颜色，3-5个精灵，多个精灵可共享同一标记色（因此共享同一锚点）... → 一对多：一个锚点、多个精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# -----------------------------------------------------------------------------
# Design analysis for the D10 variant
#
# a) Concrete behavior and trace
#    - The grid now contains exactly one isolated border anchor pixel.
#    - There are still 3-5 separated interior sprites. Each sprite remains a
#      small 2-color connected component: one large body color plus one 1-cell
#      marker color.
#    - The key visible change is that marker colors no longer determine which
#      anchor a sprite follows. Instead, every sprite slides toward the lone
#      anchor, so all marker pixels end up on the same anchor row/column.
#    - Example trace:
#      old task: anchor colors {2,4}; sprite marker=2 slides to anchor(2),
#      sprite marker=4 slides to anchor(4).
#      new task: only one anchor, e.g. color 7 on the top edge at column 5;
#      a marker-2 sprite, a marker-3 sprite, and a marker-8 sprite all slide
#      horizontally until each marker lands in column 5.
#      Difference: the solver can no longer use color matching as the routing
#      rule. It must use the single anchor as a global target line.
#
# b) Complexity calibration
#    - If all marker colors matched the anchor color, this variant would
#      collapse into a trivial subset of the original task.
#    - To keep the change observable, generator samples use 2-3 distinct marker
#      colors across the 3-5 sprites, and marker colors are chosen to differ
#      from the anchor color.
#    - We keep body generation, size ranges, and axis semantics from the
#      original task so the only conceptual leap is the one-anchor/global-target
#      relation.
#
# c) Hidden-assumption audit for main()
#    - Original anchor collection stored a dict keyed by color; that assumes
#      color is the lookup key for movement. This is no longer valid.
#    - Original component filtering moved only components whose marker color
#      appeared among anchor colors. In this variant, that would incorrectly
#      leave most sprites unmoved, because marker colors intentionally differ
#      from the anchor.
#    - Original singleton special-case relied on anchor-color matching. It is
#      removed here because a 1-color pixel has no internal minority marker, so
#      moving it would introduce a new unrelated rule.
#    - Marker detection by least-frequent color still works, but now it is used
#      only to locate each sprite's alignment pixel, not to choose a target.
#
# d) Placement legality
#    - Generator still uses output-first construction: first place the solved
#      sprites already aligned to the anchor's target line, then scatter them
#      back along the inverse axis into the input.
#    - Output placement requires sprites to stay fully inside the interior and
#      not touch the anchor or each other. Input scattering also enforces
#      interior placement plus 8-neighbor isolation, so each sprite remains a
#      separate component for the solver.
#    - Because the movement axis is global, the scatter step changes only the
#      free axis from the solved state; the solver then reverses that offset.
#
# e) Visual observability
#    - With one border anchor and several sprites with mismatched marker colors,
#      the training pairs visibly show that color matching is gone.
#    - The repeated evidence is structural: every marker, regardless of color,
#      lands on the same row or the same column after transformation.
#    - The generator avoids depending on hidden color ids, ordering by numeric
#      value, or any non-visible metadata.
#
# f) Visual cleanliness
#    - We keep the original small-body geometry and isolation gap so sprites do
#      not merge in the input.
#    - The solved state is intentionally non-overlapping in the generator even
#      though the solver would tolerate overlaps; this keeps examples legible
#      while still making the one-anchor relation obvious.
#
# g) Color-inference sanity
#    - Anchor color is intentionally not reused as a marker color. This prevents
#      a false “same color means same destination” explanation.
#    - Marker color variety is capped at 2-3 distinct colors per grid so the
#      examples are readable and the repeated one-anchor behavior is still easy
#      to spot.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every interior sprite so its marker lines up with the lone border anchor.

    Input structure:
    - The background is the most common color.
    - Exactly one isolated non-background pixel sits on the border as the only anchor.
    - Several separated interior sprites are multicolor components with a 1-cell minority marker; marker colors may differ from the anchor and from each other.

    Rule:
    - Find the unique border anchor and use its edge position to choose one global slide axis.
    - For each multicolor sprite, locate its least-frequent-color marker pixel and translate the whole sprite until that marker shares the anchor's column or row.
    - Top/bottom anchors cause horizontal sliding; left/right anchors cause vertical sliding. Pixels that move outside the grid are clipped, and moved sprites do not overwrite earlier non-background pixels.

    Key insight:
    - Because there is only one anchor, color matching disappears: all sprites share the same target line, and each sprite moves independently based on its own marker position.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            if int(g[r, c]) == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok:
                anchors.append((int(r), int(c)))

    assert len(anchors) == 1
    anchor_r, anchor_c = anchors[0]
    anchor_cell = (int(anchor_r), int(anchor_c))

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
        members = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in members and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    horizontal = bool(int(anchor_r) in (0, int(H - 1)) and int(anchor_c) not in (0, int(W - 1)))
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1 and tuple(map(int, coords[0])) == anchor_cell:
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        dr = 0
        dc = 0
        if horizontal:
            dc = int(anchor_c - mp[1])
        else:
            dr = int(anchor_r - mp[0])
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

_ANCHOR_COUNT = 1
_SPRITE_COUNT_RANGE = (3, 5)
_MARKER_COLOR_COUNT_RANGE = (2, 3)
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


def _anchor_candidates(canvas, bg, axis_filter=None):
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
    return [
        (axis, r, c)
        for axis, r, c in out
        if (axis_filter is None or axis == axis_filter) and _anchor_ok(canvas, r, c, bg)
    ]


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


def _body_color(bg, anchor_color, marker, used):
    pool = [
        c
        for c in _COLORS
        if c not in {int(bg), int(anchor_color), int(marker)} and c not in used
    ]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(anchor_color), int(marker)}]
    return random.choice(pool)


def _marker_color_sequence(anchor_color, bg, n_sprites):
    pool = [c for c in _COLORS if c not in {int(bg), int(anchor_color)}]
    k = min(len(pool), random.randint(_MARKER_COLOR_COUNT_RANGE[0], _MARKER_COLOR_COUNT_RANGE[1],))
    distinct = random.sample(pool, k)
    markers = distinct[:]
    while len(markers) < n_sprites:
        markers.append(random.choice(distinct))
    random.shuffle(markers)
    return markers


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


def _place_anchor(out, inp, bg, n_sprites):
    anchor_color = random.choice([c for c in _COLORS if c != bg])
    prefer_axis = "h" if out.shape[0] >= out.shape[1] else "v"
    if int(n_sprites) <= 3:
        prefer_axis = None
    candidates = _anchor_candidates(out, bg, prefer_axis)
    if not candidates:
        candidates = _anchor_candidates(out, bg)
    if not candidates:
        return None
    axis, r, c = random.choice(candidates)
    anchor = {
        "color": int(anchor_color),
        "r": int(r),
        "c": int(c),
        "axis": axis,
    }
    _put_anchor(out, r, c, anchor_color)
    _put_anchor(inp, r, c, anchor_color)
    return anchor


def _place_aligned_sprites(out, anchor, marker_colors, bg):
    used_body = set()
    sprites = []
    for marker in marker_colors:
        body = _body_color(bg, anchor["color"], marker, used_body)
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


def _anchor_list(grid, bg):
    H, W = grid.shape
    out = []
    for r in range(H):
        for c in range(W):
            if r not in (0, H - 1) and c not in (0, W - 1):
                continue
            if int(grid[r, c]) == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok:
                out.append((int(r), int(c), int(grid[r, c])))
    return out


def _component_marker_color(grid, coords):
    vals = [int(grid[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    anchors = _anchor_list(inp, bg)
    if len(anchors) != _ANCHOR_COUNT:
        return False

    anchor_color = anchors[0][2]
    marker_colors = []
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in comps:
        coords = comp["coords"]
        if int(coords.shape[0]) == 1 and tuple(map(int, coords[0])) == anchors[0][:2]:
            continue
        marker = _component_marker_color(inp, coords)
        if marker is None:
            return False
        marker_colors.append(int(marker))

    if not (_SPRITE_COUNT_RANGE[0] <= len(marker_colors) <= _SPRITE_COUNT_RANGE[1]):
        return False
    if len(set(marker_colors)) < 2:
        return False
    if any(int(marker) == int(anchor_color) for marker in marker_colors):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    anchor = _place_anchor(out, inp, bg, n_sprites)
    if anchor is None:
        return None

    marker_colors = _marker_color_sequence(anchor["color"], bg, n_sprites)
    sprites = _place_aligned_sprites(out, anchor, marker_colors, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D10 input")
