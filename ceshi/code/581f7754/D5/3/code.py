# Variant of 581f7754
# Dimension: D5 — 变换：裁剪规则 (transform)
# Change: 滑动后超出网格边界的像素直接丢弃（裁剪）；目标位置的背景像素被覆盖，非背景像素不被覆盖... → 精灵到达后重新着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
# ---------------------------------------------------------------------------
# a) Concrete behavior and trace
#    - The input still contains isolated border anchors and interior 2-color sprites.
#      Each sprite has one large body color and one minority marker color.
#    - Matching and motion stay the same as the original task:
#        * top/bottom anchor -> slide horizontally until marker column = anchor column
#        * left/right anchor -> slide vertically until marker row = anchor row
#    - The changed step is rendering after the slide:
#        * every non-marker sprite pixel is recolored to the anchor/marker color
#        * every marker-colored pixel disappears into background
#      Small trace:
#        body=8, marker=4, anchor=4 on bottom edge at col 5
#        original logic: move the whole {8,4} sprite; output still shows 8 body + 4 marker
#        new logic: move by the same dc, but render only former body cells, all recolored to 4;
#        the 4 marker cell is erased. The anchor itself stays isolated on the border.
#    - This changes only the final stamping stage, so the solver still identifies the
#      source sprite from the input, not from the recolored output.
#
# b) Complexity calibration
#    - If output sprites lose both body color and marker, they can become ambiguous.
#      To avoid that, the generator enforces unique body shapes across sprites.
#    - At least 2 anchor colors are used per grid so the recolor rule is visible inside
#      a single sample instead of looking like a fixed global palette replacement.
#    - Marker colors still repeat across multiple sprites sometimes, preserving the
#      original many-to-one anchor matching and preventing the task from collapsing into
#      trivial one-anchor/one-sprite pairings.
#
# c) Implicit-assumption review of the original main()
#    - Anchor detection assumed isolated non-bg border singletons. This still holds.
#    - Marker inference assumed the marker color is the least frequent color in a sprite.
#      This remains valid because the generator keeps exactly one marker pixel.
#    - Translation was computed from a chosen marker pixel and anchor axis. This stays valid.
#    - The old paste loop copied original source colors cell-by-cell. That assumption is no
#      longer valid: keeping source colors would revert to the old task. The new loop must
#      treat marker-colored cells and body-colored cells differently.
#    - The old single-color singleton distractor rule would now create vanishing objects.
#      That is logically valid but visually noisy, so the generator disables distractors.
#
# d) Placement legality
#    - Output-first generation is preserved. We first place anchors, then place the solved
#      state using a sprite patch where the marker cell is removed and the body is recolored.
#    - Input scattering still uses the original two-color patch, moved only along the inverse
#      axis. This guarantees the new solver can recover the intended output.
#    - Sprites keep a 1-cell empty moat from each other and from anchors, so 8-connected
#      component extraction remains stable in both input and output.
#
# e) Visual observability
#    - The recolor rule is visible because input sprites use a distinct body color and output
#      sprites lose that color completely.
#    - The disappearing marker is visible because output sprites are rendered from the body
#      alone; the minority pixel is gone instead of merely being recolored.
#    - The task never relies on sorting by numeric color id. Color changes are driven only by
#      the visible relation "body recolors to the matched anchor/marker color".
#
# f) Visual cleanliness
#    - No clipping-based crowding is needed. The generator keeps aligned sprites fully inside
#      the grid and non-overlapping so the recolor effect is the dominant visual change.
#    - Distractor singletons are removed because they would simply vanish and add clutter
#      without teaching a new rule.
#
# g) Color-change inferability
#    - The only mapping is local and structural: a sprite's body becomes the color of its own
#      minority marker, which is also the matching anchor color.
#    - Because each sample has 2-3 anchor colors and 3-5 sprites, examples usually contain
#      repeated evidence for the same recolor relation.
#
# Assumption audit for the new main()
#    - It assumes anchors are isolated border singletons: generator guarantees this.
#    - It assumes each movable sprite either has one minority marker color or is a singleton
#      of an anchor color: generator uses only the first case, but the solver tolerates both.
#    - It assumes all marker-colored cells of a sprite should disappear after motion. The
#      generator uses exactly one marker pixel, so this is well-defined.
#    - It assumes recolored destination cells only need background writes. The generator keeps
#      solved sprites disjoint, so order-dependent collisions are avoided.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marked sprites toward matching isolated border anchors, then repaint them.

    Input structure:
    - Background is the most frequent color.
    - Interior sprites use one main body color plus a minority marker color.
    - Isolated non-background border singletons act as anchors for the marker color.

    Rule:
    - Match each sprite to the anchor sharing its marker color.
    - Top/bottom anchors force a horizontal slide; left/right anchors force a vertical slide.
    - After the slide, repaint every non-marker sprite pixel to the anchor color and erase the marker pixel(s) to background.

    Key insight: The marker still determines both the destination axis and the final recolor, so the input's small minority pixel explains the output's monochrome settled body.
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
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixels(coords: np.ndarray, marker: int) -> list[tuple[int, int]]:
        return [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = marker_pixels(coords, marker)
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        shape = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in shape and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[dict[str, object]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
        marker_pts: list[tuple[int, int]]
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)
            marker_pts = [(int(r), int(c))]
        else:
            marker_pts = marker_pixels(coords, int(marker_col))

        if int(marker_col) not in anchors_by_color:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_color": int(marker_col),
                "marker_pts": set(marker_pts),
            }
        )

    out = g.copy()

    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for move in moves:
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        marker_col = int(move["marker_color"])
        marker_pts = move["marker_pts"]
        for r, c in coords:
            src = (int(r), int(c))
            if src in marker_pts:
                continue
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(marker_col)

    return out


# --------------- generator (o5x, recolor variant) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

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
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
        "body_signature": frozenset(_norm(body)),
    }


def _solved_patch(sprite, marker_color, bg):
    patch = np.array(sprite["patch"], copy=True)
    mask = patch != int(bg)
    patch[mask] = int(marker_color)
    off_r, off_c = sprite["offset"]
    patch[int(off_r), int(off_c)] = int(bg)
    return patch


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


def _can_place_patch(canvas, patch, top, left, bg):
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


def _can_place(canvas, sprite, marker_r, marker_c, bg, solved=False, marker_color=None):
    patch = _solved_patch(sprite, int(marker_color), bg) if solved else sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    return _can_place_patch(canvas, patch, top, left, bg)


def _place(canvas, sprite, marker_r, marker_c, bg, solved=False, marker_color=None):
    patch = _solved_patch(sprite, int(marker_color), bg) if solved else sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, patch, top, left, transparent=int(bg))


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


def _aligned_positions(canvas, sprite, anchor, marker_color, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg, solved=True, marker_color=marker_color)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg, solved=True, marker_color=marker_color)
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


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


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


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body_colors = set()
    used_shapes = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        built = None
        for _ in range(60):
            body = _body_color(bg, marker, used_body_colors)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            if sprite["body_signature"] in used_shapes:
                continue
            positions = _aligned_positions(out, sprite, anchor, marker, bg)
            if not positions:
                continue
            built = (body, sprite, positions)
            break
        if built is None:
            return None
        body, sprite, positions = built
        used_body_colors.add(body)
        used_shapes.add(sprite["body_signature"])
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg, solved=True, marker_color=marker)
        sprites.append((sprite, anchor, marker_pos, marker))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos, _marker in sprites:
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
    raise RuntimeError("failed to generate a valid 581f7754 D5 input")
