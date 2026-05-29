# Variant of 581f7754
# Dimension: D3 — 源对象：精灵结构 (src)
# Change: 多色精灵（8-连通分量），包含1个主体色（4-13像素）和1个标记色（1像素，最少数色），标记与主体4-相邻... → 三色精灵（双标记）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   task_type: move_to_match
#   unchanged_core:
#     - isolated border anchors still decide the slide axis
#     - top/bottom anchor => horizontal slide; left/right anchor => vertical slide
#     - moved pixels are clipped at the canvas edge and never overwrite existing non-bg cells
#     - generator remains output-first: build solved state, then scatter along the move axis
#   changed_sprite:
#     - every movable object is now a 3-color 8-connected sprite
#     - body: 4-13 pixels of one color
#     - marker_A ("match marker"): 1 pixel, color matches a border anchor, touches the body in exactly 1
#       four-neighbor direction, and is generated as a dangling outer nub when possible
#     - marker_B ("align marker"): 1 pixel, color does not appear on the border, touches the body in at least 2
#       four-neighbor directions, and is generated in a notch / cavity-like site when possible
#   solver_behavior:
#     - pair each sprite to the anchor whose color equals marker_A
#     - translate the whole sprite so marker_B, not marker_A, shares the anchor row/column
#
# behavior_trace:
#   original:
#     - a 2-color sprite used its only singleton-color pixel both for matching and for alignment
#     - if anchor was on the top edge, the sprite moved horizontally until that singleton column matched the anchor
#   variant:
#     - the dangling singleton pixel picks the anchor color
#     - the tucked singleton pixel supplies the row/column that must align with that anchor
#     - example: if marker_A is color 2 at column 3, marker_B is color 7 at column 5, and the matching top anchor
#       of color 2 sits at column 9, then the sprite slides horizontally by +4 so column 5, not column 3, lands on 9
#
# complexity_calibration:
#   - the two markers are not distinguished by color id ordering or frequency ties; both are singleton colors
#   - instead they are distinguished by visible local structure: one touches the body once, the other at least twice
#   - generator enforces that marker_A and marker_B differ on the active movement axis, so the variant does not
#     collapse back into the original single-marker behavior
#   - align-marker colors are chosen outside the anchor-color set when possible, making the role split visible
#
# main_assumption_audit:
#   - original component_marker_color(): invalid now because there are two singleton colors rather than one
#   - original marker_pixel(): invalid now because the pixel used for matching is no longer the pixel used for alignment
#   - original singleton-distractor branch is intentionally dropped from generation to keep every movable source
#     object in the new 3-color sprite family
#   - new solver assumption: each movable sprite has exactly one body color (count 4-13), one match marker with
#     one body contact, and one align marker with >=2 body contacts; generator constructs only such sprites
#
# placement_legality:
#   - sprites are still separated by an 8-neighborhood gap and stay one cell away from the border in the input
#   - anchor colors are unique per anchor; multiple sprites may share the same marker_A color
#   - solved placement uses marker_B as the reference point, and scattering changes only the coordinate that the
#     solver later corrects
#
# visual_observability:
#   - across a few examples, a human can see that border colors match the dangling pixel, but the object settles
#     when the tucked pixel reaches the anchor row/column
#   - because marker_A and marker_B are structurally different within the same sprite, their roles are inferable
#     from shape alone instead of hidden color metadata


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

    body_colors = [color for color, n in fg_counts.items() if int(n) >= 4]
    singleton_colors = [color for color, n in fg_counts.items() if int(n) == 1]
    if len(body_colors) != 1 or len(singleton_colors) != 2:
        return None

    body_color = int(body_colors[0])
    marker_pixels = {
        int(color): next((r, c) for r, c in pts if int(grid[r, c]) == int(color))
        for color in singleton_colors
    }
    body_set = {(r, c) for r, c in pts if int(grid[r, c]) == int(body_color)}
    contacts = {}
    for color, (r, c) in marker_pixels.items():
        contacts[int(color)] = sum((int(rr), int(cc)) in body_set for rr, cc in neighbors4(int(r), int(c)))

    match_colors = [color for color in singleton_colors if int(contacts[int(color)]) == 1]
    align_colors = [color for color in singleton_colors if int(contacts[int(color)]) >= 2]
    if len(match_colors) != 1 or len(align_colors) != 1:
        return None

    match_color = int(match_colors[0])
    align_color = int(align_colors[0])
    return {
        "body_color": int(body_color),
        "match_color": int(match_color),
        "align_color": int(align_color),
        "match_pixel": tuple(map(int, marker_pixels[match_color])),
        "align_pixel": tuple(map(int, marker_pixels[align_color])),
        "match_contacts": int(contacts[match_color]),
        "align_contacts": int(contacts[align_color]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide three-color sprites so a tucked alignment marker meets the border anchor chosen by a different dangling marker.

    Input structure:
    - bg is the most common color.
    - Each movable interior sprite is 8-connected and uses exactly 3 colors: a large body, a 1-contact match marker, and a 2+-contact align marker.
    - Isolated non-bg border singletons act as anchors.

    Rule:
    - Pair each sprite with the anchor whose color matches the dangling 1-contact marker.
    - Top/bottom anchors make the sprite slide horizontally until the tucked marker shares the anchor column.
    - Left/right anchors make the sprite slide vertically until the tucked marker shares the anchor row.

    Key insight:
    - The two singleton colors have different jobs: outer marker picks the anchor by color, inner/tucked marker provides the coordinate that must align.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color = _find_anchors(g, int(bg))
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1:
            r, c = map(int, coords[0])
            if (int(r), int(c)) in anchor_cells:
                continue
            continue

        roles = _sprite_roles(g, coords, int(bg))
        if roles is None:
            continue

        match_color = int(roles["match_color"])
        align_r, align_c = map(int, roles["align_pixel"])
        if int(match_color) not in anchors_by_color:
            continue

        ar, ac = anchors_by_color[int(match_color)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

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


def _marker_sites(body, axis):
    r0, c0, r1, c1 = _bbox(body)
    touch_counts = Counter()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p not in body:
                touch_counts[p] += 1

    match_pref, match_fallback = [], []
    align_pref, align_fallback = [], []
    for p, touches in touch_counts.items():
        inside_bbox = r0 <= p[0] <= r1 and c0 <= p[1] <= c1
        if int(touches) == 1:
            if inside_bbox:
                match_fallback.append(p)
            else:
                match_pref.append(p)
        elif int(touches) >= 2:
            if inside_bbox:
                align_pref.append(p)
            else:
                align_fallback.append(p)

    match_candidates = match_pref or match_fallback
    align_candidates = align_pref or align_fallback
    if not match_candidates or not align_candidates:
        return None

    random.shuffle(match_candidates)
    random.shuffle(align_candidates)
    for match in match_candidates:
        for align in align_candidates:
            if match == align:
                continue
            if abs(match[0] - align[0]) + abs(match[1] - align[1]) == 1:
                continue
            if axis == "h" and int(match[1]) == int(align[1]):
                continue
            if axis == "v" and int(match[0]) == int(align[0]):
                continue
            return match, align
    return None


def _make_sprite(area, body_color, match_color, align_color, bg, axis):
    for _ in range(120):
        body = _make_body(area)
        sites = _marker_sites(body, axis)
        if sites is None:
            continue
        match, align = sites
        all_cells = set(body) | {match, align}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[match[0] - r0, match[1] - c0] = int(match_color)
        patch[align[0] - r0, align[1] - c0] = int(align_color)
        sprite = {
            "patch": patch,
            "match_offset": (int(match[0] - r0), int(match[1] - c0)),
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


def _body_color(bg, anchor_colors, used):
    forbidden = {int(bg), *map(int, anchor_colors), *map(int, used)}
    pool = [c for c in _COLORS if c not in forbidden]
    if not pool:
        pool = [c for c in _COLORS if c != int(bg)]
    return random.choice(pool)


def _align_color(bg, body, match, anchor_colors):
    forbidden = {int(bg), int(body), int(match), *map(int, anchor_colors)}
    pool = [c for c in _COLORS if c not in forbidden]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(body), int(match)}]
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
    used_body = set()
    sprites = []
    for match in match_colors:
        anchor = by_color[int(match)]
        body = _body_color(bg, anchor_colors, used_body)
        used_body.add(int(body))
        align = _align_color(bg, body, match, anchor_colors)
        sprite = _make_sprite(
            random.randint(*_BODY_AREA_RANGE),
            body,
            match,
            align,
            bg,
            anchor["axis"],
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
    raise RuntimeError("failed to generate a valid 581f7754 D3 input")
