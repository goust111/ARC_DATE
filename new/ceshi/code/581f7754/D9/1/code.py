# Variant of 581f7754
# Dimension: D9 — 精灵内部标记识别规则 (meta)
# Change: 标记色为精灵内部出现次数最少的颜色（少数色），仅1个像素... → 标记由空间位置决定（非频率）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D9 variant
# a) Concrete behavior and trace:
#    - The moved object is still an interior multi-color sprite matched to an isolated border anchor by color.
#    - The changed part is only how the sprite's identifying color is read:
#      the marker is now the unique topmost occupied cell of the component
#      (equivalently the lexicographically first occupied cell because the generator enforces only one cell on the top row).
#    - Generator trace example:
#      a sprite may contain 7 teal cells and 3 red cells, with the single row-0 cell colored teal.
#      Old logic: read red because it is rarer, then fail because no red anchor exists.
#      New logic: read the topmost teal cell, match the teal border anchor, and slide correctly.
# b) Complexity calibration:
#    - Sprites remain two-color, 3-5 per grid, with repeated anchor colors across sprites.
#    - The top marker color is deliberately repeated in several cells, while the contrasting body color is a smaller blob.
#      This prevents the task from collapsing back to "minority color = marker".
#    - Body colors are chosen outside the anchor set so the old frequency-based rule would visibly break.
# c) Implicit assumptions audited from the original main():
#    - The old component_marker_color() assumed color frequency identified the anchor; no longer valid.
#    - The old marker_pixel(marker_color) was safe only because the marker color was effectively singleton.
#      In this variant the anchor color appears multiple times, so the solver must find the extremal coordinate first,
#      then read its color, not the other way around.
#    - Singleton distractors remain safe because a 1-cell component is still its own extremal marker.
# d) Placement legality:
#    - The generator still uses output-first placement, interior-only sprite placement, isolated anchors, and 8-neighbor gaps.
#    - Each sprite patch is built so row 0 contains exactly one occupied cell at column 0.
#      This guarantees a unique spatial marker and keeps solver assumptions valid.
# e) Visual observability:
#    - Humans can now follow a geometric cue: the little upper protruding pixel's color points to the anchor.
#    - Because that color repeats elsewhere in the sprite, counting colors is visibly the wrong strategy.
# f) Visual cleanliness:
#    - Sprites are small connected patches with one compact minority-colored body blob.
#    - Objects still keep a bg moat, so anchors, sprites, and optional singleton distractors remain separable.
# g) Color inference:
#    - There is no color remapping beyond "same color as the matched anchor".
#    - The only changed hidden variable is geometric marker selection, and it is made explicit by the unique top cell.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior multi-color sprites by using each sprite's extreme top-left cell as the identifying marker.

    Input structure:
    - bg is the most common color.
    - Interior sprites are multi-color connected components whose identifying pixel is the unique topmost occupied cell, not the rarest color.
    - Isolated non-bg border singletons act as anchors and still determine the motion axis.

    Rule:
    - For each sprite, read the color of its topmost/leftmost occupied cell and match it to the border anchor of the same color.
    - If that anchor sits on the top or bottom edge, slide the whole sprite horizontally until the marker column matches the anchor column.
    - If that anchor sits on the left or right edge, slide the whole sprite vertically until the marker row matches the anchor row.

    Key insight:
    - Ignore within-sprite color counts. The decisive cue is geometric: the unique upper extreme cell supplies both the marker position and the anchor color.
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

    def top_left_marker(coords: np.ndarray) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in coords]
        return min(pts, key=lambda rc: (int(rc[0]), int(rc[1])))

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        vals = {int(g[int(r), int(c)]) for r, c in coords}
        if len(vals) <= 1:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            mp = (int(r), int(c))
            marker_col = int(v)
        else:
            mp = top_left_marker(coords)
            marker_col = int(g[int(mp[0]), int(mp[1])])
            if marker_col not in anchors_by_color:
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


# --------------- generator (o5x-d9) ---------------

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
_SPRITE_AREA_RANGE = (6, 13)
_GEN_RETRY_LIMIT = 300

_SHAPE_HEIGHT_RANGE = (3, 6)
_SHAPE_WIDTH_RANGE = (3, 7)
_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

_EXTREME_SEEDS = [
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
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


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                if rr == 0 and cc != 0:
                    continue
                frontier.append((rr, cc))
    return frontier


def _grow_extreme_shape(area):
    choices = [
        (h, w)
        for h in _inclusive_range(_SHAPE_HEIGHT_RANGE)
        for w in _inclusive_range(_SHAPE_WIDTH_RANGE)
        if h * w >= area
    ]
    h, w = random.choice(choices)
    seed = set(random.choice([s for s in _EXTREME_SEEDS if len(s) <= area]))
    cells = set(seed)
    while len(cells) < area:
        frontier = _frontier_cells(cells, h, w)
        if not frontier:
            return _grow_extreme_shape(area)
        cells.add(random.choice(frontier))
    return _norm(cells)


def _body_blob(cells, count):
    marker = (0, 0)
    available = set(cells) - {marker}
    start = max(available, key=lambda rc: (int(rc[0] + rc[1]), int(rc[0]), int(rc[1])))
    blob = {start}
    while len(blob) < count:
        frontier = []
        for r, c in blob:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in available and (int(rr), int(cc)) not in blob:
                    frontier.append((int(rr), int(cc)))
        frontier = sorted(set(frontier), key=lambda rc: (-(int(rc[0] + rc[1])), -int(rc[0]), -int(rc[1])))
        if not frontier:
            break
        blob.add(frontier[0])
    if len(blob) < count:
        remaining = sorted(available - blob, key=lambda rc: (-(int(rc[0] + rc[1])), -int(rc[0]), -int(rc[1])))
        for cell in remaining:
            blob.add(cell)
            if len(blob) == count:
                break
    return blob


def _make_sprite(area, body_color, marker_color, bg):
    cells = _grow_extreme_shape(area)
    max_body = max(2, (len(cells) - 1) // 2)
    body_count = random.randint(2, max_body)
    body_cells = _body_blob(cells, body_count)
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(marker_color)
    for r, c in body_cells:
        patch[r - r0, c - c0] = int(body_color)
    return {"patch": patch, "offset": (0, 0)}


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


def _body_color(bg, anchor_colors, used):
    forbidden = {int(bg)} | {int(c) for c in anchor_colors}
    pool = [c for c in _COLORS if c not in forbidden and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in forbidden]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _add_distractors(out, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in range(_INNER_BORDER_GAP, out.shape[1] - _INNER_BORDER_GAP):
            if not _can_place(out, sprite, anchor["r"], c, bg):
                continue
            for r in range(_INNER_BORDER_GAP, out.shape[0] - _INNER_BORDER_GAP):
                if r != anchor["r"] and _can_place(inp, sprite, r, c, bg):
                    pairs.append((r, c))
        if not pairs:
            return
        in_r, col = random.choice(pairs)
        _place(out, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, anchor_colors, bg)
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


def _place_aligned_sprites(out, anchors, marker_colors, anchor_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, anchor_colors, used_body)
        used_body.add(body)
        area = random.randint(*_SPRITE_AREA_RANGE)
        sprite = _make_sprite(area, body, marker, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D9 input")
