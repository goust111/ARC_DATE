# Variant of 581f7754
# Dimension: D17 — 符号编码：锚点颜色附加信息 (symbol_instructions)
# Change: 锚点颜色仅用于配对匹配，不编码其他信息... → 锚点颜色编码旋转角度

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D17 variant
# a) Concrete behavior and trace:
#    - Keep the original scene grammar: isolated border anchor pixels, interior sprites with one minority-color
#      marker pixel, and axis-of-motion determined by which border the anchor lies on.
#    - The new visible change is attached to the anchor color. Three fixed anchor colors are used:
#      2 -> 0 degrees, 3 -> 90 degrees clockwise, 4 -> 180 degrees.
#    - A sprite still matches an anchor by its marker color, and it still slides only along the original axis.
#      After that slide, the sprite rotates around its own marker pixel. Rotating around the marker preserves the
#      original "align the marker with the anchor's row/column" logic instead of replacing it.
#    - Small trace:
#      original rule: marker color 3 with a left-edge anchor at row 6 -> translate vertically until marker row = 6.
#      new rule: do the same vertical translation first, then rotate the translated sprite 90 degrees clockwise
#      around that marker pixel because color 3 is the instruction color for quarter-turn rotation.
# b) Complexity calibration:
#    - The rotation vocabulary is intentionally small: exactly three angles (0/90/180). This is enough to make
#      color carry extra meaning without exploding the number of cases a human must infer.
#    - Every generated sample contains all three anchor colors, and each anchor color labels at least one sprite.
#      This gives repeated support for the color->angle mapping inside each example.
#    - Body colors never reuse anchor colors, so anchor colors remain visually special and act as a clear code.
# c) Implicit assumptions from the old solver that must change:
#    - The old solver assumed "move = pure translation" when it stored only (coords, dr, dc). That is no longer
#      sufficient; after computing the translation we must also transform each pixel around the marker.
#    - The old generator could allow many symmetric shapes because translation is always visible. With rotation,
#      a symmetric sprite could make 90 or 180 degrees invisible, so the generator now rejects sprites whose
#      relative non-bg pattern is unchanged by 90 or 180 degree rotation around the marker.
#    - The least-frequent-color marker heuristic still works only because generator guarantees exactly one marker
#      pixel per sprite and keeps anchor colors out of body fills.
# d) Placement legality:
#    - Generation remains output-first, but output now places the rotated solved sprite while input places the
#      unrotated version at a scattered location.
#    - Both solved and scattered placements are checked with the same non-overlap and isolation rules, and they
#      are kept fully in bounds so the rotated result is readable rather than mostly clipped.
# e) Visual observability:
#    - The new rule is directly visible because sprites are forced to be rotation-asymmetric relative to the marker.
#      Human solvers can compare examples and see that the same marker color repeatedly produces the same final
#      orientation once the sprite reaches the anchor-guided line.
# f) Visual cleanliness:
#    - Distractor singletons are disabled. They were harmless in the original task, but in this variant they add
#      clutter while not expressing the new color-coded rotation rule.
#    - The original isolation gap is preserved so anchors, bodies, and markers remain easy to parse.
# g) Color-rule inferability:
#    - The color mapping is fixed and small instead of being per-instance random. ARC solvers can infer it from a
#      few train pairs because the same three colors recur with repeated sprites across examples.
# New main() assumptions audited after the change:
#    - If a component has a marker color that is not an anchor color, it is ignored; generator guarantees markers
#      always use the instruction palette.
#    - The marker pixel must exist and be unique enough to define a rotation center; generator guarantees one
#      minority-color marker cell per sprite.
#    - Multiple sprites may share one anchor color; generator guarantees all such sprites use the same angle
#      because the angle is a property of the shared anchor color itself.


# --------------- solver ---------------

_COLORS = tuple(range(10))
_ROTATION_BY_COLOR = {
    2: 0,  # 0 degrees
    3: 1,  # 90 degrees clockwise
    4: 2,  # 180 degrees
}
_ANCHOR_COLORS = tuple(_ROTATION_BY_COLOR.keys())


def _rotate_delta(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    k = int(quarter_turns) % 4
    if k == 0:
        return int(dr), int(dc)
    if k == 1:
        return int(dc), int(-dr)
    if k == 2:
        return int(-dr), int(-dc)
    return int(-dc), int(dr)


def _rotate_cells_about(
    cells: list[tuple[int, int, int]],
    pivot_r: int,
    pivot_c: int,
    quarter_turns: int,
) -> list[tuple[int, int, int]]:
    out = []
    for r, c, value in cells:
        dr = int(r) - int(pivot_r)
        dc = int(c) - int(pivot_c)
        rr, cc = _rotate_delta(dr, dc, quarter_turns)
        out.append((int(pivot_r) + int(rr), int(pivot_c) + int(cc), int(value)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites onto the anchor-guided row/column, then orient them by anchor color.

    Input structure:
    - The background is the modal color.
    - Interior foreground components are small sprites with a single minority-color marker pixel.
    - Isolated border pixels act as anchors: their edge still determines the slide axis, and their color now also carries a rotation instruction.

    Rule:
    - Match each sprite to the isolated border anchor with the same marker color.
    - Top/bottom anchors make the sprite slide horizontally until the marker shares the anchor's column; left/right anchors make it slide vertically until the marker shares the anchor's row.
    - After the slide, rotate the translated sprite around its marker pixel by the angle encoded by that anchor color; clip out-of-bounds cells and never overwrite existing non-background cells.

    Key insight:
    - The border location answers "which line should the marker reach?", while the anchor color answers "which final orientation should that aligned sprite take?"
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
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moved_components: list[tuple[np.ndarray, list[tuple[int, int, int]]]] = []

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
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        translated = [
            (int(r) + int(dr), int(c) + int(dc), int(g[int(r), int(c)]))
            for r, c in coords
        ]
        turns = int(_ROTATION_BY_COLOR.get(int(marker_col), 0))
        moved_components.append(
            (
                coords,
                _rotate_cells_about(translated, mp[0] + dr, mp[1] + dc, turns),
            )
        )

    out = g.copy()
    for coords, _cells in moved_components:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for _coords, cells in moved_components:
        for r, c, value in cells:
            if 0 <= r < H and 0 <= c < W and int(out[r, c]) == int(bg):
                out[r, c] = int(value)

    return out


# --------------- generator ---------------

# CFG: palette and canvas bounds.
_BG_COLOR_POOL = tuple(c for c in range(1, 9) if c not in _ANCHOR_COLORS)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

# CFG: placement isolation and anchor location.
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

# CFG: anchor, sprite, and retry counts.
_SPRITE_COUNT_RANGE = (4, 6)
_BODY_AREA_RANGE = (5, 13)
_GEN_RETRY_LIMIT = 400
_SPRITE_RETRY_LIMIT = 80

# CFG: body geometry and shape mix.
_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 7)
_FRAME_HEIGHT_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (3, 7)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.22
_WIDE_BODY_THRESHOLD = 0.48
_SYMMETRIC_BODY_THRESHOLD = 0.68
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.2
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


def _make_sprite_from_patch(patch: np.ndarray, offset: tuple[int, int]) -> dict[str, object]:
    return {"patch": np.array(patch, dtype=int), "offset": (int(offset[0]), int(offset[1]))}


def _sprite_cells(sprite, bg):
    patch = sprite["patch"]
    out = []
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            value = int(patch[r, c])
            if value != int(bg):
                out.append((int(r), int(c), int(value)))
    return out


def _rotate_relative_cells(cells, quarter_turns):
    out = []
    for dr, dc, value in cells:
        rr, cc = _rotate_delta(dr, dc, quarter_turns)
        out.append((int(rr), int(cc), int(value)))
    return sorted(out)


def _sprite_is_rotation_visible(sprite, bg):
    off_r, off_c = sprite["offset"]
    cells = [(r - off_r, c - off_c, value) for r, c, value in _sprite_cells(sprite, bg)]
    base = sorted(cells)
    return _rotate_relative_cells(cells, 1) != base and _rotate_relative_cells(cells, 2) != base


def _make_sprite(area, body_color, marker_color, bg):
    for _ in range(_SPRITE_RETRY_LIMIT):
        body = _make_body(area)
        marker = _marker_cell(body)
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
        sprite = {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}
        if _sprite_is_rotation_visible(sprite, bg):
            return sprite
    return None


def _rotate_sprite(sprite, quarter_turns, bg):
    off_r, off_c = sprite["offset"]
    cells = []
    min_r = max_r = None
    min_c = max_c = None
    for r, c, value in _sprite_cells(sprite, bg):
        dr = int(r) - int(off_r)
        dc = int(c) - int(off_c)
        rr, cc = _rotate_delta(dr, dc, quarter_turns)
        cells.append((int(rr), int(cc), int(value)))
        min_r = rr if min_r is None else min(min_r, rr)
        max_r = rr if max_r is None else max(max_r, rr)
        min_c = cc if min_c is None else min(min_c, cc)
        max_c = cc if max_c is None else max(max_c, cc)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for rr, cc, value in cells:
        patch[rr - min_r, cc - min_c] = int(value)
    return _make_sprite_from_patch(patch, (-min_r, -min_c))


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


def _body_color(bg, used):
    pool = [c for c in _COLORS if c not in {int(bg), *_ANCHOR_COLORS} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), *_ANCHOR_COLORS}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.choice(_BG_COLOR_POOL)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    anchors = []
    colors = list(_ANCHOR_COLORS)
    random.shuffle(colors)
    for color in colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {
            "color": int(color),
            "r": int(r),
            "c": int(c),
            "axis": axis,
            "turns": int(_ROTATION_BY_COLOR[int(color)]),
        }
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchors


def _marker_color_sequence():
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = list(_ANCHOR_COLORS)
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(_ANCHOR_COLORS))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, used_body)
        used_body.add(body)
        base_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        if base_sprite is None:
            return None
        solved_sprite = _rotate_sprite(base_sprite, anchor["turns"], bg)
        positions = _aligned_positions(out, solved_sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, solved_sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((base_sprite, solved_sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for base_sprite, _solved_sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, base_sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, base_sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    anchors = _place_anchors(out, inp, bg)
    if anchors is None:
        return None
    sprites = _place_aligned_sprites(out, anchors, _marker_color_sequence(), bg)
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
    raise RuntimeError("failed to generate a valid D17 variant input for 581f7754")
