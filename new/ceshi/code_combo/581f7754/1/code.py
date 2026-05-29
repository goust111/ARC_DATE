# Combo variant of 581f7754 (index 1)
# combo_spec: D1/3+D10/0
# Name: 面积排序配对+严格一对一

# Design analysis
# ---------------
# This combo keeps the original "slide to a border anchor along an axis implied by the
# anchor side" mechanic, but changes the pairing relation in two coordinated ways:
#   1) the puzzle is no longer keyed by color equality
#   2) the anchor/sprite relation is a strict bijection
#
# Concrete behavior and trace:
#   - Original task:
#       anchors = isolated border pixels, often only 1-3 colors total
#       sprites = 3-5 multicolor components
#       marker color inside a sprite selects an anchor of the same color
#       several sprites may share one marker color and therefore one anchor
#   - Combo task:
#       anchors and sprites have equal count N
#       anchors are sorted by clockwise border position
#       sprites are sorted by total connected-component area, largest first
#       the k-th largest sprite moves to the k-th anchor
#       slide axis still comes from anchor side:
#         top/bottom anchor -> align marker column
#         left/right anchor -> align marker row
#   - Example:
#       clockwise anchors = [top c=4, right r=7, bottom c=11]
#       sprite areas      = [13, 9, 5]
#       pairing becomes   = 13->first, 9->second, 5->third
#       even if the largest sprite's marker color is 6 and the first anchor color is 2,
#       they still pair because size rank overrides color.
#
# Complexity calibration:
#   - A bijection alone can become too easy if there are only 1-2 pairs; ranking alone can
#     become ambiguous if areas tie. To keep the visual rule legible but nontrivial, the
#     generator uses 3-5 anchors/sprites and gives every sprite a distinct total area from
#     a small visible ladder (5,7,9,11,13 total cells).
#   - Anchors are also ordered by a visible geometric rule, not by color id. At least one
#     horizontal anchor and one vertical anchor are generated when possible so examples show
#     both motion axes.
#
# Implicit assumption audit:
#   - Original solver stored anchors in a dict keyed by color. That is invalid here because
#     colors do not determine matching.
#   - Original solver accepted multiple sprites per anchor color and sometimes single-cell
#     distractors. Those would break the strict bijection and also pollute the area ranking,
#     so the combo removes distractors and treats only multicolor sprites as movers.
#   - A ranking solver needs deterministic sprite order. The generator therefore guarantees
#     unique component areas rather than relying on invisible tie-breakers.
#   - Because color matching is removed, valid samples must provide another clear cue that
#     the old rule is wrong. The generator enforces anchor-color / marker-color set
#     separation, so marker colors never match anchor colors at all.
#
# Placement legality and cleanliness:
#   - Anchors remain isolated non-corner border singletons.
#   - Sprites remain fully interior and 8-separated from each other and from anchors.
#   - Generation is output-first: first place each sprite already aligned to its ranked
#     anchor, then scatter it only along the inverse axis to create the input.
#   - This guarantees that one constrained translation solves the puzzle and keeps the
#     input/output visually tidy instead of stacking multiple objects in one region.
#
# Observability and non-degeneracy:
#   - Human solvers can see that there are as many anchors as sprites, that marker colors do
#     not line up with anchor colors, and that larger sprites consistently map earlier along
#     the border order.
#   - The key anti-degeneration control is marker/anchor color disjointness. Under the
#     original color-match rule, no sprite would have a valid anchor at all, so the combo
#     behavior cannot silently collapse back to the source task on generated samples.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their marker pixels align with border anchors chosen by size rank.

    Input structure:
    - The background is the most common color.
    - There are N isolated non-corner border anchors and exactly N interior multicolor sprites.
    - Each sprite has one minority-color marker pixel, but marker colors do not determine which
      anchor it belongs to.

    Rule:
    - Sort anchors by clockwise border position.
    - Sort sprites by total connected-component area from largest to smallest.
    - Pair these ordered lists one-to-one. For each pair, slide the sprite along the axis implied
      by the anchor: top/bottom anchors align marker columns, left/right anchors align marker rows.

    Key insight: The correspondence is a strict ordinal bijection between sprite size rank and
    anchor border order; color is only internal decoration, not the matching key.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def border_index(r: int, c: int) -> int:
        r = int(r)
        c = int(c)
        if r == 0:
            return int(c)
        if c == int(W - 1):
            return int((W - 1) + r)
        if r == int(H - 1):
            return int((W - 1) + (H - 1) + ((W - 1) - c))
        return int(2 * (W - 1) + (H - 1) + ((H - 1) - r))

    def anchor_axis(r: int, c: int) -> str:
        if int(r) in (0, int(H - 1)) and int(c) not in (0, int(W - 1)):
            return "h"
        return "v"

    anchors: list[dict[str, int | str]] = []
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
            anchors.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "axis": anchor_axis(int(r), int(c)),
                }
            )

    if not anchors:
        raise ValueError("expected at least one isolated border anchor")

    anchors.sort(key=lambda a: border_index(int(a["r"]), int(a["c"])))
    anchor_cells = {(int(a["r"]), int(a["c"])) for a in anchors}

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
        comp_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    sprites: list[dict[str, object]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if all((int(r), int(c)) in anchor_cells for r, c in coords):
            continue
        marker_col = component_marker_color(coords)
        if marker_col is None:
            continue
        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue
        bbox = tuple(int(x) for x in comp["bbox"])
        sprites.append(
            {
                "coords": coords,
                "marker": mp,
                "area": int(coords.shape[0]),
                "bbox": bbox,
            }
        )

    if len(sprites) != len(anchors):
        raise ValueError("strict one-to-one variant expects equal anchor and sprite counts")

    sprites.sort(
        key=lambda s: (
            -int(s["area"]),
            int(s["bbox"][0]),
            int(s["bbox"][1]),
            int(s["marker"][0]),
            int(s["marker"][1]),
        )
    )

    moves: list[tuple[np.ndarray, int, int]] = []
    for sprite, anchor in zip(sprites, anchors):
        mr, mc = sprite["marker"]
        if str(anchor["axis"]) == "h":
            dr = 0
            dc = int(anchor["c"]) - int(mc)
        else:
            dr = int(anchor["r"]) - int(mr)
            dc = 0
        moves.append((sprite["coords"], int(dr), int(dc)))

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

_PAIR_COUNT_RANGE = (3, 5)
_AREA_CHOICES = (4, 6, 8, 10, 12)
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
    3: (10, 9),
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


def _border_index(H, W, r, c):
    if r == 0:
        return int(c)
    if c == W - 1:
        return int((W - 1) + r)
    if r == H - 1:
        return int((W - 1) + (H - 1) + ((W - 1) - c))
    return int(2 * (W - 1) + (H - 1) + ((H - 1) - r))


def _ordered_anchors(anchors, H, W):
    return sorted(anchors, key=lambda a: _border_index(H, W, int(a["r"]), int(a["c"])))


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


def _body_color(bg, marker, used, forbidden):
    preferred = [
        c
        for c in _COLORS
        if c not in {int(bg), int(marker)}
        and c not in forbidden
        and c not in used
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


def _new_canvas_pair(pair_count):
    min_h, min_w = _PAIR_MIN_SIDE[int(pair_count)]
    H = random.randint(int(min_h), int(_GRID_HEIGHT_RANGE[1]))
    W = random.randint(int(min_w), int(_GRID_WIDTH_RANGE[1]))
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg, pair_count):
    palette = [c for c in _COLORS if c != bg]
    anchor_colors = random.sample(palette, int(pair_count))
    anchors = []

    forced_axes = ["h", "v"] if int(pair_count) >= 2 else []
    for color, forced_axis in zip(anchor_colors[: len(forced_axes)], forced_axes):
        candidates = [p for p in _anchor_candidates(out, bg) if p[0] == forced_axis]
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)

    for color in anchor_colors[len(forced_axes) :]:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)

    return _ordered_anchors(anchors, out.shape[0], out.shape[1])


def _marker_colors(bg, anchors):
    forbidden = {int(bg)} | {int(a["color"]) for a in anchors}
    pool = [c for c in _COLORS if c not in forbidden]
    if not pool:
        return None
    colors = [int(random.choice(pool)) for _ in anchors]
    if all(colors[i] == int(anchors[i]["color"]) for i in range(len(anchors))):
        return None
    return colors


def _body_areas(pair_count):
    if len(_AREA_CHOICES) < int(pair_count):
        return None
    return sorted(random.sample(_AREA_CHOICES, int(pair_count)), reverse=True)


def _place_aligned_sprites(out, anchors, marker_colors, body_areas, bg):
    used_body = set()
    anchor_colors = {int(a["color"]) for a in anchors}
    sprites = []
    for anchor, marker, area in zip(anchors, marker_colors, body_areas):
        body = _body_color(bg, marker, used_body, anchor_colors)
        used_body.add(int(body))
        sprite = _make_sprite(int(area), int(body), int(marker), bg)
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    shuffled = sprites[:]
    random.shuffle(shuffled)
    for sprite, anchor, marker_pos in shuffled:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if np.array_equal(inp, out):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    pair_count = random.randint(*_PAIR_COUNT_RANGE)
    bg, out, inp = _new_canvas_pair(pair_count)

    anchors = _place_anchors(out, inp, bg, pair_count)
    if anchors is None:
        return None

    marker_colors = _marker_colors(bg, anchors)
    if marker_colors is None:
        return None

    body_areas = _body_areas(pair_count)
    if body_areas is None:
        return None

    sprites = _place_aligned_sprites(out, anchors, marker_colors, body_areas, bg)
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
