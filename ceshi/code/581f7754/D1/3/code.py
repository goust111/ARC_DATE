# Variant of 581f7754
# Dimension: D1 — 匹配条件：标记色与锚点色匹配 (match_cond)
# Change: 精灵内部的少数色（marker）与边界锚点颜色相同即配对：s.marker.color == d.color... → 面积排序匹配
#
# Design analysis
# YAML:
#   concrete_behavior:
#     - The original task used color identity as the pairing key: each interior sprite had
#       a minority-color marker, and that marker color selected one isolated border anchor.
#     - This variant replaces that hidden lookup with a visible ordinal rule:
#       1) order anchors by clockwise position along the border
#       2) order sprites by total component area from largest to smallest
#       3) pair the k-th sprite with the k-th anchor
#     - The slide axis is unchanged: a top/bottom anchor forces horizontal motion
#       (match marker column), and a left/right anchor forces vertical motion
#       (match marker row).
#     - Example trace:
#       anchors in clockwise order = [top col 4, right row 8, bottom col 10]
#       sprite areas = [13, 9, 5]
#       original logic would ask which marker color matches which anchor color;
#       variant logic ignores those colors and instead sends area 13 -> first anchor,
#       area 9 -> second anchor, area 5 -> third anchor.
#   complexity_calibration:
#     - The source task 31f7f899 sorts objects and slots independently, then zips them.
#       This variant uses the same complexity pattern: sort anchors by border position and
#       sort sprites by area, without introducing multi-stage transformations.
#     - To keep the ordinal cue visible, sprites use distinct body areas with a gap of 2
#       cells (body areas sampled from 4,6,8,10,12; total areas become 5,7,9,11,13).
#       Ties are therefore structurally avoided rather than resolved by invisible ids.
#     - We keep only 3 or 4 sprites/anchors. Fewer than 3 weakens the ranking signal;
#       many more would clutter the grid and make non-overlap generation brittle.
#   implicit_assumption_audit:
#     - Old anchor detection stored a dict keyed by color. That depends on color being the
#       matching relation; it must become an ordered list of anchors.
#     - Old component filtering kept any component whose marker color existed among anchors.
#       That is now invalid because marker colors are intentionally unrelated to anchor colors.
#     - Old generator expanded anchor colors into a longer marker-color list so several
#       sprites could share one anchor. Ordinal matching needs a 1:1 mapping, so the variant
#       generates exactly one sprite per anchor.
#     - Old singleton distractors are removed. Under area-ranking they would become extra
#       sortable components and destabilize the intended correspondence.
#   placement_legality:
#     - Anchors remain isolated non-corner border pixels with radius-1 clearance.
#     - Sprites are still built output-first in solved positions, then scattered only along
#       the inverse axis, so the solver can recover them by a single axis-constrained slide.
#     - The generator enforces equal counts, unique areas, full in-bounds placement, and
#       gap >= 1 between every foreground object.
#   observability_and_cleanliness:
#     - Marker colors are sampled from colors disjoint from the anchor colors. This makes
#       the old color-match rule visibly impossible, so the new behavior cannot silently
#       degenerate back to the source task.
#     - At least one horizontal anchor and one vertical anchor are generated when possible,
#       so samples exhibit both slide directions while remaining visually sparse.
#   new_solver_assumptions:
#     - The solver assumes anchors are isolated border singletons and movable sprites are
#       multi-color 8-connected components away from the border.
#     - It also assumes sprite areas are unique enough for deterministic ranking.
#       The generator guarantees this by construction.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their marker pixels align with border anchors chosen by size rank.

    Input structure:
    - bg is the most common color.
    - Several interior multi-color sprites each contain a single minority-color marker pixel.
    - Several isolated non-bg border pixels act as anchors; their border order is meaningful,
      but their colors do not determine the matching.

    Rule:
    - Order anchors clockwise along the border.
    - Order sprites by total connected-component area from largest to smallest.
    - Pair these ordered lists and slide each sprite along the axis implied by its anchor:
      top/bottom anchors align marker columns, left/right anchors align marker rows.

    Key insight: The task no longer matches marker color to anchor color; the correspondence
    is established by ordinal rank between sprite sizes and anchor border positions.
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
        on_top_bottom = int(r) in (0, int(H - 1))
        on_left_right = int(c) in (0, int(W - 1))
        return "h" if bool(on_top_bottom and not on_left_right) else "v"

    anchors: list[dict[str, int | str]] = []
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
            if ok:
                anchors.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "color": int(v),
                        "axis": anchor_axis(int(r), int(c)),
                    }
                )

    anchors.sort(key=lambda a: border_index(int(a["r"]), int(a["c"])))
    assert anchors

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
        bbox = comp["bbox"]
        sprites.append(
            {
                "coords": coords,
                "marker": mp,
                "area": int(coords.shape[0]),
                "bbox": tuple(int(x) for x in bbox),
            }
        )

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
        mp = sprite["marker"]
        if str(anchor["axis"]) == "h":
            dr = 0
            dc = int(anchor["c"]) - int(mp[1])
        else:
            dr = int(anchor["r"]) - int(mp[0])
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

_BODY_AREA_RANGE = (4, 13)
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
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in forbidden and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in forbidden]
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


def _anchor_count(H, W):
    if min(H, W) >= 15 and random.random() < 0.45:
        return 4
    return 3


def _place_anchors(out, inp, bg):
    H, W = out.shape
    n = _anchor_count(H, W)
    palette = [c for c in _COLORS if c != bg]
    if len(palette) < n:
        return None
    anchor_colors = random.sample(palette, n)
    anchors = []

    required_axes = ["h", "v"] if n >= 2 else []
    for color, forced_axis in zip(anchor_colors[: len(required_axes)], required_axes):
        candidates = [p for p in _anchor_candidates(out, bg) if p[0] == forced_axis]
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)

    for color in anchor_colors[len(required_axes) :]:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)

    return _ordered_anchors(anchors, H, W)


def _marker_colors(bg, anchors):
    forbidden = {int(bg)} | {int(a["color"]) for a in anchors}
    pool = [c for c in _COLORS if c not in forbidden]
    if len(pool) < len(anchors):
        return None
    colors = random.sample(pool, len(anchors))
    random.shuffle(colors)
    return colors


def _body_areas(H, W, n):
    pool = [a for a in _AREA_CHOICES if _BODY_AREA_RANGE[0] <= a <= _BODY_AREA_RANGE[1]]
    if min(H, W) < 12:
        pool = [a for a in pool if a <= 10]
    if len(pool) < n:
        return None
    return sorted(random.sample(pool, n), reverse=True)


def _place_aligned_sprites(out, anchors, marker_colors, body_areas, bg):
    used_body = set()
    anchor_colors = {int(a["color"]) for a in anchors}
    sprites = []
    for anchor, marker, area in zip(anchors, marker_colors, body_areas):
        body = _body_color(bg, marker, used_body, anchor_colors)
        used_body.add(body)
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


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    anchors = _place_anchors(out, inp, bg)
    if anchors is None:
        return None

    marker_colors = _marker_colors(bg, anchors)
    if marker_colors is None:
        return None

    body_areas = _body_areas(out.shape[0], out.shape[1], len(anchors))
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
    raise RuntimeError("failed to generate a valid 581f7754 D1 input")
