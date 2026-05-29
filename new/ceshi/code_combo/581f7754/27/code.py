# Combo variant of 581f7754 (index 27)
# combo_spec: D1/3+D12/3
# Name: 面积排序匹配+形状编码偏移

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# a) Concrete behavior and trace:
#    - Original 581f7754 paired a sprite to an anchor by marker color, then stopped exactly
#      when the marker reached the anchor row/column.
#    - Here the same visible quantity, total sprite area, controls two stages:
#      1) sort sprites by total occupied-cell count, largest first
#      2) pair them with anchors sorted clockwise along the border
#      3) after selecting the anchor, slide on the usual axis but stop area cells past the
#         anchor projection: target_c = anchor_c + area for horizontal slides, and
#         target_r = anchor_r + area for vertical slides.
#    - Example: if the clockwise anchors are [top@c=2, right@r=4, bottom@c=5] and the
#      sprite total areas are [11, 7, 5], then the 11-cell sprite goes to the first anchor
#      and its marker ends at column 13, the 7-cell sprite goes to the right-edge anchor and
#      its marker ends at row 11, and the 5-cell sprite goes to the bottom anchor and its
#      marker ends at column 10. The old task would instead use marker colors and no extra
#      area-based overshoot.
# b) Complexity calibration:
#    - The ranking cue must be unambiguous, so the generator uses exactly 3 sprites with
#      distinct total areas drawn from {5, 7, 9, 11}. This avoids invisible tie-breaks.
#    - The offset cue must also be visible. Using the same small set of areas makes the
#      anchor-to-marker distance countable by eye while remaining large enough to differ
#      clearly from "align exactly to the anchor".
#    - Marker colors are deliberately disjoint from anchor colors, so the source rule cannot
#      silently explain the examples.
# c) Implicit assumption audit:
#    - The original `anchors_by_color` lookup is invalid because color is no longer the
#      matching relation; solver logic must build an ordered anchor list instead.
#    - The original sprite filter `marker_color in anchors_by_color` is also invalid; every
#      interior multi-color component is now a candidate sprite regardless of anchor colors.
#    - The original motion formula `anchor - marker` is insufficient; it now requires the
#      sprite's own total area as an added term on the movement axis.
#    - Singleton distractors are removed because they would become rankable components with
#      ambiguous "shape-coded" meaning and would pollute the size order.
# d) Placement legality:
#    - Generation remains output-first. Anchors are placed first, then solved sprites are
#      placed so their markers already satisfy the fused target relation.
#    - Input is produced by scattering each sprite along the same axis it will later slide
#      back on. `_can_place` enforces in-bounds placement, non-overlap, and radius-1
#      foreground isolation.
#    - Anchor positions are restricted so `anchor + area` always stays in the interior.
# e) Visual observability:
#    - A solver can inspect the output and see two linked facts: larger sprites correspond
#      to earlier clockwise anchors, and larger sprites land farther past those anchors.
#    - Because marker colors never match anchor colors, the old color-based rule is visibly
#      unsupported; the only stable shared cue is the area ordering/counting.
# f) Visual cleanliness:
#    - Exactly 3 sprites keeps the scene sparse.
#    - Anchor, marker, and body colors are pairwise disjoint within a sample whenever
#      possible, so objects remain easy to segment and count.
#    - Markers are placed on the movement-facing outer side of each body (right for
#      horizontal, bottom for vertical), making the encoded distance easy to read.
# g) Degeneracy avoidance:
#    - Generated samples are constructed so `main(input)` differs from a D1-only "rank but
#      no offset" baseline; the area overshoot is therefore active rather than decorative.
#    - They also differ from the source task because no sprite marker color equals any anchor
#      color, so color matching cannot recover the output.
# h) New solver assumptions and generator guarantees:
#    - Each movable object is a two-color 8-connected component with exactly one minority
#      marker pixel.
#    - Sprite total areas are distinct, so size ranking is deterministic.
#    - Anchor count equals sprite count and anchors are isolated non-corner border pixels.
#    - Solved target positions keep the intended sprite fully in bounds.


_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (18, 22)
_GRID_WIDTH_RANGE = (18, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT = 3
_TOTAL_AREA_CHOICES = (5, 7, 9, 11)
_BODY_AREA_CHOICES = tuple(a - 1 for a in _TOTAL_AREA_CHOICES)
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


def _border_index(H: int, W: int, r: int, c: int) -> int:
    if r == 0:
        return int(c)
    if c == W - 1:
        return int((W - 1) + r)
    if r == H - 1:
        return int((W - 1) + (H - 1) + ((W - 1) - c))
    return int(2 * (W - 1) + (H - 1) + ((H - 1) - r))


def _anchor_axis(H: int, W: int, r: int, c: int) -> str:
    on_top_bottom = int(r) in (0, int(H - 1))
    on_left_right = int(c) in (0, int(W - 1))
    return "h" if bool(on_top_bottom and not on_left_right) else "v"


def _component_marker_color(grid: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(grid[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(grid: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    comp_set = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in comp_set and int(grid[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward ordered border anchors, with the same area controlling both pairing rank and overshoot distance.

    Input structure:
    - The background is the modal color.
    - Interior sprites are 2-color connected components, each with a single least-frequent marker pixel.
    - Border anchors are isolated non-bg singletons; their clockwise order matters, while their colors do not define the matching.

    Rule:
    - Sort anchors clockwise around the border.
    - Sort sprites by total component area from largest to smallest, pair the two lists, and keep the original axis rule:
      top/bottom anchors force horizontal motion, left/right anchors force vertical motion.
    - After selecting the paired anchor, move the sprite so its marker lands area cells past that anchor projection:
      `anchor_col + area` for horizontal slides, `anchor_row + area` for vertical slides.

    Key insight:
    Total sprite area is reused twice: first as the ranking key that chooses which anchor a sprite belongs to, and then as the numeric offset that tells how far beyond that anchor the marker must end up.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors: list[dict[str, int | str]] = []
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            isolated = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    isolated = False
                    break
            if isolated:
                anchors.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "axis": _anchor_axis(H, W, int(r), int(c)),
                    }
                )

    anchors.sort(key=lambda a: _border_index(H, W, int(a["r"]), int(a["c"])))
    assert anchors

    anchor_cells = {(int(a["r"]), int(a["c"])) for a in anchors}
    sprites: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if int(coords.shape[0]) == 1 and (int(coords[0, 0]), int(coords[0, 1])) in anchor_cells:
            continue
        marker = _component_marker_color(g, coords)
        if marker is None:
            continue
        mp = _marker_pixel(g, coords, int(marker))
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
        mp_r, mp_c = sprite["marker"]
        area = int(sprite["area"])
        if str(anchor["axis"]) == "h":
            target_c = int(anchor["c"]) + area
            dr = 0
            dc = int(target_c - mp_c)
        else:
            target_r = int(anchor["r"]) + area
            dr = int(target_r - mp_r)
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


def _marker_cell(body, axis):
    if axis == "h":
        max_c = max(c for _r, c in body)
        rows = [r for r, c in body if c == max_c]
        return (random.choice(rows), max_c + 1)
    max_r = max(r for r, _c in body)
    cols = [c for r, c in body if r == max_r]
    return (max_r + 1, random.choice(cols))


def _make_sprite(axis, body_area, body_color, marker_color, bg):
    body = _make_body(body_area)
    marker = _marker_cell(body, axis)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "area": int(body_area + 1),
        "axis": axis,
    }


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


def _edge_candidates(canvas, edge, area, bg):
    H, W = canvas.shape
    if edge in {"top", "bottom"}:
        r = 0 if edge == "top" else H - 1
        max_c = W - int(area) - 2
        cols = list(range(_ANCHOR_EDGE_MARGIN, max_c + 1))
        random.shuffle(cols)
        return [(r, c) for c in cols if _anchor_ok(canvas, r, c, bg)]
    c = 0 if edge == "left" else W - 1
    max_r = H - int(area) - 2
    rows = list(range(_ANCHOR_EDGE_MARGIN, max_r + 1))
    random.shuffle(rows)
    return [(r, c) for r in rows if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    area = int(sprite["area"])
    if anchor["axis"] == "h":
        target_c = int(anchor["c"]) + area
        if not (_INNER_BORDER_GAP <= target_c < W - _INNER_BORDER_GAP):
            return []
        return [
            (r, target_c)
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, target_c, bg)
        ]
    target_r = int(anchor["r"]) + area
    if not (_INNER_BORDER_GAP <= target_r < H - _INNER_BORDER_GAP):
        return []
    return [
        (target_r, c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, target_r, c, bg)
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


def _body_color(bg, forbidden, used):
    pool = [c for c in _COLORS if c not in forbidden and c not in used and c != int(bg)]
    if not pool:
        pool = [c for c in _COLORS if c not in forbidden and c != int(bg)]
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


def _edge_plan():
    plans = [
        ("top", "right", "bottom"),
        ("top", "right", "left"),
        ("top", "bottom", "left"),
        ("right", "bottom", "left"),
    ]
    return random.choice(plans)


def _place_anchors(out, inp, bg, total_areas):
    palette = [c for c in _COLORS if c != int(bg)]
    if len(palette) < _ANCHOR_COUNT:
        return None
    anchor_colors = random.sample(palette, _ANCHOR_COUNT)
    anchors = []
    for edge, color, area in zip(_edge_plan(), anchor_colors, total_areas):
        candidates = _edge_candidates(out, edge, int(area), bg)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        anchor = {
            "color": int(color),
            "r": int(r),
            "c": int(c),
            "axis": _anchor_axis(out.shape[0], out.shape[1], int(r), int(c)),
        }
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    anchors.sort(key=lambda a: _border_index(out.shape[0], out.shape[1], int(a["r"]), int(a["c"])))
    return anchors


def _sprite_palettes(bg, anchors):
    anchor_colors = {int(a["color"]) for a in anchors}
    pool = [c for c in _COLORS if c != int(bg) and c not in anchor_colors]
    if len(pool) < 2 * _ANCHOR_COUNT:
        return None
    marker_colors = random.sample(pool, _ANCHOR_COUNT)
    remaining = [c for c in pool if c not in marker_colors]
    body_colors = random.sample(remaining, _ANCHOR_COUNT)
    random.shuffle(marker_colors)
    random.shuffle(body_colors)
    return marker_colors, body_colors


def _place_aligned_sprites(out, anchors, total_areas, marker_colors, body_colors, bg):
    sprites = []
    used_body = set()
    for anchor, total_area, marker, body in zip(anchors, total_areas, marker_colors, body_colors):
        body_color = int(body)
        if body_color in used_body:
            body_color = _body_color(bg, {int(marker)} | {int(a["color"]) for a in anchors}, used_body)
        used_body.add(body_color)
        sprite = _make_sprite(str(anchor["axis"]), int(total_area - 1), body_color, int(marker), bg)
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
    total_areas = sorted(random.sample(_TOTAL_AREA_CHOICES, _ANCHOR_COUNT), reverse=True)
    anchors = _place_anchors(out, inp, bg, total_areas)
    if anchors is None:
        return None

    palettes = _sprite_palettes(bg, anchors)
    if palettes is None:
        return None
    marker_colors, body_colors = palettes

    sprites = _place_aligned_sprites(out, anchors, total_areas, marker_colors, body_colors, bg)
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
