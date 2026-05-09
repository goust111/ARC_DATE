# Variant of 581f7754
# Dimension: D17 — 符号编码：锚点颜色附加信息 (symbol_instructions)
# Change: 锚点颜色仅用于配对匹配，不编码其他信息... → 锚点旁边的额外像素编码偏移量

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D17 variant
# a) Concrete behavior:
#    The original task used a single border pixel as an anchor. Here that border
#    symbol becomes a same-colored border segment of length 1-3. The segment is
#    still placed on exactly one edge and still chooses the motion axis:
#    top/bottom => horizontal motion, left/right => vertical motion.
#    The new information is the segment length. We keep a fixed visual grammar:
#    horizontal border segments extend to the right, vertical border segments
#    extend downward. The sprite marker must end at the far endpoint of that
#    segment. Example: a top-edge red segment at columns 4..6 means a red marker
#    should finish at column 6, not column 4. Old rule: dc = 4 - marker_col.
#    New rule: dc = 6 - marker_col. The difference is the encoded offset +2.
# b) Complexity calibration:
#    The extra instruction only has three values (length 1/2/3 => offset 0/1/2),
#    which is small enough to infer from a few examples. The same anchor color may
#    control multiple sprites, so the same offset can repeat within one grid. We
#    also force at least one anchor length > 1 and reject samples where the new
#    output matches the old "ignore segment length" behavior.
# c) Old-code assumptions that break:
#    - Original anchor detection required a border pixel whose 4-neighbors are bg.
#      That fails immediately because a length-2/3 anchor has same-colored
#      neighbors on the border by design.
#    - Original `anchors_by_color[color] = (r,c)` assumed one reference cell.
#      Now the border symbol is a component, so the solver must recover its axis,
#      lead cell, and far endpoint.
#    - Original translation used the anchor cell itself as the target coordinate.
#      That is wrong in this variant; the target is the segment endpoint.
#    The sprite marker logic remains valid because markers are still minority-color
#    cells inside multi-color interior components.
# d) Placement legality:
#    The generator places border segments first, keeping their full 8-neighborhood
#    clear. Sprites are then placed at least one cell away from the border and from
#    each other, so border symbols never merge with sprites or with other anchors.
#    Output-first generation still works: we place each sprite at its solved target
#    endpoint, then scatter it along the orthogonal axis exactly as in the original.
# e) Visual observability:
#    A human can see that a 3-cell border run causes a farther final alignment than
#    a 1-cell run of the same type. Because the endpoint is always the rightmost or
#    bottommost cell of the border run, the relation is consistent across examples.
# f) Visual cleanliness:
#    Only one symbolic embellishment is added: the anchor pixel becomes a short,
#    isolated border line. No extra overlays are introduced on the sprites, so the
#    scene remains readable.
# g) Color inferability:
#    Colors still do only one job: pair sprite markers with border symbols. The new
#    instruction is encoded by visible segment length, not by color id or any hidden
#    ordering, so the task stays inferable from pixels alone.
# Solver assumptions after modification:
# - Each anchor color corresponds to one isolated straight border segment of length
#   1-3, entirely on a single edge, never touching a corner.
# - Horizontal anchor segments extend rightward and vertical ones extend downward;
#   therefore the far endpoint is the correct target coordinate.
# - Every movable singleton distractor uses an anchor color and is not itself part
#   of an anchor segment.
# The generator enforces all of these conditions.


def _is_border_component(
    coords: np.ndarray, color: int, H: int, W: int, use_segment_length: bool
) -> dict | None:
    pts = [(int(r), int(c)) for r, c in coords]
    if not pts:
        return None
    if any(not (r in (0, H - 1) or c in (0, W - 1)) for r, c in pts):
        return None

    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    area = len(pts)
    if not (1 <= area <= 3):
        return None

    if len(rows) == 1:
        r = rows[0]
        if r not in (0, H - 1):
            return None
        if cols[0] == 0 or cols[-1] == W - 1:
            return None
        if cols != list(range(cols[0], cols[0] + area)):
            return None
        lead_c = cols[0]
        target_c = cols[-1] if use_segment_length else lead_c
        return {
            "color": int(color),
            "axis": "h",
            "r": int(r),
            "c": int(lead_c),
            "length": int(area),
            "target_r": int(r),
            "target_c": int(target_c),
            "cells": {(int(r), int(c)) for c in cols},
        }

    if len(cols) == 1:
        c = cols[0]
        if c not in (0, W - 1):
            return None
        if rows[0] == 0 or rows[-1] == H - 1:
            return None
        if rows != list(range(rows[0], rows[0] + area)):
            return None
        lead_r = rows[0]
        target_r = rows[-1] if use_segment_length else lead_r
        return {
            "color": int(color),
            "axis": "v",
            "r": int(lead_r),
            "c": int(c),
            "length": int(area),
            "target_r": int(target_r),
            "target_c": int(c),
            "cells": {(int(r), int(c)) for r in rows},
        }

    return None


def _solve(grid: np.ndarray, *, use_segment_length: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- STEP 1) Detect anchors: straight isolated border segments (non-bg) ---
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    anchors_by_color: dict[int, dict] = {}
    anchor_cells: set[tuple[int, int]] = set()
    for comp in mono:
        anchor = _is_border_component(comp["coords"], int(comp["color"]), int(H), int(W), use_segment_length)
        if anchor is None:
            continue
        color = int(anchor["color"])
        if color not in anchors_by_color:
            anchors_by_color[color] = anchor
            anchor_cells |= anchor["cells"]

    assert anchors_by_color

    # --- STEP 2) Extract all foreground connected components as candidate sprites. ---
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
        cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
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

        anchor = anchors_by_color[int(marker_col)]
        dr = 0
        dc = 0
        if anchor["axis"] == "h":
            dc = int(anchor["target_c"] - mp[1])
        else:
            dr = int(anchor["target_r"] - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 3) Clear moved components before repasting them. ---
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    # --- STEP 4) Apply translations with clipping and non-overwrite semantics. ---
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their marker pixels land at the instructed endpoint of a border segment.

    Input structure:
    - bg is the most frequent color.
    - Interior sprites are multi-color components whose least frequent color acts as a marker.
    - Matching border symbols are 1-3 cell same-colored segments on a single edge, rather than lone anchor pixels.

    Rule:
    - Pair each sprite with the border segment sharing its marker color.
    - A top/bottom segment makes the sprite move horizontally until the marker reaches the segment's rightmost cell.
    - A left/right segment makes the sprite move vertically until the marker reaches the segment's bottommost cell.
    - Segment lengths 1, 2, and 3 therefore encode extra offsets of 0, 1, and 2 cells beyond the segment's leading cell.

    Key insight: The edge still chooses the movement axis, but the border symbol now carries a second instruction in its visible length, so the endpoint of the short run is the true destination.
    """
    return _solve(grid, use_segment_length=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Baseline solver that ignores the D17 length instruction and targets only the lead cell."""
    return _solve(grid, use_segment_length=False)


# --------------- generator (o5x) ---------------

# CFG: palette and canvas bounds.
_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

# CFG: placement isolation and anchor location.
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_ANCHOR_LENGTH_RANGE = (1, 3)

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400

# CFG: body geometry and shape mix.
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

# CFG: optional singleton distractors.
_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

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


def _segment_cells(axis, r, c, length):
    if axis == "h":
        return [(int(r), int(c) + i) for i in range(int(length))]
    return [(int(r) + i, int(c)) for i in range(int(length))]


def _segment_start_range(size, length):
    starts = list(range(_ANCHOR_EDGE_MARGIN, size - _ANCHOR_EDGE_MARGIN - int(length) + 1))
    if starts:
        return starts
    return list(range(_INNER_BORDER_GAP, size - int(length) - _INNER_BORDER_GAP + 1))


def _segment_ok(canvas, cells, bg):
    H, W = canvas.shape
    cell_set = {(int(r), int(c)) for r, c in cells}
    for r, c in cell_set:
        if canvas[int(r), int(c)] != int(bg):
            return False
        if not (int(r) in (0, H - 1) or int(c) in (0, W - 1)):
            return False
        if int(r) in (0, H - 1) and int(c) in (0, W - 1):
            return False
    for r, c in cell_set:
        r0, r1 = max(0, int(r) - _ISOLATION_RADIUS), min(H, int(r) + _ISOLATION_RADIUS + 1)
        c0, c1 = max(0, int(c) - _ISOLATION_RADIUS), min(W, int(c) + _ISOLATION_RADIUS + 1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if (int(rr), int(cc)) in cell_set:
                    continue
                if canvas[int(rr), int(cc)] != int(bg):
                    return False
    return True


def _anchor_segment_candidates(canvas, bg, length):
    H, W = canvas.shape
    rows = _segment_start_range(H, length)
    cols = _segment_start_range(W, length)
    out = []
    for c in cols:
        out.append(("h", 0, c, length))
        out.append(("h", H - 1, c, length))
    for r in rows:
        out.append(("v", r, 0, length))
        out.append(("v", r, W - 1, length))
    random.shuffle(out)
    return [
        (axis, r, c, length)
        for axis, r, c, length in out
        if _segment_ok(canvas, _segment_cells(axis, r, c, length), bg)
    ]


def _build_anchor(color, axis, r, c, length):
    cells = _segment_cells(axis, r, c, length)
    target_r = int(r + length - 1) if axis == "v" else int(r)
    target_c = int(c + length - 1) if axis == "h" else int(c)
    return {
        "color": int(color),
        "axis": axis,
        "r": int(r),
        "c": int(c),
        "length": int(length),
        "target_r": int(target_r),
        "target_c": int(target_c),
        "cells": cells,
    }


def _put_anchor(canvas, anchor):
    if anchor["axis"] == "h":
        patch = np.full((1, int(anchor["length"])), int(anchor["color"]), dtype=int)
    else:
        patch = np.full((int(anchor["length"]), 1), int(anchor["color"]), dtype=int)
    paste(canvas, patch, int(anchor["r"]), int(anchor["c"]))


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["target_c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["target_c"], bg)
        ]
    return [
        (anchor["target_r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["target_r"], c, bg)
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


def _add_distractors(out, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in range(_INNER_BORDER_GAP, out.shape[1] - _INNER_BORDER_GAP):
            if not _can_place(out, sprite, anchor["target_r"], c, bg):
                continue
            for r in range(_INNER_BORDER_GAP, out.shape[0] - _INNER_BORDER_GAP):
                if r != anchor["target_r"] and _can_place(inp, sprite, r, c, bg):
                    pairs.append((r, c))
        if not pairs:
            return
        in_r, col = random.choice(pairs)
        _place(out, sprite, anchor["target_r"], col, bg)
        _place(inp, sprite, in_r, col, bg)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _anchor_lengths(n):
    lengths = [random.randint(*_ANCHOR_LENGTH_RANGE) for _ in range(int(n))]
    if max(lengths) == 1:
        lengths[random.randrange(int(n))] = random.randint(2, _ANCHOR_LENGTH_RANGE[1])
    return lengths


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    lengths = _anchor_lengths(len(anchor_colors))
    anchors = []
    for color, length in zip(anchor_colors, lengths):
        candidates = _anchor_segment_candidates(out, bg, length)
        if not candidates:
            return None
        axis, r, c, _ = random.choice(candidates)
        anchor = _build_anchor(color, axis, r, c, length)
        anchors.append(anchor)
        _put_anchor(out, anchor)
        _put_anchor(inp, anchor)
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
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
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


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    return not bool(np.array_equal(_legacy_main(inp), out))


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

    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D17 input")
