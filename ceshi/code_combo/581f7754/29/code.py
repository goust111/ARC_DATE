# Combo variant of 581f7754 (index 29)
# combo_spec: D16/1+D17/1
# Name: 重叠标记色+锚点旁编码偏移

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior / trace:
#    Original task: a sprite matches a same-colored isolated border pixel, slides
#    along one axis, and later writes only into background cells. Combo change:
#    the border cue is now a same-colored border segment of length 1-3, and the
#    marker must align with that segment's far endpoint (rightmost for top/bottom,
#    bottommost for left/right). After every sprite declares its translated landing
#    pixels, shared landing cells are recolored to conflict color 9.
#    Example: a top-edge red segment occupying cols 3..5 encodes target col 5.
#    Sprite A and sprite B both have red markers. Old logic would target col 3 and
#    keep the first landed color on overlaps. New logic targets col 5, so both
#    shapes shift two cells farther right; if both also place a pixel at (6,5),
#    output[6,5] becomes 9 instead of either original color.
# b) Complexity calibration:
#    Segment length has only three visible values, so offset decoding stays small.
#    The generator forces at least one anchor length > 1 and also forces at least
#    one pair of sprites sharing that long anchor color to overlap in the solved
#    state. This makes both new dimensions repeat within a sample and avoids a
#    visually near-original dataset.
# c) Implicit assumptions review:
#    - Original anchor detector expected isolated singleton border pixels; that no
#      longer holds, so the solver must parse monochromatic border components.
#    - Original translation targeted the anchor cell itself; combo targets the far
#      endpoint of the border segment.
#    - Original repaste order mattered because non-bg cells blocked later writes;
#      combo removes that order dependence by collecting all moved landing pixels
#      and resolving shared cells after counting them.
#    - Marker detection via least-frequent sprite color still works because input
#      sprites remain non-overlapping and collision color 9 never appears in input.
# d) Placement legality:
#    The generator still places input sprites with full separation so component
#    parsing is stable. Output planning allows exactly one deliberate overlap pair;
#    every other solved sprite stays 8-neighbor separated from anchors and peers.
#    Anchor segments are isolated along their whole 8-neighborhood and never touch
#    corners.
# e) Visual observability:
#    Humans can read the offset from the visible length-1/2/3 border segment and
#    can also see explicit 9-colored overlap cells in the output. The collision pair
#    is tied to a long anchor, so the same sample exhibits both rules together.
# f) Visual cleanliness:
#    Only one pair is forced to collide, overlap area is capped, and both colliding
#    sprites must retain at least two non-overlapped cells. This keeps the shapes
#    legible instead of collapsing into a solid blob.
# g) Change inferability:
#    Colors still only pair markers with anchors. Offset is encoded by segment
#    length, and conflict is encoded by a dedicated visible color 9. No hidden use
#    of color order or numeric ids is introduced.
# New solver assumptions enforced by the generator:
# - Each anchor color owns one isolated straight border segment of length 1-3.
# - Horizontal segments extend rightward and vertical segments extend downward, so
#   the far endpoint is well-defined.
# - Input movable objects are pairwise separated and do not include conflict color 9.


_CONFLICT_COLOR = 9


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


def _solve(grid: np.ndarray, *, use_segment_length: bool, use_conflict_color: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def apply_landing_pixels(base: np.ndarray, landed: list[tuple[int, int, int]]) -> np.ndarray:
        if not use_conflict_color:
            out = base.copy()
            for rr, cc, color in landed:
                if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(color)
            return out

        out = base.copy()
        blocked = out != int(bg)
        counts: dict[tuple[int, int], list[int]] = {}
        for rr, cc, color in landed:
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if bool(blocked[rr, cc]):
                continue
            counts.setdefault((int(rr), int(cc)), []).append(int(color))
        for (rr, cc), vals in counts.items():
            out[rr, cc] = int(vals[0]) if len(vals) == 1 else int(_CONFLICT_COLOR)
        return out

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
            anchor_cells |= set(anchor["cells"])

    assert anchors_by_color

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

    base = g.copy()
    for coords, _dr, _dc in moves:
        base[coords[:, 0], coords[:, 1]] = int(bg)

    landed: list[tuple[int, int, int]] = []
    for coords, dr, dc in moves:
        for r, c in coords:
            landed.append((int(r) + int(dr), int(c) + int(dc), int(g[int(r), int(c)])))

    return apply_landing_pixels(base, landed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border segments, and recolor shared landing cells with 9.

    Input structure:
    - The background is the modal color.
    - Interior movable objects are separated multi-color sprites, plus occasional singleton distractors, whose rare marker color identifies a border symbol.
    - Each border symbol is a 1-3 cell straight same-colored segment on one edge; its length encodes an extra offset.

    Rule:
    - Pair each movable object with the border segment of the same marker color.
    - A top/bottom segment makes the object slide horizontally until its marker reaches the segment's rightmost cell; a left/right segment makes it slide vertically until its marker reaches the segment's bottommost cell.
    - After clearing the source locations, combine all moved landing pixels at once: in-bounds cells hit by exactly one moved pixel keep that color, while cells hit by two or more moved pixels become conflict color 9.

    Key insight:
    - The border segment now carries two instructions at once: its edge still selects the motion axis, and its visible length selects the final endpoint; overlaps cannot be resolved greedily, because conflict color 9 depends on counting all moved pixels that target the same cell.
    """
    return _solve(grid, use_segment_length=True, use_conflict_color=True)


def _legacy_no_segment(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_segment_length=False, use_conflict_color=True)


def _legacy_no_conflict(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_segment_length=True, use_conflict_color=False)


# --------------- generator ---------------

_COLORS = tuple(range(10))
_USABLE_COLORS = tuple(c for c in _COLORS if c != _CONFLICT_COLOR)
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_ANCHOR_LENGTH_RANGE = (1, 3)

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500

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

_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

_COLLISION_MIN_OVERLAP = 2
_COLLISION_MAX_OVERLAP = 5
_VISIBLE_REMAINDER_MIN = 2
_PAIR_BUILD_ATTEMPTS = 60

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


def _sprite_pixels(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    coords = np.argwhere(patch != int(bg))
    return [
        (int(top + r), int(left + c), int(patch[int(r), int(c)]))
        for r, c in coords
    ]


def _sprite_cells(sprite, marker_r, marker_c, bg):
    return {(r, c) for r, c, _color in _sprite_pixels(sprite, marker_r, marker_c, bg)}


def _expand_cells(cells, shape):
    H, W = shape
    out = set()
    for r, c in cells:
        for dr, dc in _NEIGHBORS8:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


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


def _aligned_positions_free(shape, sprite, anchor):
    H, W = shape
    h, w = sprite["patch"].shape
    off_r, off_c = sprite["offset"]
    positions = []
    if anchor["axis"] == "h":
        col = int(anchor["target_c"])
        left = col - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            return []
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            top = int(r) - int(off_r)
            if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
                continue
            positions.append((int(r), int(col)))
        return positions

    row = int(anchor["target_r"])
    top = row - int(off_r)
    if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
        return []
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        left = int(c) - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            continue
        positions.append((int(row), int(c)))
    return positions


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
    pool = [c for c in _USABLE_COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _USABLE_COLORS if c not in {int(bg), int(marker)}]
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


def _anchor_lengths(n):
    lengths = [random.randint(*_ANCHOR_LENGTH_RANGE) for _ in range(int(n))]
    if max(lengths) == 1:
        lengths[random.randrange(int(n))] = random.randint(2, _ANCHOR_LENGTH_RANGE[1])
    return lengths


def _place_anchors(out, inp, bg):
    palette = [c for c in _USABLE_COLORS if c != int(bg)]
    anchor_colors = random.sample(palette, random.randint(*_ANCHOR_COUNT_RANGE))
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


def _marker_color_sequence(anchor_colors, collision_color):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    rest = [random.choice(anchor_colors) for _ in range(max(0, n_sprites - 2))]
    random.shuffle(rest)
    return [int(collision_color), int(collision_color), *rest]


def _anchor_keepout(anchors, shape):
    cells = set()
    for anchor in anchors:
        cells |= _expand_cells(set(anchor["cells"]), shape)
    return cells


def _make_item(sprite, anchor, marker_pos, bg):
    cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
    return {
        "sprite": sprite,
        "anchor": anchor,
        "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
        "cells": cells,
    }


def _occupied_cells(items):
    cells = set()
    for item in items:
        cells |= set(item["cells"])
    return cells


def _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg):
    occupied = _occupied_cells(items)
    positions = []
    for marker_pos in _aligned_positions_free(shape, sprite, anchor):
        cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
        if cells & anchor_keepout:
            continue
        if _expand_cells(cells, shape) & occupied:
            continue
        positions.append((int(marker_pos[0]), int(marker_pos[1])))
    return positions


def _colliding_positions(shape, sprite, anchor, target_item, other_items, anchor_keepout, bg):
    other_occupied = _occupied_cells(other_items)
    target_cells = set(target_item["cells"])
    positions = []
    for marker_pos in _aligned_positions_free(shape, sprite, anchor):
        cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
        if cells & anchor_keepout:
            continue
        overlap = cells & target_cells
        if not (_COLLISION_MIN_OVERLAP <= len(overlap) <= _COLLISION_MAX_OVERLAP):
            continue
        if len(cells - overlap) < _VISIBLE_REMAINDER_MIN:
            continue
        if len(target_cells - overlap) < _VISIBLE_REMAINDER_MIN:
            continue
        if _expand_cells(cells, shape) & other_occupied:
            continue
        positions.append((int(marker_pos[0]), int(marker_pos[1]), len(overlap)))
    return positions


def _render_output(shape, anchors, items, bg):
    H, W = shape
    base = np.full((H, W), int(bg), dtype=int)
    for anchor in anchors:
        for r, c in anchor["cells"]:
            base[int(r), int(c)] = int(anchor["color"])

    blocked = base != int(bg)
    landed: dict[tuple[int, int], list[int]] = {}
    for item in items:
        for rr, cc, color in _sprite_pixels(item["sprite"], item["marker_pos"][0], item["marker_pos"][1], bg):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if bool(blocked[rr, cc]):
                continue
            landed.setdefault((int(rr), int(cc)), []).append(int(color))

    out = base.copy()
    for (rr, cc), vals in landed.items():
        out[rr, cc] = int(vals[0]) if len(vals) == 1 else int(_CONFLICT_COLOR)
    return out


def _place_collision_pair(shape, anchors, collision_anchor, bg, used_body):
    anchor_keepout = _anchor_keepout(anchors, shape)
    for _ in range(_PAIR_BUILD_ATTEMPTS):
        body_a = _body_color(bg, collision_anchor["color"], used_body)
        sprite_a = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_a, collision_anchor["color"], bg)
        positions_a = _noncolliding_positions(shape, sprite_a, collision_anchor, [], anchor_keepout, bg)
        if not positions_a:
            continue
        random.shuffle(positions_a)
        for marker_a in positions_a[: min(len(positions_a), 8)]:
            item_a = _make_item(sprite_a, collision_anchor, marker_a, bg)
            for _ in range(_PAIR_BUILD_ATTEMPTS):
                body_b = _body_color(bg, collision_anchor["color"], used_body | {body_a})
                sprite_b = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_b, collision_anchor["color"], bg)
                positions_b = _colliding_positions(shape, sprite_b, collision_anchor, item_a, [], anchor_keepout, bg)
                if not positions_b:
                    continue
                positions_b.sort(key=lambda x: (abs(x[2] - 3), random.random()))
                marker_b = random.choice(positions_b[: min(len(positions_b), 6)])
                item_b = _make_item(sprite_b, collision_anchor, (marker_b[0], marker_b[1]), bg)
                used_body.add(int(body_a))
                used_body.add(int(body_b))
                return [item_a, item_b]
    return None


def _place_remaining_sprites(shape, anchors, marker_colors, bg, used_body, items):
    anchor_keepout = _anchor_keepout(anchors, shape)
    by_color = {a["color"]: a for a in anchors}
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        placed = False
        for _ in range(_PAIR_BUILD_ATTEMPTS):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            positions = _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            items.append(_make_item(sprite, anchor, marker_pos, bg))
            used_body.add(int(body))
            placed = True
            break
        if not placed:
            return False
    return True


def _add_distractors(inp, anchors, bg, items, shape):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return True

    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    anchor_keepout = _anchor_keepout(anchors, shape)
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        out_positions = _noncolliding_positions(shape, sprite, anchor, items, anchor_keepout, bg)
        if not out_positions:
            return True
        marker_pos = random.choice(out_positions)
        candidates = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not candidates:
            return False
        in_pos = random.choice(candidates)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
        items.append(_make_item(sprite, anchor, marker_pos, bg))
    return True


def _place_aligned_sprites(shape, anchors, marker_colors, collision_color, bg):
    by_color = {a["color"]: a for a in anchors}
    collision_anchor = by_color[int(collision_color)]
    used_body = set()
    items = _place_collision_pair(shape, anchors, collision_anchor, bg, used_body)
    if items is None:
        return None
    if not _place_remaining_sprites(shape, anchors, marker_colors[2:], bg, used_body, items):
        return None
    return items


def _scatter_sprites(inp, items, bg):
    shuffled = list(items)
    random.shuffle(shuffled)
    for item in shuffled:
        positions = _scatter_positions(inp, item["sprite"], item["anchor"], item["marker_pos"], bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, item["sprite"], in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool((out == int(_CONFLICT_COLOR)).any()):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    if bool(np.array_equal(_legacy_no_segment(inp), out)):
        return False
    if bool(np.array_equal(_legacy_no_conflict(inp), out)):
        return False
    return True


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    long_anchors = [a for a in anchors if int(a["length"]) > 1]
    collision_anchor = random.choice(long_anchors)
    marker_colors = _marker_color_sequence(anchor_colors, collision_anchor["color"])

    items = _place_aligned_sprites(out.shape, anchors, marker_colors, collision_anchor["color"], bg)
    if items is None:
        return None

    if not _scatter_sprites(inp, items, bg):
        return None
    if not _add_distractors(inp, anchors, bg, items, out.shape):
        return None

    out = _render_output(out.shape, anchors, items, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
