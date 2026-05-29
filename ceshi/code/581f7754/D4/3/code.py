# Variant of 581f7754
# Dimension: D4 — 目标位置：锚点类型与位置 (dst)
# Change: 边界上的孤立单像素（非角落），4-邻域均为背景色，1-3个不同颜色锚点... → 锚点为短线段而非单像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     original: each target anchor is a single isolated border pixel; solver aligns the
#       sprite marker to that pixel's row or column.
#     variant: each target anchor is instead a same-color border segment of length 2 or 3.
#       The sprite marker aligns to the segment midpoint. The segment orientation is always
#       parallel to the border it lies on, so top/bottom segments still imply horizontal
#       sliding and left/right segments still imply vertical sliding.
#   concrete_visualization:
#     - A horizontal anchor is a 1x2 or 1x3 colored dash on the top or bottom edge.
#       Example: top edge cells (0,5)-(0,7) are color 4. The target column is the dash
#       midpoint column: for length 3 it is the center cell, for length 2 we discretize the
#       midpoint by rounding toward the larger index, i.e. cols [5,6] target col 6.
#     - A vertical anchor is a 2x1 or 3x1 dash on the left or right edge, with the target
#       row defined analogously.
#     - Sprites remain 2-color connected objects with a rare-color marker pixel; multiple
#       sprites may share one anchor color and therefore align to the same segment midpoint.
#   behavior_trace:
#     - Original: a top-edge singleton anchor at (0,5) makes every matching sprite slide
#       horizontally until its marker column becomes 5.
#     - Variant: a top-edge length-3 dash at (0,4..6) still causes horizontal sliding, but
#       the marker must stop at col 5 because that is the dash midpoint; a length-2 dash at
#       (0,4..5) stops at col 5 after midpoint rounding.
#     - Difference: solver can no longer inspect one pixel and use its coordinate directly.
#       It must first recover a border line segment, infer its axis, then compute its target
#       midpoint coordinate before translating the sprite.
#   complexity_calibration:
#     - Length-3 anchors are visually clearest, so the generator biases toward them.
#     - Length-2 anchors are still generated sometimes, which prevents the task from
#       collapsing back to the original singleton-anchor behavior, but they are not the
#       majority because discrete midpoint tie-breaking is less obvious.
#     - Anchor colors stay limited to 1-3 and are reused across 3-5 sprites, so each sample
#       still contains repeated evidence for the color-to-anchor matching rule.
#   implicit_assumption_audit:
#     - Original STEP 1 in main() assumes any isolated border non-bg pixel is an anchor.
#       That is invalid now: a single border pixel might be noise, while true anchors are
#       border components of size 2 or 3 with straight geometry.
#     - Original anchor_cells stores one coordinate per color; now it must store every cell
#       of every detected segment so singleton distractors are not mistaken for anchor cells.
#     - Original translation uses the anchor cell position directly; now it must use the
#       discrete midpoint coordinate.
#     - Generator helpers `_anchor_ok`, `_anchor_candidates`, `_put_anchor`,
#       `_aligned_positions`, and `_add_distractors` all depend on anchors being size-1 and
#       therefore need synchronized updates.
#   placement_legality:
#     - Each anchor segment is isolated by a 1-cell 8-neighborhood gap, never touches a
#       corner, and never overlaps another object.
#     - Sprites are placed fully inside the canvas with the same padding rules as before.
#       In solved/output grids, their marker aligns to the anchor midpoint but sprites remain
#       at least one cell away from the border, so the anchor segment stays visually separate.
#     - Input grids are produced by scattering along the inverse axis, exactly as before, so
#       running the solver reconstructs the settled output deterministically.
#   observability:
#     - Human-solvable signal: the border target is visibly a dash instead of a dot, and the
#       sprite marker aligns with the center of that dash rather than an endpoint.
#     - The generator mixes top/bottom and left/right anchors and reuses colors across
#       multiple sprites, so examples show both the axis rule and the midpoint rule.
#   cleanliness:
#     - Objects remain sparse, do not touch, and use only one body color plus one marker
#       color each, preserving the original visual clarity.
#   color_inference:
#     - No color-id arithmetic or sorting by numeric color is used for visible behavior.
#       Colors only serve as equality labels between a sprite marker and its anchor.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites so their marker pixels line up with matching border dash anchors.

    Input structure:
    - bg is the dominant color.
    - Interior sprites are multi-color connected components; each has a rare marker color.
    - Border anchors are isolated same-color line segments of length 2 or 3 on one edge.

    Rule:
    - Pair each sprite with the border segment whose color matches the sprite marker color.
    - A top/bottom segment makes the sprite slide horizontally until the marker reaches the segment midpoint column.
    - A left/right segment makes the sprite slide vertically until the marker reaches the segment midpoint row.

    Key insight:
    - The anchor is no longer one pixel: first detect a short border dash, infer its edge-aligned direction, then use the dash midpoint as the alignment target.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _detect_anchors() -> list[dict]:
        anchors: list[dict] = []
        comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
        for comp in comps:
            coords = comp["coords"]
            if coords.size == 0:
                continue
            color = int(comp["color"])
            cells = {(int(r), int(c)) for r, c in coords}
            if len(cells) not in (2, 3):
                continue

            rows = sorted({r for r, _c in cells})
            cols = sorted({c for _r, c in cells})
            axis = None
            target_r = None
            target_c = None

            if len(rows) == 1 and rows[0] in (0, H - 1) and cols == list(range(cols[0], cols[0] + len(cols))):
                axis = "h"
                target_r = rows[0]
                target_c = cols[0] + len(cols) // 2
            elif len(cols) == 1 and cols[0] in (0, W - 1) and rows == list(range(rows[0], rows[0] + len(rows))):
                axis = "v"
                target_r = rows[0] + len(rows) // 2
                target_c = cols[0]
            if axis is None:
                continue

            isolated = True
            for r, c in cells:
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells and int(g[rr, cc]) != bg:
                        isolated = False
                        break
                if not isolated:
                    break
            if not isolated:
                continue

            anchors.append(
                {
                    "color": color,
                    "axis": axis,
                    "target_r": int(target_r),
                    "target_c": int(target_c),
                    "cells": cells,
                }
            )
        return anchors

    anchors = _detect_anchors()
    assert anchors

    anchors_by_color: dict[int, dict] = {}
    anchor_cells: set[tuple[int, int]] = set()
    for anchor in anchors:
        color = int(anchor["color"])
        if color not in anchors_by_color:
            anchors_by_color[color] = anchor
        anchor_cells |= anchor["cells"]

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
        component_cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(r, c):
                if (int(rr), int(cc)) in component_cells and int(g[int(rr), int(cc)]) != int(marker):
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
            if v == bg or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = v

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
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    return out


# --------------- generator (o5x) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_ANCHOR_SEGMENT_LENGTHS = (2, 3)
_ANCHOR_LEN3_PROB = 0.7
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
        return [(int(r), int(c + k)) for k in range(int(length))]
    return [(int(r + k), int(c)) for k in range(int(length))]


def _anchor_segment_ok(canvas, axis, r, c, length, bg):
    cells = _segment_cells(axis, r, c, length)
    H, W = canvas.shape
    for rr, cc in cells:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if rr in (0, H - 1) and cc in (0, W - 1):
            return False
    rows = [rr for rr, _cc in cells]
    cols = [cc for _rr, cc in cells]
    r0, r1 = max(0, min(rows) - _ISOLATION_RADIUS), min(H, max(rows) + _ISOLATION_RADIUS + 1)
    c0, c1 = max(0, min(cols) - _ISOLATION_RADIUS), min(W, max(cols) + _ISOLATION_RADIUS + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    out = []
    for length in _ANCHOR_SEGMENT_LENGTHS:
        col_start = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN - length + 1))
        if not col_start:
            col_start = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP - length + 1))
        row_start = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN - length + 1))
        if not row_start:
            row_start = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP - length + 1))

        for c in col_start:
            out.append(("h", 0, c, length))
            out.append(("h", H - 1, c, length))
        for r in row_start:
            out.append(("v", r, 0, length))
            out.append(("v", r, W - 1, length))
    random.shuffle(out)
    return [(axis, r, c, length) for axis, r, c, length in out if _anchor_segment_ok(canvas, axis, r, c, length, bg)]


def _put_anchor(canvas, anchor):
    if anchor["axis"] == "h":
        patch = np.full((1, anchor["length"]), int(anchor["color"]), dtype=int)
    else:
        patch = np.full((anchor["length"], 1), int(anchor["color"]), dtype=int)
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


def _choose_anchor_length():
    return 3 if random.random() < _ANCHOR_LEN3_PROB else 2


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
        preferred_length = _choose_anchor_length()
        pool = [item for item in candidates if item[3] == preferred_length] or candidates
        axis, r, c, length = random.choice(pool)
        if axis == "h":
            target_r, target_c = int(r), int(c + length // 2)
        else:
            target_r, target_c = int(r + length // 2), int(c)
        anchor = {
            "color": int(color),
            "r": int(r),
            "c": int(c),
            "axis": axis,
            "length": int(length),
            "target_r": int(target_r),
            "target_c": int(target_c),
        }
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
    return bool(np.array_equal(main(inp), out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D4 input")
