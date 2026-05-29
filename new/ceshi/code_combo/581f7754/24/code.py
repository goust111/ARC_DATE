# Combo variant of 581f7754 (index 24)
# combo_spec: D5/3+D6/2+D17/0
# Name: 旋转+重着色+锚点消失

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# a) Concrete behavior and trace
#    - Keep the original scene grammar: isolated border anchors, interior 2-color sprites,
#      and axis-of-motion determined by the anchor edge.
#    - The fused rule applies in a strict pipeline:
#        1. match a sprite to the anchor sharing its minority marker color
#        2. slide along the original axis until the marker reaches the anchor-guided row/column
#        3. rotate the translated sprite around that marker pixel, using the anchor color as a
#           fixed instruction code
#        4. erase the marker pixel, recolor every remaining sprite pixel to the anchor color,
#           and erase the anchor itself
#    - Small trace:
#        body=8, marker=3, left-edge anchor color 3 at row 6
#        original task: slide vertically until the marker reaches row 6; keep body color 8,
#        keep marker 3, keep border anchor 3.
#        combo task: perform the same vertical slide first; then rotate the sprite 90 degrees
#        clockwise because color 3 encodes one quarter-turn; then repaint the body to color 3,
#        delete the marker at the rotation pivot, and also delete the border anchor.
#        final output shows only a monochrome rotated body, no guide pixel and no marker.
#
# b) Complexity calibration
#    - The rotation vocabulary is intentionally small and fixed: 2 -> 0 turns, 3 -> 1 turn,
#      4 -> 2 turns. This keeps the symbolic layer inferable across 2-4 ARC examples.
#    - Every generated grid contains all three anchor colors and therefore all three rotation
#      instructions. Sprite count is 4-6 so some anchor colors repeat, giving the solver
#      repeated evidence for the same color->angle relation.
#    - Recoloring and marker deletion can otherwise make different sprites look interchangeable,
#      so the generator enforces unique body shapes and unique solved body silhouettes.
#
# c) Implicit-assumption review of the original main()
#    - Original anchor detection assumed isolated border singletons. This still holds and is
#      still the only legal way to read the hidden instruction color.
#    - Original marker inference assumed the least-frequent sprite color is the marker. The
#      combo generator keeps exactly one marker pixel, so that assumption still holds.
#    - Original movement state stored only a translation. That is no longer sufficient because
#      the destination geometry now depends on a post-translation rotation around the marker.
#    - Original paste logic copied each source pixel with its original value. That would break
#      two combo requirements at once: body recoloring and marker disappearance.
#    - Original output started from the input copy, which implicitly preserved anchors. The
#      combo must explicitly erase anchors before rendering the solved sprites.
#    - Rotating the full sprite can be visible while rotating the body-only output is not, so
#      the generator checks visibility after marker removal, not only on the original patch.
#
# d) Placement legality
#    - Generation stays output-first, but the solved canvas is built from rotated, recolored,
#      marker-erased bodies while the input canvas receives the unrotated original sprites.
#    - A temporary layout canvas keeps anchors present during solved placement so bodies do not
#      touch anchors or each other even though anchors are later erased from the final output.
#    - Solved sprites are kept fully inside the inner frame. The solver still supports clipping,
#      but the generator avoids heavy clipping because it would hide the rotation signal.
#
# e) Visual observability
#    - A human can see three independent changes in each pair:
#        * input has border anchors, output does not
#        * input bodies have their own colors, output bodies adopt the anchor color
#        * at least one color class repeatedly produces a rotated final orientation
#    - The color code is not hidden in numeric ordering at solve time; it is a fixed small
#      vocabulary repeated across all generated samples.
#
# f) Visual cleanliness
#    - Distractor singletons are disabled. In this combo they would either vanish completely or
#      create noisy exceptions that do not teach the rotation-recolor rule.
#    - The original 1-cell moat is preserved so anchors, source sprites, and solved bodies stay
#      visually separable after rotation.
#
# g) Change inferability
#    - The final recolor is structural and local: a sprite body always becomes the color of its
#      own matched anchor.
#    - The generator rejects sprites whose body-only solved silhouette is unchanged by 90 or
#      180 degrees. This prevents the rotation code from degenerating into an invisible factor.
#
# Assumption audit for the new main()
#    - Every moved sprite must contain a unique enough marker pixel to define the slide target
#      and rotation pivot. The generator guarantees exactly one minority marker cell.
#    - Marker colors must belong to the anchor instruction palette; otherwise the component is
#      ignored. The generator enforces that contract.
#    - Clearing anchors before stamping assumes solved sprites never rely on the border cell
#      itself for visibility. The generator keeps all solved bodies off the border.
#    - Non-overwrite stamping assumes solved bodies are mutually disjoint after rotation. The
#      generator validates placement on the solved layout before scattering to the input.


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


def _find_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = g.shape

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            value = int(g[r, c])
            if value == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and value not in anchors_by_color:
                anchors_by_color[int(value)] = (int(r), int(c))
    return anchors_by_color


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixels(g: np.ndarray, coords: np.ndarray, marker: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = _marker_pixels(g, coords, marker)
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


def _solve_core(
    grid: np.ndarray,
    *,
    rotate: bool,
    recolor_body: bool,
    erase_marker: bool,
    erase_anchor: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _find_anchors(g, bg)
    assert anchors_by_color

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    moves: list[dict[str, object]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        marker_pts: list[tuple[int, int]]
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            value = int(g[r, c])
            if value == int(bg) or value not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(value)
            marker_pts = [(int(r), int(c))]
        else:
            marker_pts = _marker_pixels(g, coords, int(marker_col))

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
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
        turns = int(_ROTATION_BY_COLOR.get(int(marker_col), 0)) if rotate else 0
        pivot_r = int(mp[0]) + int(dr)
        pivot_c = int(mp[1]) + int(dc)
        cells = _rotate_cells_about(translated, pivot_r, pivot_c, turns)
        moves.append(
            {
                "coords": coords,
                "cells": cells,
                "marker_color": int(marker_col),
            }
        )

    out = g.copy()
    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    if erase_anchor:
        for r, c in anchor_cells:
            out[int(r), int(c)] = int(bg)

    for move in moves:
        marker_col = int(move["marker_color"])
        for r, c, value in move["cells"]:
            if erase_marker and int(value) == int(marker_col):
                continue
            if not (0 <= int(r) < H and 0 <= int(c) < W):
                continue
            if int(out[int(r), int(c)]) != int(bg):
                continue
            out[int(r), int(c)] = int(marker_col) if recolor_body else int(value)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marked sprites onto their anchor-guided line, then rotate and repaint them.

    Input structure:
    - The background is the modal color.
    - Interior foreground components are small sprites with one minority-color marker pixel.
    - Isolated border singletons provide both the matching color and an extra rotation instruction.

    Rule:
    - Pair each sprite with the anchor of the same marker color.
    - Top/bottom anchors make the sprite slide horizontally to the anchor column; left/right anchors make it slide vertically to the anchor row.
    - After the slide, rotate the sprite around the translated marker pixel by the angle encoded by that anchor color, recolor every remaining body pixel to the anchor color, erase the marker, and erase the anchor.

    Key insight: The anchor position still tells you which row or column the marker must reach, but the anchor color now also tells you how that aligned sprite should turn before only its recolored body remains.
    """
    return _solve_core(
        grid,
        rotate=True,
        recolor_body=True,
        erase_marker=True,
        erase_anchor=True,
    )


# --------------- generator ---------------

_BG_COLOR_POOL = tuple(c for c in range(1, 9) if c not in _ANCHOR_COLORS)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_SPRITE_COUNT_RANGE = (4, 6)
_BODY_AREA_RANGE = (5, 13)
_GEN_RETRY_LIMIT = 500
_SPRITE_RETRY_LIMIT = 90

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


def _norm(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells = list(cells)
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r) - int(r0), int(c) - int(c0)) for r, c in cells}


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
        patch[int(rr - min_r), int(cc - min_c)] = int(value)
    return _make_sprite_from_patch(patch, (-int(min_r), -int(min_c)))


def _solved_sprite(base_sprite, marker_color, turns, bg):
    sprite = _rotate_sprite(base_sprite, turns, bg)
    patch = np.array(sprite["patch"], copy=True)
    off_r, off_c = sprite["offset"]
    mask = patch != int(bg)
    patch[mask] = int(marker_color)
    patch[int(off_r), int(off_c)] = int(bg)
    return _make_sprite_from_patch(patch, sprite["offset"])


def _body_signature_after_turn(base_sprite, turns, bg):
    rotated = _rotate_sprite(base_sprite, turns, bg)
    off_r, off_c = rotated["offset"]
    coords = [
        (r, c)
        for r, c, _value in _sprite_cells(rotated, bg)
        if not (int(r) == int(off_r) and int(c) == int(off_c))
    ]
    return frozenset(_norm(coords))


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


def _anchor_candidates(canvas, bg, axis=None):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    if axis in (None, "h"):
        for c in cols:
            out.append(("h", 0, c))
            out.append(("h", H - 1, c))
    if axis in (None, "v"):
        for r in rows:
            out.append(("v", r, 0))
            out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(a, r, c) for a, r, c in out if _isolated_bg_cell(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _clear_anchors(canvas, anchors, bg):
    for anchor in anchors:
        canvas[int(anchor["r"]), int(anchor["c"])] = int(bg)


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
    layout = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, layout, inp


def _place_anchors(layout, inp, bg):
    anchors = []
    colors = list(_ANCHOR_COLORS)
    random.shuffle(colors)
    axis_plan = ["h", "v", random.choice(["h", "v"])]
    random.shuffle(axis_plan)
    for color, axis in zip(colors, axis_plan):
        candidates = _anchor_candidates(layout, bg, axis=axis)
        if not candidates:
            return None
        chosen_axis, r, c = random.choice(candidates)
        anchor = {
            "color": int(color),
            "r": int(r),
            "c": int(c),
            "axis": chosen_axis,
            "turns": int(_ROTATION_BY_COLOR[int(color)]),
        }
        anchors.append(anchor)
        _put_anchor(layout, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchors


def _marker_color_sequence():
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = list(_ANCHOR_COLORS)
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(_ANCHOR_COLORS))
    random.shuffle(marker_colors)
    return marker_colors


def _build_base_sprite(area, body_color, marker_color, turns, bg):
    for _ in range(_SPRITE_RETRY_LIMIT):
        body = _make_body(area)
        marker = _marker_cell(body)
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
        sprite = _make_sprite_from_patch(patch, (marker[0] - r0, marker[1] - c0))
        base_sig = frozenset(_norm(body))
        no_turn_sig = _body_signature_after_turn(sprite, 0, bg)
        turn_sig = _body_signature_after_turn(sprite, turns, bg)
        if turns and turn_sig == no_turn_sig:
            continue
        return sprite, base_sig, turn_sig
    return None


def _place_solved_sprites(layout, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body_colors = set()
    used_base_shapes = set()
    used_solved_shapes = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        built = None
        for _ in range(_SPRITE_RETRY_LIMIT):
            body = _body_color(bg, used_body_colors)
            built_sprite = _build_base_sprite(
                random.randint(*_BODY_AREA_RANGE),
                body_color=body,
                marker_color=marker,
                turns=anchor["turns"],
                bg=bg,
            )
            if built_sprite is None:
                continue
            base_sprite, base_sig, solved_sig = built_sprite
            if base_sig in used_base_shapes or solved_sig in used_solved_shapes:
                continue
            solved_sprite = _solved_sprite(base_sprite, marker, anchor["turns"], bg)
            positions = _aligned_positions(layout, solved_sprite, anchor, bg)
            if not positions:
                continue
            built = (body, base_sprite, solved_sprite, base_sig, solved_sig, positions)
            break
        if built is None:
            return None
        body, base_sprite, solved_sprite, base_sig, solved_sig, positions = built
        used_body_colors.add(int(body))
        used_base_shapes.add(base_sig)
        used_solved_shapes.add(solved_sig)
        marker_pos = random.choice(positions)
        _place(layout, solved_sprite, marker_pos[0], marker_pos[1], bg)
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


def _fusion_visibility_metrics(inp: np.ndarray) -> dict[str, int | bool]:
    combo = _solve_core(inp, rotate=True, recolor_body=True, erase_marker=True, erase_anchor=True)
    no_rotation = _solve_core(inp, rotate=False, recolor_body=True, erase_marker=True, erase_anchor=True)
    no_recolor = _solve_core(inp, rotate=True, recolor_body=False, erase_marker=True, erase_anchor=True)
    bg = int(get_bg(combo))
    border_fg = int(np.count_nonzero(combo[0, :] != bg))
    border_fg += int(np.count_nonzero(combo[-1, :] != bg))
    if combo.shape[0] > 2:
        border_fg += int(np.count_nonzero(combo[1:-1, 0] != bg))
        border_fg += int(np.count_nonzero(combo[1:-1, -1] != bg))
    return {
        "rotation_changes_output": bool(not np.array_equal(combo, no_rotation)),
        "recolor_changes_output": bool(not np.array_equal(combo, no_recolor)),
        "border_foreground": int(border_fg),
    }


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not np.array_equal(main(inp), out):
        return False
    metrics = _fusion_visibility_metrics(inp)
    return bool(
        metrics["rotation_changes_output"]
        and metrics["recolor_changes_output"]
        and metrics["border_foreground"] == 0
    )


def _gen_attempt():
    bg, layout, inp = _new_canvas_pair()
    anchors = _place_anchors(layout, inp, bg)
    if anchors is None:
        return None
    sprites = _place_solved_sprites(layout, anchors, _marker_color_sequence(), bg)
    if sprites is None:
        return None
    if not _scatter_sprites(inp, sprites, bg):
        return None
    out = np.array(layout, copy=True)
    _clear_anchors(out, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input for 581f7754/24")
