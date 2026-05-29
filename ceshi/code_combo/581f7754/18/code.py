# Combo variant of 581f7754 (index 18)
# combo_spec: D2/0+D6/3+D15/0
# Name: 轴规则取反+原位留标记轮廓+角落锚点

from __future__ import annotations

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Seed task: a border anchor color matches a sprite's rare marker color; top/bottom anchors
#      cause horizontal slides and left/right anchors cause vertical slides; the source location is
#      fully erased; corners are excluded.
#    - Combo task:
#        1) invert the non-corner axis rule;
#        2) allow corner anchors;
#        3) after a sprite moves away, repaint its original 4-neighbor outline in marker color and
#           keep the original marker pixel(s) as part of that source trail.
#    - Example:
#      * input has a top-edge anchor of color 4 and an interior sprite whose only 4-colored cell is
#        at row 6, col 9.
#      * seed behavior: shift horizontally until the marker reaches anchor column; source vanishes.
#      * combo behavior: shift vertically until the marker reaches row 0, keep its input column, and
#        repaint the original sprite boundary in color 4 while also preserving the original marker cell.
#      * if the anchor is at the top-left corner instead, two border families are possible. The combo
#        resolves this by comparing the marker's current row-gap and column-gap to the corner: the
#        larger gap selects the slide axis. Larger column-gap => horizontal slide to the corner column;
#        otherwise vertical slide to the corner row.
# b) Complexity calibration:
#    - The rule still uses only a few visible categories: top/bottom, left/right, and corner.
#    - Corner samples are forced in most generated grids so the new branch is learnable from 2-4
#      examples rather than being a rare edge case.
#    - Corner tie cases are avoided: the generator enforces a strict gap dominance margin so the
#      chosen corner axis is visually recoverable and deterministic.
#    - Optional singleton distractors are disabled. Under the inverted rule they would solve onto the
#      border and become too easy to confuse with anchors, especially once trails are added.
# c) Implicit assumptions audited from main():
#    - Anchor detection assumes isolated border singletons; the generator preserves 8-neighbor
#      isolation and distinct colors per anchor.
#    - Marker-color inference assumes the least frequent component color identifies the sprite; the
#      generator keeps two-color sprites with exactly one marker pixel.
#    - Movement assumes one anchor per marker color; generator uses distinct anchor colors and maps
#      repeated sprites back to that single anchor.
#    - Trail repaint assumes source sites of different sprites are disjoint and do not collide with
#      anchors or solved sprites; generator checks trail placement on the output canvas explicitly.
#    - Corner handling assumes strict horizontal-gap vs vertical-gap comparison; generator constructs
#      corner inputs with a margin so equality never decides the branch.
# d) Placement legality:
#    - Output-first generation is still used.
#    - Solved sprites are placed first, touching the relevant border but not overlapping anchors.
#    - Input scattering is then chosen along the inverse construction axis while simultaneously
#      checking that the source sprite fits in the input and its marker-outline trail fits in the
#      output without overlap or 8-neighbor contact.
# e) Visual observability:
#    - In every non-corner case the solved sprite now hugs the border orthogonal to the seed task,
#      which is easy to notice across examples.
#    - Corner anchors are visible because solved sprites are kept a few cells away from the corner
#      along the free coordinate, leaving the anchor isolated and readable.
#    - Trails are monochrome and use the sprite's own marker color, so they clearly read as source
#      footprints instead of unrelated new objects.
# f) Visual cleanliness:
#    - At most one corner anchor is placed per grid.
#    - Source trails are separated from one another and from solved sprites by the same 8-neighbor
#      clearance used for sprites, preventing tangled overlays.
# g) Change inferability:
#    - No hidden color ordering is introduced. Color only indicates anchor-marker matching and trail
#      repainting, both directly visible in the grids.

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors with the inverted axis rule, while leaving marker-colored source outlines behind.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are sprites with a dominant body color and a rarer marker color.
    - Isolated non-background border singletons are anchors; unlike the seed task, anchors may also occupy corners.

    Rule:
    - Pair each sprite with the anchor whose color matches its marker color.
    - Non-corner top/bottom anchors now cause vertical motion so the marker reaches the anchor row; non-corner left/right anchors now cause horizontal motion so the marker reaches the anchor column.
    - Corner anchors use the marker's larger current offset from the corner to choose the axis: larger column-gap means horizontal motion, otherwise vertical motion. After moving the sprite, repaint its original 4-neighbor boundary cells in the marker color and keep the original marker cell(s) instead of clearing the whole source site.

    Key insight:
    - The border still selects how the sprite moves, but the mapping is intentionally inverted, and corner anchors need one more visible cue: whichever offset from the corner is larger becomes the direction that gets corrected.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

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

    def trail_cells(coords: np.ndarray, marker: int) -> list[tuple[int, int]]:
        cells = {(int(r), int(c)) for r, c in coords}
        out = {
            (int(r), int(c))
            for r, c in coords
            if int(g[int(r), int(c)]) == int(marker)
        }
        for r, c in cells:
            if any((int(rr), int(cc)) not in cells for rr, cc in neighbors4(int(r), int(c))):
                out.add((int(r), int(c)))
        return list(out) or [(int(coords[0, 0]), int(coords[0, 1]))]

    def anchor_axis(ar: int, ac: int, mr: int, mc: int) -> str:
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        if on_top_bottom and on_left_right:
            horizontal_gap = abs(int(ac) - int(mc))
            vertical_gap = abs(int(ar) - int(mr))
            return "h" if horizontal_gap > vertical_gap else "v"
        if on_top_bottom:
            return "v"
        return "h"

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

    moves: list[dict[str, object]] = []
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
        axis = anchor_axis(int(ar), int(ac), int(mp[0]), int(mp[1]))

        dr = 0
        dc = 0
        if axis == "h":
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker": int(marker_col),
                "trail": trail_cells(coords, int(marker_col)),
            }
        )

    out = g.copy()
    trail_owner = np.full((H, W), -1, dtype=int)

    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for idx, move in enumerate(moves):
        marker = int(move["marker"])
        for r, c in move["trail"]:
            out[int(r), int(c)] = int(marker)
            trail_owner[int(r), int(c)] = int(idx)

    for idx, move in enumerate(moves):
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if int(out[rr, cc]) == int(bg) or int(trail_owner[rr, cc]) == int(idx):
                out[rr, cc] = int(g[int(r), int(c)])
                trail_owner[rr, cc] = -1

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
_GEN_RETRY_LIMIT = 600
_SPRITE_BUILD_TRIES = 80

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

_FORCE_CORNER_PROB = 0.8
_CORNER_FREE_GAP_RANGE = (2, 4)
_CORNER_DOMINANCE_MARGIN = 1

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


def _marker_candidates(body):
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
    return (r0, c0, r1, c1), interior, exterior


def _marker_cell(body, side=None):
    (r0, c0, r1, c1), interior, exterior = _marker_candidates(body)
    if side is None:
        pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
        return random.choice(pool)

    def on_side(cell):
        r, c = cell
        if side == "top":
            return r <= r0
        if side == "bottom":
            return r >= r1
        if side == "left":
            return c <= c0
        return c >= c1

    outward = [cell for cell in exterior if on_side(cell)]
    if outward:
        return random.choice(outward)

    side_pool = [cell for cell in (exterior | interior) if on_side(cell)]
    if side_pool:
        return random.choice(side_pool)

    return random.choice(list(exterior | interior))


def _outline_mask(patch, bg):
    h, w = patch.shape
    mask = patch != int(bg)
    outline = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            for dr, dc in _NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w) or not bool(mask[rr, cc]):
                    outline[r, c] = True
                    break
    return outline


def _trail_mask(patch, marker_color, bg):
    return np.logical_or(_outline_mask(patch, bg), patch == int(marker_color))


def _make_sprite(area, body_color, marker_color, bg, side=None):
    body = _make_body(area)
    marker = _marker_cell(body, side=side)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    trail = np.full_like(patch, int(bg))
    trail[_trail_mask(patch, marker_color, bg)] = int(marker_color)
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
        "trail": trail,
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas, mask, top, left, bg):
    H, W = canvas.shape
    coords = np.argwhere(mask)
    occupied = {(int(top) + int(r), int(left) + int(c)) for r, c in coords}
    for pr, pc in coords:
        rr = int(top) + int(pr)
        cc = int(left) + int(pc)
        for nr in range(max(0, rr - _ISOLATION_RADIUS), min(H, rr + _ISOLATION_RADIUS + 1)):
            for nc in range(max(0, cc - _ISOLATION_RADIUS), min(W, cc + _ISOLATION_RADIUS + 1)):
                if (int(nr), int(nc)) in occupied:
                    continue
                if int(canvas[int(nr), int(nc)]) != int(bg):
                    return True
    return False


def _can_place(canvas, sprite, marker_r, marker_c, bg, margin=_INNER_BORDER_GAP):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < int(margin) or left < int(margin):
        return False
    if top + h > H - int(margin) or left + w > W - int(margin):
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _trail_sprite(sprite):
    return {"patch": sprite["trail"], "offset": sprite["offset"]}


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _is_corner(r, c, H, W):
    return int(r) in (0, int(H - 1)) and int(c) in (0, int(W - 1))


def _anchor_candidates(canvas, bg, *, corners="any"):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append({"kind": "top", "r": 0, "c": int(c)})
        out.append({"kind": "bottom", "r": H - 1, "c": int(c)})
    for r in rows:
        out.append({"kind": "left", "r": int(r), "c": 0})
        out.append({"kind": "right", "r": int(r), "c": W - 1})
    if corners != "exclude":
        out.extend(
            [
                {"kind": "corner", "r": 0, "c": 0},
                {"kind": "corner", "r": 0, "c": W - 1},
                {"kind": "corner", "r": H - 1, "c": 0},
                {"kind": "corner", "r": H - 1, "c": W - 1},
            ]
        )
    random.shuffle(out)
    filtered = []
    for cand in out:
        is_corner = _is_corner(cand["r"], cand["c"], H, W)
        if corners == "only" and not is_corner:
            continue
        if corners == "exclude" and is_corner:
            continue
        if _anchor_ok(canvas, cand["r"], cand["c"], bg):
            filtered.append(cand)
    return filtered


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _anchor_side(anchor, move_axis):
    kind = anchor["kind"]
    if kind == "top":
        return "top"
    if kind == "bottom":
        return "bottom"
    if kind == "left":
        return "left"
    if kind == "right":
        return "right"
    if move_axis == "h":
        return "left" if int(anchor["c"]) == 0 else "right"
    return "top" if int(anchor["r"]) == 0 else "bottom"


def _corner_free_values(limit, edge_index):
    lo, hi = _CORNER_FREE_GAP_RANGE
    out = []
    for gap in range(int(lo), int(hi) + 1):
        v = int(gap) if int(edge_index) == 0 else int(limit - 1 - gap)
        if 0 < v < int(limit - 1):
            out.append(v)
    return out


def _aligned_positions(canvas, sprite, anchor, move_axis, bg):
    H, W = canvas.shape
    if move_axis == "v":
        if anchor["kind"] == "corner":
            cols = _corner_free_values(W, anchor["c"])
        else:
            cols = [c for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP) if c != int(anchor["c"])]
        return [
            (int(anchor["r"]), int(c))
            for c in cols
            if _can_place(canvas, sprite, int(anchor["r"]), int(c), bg, margin=0)
        ]

    if anchor["kind"] == "corner":
        rows = _corner_free_values(H, anchor["r"])
    else:
        rows = [r for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP) if r != int(anchor["r"])]
    return [
        (int(r), int(anchor["c"]))
        for r in rows
        if _can_place(canvas, sprite, int(r), int(anchor["c"]), bg, margin=0)
    ]


def _corner_axis_ok(anchor, marker_r, marker_c, move_axis):
    vertical_gap = abs(int(marker_r) - int(anchor["r"]))
    horizontal_gap = abs(int(marker_c) - int(anchor["c"]))
    if move_axis == "h":
        return horizontal_gap > vertical_gap + _CORNER_DOMINANCE_MARGIN
    return vertical_gap >= horizontal_gap + _CORNER_DOMINANCE_MARGIN


def _scatter_positions(inp, out, sprite, anchor, solved_marker, move_axis, bg):
    H, W = inp.shape
    mr, mc = map(int, solved_marker)
    trail = _trail_sprite(sprite)
    if move_axis == "v":
        out_positions = []
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if int(r) == int(mr):
                continue
            if anchor["kind"] == "corner" and not _corner_axis_ok(anchor, r, mc, move_axis):
                continue
            if _can_place(inp, sprite, int(r), int(mc), bg) and _can_place(out, trail, int(r), int(mc), bg):
                out_positions.append((int(r), int(mc)))
        return out_positions

    out_positions = []
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        if int(c) == int(mc):
            continue
        if anchor["kind"] == "corner" and not _corner_axis_ok(anchor, mr, c, move_axis):
            continue
        if _can_place(inp, sprite, int(mr), int(c), bg) and _can_place(out, trail, int(mr), int(c), bg):
            out_positions.append((int(mr), int(c)))
    return out_positions


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _add_distractors(out, inp, anchors, bg):
    return


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
    force_corner = bool(anchor_colors) and random.random() < _FORCE_CORNER_PROB
    for idx, color in enumerate(anchor_colors):
        if force_corner and idx == 0:
            candidates = _anchor_candidates(out, bg, corners="only")
        else:
            candidates = _anchor_candidates(out, bg, corners="exclude" if force_corner else "any")
        if not candidates:
            return None
        anchor = random.choice(candidates).copy()
        anchor["color"] = int(color)
        anchors.append(anchor)
        _put_anchor(out, anchor["r"], anchor["c"], color)
        _put_anchor(inp, anchor["r"], anchor["c"], color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _corner_axis_plan(marker_colors, anchors):
    counts = Counter(map(int, marker_colors))
    plan = {}
    for anchor in anchors:
        if anchor["kind"] != "corner":
            continue
        n = int(counts[int(anchor["color"])])
        seq = []
        for i in range(n):
            seq.append("h" if i % 2 == 0 else "v")
        if n == 1 and random.random() < 0.5:
            seq[0] = "v"
        plan[int(anchor["color"])] = seq
    return plan


def _default_move_axis(anchor):
    if anchor["kind"] in ("top", "bottom"):
        return "v"
    if anchor["kind"] in ("left", "right"):
        return "h"
    raise ValueError("corner axis must be planned explicitly")


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    corner_plan = _corner_axis_plan(marker_colors, anchors)
    corner_ix = defaultdict(int)
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        move_axis = (
            corner_plan[int(marker)][int(corner_ix[int(marker)])]
            if anchor["kind"] == "corner"
            else _default_move_axis(anchor)
        )
        placed = False
        for _ in range(_SPRITE_BUILD_TRIES):
            body = _body_color(bg, marker, used_body)
            side = _anchor_side(anchor, move_axis)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg, side=side)
            positions = _aligned_positions(out, sprite, anchor, move_axis, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg)
            used_body.add(body)
            sprites.append((sprite, anchor, marker_pos, move_axis))
            if anchor["kind"] == "corner":
                corner_ix[int(marker)] += 1
            placed = True
            break
        if not placed:
            return None
    return sprites


def _scatter_sprites(inp, out, sprites, bg):
    random.shuffle(sprites)
    placements = []
    for sprite, anchor, marker_pos, move_axis in sprites:
        positions = _scatter_positions(inp, out, sprite, anchor, marker_pos, move_axis, bg)
        if not positions:
            return None
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
        _place(out, _trail_sprite(sprite), in_pos[0], in_pos[1], bg)
        placements.append(
            {
                "anchor": {"kind": anchor["kind"], "r": int(anchor["r"]), "c": int(anchor["c"]), "color": int(anchor["color"])},
                "move_axis": str(move_axis),
                "solved_marker": (int(marker_pos[0]), int(marker_pos[1])),
                "input_marker": (int(in_pos[0]), int(in_pos[1])),
            }
        )
    return placements


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt(return_meta=False):
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg)
    if sprites is None:
        return None

    placements = _scatter_sprites(inp, out, sprites, bg)
    if placements is None:
        return None

    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None

    if not return_meta:
        return inp

    return {
        "input": inp,
        "output": out,
        "bg": int(bg),
        "anchors": [
            {"kind": str(a["kind"]), "r": int(a["r"]), "c": int(a["c"]), "color": int(a["color"])}
            for a in anchors
        ],
        "placements": placements,
    }


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt(return_meta=False)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
