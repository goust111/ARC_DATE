# Variant of 581f7754
# Dimension: D15 — 锚点边缘约束 (meta)
# Change: 锚点必须位于边界行/列上，且不在角落；4-邻域全部为背景色（孤立）... → 锚点可在角落

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D15 variant
# a) Corner anchors are now legal border anchors. A corner belongs to both a row edge
#    and a column edge, so the base task's "edge determines axis" rule is incomplete.
#    I resolve that ambiguity by using the marker pixel's current offset from the corner:
#    compare row-gap vs column-gap in the input; the larger gap picks the slide axis.
#    Example with a top-left corner anchor:
#    - old code path: any corner anchor falls into the left/right branch, so the sprite
#      always slides vertically until the marker reaches row 0.
#    - new code path: if the marker is much farther to the right than downward, the
#      sprite instead slides horizontally until the marker reaches column 0.
#    - difference: the same input can now produce a different output solely because the
#      anchor sits in a corner and the marker's stronger offset chooses the active edge.
# b) The change must stay visible. If corner anchors were merely allowed but every
#    generated sprite still used the old vertical behavior, the variant would collapse
#    back to the base task. The generator therefore forces corner anchors in most grids
#    and guarantees at least one horizontally resolved corner case.
# c) Hidden assumptions in the original `main()`:
#    - `horizontal = on_top_bottom and not on_left_right` silently treated every corner
#      as vertical. That is no longer valid.
#    - The rest of the solver assumes only one anchor per color and isolated anchors.
#      I keep those assumptions and make the generator satisfy them.
#    - The original generator assumed solved sprites stay one cell away from all borders.
#      That becomes invalid for corner anchors because aligning to row 0 or column 0/W-1
#      necessarily makes the solved sprite touch the border and sometimes clip.
# d) Placement legality changes only where needed:
#    - input sprites remain fully interior and separated as before;
#    - non-corner solved sprites remain fully interior as before;
#    - corner-resolved solved sprites may touch or clip against the relevant border, but
#      they still may not overlap existing non-bg cells, and the visible remainder must
#      stay large enough to read as a sprite.
# e) Visual observability:
#    - corner outputs visibly hug the border while non-corner outputs do not;
#    - among corner cases, some sprites end on the border row and others on the border
#      column, and that choice correlates with which input offset is larger.
# f) Cleanliness:
#    - at most one corner anchor is placed per grid, which keeps the border readable;
#    - corner-aligned sprites are kept a small distance from the anchor along the free
#      coordinate so the anchor itself remains visually distinct.
# g) No new color rule is introduced. Colors keep the original role: marker color picks
#    the anchor, body color only distinguishes sprites when possible.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift interior sprites so their rare-color marker lines up with the matching isolated border anchor.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are sprites with a dominant body color plus a rarer marker color.
    - Matching anchor pixels are isolated non-background cells on the border, and this variant also allows them in corners.

    Rule:
    - Pair each sprite with the anchor whose color matches the sprite's marker color.
    - A non-corner top/bottom anchor means a horizontal slide to the anchor column; a non-corner left/right anchor means a vertical slide to the anchor row.
    - A corner anchor is ambiguous, so compare the marker's column-gap and row-gap to that corner in the input: larger column-gap selects a horizontal slide, otherwise use a vertical slide.

    Key insight:
    - Normal anchors still encode the slide axis by which edge they sit on, but corner anchors need a second cue: the marker's stronger offset from the corner chooses which incident edge is active.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def anchor_axis(ar: int, ac: int, mr: int, mc: int) -> str:
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        if on_top_bottom and on_left_right:
            vertical_gap = abs(int(ar) - int(mr))
            horizontal_gap = abs(int(ac) - int(mc))
            return "h" if horizontal_gap > vertical_gap else "v"
        if on_top_bottom:
            return "h"
        return "v"

    # --- STEP 1) Detect anchors: isolated border pixels (non-bg) ---
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
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        # --- STEP 3) Decide marker color for this component. ---
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

        # --- STEP 4) Compute translation (dr,dc). ---
        dr = 0
        dc = 0
        if axis == "h":
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 5) Apply all moves. ---
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

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_TRIES = 60

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

_FORCE_CORNER_PROB = 0.8
_CORNER_FREE_GAP_RANGE = (2, 4)
_CORNER_EDGE_TOL = 1
_CORNER_MIN_VISIBLE_CELLS = 3

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
    cells = [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != int(bg))]
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "cells": cells,
        "area": len(cells),
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _placement_cells(canvas, sprite, marker_r, marker_c, bg, allow_clip=False):
    H, W = canvas.shape
    top, left = _top_left(sprite, marker_r, marker_c)
    visible = []
    marker_top, marker_left = sprite["offset"]
    marker_rr = top + int(marker_top)
    marker_cc = left + int(marker_left)
    if not (0 <= marker_rr < H and 0 <= marker_cc < W):
        return None
    for pr, pc, color in sprite["cells"]:
        rr = int(top) + int(pr)
        cc = int(left) + int(pc)
        if 0 <= rr < H and 0 <= cc < W:
            visible.append((rr, cc, int(color)))
        elif not allow_clip:
            return None
    if not visible:
        return None
    return visible


def _can_place(canvas, sprite, marker_r, marker_c, bg, *, allow_clip=False, require_inner_gap=True):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if require_inner_gap:
        if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
            return False
        if top + h > H - _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            return False
    visible = _placement_cells(canvas, sprite, marker_r, marker_c, bg, allow_clip=allow_clip)
    if visible is None:
        return False
    if allow_clip and len(visible) < min(_CORNER_MIN_VISIBLE_CELLS, int(sprite["area"])):
        return False
    occupied = {(int(r), int(c)) for r, c, _v in visible}
    for r, c, _v in visible:
        if int(canvas[int(r), int(c)]) != int(bg):
            return False
    for r, c, _v in visible:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (int(rr), int(cc)) in occupied:
                    continue
                if int(canvas[int(rr), int(cc)]) != int(bg):
                    return False
    return True


def _place(canvas, sprite, marker_r, marker_c, bg, *, allow_clip=False):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), clip=bool(allow_clip))


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


def _anchor_candidates(canvas, bg, *, corners: str = "any"):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append({"axis": "h", "r": 0, "c": int(c)})
        out.append({"axis": "h", "r": H - 1, "c": int(c)})
    for r in rows:
        out.append({"axis": "v", "r": int(r), "c": 0})
        out.append({"axis": "v", "r": int(r), "c": W - 1})
    corners_list = [
        {"axis": "corner", "r": 0, "c": 0},
        {"axis": "corner", "r": 0, "c": W - 1},
        {"axis": "corner", "r": H - 1, "c": 0},
        {"axis": "corner", "r": H - 1, "c": W - 1},
    ]
    if corners != "exclude":
        out.extend(corners_list)
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


def _corner_axis_for_marker(anchor, marker_r, marker_c):
    vertical_gap = abs(int(anchor["r"]) - int(marker_r))
    horizontal_gap = abs(int(anchor["c"]) - int(marker_c))
    return "h" if horizontal_gap > vertical_gap else "v"


def _corner_gap_values(limit, edge_index):
    lo, hi = _CORNER_FREE_GAP_RANGE
    vals = []
    for gap in range(int(lo), int(hi) + 1):
        v = int(gap) if int(edge_index) == 0 else int(limit - 1 - gap)
        if 0 < v < int(limit - 1):
            vals.append(v)
    return vals


def _corner_axis_feasible(sprite, anchor, move_axis):
    off_r, off_c = sprite["offset"]
    h, w = sprite["patch"].shape
    if move_axis == "h":
        clipped_cols = int(off_c) if int(anchor["c"]) == 0 else int(w - 1 - off_c)
        return clipped_cols <= _CORNER_EDGE_TOL
    clipped_rows = int(off_r) if int(anchor["r"]) == 0 else int(h - 1 - off_r)
    return clipped_rows <= _CORNER_EDGE_TOL


def _aligned_positions(canvas, sprite, anchor, move_axis, bg):
    H, W = canvas.shape
    allow_clip = anchor["axis"] == "corner"
    require_inner_gap = not allow_clip
    if move_axis == "h":
        if anchor["axis"] == "corner":
            rows = _corner_gap_values(H, anchor["r"])
        else:
            rows = range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        return [
            (int(r), int(anchor["c"]))
            for r in rows
            if _can_place(
                canvas,
                sprite,
                int(r),
                int(anchor["c"]),
                bg,
                allow_clip=allow_clip,
                require_inner_gap=require_inner_gap,
            )
        ]
    if anchor["axis"] == "corner":
        cols = _corner_gap_values(W, anchor["c"])
    else:
        cols = range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
    return [
        (int(anchor["r"]), int(c))
        for c in cols
        if _can_place(
            canvas,
            sprite,
            int(anchor["r"]),
            int(c),
            bg,
            allow_clip=allow_clip,
            require_inner_gap=require_inner_gap,
        )
    ]


def _scatter_positions(canvas, sprite, anchor, solved_marker, move_axis, bg):
    H, W = canvas.shape
    mr, mc = map(int, solved_marker)
    if move_axis == "h":
        out = []
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if int(c) == int(mc):
                continue
            if anchor["axis"] == "corner" and _corner_axis_for_marker(anchor, mr, c) != "h":
                continue
            if _can_place(canvas, sprite, mr, c, bg):
                out.append((int(mr), int(c)))
        return out
    out = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        if int(r) == int(mr):
            continue
        if anchor["axis"] == "corner" and _corner_axis_for_marker(anchor, r, mc) != "v":
            continue
        if _can_place(canvas, sprite, r, mc, bg):
            out.append((int(r), int(mc)))
    return out


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
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0), "cells": [(0, 0, int(anchor["color"]))], "area": 1}
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
    force_corner = bool(anchor_colors) and random.random() < _FORCE_CORNER_PROB
    for idx, color in enumerate(anchor_colors):
        if force_corner and idx == 0:
            candidates = _anchor_candidates(out, bg, corners="only")
        else:
            candidates = _anchor_candidates(out, bg, corners="exclude" if force_corner else "any")
        if not candidates:
            return None
        chosen = random.choice(candidates).copy()
        chosen["color"] = int(color)
        anchors.append(chosen)
        _put_anchor(out, chosen["r"], chosen["c"], color)
        _put_anchor(inp, chosen["r"], chosen["c"], color)
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
        if anchor["axis"] != "corner":
            continue
        n = int(counts[int(anchor["color"])])
        seq = ["h"]
        if n >= 2:
            seq.append("v")
        while len(seq) < n:
            seq.append("h" if len(seq) % 2 == 0 else "v")
        plan[int(anchor["color"])] = seq
    return plan


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    corner_plan = _corner_axis_plan(marker_colors, anchors)
    corner_ix = defaultdict(int)
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        if anchor["axis"] == "corner":
            move_axis = corner_plan[int(marker)][int(corner_ix[int(marker)])]
        else:
            move_axis = anchor["axis"]
        placed = False
        for _ in range(_SPRITE_BUILD_TRIES):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            if anchor["axis"] == "corner" and not _corner_axis_feasible(sprite, anchor, move_axis):
                continue
            positions = _aligned_positions(out, sprite, anchor, move_axis, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg, allow_clip=(anchor["axis"] == "corner"))
            used_body.add(body)
            sprites.append((sprite, anchor, marker_pos, move_axis))
            if anchor["axis"] == "corner":
                corner_ix[int(marker)] += 1
            placed = True
            break
        if not placed:
            return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos, move_axis in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, move_axis, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D15 input")
