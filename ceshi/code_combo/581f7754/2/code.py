# Combo variant of 581f7754 (index 2)
# combo_spec: D2/3+D5/0
# Name: 双轴对角滑动+环绕

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for combo D2/3 + D5/0:
# - Concrete behavior:
#   Each anchor still matches sprites by marker color, but it no longer chooses a single slide
#   axis. Instead it defines one exact destination cell one step inside the border:
#   top -> (1,c), bottom -> (H-2,c), left -> (r,1), right -> (r,W-2). The whole sprite is
#   translated by (dr, dc) so its marker lands on that cell, and any translated pixel that
#   leaves the grid re-enters from the opposite side by modulo arithmetic.
# - Behavior trace:
#   Example on a 12x12 grid: a top anchor of color 4 sits at (0,2), so the marker target is
#   (1,2). Suppose a sprite with marker color 4 starts with its marker at (7,8).
#   Original task: only the column matters, so dc = -6 and the marker ends at (7,2).
#   D2-only change: dr = -6 and dc = -6, so the marker ends at (1,2).
#   Fused change: the same diagonal shift happens, but if one body pixel was at (5,0), after
#   translation it would land at (-1,-6) and reappear at (11,6) instead of being clipped.
#   The visible delta is therefore both "marker reaches the inward-neighbor cell" and "missing
#   off-grid pieces are preserved on the opposite side".
# - Complexity calibration:
#   The dual-axis target gives each anchor color a unique landing cell, so the generator uses
#   exactly one sprite per anchor color and disables repeated-color sprites and singleton
#   distractors. This prevents ambiguous stacking and keeps the diagonal rule readable. At least
#   one generated sprite must genuinely wrap and every generated sprite must move on both axes,
#   so samples cannot collapse back to the original axis-only task.
# - Implicit-assumption audit of `main()`:
#   Anchor detection, component extraction, and marker-color inference survive unchanged because
#   they do not depend on the movement rule. The old assumption "one of dr/dc is zero" is false
#   now, so move collection must compute both components from the inward target cell. The old
#   clipping check `0 <= rr < H and 0 <= cc < W` is also false now; paste coordinates must be
#   reduced modulo grid size. "Clear all sources first" still holds and remains necessary so the
#   move is simultaneous. "Do not overwrite non-background cells" is retained, otherwise wrapped
#   fragments could erase anchors or other sprites.
# - Placement legality:
#   Input sprites are placed fully inside the border on an anchor-populated canvas, so object
#   recognition never depends on wraparound. Output sprites are built on an anchor-free sprite
#   canvas using wrapped placement checks: no wrapped pixel may collide with another sprite, with
#   any anchor cell, or with its own wrapped image. The only allowed anchor contact is the marker
#   target cell sitting immediately inside its matching anchor.
# - Visual observability and neatness:
#   Using 2-3 anchors with one sprite each gives repeated evidence for the color-pairing rule
#   without forcing collisions at the unique target cells. One designated anchor is chosen near a
#   corner-side band so its solved sprite visibly wraps, often splitting across two sides. The
#   input placements are forced off both the target row and the target column, which makes the
#   diagonal motion obvious across a few examples.
# - Human-inferable structure:
#   Colors only encode anchor-marker pairing. No ordering over color ids is used. The visible
#   structure is purely geometric: align the marker to the inward cell next to the same-colored
#   anchor, and preserve off-grid parts by toroidal wrap instead of clipping.


def _collect_move_data(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def anchor_target(r: int, c: int) -> tuple[int, int]:
        tr = int(r)
        tc = int(c)
        if tr == 0:
            tr = 1
        elif tr == H - 1:
            tr = H - 2
        if tc == 0:
            tc = 1
        elif tc == W - 1:
            tc = W - 2
        return int(tr), int(tc)

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

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
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
        tr, tc = anchor_target(int(ar), int(ac))
        dual_dr = int(tr - mp[0])
        dual_dc = int(tc - mp[1])

        axis_dr = 0
        axis_dc = 0
        if int(ar) in (0, int(H - 1)):
            axis_dc = int(ac - mp[1])
        else:
            axis_dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords.astype(int),
                "dual_dr": int(dual_dr),
                "dual_dc": int(dual_dc),
                "axis_dr": int(axis_dr),
                "axis_dc": int(axis_dc),
            }
        )

    return g, bg, moves


def _apply_moves(
    g: np.ndarray,
    bg: int,
    moves: list[dict[str, object]],
    *,
    wrap: bool,
    move_kind: str,
) -> np.ndarray:
    H, W = g.shape
    out = g.copy()

    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    dr_key = f"{move_kind}_dr"
    dc_key = f"{move_kind}_dc"
    for move in moves:
        coords = move["coords"]
        dr = int(move[dr_key])
        dc = int(move[dc_key])
        for r, c in coords:
            rr = int(r) + dr
            cc = int(c) + dc
            if wrap:
                rr %= H
                cc %= W
                if int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])
            else:
                if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])

    return out


def variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g, bg, moves = _collect_move_data(grid)
    H, W = g.shape
    wrapped_pixels = 0
    wrapped_components = 0
    diagonal_components = 0
    axis_changed_components = 0

    for move in moves:
        dual_dr = int(move["dual_dr"])
        dual_dc = int(move["dual_dc"])
        axis_dr = int(move["axis_dr"])
        axis_dc = int(move["axis_dc"])
        coords = move["coords"]

        if dual_dr != 0 and dual_dc != 0:
            diagonal_components += 1
        if dual_dr != axis_dr or dual_dc != axis_dc:
            axis_changed_components += 1

        comp_wrap = 0
        for r, c in coords:
            rr = int(r) + dual_dr
            cc = int(c) + dual_dc
            if not (0 <= rr < H and 0 <= cc < W):
                comp_wrap += 1
        wrapped_pixels += comp_wrap
        if comp_wrap:
            wrapped_components += 1

    clipped = _apply_moves(g, bg, moves, wrap=False, move_kind="dual")
    wrapped = _apply_moves(g, bg, moves, wrap=True, move_kind="dual")
    axis_wrapped = _apply_moves(g, bg, moves, wrap=True, move_kind="axis")
    return {
        "move_count": int(len(moves)),
        "diagonal_components": int(diagonal_components),
        "axis_changed_components": int(axis_changed_components),
        "wrapped_pixels": int(wrapped_pixels),
        "wrapped_components": int(wrapped_components),
        "changes_vs_clipped": bool(not np.array_equal(wrapped, clipped)),
        "changes_vs_axis_only": bool(not np.array_equal(wrapped, axis_wrapped)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior marker-bearing sprite so its marker lands in the unique interior cell next to the matching border anchor.

    Input structure:
    - The background is the most common grid color.
    - Interior 8-connected components are mostly two-color sprites: a larger body plus a rarer marker pixel.
    - Isolated border singletons act as anchors, and a sprite is paired to the anchor with the same marker color.

    Rule:
    - Convert each anchor into the one-step-inward target cell beside it, then translate the whole sprite in both row and column so its marker lands exactly on that cell.
    - Clear the original sprite locations before pasting, keep anchors fixed, and refuse to overwrite existing non-background cells.
    - If a translated pixel leaves the grid, wrap it to the opposite side instead of clipping it away.

    Key insight: The border anchor now specifies a single destination cell rather than a slide axis, so the motion is diagonal in general, and the final paste happens on a torus.
    """
    g, bg, moves = _collect_move_data(grid)
    return _apply_moves(g, bg, moves, wrap=True, move_kind="dual")


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_WRAP_EDGE_BAND = 3

_ANCHOR_COUNT_RANGE = (2, 3)
_BODY_AREA_RANGE = (4, 13)
_WRAP_BODY_AREA_MIN = 7
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_TRIES = 100
_STRONG_SHIFT_MIN = 2
_MIN_WRAP_PIXELS = 2

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
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "area": int((patch != int(bg)).sum()),
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


def _wrapped_sprite_cells(canvas_shape, sprite, marker_r, marker_c, bg):
    H, W = canvas_shape
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    wrapped_pixels = 0
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            color = int(patch[pr, pc])
            if color == int(bg):
                continue
            raw_r = top + pr
            raw_c = left + pc
            if not (0 <= raw_r < H and 0 <= raw_c < W):
                wrapped_pixels += 1
            cells.append((int(raw_r % H), int(raw_c % W), color))
    return cells, int(wrapped_pixels)


def _touches_foreground_cells(canvas, occupied_cells, bg):
    H, W = canvas.shape
    occupied = set(occupied_cells)
    for r, c in occupied:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (int(rr), int(cc)) in occupied:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return True
    return False


def _touches_anchor_illegally(occupied_cells, anchors, own_anchor, target):
    target = (int(target[0]), int(target[1]))
    own_cell = (int(own_anchor["r"]), int(own_anchor["c"]))
    for rr, cc in occupied_cells:
        for anchor in anchors:
            ar = int(anchor["r"])
            ac = int(anchor["c"])
            if (rr, cc) == (ar, ac):
                return True
            if max(abs(int(rr) - ar), abs(int(cc) - ac)) > 1:
                continue
            if (rr, cc) == target and (ar, ac) == own_cell and abs(int(rr) - ar) + abs(int(cc) - ac) == 1:
                continue
            return True
    return False


def _wrapped_placement_ok(canvas, sprite, marker_r, marker_c, bg, anchors, own_anchor, target, *, min_wrap_pixels=0):
    cells, wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, marker_r, marker_c, bg)
    if wrapped_pixels < int(min_wrap_pixels):
        return None

    occupied = set()
    for rr, cc, _color in cells:
        key = (int(rr), int(cc))
        if key in occupied:
            return None
        occupied.add(key)
        if int(canvas[rr, cc]) != int(bg):
            return None

    if _touches_foreground_cells(canvas, occupied, bg):
        return None
    if _touches_anchor_illegally(occupied, anchors, own_anchor, target):
        return None
    return int(wrapped_pixels)


def _place_wrapped(canvas, sprite, marker_r, marker_c, bg):
    cells, _wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, marker_r, marker_c, bg)
    for rr, cc, color in cells:
        canvas[rr, cc] = int(color)


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("top", 0, c))
        out.append(("bottom", H - 1, c))
    for r in rows:
        out.append(("left", r, 0))
        out.append(("right", r, W - 1))
    random.shuffle(out)
    return [(side, r, c) for side, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _side_family(side):
    return "tb" if side in ("top", "bottom") else "lr"


def _anchor_is_wrap_friendly(side, r, c, shape):
    H, W = shape
    if side in ("top", "bottom"):
        return int(c) <= _WRAP_EDGE_BAND or int(c) >= int(W - 1 - _WRAP_EDGE_BAND)
    return int(r) <= _WRAP_EDGE_BAND or int(r) >= int(H - 1 - _WRAP_EDGE_BAND)


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _anchor_target(anchor, shape):
    H, W = shape
    if anchor["side"] == "top":
        return 1, int(anchor["c"])
    if anchor["side"] == "bottom":
        return int(H - 2), int(anchor["c"])
    if anchor["side"] == "left":
        return int(anchor["r"]), 1
    if anchor["side"] == "right":
        return int(anchor["r"]), int(W - 2)
    raise ValueError(f"unknown side {anchor['side']}")


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
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


def _choose_anchors(shape, bg):
    H, W = shape
    anchor_canvas = np.full((H, W), int(bg), dtype=int)
    anchor_count = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _COLORS if c != bg], anchor_count)
    wrap_color = int(anchor_colors[0])
    anchors = []
    seen_families = set()
    family_order = ["tb", "lr"]
    random.shuffle(family_order)

    for idx, color in enumerate(anchor_colors):
        candidates = _anchor_candidates(anchor_canvas, bg)
        if not candidates:
            return None

        missing_families = [fam for fam in family_order if fam not in seen_families]
        required_family = missing_families[0] if len(missing_families) >= len(anchor_colors) - idx else None

        if required_family is not None:
            family_candidates = [cand for cand in candidates if _side_family(cand[0]) == required_family]
            if not family_candidates:
                return None
            candidates = family_candidates

        if idx == 0:
            preferred = [cand for cand in candidates if _anchor_is_wrap_friendly(*cand, shape)]
            if preferred:
                candidates = preferred

        side, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "side": side}
        anchors.append(anchor)
        seen_families.add(_side_family(side))
        _put_anchor(anchor_canvas, r, c, color)
    return anchor_canvas, anchors, wrap_color


def _put_anchors(canvas, anchors):
    for anchor in anchors:
        _put_anchor(canvas, anchor["r"], anchor["c"], anchor["color"])


def _build_solved_sprite(sprite_canvas, anchors, anchor, bg, used_body, *, require_wrap):
    target = _anchor_target(anchor, sprite_canvas.shape)
    area_lo = max(_WRAP_BODY_AREA_MIN, _BODY_AREA_RANGE[0]) if require_wrap else _BODY_AREA_RANGE[0]
    for _ in range(_SPRITE_BUILD_TRIES):
        area = random.randint(area_lo, _BODY_AREA_RANGE[1])
        body = _body_color(bg, anchor["color"], used_body)
        sprite = _make_sprite(area, body, anchor["color"], bg)
        wrapped_pixels = _wrapped_placement_ok(
            sprite_canvas,
            sprite,
            target[0],
            target[1],
            bg,
            anchors,
            anchor,
            target,
            min_wrap_pixels=_MIN_WRAP_PIXELS if require_wrap else 0,
        )
        if wrapped_pixels is None:
            continue
        _place_wrapped(sprite_canvas, sprite, target[0], target[1], bg)
        used_body.add(body)
        return {"sprite": sprite, "anchor": anchor, "target": target, "wrapped_pixels": int(wrapped_pixels)}
    return None


def _place_solved_sprites(out, anchors, bg, wrap_color):
    sprite_canvas = np.full_like(out, int(bg))
    used_body = set()
    sprites = []
    for anchor in anchors:
        item = _build_solved_sprite(
            sprite_canvas,
            anchors,
            anchor,
            bg,
            used_body,
            require_wrap=int(anchor["color"]) == int(wrap_color),
        )
        if item is None:
            return None
        sprites.append(item)
    if max((item["wrapped_pixels"] for item in sprites), default=0) < _MIN_WRAP_PIXELS:
        return None
    out[:, :] = sprite_canvas
    _put_anchors(out, anchors)
    return sprites


def _scatter_positions(canvas, sprite, target, bg):
    H, W = canvas.shape
    tr, tc = target
    candidates = []
    strong = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if r == int(tr) or c == int(tc):
                continue
            if not _can_place(canvas, sprite, r, c, bg):
                continue
            candidates.append((int(r), int(c)))
            if abs(int(r) - int(tr)) >= _STRONG_SHIFT_MIN and abs(int(c) - int(tc)) >= _STRONG_SHIFT_MIN:
                strong.append((int(r), int(c)))
    return strong if strong else candidates


def _scatter_sprites(inp, sprites, bg):
    ordered = sorted(sprites, key=lambda item: item["sprite"]["area"], reverse=True)
    for item in ordered:
        positions = _scatter_positions(inp, item["sprite"], item["target"], bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, item["sprite"], in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg, expected_moves):
    if not _bg_is_mode(inp, bg):
        return False
    metrics = variant_metrics(inp)
    if int(metrics["move_count"]) != int(expected_moves):
        return False
    if int(metrics["diagonal_components"]) != int(expected_moves):
        return False
    if int(metrics["axis_changed_components"]) != int(expected_moves):
        return False
    if int(metrics["wrapped_pixels"]) < _MIN_WRAP_PIXELS:
        return False
    if int(metrics["wrapped_components"]) < 1:
        return False
    if not bool(metrics["changes_vs_clipped"]):
        return False
    if not bool(metrics["changes_vs_axis_only"]):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()

    chosen = _choose_anchors(out.shape, bg)
    if chosen is None:
        return None
    anchor_canvas, anchors, wrap_color = chosen

    sprites = _place_solved_sprites(out, anchors, bg, wrap_color)
    if sprites is None:
        return None

    inp[:, :] = anchor_canvas
    if not _scatter_sprites(inp, sprites, bg):
        return None

    if not _generated_grid_is_valid(inp, out, bg, len(sprites)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
