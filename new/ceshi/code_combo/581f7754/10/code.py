# Combo variant of 581f7754 (index 10)
# combo_spec: D11/0+D16/2
# Name: 静态诱饵+禁止重叠

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import find_components, get_bg, neighbors4, paste


# -----------------------------------------------------------------------------
# Design analysis for combo D11/0 + D16/2
#
# concrete_behavior:
#   - Base task: an interior multicolor sprite has one rare marker pixel; an
#     isolated border anchor of the same color tells that sprite which axis to
#     slide on and which row/column to match.
#   - D11/0 change: isolated interior same-color singletons are no longer tiny
#     movable sprites. They are static decoys and remain in place.
#   - D16/2 change: movement is no longer pixelwise "paste what fits". A sprite
#     either lands with its whole in-bounds translated footprint or does not
#     move at all.
#   - Fused behavior: the static singleton decoys are also the blockers. Example:
#     color 6 has a left-edge anchor at row 8, so every marker-6 sprite wants to
#     slide vertically until its marker reaches row 8. One marker-6 singleton
#     decoy sits on a body cell of a candidate landing footprint. Original logic
#     would move that singleton and let the larger sprite keep translating.
#     Fused logic keeps the singleton fixed, detects the overlap, and leaves the
#     whole larger sprite at its input location.
#
# complexity_calibration:
#   - Each generated board forces both outcomes: at least one sprite moves and at
#     least one sprite is blocked by a static decoy. This makes the collision-stop
#     branch visible instead of degenerating back to the base task.
#   - A blocked color is also used by a successful sprite, so the learner sees a
#     same-color singleton stay fixed while a larger same-color-marked sprite can
#     still move when unblocked. That makes "connectivity/size matters" observable.
#   - Body colors avoid anchor colors whenever possible so color matching stays
#     anchored on the minority marker pixel rather than on unrelated body regions.
#
# hidden_assumption_review:
#   - The original singleton branch in main() assumed some size-1 components of
#     anchor colors should move. That premise is false here; singleton components
#     must be ignored unless they are border anchors.
#   - The original clear-first pass is still valid, but only for movable
#     multicolor sprites. Static decoys must remain on the base board so they can
#     block translated destinations.
#   - The original paste loop assumed partial landing was acceptable. That is now
#     invalid: any overlap with surviving non-background pixels, including static
#     decoys, cancels the whole sprite move.
#   - Pairwise target overlaps among translated sprites are also treated as
#     blocked for every involved sprite. The generator keeps successful targets
#     disjoint, but the solver handles the general case defensively.
#
# placement_legality:
#   - The generator is still output-first. Successful sprites are placed solved in
#     `out`, then scattered onto `inp`.
#   - A blocked sprite is generated from a valid aligned target. One isolated
#     singleton decoy of the same marker/anchor color is planted on a body cell of
#     that target in both `out` and `inp`, and the blocked sprite itself is placed
#     at its scattered input location in both `out` and `inp`.
#   - All placements use the same 8-neighborhood isolation checks as the original
#     task, so anchors, sprites, and decoys stay visually separated and legal.
#
# observability_and_cleanliness:
#   - The rule is visible from 2-4 examples because every board contains:
#       1) a same-color border anchor,
#       2) a larger multicolor sprite reacting to that color,
#       3) an isolated same-color singleton that does not move,
#       4) at least one blocked-vs-success contrast.
#   - No hidden ordering on raw color ids is used. The only distinctions are
#     human-visible: border vs interior, singleton vs connected sprite, and
#     collision-free destination vs blocked destination.
# -----------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the composite interior sprites toward matching border anchors, while static singleton decoys stay fixed and can block a move entirely.

    Input structure:
    - The background is the most common color.
    - Interior foreground contains multicolor 8-connected sprites whose least frequent color is a marker, plus some isolated interior singleton decoys that share anchor colors but are not movable.
    - Border anchors are isolated non-background border pixels; their edge position still determines whether matching sprites slide horizontally or vertically.

    Rule:
    - Match each multicolor sprite to the border anchor with the same marker color, then compute the one-axis translation that aligns the marker with that anchor row or column.
    - Top/bottom anchors cause horizontal sliding; left/right anchors cause vertical sliding.
    - Ignore isolated interior singleton decoys as moving objects, and if a translated sprite would overlap any remaining non-background pixel or another translated sprite footprint, cancel that whole sprite move; otherwise paste the translated in-bounds cells with clipping.

    Key insight:
    - Color alone is no longer enough: the movable object is the full multicolor connected sprite, whereas an isolated same-color singleton is a stationary lure, and those stationary lures can prevent a same-color sprite from moving at all.
    """
    out, _details = _solve_variant(grid, return_details=True)
    return out


def _solve_variant(grid: np.ndarray, *, return_details: bool = False) -> tuple[np.ndarray, dict] | np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

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
            if ok and int(v) not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    moves: list[dict] = []
    static_decoy_cells: set[tuple[int, int]] = set()
    static_decoy_count = 0

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) == 1:
                r, c = map(int, coords[0])
                v = int(g[r, c])
                if int(v) in anchors_by_color and (int(r), int(c)) not in anchor_cells:
                    static_decoy_cells.add((int(r), int(c)))
                    static_decoy_count += 1
            continue

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        dest_cells: list[tuple[int, int, int]] = []
        dest_set: set[tuple[int, int]] = set()
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                dest_cells.append((int(rr), int(cc), int(g[int(r), int(c)])))
                dest_set.add((int(rr), int(cc)))

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "dest_cells": dest_cells,
                "dest_set": dest_set,
                "marker_color": int(marker_col),
            }
        )

    base = g.copy()
    for move in moves:
        coords = move["coords"]
        base[coords[:, 0], coords[:, 1]] = int(bg)

    blocked = [False] * len(moves)
    blocked_by_static = [False] * len(moves)
    static_collision_count = 0
    for i, move in enumerate(moves):
        hits_static = bool(move["dest_set"] & static_decoy_cells)
        hits_other = any(int(base[int(rr), int(cc)]) != int(bg) for rr, cc, _v in move["dest_cells"])
        if hits_static or hits_other:
            blocked[i] = True
            if hits_static:
                blocked_by_static[i] = True
            static_collision_count += 1

    target_overlap_count = 0
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            if moves[i]["dest_set"] & moves[j]["dest_set"]:
                blocked[i] = True
                blocked[j] = True
                target_overlap_count += 1

    out = base.copy()
    for is_blocked, move in zip(blocked, moves):
        coords = move["coords"]
        if is_blocked:
            out[coords[:, 0], coords[:, 1]] = g[coords[:, 0], coords[:, 1]]
            continue
        for rr, cc, v in move["dest_cells"]:
            out[int(rr), int(cc)] = int(v)

    details = {
        "candidate_count": int(len(moves)),
        "moved_count": int(sum(1 for x in blocked if not x)),
        "blocked_count": int(sum(1 for x in blocked if x)),
        "static_decoy_count": int(static_decoy_count),
        "blocked_by_static_count": int(sum(1 for x in blocked_by_static if x)),
        "static_collision_count": int(static_collision_count),
        "target_overlap_count": int(target_overlap_count),
    }
    if not return_details:
        return out
    return out, details


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
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in coord_set and int(grid[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

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
            if ok and int(v) not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if int(v) == int(bg) or int(v) not in anchors_by_color or (int(r), int(c)) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue
        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc)))

    out = g.copy()
    for coords, _dr, _dc in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[int(rr), int(cc)]) == int(bg):
                out[int(rr), int(cc)] = int(g[int(r), int(c)])
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
_BLOCKED_SPRITE_COUNT_RANGE = (1, 2)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500
_BLOCKED_SPRITE_ATTEMPTS = 30

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
        if not frontier:
            return _grow_body(area)
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
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "body_color": int(body_color),
        "marker_color": int(marker_color),
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _body_world_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    off_r, off_c = sprite["offset"]
    cells = []
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            v = int(patch[pr, pc])
            if v == int(bg):
                continue
            if int(pr) == int(off_r) and int(pc) == int(off_c):
                continue
            cells.append((int(top + pr), int(left + pc)))
    return cells


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
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


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


def _aligned_positions_both(out, inp, sprite, anchor, bg):
    H, W = out.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(out, sprite, r, anchor["c"], bg) and _can_place(inp, sprite, r, anchor["c"], bg)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(out, sprite, anchor["r"], c, bg) and _can_place(inp, sprite, anchor["r"], c, bg)
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


def _scatter_positions_both(out, inp, sprite, anchor, marker, bg):
    H, W = out.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(out, sprite, mr, c, bg) and _can_place(inp, sprite, mr, c, bg)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(out, sprite, r, mc, bg) and _can_place(inp, sprite, r, mc, bg)
    ]


def _body_color(bg, marker, used, banned_colors):
    banned = {int(bg), int(marker), *map(int, banned_colors)}
    pool = [c for c in _COLORS if c not in banned and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in banned]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    blocked_count = min(random.randint(*_BLOCKED_SPRITE_COUNT_RANGE), total_sprites - 1)
    success_count = total_sprites - blocked_count
    anchor_count = random.randint(_ANCHOR_COUNT_RANGE[0], min(_ANCHOR_COUNT_RANGE[1], success_count))

    placed = _place_anchors(out, inp, bg, anchor_count)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    success_markers = _success_marker_colors(anchor_colors, success_count)
    blocked_markers = [random.choice(success_markers) for _ in range(blocked_count)]

    used_body: set[int] = set()
    if not _place_successful_sprites(out, inp, anchors, success_markers, anchor_colors, bg, used_body):
        return None
    if not _place_blocked_sprites(out, inp, anchors, blocked_markers, anchor_colors, bg, used_body):
        return None

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


def _place_anchors(out, inp, bg, anchor_count):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], int(anchor_count))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _success_marker_colors(anchor_colors, success_count):
    marker_colors = list(anchor_colors)
    while len(marker_colors) < int(success_count):
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors[:success_count]


def _place_successful_sprites(out, inp, anchors, marker_colors, anchor_colors, bg, used_body):
    by_color = {a["color"]: a for a in anchors}
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body, anchor_colors)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return False
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
        used_body.add(int(body))

    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _place_blocked_sprites(out, inp, anchors, marker_colors, anchor_colors, bg, used_body):
    by_color = {a["color"]: a for a in anchors}
    decoy_cache: dict[int, dict] = {}

    for marker in marker_colors:
        anchor = by_color[int(marker)]
        placed = False
        for _ in range(_BLOCKED_SPRITE_ATTEMPTS):
            body = _body_color(bg, marker, used_body, anchor_colors)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            target_positions = _aligned_positions_both(out, inp, sprite, anchor, bg)
            if not target_positions:
                continue
            random.shuffle(target_positions)

            for target_pos in target_positions:
                input_positions = _scatter_positions_both(out, inp, sprite, anchor, target_pos, bg)
                if not input_positions:
                    continue
                random.shuffle(input_positions)
                body_cells = _body_world_cells(sprite, target_pos[0], target_pos[1], bg)
                if not body_cells:
                    continue
                random.shuffle(body_cells)

                if int(marker) not in decoy_cache:
                    decoy_cache[int(marker)] = {
                        "patch": np.array([[int(marker)]], dtype=int),
                        "offset": (0, 0),
                    }
                decoy_sprite = decoy_cache[int(marker)]

                for input_pos in input_positions:
                    for decoy_r, decoy_c in body_cells:
                        temp_out = out.copy()
                        temp_inp = inp.copy()
                        if not _can_place(temp_out, decoy_sprite, decoy_r, decoy_c, bg):
                            continue
                        if not _can_place(temp_inp, decoy_sprite, decoy_r, decoy_c, bg):
                            continue
                        _place(temp_out, decoy_sprite, decoy_r, decoy_c, bg)
                        _place(temp_inp, decoy_sprite, decoy_r, decoy_c, bg)
                        if not _can_place(temp_out, sprite, input_pos[0], input_pos[1], bg):
                            continue
                        if not _can_place(temp_inp, sprite, input_pos[0], input_pos[1], bg):
                            continue
                        _place(temp_out, sprite, input_pos[0], input_pos[1], bg)
                        _place(temp_inp, sprite, input_pos[0], input_pos[1], bg)
                        out[:, :] = temp_out
                        inp[:, :] = temp_inp
                        used_body.add(int(body))
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False

    solved, details = _solve_variant(inp, return_details=True)
    if not bool(np.array_equal(solved, out)):
        return False
    if int(details["moved_count"]) < 1:
        return False
    if int(details["blocked_by_static_count"]) < 1:
        return False
    if int(details["static_decoy_count"]) < 1:
        return False

    legacy = _legacy_main(inp)
    if bool(np.array_equal(legacy, out)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
