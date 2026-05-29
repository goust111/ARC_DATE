# Variant of 581f7754
# Dimension: D16 — 非背景像素碰撞处理 (transform)
# Change: 滑动目标位置的非背景像素不被覆盖（精灵像素跳过这些位置）... → 禁止重叠（精灵互不干涉）
#
# Design analysis:
# spec:
#   preserved_core:
#     - Border anchors are still isolated single pixels whose colors identify which sprite should react.
#     - Interior multi-color sprites still have one least-frequent marker pixel and slide on one axis only.
#     - Top/bottom anchors still mean horizontal alignment; left/right anchors still mean vertical alignment.
#   d16_change:
#     - Old behavior cleared all candidate sprites, then pasted translated pixels one by one.
#       Any translated pixel landing on a surviving non-bg cell was simply skipped, so the sprite could move
#       partially and arrive with a missing notch.
#     - New behavior keeps the same anchor detection and translation computation, but collision handling is
#       promoted from pixel-level to object-level: after removing all candidate sprites from the board,
#       a sprite moves only if every in-bounds translated cell lands on bg and its translated footprint
#       does not overlap any other translated sprite footprint. Otherwise the whole sprite stays at its
#       original input position.
#   concrete_trace:
#     - Example: a 5-cell sprite with marker color 4 wants to slide right until its marker column matches
#       a bottom-edge anchor. A fixed interior singleton sits on one body cell of the target footprint.
#     - Old logic: 4 of the 5 translated cells would paste; the colliding cell is skipped, producing an
#       aligned-but-damaged sprite in the output.
#     - New logic: because one translated cell would collide, none of the sprite moves; the whole component
#       remains in the scattered input location.
#   complexity_calibration:
#     - The task still needs repeated anchor colors and 3-5 sprites so the learner must infer color matching
#       and axis choice, not just one-off object motion.
#     - At least one blocked sprite is forced in each generated example, and at least one sprite still moves,
#       so the D16 branch is visible without collapsing the original rule.
#     - Fixed blockers are kept tiny (isolated singleton obstacles) so they make the collision rule obvious
#       instead of visually overwhelming the scene.
#   implicit_assumptions_review:
#     - Original Step 5 assumed partial translation was acceptable; that is exactly what now becomes invalid.
#     - Original clear-first pass is still safe and is intentionally preserved, because D16 changes collision
#       resolution, not anchor matching or the fact that sprites vacate their start cells before landing.
#     - Because "whole sprite stays" reintroduces source cells after the clear-first pass, generator must
#       guarantee that blocked sprites' original cells do not overlap any successful target footprints.
#     - Pairwise target overlaps are now treated symmetrically as blocked for both sprites; generator keeps
#       successful targets disjoint so no hidden ordering rule is required for normal cases.
#   placement_legality:
#     - Successful sprites are still generated output-first at aligned positions and scattered onto the input.
#     - Blocked sprites are generated from a valid aligned target candidate, then a fixed singleton obstacle
#       is planted on one body cell of that target; the blocked sprite itself is written to the same scattered
#       position in both input and output.
#     - Obstacles use colors outside the anchor-color set, so solver never misclassifies them as movable
#       single-pixel sprites.
#   observability_and_cleanliness:
#     - Mobile distractor singletons from the original generator are omitted here. They are not part of the
#       D16 change and they reduce readability by adding extra moving one-pixel components unrelated to the
#       new stop-vs-partial-move distinction.
#     - Validation rejects degenerate samples where old and new collision semantics happen to produce the
#       same output.
#   post_change_solver_assumptions:
#     - Every movable component still has either a least-frequent marker color matching an anchor or is a
#       singleton whose own color matches an anchor and is not itself that anchor.
#     - Fixed blockers never use anchor colors, so they remain stationary under `main()`.
#     - Generated successful destinations are pairwise disjoint; blocked cases are created via explicit
#       static obstacles, not via hidden tie-breaking on iteration order.

from __future__ import annotations

from collections import Counter
import random
from pathlib import Path
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import find_components, get_bg, neighbors4, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, but cancel moves that would collide.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are mostly two-color sprites with one minority-color marker pixel.
    - Isolated border pixels provide anchor colors, and some examples also include isolated interior blockers.

    Rule:
    - Match each sprite to the border anchor with the same marker color.
    - Top/bottom anchors make the sprite slide horizontally; left/right anchors make it slide vertically.
    - After removing all movable sprites from the board, a sprite moves only if its translated in-bounds
      footprint is entirely empty and does not overlap another translated sprite footprint; otherwise it stays put.

    Key insight: The anchor still decides the slide axis, but D16 changes collision handling from "skip
    blocked pixels" to "all-or-nothing movement" for the whole sprite.
    """
    out, _details = _solve_with_mode(grid, collision_mode="stop", return_details=True)
    return out


def _solve_with_mode(
    grid: np.ndarray,
    *,
    collision_mode: str,
    return_details: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    collision_mode = str(collision_mode)

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
            if v == int(bg) or int(v) not in anchors_by_color or (int(r), int(c)) in anchor_cells:
                continue
            marker_col = int(v)

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

    if collision_mode == "skip":
        out = base.copy()
        for move in moves:
            for rr, cc, v in move["dest_cells"]:
                if int(out[int(rr), int(cc)]) == int(bg):
                    out[int(rr), int(cc)] = int(v)
        if not return_details:
            return out
        return out, {
            "candidate_count": int(len(moves)),
            "moved_count": int(len(moves)),
            "blocked_count": 0,
            "static_collision_count": 0,
            "target_overlap_count": 0,
        }

    if collision_mode != "stop":
        raise ValueError(f"unsupported collision_mode: {collision_mode}")

    blocked = [False] * len(moves)
    static_collision_count = 0
    for i, move in enumerate(moves):
        if any(int(base[int(rr), int(cc)]) != int(bg) for rr, cc, _v in move["dest_cells"]):
            blocked[i] = True
            static_collision_count += 1

    target_overlap_count = 0
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            if moves[i]["dest_set"] & moves[j]["dest_set"]:
                if not blocked[i]:
                    blocked[i] = True
                if not blocked[j]:
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
    return _solve_with_mode(grid, collision_mode="skip", return_details=False)


# --------------- generator (collision-stop hybrid: output-first success + explicit blockers) ---------------

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
_BLOCKED_SPRITE_ATTEMPTS = 24

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


def _sprite_cells(sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    cells = []
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            v = int(patch[pr, pc])
            if v == int(bg):
                continue
            cells.append((int(top + pr), int(left + pc), int(v), int(pr), int(pc)))
    return cells


def _body_world_cells(sprite, marker_r, marker_c, bg):
    off_r, off_c = sprite["offset"]
    cells = []
    for rr, cc, _v, pr, pc in _sprite_cells(sprite, marker_r, marker_c, bg):
        if int(pr) == int(off_r) and int(pc) == int(off_c):
            continue
        cells.append((int(rr), int(cc)))
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


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _obstacle_color(bg, marker, body, anchor_colors):
    pool = [
        c
        for c in _COLORS
        if c not in {int(bg), int(marker), int(body)} and c not in set(map(int, anchor_colors))
    ]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker), int(body)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    blocked_count = min(random.randint(*_BLOCKED_SPRITE_COUNT_RANGE), total_sprites - 1)
    success_count = total_sprites - blocked_count

    marker_colors = _marker_color_sequence(anchor_colors, total_sprites)
    success_markers = marker_colors[:success_count]
    blocked_markers = marker_colors[success_count:]

    used_body = set()
    if not _place_successful_sprites(out, inp, anchors, success_markers, bg, used_body):
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


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
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


def _marker_color_sequence(anchor_colors, total_sprites):
    marker_colors = anchor_colors[:]
    while len(marker_colors) < total_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors[:total_sprites]


def _place_successful_sprites(out, inp, anchors, marker_colors, bg, used_body):
    by_color = {a["color"]: a for a in anchors}
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
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
    obstacle_sprite_cache: dict[int, dict] = {}

    for marker in marker_colors:
        anchor = by_color[int(marker)]
        placed = False
        for _ in range(_BLOCKED_SPRITE_ATTEMPTS):
            body = _body_color(bg, marker, used_body)
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

                for input_pos in input_positions:
                    obstacle_color = _obstacle_color(bg, marker, body, anchor_colors)
                    if int(obstacle_color) not in obstacle_sprite_cache:
                        obstacle_sprite_cache[int(obstacle_color)] = {
                            "patch": np.array([[int(obstacle_color)]], dtype=int),
                            "offset": (0, 0),
                        }
                    obstacle_sprite = obstacle_sprite_cache[int(obstacle_color)]

                    for obst_r, obst_c in body_cells:
                        temp_out = out.copy()
                        temp_inp = inp.copy()
                        if not _can_place(temp_out, obstacle_sprite, obst_r, obst_c, bg):
                            continue
                        if not _can_place(temp_inp, obstacle_sprite, obst_r, obst_c, bg):
                            continue
                        _place(temp_out, obstacle_sprite, obst_r, obst_c, bg)
                        _place(temp_inp, obstacle_sprite, obst_r, obst_c, bg)
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
    solved, details = _solve_with_mode(inp, collision_mode="stop", return_details=True)
    if not bool(np.array_equal(solved, out)):
        return False
    if int(details["blocked_count"]) < 1 or int(details["moved_count"]) < 1:
        return False
    legacy = _legacy_main(inp)
    return not bool(np.array_equal(legacy, out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D16 input")
