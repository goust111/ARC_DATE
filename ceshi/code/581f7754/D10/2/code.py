# Variant of 581f7754
# Dimension: D10 — 锚点-精灵基数关系 (meta)
# Change: 多对多：1-3个锚点颜色，3-5个精灵，多个精灵可共享同一标记色（因此共享同一锚点）... → 多锚点竞争：精灵选择最近锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis
# a) Concrete behavior and trace
#    - The visible change is that a marker color may now appear on several isolated border anchors.
#    - Interior sprites still look like the original task: one body color plus a minority marker pixel,
#      with 3-5 disconnected sprites placed away from the border.
#    - The solver can no longer map "marker color -> one anchor". It must inspect every anchor of that
#      color, measure the marker pixel's Manhattan distance to each anchor, choose the nearest one,
#      and only then apply the old slide rule determined by that chosen anchor's edge.
#    - Example trace:
#        old task: color 4 had only one anchor, so marker color 4 always slid to that anchor's row/col.
#        variant: color 4 may have anchors at top col 2 and left row 7. A sprite with marker at (8, 5)
#        is distance 11 from the top anchor (0,2) and distance 6 from the left anchor (7,0), so it now
#        chooses the left anchor and slides vertically to row 7 instead of horizontally to col 2.
#    - This changes both solver behavior and generator construction. Output-first generation must decide
#      which same-color anchor each sprite is meant to choose, then scatter the solved sprite to an input
#      position where that anchor is indeed the nearest one.
# b) Complexity calibration
#    - To make the new rule noticeable, every generated grid has one "contested" marker color with two
#      anchors of that same color, one on a top/bottom edge and one on a left/right edge.
#    - At least two sprites use that contested color, and at least one sprite goes to each of the two
#      competing anchors. This makes the duplicated-color competition visible in only 2-3 examples.
#    - Other colors keep a single anchor so the task does not explode into too many simultaneous cases.
# c) Implicit-assumption audit against the original main()
#    - The original `anchors_by_color: dict[int, tuple[int, int]]` assumed one anchor per color.
#      That is invalid now; it must become `dict[int, list[anchor]]`.
#    - The original `if ok and v not in anchors_by_color:` silently discarded later same-color anchors.
#      That would erase the new signal entirely, so the variant must keep every isolated border anchor.
#    - The original `ar, ac = anchors_by_color[marker_col]` assumed color lookup already identifies a
#      unique target. That line now needs a nearest-anchor selection step based on the marker position.
#    - The original singleton fallback remains valid, but it must also choose among multiple same-color
#      anchors by distance.
#    - Original collision handling, clearing-before-pasting, and clipping remain correct because the
#      variant changes target selection only, not the translation mechanics.
# d) Placement legality
#    - Anchors stay isolated single border pixels with a 1-cell empty moat so they are unambiguously
#      detectable by the solver.
#    - Output-first generation places solved sprites exactly as before, except the chosen anchor may be
#      one of two same-color anchors.
#    - Scattering is constrained so the chosen anchor is the unique nearest same-color anchor from the
#      sprite's input marker position; if no such placement exists, the attempt is discarded.
#    - Sprites still cannot overlap or touch each other or anchors.
# e) Visual observability
#    - The key cue is repeated anchor color on different edges, plus same-color sprites ending at
#      different destinations in the output. That is visible without using hidden properties such as
#      numeric color ordering.
#    - Manhattan distance is inferable because the winning anchor is spatially closer to each sprite's
#      marker, and the chosen anchor's edge then explains whether the movement is horizontal or vertical.
# f) Visual cleanliness
#    - Only one color is forced into competition, keeping the scene readable.
#    - Optional singleton distractors from the original generator are omitted here so the new anchor
#      competition is the dominant visible structure instead of extra isolated pixels.
# g) Color-rule inferability
#    - Colors themselves do not transform. They only label which anchors a marker may consider.
#    - Because one duplicated color repeats across multiple sprites and examples, the human solver gets
#      repeated evidence that "same color" defines the candidate set, while "nearest anchor" chooses the
#      actual destination among those candidates.


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _anchor_axis(r: int, c: int, H: int, W: int) -> str:
    if int(r) in (0, int(H - 1)) and int(c) not in (0, int(W - 1)):
        return "h"
    return "v"


def _detect_anchors(g: np.ndarray, bg: int) -> tuple[dict[int, list[dict[str, int | str]]], set[tuple[int, int]]]:
    H, W = g.shape
    anchors_by_color: dict[int, list[dict[str, int | str]]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), H, W):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if not ok:
                continue
            anchors_by_color.setdefault(v, []).append(
                {
                    "color": int(v),
                    "r": int(r),
                    "c": int(c),
                    "axis": _anchor_axis(int(r), int(c), H, W),
                }
            )
    for anchors in anchors_by_color.values():
        anchors.sort(key=lambda a: (int(a["r"]), int(a["c"])))
    anchor_cells = {
        (int(anchor["r"]), int(anchor["c"]))
        for anchors in anchors_by_color.values()
        for anchor in anchors
    }
    return anchors_by_color, anchor_cells


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
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


def _manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(int(p[0]) - int(q[0])) + abs(int(p[1]) - int(q[1]))


def _nearest_anchor(marker: tuple[int, int], anchors: list[dict[str, int | str]]) -> dict[str, int | str]:
    return min(
        anchors,
        key=lambda a: (
            _manhattan(marker, (int(a["r"]), int(a["c"]))),
            int(a["r"]),
            int(a["c"]),
        ),
    )


def _nearest_anchor_data(
    marker: tuple[int, int], anchors: list[dict[str, int | str]]
) -> tuple[dict[str, int | str], int, bool]:
    ranked = sorted(
        (
            (
                _manhattan(marker, (int(anchor["r"]), int(anchor["c"]))),
                int(anchor["r"]),
                int(anchor["c"]),
                anchor,
            )
            for anchor in anchors
        ),
        key=lambda item: (int(item[0]), int(item[1]), int(item[2])),
    )
    chosen = ranked[0][3]
    tied = len(ranked) > 1 and int(ranked[1][0]) == int(ranked[0][0])
    margin = 999 if len(ranked) == 1 else int(ranked[1][0]) - int(ranked[0][0])
    return chosen, margin, tied


def _collect_moves(
    g: np.ndarray,
    bg: int,
    anchors_by_color: dict[int, list[dict[str, int | str]]],
    anchor_cells: set[tuple[int, int]],
    *,
    choose_first_anchor: bool,
) -> list[dict[str, object]]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[dict[str, object]] = []

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
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        anchors = anchors_by_color[int(marker_col)]
        anchor = anchors[0] if choose_first_anchor else _nearest_anchor(mp, anchors)

        dr = 0
        dc = 0
        if str(anchor["axis"]) == "h":
            dc = int(anchor["c"]) - int(mp[1])
        else:
            dr = int(anchor["r"]) - int(mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_color": int(marker_col),
                "marker": (int(mp[0]), int(mp[1])),
                "anchor": anchor,
            }
        )

    return moves


def _apply_moves(g: np.ndarray, bg: int, moves: list[dict[str, object]]) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    for move in moves:
        coords = move["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for move in moves:
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each interior sprite toward the closest border anchor that shares its marker color.

    Input structure:
    - The modal color is the background.
    - Interior foreground components are multi-color sprites whose rarest color acts as a marker.
    - Border anchors are isolated single pixels, and one marker color may appear on several anchors.

    Rule:
    - For each sprite, compare its marker pixel to every same-color anchor and choose the nearest one.
    - If that chosen anchor sits on the top/bottom edge, slide the sprite horizontally so the marker
      lands in the anchor's column; if it sits on the left/right edge, slide vertically so the marker
      lands in the anchor's row.
    - Clear the source, clip off-grid pixels, and never overwrite an already occupied destination cell.

    Key insight: matching the marker color only gives a candidate set; the actual target is the nearest
    same-color anchor, and that selected anchor then determines the slide axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    assert anchors_by_color
    moves = _collect_moves(g, bg, anchors_by_color, anchor_cells, choose_first_anchor=False)
    return _apply_moves(g, bg, moves)


def _solve_first_anchor_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    if not anchors_by_color:
        raise AssertionError("no anchors")
    moves = _collect_moves(g, bg, anchors_by_color, anchor_cells, choose_first_anchor=True)
    return _apply_moves(g, bg, moves)


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_COMPETING_ANCHOR_MIN_DISTANCE = 5

_ANCHOR_COUNT_RANGE = (1, 3)  # distinct colors
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
            (r, int(anchor["c"]))
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, int(anchor["c"]), bg)
        ]
    return [
        (int(anchor["r"]), c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, int(anchor["r"]), c, bg)
    ]


def _scatter_positions(canvas, sprite, anchor, marker, same_color_anchors, bg):
    H, W = canvas.shape
    mr, mc = marker
    scored = []
    if anchor["axis"] == "h":
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc or not _can_place(canvas, sprite, mr, c, bg):
                continue
            pos = (mr, c)
            chosen, margin, tied = _nearest_anchor_data(pos, same_color_anchors)
            if tied or chosen is not anchor:
                continue
            scored.append((int(margin), abs(int(c) - int(mc)), pos))
    else:
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if r == mr or not _can_place(canvas, sprite, r, mc, bg):
                continue
            pos = (r, mc)
            chosen, margin, tied = _nearest_anchor_data(pos, same_color_anchors)
            if tied or chosen is not anchor:
                continue
            scored.append((int(margin), abs(int(r) - int(mr)), pos))
    if not scored:
        return []
    best_margin = max(int(item[0]) for item in scored)
    best_shift = max(int(item[1]) for item in scored if int(item[0]) == best_margin)
    good = [
        pos
        for margin, shift, pos in scored
        if int(margin) == best_margin and int(shift) == best_shift
    ]
    random.shuffle(good)
    return good


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


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample(
        [c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE)
    )
    contest_color = random.choice(anchor_colors)
    anchors = []

    h_candidates = [cand for cand in _anchor_candidates(out, bg) if cand[0] == "h"]
    if not h_candidates:
        return None
    axis, r, c = random.choice(h_candidates)
    first = {"color": int(contest_color), "r": int(r), "c": int(c), "axis": axis}
    anchors.append(first)
    _put_anchor(out, r, c, contest_color)
    _put_anchor(inp, r, c, contest_color)

    v_candidates = [
        cand
        for cand in _anchor_candidates(out, bg)
        if cand[0] == "v" and _manhattan((cand[1], cand[2]), (first["r"], first["c"])) >= _COMPETING_ANCHOR_MIN_DISTANCE
    ]
    if not v_candidates:
        return None
    axis, r, c = random.choice(v_candidates)
    second = {"color": int(contest_color), "r": int(r), "c": int(c), "axis": axis}
    anchors.append(second)
    _put_anchor(out, r, c, contest_color)
    _put_anchor(inp, r, c, contest_color)

    for color in anchor_colors:
        if int(color) == int(contest_color):
            continue
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)

    return anchor_colors, contest_color, anchors


def _sprite_targets(anchor_colors, contest_color, anchors):
    by_color = {}
    for anchor in anchors:
        by_color.setdefault(int(anchor["color"]), []).append(anchor)
    contest_anchors = by_color[int(contest_color)]
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    targets = [contest_anchors[0], contest_anchors[1]]
    weighted_colors = list(anchor_colors) + [contest_color]
    while len(targets) < n_sprites:
        color = random.choice(weighted_colors)
        targets.append(random.choice(by_color[int(color)]))
    random.shuffle(targets)
    return targets


def _place_aligned_sprites(out, targets, bg):
    used_body = set()
    sprites = []
    for anchor in targets:
        marker = int(anchor["color"])
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        random.shuffle(positions)
        marker_pos = positions[0]
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "marker_color": marker,
                "out_marker": (int(marker_pos[0]), int(marker_pos[1])),
                "area": int((sprite["patch"] != int(bg)).sum()),
            }
        )
    return sprites


def _scatter_sprites(inp, sprites, anchors_by_color, bg):
    order = sorted(
        range(len(sprites)),
        key=lambda idx: (
            len(anchors_by_color[int(sprites[idx]["marker_color"])]),
            int(sprites[idx]["area"]),
        ),
        reverse=True,
    )
    for idx in order:
        meta = sprites[idx]
        positions = _scatter_positions(
            inp,
            meta["sprite"],
            meta["anchor"],
            meta["out_marker"],
            anchors_by_color[int(meta["marker_color"])],
            bg,
        )
        if not positions:
            return False
        marker_pos = random.choice(positions)
        meta["in_marker"] = (int(marker_pos[0]), int(marker_pos[1]))
        _place(inp, meta["sprite"], marker_pos[0], marker_pos[1], bg)
    return True


def _variant_visible(inp):
    return not bool(np.array_equal(main(inp), _solve_first_anchor_baseline(inp)))


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    return _variant_visible(inp)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, contest_color, anchors = placed
    targets = _sprite_targets(anchor_colors, contest_color, anchors)
    sprites = _place_aligned_sprites(out, targets, bg)
    if sprites is None:
        return None
    anchors_by_color = {}
    for anchor in anchors:
        anchors_by_color.setdefault(int(anchor["color"]), []).append(anchor)
    if not _scatter_sprites(inp, sprites, anchors_by_color, bg):
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D10 variant input")
