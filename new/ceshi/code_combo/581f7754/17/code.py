# Combo variant of 581f7754 (index 17)
# combo_spec: D4/2+D10/2
# Name: 成对锚点选近+多锚点竞争

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the fused paired-anchor + multi-anchor-competition variant
# a) Concrete behavior trace:
#    - Every used marker color now has at least one opposite-edge anchor pair:
#      either top+bottom or left+right.
#    - One highlighted color is upgraded further and receives both opposite-edge pairs,
#      giving it four same-color anchors on all four sides.
#    - Example: color 4 anchors at top (0,2), bottom (11,8), left (6,0), right (3,12).
#      A sprite with marker at (6,7) compares Manhattan distances:
#        top=11, bottom=6, left=7, right=8, so it chooses the bottom anchor and slides
#        horizontally to column 8.
#      Another color-4 sprite at (4,1) gets top=3, bottom=14, left=3, right=12; ties are
#      rejected by the generator, so only placements with a unique nearest anchor are used.
#      A third color-4 sprite can be placed so the left anchor wins, making the same color
#      use both horizontal and vertical motion across one grid.
# b) Complexity calibration:
#    - The canvas/object scale stays near the source task, but the generator forces one
#      contest color to demonstrate both parts of the combo: two sprites land on different
#      members of one opposite-edge pair, and at least one sprite of that color lands on
#      an anchor from the perpendicular pair.
#    - This keeps the rule visible in 2-4 examples without flooding the border with many
#      unrelated same-color anchors.
# c) Implicit assumptions audited in the original solver:
#    - `anchors_by_color[color] = one_position` is invalid; the fused task needs every
#      isolated same-color border anchor to be retained.
#    - `if ok and color not in anchors_by_color` would silently drop later anchors and erase
#      the multi-anchor signal, so anchor detection now accumulates lists.
#    - `ar, ac = anchors_by_color[marker_color]` assumed matching color already determines a
#      unique destination. The fused rule inserts a nearest-anchor choice before translation.
#    - Border anchor cells must all be tracked so they are not misread as movable singleton
#      sprites in the fallback branch.
# d) Placement legality:
#    - Generation remains output-first: place all anchors and solved sprites on the output,
#      then scatter each sprite to an input position that preserves its intended nearest
#      same-color anchor.
#    - Every scatter candidate is filtered to require a unique nearest target anchor, which
#      prevents solver ambiguity and enforces the new hidden assumption that ties should not
#      occur in generated training/test examples.
#    - All anchors remain isolated border singletons; sprites remain interior, separated, and
#      non-overlapping.
# e) Visual observability:
#    - Humans can see repeated anchor colors on opposite edges and, for one color, across both
#      axes. The output shows that same-color sprites can settle into different rows/columns,
#      which points directly to nearest-anchor choice rather than a fixed per-color mapping.
#    - The highlighted contest color provides repeated evidence inside one sample instead of
#      requiring cross-sample comparison to notice the fusion.
# f) Visual cleanliness:
#    - Only one color is allowed to expand into four anchors; other colors stay at two anchors.
#      This keeps the border readable and preserves clear whitespace around sprites.
#    - Optional singleton distractors from the source task are intentionally omitted so the
#      new anchor competition stays dominant.
# g) Rule inferability:
#    - No logic depends on numeric color IDs. Color only marks candidate anchor membership.
#    - The decisive factors are all visible: same-color membership, spatial distance, and the
#      chosen anchor's edge determining whether the shift is horizontal or vertical.

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
_GEN_RETRY_LIMIT = 1200

_PAIR_OFFSET_MIN = 2
_SCATTER_MOVE_MIN = 2
_NEAREST_MARGIN_MIN = 1
_CONTEST_ANCHOR_MIN_DISTANCE = 4

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


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _anchor_axis(r: int, c: int, H: int, W: int) -> str:
    if int(r) in (0, int(H - 1)) and int(c) not in (0, int(W - 1)):
        return "h"
    return "v"


def _anchor_side(r: int, c: int, H: int, W: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(H - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    return "right"


def _anchor_priority(anchor: dict[str, int | str]) -> tuple[int, int, int]:
    side_rank = {"top": 0, "left": 1, "right": 2, "bottom": 3}
    return (
        side_rank[str(anchor["side"])],
        int(anchor["r"]),
        int(anchor["c"]),
    )


def _detect_anchors(
    g: np.ndarray, bg: int
) -> tuple[dict[int, list[dict[str, int | str]]], set[tuple[int, int]]]:
    H, W = g.shape
    anchors_by_color: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), H, W):
                continue
            value = int(g[r, c])
            if value == int(bg):
                continue
            isolated = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    isolated = False
                    break
            if not isolated:
                continue
            anchors_by_color[int(value)].append(
                {
                    "color": int(value),
                    "r": int(r),
                    "c": int(c),
                    "axis": _anchor_axis(int(r), int(c), H, W),
                    "side": _anchor_side(int(r), int(c), H, W),
                }
            )
    for anchors in anchors_by_color.values():
        anchors.sort(key=_anchor_priority)
    anchor_cells = {
        (int(anchor["r"]), int(anchor["c"]))
        for anchors in anchors_by_color.values()
        for anchor in anchors
    }
    return anchors_by_color, anchor_cells


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    values = [int(g[int(r), int(c)]) for r, c in coords]
    counts = Counter(values)
    if len(counts) <= 1:
        return None
    return int(min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    points = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    cells = {(int(r), int(c)) for r, c in coords}
    for r, c in points:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return points[0]


def _manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(int(p[0]) - int(q[0])) + abs(int(p[1]) - int(q[1]))


def _same_anchor(a: dict[str, int | str], b: dict[str, int | str]) -> bool:
    return int(a["r"]) == int(b["r"]) and int(a["c"]) == int(b["c"])


def _nearest_anchor_data(
    marker: tuple[int, int], anchors: list[dict[str, int | str]]
) -> tuple[dict[str, int | str], int, bool]:
    ranked = sorted(
        (
            (
                _manhattan(marker, (int(anchor["r"]), int(anchor["c"]))),
                _anchor_priority(anchor),
                anchor,
            )
            for anchor in anchors
        ),
        key=lambda item: (int(item[0]), item[1]),
    )
    chosen = ranked[0][2]
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

        marker_color = _component_marker_color(g, coords)
        if marker_color is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            value = int(g[r, c])
            if value == int(bg) or value not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_color = int(value)

        if int(marker_color) not in anchors_by_color:
            continue

        marker = _marker_pixel(g, coords, int(marker_color))
        if marker is None:
            continue

        anchors = anchors_by_color[int(marker_color)]
        chosen = anchors[0] if choose_first_anchor else _nearest_anchor_data(marker, anchors)[0]

        dr = 0
        dc = 0
        if str(chosen["axis"]) == "h":
            dc = int(chosen["c"]) - int(marker[1])
        else:
            dr = int(chosen["r"]) - int(marker[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_color": int(marker_color),
                "marker": (int(marker[0]), int(marker[1])),
                "anchor": chosen,
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


def _choice_records(grid: np.ndarray, *, choose_first_anchor: bool) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    return _collect_moves(g, bg, anchors_by_color, anchor_cells, choose_first_anchor=choose_first_anchor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior sprite toward the unique nearest border anchor among all same-color options.

    Input structure:
    - The modal color is the background.
    - Interior connected components are mostly two-color sprites whose rare color is the marker.
    - Each marker color has an opposite-edge border-anchor pair, and one color may have extra same-color anchors on the perpendicular border pair as well.

    Rule:
    - For each sprite, gather every isolated border anchor with the marker's color and choose the nearest one by Manhattan distance.
    - If the chosen anchor sits on the top or bottom edge, slide the sprite horizontally so the marker reaches that anchor's column; if it sits on the left or right edge, slide vertically so the marker reaches that anchor's row.
    - Clear source cells first, clip pixels that move off-grid, and never overwrite an already occupied non-background destination.

    Key insight:
    - Matching color only identifies the candidate anchors; the sprite's current marker position selects the controlling anchor, and that selected border side then determines the motion axis.
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
            pos = (r + dr, c + dc)
            if pos in body:
                continue
            if r0 <= pos[0] <= r1 and c0 <= pos[1] <= c1:
                interior.add(pos)
            else:
                exterior.add(pos)
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (interior | exterior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg):
    body = _make_body(area)
    marker = _marker_cell(body)
    cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(cells)
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


def _edge_anchor_candidates(canvas, bg, side: str):
    H, W = canvas.shape
    if side in {"top", "bottom"}:
        row = 0 if side == "top" else H - 1
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
        cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
        random.shuffle(cols)
        return [(row, c) for c in cols if _anchor_ok(canvas, row, c, bg)]
    col = 0 if side == "left" else W - 1
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    random.shuffle(rows)
    return [(r, col) for r in rows if _anchor_ok(canvas, r, col, bg)]


def _pair_span(axis: str, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
    if axis == "h":
        return abs(int(pos_a[1]) - int(pos_b[1]))
    return abs(int(pos_a[0]) - int(pos_b[0]))


def _make_anchor(r: int, c: int, axis: str, side: str, color: int) -> dict[str, int | str]:
    return {"r": int(r), "c": int(c), "axis": axis, "side": side, "color": int(color)}


def _pair_for_axis(
    canvas,
    bg,
    axis: str,
    color: int,
    *,
    avoid: list[tuple[int, int]] | None = None,
    min_distance: int = 0,
) -> list[dict[str, int | str]] | None:
    avoid = avoid or []
    side_a, side_b = ("top", "bottom") if axis == "h" else ("left", "right")
    cand_a = _edge_anchor_candidates(canvas, bg, side_a)
    cand_b = _edge_anchor_candidates(canvas, bg, side_b)
    good = []
    weak = []
    for pos_a in cand_a:
        for pos_b in cand_b:
            if any(_manhattan(pos_a, pos) < min_distance or _manhattan(pos_b, pos) < min_distance for pos in avoid):
                continue
            span = _pair_span(axis, pos_a, pos_b)
            pair = [
                _make_anchor(pos_a[0], pos_a[1], axis, side_a, color),
                _make_anchor(pos_b[0], pos_b[1], axis, side_b, color),
            ]
            if span >= _PAIR_OFFSET_MIN:
                good.append(pair)
            elif span > 0:
                weak.append(pair)
    pairs = good or weak
    if not pairs:
        return None
    return random.choice(pairs)


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _anchors_by_color(anchors: list[dict[str, int | str]]) -> dict[int, list[dict[str, int | str]]]:
    grouped: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    for anchor in anchors:
        grouped[int(anchor["color"])].append(anchor)
    for color in grouped:
        grouped[color].sort(key=_anchor_priority)
    return grouped


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if str(anchor["axis"]) == "h":
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


def _scatter_positions(canvas, sprite, target_anchor, marker, same_color_anchors, bg):
    H, W = canvas.shape
    mr, mc = map(int, marker)
    strong = []
    weak = []
    if str(target_anchor["axis"]) == "h":
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc or not _can_place(canvas, sprite, mr, c, bg):
                continue
            pos = (mr, c)
            chosen, margin, tied = _nearest_anchor_data(pos, same_color_anchors)
            if tied or not _same_anchor(chosen, target_anchor) or margin < _NEAREST_MARGIN_MIN:
                continue
            item = (int(margin), abs(int(c) - int(mc)), pos)
            if abs(int(c) - int(mc)) >= _SCATTER_MOVE_MIN:
                strong.append(item)
            else:
                weak.append(item)
    else:
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if r == mr or not _can_place(canvas, sprite, r, mc, bg):
                continue
            pos = (r, mc)
            chosen, margin, tied = _nearest_anchor_data(pos, same_color_anchors)
            if tied or not _same_anchor(chosen, target_anchor) or margin < _NEAREST_MARGIN_MIN:
                continue
            item = (int(margin), abs(int(r) - int(mr)), pos)
            if abs(int(r) - int(mr)) >= _SCATTER_MOVE_MIN:
                strong.append(item)
            else:
                weak.append(item)
    scored = strong or weak
    if not scored:
        return []
    best_margin = max(int(item[0]) for item in scored)
    best_shift = max(int(item[1]) for item in scored if int(item[0]) == best_margin)
    candidates = [pos for margin, shift, pos in scored if int(margin) == best_margin and int(shift) == best_shift]
    random.shuffle(candidates)
    return candidates


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((count for color, count in counts.items() if color != int(bg)), default=0)


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchor_sets(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    contest_color = random.choice(anchor_colors)
    contest_axis = random.choice(["h", "v"])
    anchors: list[dict[str, int | str]] = []

    contest_pair = _pair_for_axis(out, bg, contest_axis, contest_color)
    if contest_pair is None:
        return None
    for anchor in contest_pair:
        anchors.append(anchor)
        _put_anchor(out, anchor["r"], anchor["c"], contest_color)
        _put_anchor(inp, anchor["r"], anchor["c"], contest_color)

    orth_axis = "v" if contest_axis == "h" else "h"
    avoid = [(int(anchor["r"]), int(anchor["c"])) for anchor in contest_pair]
    extra_pair = _pair_for_axis(
        out,
        bg,
        orth_axis,
        contest_color,
        avoid=avoid,
        min_distance=_CONTEST_ANCHOR_MIN_DISTANCE,
    )
    if extra_pair is None:
        return None
    for anchor in extra_pair:
        anchors.append(anchor)
        _put_anchor(out, anchor["r"], anchor["c"], contest_color)
        _put_anchor(inp, anchor["r"], anchor["c"], contest_color)

    for color in anchor_colors:
        if int(color) == int(contest_color):
            continue
        axes = ["h", "v"]
        random.shuffle(axes)
        pair = None
        for axis in axes:
            pair = _pair_for_axis(out, bg, axis, color)
            if pair is not None:
                break
        if pair is None:
            return None
        for anchor in pair:
            anchors.append(anchor)
            _put_anchor(out, anchor["r"], anchor["c"], color)
            _put_anchor(inp, anchor["r"], anchor["c"], color)

    return anchor_colors, contest_color, contest_axis, anchors


def _sprite_targets(anchor_colors, contest_color, contest_axis, anchors):
    grouped = _anchors_by_color(anchors)
    contest_anchors = grouped[int(contest_color)]
    base_pair = [anchor for anchor in contest_anchors if str(anchor["axis"]) == contest_axis]
    cross_pair = [anchor for anchor in contest_anchors if str(anchor["axis"]) != contest_axis]
    if len(base_pair) != 2 or len(cross_pair) != 2:
        return None

    targets = [base_pair[0], base_pair[1], random.choice(cross_pair)]
    for color in anchor_colors:
        if int(color) == int(contest_color):
            continue
        targets.append(random.choice(grouped[int(color)]))

    min_sprites = max(_SPRITE_COUNT_RANGE[0], len(targets))
    max_sprites = _SPRITE_COUNT_RANGE[1]
    if min_sprites > max_sprites:
        return None
    n_sprites = random.randint(min_sprites, max_sprites)

    weighted_colors = [contest_color, contest_color] + list(anchor_colors)
    while len(targets) < n_sprites:
        color = random.choice(weighted_colors)
        targets.append(random.choice(grouped[int(color)]))
    random.shuffle(targets)
    return targets


def _place_aligned_sprites(out, targets, bg):
    used_body = set()
    sprites: list[dict[str, object]] = []
    for anchor in targets:
        marker_color = int(anchor["color"])
        body = _body_color(bg, marker_color, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker_color, bg)
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
                "marker_color": marker_color,
                "out_marker": (int(marker_pos[0]), int(marker_pos[1])),
                "area": int((sprite["patch"] != int(bg)).sum()),
            }
        )
    return sprites


def _scatter_sprites(inp, sprites, anchors_by_color, bg):
    order = sorted(
        range(len(sprites)),
        key=lambda idx: (len(anchors_by_color[int(sprites[idx]["marker_color"])]), int(sprites[idx]["area"])),
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


def _fusion_profile(inp: np.ndarray) -> dict[str, object]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    moves = _collect_moves(g, bg, anchors_by_color, anchor_cells, choose_first_anchor=False)
    profile = {
        "contest_colors": 0,
        "axis_split_colors": 0,
        "pair_split_colors": 0,
        "usable_colors": [],
    }
    for color, anchors in anchors_by_color.items():
        axes = {str(anchor["axis"]) for anchor in anchors}
        if len(anchors) < 3 or len(axes) < 2:
            continue
        chosen = [move["anchor"] for move in moves if int(move["marker_color"]) == int(color)]
        if not chosen:
            continue
        chosen_axes = {str(anchor["axis"]) for anchor in chosen}
        distinct_by_axis = {
            axis: {
                (int(anchor["r"]), int(anchor["c"]))
                for anchor in chosen
                if str(anchor["axis"]) == axis
            }
            for axis in ("h", "v")
        }
        profile["contest_colors"] += 1
        if len(chosen_axes) >= 2:
            profile["axis_split_colors"] += 1
        if any(len(coords) >= 2 for coords in distinct_by_axis.values()):
            profile["pair_split_colors"] += 1
        profile["usable_colors"].append(
            {
                "color": int(color),
                "anchor_count": len(anchors),
                "chosen_axes": sorted(chosen_axes),
                "distinct_h": len(distinct_by_axis["h"]),
                "distinct_v": len(distinct_by_axis["v"]),
            }
        )
    return profile


def _fusion_visible(inp):
    profile = _fusion_profile(inp)
    return bool(profile["contest_colors"] and profile["axis_split_colors"] and profile["pair_split_colors"])


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    if not bool(np.array_equal(main(inp), out)):
        return False
    if not _variant_visible(inp):
        return False
    return _fusion_visible(inp)


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchor_sets(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, contest_color, contest_axis, anchors = placed
    targets = _sprite_targets(anchor_colors, contest_color, contest_axis, anchors)
    if targets is None:
        return None
    sprites = _place_aligned_sprites(out, targets, bg)
    if sprites is None:
        return None
    anchors_by_color = _anchors_by_color(anchors)
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
    raise RuntimeError("failed to generate a valid fused 581f7754 input")
