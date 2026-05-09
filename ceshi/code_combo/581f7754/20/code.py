# Combo variant of 581f7754 (index 20)
# combo_spec: D1/1+D9/0
# Name: 体色匹配锚点+多数即标记
#
# Design analysis
# a) Concrete behavior / trace:
#    - Original task: a 2-color sprite uses its rare color twice: that rare color chooses the
#      matching border anchor, and the rare pixel itself is the alignment point that must share
#      the anchor's row/column after sliding.
#    - This combo inverts both decisions onto the majority color. A sprite is built as:
#        * a large majority-colored region,
#        * a small minority-colored appendage,
#        * exactly one majority-colored connector cell touching that appendage.
#      The majority color now picks the anchor, and that connector cell is the point aligned to
#      the anchor axis. The minority patch is no longer semantically privileged.
#    - Example: suppose a sprite has 7 red cells and 2 blue cells, with one red connector cell
#      touching the blue tail. If the red anchor sits on the top edge at column 8, the combo
#      solver slides the whole sprite horizontally so that the connector red cell reaches column 8.
#      The original solver would instead read blue as the marker and either target a blue anchor
#      or align a blue tail cell, producing a different motion.
# b) Complexity calibration:
#    - Majority area is kept in 5-12 cells and minority area in 1-3 cells so the "majority wins"
#      cue is visually obvious and ties cannot occur.
#    - Anchors remain 1-3 colors and sprites remain 3-5, so anchor colors repeat across multiple
#      sprites and the solver still needs grouping rather than one-off memorization.
#    - When multiple anchors exist, minority colors are often chosen from a different anchor color.
#      This creates visible false leads: the small patch looks anchor-like, but the large patch is
#      the true control signal.
# c) Hidden assumptions review:
#    - Anchor detection stays valid because anchors are still isolated border singletons.
#    - Component role parsing changes: the solver now needs a unique majority color, not a unique
#      minority color. The generator guarantees marker_area > body_area.
#    - Alignment-point extraction changes more substantially: the majority color can occupy many
#      cells, so the solver assumes exactly one majority-colored cell is 4-neighbor-adjacent to
#      the minority patch. The generator enforces this by attaching the tail through one connector.
#    - The old "least frequent color decides everything" assumption is intentionally broken. The
#      generator also rejects samples where the old solver or the D1-only solver would coincidentally
#      produce the same output, preventing silent degeneration.
# d) Placement legality:
#    - Generation remains output-first. Solved sprites are placed first, then scattered along the
#      inverse axis to form the input.
#    - `_can_place(...)` keeps every sprite in-bounds, non-overlapping, and separated from anchors
#      and other sprites by a one-cell moat, so connected-component parsing is stable.
#    - The minority tail is forced to start in a direction that changes the relevant alignment
#      coordinate, guaranteeing that "align majority connector" and "align minority tail" differ.
# e) Visual observability:
#    - Across examples, the large obvious color consistently matches the anchor that governs motion.
#      The small odd-colored tail often reuses another anchor color, so a human can see that the
#      tempting small patch is a decoy rather than the true key.
#    - No hidden ordering on color ids is used; only visible equality classes matter.
# f) Visual cleanliness:
#    - Sprites use only two colors and remain well separated by background.
#    - The minority patch is a compact appendage rather than scattered noise, so the connector cell
#      and the dominant color remain easy to parse by eye.
# g) Change inferability:
#    - The combo collapses both modified dimensions onto the same visible cue: the dominant color
#      controls both anchor selection and alignment semantics. Because the minority patch is small
#      and often misleading, a solver must infer the majority-based rule from the examples instead
#      of relying on raw color ids or hidden metadata.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


def _detect_anchors(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    H, W = g.shape

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
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
    return anchors_by_color, anchor_cells


def _majority_color(coords: np.ndarray, g: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    top = max(int(n) for n in cnt.values())
    colors = [int(col) for col, n in cnt.items() if int(n) == int(top)]
    if len(colors) != 1:
        return None
    return int(colors[0])


def _minority_color(coords: np.ndarray, g: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    bot = min(int(n) for n in cnt.values())
    colors = [int(col) for col, n in cnt.items() if int(n) == int(bot)]
    if len(colors) != 1:
        return None
    return int(colors[0])


def _majority_connector(coords: np.ndarray, g: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    comp_cells = {(int(r), int(c)) for r, c in coords}
    touching_other = []
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in comp_cells and int(g[int(rr), int(cc)]) != int(marker):
                touching_other.append((int(r), int(c)))
                break
    if len(touching_other) == 1:
        return touching_other[0]
    return None


def _minority_connector(coords: np.ndarray, g: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    comp_cells = {(int(r), int(c)) for r, c in coords}
    touching_other = []
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in comp_cells and int(g[int(rr), int(cc)]) != int(marker):
                touching_other.append((int(r), int(c)))
                break
    if len(touching_other) == 1:
        return touching_other[0]
    return touching_other[0] if touching_other else None


def _apply_moves(
    g: np.ndarray,
    anchors_by_color: dict[int, tuple[int, int]],
    moves: list[tuple[np.ndarray, int, int]],
    bg: int,
) -> np.ndarray:
    H, W = g.shape
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


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    assert anchors_by_color

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    H, W = g.shape
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _majority_color(coords, g)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)
            mp = (r, c)
        else:
            if marker_col not in anchors_by_color:
                continue
            mp = _majority_connector(coords, g, marker_col)
            if mp is None:
                continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr, dc = 0, 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc)))

    return _apply_moves(g, anchors_by_color, moves, bg)


def _solve_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    if not anchors_by_color:
        raise AssertionError("missing anchors")

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []
    H, W = g.shape

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _minority_color(coords, g)
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

        mp = _minority_connector(coords, g, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr, dc = 0, 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc)))

    return _apply_moves(g, anchors_by_color, moves, bg)


def _solve_d1_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    anchors_by_color, anchor_cells = _detect_anchors(g, bg)
    if not anchors_by_color:
        raise AssertionError("missing anchors")

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []
    H, W = g.shape

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        body_col = _majority_color(coords, g)
        marker_col = _minority_color(coords, g)
        if body_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            body_col = int(v)
            mp = (r, c)
        else:
            if body_col not in anchors_by_color or marker_col is None:
                continue
            mp = _minority_connector(coords, g, int(marker_col))
            if mp is None:
                continue

        ar, ac = anchors_by_color[int(body_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr, dc = 0, 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        moves.append((coords, int(dr), int(dc)))

    return _apply_moves(g, anchors_by_color, moves, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so the dominant-color connector cell lines up with an isolated border anchor of the same dominant color.

    Input structure:
    - The background is the most frequent color in the grid.
    - Interior sprites are isolated two-color components with a large dominant region and a small attached minority tail.
    - Isolated non-background border singletons act as anchors.

    Rule:
    - Treat each sprite's majority color as the anchor-matching color.
    - Within that majority region, use the unique majority-colored cell touching the minority tail as the alignment point.
    - Top/bottom anchors force a horizontal slide to match columns, while left/right anchors force a vertical slide to match rows; moved pixels clip at the border and never overwrite existing non-background cells.

    Key insight:
    - The conspicuous large color now controls both which anchor a sprite follows and which internal point is aligned, so the small odd-colored tail is a decoy rather than the key.
    """
    return _solve_combo(grid)


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
_SPRITE_COUNT_RANGE = (3, 5)
_MARKER_AREA_RANGE = (5, 12)
_BODY_AREA_RANGE = (1, 3)
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_RETRY_LIMIT = 120
_CROSS_ANCHOR_BODY_PROB = 0.75

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
_FLIP_PROB = 0.5

_DISTRACTOR_PROB = 0.12
_DISTRACTOR_COUNT_RANGE = (1, 3)

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


def _make_majority_shape(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _marker_neighbor_count(cell, marker_cells):
    r, c = cell
    return sum((r + dr, c + dc) in marker_cells for dr, dc in _NEIGHBORS4)


def _tail_start_candidates(marker_cells, connector, motion_axis):
    cr, cc = connector
    out = []
    for dr, dc in _NEIGHBORS4:
        cell = (cr + dr, cc + dc)
        if cell in marker_cells:
            continue
        if _marker_neighbor_count(cell, marker_cells) != 1:
            continue
        if motion_axis == "h" and dc == 0:
            continue
        if motion_axis == "v" and dr == 0:
            continue
        out.append(cell)
    random.shuffle(out)
    return out


def _tail_frontier(tail, marker_cells):
    out = set()
    for r, c in tail:
        for dr, dc in _NEIGHBORS4:
            cell = (r + dr, c + dc)
            if cell in tail or cell in marker_cells:
                continue
            if _marker_neighbor_count(cell, marker_cells) != 0:
                continue
            out.add(cell)
    return list(out)


def _grow_tail(marker_cells, connector, area, motion_axis):
    for start in _tail_start_candidates(marker_cells, connector, motion_axis):
        tail = {start}
        while len(tail) < area:
            frontier = _tail_frontier(tail, marker_cells)
            if not frontier:
                break
            tail.add(random.choice(frontier))
        if len(tail) == area:
            return tail
    return None


def _make_sprite(marker_area, body_area, marker_color, body_color, bg, motion_axis):
    for _ in range(_SPRITE_BUILD_RETRY_LIMIT):
        marker_cells = _make_majority_shape(marker_area)
        connectors = list(marker_cells)
        random.shuffle(connectors)
        for connector in connectors:
            tail = _grow_tail(marker_cells, connector, body_area, motion_axis)
            if tail is None:
                continue
            all_cells = set(marker_cells) | set(tail)
            r0, c0, r1, c1 = _bbox(all_cells)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
            for r, c in marker_cells:
                patch[r - r0, c - c0] = int(marker_color)
            for r, c in tail:
                patch[r - r0, c - c0] = int(body_color)
            return {
                "patch": patch,
                "offset": (connector[0] - r0, connector[1] - c0),
            }
    return None


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


def _body_color_sequence(anchor_colors, bg, marker_colors):
    body_colors = []
    cross_count = 0
    for marker in marker_colors:
        other_anchors = [c for c in anchor_colors if c != int(marker)]
        off_anchor_pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in anchor_colors]
        if other_anchors and random.random() < _CROSS_ANCHOR_BODY_PROB:
            body = random.choice(other_anchors)
            cross_count += 1
        else:
            pool = off_anchor_pool or other_anchors or [c for c in _COLORS if c not in {int(bg), int(marker)}]
            body = random.choice(pool)
        body_colors.append(int(body))

    if len(anchor_colors) >= 2 and cross_count == 0:
        i = random.randrange(len(marker_colors))
        body_colors[i] = int(random.choice([c for c in anchor_colors if c != int(marker_colors[i])]))
    return body_colors


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _add_distractors(out, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
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


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(out, inp, bg):
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    required_axes: list[str | None] = [None] * len(anchor_colors)
    if len(anchor_colors) >= 2:
        axis_pair = ["h", "v"]
        random.shuffle(axis_pair)
        required_axes[0], required_axes[1] = axis_pair
    anchors = []
    for color, required_axis in zip(anchor_colors, required_axes):
        candidates = _anchor_candidates(out, bg)
        if not candidates:
            return None
        if required_axis is not None:
            axis_candidates = [cand for cand in candidates if cand[0] == required_axis]
            if not axis_candidates:
                return None
            candidates = axis_candidates
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, body_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    sprites = []
    for marker_color, body_color in zip(marker_colors, body_colors):
        anchor = by_color[int(marker_color)]
        sprite = _make_sprite(
            random.randint(*_MARKER_AREA_RANGE),
            random.randint(*_BODY_AREA_RANGE),
            int(marker_color),
            int(body_color),
            bg,
            anchor["axis"],
        )
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos, int(body_color)))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos, _body_color in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    combo = main(inp)
    if not bool(np.array_equal(combo, out)):
        return False
    if bool(np.array_equal(_solve_original(inp), out)):
        return False
    if bool(np.array_equal(_solve_d1_only(inp), out)):
        return False
    return True


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    body_colors = _body_color_sequence(anchor_colors, bg, marker_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, body_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
        return None

    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
