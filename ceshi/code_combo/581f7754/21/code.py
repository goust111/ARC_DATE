# Combo variant of 581f7754 (index 21)
# combo_spec: D5/2+D8/3
# Name: 覆盖优先+按颜色分组处理

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The base task still consists of isolated border anchors plus interior sprites whose least
#    frequent color is a marker. Each sprite slides toward the anchor of the same marker color,
#    using the anchor edge to choose the slide axis. The combo changes two execution stages at
#    once:
#    - sprites are no longer resolved in one global simultaneous pass;
#    - moved pixels now overwrite any occupied destination cell.
#    A small concrete trace is:
#      * color-1 group is earlier, color-2 group is later;
#      * a color-1 sprite wants to land on cells currently occupied by a color-2 source sprite;
#      * because overwrite is allowed, color-1 temporarily paints over those cells;
#      * when color-2's turn arrives, clearing its own source cells deletes that temporary paint;
#      * color-2 then lands on top of some color-1 destination cells, and because overwrite is
#        allowed again, the later group visibly covers the earlier group.
#    Compared with the original task, both the timing of clearing and the destination collision
#    rule change, so the output can differ in two separate ways on the same board.
# b) Complexity calibration:
#    The generator uses exactly two anchor colors and repeats both colors across multiple sprites.
#    That keeps the hidden state small enough to infer while still making "same marker color moves
#    together" observable. Only one deliberate cross-group overwrite pair and one deliberate
#    cross-group source/destination interference pair are forced, so the board stays readable.
# c) Implicit assumptions audit for main():
#    - Anchor detection still assumes border singletons with 4-neighbor isolation; generator keeps
#      that invariant.
#    - Marker detection still assumes the minority color in a multi-color component; generator
#      keeps sprites bi-color with one marker pixel.
#    - Single-cell movable distractors are still supported by main(), but the generator does not
#      need them for this combo.
#    - The old "clear everything, then paste everything" assumption is invalid here; main() must
#      instead clear and paste one marker-color group at a time.
#    - Overwrite now matters, so main() cannot skip writes onto non-background cells.
#    - Same-color sprites are conceptually simultaneous; therefore the generator guarantees that
#      sprites within one color group do not overlap after translation, avoiding a hidden
#      within-group tie-break rule.
# d) Placement legality:
#    Input sprites are still scattered as separated interior components with a background gap.
#    Planned aligned positions are also separated except for one controlled cross-group overlap
#    pair. All candidate placements are checked against bounds, anchors, and unwanted contacts.
# e) Visual observability:
#    Numeric color order is normally a weak cue, so the generator fixes the marker palette to the
#    same two recurring visible colors across samples and repeats each color within a board. A
#    human can infer from examples that one color-group consistently settles first and the other
#    settles second, while matching sprites of the same marker color move together.
# f) Visual cleanliness:
#    The combo effect is localized: one later group overwrites one earlier group's settled area,
#    and one later source blocks cells that an earlier group would otherwise keep. Other sprites
#    remain separated, so the overlap does not turn the whole board into clutter.
# g) Color-change / order inference:
#    No recoloring happens. Colors still serve the visible marker-to-anchor match, and only two
#    marker colors are used, reducing ambiguity about which group is earlier.
# h) Generator-side anti-degeneration checks:
#    The generator requires the fused solver output to differ from:
#    - grouped processing without overwrite (so D5 is active), and
#    - simultaneous-clear processing with overwrite (so D8 timing is active).
#    It also measures the underlying conflict counts directly.
# i) New assumptions introduced by the combo:
#    main() assumes each marker color maps to exactly one anchor and that grouped clearing never
#    erases another source component from the input. The generator guarantees unique anchors per
#    marker color and keeps input components non-overlapping.


_COLORS = tuple(range(10))
_MARKER_COLORS = (1, 2)
_BG_COLOR_RANGE = (5, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 700
_MIN_OVERWRITE_DIFF = 1
_MIN_TIMING_DIFF = 1
_MIN_OVERWRITE_CONFLICTS = 1
_MIN_TIMING_CONFLICTS = 1

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

_GROUP_SIZES = {1: 2, 2: 2}
_TARGET_OVERLAP_TOP_K = 5

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


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


def _collect_scene(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(r, c, H, W):
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
    group_order = sorted(int(color) for color in anchors_by_color)
    moves_by_color: dict[int, list[dict]] = {int(color): [] for color in group_order}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
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

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves_by_color[int(marker_col)].append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker": int(marker_col),
            }
        )

    return g, bg, anchors_by_color, group_order, moves_by_color


def _apply_moves(
    g: np.ndarray,
    bg: int,
    group_order: list[int],
    moves_by_color: dict[int, list[dict]],
    *,
    grouped: bool,
    overwrite_non_bg: bool,
) -> np.ndarray:
    H, W = g.shape
    out = g.copy()

    if grouped:
        groups = [moves_by_color.get(int(color), []) for color in group_order]
    else:
        groups = [[move for color in group_order for move in moves_by_color.get(int(color), [])]]

    for group in groups:
        if not group:
            continue
        for move in group:
            coords = move["coords"]
            out[coords[:, 0], coords[:, 1]] = int(bg)
        for move in group:
            coords = move["coords"]
            dr = int(move["dr"])
            dc = int(move["dc"])
            for r, c in coords:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if overwrite_non_bg or int(out[rr, cc]) == int(bg):
                        out[rr, cc] = int(g[int(r), int(c)])
    return out


def _solve_variant(grid: np.ndarray, *, grouped: bool, overwrite_non_bg: bool) -> np.ndarray:
    g, bg, _anchors_by_color, group_order, moves_by_color = _collect_scene(np.array(grid, dtype=int))
    return _apply_moves(g, bg, group_order, moves_by_color, grouped=grouped, overwrite_non_bg=overwrite_non_bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior marker-bearing sprites toward matching border anchors, resolving one marker-color group at a time with overwrite priority.

    Input structure:
    - The modal color is the background.
    - Interior foreground components are mostly two-color sprites whose minority color is the marker; isolated interior singletons of an anchor color are also valid movers.
    - Border anchors are isolated single pixels, and the side they sit on still determines whether matching movers slide horizontally or vertically.

    Rule:
    - Match each movable component to the unique anchor of the same marker color.
    - Top or bottom anchors force a horizontal slide to the anchor column; left or right anchors force a vertical slide to the anchor row.
    - Components sharing one marker color move together, different marker colors are processed in ascending color order, and each group's translated pixels overwrite whatever is currently present; later groups therefore cover earlier groups, while pixels pushed out of bounds are clipped.

    Key insight:
    - The puzzle is no longer one simultaneous translation: you must infer both the color-group schedule and the fact that every moved pixel is painted with priority, so later groups can erase or repaint what earlier groups created.
    """
    return _solve_variant(grid, grouped=True, overwrite_non_bg=True)


def _grouped_no_overwrite(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, grouped=True, overwrite_non_bg=False)


def _simultaneous_overwrite(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, grouped=False, overwrite_non_bg=True)


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


def _sprite_cell_map(sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    patch = sprite["patch"]
    out = {}
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v != int(bg):
                out[(int(top + r), int(left + c))] = int(v)
    return out


def _sprite_cells(sprite, marker_r, marker_c, bg):
    return set(_sprite_cell_map(sprite, marker_r, marker_c, bg))


def _cells_touch(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for r, c in cells_a:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if (int(rr), int(cc)) in cells_b:
                    return True
    return False


def _map_overlap_diff(a: dict[tuple[int, int], int], b: dict[tuple[int, int], int]) -> int:
    return sum(1 for pos, col in a.items() if pos in b and int(b[pos]) != int(col))


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


def _place(canvas, sprite, marker_r, marker_c, bg, *, overwrite=True):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), overwrite=overwrite, clip=True)


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


def _anchor_candidates_for_axis(canvas, bg, axis):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    if axis == "h":
        raw = [("h", 0, c) for c in cols] + [("h", H - 1, c) for c in cols]
    else:
        raw = [("v", r, 0) for r in rows] + [("v", r, W - 1) for r in rows]
    random.shuffle(raw)
    return [(kind, r, c) for kind, r, c in raw if _anchor_ok(canvas, r, c, bg)]


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
    plan = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, plan, inp


def _place_anchors(plan, inp, bg):
    colors = list(_MARKER_COLORS)
    axes = ["h", "v"]
    random.shuffle(axes)
    anchors = []
    for color, axis in zip(colors, axes):
        candidates = _anchor_candidates_for_axis(plan, bg, axis)
        if not candidates:
            return None
        _, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(plan, r, c, color)
        _put_anchor(inp, r, c, color)
    anchors.sort(key=lambda a: int(a["color"]))
    return anchors


def _choose_disjoint_position(anchor_canvas, sprite, anchor, existing_cells, bg):
    positions = _aligned_positions(anchor_canvas, sprite, anchor, bg)
    random.shuffle(positions)
    for marker_pos in positions:
        cells = _sprite_cells(sprite, marker_pos[0], marker_pos[1], bg)
        if not _cells_touch(cells, existing_cells):
            return (int(marker_pos[0]), int(marker_pos[1])), cells
    return None


def _choose_overlap_position(anchor_canvas, sprite, anchor, target_map, forbidden_cells, bg):
    positions = _aligned_positions(anchor_canvas, sprite, anchor, bg)
    scored = []
    for marker_pos in positions:
        cell_map = _sprite_cell_map(sprite, marker_pos[0], marker_pos[1], bg)
        cells = set(cell_map)
        if forbidden_cells and _cells_touch(cells, forbidden_cells):
            continue
        overlap = _map_overlap_diff(cell_map, target_map)
        if overlap > 0:
            scored.append((int(overlap), (int(marker_pos[0]), int(marker_pos[1])), cell_map))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    window = scored[: min(_TARGET_OVERLAP_TOP_K, len(scored))]
    overlap, marker_pos, cell_map = random.choice(window)
    return marker_pos, cell_map, int(overlap)


def _plan_records(plan, anchors, bg):
    anchor_canvas = plan.copy()
    by_color = {int(a["color"]): a for a in anchors}
    used_body = set()
    records: list[dict] = []
    occupied_without_overlap: set[tuple[int, int]] = set()

    color1, color2 = (int(c) for c in _MARKER_COLORS)
    anchor1 = by_color[int(color1)]
    anchor2 = by_color[int(color2)]

    for _ in range(_GROUP_SIZES[int(color1)]):
        body = _body_color(bg, color1, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, color1, bg)
        chosen = _choose_disjoint_position(anchor_canvas, sprite, anchor1, occupied_without_overlap, bg)
        if chosen is None:
            return None
        marker_pos, cells = chosen
        records.append(
            {
                "sprite": sprite,
                "anchor": anchor1,
                "marker_color": int(color1),
                "final_marker": marker_pos,
                "final_map": _sprite_cell_map(sprite, marker_pos[0], marker_pos[1], bg),
                "role": "regular",
            }
        )
        occupied_without_overlap |= set(cells)

    target = random.choice([rec for rec in records if int(rec["marker_color"]) == int(color1)])
    body = _body_color(bg, color2, used_body)
    used_body.add(body)
    overlap_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, color2, bg)
    chosen_overlap = _choose_overlap_position(
        anchor_canvas,
        overlap_sprite,
        anchor2,
        target["final_map"],
        occupied_without_overlap - set(target["final_map"]),
        bg,
    )
    if chosen_overlap is None:
        return None
    marker_pos, final_map, overlap = chosen_overlap
    overlap_record = {
        "sprite": overlap_sprite,
        "anchor": anchor2,
        "marker_color": int(color2),
        "final_marker": marker_pos,
        "final_map": final_map,
        "role": "overlap",
        "overwrites_target": target,
        "planned_overlap_cells": int(overlap),
    }
    records.append(overlap_record)

    occupied_for_regular = occupied_without_overlap | set(final_map)
    for _ in range(_GROUP_SIZES[int(color2)] - 1):
        body = _body_color(bg, color2, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, color2, bg)
        chosen = _choose_disjoint_position(anchor_canvas, sprite, anchor2, occupied_for_regular, bg)
        if chosen is None:
            return None
        marker_pos, cells = chosen
        rec = {
            "sprite": sprite,
            "anchor": anchor2,
            "marker_color": int(color2),
            "final_marker": marker_pos,
            "final_map": _sprite_cell_map(sprite, marker_pos[0], marker_pos[1], bg),
            "role": "regular",
        }
        records.append(rec)
        occupied_for_regular |= set(cells)

    return records


def _choose_source_with_timing_overlap(inp, record, target_cells, bg):
    positions = _scatter_positions(inp, record["sprite"], record["anchor"], record["final_marker"], bg)
    scored = []
    for marker_pos in positions:
        cells = _sprite_cells(record["sprite"], marker_pos[0], marker_pos[1], bg)
        overlap = len(cells & target_cells)
        if overlap > 0:
            scored.append((int(overlap), (int(marker_pos[0]), int(marker_pos[1])), cells))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    window = scored[: min(_TARGET_OVERLAP_TOP_K, len(scored))]
    overlap, marker_pos, cells = random.choice(window)
    return marker_pos, cells, int(overlap)


def _place_any_source(inp, record, bg):
    positions = _scatter_positions(inp, record["sprite"], record["anchor"], record["final_marker"], bg)
    if not positions:
        return None
    marker_pos = random.choice(positions)
    cells = _sprite_cells(record["sprite"], marker_pos[0], marker_pos[1], bg)
    return (int(marker_pos[0]), int(marker_pos[1])), cells


def _scatter_records(inp, records, bg):
    overlap_record = next(rec for rec in records if rec["role"] == "overlap")
    target_cells = set(overlap_record["overwrites_target"]["final_map"])
    chosen = _choose_source_with_timing_overlap(inp, overlap_record, target_cells, bg)
    if chosen is None:
        return False
    marker_pos, cells, overlap = chosen
    _place(inp, overlap_record["sprite"], marker_pos[0], marker_pos[1], bg)
    overlap_record["source_marker"] = marker_pos
    overlap_record["source_cells"] = cells
    overlap_record["timing_overlap_cells"] = int(overlap)

    others = [rec for rec in records if rec is not overlap_record]
    random.shuffle(others)
    for rec in others:
        chosen = _place_any_source(inp, rec, bg)
        if chosen is None:
            return False
        marker_pos, cells = chosen
        _place(inp, rec["sprite"], marker_pos[0], marker_pos[1], bg)
        rec["source_marker"] = marker_pos
        rec["source_cells"] = cells
    return True


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    fused = main(grid)
    grouped_no_overwrite = _grouped_no_overwrite(grid)
    simultaneous_overwrite = _simultaneous_overwrite(grid)

    g, _bg, _anchors_by_color, group_order, moves_by_color = _collect_scene(np.array(grid, dtype=int))
    H, W = g.shape

    later_sources: dict[int, set[tuple[int, int]]] = {}
    running_sources: set[tuple[int, int]] = set()
    for color in reversed(group_order):
        later_sources[int(color)] = set(running_sources)
        for move in moves_by_color.get(int(color), []):
            running_sources |= {(int(r), int(c)) for r, c in move["coords"]}

    earlier_destinations: dict[int, dict[tuple[int, int], int]] = {}
    running_destinations: dict[tuple[int, int], int] = {}
    for color in group_order:
        earlier_destinations[int(color)] = dict(running_destinations)
        for move in moves_by_color.get(int(color), []):
            for r, c in move["coords"]:
                rr = int(r) + int(move["dr"])
                cc = int(c) + int(move["dc"])
                if 0 <= rr < H and 0 <= cc < W:
                    running_destinations[(int(rr), int(cc))] = int(g[int(r), int(c)])

    timing_conflicts = 0
    overwrite_conflicts = 0
    repeated_groups = 0
    for color in group_order:
        group = moves_by_color.get(int(color), [])
        if len(group) >= 2:
            repeated_groups += 1
        blockers = later_sources[int(color)]
        prior = earlier_destinations[int(color)]
        for move in group:
            for r, c in move["coords"]:
                rr = int(r) + int(move["dr"])
                cc = int(c) + int(move["dc"])
                if 0 <= rr < H and 0 <= cc < W:
                    pos = (int(rr), int(cc))
                    src_col = int(g[int(r), int(c)])
                    if pos in blockers:
                        timing_conflicts += 1
                    if pos in prior and int(prior[pos]) != int(src_col):
                        overwrite_conflicts += 1

    return {
        "overwrite_diff": int(np.count_nonzero(fused != grouped_no_overwrite)),
        "timing_diff": int(np.count_nonzero(fused != simultaneous_overwrite)),
        "overwrite_conflicts": int(overwrite_conflicts),
        "timing_conflicts": int(timing_conflicts),
        "repeated_groups": int(repeated_groups),
    }


def _generated_grid_is_valid(inp, bg):
    if not _bg_is_mode(inp, bg):
        return False
    metrics = _variant_metrics(inp)
    if metrics["repeated_groups"] < 2:
        return False
    if metrics["overwrite_conflicts"] < _MIN_OVERWRITE_CONFLICTS:
        return False
    if metrics["timing_conflicts"] < _MIN_TIMING_CONFLICTS:
        return False
    if metrics["overwrite_diff"] < _MIN_OVERWRITE_DIFF:
        return False
    if metrics["timing_diff"] < _MIN_TIMING_DIFF:
        return False
    return True


def _gen_attempt():
    bg, plan, inp = _new_canvas_pair()
    anchors = _place_anchors(plan, inp, bg)
    if anchors is None:
        return None

    records = _plan_records(plan, anchors, bg)
    if records is None:
        return None

    if not _scatter_records(inp, records, bg):
        return None

    if not _generated_grid_is_valid(inp, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
