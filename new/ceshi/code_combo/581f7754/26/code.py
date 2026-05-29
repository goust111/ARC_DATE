# Combo variant of 581f7754 (index 26)
# combo_spec: D2/3+D3/0+D5/0
# Name: 双轴对齐+多标记+环绕

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant:
# - Concrete fused behavior:
#   The old task used one marker pixel, one active slide axis, and clipping. This combo changes
#   all three together: each sprite now has a 2- or 3-pixel collinear marker set, the effective
#   point is the marker-line center, the destination is the inward-neighbor cell of the matching
#   border anchor, and any translated pixel that leaves the grid wraps around toroidally.
# - Behavior trace:
#   Suppose a top anchor of color 4 is at (0,2), so its inward target is (1,2). In the original
#   task a single marker at (6,8) would only match the column and slide to (6,2), clipping any
#   out-of-bounds cells. In this combo a sprite might instead have three color-4 markers at
#   (7,7),(7,8),(7,9), whose center is (7,8). The whole sprite translates by (-6,-6) so that
#   the center lands at (1,2); if some body cells move to column -1 they reappear at W-1 rather
#   than disappearing. The visible delta is therefore both "2D landing instead of 1D landing"
#   and "split-across-edges instead of cropped".
# - Complexity calibration:
#   A dual-axis inward target gives each anchor color one exact landing point, so allowing
#   repeated marker colors would mostly create forced collisions. The generator therefore uses
#   2-3 anchors and exactly one sprite per anchor color. To keep the multi-marker rule obvious,
#   every sample contains at least one 2-marker sprite and one 3-marker sprite, and scattered
#   inputs always differ from the target on both row and column. To keep the wrap rule obvious,
#   one designated sprite is forced to wrap by at least two pixels in the solved output.
# - Implicit-assumption audit:
#   The old `marker_pixel(...)` assumed one privileged pixel; that fails and is replaced by a
#   center-of-marker-line calculation. The old translation logic assumed either `dr` or `dc`
#   stayed zero; that also fails because the inward target fixes both coordinates. The old paste
#   step clipped with bounds checks; this combo must paste with modulo arithmetic instead. The
#   generator can no longer rely on a free alignment axis or on repeated marker colors, so it
#   builds one fixed solved target per anchor and scatters sprites in full 2D.
# - Placement legality:
#   Input sprites are still placed fully inside the border with 8-neighborhood isolation, so
#   object recognition remains clean. Solved sprites are placed on an anchor-free output canvas
#   first because they are allowed to sit adjacent to their own anchor after the overlay step.
#   A separate anchor-respect check forbids overlapping any anchor cell and forbids touching
#   non-matching anchors; wrapped destinations also reject self-overlap and collisions.
# - Visual observability:
#   Across 2-4 examples, a solver can see three stable cues: marker groups are always 2-3 cells
#   in one row or one column, outputs land immediately inside the matching anchor rather than
#   merely sharing one coordinate, and at least one solved sprite comes back from the opposite
#   side instead of losing pixels. No color-id arithmetic or hidden ordering is used.
# - Cleanliness and new solver assumptions:
#   The new `main()` assumes the minority color inside each movable sprite is the marker color
#   and that its 2-3 pixels are collinear; the generator enforces both. It also assumes there is
#   at most one anchor per marker color and that wrapped placements are collision-free; the
#   generator enforces unique anchor colors and validates the toroidal destination layout.


def _median_int(values: list[int]) -> int:
    values = sorted(int(v) for v in values)
    n = len(values)
    if n == 0:
        raise ValueError("empty median")
    if n % 2 == 1:
        return int(values[n // 2])
    total = int(values[n // 2 - 1] + values[n // 2])
    if total % 2 == 0:
        return int(total // 2)
    return int((total + 1) // 2)


def _anchor_target(anchor_r: int, anchor_c: int, shape: tuple[int, int]) -> tuple[int, int]:
    H, W = map(int, shape)
    tr = int(anchor_r)
    tc = int(anchor_c)
    if tr == 0:
        tr = 1
    elif tr == H - 1:
        tr = H - 2
    if tc == 0:
        tc = 1
    elif tc == W - 1:
        tc = W - 2
    return int(tr), int(tc)


def _detect_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
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
    return anchors_by_color


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_points(g: np.ndarray, coords: np.ndarray, marker: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]


def _marker_alignment_point(marker_pts: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not marker_pts:
        return None
    rows = [int(r) for r, _c in marker_pts]
    cols = [int(c) for _r, c in marker_pts]
    row_set = set(rows)
    col_set = set(cols)
    if len(row_set) == 1:
        row = int(rows[0])
        total = sum(cols)
        if total % len(cols) == 0:
            return int(row), int(total // len(cols))
        return int(row), int(_median_int(cols))
    if len(col_set) == 1:
        col = int(cols[0])
        total = sum(rows)
        if total % len(rows) == 0:
            return int(total // len(rows)), int(col)
        return int(_median_int(rows)), int(col)
    return int(_median_int(rows)), int(_median_int(cols))


def _collect_moves(grid: np.ndarray, *, dual_axis: bool) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors(g, bg)
    assert anchors_by_color

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[dict[str, object]] = []

    for comp in comps:
        coords = comp["coords"].astype(int)
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None or int(marker_col) not in anchors_by_color:
            continue

        marker_pts = _marker_points(g, coords, int(marker_col))
        if len(marker_pts) < 2:
            continue

        align_pt = _marker_alignment_point(marker_pts)
        if align_pt is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        if dual_axis:
            tr, tc = _anchor_target(int(ar), int(ac), g.shape)
            dr = int(tr - align_pt[0])
            dc = int(tc - align_pt[1])
        else:
            on_top_bottom = int(ar) in (0, int(H - 1))
            on_left_right = int(ac) in (0, int(W - 1))
            horizontal = bool(on_top_bottom and not on_left_right)
            dr = 0
            dc = 0
            if horizontal:
                dc = int(ac - align_pt[1])
            else:
                dr = int(ar - align_pt[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_color": int(marker_col),
                "marker_count": int(len(marker_pts)),
                "align_point": (int(align_pt[0]), int(align_pt[1])),
                "collinear": bool(len({r for r, _c in marker_pts}) == 1 or len({c for _r, c in marker_pts}) == 1),
            }
        )

    return g, bg, moves


def _apply_moves(g: np.ndarray, bg: int, moves: list[dict[str, object]], *, wrap: bool) -> np.ndarray:
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
    g, bg, dual_moves = _collect_moves(grid, dual_axis=True)
    _g2, _bg2, axis_moves = _collect_moves(grid, dual_axis=False)
    H, W = g.shape

    wrapped_pixels = 0
    wrapped_components = 0
    dual_axis_moves = 0
    marker2_count = 0
    marker3_count = 0
    all_markers_collinear = True

    for move in dual_moves:
        dr = int(move["dr"])
        dc = int(move["dc"])
        if dr != 0 and dc != 0:
            dual_axis_moves += 1
        if int(move["marker_count"]) == 2:
            marker2_count += 1
        if int(move["marker_count"]) == 3:
            marker3_count += 1
        all_markers_collinear &= bool(move["collinear"])

        comp_wrap = 0
        for r, c in move["coords"]:
            rr = int(r) + dr
            cc = int(c) + dc
            if not (0 <= rr < H and 0 <= cc < W):
                comp_wrap += 1
        wrapped_pixels += comp_wrap
        if comp_wrap:
            wrapped_components += 1

    combo = _apply_moves(g, bg, dual_moves, wrap=True)
    single_axis_wrap = _apply_moves(g, bg, axis_moves, wrap=True)
    dual_axis_clip = _apply_moves(g, bg, dual_moves, wrap=False)
    return {
        "move_count": int(len(dual_moves)),
        "dual_axis_moves": int(dual_axis_moves),
        "marker2_count": int(marker2_count),
        "marker3_count": int(marker3_count),
        "wrapped_pixels": int(wrapped_pixels),
        "wrapped_components": int(wrapped_components),
        "all_markers_collinear": bool(all_markers_collinear),
        "diff_vs_single_axis_wrap": bool(not np.array_equal(combo, single_axis_wrap)),
        "diff_vs_dual_axis_clip": bool(not np.array_equal(combo, dual_axis_clip)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move interior multi-marker sprites so each marker group's center lands just inside the border anchor of the same color.

    Input structure:
    - The background is the most frequent color in the grid.
    - Each movable interior sprite is an 8-connected multi-color component with one body color and 2-3 minority-color marker pixels.
    - The marker pixels of a sprite lie on one shared row or one shared column, and isolated border singletons of the same color act as anchors.

    Rule:
    - Match every sprite to the border anchor that shares its marker color.
    - Compute the effective marker position from the marker-line center, then translate the whole sprite in both row and column so that center reaches the inward-neighbor cell of the anchor.
    - After translation, any pixel that crosses a grid edge wraps to the opposite side; anchors stay fixed, source cells are cleared, and existing non-background cells are not overwritten.

    Key insight: The border pixel no longer picks a single slide axis; it fixes one exact interior landing point, so you must aggregate a multi-pixel marker into one center and then apply a toroidal 2D translation.
    """
    g, bg, moves = _collect_moves(grid, dual_axis=True)
    return _apply_moves(g, bg, moves, wrap=True)


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
_GEN_RETRY_LIMIT = 600
_SPRITE_BUILD_TRIES = 90
_MIN_WRAP_PIXELS = 2
_STRONG_SHIFT_MIN = 2

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

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _norm(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(int(r), int(mc - c)) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(int(mr - r), int(c)) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds: tuple[int, int]) -> range:
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.append((int(rr), int(cc)))
    return frontier


def _grow_body(area: int) -> set[tuple[int, int]]:
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


def _symmetric_body(area: int) -> set[tuple[int, int]]:
    choices = []
    for height in _inclusive_range(_BODY_HEIGHT_RANGE):
        extra = area - height
        if extra >= 2 and extra % 2 == 0 and extra // 2 <= height:
            choices.append((height, extra // 2))
    if not choices:
        return _grow_body(area)

    height, pair_count = random.choice(choices)
    cells = {(int(r), 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((int(r), 0))
        cells.add((int(r), 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(int(c), int(r)) for r, c in cells}
    return _flip(cells)


def _wide_body(area: int) -> set[tuple[int, int]]:
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, int(c)) for c in range(width)}
    cells.add((1, random.randrange(width)))
    while len(cells) < area:
        frontier = _frontier_cells(cells, height, width)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _frame_body(area: int) -> set[tuple[int, int]]:
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
        (int(r), int(c))
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


def _make_body(area: int) -> set[tuple[int, int]]:
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _marker_line_options(body: set[tuple[int, int]], orientation: str) -> list[dict[str, object]]:
    groups: dict[int, set[int]] = {}
    if orientation == "row":
        for r, c in body:
            for rr in (int(r - 1), int(r + 1)):
                if (rr, int(c)) not in body:
                    groups.setdefault(int(rr), set()).add(int(c))
    else:
        for r, c in body:
            for cc in (int(c - 1), int(c + 1)):
                if (int(r), cc) not in body:
                    groups.setdefault(int(cc), set()).add(int(r))

    options = []
    seen = set()
    for fixed, vars_ in groups.items():
        vals = sorted(int(v) for v in vars_)
        valset = set(vals)
        for center in vals:
            if {center - 1, center, center + 1}.issubset(valset):
                if orientation == "row":
                    cells = ((int(fixed), int(center - 1)), (int(fixed), int(center)), (int(fixed), int(center + 1)))
                    align = (int(fixed), int(center))
                else:
                    cells = ((int(center - 1), int(fixed)), (int(center), int(fixed)), (int(center + 1), int(fixed)))
                    align = (int(center), int(fixed))
                if cells not in seen:
                    seen.add(cells)
                    options.append({"cells": cells, "align": align, "count": 3, "orientation": orientation})
            if {center - 1, center + 1}.issubset(valset):
                if orientation == "row":
                    if (int(fixed), int(center)) in body:
                        continue
                    cells = ((int(fixed), int(center - 1)), (int(fixed), int(center + 1)))
                    align = (int(fixed), int(center))
                else:
                    if (int(center), int(fixed)) in body:
                        continue
                    cells = ((int(center - 1), int(fixed)), (int(center + 1), int(fixed)))
                    align = (int(center), int(fixed))
                if cells not in seen:
                    seen.add(cells)
                    options.append({"cells": cells, "align": align, "count": 2, "orientation": orientation})
    return options


def _make_sprite(
    area: int,
    body_color: int,
    marker_color: int,
    bg: int,
    *,
    required_count: int | None,
) -> dict[str, object] | None:
    for _ in range(_SPRITE_BUILD_TRIES):
        body = _make_body(area)
        options = _marker_line_options(body, "row") + _marker_line_options(body, "col")
        if required_count is not None:
            options = [opt for opt in options if int(opt["count"]) == int(required_count)]
        if not options:
            continue
        option = random.choice(options)
        markers = set(option["cells"])
        all_cells = set(body) | markers
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[int(r - r0), int(c - c0)] = int(body_color)
        for r, c in markers:
            patch[int(r - r0), int(c - c0)] = int(marker_color)
        ar, ac = option["align"]
        return {
            "patch": patch,
            "align_offset": (int(ar - r0), int(ac - c0)),
            "marker_count": int(option["count"]),
            "orientation": str(option["orientation"]),
            "nonbg_area": int((patch != int(bg)).sum()),
        }
    return None


def _top_left(sprite: dict[str, object], align_r: int, align_c: int) -> tuple[int, int]:
    off_r, off_c = sprite["align_offset"]
    return int(align_r) - int(off_r), int(align_c) - int(off_c)


def _touches_foreground(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas: np.ndarray, sprite: dict[str, object], align_r: int, align_c: int, bg: int) -> bool:
    patch = sprite["patch"]
    top, left = _top_left(sprite, align_r, align_c)
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


def _place(canvas: np.ndarray, sprite: dict[str, object], align_r: int, align_c: int, bg: int) -> None:
    top, left = _top_left(sprite, align_r, align_c)
    paste(canvas, sprite["patch"], int(top), int(left), transparent=int(bg))


def _wrapped_sprite_cells(
    canvas_shape: tuple[int, int],
    sprite: dict[str, object],
    align_r: int,
    align_c: int,
    bg: int,
) -> tuple[list[tuple[int, int, int]], int]:
    H, W = canvas_shape
    patch = sprite["patch"]
    top, left = _top_left(sprite, align_r, align_c)
    cells = []
    wrapped_pixels = 0
    for pr in range(patch.shape[0]):
        for pc in range(patch.shape[1]):
            color = int(patch[pr, pc])
            if color == int(bg):
                continue
            raw_r = int(top + pr)
            raw_c = int(left + pc)
            if not (0 <= raw_r < H and 0 <= raw_c < W):
                wrapped_pixels += 1
            cells.append((int(raw_r % H), int(raw_c % W), color))
    return cells, int(wrapped_pixels)


def _touches_foreground_cells(canvas: np.ndarray, occupied_cells: set[tuple[int, int]], bg: int) -> bool:
    H, W = canvas.shape
    for r, c in occupied_cells:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (int(rr), int(cc)) in occupied_cells:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return True
    return False


def _can_place_wrapped(
    canvas: np.ndarray,
    sprite: dict[str, object],
    align_r: int,
    align_c: int,
    bg: int,
    *,
    min_wrap_pixels: int = 0,
) -> bool:
    cells, wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, align_r, align_c, bg)
    if wrapped_pixels < int(min_wrap_pixels):
        return False

    occupied: set[tuple[int, int]] = set()
    for rr, cc, _color in cells:
        key = (int(rr), int(cc))
        if key in occupied:
            return False
        occupied.add(key)
        if int(canvas[rr, cc]) != int(bg):
            return False

    return not _touches_foreground_cells(canvas, occupied, bg)


def _place_wrapped(canvas: np.ndarray, sprite: dict[str, object], align_r: int, align_c: int, bg: int) -> int:
    cells, wrapped_pixels = _wrapped_sprite_cells(canvas.shape, sprite, align_r, align_c, bg)
    for rr, cc, color in cells:
        canvas[int(rr), int(cc)] = int(color)
    return int(wrapped_pixels)


def _isolated_bg_cell(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _anchor_candidates(canvas: np.ndarray, bg: int) -> list[tuple[str, int, int]]:
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("top", 0, int(c)))
        out.append(("bottom", int(H - 1), int(c)))
    for r in rows:
        out.append(("left", int(r), 0))
        out.append(("right", int(r), int(W - 1)))
    random.shuffle(out)
    return [(side, int(r), int(c)) for side, r, c in out if _anchor_ok(canvas, int(r), int(c), bg)]


def _anchor_is_wrap_friendly(side: str, r: int, c: int, shape: tuple[int, int]) -> bool:
    H, W = shape
    if side in ("top", "bottom"):
        return int(c) <= _WRAP_EDGE_BAND or int(c) >= int(W - 1 - _WRAP_EDGE_BAND)
    return int(r) <= _WRAP_EDGE_BAND or int(r) >= int(H - 1 - _WRAP_EDGE_BAND)


def _put_anchor(canvas: np.ndarray, r: int, c: int, color: int) -> None:
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _put_anchors(canvas: np.ndarray, anchors: list[dict[str, int | str]]) -> None:
    for anchor in anchors:
        _put_anchor(canvas, int(anchor["r"]), int(anchor["c"]), int(anchor["color"]))


def _body_color(bg: int, marker: int, used: set[int]) -> int:
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return int(random.choice(pool))


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair() -> tuple[int, np.ndarray, np.ndarray]:
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    return int(bg), np.full((H, W), int(bg), dtype=int), np.full((H, W), int(bg), dtype=int)


def _choose_anchors(shape: tuple[int, int], bg: int) -> tuple[np.ndarray, list[dict[str, int | str]], int] | None:
    anchor_canvas = np.full(shape, int(bg), dtype=int)
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], random.randint(*_ANCHOR_COUNT_RANGE))
    wrap_color = int(anchor_colors[0])
    anchors: list[dict[str, int | str]] = []
    for idx, color in enumerate(anchor_colors):
        candidates = _anchor_candidates(anchor_canvas, bg)
        if not candidates:
            return None
        if idx == 0:
            preferred = [cand for cand in candidates if _anchor_is_wrap_friendly(*cand, shape)]
            if preferred:
                candidates = preferred
        side, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "side": str(side)}
        anchors.append(anchor)
        _put_anchor(anchor_canvas, r, c, color)
    return anchor_canvas, anchors, wrap_color


def _cells_respect_anchors(
    cells: list[tuple[int, int, int]],
    anchors: list[dict[str, int | str]],
    own_color: int,
) -> bool:
    positions = {int(anchor["color"]): (int(anchor["r"]), int(anchor["c"])) for anchor in anchors}
    for rr, cc, _color in cells:
        for color, (ar, ac) in positions.items():
            if (int(rr), int(cc)) == (int(ar), int(ac)):
                return False
            if color == int(own_color):
                continue
            if abs(int(rr) - int(ar)) <= 1 and abs(int(cc) - int(ac)) <= 1:
                return False
    return True


def _required_counts_by_color(anchors: list[dict[str, int | str]], wrap_color: int) -> dict[int, int]:
    others = [int(anchor["color"]) for anchor in anchors if int(anchor["color"]) != int(wrap_color)]
    required = {int(wrap_color): 3}
    if others:
        required[int(others[0])] = 2
    for color in others[1:]:
        required[int(color)] = int(random.choice([2, 3]))
    return required


def _build_aligned_sprite(
    sprite_canvas: np.ndarray,
    anchors: list[dict[str, int | str]],
    anchor: dict[str, int | str],
    bg: int,
    used_body: set[int],
    *,
    required_count: int,
    require_wrap: bool,
) -> dict[str, object] | None:
    target = _anchor_target(int(anchor["r"]), int(anchor["c"]), sprite_canvas.shape)
    marker = int(anchor["color"])
    for _ in range(_SPRITE_BUILD_TRIES):
        area_lo = max(_WRAP_BODY_AREA_MIN, _BODY_AREA_RANGE[0]) if require_wrap else _BODY_AREA_RANGE[0]
        area = random.randint(area_lo, _BODY_AREA_RANGE[1])
        body = _body_color(bg, marker, used_body)
        sprite = _make_sprite(area, body, marker, bg, required_count=required_count)
        if sprite is None:
            continue
        if not _can_place_wrapped(
            sprite_canvas,
            sprite,
            target[0],
            target[1],
            bg,
            min_wrap_pixels=_MIN_WRAP_PIXELS if require_wrap else 0,
        ):
            continue
        cells, wrapped_pixels = _wrapped_sprite_cells(sprite_canvas.shape, sprite, target[0], target[1], bg)
        if not _cells_respect_anchors(cells, anchors, marker):
            continue
        _place_wrapped(sprite_canvas, sprite, target[0], target[1], bg)
        used_body.add(body)
        return {
            "sprite": sprite,
            "anchor": anchor,
            "target": (int(target[0]), int(target[1])),
            "wrapped_pixels": int(wrapped_pixels),
        }
    return None


def _place_aligned_sprites(
    sprite_canvas: np.ndarray,
    anchors: list[dict[str, int | str]],
    bg: int,
    wrap_color: int,
) -> list[dict[str, object]] | None:
    required_counts = _required_counts_by_color(anchors, wrap_color)
    ordered = sorted(anchors, key=lambda a: (int(a["color"]) != int(wrap_color), random.random()))
    used_body: set[int] = set()
    sprites = []
    for anchor in ordered:
        item = _build_aligned_sprite(
            sprite_canvas,
            anchors,
            anchor,
            bg,
            used_body,
            required_count=int(required_counts[int(anchor["color"])]),
            require_wrap=bool(int(anchor["color"]) == int(wrap_color)),
        )
        if item is None:
            return None
        sprites.append(item)

    if max((int(item["wrapped_pixels"]) for item in sprites), default=0) < _MIN_WRAP_PIXELS:
        return None
    if not any(int(item["sprite"]["marker_count"]) == 2 for item in sprites):
        return None
    if not any(int(item["sprite"]["marker_count"]) == 3 for item in sprites):
        return None
    return sprites


def _scatter_positions(
    canvas: np.ndarray,
    sprite: dict[str, object],
    target: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    tr, tc = map(int, target)
    candidates = []
    strong = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if r == tr or c == tc:
                continue
            if not _can_place(canvas, sprite, r, c, bg):
                continue
            candidates.append((int(r), int(c)))
            if abs(int(r) - tr) >= _STRONG_SHIFT_MIN and abs(int(c) - tc) >= _STRONG_SHIFT_MIN:
                strong.append((int(r), int(c)))
    return strong if strong else candidates


def _scatter_sprites(inp: np.ndarray, sprites: list[dict[str, object]], bg: int) -> bool:
    ordered = sorted(sprites, key=lambda item: int(item["sprite"]["nonbg_area"]), reverse=True)
    for item in ordered:
        positions = _scatter_positions(inp, item["sprite"], item["target"], bg)
        if not positions:
            return False
        align_r, align_c = random.choice(positions)
        _place(inp, item["sprite"], align_r, align_c, bg)
    return True


def _generated_grid_is_valid(inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    if not _bg_is_mode(inp, bg):
        return False
    metrics = variant_metrics(inp)
    if int(metrics["move_count"]) < 2:
        return False
    if int(metrics["dual_axis_moves"]) != int(metrics["move_count"]):
        return False
    if int(metrics["marker2_count"]) < 1 or int(metrics["marker3_count"]) < 1:
        return False
    if int(metrics["wrapped_pixels"]) < _MIN_WRAP_PIXELS:
        return False
    if not bool(metrics["all_markers_collinear"]):
        return False
    if not bool(metrics["diff_vs_single_axis_wrap"]):
        return False
    if not bool(metrics["diff_vs_dual_axis_clip"]):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt() -> np.ndarray | None:
    bg, out, _inp = _new_canvas_pair()
    chosen = _choose_anchors(out.shape, bg)
    if chosen is None:
        return None
    anchor_canvas, anchors, wrap_color = chosen

    out_sprites = np.full(out.shape, int(bg), dtype=int)
    sprites = _place_aligned_sprites(out_sprites, anchors, bg, wrap_color)
    if sprites is None:
        return None

    inp = anchor_canvas.copy()
    if not _scatter_sprites(inp, sprites, bg):
        return None

    out = out_sprites.copy()
    _put_anchors(out, anchors)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
