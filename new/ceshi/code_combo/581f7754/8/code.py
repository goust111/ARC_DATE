# Combo variant of 581f7754 (index 8)
# combo_spec: D2/1+D3/0
# Name: 标记位置决定轴+多标记精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The scene still contains isolated border anchors plus interior multicolor sprites.
#    - Each movable sprite now has 2 or 3 marker pixels of the minority color, all on one outer side of the body.
#      They are collinear on one row or one column, and the effective target point is the marker-line center.
#    - The D2 change and D3 change are fused into one object-level rule:
#      old task: anchor edge chose the move axis, and a single marker pixel gave the target coordinate.
#      combo task: marker side chooses the move axis, while the marker-line center gives the target coordinate.
#    - Small trace:
#      * old: a top-border red anchor implied horizontal motion; if the red marker sat at column 4 and the anchor was at column 8,
#        then dc = 4.
#      * combo: suppose the same red sprite has two red markers on its left edge at rows 3 and 5, so their center row is 4.
#        The left-edge marker placement means horizontal motion, even though the anchor is on the top border.
#        The anchor still contributes the target column 8, so dc = 8 - current_center_col.
#      * second combo example: a left-border blue anchor can pair with three blue markers on the sprite's top edge.
#        The top-edge marker line means vertical motion, and the center row of that line must match the anchor row.
#    - The visible difference from the original is therefore twofold in the same sprite:
#      the solver must read a whole marker run rather than one pixel, and it must read the run's side relative to the body
#      rather than the anchor edge to decide the axis.
#
# b) Complexity calibration
#    - Marker counts are limited to 2 or 3, and marker lines are limited to one straight outer-side run.
#      This keeps the search space small enough to infer from a few examples.
#    - The generator forces both anchor edge families to appear in a sample and mixes "same-edge" vs "orthogonal" marker placements.
#      Same-edge cases flip the original axis rule; orthogonal cases preserve it. This prevents regression to the original task.
#    - Multiple sprites still reuse anchor colors, so the task stays about structured matching rather than one-off color lookup.
#
# c) Hidden assumptions in the original main()
#    - Original anchor detection still works because anchors remain isolated border singletons.
#    - Original minority-color detection still works because body area is at least 4 while marker count is at most 3.
#    - Original `marker_pixel(...)` assumed one decisive marker cell. That assumption is broken and replaced by
#      `marker_alignment_info(...)`, which aggregates all marker cells into one center point.
#    - Original axis logic depended only on anchor coordinates. That assumption is broken and replaced by body-relative side analysis.
#    - Original singleton distractors would violate the new side-based rule because a lone pixel has no body-relative side.
#      The combo generator removes that branch entirely.
#
# d) Placement legality
#    - Output-first generation is preserved: build the solved output, then scatter each sprite along its move axis to obtain the input.
#    - Marker lines are generated only from exposed cells on one body side, so every marker is 4-adjacent to the body and the sprite
#      remains one 8-connected component.
#    - The solved placement uses the marker-line center as the geometric anchor. Pair markers are symmetric about the center and
#      triplets are consecutive, so the center is always an integer grid location.
#    - Border-touching is allowed only when the marker side equals the anchor edge, which is exactly the case where the aligned marker
#      line can sit on that border without pushing the body outside the grid.
#
# e) Visual observability
#    - D2 is visible because anchor location alone is insufficient: some top/bottom anchors produce horizontal motion and others
#      produce vertical motion depending on the sprite's marker side.
#    - D3 is visible because the moved feature is a short marker run, not one pixel; pair markers leave a gap-centered midpoint and
#      triplets expose a literal center marker.
#    - No invisible properties such as color ordering or numeric color ids determine motion.
#
# f) Visual cleanliness
#    - Sprites and anchors keep an 8-neighborhood gap, so marker runs do not merge with neighbors.
#    - Each sprite still uses only one body color plus one marker color, avoiding clutter despite the extra marker cells.
#    - Marker runs are kept on the body's exterior boundary rather than inside cavities, which makes the side-to-axis cue legible.
#
# g) Change inferability
#    - Colors only specify which anchor matches which sprite. They do not encode axis, count, or order.
#    - The repeated use of 2-marker and 3-marker runs, together with both preserved-axis and flipped-axis cases, gives enough repeated
#      structure for a human to infer the combo rule from 2-4 examples.
#
# h) Post-change solver assumptions audited
#    - main() now assumes every movable sprite has at least one non-marker body pixel and 2-3 marker pixels of the minority color.
#      generate_input() guarantees this.
#    - main() assumes those marker pixels lie on one row or one column and occupy one outer side of the body. generate_input()
#      constructs markers only from explicit side options.
#    - main() assumes the marker-line center is integral. generate_input() only emits symmetric pairs or 3-cell runs, so this holds.
#    - main() ignores monochrome interior components. generate_input() produces none besides the isolated border anchors.
#
# i) Non-degeneracy target
#    - The important failure mode would be generating only cases where the combo solver behaves exactly like the original task.
#      To block that, the generator explicitly includes same-edge cases where the new side-based axis disagrees with the original
#      anchor-based axis. Post-generation evaluation compares the combo solver against the original solver on generated inputs.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-marker sprites so their marker-line centers align with matching isolated border anchors.

    Input structure:
    - The background is the most frequent color.
    - Interior sprites are 8-connected two-color components with one larger body color and 2-3 minority-color marker pixels.
    - The marker pixels of each sprite lie on one shared row or one shared column along a single outer side of the body.
    - Border anchors are isolated non-background singletons, and their colors match sprite marker colors.

    Rule:
    - Pair each sprite with the anchor that shares its marker color.
    - Use the marker-line center as the sprite's effective alignment point.
    - If the marker line sits on the top or bottom side of the body, slide the sprite vertically so that center row matches the anchor row.
    - If the marker line sits on the left or right side of the body, slide the sprite horizontally so that center column matches the anchor column.
    - Clear source cells first; translated pixels that leave the grid are clipped, and existing non-background destination cells are preserved.

    Key insight: The anchor still tells you which color to match and which row or column to hit, but the move axis comes from the marker line's side on the sprite, and the aligned location is the center of that whole line rather than any single marker pixel.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def median_int(values: list[int]) -> int:
        values = sorted(int(v) for v in values)
        n = len(values)
        if n == 0:
            raise ValueError("empty median")
        return int(values[n // 2]) if n % 2 == 1 else int((values[n // 2 - 1] + values[n // 2]) // 2)

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

    def marker_alignment_info(coords: np.ndarray, marker: int) -> tuple[tuple[int, int], str] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        body = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker)]
        if len(pts) < 2 or not body:
            return None

        body_set = set(body)
        rows = sorted({r for r, _c in pts})
        cols = sorted({c for _r, c in pts})
        r0 = min(r for r, _c in body)
        c0 = min(c for _r, c in body)
        r1 = max(r for r, _c in body)
        c1 = max(c for _r, c in body)

        if len(rows) == 1:
            mr = int(rows[0])
            mc = int(median_int([c for _r, c in pts]))
            if mr < r0:
                return (mr, mc), "v"
            if mr > r1:
                return (mr, mc), "v"
            attached = [(rr, cc) for r, c in pts for rr, cc in neighbors4(r, c) if (int(rr), int(cc)) in body_set]
            if attached:
                if all(int(rr) > mr for rr, _cc in attached):
                    return (mr, mc), "v"
                if all(int(rr) < mr for rr, _cc in attached):
                    return (mr, mc), "v"
            return None

        if len(cols) == 1:
            mr = int(median_int([r for r, _c in pts]))
            mc = int(cols[0])
            if mc < c0:
                return (mr, mc), "h"
            if mc > c1:
                return (mr, mc), "h"
            attached = [(rr, cc) for r, c in pts for rr, cc in neighbors4(r, c) if (int(rr), int(cc)) in body_set]
            if attached:
                if all(int(cc) > mc for _rr, cc in attached):
                    return (mr, mc), "h"
                if all(int(cc) < mc for _rr, cc in attached):
                    return (mr, mc), "h"
            return None

        return None

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None or int(marker_col) not in anchors_by_color:
            continue

        anchor = anchors_by_color[int(marker_col)]
        info = marker_alignment_info(coords, int(marker_col))
        if info is None:
            continue

        (mr, mc), axis = info
        if (mr, mc) in anchor_cells and int(coords.shape[0]) == 1:
            continue

        ar, ac = anchor
        dr = 0
        dc = 0
        if axis == "h":
            dc = int(ac - mc)
        else:
            dr = int(ar - mr)
        moves.append((coords, int(dr), int(dc)))

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


_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 700
_SPRITE_RETRY_LIMIT = 80

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

_PAIR_MARKER_PROB = 0.45

_SIDE_TO_AXIS = {"top": "v", "bottom": "v", "left": "h", "right": "h"}
_TB_SIDES = ("top", "bottom")
_LR_SIDES = ("left", "right")

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


def _marker_cells_by_side(body):
    r0, c0, r1, c1 = _bbox(body)
    body = set(body)
    pools = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if (rr, cc) in body:
                continue
            if rr == r0 - 1 and c0 <= cc <= c1:
                pools["top"].add((rr, cc))
            elif rr == r1 + 1 and c0 <= cc <= c1:
                pools["bottom"].add((rr, cc))
            elif cc == c0 - 1 and r0 <= rr <= r1:
                pools["left"].add((rr, cc))
            elif cc == c1 + 1 and r0 <= rr <= r1:
                pools["right"].add((rr, cc))
    return {k: sorted(v) for k, v in pools.items()}


def _edge_family(edge):
    return "tb" if edge in _TB_SIDES else "lr"


def _orthogonal_sides(anchor):
    return _LR_SIDES if _edge_family(anchor["edge"]) == "tb" else _TB_SIDES


def _marker_side_options(anchor, mode):
    if mode == "same_edge":
        return [anchor["edge"]]
    return list(_orthogonal_sides(anchor))


def _marker_line_options_for_side(body, side):
    cells = _marker_cells_by_side(body).get(side, [])
    if not cells:
        return []

    options = []
    seen = set()
    if side in _TB_SIDES:
        fixed = int(cells[0][0])
        vals = sorted({int(c) for _r, c in cells})
        valset = set(vals)
        for center in vals:
            if {center - 1, center, center + 1}.issubset(valset):
                marker_cells = ((fixed, center - 1), (fixed, center), (fixed, center + 1))
                if marker_cells not in seen:
                    seen.add(marker_cells)
                    options.append({"cells": marker_cells, "align": (fixed, center), "count": 3})
            if {center - 1, center + 1}.issubset(valset):
                marker_cells = ((fixed, center - 1), (fixed, center + 1))
                if marker_cells not in seen:
                    seen.add(marker_cells)
                    options.append({"cells": marker_cells, "align": (fixed, center), "count": 2})
    else:
        fixed = int(cells[0][1])
        vals = sorted({int(r) for r, _c in cells})
        valset = set(vals)
        for center in vals:
            if {center - 1, center, center + 1}.issubset(valset):
                marker_cells = ((center - 1, fixed), (center, fixed), (center + 1, fixed))
                if marker_cells not in seen:
                    seen.add(marker_cells)
                    options.append({"cells": marker_cells, "align": (center, fixed), "count": 3})
            if {center - 1, center + 1}.issubset(valset):
                marker_cells = ((center - 1, fixed), (center + 1, fixed))
                if marker_cells not in seen:
                    seen.add(marker_cells)
                    options.append({"cells": marker_cells, "align": (center, fixed), "count": 2})
    return options


def _choose_marker_option(options, preferred_count):
    by_count = {
        2: [opt for opt in options if int(opt["count"]) == 2],
        3: [opt for opt in options if int(opt["count"]) == 3],
    }
    order = [int(preferred_count), 5 - int(preferred_count)]
    for count in order:
        if by_count[count]:
            return random.choice(by_count[count])
    return None


def _make_sprite(area, body_color, marker_color, bg, marker_side, preferred_count):
    for _ in range(_SPRITE_RETRY_LIMIT):
        body = _make_body(area)
        option = _choose_marker_option(_marker_line_options_for_side(body, marker_side), preferred_count)
        if option is None:
            continue
        markers = set(option["cells"])
        all_cells = set(body) | markers
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        for r, c in markers:
            patch[r - r0, c - c0] = int(marker_color)
        ar, ac = option["align"]
        return {
            "patch": patch,
            "align_offset": (int(ar - r0), int(ac - c0)),
            "marker_side": str(marker_side),
            "move_axis": _SIDE_TO_AXIS[str(marker_side)],
            "marker_count": int(option["count"]),
        }
    return None


def _top_left(sprite, align_r, align_c):
    off_r, off_c = sprite["align_offset"]
    return int(align_r) - int(off_r), int(align_c) - int(off_c)


def _visible_sprite_cells(sprite, align_r, align_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, align_r, align_c)
    h, w = patch.shape
    cells = []
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v == int(bg):
                continue
            cells.append((top + r, left + c, v))
    return cells, top, left, h, w


def _can_place(canvas, sprite, align_r, align_c, bg, *, allow_border):
    H, W = canvas.shape
    cells, top, left, h, w = _visible_sprite_cells(sprite, align_r, align_c, bg)
    margin = 0 if allow_border else _INNER_BORDER_GAP
    if top < margin or left < margin:
        return False
    if top + h > H - margin or left + w > W - margin:
        return False

    occupied = {(r, c) for r, c, _v in cells}
    for r, c, _v in cells:
        if int(canvas[r, c]) != int(bg):
            return False
        for rr in range(max(0, r - 1), min(H, r + 2)):
            for cc in range(max(0, c - 1), min(W, c + 2)):
                if (rr, cc) in occupied:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return False
    return True


def _place(canvas, sprite, align_r, align_c, bg):
    top, left = _top_left(sprite, align_r, align_c)
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
        out.append(("top", 0, c))
        out.append(("bottom", H - 1, c))
    for r in rows:
        out.append(("left", r, 0))
        out.append(("right", r, W - 1))
    random.shuffle(out)
    return [(edge, r, c) for edge, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    allow_border = sprite["marker_side"] == anchor["edge"]
    if sprite["move_axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg, allow_border=allow_border)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg, allow_border=allow_border)
    ]


def _scatter_positions(canvas, sprite, align_pos, bg):
    H, W = canvas.shape
    ar, ac = align_pos
    if sprite["move_axis"] == "h":
        return [
            (ar, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != ac and _can_place(canvas, sprite, ar, c, bg, allow_border=False)
        ]
    return [
        (r, ac)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != ar and _can_place(canvas, sprite, r, ac, bg, allow_border=False)
    ]


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
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

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
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
    families = ["tb", "lr"]
    while len(families) < len(anchor_colors):
        families.append(random.choice(["tb", "lr"]))
    random.shuffle(families)

    anchors = []
    for color, family in zip(anchor_colors, families):
        candidates = [cand for cand in _anchor_candidates(out, bg) if _edge_family(cand[0]) == family]
        if not candidates:
            return None
        edge, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "edge": str(edge)}
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


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []

    modes = ["same_edge", "orthogonal"]
    while len(modes) < len(marker_colors):
        modes.append(random.choice(["same_edge", "orthogonal"]))
    random.shuffle(modes)

    preferred_counts = [2, 3]
    while len(preferred_counts) < len(marker_colors):
        preferred_counts.append(2 if random.random() < _PAIR_MARKER_PROB else 3)
    random.shuffle(preferred_counts)

    for marker, mode, preferred_count in zip(marker_colors, modes, preferred_counts):
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)

        side_options = _marker_side_options(anchor, mode)
        random.shuffle(side_options)
        placed = False
        for marker_side in side_options:
            sprite = _make_sprite(
                random.randint(*_BODY_AREA_RANGE),
                body,
                marker,
                bg,
                marker_side,
                preferred_count,
            )
            if sprite is None:
                continue
            positions = _aligned_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            align_pos = random.choice(positions)
            _place(out, sprite, align_pos[0], align_pos[1], bg)
            sprites.append((sprite, align_pos))
            placed = True
            break
        if not placed:
            return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, align_pos in sprites:
        positions = _scatter_positions(inp, sprite, align_pos, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
