# Combo variant of 581f7754 (index 6)
# combo_spec: D1/0+D4/3
# Name: 形状匹配+线段锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) Concreteization + behavior trace:
#    - Border anchors are no longer singleton dots. They are isolated 2- or 3-cell straight segments on one
#      edge: horizontal on the top/bottom edge, vertical on the left/right edge.
#    - Each interior sprite still has a large body color and a rarer marker color, but the marker is now a
#      2- or 3-cell straight segment too. Example: if the border shows a top-edge horizontal domino and a
#      left-edge vertical triomino, then a sprite whose minority region is a horizontal domino slides only
#      horizontally until that marker segment's discrete midpoint column matches the top-edge domino midpoint.
#      A sprite whose minority region is the vertical triomino slides only vertically until midpoint rows
#      match. The difference from the original task is twofold at once: pairing is by geometry instead of
#      color, and the target reference is a line midpoint rather than one anchor pixel.
# b) Complexity calibration:
#    - The shape space is deliberately small: horizontal/vertical segments of length 2 or 3. That keeps the
#      task human-readable while still making color-based matching fail.
#    - Each sample uses 2-3 distinct anchor shapes and 3-5 sprites, with shapes repeated across sprites. This
#      gives repeated support for the shape rule without flooding the board with too many cases.
# c) Implicit-assumption audit of the original solver:
#    - `anchors_by_color[...] = (r,c)` is invalid: anchors are multi-cell and colors are intentionally shared.
#    - `component_marker_color(...)` is still useful only to isolate the minority region inside a sprite; it
#      no longer identifies the destination.
#    - `marker_pixel(...)` is invalid because the marker is not a singleton. The alignment reference must be
#      the discrete midpoint of the marker segment.
#    - Allowing duplicate anchor shapes would make shape pairing ambiguous, so the generator guarantees one
#      border anchor per shape signature.
# d) Placement legality:
#    - Anchors are placed with a full 8-neighborhood moat of background, never touching corners.
#    - Sprites are placed fully inside the interior with at least one background cell of padding, and their
#      marker segment is 4-neighbor attached to the body so the whole sprite remains one 8-connected object.
# e) Visual observability:
#    - All anchors share one color and all markers share another color. A human can immediately see that
#      color equality cannot explain the motion; the only stable clue is segment shape and axis.
#    - Length-2 midpoint rounding is visible because the aligned marker ends on the segment's trailing cell,
#      matching the dash's own discrete center.
# f) Visual cleanliness:
#    - There are no extra distractor singletons. Objects stay sparse and separated, so each border dash and
#      each internal marker segment remains easy to parse by eye.
# g) Change inferability:
#    - No hidden numeric color ordering is used. Colors act only as role labels (bg / anchor / marker / body),
#      while the transform is entirely recoverable from visible shape repetition and border position.


_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (16, 22)
_GRID_WIDTH_RANGE = (16, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 2)
_SPRITE_COUNT_RANGE = (3, 4)
_BODY_AREA_RANGE = (4, 9)
_GEN_RETRY_LIMIT = 800

_BODY_HEIGHT_RANGE = (2, 4)
_BODY_WIDTH_RANGE = (3, 5)
_FRAME_HEIGHT_RANGE = (3, 4)
_FRAME_WIDTH_RANGE = (3, 5)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.28
_WIDE_BODY_THRESHOLD = 0.55
_SYMMETRIC_BODY_THRESHOLD = 0.82
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.45
_FLIP_PROB = 0.5
_MAX_SPRITE_SPAN_H = _BODY_HEIGHT_RANGE[1] + 3
_MAX_SPRITE_SPAN_W = _BODY_WIDTH_RANGE[1] + 3

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]

_SEGMENT_SHAPES = (
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites until their marker segments align with matching border dash anchors.

    Input structure:
    - bg is the most common color.
    - Border anchors are isolated 2-3 cell straight segments on exactly one edge.
    - Interior sprites are 8-connected multi-color components with one larger body color and one smaller
      marker color; the marker cells form a 2-3 cell straight segment.

    Rule:
    - Pair each sprite with the border anchor whose normalized segment shape matches the sprite's marker shape.
    - A top/bottom anchor moves the sprite horizontally until the marker midpoint column equals the anchor
      midpoint column.
    - A left/right anchor moves the sprite vertically until the marker midpoint row equals the anchor midpoint row.

    Key insight:
    - The minority color still reveals which cells belong to the marker, but destination matching is geometric:
      recover the marker segment shape, find the same border dash, and align their discrete centers along the
      axis implied by the anchor's edge.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _norm_sig(cells) -> tuple[tuple[int, int], ...]:
        r0 = min(int(r) for r, _c in cells)
        c0 = min(int(c) for _r, c in cells)
        return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))

    def _is_segment(cells) -> tuple[str, int, int, int] | None:
        pts = {(int(r), int(c)) for r, c in cells}
        if len(pts) not in (2, 3):
            return None
        rows = sorted({r for r, _c in pts})
        cols = sorted({c for _r, c in pts})
        if len(rows) == 1 and cols == list(range(cols[0], cols[0] + len(cols))):
            return ("h", int(rows[0]), int(cols[0]), int(len(cols)))
        if len(cols) == 1 and rows == list(range(rows[0], rows[0] + len(rows))):
            return ("v", int(rows[0]), int(cols[0]), int(len(rows)))
        return None

    def _segment_midpoint(cells) -> tuple[int, int]:
        seg = _is_segment(cells)
        assert seg is not None
        axis, r0, c0, length = seg
        if axis == "h":
            return int(r0), int(c0 + length // 2)
        return int(r0 + length // 2), int(c0)

    def _isolated_component(cells) -> bool:
        S = {(int(r), int(c)) for r, c in cells}
        for r, c in S:
            for rr in range(int(r) - _ISOLATION_RADIUS, int(r) + _ISOLATION_RADIUS + 1):
                for cc in range(int(c) - _ISOLATION_RADIUS, int(c) + _ISOLATION_RADIUS + 1):
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if (int(rr), int(cc)) in S:
                        continue
                    if int(g[int(rr), int(cc)]) != int(bg):
                        return False
        return True

    anchors_by_shape: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    anchor_cells: set[tuple[int, int]] = set()
    mono_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in mono_comps:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        seg = _is_segment(coords)
        if seg is None:
            continue
        axis, r0, c0, length = seg
        if axis == "h":
            if r0 not in (0, H - 1):
                continue
        else:
            if c0 not in (0, W - 1):
                continue
        if any((int(r), int(c)) in {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)} for r, c in coords):
            continue
        if not _isolated_component(coords):
            continue
        sig = _norm_sig(coords)
        if sig in anchors_by_shape:
            continue
        target_r, target_c = _segment_midpoint(coords)
        anchors_by_shape[sig] = {
            "axis": axis,
            "target_r": int(target_r),
            "target_c": int(target_c),
        }
        anchor_cells |= {(int(r), int(c)) for r, c in coords}

    assert anchors_by_shape

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        coord_list = [(int(r), int(c)) for r, c in coords]
        if any((int(r), int(c)) in anchor_cells for r, c in coord_list):
            continue

        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            continue
        min_n = min(int(n) for n in cnt.values())
        if sum(int(n) == int(min_n) for n in cnt.values()) != 1:
            continue
        marker_color = next(int(color) for color, n in cnt.items() if int(n) == int(min_n))
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker_color)]
        if _is_segment(marker_cells) is None:
            continue

        comp_set = {(int(r), int(c)) for r, c in coords}
        if not any(
            (int(r + dr), int(c + dc)) in comp_set and int(g[int(r + dr), int(c + dc)]) != int(marker_color)
            for r, c in marker_cells
            for dr, dc in _NEIGHBORS4
        ):
            continue

        sig = _norm_sig(marker_cells)
        if sig not in anchors_by_shape:
            continue

        anchor = anchors_by_shape[sig]
        mid_r, mid_c = _segment_midpoint(marker_cells)
        dr = 0
        dc = 0
        if anchor["axis"] == "h":
            dc = int(anchor["target_c"]) - int(mid_c)
        else:
            dr = int(anchor["target_r"]) - int(mid_r)
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


def _norm(cells):
    r0 = min(int(r) for r, _c in cells)
    c0 = min(int(c) for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _sig(cells):
    return tuple(sorted(_norm(cells)))


def _bbox(cells):
    return (
        min(int(r) for r, _c in cells),
        min(int(c) for _r, c in cells),
        max(int(r) for r, _c in cells),
        max(int(c) for _r, c in cells),
    )


def _is_segment(cells) -> tuple[str, int]:
    cells = _norm(cells)
    r0, c0, r1, c1 = _bbox(cells)
    if r0 == r1 and c1 - c0 + 1 == len(cells):
        return ("h", int(len(cells)))
    if c0 == c1 and r1 - r0 + 1 == len(cells):
        return ("v", int(len(cells)))
    raise ValueError("shape is not a straight segment")


def _segment_midpoint(cells):
    cells = _norm(cells)
    axis, length = _is_segment(cells)
    if axis == "h":
        return 0, int(length // 2)
    return int(length // 2), 0


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(int(c) for _r, c in cells)
        cells = {(int(r), int(mc - c)) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(int(r) for r, _c in cells)
        cells = {(int(mr - r), int(c)) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and (int(rr), int(cc)) not in cells:
                frontier.append((int(rr), int(cc)))
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
    cells = {(int(r), 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((int(r), 0))
        cells.add((int(r), 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(int(c), int(r)) for r, c in cells}
    return _flip(cells)


def _wide_body(area):
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, int(c)) for c in range(width)}
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
        (int(r), int(c))
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    gap = len(cells) - area
    if gap:
        removable = [
            (int(r), int(c))
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


def _translate(cells, dr, dc):
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _shape_placement_candidates(body, marker_shape):
    body = set(body)
    r0, c0, r1, c1 = _bbox(body)
    shape = set(marker_shape)
    shape_h = max(int(r) for r, _c in shape) + 1
    shape_w = max(int(c) for _r, c in shape) + 1
    scored = []
    for top in range(int(r0 - shape_h - 1), int(r1 + 3)):
        for left in range(int(c0 - shape_w - 1), int(c1 + 3)):
            placed = _translate(shape, top, left)
            if placed & body:
                continue
            if not any(
                (int(r + dr), int(c + dc)) in body
                for r, c in placed
                for dr, dc in _NEIGHBORS4
            ):
                continue
            all_cells = body | placed
            rr0, cc0, rr1, cc1 = _bbox(all_cells)
            if rr1 - rr0 + 1 > _MAX_SPRITE_SPAN_H:
                continue
            if cc1 - cc0 + 1 > _MAX_SPRITE_SPAN_W:
                continue
            mid_r, mid_c = _segment_midpoint(placed)
            off_r = int(mid_r - rr0)
            off_c = int(mid_c - cc0)
            score = ((off_r > 2) + (off_c > 2), off_r + off_c, rr1 - rr0, cc1 - cc0)
            scored.append((score, placed))
    scored.sort(key=lambda x: x[0])
    return [placed for _score, placed in scored[:12]]


def _make_sprite(area, body_color, marker_color, marker_shape, bg):
    body = _make_body(area)
    placements = _shape_placement_candidates(body, marker_shape)
    if not placements:
        return None
    marker = random.choice(placements[: min(4, len(placements))])
    all_cells = set(body) | set(marker)
    r0, c0, r1, c1 = _bbox(all_cells)
    mid_r, mid_c = _segment_midpoint(marker)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    for r, c in marker:
        patch[int(r - r0), int(c - c0)] = int(marker_color)
    return {
        "patch": patch,
        "mid_offset": (int(mid_r - r0), int(mid_c - c0)),
        "marker_sig": _sig(marker_shape),
    }


def _top_left(sprite, ref_r, ref_c):
    off_r, off_c = sprite["mid_offset"]
    return int(ref_r) - int(off_r), int(ref_c) - int(off_c)


def _can_place(canvas, sprite, ref_r, ref_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, ref_r, ref_c)
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


def _place(canvas, sprite, ref_r, ref_c, bg):
    top, left = _top_left(sprite, ref_r, ref_c)
    paste(canvas, sprite["patch"], int(top), int(left), transparent=int(bg))


def _anchor_cells(shape, side, H, W, offset):
    shape = _norm(shape)
    _r0, _c0, r1, c1 = _bbox(shape)
    h = int(r1 + 1)
    w = int(c1 + 1)
    if side == "top":
        top, left = 0, int(offset)
    elif side == "bottom":
        top, left = int(H - h), int(offset)
    elif side == "left":
        top, left = int(offset), 0
    else:
        top, left = int(offset), int(W - w)
    return _translate(shape, top, left)


def _pick_anchor_shapes(n_anchors):
    horizontal = [shape for shape in _SEGMENT_SHAPES if len({int(r) for r, _c in shape}) == 1]
    vertical = [shape for shape in _SEGMENT_SHAPES if len({int(c) for _r, c in shape}) == 1]
    assert horizontal and vertical

    chosen = [random.choice(horizontal), random.choice(vertical)]
    if n_anchors > 2:
        remaining = [shape for shape in _SEGMENT_SHAPES if shape not in chosen]
        chosen.extend(random.sample(remaining, n_anchors - 2))
    random.shuffle(chosen)
    return chosen


def _center_order(positions, shape):
    H, W = shape
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0
    return sorted(
        positions,
        key=lambda pos: (
            abs(float(pos[0]) - center_r) + abs(float(pos[1]) - center_c),
            int(pos[0]),
            int(pos[1]),
        ),
    )


def _scatter_order(positions, anchor, ref_pos, shape):
    H, W = shape
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0
    if anchor["axis"] == "h":
        distance = lambda pos: abs(int(pos[1]) - int(ref_pos[1]))
    else:
        distance = lambda pos: abs(int(pos[0]) - int(ref_pos[0]))
    return sorted(
        positions,
        key=lambda pos: (
            -distance(pos),
            abs(float(pos[0]) - center_r) + abs(float(pos[1]) - center_c),
            int(pos[0]),
            int(pos[1]),
        ),
    )


def _isolated_bg_cells(canvas, cells, bg):
    H, W = canvas.shape
    S = {(int(r), int(c)) for r, c in cells}
    for r, c in S:
        for rr in range(int(r) - _ISOLATION_RADIUS, int(r) + _ISOLATION_RADIUS + 1):
            for cc in range(int(c) - _ISOLATION_RADIUS, int(c) + _ISOLATION_RADIUS + 1):
                if 0 <= rr < H and 0 <= cc < W and (int(rr), int(cc)) not in S and int(canvas[int(rr), int(cc)]) != int(bg):
                    return False
    return True


def _anchor_candidates(canvas, shape, bg):
    H, W = canvas.shape
    axis, _length = _is_segment(shape)
    out = []
    if axis == "h":
        width = max(int(c) for _r, c in shape) + 1
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - width - _ANCHOR_EDGE_MARGIN + 1))
        if not cols:
            cols = list(range(_INNER_BORDER_GAP, W - width - _INNER_BORDER_GAP + 1))
        for left in cols:
            out.append(("top", _anchor_cells(shape, "top", H, W, left)))
            out.append(("bottom", _anchor_cells(shape, "bottom", H, W, left)))
    else:
        height = max(int(r) for r, _c in shape) + 1
        rows = list(range(_ANCHOR_EDGE_MARGIN, H - height - _ANCHOR_EDGE_MARGIN + 1))
        if not rows:
            rows = list(range(_INNER_BORDER_GAP, H - height - _INNER_BORDER_GAP + 1))
        for top in rows:
            out.append(("left", _anchor_cells(shape, "left", H, W, top)))
            out.append(("right", _anchor_cells(shape, "right", H, W, top)))
    random.shuffle(out)
    good = []
    corners = {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}
    for side, cells in out:
        if any((int(r), int(c)) in corners for r, c in cells):
            continue
        if _isolated_bg_cells(canvas, cells, bg):
            good.append((side, cells))
    return good


def _put_anchor(canvas, cells, color):
    for r, c in cells:
        canvas[int(r), int(c)] = int(color)


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (ref_r, anchor["target_c"])
            for ref_r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, ref_r, anchor["target_c"], bg)
        ]
    return [
        (anchor["target_r"], ref_c)
        for ref_c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["target_r"], ref_c, bg)
    ]


def _scatter_positions(canvas, sprite, anchor, ref_pos, bg):
    H, W = canvas.shape
    ref_r, ref_c = ref_pos
    if anchor["axis"] == "h":
        return [
            (ref_r, other_c)
            for other_c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if other_c != ref_c and _can_place(canvas, sprite, ref_r, other_c, bg)
        ]
    return [
        (other_r, ref_c)
        for other_r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if other_r != ref_r and _can_place(canvas, sprite, other_r, ref_c, bg)
    ]


def _body_color(bg, anchor_color, marker_color, used):
    blocked = {int(bg), int(anchor_color), int(marker_color)}
    pool = [c for c in _COLORS if c not in blocked and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in blocked]
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


def _pick_role_colors(bg):
    anchor_color = random.choice([c for c in _COLORS if c != int(bg)])
    marker_color = random.choice([c for c in _COLORS if c not in {int(bg), int(anchor_color)}])
    return int(anchor_color), int(marker_color)


def _place_anchors(out, inp, bg, anchor_color):
    n_anchors = random.randint(*_ANCHOR_COUNT_RANGE)
    chosen_shapes = _pick_anchor_shapes(n_anchors)
    anchors = []
    for shape in chosen_shapes:
        candidates = _anchor_candidates(out, shape, bg)
        if not candidates:
            return None
        side, cells = random.choice(candidates)
        axis, length = _is_segment(shape)
        cell_sig = _sig(shape)
        if axis == "h":
            target_r = next(iter({int(r) for r, _c in cells}))
            target_c = min(int(c) for _r, c in cells) + length // 2
        else:
            target_r = min(int(r) for r, _c in cells) + length // 2
            target_c = next(iter({int(c) for _r, c in cells}))
        anchor = {
            "shape": _norm(shape),
            "sig": cell_sig,
            "axis": axis,
            "side": side,
            "target_r": int(target_r),
            "target_c": int(target_c),
            "cells": {(int(r), int(c)) for r, c in cells},
            "color": int(anchor_color),
        }
        anchors.append(anchor)
        _put_anchor(out, cells, anchor_color)
        _put_anchor(inp, cells, anchor_color)
    return anchors


def _marker_shape_sequence(anchors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_shapes = [a["shape"] for a in anchors]
    while len(marker_shapes) < n_sprites:
        marker_shapes.append(random.choice([a["shape"] for a in anchors]))
    random.shuffle(marker_shapes)
    return marker_shapes


def _search_aligned_layout(canvas, pending, bg):
    if not pending:
        return []

    options = []
    for idx, item in enumerate(pending):
        positions = _aligned_positions(canvas, item["sprite"], item["anchor"], bg)
        if not positions:
            return None
        options.append((len(positions), idx, _center_order(positions, canvas.shape)))

    _count, idx, positions = min(options, key=lambda item: item[0])
    item = pending[idx]
    rest = pending[:idx] + pending[idx + 1 :]
    for ref_pos in positions:
        next_canvas = canvas.copy()
        _place(next_canvas, item["sprite"], ref_pos[0], ref_pos[1], bg)
        placed = _search_aligned_layout(next_canvas, rest, bg)
        if placed is not None:
            return [(item["sprite"], item["anchor"], ref_pos)] + placed
    return None


def _place_aligned_sprites(out, anchors, marker_shapes, marker_color, anchor_color, bg):
    by_sig = {a["sig"]: a for a in anchors}
    used_body = set()
    pending = []
    for marker_shape in marker_shapes:
        sig = _sig(marker_shape)
        anchor = by_sig[sig]
        body = _body_color(bg, anchor_color, marker_color, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker_color, marker_shape, bg)
        if sprite is None:
            return None
        pending.append({"sprite": sprite, "anchor": anchor})

    placed = _search_aligned_layout(out, pending, bg)
    if placed is None:
        return None
    for sprite, _anchor, ref_pos in placed:
        _place(out, sprite, ref_pos[0], ref_pos[1], bg)
    return placed


def _search_scatter_layout(canvas, pending, bg):
    if not pending:
        return []

    options = []
    for idx, item in enumerate(pending):
        positions = _scatter_positions(canvas, item["sprite"], item["anchor"], item["ref_pos"], bg)
        if not positions:
            return None
        ordered = _scatter_order(positions, item["anchor"], item["ref_pos"], canvas.shape)
        options.append((len(ordered), idx, ordered))

    _count, idx, positions = min(options, key=lambda item: item[0])
    item = pending[idx]
    rest = pending[:idx] + pending[idx + 1 :]
    for in_pos in positions:
        next_canvas = canvas.copy()
        _place(next_canvas, item["sprite"], in_pos[0], in_pos[1], bg)
        placed = _search_scatter_layout(next_canvas, rest, bg)
        if placed is not None:
            return [(item["sprite"], in_pos)] + placed
    return None


def _scatter_sprites(inp, sprites, bg):
    pending = [
        {"sprite": sprite, "anchor": anchor, "ref_pos": ref_pos}
        for sprite, anchor, ref_pos in sprites
    ]
    placed = _search_scatter_layout(inp, pending, bg)
    if placed is None:
        return False
    for sprite, in_pos in placed:
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    anchor_color, marker_color = _pick_role_colors(bg)
    anchors = _place_anchors(out, inp, bg, anchor_color)
    if anchors is None:
        return None

    marker_shapes = _marker_shape_sequence(anchors)
    sprites = _place_aligned_sprites(out, anchors, marker_shapes, marker_color, anchor_color, bg)
    if sprites is None:
        return None
    if not _scatter_sprites(inp, sprites, bg):
        return None
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
