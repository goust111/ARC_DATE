# Variant of 581f7754
# Dimension: D1 — 匹配条件：标记色与锚点色匹配 (match_cond)
# Change: 精灵内部的少数色（marker）与边界锚点颜色相同即配对：s.marker.color == d.color... → 形状匹配替代颜色匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D1 variant
# a) Concreteization + behavior trace:
#    - Border anchors are no longer singleton pixels. Each anchor is an isolated 2-3 cell monochrome shape
#      placed on exactly one edge. Interior sprites still have a dominant body color plus a minority marker
#      region, but the marker is now a 2-3 cell connected subshape rather than one pixel.
#    - Matching changes at one specific point in the pipeline: original code chose an anchor by
#      `marker_color -> anchor_color`. Here the marker color only identifies the marker region inside the
#      sprite; the actual pairing key is the normalized geometry of that region.
#    - Small trace: originally a sprite with minority color 4 moved to the border anchor colored 4. Now
#      a sprite whose minority region is an L triomino moves to the border anchor that is also an L triomino,
#      even when the anchor color is shared by all anchors and differs from the marker color.
# b) Complexity calibration:
#    - A tiny shape library (domino, I3, L variants) keeps the matching space human-readable while still
#      making color-based heuristics fail. Each grid uses 2-3 distinct anchor shapes, and multiple sprites
#      may reuse the same shape so the rule has repeated support within one example.
# c) Implicit-assumption audit of the original solver:
#    - `anchors_by_color[...] = (r,c)` assumed every anchor was a singleton pixel and colors were unique.
#      This is invalid now, so anchors must be collected as border components indexed by shape signature.
#    - `component_marker_color(...)` remains usable only to locate the minority region inside a sprite; it
#      can no longer determine the destination.
#    - `marker_pixel(...)` assumed a one-cell target reference. This variant replaces it with marker bbox
#      alignment: for top/bottom anchors we match the marker shape's column footprint, for left/right
#      anchors we match its row footprint.
# d) Placement legality:
#    - The generator keeps anchors isolated with an 8-neighborhood bg moat and keeps sprites fully inside
#      the interior with a one-cell gap. Marker subshapes are attached 4-neighbor to the body so each sprite
#      stays a single 8-connected component and the solver can reliably extract it.
# e) Visual observability:
#    - All anchors share one color and all marker regions share another color, so color visibly cannot be
#      the matching key. The only persistent correspondence across I/O pairs is the repeated geometry.
# f) Visual cleanliness:
#    - Shape library size is small, anchors are sparse, and sprites do not touch. This avoids mixed border
#      clutter and keeps each marker region visually readable inside its sprite.
# g) Color-change inferability:
#    - No hidden color arithmetic is introduced. Colors stay locally meaningful (bg / anchor / marker / body),
#      and movement is still the original axis-constrained slide, now keyed by shape instead of by color id.


# --------------- shared config ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (16, 22)
_GRID_WIDTH_RANGE = (16, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 2)
_SPRITE_COUNT_RANGE = (2, 3)
_BODY_AREA_RANGE = (4, 10)
_GEN_RETRY_LIMIT = 400

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

_DISTRACTOR_PROB = 0.0
_DISTRACTOR_COUNT_RANGE = (0, 0)

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]

_SHAPE_LIBRARY = (
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites until their marker subshapes line up with matching border anchor shapes.

    Input structure:
    - bg = most common color.
    - Border anchors are isolated 2-3 cell monochrome shapes on exactly one edge.
    - Interior sprites are mostly one body color plus a smaller connected marker region of another color.

    Rule:
    - Pair each sprite with the border anchor whose shape matches the sprite's marker subshape.
    - Top/bottom-edge anchors shift the sprite horizontally so the marker shape's column footprint matches the anchor.
    - Left/right-edge anchors shift the sprite vertically so the marker shape's row footprint matches the anchor.

    Key insight: Color no longer decides the pairing; the minority region only marks the subshape, and the
    repeated geometry on the border tells you which axis-constrained slide to apply.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _norm_sig(cells) -> tuple[tuple[int, int], ...]:
        r0 = min(r for r, _c in cells)
        c0 = min(c for _r, c in cells)
        return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))

    def _bbox_of(cells) -> tuple[int, int, int, int]:
        return (
            min(int(r) for r, _c in cells),
            min(int(c) for _r, c in cells),
            max(int(r) for r, _c in cells),
            max(int(c) for _r, c in cells),
        )

    def _touches_single_edge(cells) -> str | None:
        top = any(int(r) == 0 for r, _c in cells)
        bottom = any(int(r) == int(H - 1) for r, _c in cells)
        left = any(int(c) == 0 for _r, c in cells)
        right = any(int(c) == int(W - 1) for _r, c in cells)
        touched = [name for name, ok in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if ok]
        if len(touched) != 1:
            return None
        return touched[0]

    def _isolated_component(cells) -> bool:
        S = {(int(r), int(c)) for r, c in cells}
        for r, c in S:
            for rr in range(int(r) - 1, int(r) + 2):
                for cc in range(int(c) - 1, int(c) + 2):
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if (int(rr), int(cc)) in S:
                        continue
                    if int(g[int(rr), int(cc)]) != int(bg):
                        return False
        return True

    def _is_conn4(cells) -> bool:
        pts = {(int(r), int(c)) for r, c in cells}
        if not pts:
            return False
        stack = [next(iter(pts))]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nb in neighbors4(cur[0], cur[1]):
                if nb in pts and nb not in seen:
                    stack.append(nb)
        return len(seen) == len(pts)

    # --- STEP 1) Detect anchors: isolated border components indexed by shape. ---
    anchors_by_shape: dict[tuple[tuple[int, int], ...], dict[str, int | tuple[tuple[int, int], ...]]] = {}
    anchor_cells = set()
    mono_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in mono_comps:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if len(coords) not in (2, 3):
            continue
        edge = _touches_single_edge(coords)
        if edge is None or not _isolated_component(coords):
            continue
        sig = _norm_sig(coords)
        r0, c0, _r1, _c1 = _bbox_of(coords)
        anchors_by_shape[sig] = {
            "sig": sig,
            "r0": int(r0),
            "c0": int(c0),
            "axis": "h" if edge in ("top", "bottom") else "v",
        }
        anchor_cells |= {(int(r), int(c)) for r, c in coords}

    assert anchors_by_shape

    # --- STEP 2) Extract all foreground connected components as candidate sprites. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        coord_list = [(int(r), int(c)) for r, c in coords]
        if set(coord_list) <= anchor_cells:
            continue

        marker_cells: list[tuple[int, int]] | None = None
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)

        # --- STEP 3) Extract the marker subshape for this component. ---
        if len(cnt) > 1:
            min_n = min(int(n) for n in cnt.values())
            if sum(int(n) == int(min_n) for n in cnt.values()) != 1:
                continue
            marker_color = next(int(color) for color, n in cnt.items() if int(n) == int(min_n))
            marker_cells = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker_color)]
            if len(marker_cells) not in (2, 3) or not _is_conn4(marker_cells):
                continue
        else:
            # Shape-only distractor: whole component is the marker shape, as long as it is not a border anchor.
            if any((int(r), int(c)) in anchor_cells for r, c in coord_list):
                continue
            if len(coord_list) not in (2, 3) or not _is_conn4(coord_list):
                continue
            marker_cells = coord_list

        sig = _norm_sig(marker_cells)
        if sig not in anchors_by_shape:
            continue

        anchor = anchors_by_shape[sig]
        mr0, mc0, _mr1, _mc1 = _bbox_of(marker_cells)

        # --- STEP 4) Compute translation (dr,dc) from marker-bbox alignment. ---
        dr = 0
        dc = 0
        if anchor["axis"] == "h":
            dc = int(anchor["c0"]) - int(mc0)
        else:
            dr = int(anchor["r0"]) - int(mr0)

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 5) Apply all moves. ---
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
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
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


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(int(r), int(mc - c)) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(int(mr - r), int(c)) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = int(r + dr), int(c + dc)
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
    shape_h = max(r for r, _c in shape) + 1
    shape_w = max(c for _r, c in shape) + 1
    scored = []
    for top in range(r0 - shape_h - 1, r1 + 3):
        for left in range(c0 - shape_w - 1, c1 + 3):
            placed = _translate(shape, top, left)
            if placed & body:
                continue
            if not any((int(r + dr), int(c + dc)) in body for r, c in placed for dr, dc in _NEIGHBORS4):
                continue
            all_cells = body | placed
            rr0, cc0, rr1, cc1 = _bbox(all_cells)
            if rr1 - rr0 + 1 > _BODY_HEIGHT_RANGE[1] + 3:
                continue
            if cc1 - cc0 + 1 > _BODY_WIDTH_RANGE[1] + 3:
                continue
            mr0, mc0, _mr1, _mc1 = _bbox(placed)
            off_r = int(mr0 - rr0)
            off_c = int(mc0 - cc0)
            # Prefer marker regions that sit near the sprite's top-left; this keeps axis-aligned
            # placements feasible for many anchor positions.
            score = (off_r > 2) + (off_c > 2), off_r + off_c, rr1 - rr0, cc1 - cc0
            scored.append((score, placed))
    scored.sort(key=lambda x: x[0])
    return [placed for _score, placed in scored[:12]]


def _make_sprite(area, body_color, marker_color, marker_shape, bg):
    body = _make_body(area)
    placements = _shape_placement_candidates(body, marker_shape)
    if not placements:
        return None
    marker = placements[0]
    all_cells = set(body) | set(marker)
    r0, c0, r1, c1 = _bbox(all_cells)
    mr0, mc0, _mr1, _mc1 = _bbox(marker)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    for r, c in marker:
        patch[int(r - r0), int(c - c0)] = int(marker_color)
    return {
        "patch": patch,
        "marker_offset": (int(mr0 - r0), int(mc0 - c0)),
        "marker_shape": _sig(marker_shape),
    }


def _top_left(sprite, marker_r0, marker_c0):
    off_r, off_c = sprite["marker_offset"]
    return int(marker_r0) - int(off_r), int(marker_c0) - int(off_c)


def _can_place(canvas, sprite, marker_r0, marker_c0, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r0, marker_c0)
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


def _place(canvas, sprite, marker_r0, marker_c0, bg):
    top, left = _top_left(sprite, marker_r0, marker_c0)
    paste(canvas, sprite["patch"], int(top), int(left), transparent=int(bg))


def _anchor_cells(shape, side, H, W, offset):
    shape = _norm(shape)
    r0, c0, r1, c1 = _bbox(shape)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    if side == "top":
        top, left = 0, int(offset)
    elif side == "bottom":
        top, left = int(H - h), int(offset)
    elif side == "left":
        top, left = int(offset), 0
    else:
        top, left = int(offset), int(W - w)
    return _translate(shape, top, left)


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
    shape = _norm(shape)
    _r0, _c0, r1, c1 = _bbox(shape)
    h, w = int(r1 + 1), int(c1 + 1)
    out = []
    if W - w >= 2:
        max_left = min(W - w - 1, W - _MAX_SPRITE_SPAN_W - 1)
        cols = list(range(1, max_left + 1))
        cols = [c for c in cols if _ANCHOR_EDGE_MARGIN <= c <= W - w - _ANCHOR_EDGE_MARGIN] or cols
        for left in cols:
            out.append(("h", "top", _anchor_cells(shape, "top", H, W, left)))
            out.append(("h", "bottom", _anchor_cells(shape, "bottom", H, W, left)))
    if H - h >= 2:
        max_top = min(H - h - 1, H - _MAX_SPRITE_SPAN_H - 1)
        rows = list(range(1, max_top + 1))
        rows = [r for r in rows if _ANCHOR_EDGE_MARGIN <= r <= H - h - _ANCHOR_EDGE_MARGIN] or rows
        for top in rows:
            out.append(("v", "left", _anchor_cells(shape, "left", H, W, top)))
            out.append(("v", "right", _anchor_cells(shape, "right", H, W, top)))
    random.shuffle(out)
    good = []
    for axis, side, cells in out:
        if not _isolated_bg_cells(canvas, cells, bg):
            continue
        good.append((axis, side, cells))
    return good


def _put_anchor(canvas, cells, color):
    for r, c in cells:
        canvas[int(r), int(c)] = int(color)


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (marker_r0, anchor["c0"])
            for marker_r0 in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, marker_r0, anchor["c0"], bg)
        ]
    return [
        (anchor["r0"], marker_c0)
        for marker_c0 in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r0"], marker_c0, bg)
    ]


def _scatter_positions(canvas, sprite, anchor, marker_pos, bg):
    H, W = canvas.shape
    mr0, mc0 = marker_pos
    if anchor["axis"] == "h":
        return [
            (mr0, other_c0)
            for other_c0 in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if other_c0 != mc0 and _can_place(canvas, sprite, mr0, other_c0, bg)
        ]
    return [
        (other_r0, mc0)
        for other_r0 in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if other_r0 != mr0 and _can_place(canvas, sprite, other_r0, mc0, bg)
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
    shape_pool = random.sample(list(_SHAPE_LIBRARY), random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    used_sigs = set()
    for shape in shape_pool:
        candidates = _anchor_candidates(out, shape, bg)
        if not candidates:
            return None
        axis, side, cells = random.choice(candidates)
        sig = _sig(shape)
        if sig in used_sigs:
            return None
        used_sigs.add(sig)
        r0, c0, _r1, _c1 = _bbox(cells)
        anchor = {
            "shape": _norm(shape),
            "sig": sig,
            "axis": axis,
            "side": side,
            "r0": int(r0),
            "c0": int(c0),
            "cells": cells,
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


def _place_aligned_sprites(out, anchors, marker_shapes, marker_color, anchor_color, bg):
    by_sig = {a["sig"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker_shape in marker_shapes:
        sig = _sig(marker_shape)
        anchor = by_sig[sig]
        body = _body_color(bg, anchor_color, marker_color, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker_color, marker_shape, bg)
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _add_distractors(out, inp, anchors, marker_color, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return
    count = random.randint(*_DISTRACTOR_COUNT_RANGE)
    for _ in range(count):
        anchor = random.choice(anchors)
        sprite = {
            "patch": np.full((1, 1), int(bg), dtype=int),  # replaced below
            "marker_offset": (0, 0),
            "marker_shape": anchor["sig"],
        }
        shape = anchor["shape"]
        _r0, _c0, r1, c1 = _bbox(shape)
        patch = np.full((int(r1 + 1), int(c1 + 1)), int(bg), dtype=int)
        for r, c in shape:
            patch[int(r), int(c)] = int(marker_color)
        sprite["patch"] = patch
        aligned = _aligned_positions(out, sprite, anchor, bg)
        if not aligned:
            return
        out_pos = random.choice(aligned)
        scatter = _scatter_positions(inp, sprite, anchor, out_pos, bg)
        if not scatter:
            return
        in_pos = random.choice(scatter)
        _place(out, sprite, out_pos[0], out_pos[1], bg)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)


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

    _add_distractors(out, inp, anchors, marker_color, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D1 input")
