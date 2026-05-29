# Variant of 581f7754
# Dimension: D12 — 精灵体形状多样性 (meta)
# Change: 体形为不规则小形状（空心矩形、十字、L形、环形等），外接矩形2x3到5x7，约60%具有左右或上下对称... → 精灵形状编码信息（形状决定滑动距离）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D12 variant
# a) Concrete behavior:
#    The original task stops a sprite when its marker reaches the anchor column/row.
#    This variant keeps the same anchor-color matching and axis choice, but the stopping
#    point now includes an extra offset derived from the sprite body itself.
#    For a top/bottom anchor, the sprite still slides horizontally, yet the marker ends
#    at anchor_col + body_bbox_width. For a left/right anchor, it still slides vertically,
#    yet the marker ends at anchor_row + body_bbox_height.
#    Example: suppose a top anchor of color 4 is at column 2 and a matching sprite body
#    has width 4. The old rule moved its marker to column 2; the new rule moves it to
#    column 6. Only the slide axis is inherited unchanged; the stop distance is now shape-
#    dependent. This changes the solver exactly at the translation-computation stage and
#    changes the generator at the solved-state placement stage.
# b) Complexity calibration:
#    Using exact body width/height is visible and low-ambiguity, unlike hidden color ids.
#    The allowed spans are restricted to 3..5 and each puzzle reuses 2-3 span values across
#    multiple sprites, so the rule is neither trivial nor too unconstrained.
# c) Old implicit assumptions that break:
#    - Old `dc = ac - mp[1]` / `dr = ar - mp[0]` assumed the anchor projection itself was
#      the final stop. That is no longer valid; we must add a body-span term.
#    - Old singleton distractors were harmless because all movable objects shared the same
#      "align exactly with anchor" rule. Here a singleton has no meaningful width/height
#      code, so keeping it would introduce a second unrelated rule. The generator therefore
#      removes that branch.
#    - Old random marker placement did not matter for the stopping rule. In this variant,
#      the shape-coded offset must remain easy to see, so the generator places markers on
#      the movement-facing outer side of the body, which makes the width/height cue legible.
# d) Placement legality:
#    Output is still built first. Anchors are placed with isolation. Each solved sprite is
#    positioned by searching valid placements whose marker already satisfies the new target
#    row/column. Input is obtained by scattering each sprite away along the same slide axis,
#    while `_can_place` enforces non-overlap, border gaps, and 8-neighbor isolation.
# e) Visual observability:
#    A human can compare anchor-to-marker distance with sprite width/height directly. Because
#    span values repeat within a puzzle, samples show that equal-width sprites land equally
#    far from their anchors, while different widths produce different landing distances.
# f) Visual cleanliness:
#    Components remain separated by background. No distractor singletons are mixed in, so
#    every non-anchor movable object follows the same two-color sprite grammar.
# g) Color inference:
#    Colors keep their original role only: marker color selects the anchor. No sorting or
#    arithmetic uses color ids. The new control variable is purely geometric and visible.
# h) New solver assumptions and generator guarantees:
#    - Every movable sprite has exactly one least-frequent marker color pixel.
#    - The rest of the component forms the body whose bbox width/height is well-defined.
#    - Each marker color has exactly one anchor in generated data.
#    - No two moved sprites need to overwrite the same destination cell.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior sprite toward its matching border anchor, then continue by a shape-coded extra distance.

    Input structure:
    - The background is the modal color.
    - Interior sprites are 2-color connected components: one body color plus one marker color.
    - Border anchors are isolated single pixels whose color matches the marker color of one or more sprites.

    Rule:
    - A top/bottom anchor still means horizontal motion; a left/right anchor still means vertical motion.
    - After choosing the axis from the anchor edge, move the sprite so its marker lands at:
      anchor column + body width for horizontal slides, or anchor row + body height for vertical slides.
    - Source cells are cleared first; pasted pixels clip at the grid boundary and do not overwrite existing non-bg cells.

    Key insight:
    The anchor tells you which axis to use, while the sprite body's bbox span on that axis tells you how much farther past the anchor projection the marker must travel.
    """
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
            isolated = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    isolated = False
                    break
            if isolated and v not in anchors_by_color:
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

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if len(pts) == 1:
            return pts[0]
        if not pts:
            return None
        body = {(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker)}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in body:
                    return (int(r), int(c))
        return pts[0]

    def body_axis_span(coords: np.ndarray, marker: int, horizontal: bool) -> int | None:
        body = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker)]
        if not body:
            return None
        rows = [r for r, _c in body]
        cols = [c for _r, c in body]
        if horizontal:
            return int(max(cols) - min(cols) + 1)
        return int(max(rows) - min(rows) + 1)

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None or int(marker_col) not in anchors_by_color:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        span = body_axis_span(coords, int(marker_col), horizontal)
        if span is None:
            continue

        dr = 0
        dc = 0
        if horizontal:
            target_c = int(ac + span)
            dc = int(target_c - mp[1])
        else:
            target_r = int(ar + span)
            dr = int(target_r - mp[0])

        if tuple(map(int, mp)) in anchor_cells:
            continue
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


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (12, 22)
_GRID_WIDTH_RANGE = (12, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (4, 5)
_BODY_AREA_RANGE = (4, 13)
_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 5)
_SPAN_VALUES = (3, 4, 5)
_SYMMETRIC_BODY_PROB = 0.6
_GEN_RETRY_LIMIT = 400


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
    if random.random() < 0.5:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < 0.5:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _valid_body(cells, h, w):
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if (r1 - r0 + 1, c1 - c0 + 1) != (h, w):
        return False
    return _BODY_AREA_RANGE[0] <= len(cells) <= _BODY_AREA_RANGE[1]


def _body_templates(h, w):
    sym = []
    asym = []

    l_shape = {(0, c) for c in range(w)} | {(r, 0) for r in range(h)}
    asym.append(l_shape)

    t_shape = {(0, c) for c in range(w)} | {(r, w // 2) for r in range(h)}
    sym.append(t_shape)

    upside_t = {(h - 1, c) for c in range(w)} | {(r, w // 2) for r in range(h)}
    sym.append(upside_t)

    comb = {(r, 0) for r in range(h)} | {(h // 2, c) for c in range(w)}
    sym.append(comb)

    if h >= 3 and w >= 3:
        ring = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        }
        sym.append(ring)

        c_shape = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
        asym.append(c_shape)

        u_shape = {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
        sym.append(u_shape)

    full = {(r, c) for r in range(h) for c in range(w)}
    if h * w - 1 >= _BODY_AREA_RANGE[0]:
        notch = set(full)
        notch.remove((0, 0))
        asym.append(notch)
    if h >= 3 and w >= 3 and h * w - 2 >= _BODY_AREA_RANGE[0]:
        notch2 = set(full)
        notch2.remove((0, 0))
        notch2.remove((1, 1))
        asym.append(notch2)

    sym = [_norm(s) for s in sym if _valid_body(s, h, w)]
    asym = [_norm(s) for s in asym if _valid_body(s, h, w)]
    return sym, asym


def _make_body(h, w):
    sym, asym = _body_templates(h, w)
    if not sym and not asym:
        return None
    pick_sym = bool(sym) and (not asym or random.random() < _SYMMETRIC_BODY_PROB)
    pool = sym if pick_sym else asym
    if not pool:
        pool = sym or asym
    return _flip(random.choice(pool))


def _marker_cell(body, axis):
    if axis == "h":
        max_c = max(c for _r, c in body)
        rows = [r for r, c in body if c == max_c]
        return (random.choice(rows), max_c + 1)
    max_r = max(r for r, _c in body)
    cols = [c for r, c in body if r == max_r]
    return (max_r + 1, random.choice(cols))


def _make_sprite(axis, span, body_color, marker_color, bg):
    for _ in range(40):
        if axis == "h":
            h = random.randint(*_BODY_HEIGHT_RANGE)
            w = int(span)
        else:
            h = int(span)
            w = random.randint(*_BODY_WIDTH_RANGE)
        body = _make_body(int(h), int(w))
        if body is None:
            continue
        marker = _marker_cell(body, axis)
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
        return {
            "patch": patch,
            "offset": (marker[0] - r0, marker[1] - c0),
            "span": int(span),
            "axis": axis,
        }
    return None


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


def _anchor_candidates(canvas, axis, bg):
    H, W = canvas.shape
    out = []
    if axis == "h":
        max_c = W - max(_SPAN_VALUES) - 3
        cols = list(range(_ANCHOR_EDGE_MARGIN, max(max_c + 1, _ANCHOR_EDGE_MARGIN + 1)))
        for c in cols:
            out.append((0, c))
            out.append((H - 1, c))
    else:
        max_r = H - max(_SPAN_VALUES) - 3
        rows = list(range(_ANCHOR_EDGE_MARGIN, max(max_r + 1, _ANCHOR_EDGE_MARGIN + 1)))
        for r in rows:
            out.append((r, 0))
            out.append((r, W - 1))
    random.shuffle(out)
    return [(r, c) for r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


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
    n_anchors = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _COLORS if c != bg], n_anchors)
    axis_plan = ["h", "v"]
    while len(axis_plan) < n_anchors:
        axis_plan.append(random.choice(("h", "v")))
    random.shuffle(axis_plan)
    anchors = []
    for color, axis in zip(anchor_colors, axis_plan):
        candidates = _anchor_candidates(out, axis, bg)
        if not candidates:
            return None
        r, c = random.choice(candidates)
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


def _span_sequence(n_sprites):
    palette_size = random.randint(2, min(3, len(_SPAN_VALUES)))
    palette = random.sample(list(_SPAN_VALUES), palette_size)
    spans = palette[:]
    while len(spans) < n_sprites:
        spans.append(random.choice(palette))
    random.shuffle(spans)
    return spans


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    span = int(sprite["span"])
    if anchor["axis"] == "h":
        target_c = int(anchor["c"] + span)
        if not (_INNER_BORDER_GAP <= target_c < W - _INNER_BORDER_GAP):
            return []
        return [
            (r, target_c)
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, target_c, bg)
        ]
    target_r = int(anchor["r"] + span)
    if not (_INNER_BORDER_GAP <= target_r < H - _INNER_BORDER_GAP):
        return []
    return [
        (target_r, c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, target_r, c, bg)
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


def _place_aligned_sprites(out, anchors, marker_colors, spans, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker, span in zip(marker_colors, spans):
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(anchor["axis"], int(span), body, marker, bg)
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


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed
    marker_colors = _marker_color_sequence(anchor_colors)
    spans = _span_sequence(len(marker_colors))
    sprites = _place_aligned_sprites(out, anchors, marker_colors, spans, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D12 input")
