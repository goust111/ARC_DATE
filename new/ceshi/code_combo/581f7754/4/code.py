# Combo variant of 581f7754 (index 4)
# combo_spec: D6/1+D14/0
# Name: 滑动留痕+反向滑动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Keep the original object vocabulary: isolated non-bg border anchors and interior 8-connected
#      sprites whose least-frequent color is the marker. Matching is still by marker color.
#    - The fused rule changes two linked stages at once:
#        1. the slide direction is inverted relative to the base task;
#        2. after clearing the source sprite, the marker's swept path is redrawn as a thin trail.
#    - Small trace:
#        top-edge anchor color 4 at column 3, input marker color 4 at (5,6).
#        Base task: horizontal move dc = 3 - 6 = -3, so marker lands at (5,3).
#        Combo task: horizontal move dc = 6 - 3 = +3, so marker lands at (5,9).
#        Then the source component is cleared and the cells (5,6),(5,7),(5,8) are painted 4.
#      The anchor stays fixed on the border, while the sprite ends farther away and leaves a clue line.
# b) Complexity calibration
#    - Keep 1-3 anchor colors and 3-5 sprites so the direction inversion repeats within a sample.
#    - Use the marker color itself for the trail; this makes the correspondence visible without adding
#      a second hidden color mapping.
#    - Require a minimum slide length of 2 so the away-from-anchor behavior and the trail are both easy
#      to see. Multiple sprites may reuse one anchor color so the same rule is reinforced repeatedly.
# c) Hidden assumptions in the original main()
#    - Anchor detection, component extraction, marker-color inference, and axis selection still hold
#      because the input format is unchanged.
#    - The original translation formula does not hold; it must flip sign:
#        horizontal: dc = marker_c - anchor_c
#        vertical:   dr = marker_r - anchor_r
#    - The original clear-then-paste pass is still valid for the moved sprite, but the trail must be
#      painted after pasting. Otherwise the trail could occupy destination cells and block the sprite,
#      because the paste logic only writes onto background.
# d) Placement legality
#    - Generator is input-first because the output no longer aligns markers onto the anchor row/column.
#    - Every candidate input marker position must satisfy three conditions simultaneously:
#        1. the sprite fits on the input canvas,
#        2. the mirrored-away destination fits on the output base canvas,
#        3. the marker trail corridor is background in the output base and unused by earlier trails.
#    - This guarantees no overlap, no clipping accidents, and no anchor/trail collisions in generated
#      training cases, even though the solver itself still supports clipping by construction.
# e) Visual observability
#    - A human can still match each sprite to an anchor by color, then notice that the final sprite is
#      not on the anchor line but farther from that border by the same distance.
#    - The trail is a straight marker-colored segment starting at the source marker and pointing toward
#      the final sprite, which makes the inverted direction easier to infer from 2-4 examples.
# f) Visual tidiness
#    - Trails are one cell wide and cannot pass through anchors, settled sprites, or older trails.
#    - Sprites remain isolated by at least one background cell in both input and output base layouts,
#      so the added trails do not turn the scene into merged clutter.
# g) Inferability of the change
#    - No color-id arithmetic or hidden ordering is used. All logic is visible through edge position,
#      shared color, signed distance, and the explicit trail segment.
#    - Because trail color equals marker color and each sample repeats colors across multiple sprites,
#      the fused rule is recoverable without needing many distinct examples.
# h) Quantitative anti-degeneration target
#    - Generated motions must have distance >= 2, so the combo cannot silently collapse into "no move".
#    - Since the direction is reversed, the combo output must differ from the base-task output on the
#      same input whenever the move is unobstructed; the generator keeps destination cells free so this
#      difference remains visible instead of being masked by collisions.
# i) New assumptions introduced by the fused main()
#    - The trail is assumed to be axis-aligned, which holds because every move changes only one axis.
#    - Trail rendering assumes the source marker cell becomes visible after clearing the source sprite;
#      the generator therefore keeps the output base background at every trail cell.
#    - Marker uniqueness is not strictly required by main(), but the generator still builds sprites with
#      a single minority marker so the intended correspondence stays unambiguous.


def _marker_trail_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, src)
    dr = int(dst[0] - sr)
    dc = int(dst[1] - sc)
    steps = int(max(abs(dr), abs(dc)))
    if steps <= 0:
        return []
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    return [(int(sr + k * step_r), int(sc + k * step_c)) for k in range(steps)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites away from their matching border anchors and leave a marker-colored trail.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are sparse 8-connected sprites; each multi-color sprite has a rare marker pixel.
    - Isolated non-background border singletons are anchors, and anchor colors match sprite marker colors.

    Rule:
    - Pair each sprite with the anchor of the same marker color.
    - Top or bottom anchors still force horizontal motion, while left or right anchors still force vertical motion.
    - Instead of moving toward the anchor, the sprite moves the same distance away from it, then the cleared source marker path is redrawn as a continuous trail in the marker color.

    Key insight:
    - The border tells you which axis matters, but the final landing point is the mirror image across the anchor coordinate; the extra colored line is the swept path of that reversed move.
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
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color

    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        comp_set = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[dict[str, object]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
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

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(mp[1] - ac)
        else:
            dr = int(mp[0] - ar)

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
            }
        )

    out = g.copy()

    for move in moves:
        coords = move["coords"]
        assert isinstance(coords, np.ndarray)
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for move in moves:
        coords = move["coords"]
        dr = int(move["dr"])
        dc = int(move["dc"])
        assert isinstance(coords, np.ndarray)
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    for move in moves:
        src = move["marker_pixel"]
        assert isinstance(src, tuple)
        dst = (int(src[0] + int(move["dr"])), int(src[1] + int(move["dc"])))
        color = int(move["marker_color"])
        for r, c in _marker_trail_cells(src, dst):
            if 0 <= r < H and 0 <= c < W and int(out[r, c]) == int(bg):
                out[r, c] = int(color)

    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_MIN_SLIDE_DISTANCE = 2
_CANDIDATE_POOL_LIMIT = 20

_ANCHOR_COUNT_RANGE = (1, 3)
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

_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

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


def _variant_output_marker(anchor, marker_r, marker_c):
    if anchor["axis"] == "h":
        return int(marker_r), int(2 * int(marker_c) - int(anchor["c"]))
    return int(2 * int(marker_r) - int(anchor["r"])), int(marker_c)


def _trail_clear(base_out, trail_cells, trail_used, bg):
    H, W = base_out.shape
    for r, c in trail_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(base_out[r, c]) != int(bg):
            return False
        if (int(r), int(c)) in trail_used:
            return False
    return True


def _paired_positions(inp, out_base, trail_used, sprite, anchor, bg):
    H, W = inp.shape
    rows = list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    random.shuffle(rows)
    random.shuffle(cols)
    pairs = []

    if anchor["axis"] == "h":
        for r in rows:
            for c in cols:
                dist = abs(int(c) - int(anchor["c"]))
                if dist < _MIN_SLIDE_DISTANCE:
                    continue
                out_r, out_c = _variant_output_marker(anchor, r, c)
                if not (_INNER_BORDER_GAP <= out_c < W - _INNER_BORDER_GAP):
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                if not _can_place(out_base, sprite, out_r, out_c, bg):
                    continue
                trail = _marker_trail_cells((int(r), int(c)), (int(out_r), int(out_c)))
                if not _trail_clear(out_base, trail, trail_used, bg):
                    continue
                pairs.append(
                    {
                        "src": (int(r), int(c)),
                        "dst": (int(out_r), int(out_c)),
                        "trail": trail,
                        "distance": int(dist),
                        "axis": "h",
                    }
                )
    else:
        for c in cols:
            for r in rows:
                dist = abs(int(r) - int(anchor["r"]))
                if dist < _MIN_SLIDE_DISTANCE:
                    continue
                out_r, out_c = _variant_output_marker(anchor, r, c)
                if not (_INNER_BORDER_GAP <= out_r < H - _INNER_BORDER_GAP):
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                if not _can_place(out_base, sprite, out_r, out_c, bg):
                    continue
                trail = _marker_trail_cells((int(r), int(c)), (int(out_r), int(out_c)))
                if not _trail_clear(out_base, trail, trail_used, bg):
                    continue
                pairs.append(
                    {
                        "src": (int(r), int(c)),
                        "dst": (int(out_r), int(out_c)),
                        "trail": trail,
                        "distance": int(dist),
                        "axis": "v",
                    }
                )

    if not pairs:
        return None
    pairs.sort(key=lambda item: int(item["distance"]), reverse=True)
    trimmed = pairs[: min(len(pairs), _CANDIDATE_POOL_LIMIT)]
    return random.choice(trimmed)


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _render_output(base_out, moves, bg):
    out = base_out.copy()
    for move in moves:
        color = int(move["color"])
        for r, c in _marker_trail_cells(move["src"], move["dst"]):
            if int(out[r, c]) == int(bg):
                out[r, c] = int(color)
    return out


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


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_sprite_pairs(out_base, inp, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    trail_used: set[tuple[int, int]] = set()
    moves = []

    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        pair = _paired_positions(inp, out_base, trail_used, sprite, anchor, bg)
        if pair is None:
            return None
        _place(inp, sprite, pair["src"][0], pair["src"][1], bg)
        _place(out_base, sprite, pair["dst"][0], pair["dst"][1], bg)
        trail_used.update((int(r), int(c)) for r, c in pair["trail"])
        moves.append(
            {
                "src": pair["src"],
                "dst": pair["dst"],
                "trail": pair["trail"],
                "distance": int(pair["distance"]),
                "axis": pair["axis"],
                "color": int(marker),
            }
        )

    return moves, trail_used


def _add_distractors(out_base, inp, anchors, trail_used, bg):
    moves = []
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return moves

    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pair = _paired_positions(inp, out_base, trail_used, sprite, anchor, bg)
        if pair is None:
            break
        _place(inp, sprite, pair["src"][0], pair["src"][1], bg)
        _place(out_base, sprite, pair["dst"][0], pair["dst"][1], bg)
        trail_used.update((int(r), int(c)) for r, c in pair["trail"])
        moves.append(
            {
                "src": pair["src"],
                "dst": pair["dst"],
                "trail": pair["trail"],
                "distance": int(pair["distance"]),
                "axis": pair["axis"],
                "color": int(anchor["color"]),
            }
        )
    return moves


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _generate_case():
    bg, out_base, inp = _new_canvas_pair()
    placed = _place_anchors(out_base, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    placed_sprites = _place_sprite_pairs(out_base, inp, anchors, marker_colors, bg)
    if placed_sprites is None:
        return None
    sprite_moves, trail_used = placed_sprites

    distractor_moves = _add_distractors(out_base, inp, anchors, trail_used, bg)
    moves = sprite_moves + distractor_moves
    if not moves:
        return None
    if max(int(move["distance"]) for move in moves) < _MIN_SLIDE_DISTANCE:
        return None

    out = _render_output(out_base, moves, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None

    return {"input": inp, "output": out, "anchors": anchors, "moves": moves, "bg": int(bg)}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate combo variant for 581f7754")
