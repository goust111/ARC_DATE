# Variant of 581f7754
# Dimension: D6 — 变换：原始位置处理 (trajectory)
# Change: 精灵的原始位置像素全部变为背景色（清除），锚点保持不变... → 滑动轨迹留痕

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D6 trajectory variant
# ============================================
# a) Concrete behavior and trace
#    - Keep the original object vocabulary: isolated border anchors and interior multi-color sprites
#      whose least-frequent color is the marker. Matching is still by marker color.
#    - The only rule change is what happens to the source location after a slide. Instead of simply
#      erasing the whole source sprite to background, the marker's swept path is re-drawn as a 1-cell
#      wide continuous line in the marker color.
#    - Small trace:
#        input marker at (5,2), matching top anchor at column 7 -> horizontal slide by dc=+5.
#        Original task output: clear source sprite, paste sprite with marker at (5,7).
#        Variant output: clear source sprite, paste sprite with marker at (5,7), then color
#        (5,2),(5,3),(5,4),(5,5),(5,6) with the marker color. The anchor stays on the border.
#      The visible difference is therefore entirely in the "old position handling" stage.
# b) Complexity calibration
#    - Use the marker color itself as the trail color. This avoids introducing a second color mapping
#      that would be hard to infer from 2-4 examples, while still making the variant visually distinct.
#    - The trail is only the marker path, not the full swept footprint of the body. Full-body sweeps
#      would create thick overlaps and would materially change the task rather than only D6.
#    - Generator enforces a minimum slide length of 2 cells so the trail is normally visible.
# c) Hidden assumptions in the original main()
#    - Anchor detection, component extraction, marker-color inference, and axis selection all remain
#      valid because the input format is unchanged; only the output rendering changes.
#    - The original "clear everything first, then paste moved sprites into empty cells" is not enough
#      anymore. If a trail were drawn before the final sprite paste, the trail could block the sprite's
#      destination because the old paste logic only writes onto background.
#    - Therefore the new order is:
#        1. clear moved source components
#        2. paste final moved sprites
#        3. paint the trail only onto cells that are still background
#      This preserves the original non-overwrite collision behavior while leaving visible traces.
# d) Placement legality guarantees in the generator
#    - Input sprites are still placed with the original non-touching interior constraints.
#    - Output-aligned sprites are placed first on a clean "solved" canvas.
#    - When choosing each scattered input position, the generator checks that the marker trail cells
#      (source marker through the cell before the destination marker) are background on the solved
#      canvas and do not intersect previously reserved trail cells.
#    - Distractor singletons, when present, obey the same reservation rule.
# e) Visual observability
#    - A human sees the same anchor-to-marker matching as in the base task, plus a colored stripe that
#      reveals where the marker traveled. Because the stripe is axis-aligned and shares the marker's
#      color, it points back to the correct anchor relation rather than looking arbitrary.
#    - Samples still mix multiple sprites and repeated marker colors, so the solver must infer the axis
#      rule, not memorize one layout.
# f) Visual tidiness
#    - Trails are one pixel wide.
#    - Generator rejects placements where trails would cross anchors, settled sprites, or earlier trails.
#    - This keeps the output readable instead of producing merged clutter.
# g) Color inferability
#    - No invisible ordering or color-id arithmetic is used.
#    - Trail color = marker color, which is directly visible in both input sprite and border anchor.
# h) Reference task calibration
#    - The source task 8b9c3697 also clears the original object and paints an intermediate trail before
#      showing the final placement. This variant adopts that same "clear + trail + final object" idea,
#      but only for the moving marker path so the rest of 581f7754 remains intact.
# i) New assumptions introduced by main()
#    - The trail is assumed to be a straight line because every move is purely horizontal or vertical.
#    - The generator guarantees nonzero motion and usually at least two trail cells.
#    - Because main() paints trails only on background, generator must keep the trail corridor free in
#      the solved layout; _scatter_positions() and _add_distractors() enforce exactly that.


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
    Slide interior sprites to match border anchors, leaving a marker-colored trail behind the motion.

    Input structure:
    - bg is the most common color.
    - Interior sprites are 8-connected foreground components; each multi-color sprite has a rare marker pixel.
    - Isolated non-bg border pixels act as anchors, and the anchor color matches the sprite marker color.

    Rule:
    - Match each sprite to the anchor with the same marker color.
    - Top/bottom anchors force a horizontal slide so the marker lands in the anchor's column; left/right anchors force a vertical slide so the marker lands in the anchor's row.
    - After removing the source sprite, redraw the marker's traversed cells as a continuous trail in the marker color, while keeping the anchor unchanged.

    Key insight:
    - The border edge still tells you the slide axis; the new non-bg line is simply the marker pixel's travel path from the original placement to the aligned placement.
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

    assert not (not anchors_by_color)

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
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[dict] = []

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
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "marker_color": int(marker_col),
            }
        )

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

    for move in moves:
        marker_col = int(move["marker_color"])
        src = move["marker_pixel"]
        dst = (int(src[0] + move["dr"]), int(src[1] + move["dc"]))
        for r, c in _marker_trail_cells(src, dst):
            if 0 <= r < H and 0 <= c < W and int(out[r, c]) == int(bg):
                out[r, c] = int(marker_col)

    return out


# --------------- generator (D6 trajectory variant) ---------------

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
_GEN_RETRY_LIMIT = 300
_MIN_TRAIL_STEPS = 2

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


def _scatter_positions(inp, out_base, trail_used, sprite, anchor, marker, bg):
    H, W = inp.shape
    mr, mc = marker
    positions = []
    if anchor["axis"] == "h":
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc or abs(int(c - mc)) < _MIN_TRAIL_STEPS:
                continue
            if not _can_place(inp, sprite, mr, c, bg):
                continue
            trail = _marker_trail_cells((mr, c), (mr, mc))
            if _trail_clear(out_base, trail, trail_used, bg):
                positions.append({"marker": (int(mr), int(c)), "trail": trail})
    else:
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if r == mr or abs(int(r - mr)) < _MIN_TRAIL_STEPS:
                continue
            if not _can_place(inp, sprite, r, mc, bg):
                continue
            trail = _marker_trail_cells((r, mc), (mr, mc))
            if _trail_clear(out_base, trail, trail_used, bg):
                positions.append({"marker": (int(r), int(mc)), "trail": trail})
    return positions


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _render_output(base_out, trail_moves, bg):
    out = base_out.copy()
    for move in trail_moves:
        color = int(move["color"])
        for r, c in _marker_trail_cells(move["src"], move["dst"]):
            if int(out[r, c]) == int(bg):
                out[r, c] = int(color)
    return out


def _add_distractors(out_base, inp, anchors, trail_used, bg):
    moves = []
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return moves
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        for c in range(_INNER_BORDER_GAP, out_base.shape[1] - _INNER_BORDER_GAP):
            if (int(anchor["r"]), int(c)) in trail_used:
                continue
            if not _can_place(out_base, sprite, anchor["r"], c, bg):
                continue
            dst = (int(anchor["r"]), int(c))
            for r in range(_INNER_BORDER_GAP, out_base.shape[0] - _INNER_BORDER_GAP):
                if r == anchor["r"] or abs(int(r - anchor["r"])) < _MIN_TRAIL_STEPS:
                    continue
                if not _can_place(inp, sprite, r, c, bg):
                    continue
                trail = _marker_trail_cells((r, c), dst)
                if _trail_clear(out_base, trail, trail_used, bg):
                    pairs.append({"src": (int(r), int(c)), "dst": dst, "trail": trail})
        if not pairs:
            return moves
        chosen = random.choice(pairs)
        _place(out_base, sprite, chosen["dst"][0], chosen["dst"][1], bg)
        _place(inp, sprite, chosen["src"][0], chosen["src"][1], bg)
        trail_used.update((int(r), int(c)) for r, c in chosen["trail"])
        moves.append({"src": chosen["src"], "dst": chosen["dst"], "color": int(anchor["color"])})
    return moves


def _gen_attempt():
    bg, out_base, inp = _new_canvas_pair()
    placed = _place_anchors(out_base, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out_base, anchors, marker_colors, bg)
    if sprites is None:
        return None

    trail_used: set[tuple[int, int]] = set()
    moves = _scatter_sprites(inp, out_base, sprites, trail_used, bg)
    if moves is None:
        return None

    moves.extend(_add_distractors(out_base, inp, anchors, trail_used, bg))
    out = _render_output(out_base, moves, bg)
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


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(out, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append({"sprite": sprite, "anchor": anchor, "marker_pos": marker_pos})
    return sprites


def _scatter_sprites(inp, out_base, sprites, trail_used, bg):
    random.shuffle(sprites)
    moves = []
    for item in sprites:
        sprite = item["sprite"]
        anchor = item["anchor"]
        marker_pos = item["marker_pos"]
        positions = _scatter_positions(inp, out_base, trail_used, sprite, anchor, marker_pos, bg)
        if not positions:
            return None
        chosen = random.choice(positions)
        _place(inp, sprite, chosen["marker"][0], chosen["marker"][1], bg)
        trail_used.update((int(r), int(c)) for r, c in chosen["trail"])
        moves.append({"src": chosen["marker"], "dst": marker_pos, "color": int(anchor["color"])})
    return moves


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D6 input")
