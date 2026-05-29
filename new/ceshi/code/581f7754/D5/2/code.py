# Variant of 581f7754
# Dimension: D5 — 变换：裁剪规则 (transform)
# Change: 滑动后超出网格边界的像素直接丢弃（裁剪）；目标位置的背景像素被覆盖，非背景像素不被覆盖... → 覆盖而非保留非背景

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
#    The original task already identifies interior sprites by a minority-color marker and slides
#    them toward a matching isolated border anchor. The changed behavior appears only when two
#    moved sprites want the same destination cells: the old task kept the first-arriving non-bg
#    pixel and rejected later ones, while this variant paints later sprites on top. To make that
#    visible, the generator deliberately creates at least one pair of sprites whose solved
#    locations overlap after sliding to the same anchor line. Example on a top-edge anchor:
#    sprite A and sprite B both slide horizontally toward the anchor column; after translation
#    their bodies intersect, so old logic would preserve A at the overlap, while the variant
#    shows B on top if B is later in source reading order.
# b) Complexity calibration:
#    Overlap is forced only for one collision pair, while other sprites remain ordinary examples
#    of the original rule. This keeps the puzzle readable: the solver still needs the original
#    marker/anchor matching and axis inference, but now must also infer a small number of order
#    states rather than a large hidden permutation.
# c) Implicit assumptions audit for main():
#    - Anchor detection still assumes isolated border singletons; generator preserves that.
#    - Marker detection still assumes the least frequent color inside a sprite; generator keeps
#      sprites bi-color with one marker pixel.
#    - Connected components must stay separated in the input; generator scatters sprites with the
#      old non-touching placement rules, so collision happens only after movement.
#    - The old paste loop assumed collision order was irrelevant because non-bg cells blocked
#      later writes. That assumption breaks here, so moves are explicitly sorted by visible source
#      reading order `(top_row, left_col)` before repainting.
#    - Clearing source components before repainting is still necessary; otherwise a later sprite
#      could overwrite a stale source pixel and create a spurious order cue unrelated to the task.
# d) Placement legality:
#    Input placement still uses strict non-overlap/non-touch rules. Output planning allows exactly
#    the intended overlap pair while keeping anchors isolated and other sprites disjoint. Final
#    solved positions are chosen so every sprite individually fits inside the grid; clipping is
#    still supported by the solver but is not required for every sample.
# e) Visual observability:
#    The ordering cue is based on source reading order, which is directly visible from the input.
#    With 2-4 examples, a human can see that overlapping solved sprites inherit the colors of the
#    lower/right-later source object rather than preserving the first one encountered.
# f) Visual cleanliness:
#    Only one overlap group is forced, and the rest of the board remains sparse. That keeps the
#    output legible; the overlap reads as a local interaction rather than turning the whole canvas
#    into layered noise.
# g) Color inference:
#    No hidden color-id ordering is used. Colors still only determine marker-to-anchor matching.
#    Overwrite order depends solely on visible source geometry.
# h) Generator-side anti-degeneration check:
#    Each generated sample must contain at least one overwrite conflict in the solved rendering,
#    and the new solver's output must differ from the legacy non-overwrite rendering on that same
#    input. This prevents the generator from accidentally producing samples where the variant has
#    no observable effect.
# i) New assumptions introduced by the variant:
#    main() now assumes source reading order is the intended painting order. The generator
#    guarantees this is meaningful by separating all input sprites and by choosing collision pairs
#    whose final overlap actually changes visible colors when that order is applied.


def _solve(grid: np.ndarray, *, overwrite_non_bg: bool) -> np.ndarray:
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
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int, tuple[int, int]]] = []

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

        key = (
            int(coords[:, 0].min()),
            int(coords[:, 1].min()),
        )
        moves.append((coords, int(dr), int(dc), key))

    moves.sort(key=lambda item: item[3])

    out = g.copy()
    for coords, _dr, _dc, _key in moves:
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for coords, dr, dc, _key in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                if overwrite_non_bg or int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(g[int(r), int(c)])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior marked sprites toward matching isolated border anchors, allowing later arrivals to repaint earlier ones.

    Input structure:
    - The background is the dominant color.
    - Interior connected sprites are mostly one body color with a rarer marker color.
    - Isolated non-background border pixels act as anchors for the matching marker color.

    Rule:
    - A sprite moves toward the anchor sharing its marker color.
    - Anchors on the top/bottom edges force a horizontal slide to the anchor column; anchors on the left/right edges force a vertical slide to the anchor row.
    - Moved pixels that leave the grid are still clipped, but moved sprite pixels repaint whatever is already at the destination; if two moved sprites overlap, the one painted later covers the earlier one.

    Key insight:
    - The anchor location still chooses the slide axis, but overlap is now resolved by source reading order: sprites higher/lefther in the input paint first, and later ones overwrite them.
    """
    return _solve(grid, overwrite_non_bg=True)


# --------------- generator ---------------

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

_DISTRACTOR_PROB = 0.12
_DISTRACTOR_COUNT_RANGE = (1, 3)
_MIN_OVERWRITE_CONFLICTS = 1

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


def _place(canvas, sprite, marker_r, marker_c, bg, *, overwrite=True):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg), overwrite=overwrite, clip=True)


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
    collision_color = random.choice(anchor_colors)
    marker_colors = [collision_color, collision_color]
    extras = anchor_colors[:]
    random.shuffle(extras)
    while len(marker_colors) < n_sprites:
        if extras:
            marker_colors.append(extras.pop())
        else:
            marker_colors.append(random.choice(anchor_colors))
    tail = marker_colors[2:]
    random.shuffle(tail)
    marker_colors = marker_colors[:2] + tail
    return collision_color, marker_colors


def _translated_cells(sprite, marker_pos, bg):
    top, left = _top_left(sprite, marker_pos[0], marker_pos[1])
    patch = sprite["patch"]
    cells = {}
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v != int(bg):
                cells[(top + r, left + c)] = v
    return cells


def _overwrite_conflicts(cells_a, cells_b):
    return sum(1 for pos, col in cells_a.items() if pos in cells_b and int(cells_b[pos]) != int(col))


def _find_collision_positions(canvas, anchor, sprite_a, sprite_b, bg):
    pos_a = _aligned_positions(canvas, sprite_a, anchor, bg)
    pos_b = _aligned_positions(canvas, sprite_b, anchor, bg)
    if not pos_a or not pos_b:
        return None

    random.shuffle(pos_a)
    random.shuffle(pos_b)

    best = None
    best_score = 0
    cache_a = {p: _translated_cells(sprite_a, p, bg) for p in pos_a}
    cache_b = {p: _translated_cells(sprite_b, p, bg) for p in pos_b}
    for pa in pos_a:
        cells_a = cache_a[pa]
        for pb in pos_b:
            score = _overwrite_conflicts(cells_a, cache_b[pb])
            if score > best_score:
                best = (pa, pb)
                best_score = score
    if best_score < _MIN_OVERWRITE_CONFLICTS:
        return None
    return best


def _entry_order_key(entry):
    return tuple(entry["input_top_left"])


def _place_collision_pair(inp, first, second, bg):
    positions_first = _scatter_positions(inp, first["sprite"], first["anchor"], first["final_marker"], bg)
    random.shuffle(positions_first)
    for p1 in positions_first:
        top1 = _top_left(first["sprite"], p1[0], p1[1])
        temp = inp.copy()
        _place(temp, first["sprite"], p1[0], p1[1], bg)
        positions_second = _scatter_positions(temp, second["sprite"], second["anchor"], second["final_marker"], bg)
        random.shuffle(positions_second)
        valid_second = [p2 for p2 in positions_second if _top_left(second["sprite"], p2[0], p2[1]) > top1]
        if not valid_second:
            continue
        p2 = random.choice(valid_second)
        _place(inp, first["sprite"], p1[0], p1[1], bg)
        _place(inp, second["sprite"], p2[0], p2[1], bg)
        first["input_marker"] = p1
        second["input_marker"] = p2
        first["input_top_left"] = _top_left(first["sprite"], p1[0], p1[1])
        second["input_top_left"] = _top_left(second["sprite"], p2[0], p2[1])
        return True
    return False


def _render_output(shape, bg, anchors, entries):
    out = np.full(shape, int(bg), dtype=int)
    for anchor in anchors:
        _put_anchor(out, anchor["r"], anchor["c"], anchor["color"])
    for entry in sorted(entries, key=_entry_order_key):
        mr, mc = entry["final_marker"]
        _place(out, entry["sprite"], mr, mc, bg, overwrite=True)
    return out


def _overwrite_count(entries, bg):
    occupied = {}
    count = 0
    for entry in sorted(entries, key=_entry_order_key):
        for pos, col in _translated_cells(entry["sprite"], entry["final_marker"], bg).items():
            if pos in occupied and int(occupied[pos]) != int(col):
                count += 1
            occupied[pos] = int(col)
    return count


def _place_remaining_sprites(anchor_canvas, anchors, marker_colors, collision_color, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    entries = []

    collision_anchor = by_color[int(collision_color)]
    body_a = _body_color(bg, collision_color, used_body)
    used_body.add(body_a)
    body_b = _body_color(bg, collision_color, used_body)
    used_body.add(body_b)
    sprite_a = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_a, collision_color, bg)
    sprite_b = _make_sprite(random.randint(*_BODY_AREA_RANGE), body_b, collision_color, bg)
    collision_positions = _find_collision_positions(anchor_canvas, collision_anchor, sprite_a, sprite_b, bg)
    if collision_positions is None:
        return None

    pa, pb = collision_positions
    collision_entries = [
        {"sprite": sprite_a, "anchor": collision_anchor, "final_marker": pa},
        {"sprite": sprite_b, "anchor": collision_anchor, "final_marker": pb},
    ]
    entries.extend(collision_entries)

    reserved_out = anchor_canvas.copy()
    _place(reserved_out, sprite_a, pa[0], pa[1], bg, overwrite=True)
    _place(reserved_out, sprite_b, pb[0], pb[1], bg, overwrite=True)

    remaining_colors = marker_colors[:]
    remaining_colors.remove(collision_color)
    remaining_colors.remove(collision_color)
    random.shuffle(remaining_colors)
    for marker in remaining_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(reserved_out, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(reserved_out, sprite, marker_pos[0], marker_pos[1], bg, overwrite=True)
        entries.append({"sprite": sprite, "anchor": anchor, "final_marker": marker_pos})

    return entries


def _scatter_entries(inp, entries, bg):
    if len(entries) < 2:
        return False

    collision_pair = entries[:2]
    if random.random() < 0.5:
        collision_pair = collision_pair[::-1]
    if not _place_collision_pair(inp, collision_pair[0], collision_pair[1], bg):
        return False

    others = entries[2:]
    random.shuffle(others)
    for entry in others:
        positions = _scatter_positions(inp, entry["sprite"], entry["anchor"], entry["final_marker"], bg)
        if not positions:
            return False
        marker_pos = random.choice(positions)
        _place(inp, entry["sprite"], marker_pos[0], marker_pos[1], bg)
        entry["input_marker"] = marker_pos
        entry["input_top_left"] = _top_left(entry["sprite"], marker_pos[0], marker_pos[1])
    return True


def _add_distractors(out, inp, anchors, bg):
    vertical = [a for a in anchors if a["axis"] == "v"]
    if not vertical or random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(vertical)
    sprite = {"patch": np.array([[anchor["color"]]], dtype=int), "offset": (0, 0)}
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


def _generated_grid_is_valid(inp, out, bg, entries):
    if not _bg_is_mode(inp, bg):
        return False
    if _overwrite_count(entries, bg) < _MIN_OVERWRITE_CONFLICTS:
        return False
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False
    legacy = _solve(inp, overwrite_non_bg=False)
    return not np.array_equal(legacy, out)


def _gen_attempt():
    bg, out_seed, inp = _new_canvas_pair()
    placed = _place_anchors(out_seed, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    collision_color, marker_colors = _marker_color_sequence(anchor_colors)
    entries = _place_remaining_sprites(out_seed.copy(), anchors, marker_colors, collision_color, bg)
    if entries is None:
        return None

    if not _scatter_entries(inp, entries, bg):
        return None

    out = _render_output(inp.shape, bg, anchors, entries)
    _add_distractors(out, inp, anchors, bg)
    if not _generated_grid_is_valid(inp, out, bg, entries):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D5 variant input")
