# Variant of 581f7754
# Dimension: D1 — 匹配条件：标记色与锚点色匹配 (match_cond)
# Change: 精灵内部的少数色（marker）与边界锚点颜色相同即配对：s.marker.color == d.color... → 连接器颜色+方向匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for this D1 variant
# -----------------------------------
# variant_spec:
#   preserved_core:
#     - Sprites still slide by pure translation.
#     - Horizontal motion still means "align a sprite column to a border anchor column".
#     - Vertical motion still means "align a sprite row to a border anchor row".
#     - Clipping and non-overwrite behavior stay unchanged.
#   changed_core:
#     - A sprite no longer matches an anchor by color alone.
#     - Each sprite has a 1-pixel connector whose color is still the minority color, but its
#       direction now matters too. The direction is the outward side of the connector relative
#       to the sprite body: body->connector = left/right/up/down.
#     - Each border anchor is now a 2-pixel motif:
#         * one border connector pixel in the match color
#         * one adjacent direction pixel in a neutral direction-indicator color
#       The relative position of the indicator pixel encodes the direction.
# concrete_behavior_trace:
#   - Old rule:
#       sprite connector color=4 -> match the first border anchor colored 4.
#   - New rule:
#       sprite signature=(4, right) -> ignore a (4, left) anchor and match only the (4, right) anchor.
#       After the match is chosen, motion is computed exactly as before:
#         * left/right direction => horizontal slide so connector column matches anchor column
#         * up/down direction => vertical slide so connector row matches anchor row
#   - Difference:
#       repeated anchor colors are now safe and intentional, because direction disambiguates them.
# complexity_calibration:
#   - Every generated grid contains at least one repeated connector color used by two anchors with
#     opposite directions. This makes the new control variable observable and prevents regression
#     back to color-only behavior.
#   - The direction set is limited to the four cardinal directions; in one grid only 2-3 anchors
#     appear, so the scene remains legible.
# implicit_assumption_audit:
#   - Original anchor detection assumed an anchor was a 1-pixel isolated border component.
#     That is no longer true, so the solver must parse 2-pixel border components instead.
#   - Original matching used `anchors_by_color[color]`; this silently breaks once one color may
#     appear on multiple anchors. The key must change to `(color, direction)`.
#   - Original marker extraction only needed a minority color. That is insufficient now; the
#     minority pixel must also reveal a unique body-adjacent direction. The generator therefore
#     guarantees the connector is an exterior protrusion with exactly one body neighbor.
# placement_legality:
#   - Sprites remain fully interior with a 1-cell bg moat, so border-touching components can be
#     treated as anchors without ambiguity.
#   - Anchor motifs are isolated 2-pixel components on the border, separated by at least one bg
#     cell from everything else.
# visual_observability:
#   - The neutral direction-indicator color repeats across all anchors, so humans can notice that
#     the side of the extra pixel is meaningful.
#   - The repeated connector color across opposite directions ensures at least one sample-level
#     conflict that color-only matching cannot explain.
# cleanliness:
#   - Direction pixels live only inside anchor motifs; sprite connectors are single-pixel protrusions.
#     This avoids multi-layer overlaps and keeps anchors visually distinct from sprites.

_DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_VEC_TO_DIR = {v: k for k, v in _DIR_VECS.items()}
_H_DIRS = ("left", "right")
_V_DIRS = ("up", "down")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their connector pixels dock onto matching directional border anchors.

    Input structure:
    - bg = most common color.
    - Interior multi-color sprites; each has one minority-color connector pixel protruding from the body, and its protrusion side gives a direction.
    - Border anchors are isolated 2-pixel motifs: a border connector pixel plus an adjacent direction pixel.

    Rule:
    - Match each sprite to the anchor sharing both connector color and connector direction.
    - Left/right-facing anchors make the sprite slide horizontally so connector column = anchor column.
    - Up/down-facing anchors make the sprite slide vertically so connector row = anchor row.

    Key insight: The extra pixel attached to each border anchor encodes a facing, so the correct pairing is determined by the connector signature (color + direction), not by color alone.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def touches_border(coords: np.ndarray) -> bool:
        return any(is_border(int(r), int(c)) for r, c in coords)

    def sprite_connector_info(coords: np.ndarray) -> tuple[int, tuple[int, int], str] | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None

        connector_color = int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(connector_color)]
        if not pts:
            return None

        comp_set = {(int(r), int(c)) for r, c in coords}
        choices: list[tuple[tuple[int, int], str]] = []
        for r, c in pts:
            body_neighbors = []
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(connector_color):
                    body_neighbors.append((int(rr), int(cc)))
            if len(body_neighbors) != 1:
                continue
            br, bc = body_neighbors[0]
            direction = _VEC_TO_DIR.get((int(r - br), int(c - bc)))
            if direction is None:
                continue
            choices.append(((int(r), int(c)), direction))

        if not choices:
            return None

        connector_pixel, direction = choices[0]
        return int(connector_color), connector_pixel, direction

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    sprite_infos: list[tuple[np.ndarray, int, tuple[int, int], str]] = []
    border_comps: list[np.ndarray] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if touches_border(coords):
            border_comps.append(coords)
            continue
        info = sprite_connector_info(coords)
        if info is None:
            continue
        connector_color, connector_pixel, direction = info
        sprite_infos.append((coords, connector_color, connector_pixel, direction))

    assert sprite_infos

    sprite_connector_colors = {int(color) for _coords, color, _pixel, _direction in sprite_infos}
    anchors_by_sig: dict[tuple[int, str], tuple[int, int]] = {}

    for coords in border_comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = pts
        if abs(int(r0 - r1)) + abs(int(c0 - c1)) != 1:
            continue

        connector_candidates = [p for p in pts if int(g[p[0], p[1]]) in sprite_connector_colors]
        if len(connector_candidates) != 1:
            continue

        ar, ac = connector_candidates[0]
        other = pts[0] if pts[1] == (ar, ac) else pts[1]
        direction = _VEC_TO_DIR.get((int(other[0] - ar), int(other[1] - ac)))
        if direction is None:
            continue
        sig = (int(g[ar, ac]), direction)
        anchors_by_sig[sig] = (int(ar), int(ac))

    assert anchors_by_sig

    moves: list[tuple[np.ndarray, int, int]] = []
    for coords, connector_color, connector_pixel, direction in sprite_infos:
        sig = (int(connector_color), direction)
        if sig not in anchors_by_sig:
            continue

        ar, ac = anchors_by_sig[sig]
        dr = 0
        dc = 0
        if direction in _H_DIRS:
            dc = int(ac - connector_pixel[1])
        else:
            dr = int(ar - connector_pixel[0])
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


# --------------- generator (o5x) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = tuple(_DIR_VECS.values())
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_RETRIES = 60

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
_EXTRA_ANCHOR_PROB = 0.55

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


def _connector_candidates(body):
    r0, c0, r1, c1 = _bbox(body)
    candidates = []
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if (rr, cc) in body:
                continue
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            body_neighbors = []
            for nr, nc in neighbors4(int(rr), int(cc)):
                if (int(nr), int(nc)) in body:
                    body_neighbors.append((int(nr), int(nc)))
            if len(body_neighbors) != 1:
                continue
            direction = _VEC_TO_DIR[(int(rr - body_neighbors[0][0]), int(cc - body_neighbors[0][1]))]
            candidates.append(((int(rr), int(cc)), direction))
    uniq = {}
    for cell, direction in candidates:
        uniq[(cell, direction)] = (cell, direction)
    return list(uniq.values())


def _make_sprite(area, body_color, connector_color, direction, bg):
    for _ in range(_SPRITE_BUILD_RETRIES):
        body = _make_body(area)
        choices = [cell for cell, d in _connector_candidates(body) if d == direction]
        if not choices:
            continue
        connector = random.choice(choices)
        all_cells = set(body) | {connector}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[connector[0] - r0, connector[1] - c0] = int(connector_color)
        return {
            "patch": patch,
            "offset": (connector[0] - r0, connector[1] - c0),
            "direction": direction,
        }
    return None


def _top_left(sprite, connector_r, connector_c):
    off_r, off_c = sprite["offset"]
    return int(connector_r) - int(off_r), int(connector_c) - int(off_c)


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


def _can_place(canvas, sprite, connector_r, connector_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, connector_r, connector_c)
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


def _place(canvas, sprite, connector_r, connector_c, bg):
    top, left = _top_left(sprite, connector_r, connector_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _anchor_cells(anchor):
    dr, dc = _DIR_VECS[anchor["direction"]]
    return [(int(anchor["r"]), int(anchor["c"])), (int(anchor["r"] + dr), int(anchor["c"] + dc))]


def _anchor_ok(canvas, cells, bg):
    H, W = canvas.shape
    if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
        return False
    if any(canvas[r, c] != int(bg) for r, c in cells):
        return False
    keep = set(cells)
    for r, c in cells:
        r0, r1 = max(0, r - _ISOLATION_RADIUS), min(H, r + _ISOLATION_RADIUS + 1)
        c0, c1 = max(0, c - _ISOLATION_RADIUS), min(W, c + _ISOLATION_RADIUS + 1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if (rr, cc) in keep:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return False
    return True


def _anchor_candidates(canvas, direction, bg):
    H, W = canvas.shape
    out = []
    if direction == "left":
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - 1))
        rows = [0, H - 1]
        for r in rows:
            for c in cols:
                anchor = {"r": int(r), "c": int(c), "direction": direction}
                cells = _anchor_cells(anchor)
                if min(cc for _rr, cc in cells) <= 0:
                    continue
                if _anchor_ok(canvas, cells, bg):
                    out.append(anchor)
    elif direction == "right":
        cols = list(range(1, W - _ANCHOR_EDGE_MARGIN))
        rows = [0, H - 1]
        for r in rows:
            for c in cols:
                anchor = {"r": int(r), "c": int(c), "direction": direction}
                cells = _anchor_cells(anchor)
                if max(cc for _rr, cc in cells) >= W - 1:
                    continue
                if _anchor_ok(canvas, cells, bg):
                    out.append(anchor)
    elif direction == "up":
        rows = list(range(_ANCHOR_EDGE_MARGIN, H - 1))
        cols = [0, W - 1]
        for r in rows:
            for c in cols:
                anchor = {"r": int(r), "c": int(c), "direction": direction}
                cells = _anchor_cells(anchor)
                if min(rr for rr, _cc in cells) <= 0:
                    continue
                if _anchor_ok(canvas, cells, bg):
                    out.append(anchor)
    else:
        rows = list(range(1, H - _ANCHOR_EDGE_MARGIN))
        cols = [0, W - 1]
        for r in rows:
            for c in cols:
                anchor = {"r": int(r), "c": int(c), "direction": direction}
                cells = _anchor_cells(anchor)
                if max(rr for rr, _cc in cells) >= H - 1:
                    continue
                if _anchor_ok(canvas, cells, bg):
                    out.append(anchor)
    random.shuffle(out)
    return out


def _put_anchor(canvas, anchor, dir_color):
    dr, dc = _DIR_VECS[anchor["direction"]]
    canvas[int(anchor["r"]), int(anchor["c"])] = int(anchor["color"])
    canvas[int(anchor["r"] + dr), int(anchor["c"] + dc)] = int(dir_color)


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


def _scatter_positions(canvas, sprite, anchor, connector_pos, bg):
    H, W = canvas.shape
    cr, cc = connector_pos
    if anchor["axis"] == "h":
        return [
            (cr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != cc and _can_place(canvas, sprite, cr, c, bg)
        ]
    return [
        (r, cc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != cr and _can_place(canvas, sprite, r, cc, bg)
    ]


def _body_color(bg, connector_color, dir_color, used):
    banned = {int(bg), int(connector_color), int(dir_color)}
    pool = [c for c in _COLORS if c not in banned and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in banned]
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


def _choose_anchor_signatures(bg):
    pool = [c for c in _COLORS if c != int(bg)]
    repeat_color = random.choice(pool)
    if random.random() < 0.5:
        repeated_dirs = ["left", "right"]
    else:
        repeated_dirs = ["up", "down"]
    signatures = [(int(repeat_color), d) for d in repeated_dirs]
    if random.random() < _EXTRA_ANCHOR_PROB:
        extra_pool = [c for c in pool if c != int(repeat_color)]
        if extra_pool:
            extra_color = random.choice(extra_pool)
            extra_dir = random.choice(list(_DIR_VECS))
            signatures.append((int(extra_color), extra_dir))
    return signatures


def _place_anchors(out, inp, bg):
    signatures = _choose_anchor_signatures(bg)
    connector_colors = {int(color) for color, _direction in signatures}
    dir_pool = [c for c in _COLORS if c not in connector_colors and c != int(bg)]
    if not dir_pool:
        return None
    dir_color = random.choice(dir_pool)

    anchors = []
    for color, direction in signatures:
        candidates = _anchor_candidates(out, direction, bg)
        if not candidates:
            return None
        anchor = random.choice(candidates)
        anchor = {
            "color": int(color),
            "direction": direction,
            "r": int(anchor["r"]),
            "c": int(anchor["c"]),
            "axis": "h" if direction in _H_DIRS else "v",
        }
        anchors.append(anchor)
        _put_anchor(out, anchor, dir_color)
        _put_anchor(inp, anchor, dir_color)
    return anchors, dir_color


def _sprite_signature_sequence(anchors):
    n_sprites = random.randint(max(3, len(anchors)), _SPRITE_COUNT_RANGE[1])
    signatures = [(int(a["color"]), a["direction"]) for a in anchors]
    out = signatures[:]
    while len(out) < n_sprites:
        out.append(random.choice(signatures))
    random.shuffle(out)
    return out


def _place_aligned_sprites(out, anchors, sprite_signatures, dir_color, bg):
    by_sig = {(int(a["color"]), a["direction"]): a for a in anchors}
    used_body = set()
    sprites = []
    for connector_color, direction in sprite_signatures:
        anchor = by_sig[(int(connector_color), direction)]
        body = _body_color(bg, connector_color, dir_color, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, connector_color, direction, bg)
        if sprite is None:
            return None
        positions = _aligned_positions(out, sprite, anchor, bg)
        if not positions:
            return None
        connector_pos = random.choice(positions)
        _place(out, sprite, connector_pos[0], connector_pos[1], bg)
        sprites.append((sprite, anchor, connector_pos))
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, connector_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, connector_pos, bg)
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
    anchors, dir_color = placed

    sprite_signatures = _sprite_signature_sequence(anchors)
    sprites = _place_aligned_sprites(out, anchors, sprite_signatures, dir_color, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D1 input")
