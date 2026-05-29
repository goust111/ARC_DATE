# Combo variant of 581f7754 (index 13)
# combo_spec: D1/2+D2/2+D5/3
# Name: 连接器方向匹配+颜色编码轴+到达后重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Base task:
#      isolated border singleton color -> same-colored sprite marker -> edge chooses slide axis.
#    - Combo task:
#      each sprite has a 1-pixel connector whose color is still the minority color, but it also
#      has a facing determined by body->connector = left/right/up/down.
#      Each anchor is now a 2-pixel border motif:
#        * a border pixel in the connector color
#        * one neutral indicator pixel immediately inside the grid
#      The indicator does two jobs:
#        * together with the border pixel it encodes the anchor facing
#        * its row/col is the docking coordinate that the sprite connector aligns to
#      The facing is read from indicator -> border pixel, so a left-edge motif encodes "left",
#      a right-edge motif encodes "right", etc. This keeps the sprite body inside the grid after
#      docking even when the slide axis contradicts the old edge-based rule.
#      Finally, after the move, body pixels recolor to the anchor color and the connector pixel
#      disappears to background.
#    - Small trace:
#      warm anchor color 2 on the left edge: cells [(6,0)=2, (6,1)=dir].
#      Facing = left, because the inner pixel points toward the border pixel.
#      A sprite whose connector color=2 and facing=left matches it.
#      Since 2 is warm, the sprite slides horizontally, so connector column becomes 1
#      (the indicator column), not vertically as in the original task.
#      After landing, the connector vanishes and the remaining body recolors to 2.
#
# b) Complexity calibration
#    - Every generated grid contains:
#        * one warm color used by two opposite-direction anchors on left/right edges
#        * one cool anchor on the top or bottom edge
#      This makes direction matching necessary and makes the color-coded axis visibly contradict
#      the original edge-based rule in both color families.
#    - Sprites are 4-5 per grid, with at least one sprite for every anchor signature, so the
#      repeated warm color cannot be ignored.
#    - Solved body shapes are kept unique to prevent recolored outputs from collapsing into
#      ambiguous same-color blobs.
#
# c) Implicit-assumption review of the original main()
#    - Original anchor parser assumed 1-pixel border singletons. That breaks immediately; the
#      solver must now parse 2-pixel border components and distinguish the color pixel from the
#      neutral indicator.
#    - Original matching key was just marker color. That is no longer sufficient because two
#      anchors intentionally share the same warm color. The matching key becomes (color, facing).
#    - Original axis branch depended on anchor edge. The combo replaces that with a color-family
#      lookup: warm colors -> horizontal, cool colors -> vertical.
#    - Original paste loop preserved source colors. That would destroy the recolor rule, so the
#      destination renderer must treat connector pixels as disappearing and repaint all remaining
#      sprite pixels to the anchor color.
#
# d) Placement legality
#    - Output-first generation is preserved.
#    - The solved state places the connector on the anchor's inner indicator cell, not on the
#      border color cell. This leaves the anchor motif intact and keeps the recolored body inside.
#    - Solved bodies may touch their own anchor motif, but placement rejects overlap/contact with
#      other objects. Scattered input sprites keep a 1-cell moat from everything.
#
# e) Visual observability
#    - The warm repeated color on left/right edges disproves color-only matching.
#    - Warm anchors live on left/right edges and the cool anchor lives on top/bottom, so a solver
#      that still uses the original edge->axis rule gives the wrong motion on nearly every sample.
#    - The indicator color is shared across anchors, making the tiny 2-pixel motif readable as a
#      structural marker rather than as unrelated decoration.
#
# f) Visual cleanliness
#    - Only 3 anchors are used, and the solved sprites stay inside the frame instead of relying on
#      heavy clipping. The recolored body remains visually separate from the border color pixel by
#      the neutral indicator cell.
#    - Distractor singletons are disabled; otherwise they would simply vanish after recoloring and
#      add noise without carrying the new rule.
#
# g) Change inferability
#    - The color-family mapping uses a small, human-visible palette split:
#        warm = {2 red, 4 yellow, 7 orange} -> horizontal
#        cool = {1 blue, 3 green, 6 pink/purple} -> vertical
#    - The recolor is local and structural: a sprite's body becomes the matched anchor color, and
#      the connector pixel disappears. Because each grid has both warm and cool evidence plus
#      repeated warm signatures, the fusion rule is recoverable from a few examples.
#
# Assumption audit for the new main()
#    - Border anchors are exactly 2-pixel components and use a neutral indicator color that is not
#      in the warm/cool anchor lookup. The generator guarantees this.
#    - Every movable sprite has exactly one connector pixel whose 4-neighborhood identifies a
#      unique facing. The generator constructs connectors that satisfy that uniqueness.
#    - Every anchor signature appears in at least one sprite. The generator enforces this so
#      repeated-color directional matching is behaviorally active, not just present in decoration.
#    - Solved bodies are disjoint except for allowed contact with their own anchor motif. The
#      generator checks those contacts explicitly when placing solved patches.


_DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_VEC_TO_DIR = {v: k for k, v in _DIR_VECS.items()}
_WARM_COLORS = frozenset({2, 4, 7})
_COOL_COLORS = frozenset({1, 3, 6})
_AXIS_BY_COLOR = {c: "h" for c in _WARM_COLORS} | {c: "v" for c in _COOL_COLORS}
_EDGE_TO_DIRECTION = {"left": "left", "right": "right", "top": "up", "bottom": "down"}


def _axis_from_color(color: int) -> str | None:
    return _AXIS_BY_COLOR.get(int(color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide connector-marked sprites toward directional border anchors, then repaint them.

    Input structure:
    - Background is the most frequent color.
    - Interior sprites have one body color plus one minority connector pixel whose outward side defines a facing.
    - Border anchors are isolated 2-pixel motifs: a border pixel in the connector color and one adjacent inner indicator pixel.

    Rule:
    - Match each sprite to the anchor sharing both connector color and facing; the anchor facing is read from the inner indicator pixel toward the border color pixel.
    - Warm anchor colors (red / yellow / orange) move the sprite horizontally so the connector reaches the indicator column; cool anchor colors (blue / green / pink-purple) move it vertically so the connector reaches the indicator row.
    - After the slide, repaint every non-connector sprite pixel to the anchor color and erase the connector pixel to background.

    Key insight: The border motif provides two independent cues at once: its color family chooses the slide axis, while its tiny two-pixel orientation disambiguates same-colored anchors by facing.
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
        for r, c in pts:
            body_neighbors = []
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in comp_set and int(g[int(rr), int(cc)]) != int(connector_color):
                    body_neighbors.append((int(rr), int(cc)))
            if len(body_neighbors) != 1:
                continue
            br, bc = body_neighbors[0]
            direction = _VEC_TO_DIR.get((int(r - br), int(c - bc)))
            if direction is not None:
                return int(connector_color), (int(r), int(c)), direction
        return None

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

    anchors_by_sig: dict[tuple[int, str], tuple[int, int]] = {}
    sprite_connector_colors = {int(color) for _coords, color, _pixel, _direction in sprite_infos}

    for coords in border_comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = pts
        if abs(int(r0 - r1)) + abs(int(c0 - c1)) != 1:
            continue

        border_pts = [p for p in pts if is_border(*p)]
        if len(border_pts) != 1:
            continue
        color_pt = border_pts[0]
        inner_pt = pts[0] if pts[1] == color_pt else pts[1]
        color = int(g[color_pt[0], color_pt[1]])
        if color not in sprite_connector_colors:
            continue
        if _axis_from_color(color) is None:
            continue

        direction = _VEC_TO_DIR.get((int(color_pt[0] - inner_pt[0]), int(color_pt[1] - inner_pt[1])))
        if direction is None:
            continue
        anchors_by_sig[(color, direction)] = (int(inner_pt[0]), int(inner_pt[1]))

    assert anchors_by_sig

    moves: list[dict[str, object]] = []
    for coords, connector_color, connector_pixel, direction in sprite_infos:
        sig = (int(connector_color), direction)
        if sig not in anchors_by_sig:
            continue
        axis = _axis_from_color(int(connector_color))
        if axis is None:
            continue

        tr, tc = anchors_by_sig[sig]
        dr = 0
        dc = 0
        if axis == "h":
            dc = int(tc - connector_pixel[1])
        else:
            dr = int(tr - connector_pixel[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "anchor_color": int(connector_color),
                "connector_pixel": tuple(map(int, connector_pixel)),
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
        anchor_color = int(move["anchor_color"])
        connector_pixel = move["connector_pixel"]
        for r, c in coords:
            src = (int(r), int(c))
            if src == connector_pixel:
                continue
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(anchor_color)

    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = tuple(_DIR_VECS.values())
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_SPRITE_COUNT_RANGE = (4, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500
_SPRITE_BUILD_RETRIES = 80
_SPRITE_PLACE_RETRIES = 80

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
            "offset": (int(connector[0] - r0), int(connector[1] - c0)),
            "direction": direction,
            "body_signature": frozenset(_norm(body)),
        }
    return None


def _solved_patch(sprite, anchor_color, bg):
    patch = np.array(sprite["patch"], copy=True)
    mask = patch != int(bg)
    patch[mask] = int(anchor_color)
    off_r, off_c = sprite["offset"]
    patch[int(off_r), int(off_c)] = int(bg)
    return patch


def _top_left(sprite, connector_r, connector_c):
    off_r, off_c = sprite["offset"]
    return int(connector_r) - int(off_r), int(connector_c) - int(off_c)


def _patch_cells(sprite, connector_r, connector_c, bg, solved=False, anchor_color=None):
    patch = _solved_patch(sprite, int(anchor_color), bg) if solved else sprite["patch"]
    top, left = _top_left(sprite, connector_r, connector_c)
    cells = []
    for pr, pc in np.argwhere(patch != int(bg)):
        cells.append((top + int(pr), left + int(pc), int(patch[int(pr), int(pc)])))
    return cells


def _neighbor_coords(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r) + dr, int(c) + dc


def _can_place(canvas, sprite, connector_r, connector_c, bg, *, solved=False, anchor_color=None, border_gap=1, allow_contact=()):
    patch = _solved_patch(sprite, int(anchor_color), bg) if solved else sprite["patch"]
    top, left = _top_left(sprite, connector_r, connector_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < border_gap or left < border_gap:
        return False
    if top + h > H - border_gap or left + w > W - border_gap:
        return False

    allowed = {(int(r), int(c)) for r, c in allow_contact}
    placed = {(r, c): v for r, c, v in _patch_cells(sprite, connector_r, connector_c, bg, solved=solved, anchor_color=anchor_color)}

    for (r, c), v in placed.items():
        cell = int(canvas[r, c])
        if cell != int(bg):
            if (r, c) not in allowed or cell != int(v):
                return False

    for r, c in placed:
        for rr, cc in _neighbor_coords(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if int(canvas[rr, cc]) == int(bg):
                continue
            if (rr, cc) in allowed:
                continue
            return False
    return True


def _place(canvas, sprite, connector_r, connector_c, bg, *, solved=False, anchor_color=None):
    patch = _solved_patch(sprite, int(anchor_color), bg) if solved else sprite["patch"]
    top, left = _top_left(sprite, connector_r, connector_c)
    paste(canvas, patch, top, left, transparent=int(bg))


def _anchor_cells(anchor):
    return [(int(anchor["r"]), int(anchor["c"])), (int(anchor["tr"]), int(anchor["tc"]))]


def _anchor_ok(canvas, anchor, bg):
    H, W = canvas.shape
    cells = _anchor_cells(anchor)
    if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
        return False
    if any(int(canvas[r, c]) != int(bg) for r, c in cells):
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


def _anchor_candidates(canvas, edge, bg):
    H, W = canvas.shape
    out = []
    if edge == "left":
        rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
        random.shuffle(rows)
        for r in rows:
            anchor = {
                "edge": edge,
                "direction": "left",
                "r": int(r),
                "c": 0,
                "tr": int(r),
                "tc": 1,
            }
            if _anchor_ok(canvas, anchor, bg):
                out.append(anchor)
    elif edge == "right":
        rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
        random.shuffle(rows)
        for r in rows:
            anchor = {
                "edge": edge,
                "direction": "right",
                "r": int(r),
                "c": W - 1,
                "tr": int(r),
                "tc": W - 2,
            }
            if _anchor_ok(canvas, anchor, bg):
                out.append(anchor)
    elif edge == "top":
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
        random.shuffle(cols)
        for c in cols:
            anchor = {
                "edge": edge,
                "direction": "up",
                "r": 0,
                "c": int(c),
                "tr": 1,
                "tc": int(c),
            }
            if _anchor_ok(canvas, anchor, bg):
                out.append(anchor)
    else:
        cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
        random.shuffle(cols)
        for c in cols:
            anchor = {
                "edge": edge,
                "direction": "down",
                "r": H - 1,
                "c": int(c),
                "tr": H - 2,
                "tc": int(c),
            }
            if _anchor_ok(canvas, anchor, bg):
                out.append(anchor)
    return out


def _put_anchor(canvas, anchor, dir_color):
    canvas[int(anchor["r"]), int(anchor["c"])] = int(anchor["color"])
    canvas[int(anchor["tr"]), int(anchor["tc"])] = int(dir_color)


def _aligned_positions(canvas, sprite, anchor, bg):
    allow_contact = _anchor_cells(anchor)
    if anchor["axis"] == "h":
        return [
            (r, anchor["tc"])
            for r in range(_INNER_BORDER_GAP, canvas.shape[0] - _INNER_BORDER_GAP)
            if _can_place(
                canvas,
                sprite,
                r,
                anchor["tc"],
                bg,
                solved=True,
                anchor_color=anchor["color"],
                border_gap=_INNER_BORDER_GAP,
                allow_contact=allow_contact,
            )
        ]
    return [
        (anchor["tr"], c)
        for c in range(_INNER_BORDER_GAP, canvas.shape[1] - _INNER_BORDER_GAP)
        if _can_place(
            canvas,
            sprite,
            anchor["tr"],
            c,
            bg,
            solved=True,
            anchor_color=anchor["color"],
            border_gap=_INNER_BORDER_GAP,
            allow_contact=allow_contact,
        )
    ]


def _scatter_positions(canvas, sprite, anchor, connector_pos, bg):
    H, W = canvas.shape
    cr, cc = connector_pos
    if anchor["axis"] == "h":
        return [
            (cr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != cc and _can_place(canvas, sprite, cr, c, bg, border_gap=_INNER_BORDER_GAP)
        ]
    return [
        (r, cc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != cr and _can_place(canvas, sprite, r, cc, bg, border_gap=_INNER_BORDER_GAP)
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


def _choose_anchor_colors(bg):
    warm_pool = [c for c in _WARM_COLORS if c != int(bg)]
    cool_pool = [c for c in _COOL_COLORS if c != int(bg)]
    if not warm_pool or not cool_pool:
        return None
    return int(random.choice(warm_pool)), int(random.choice(cool_pool))


def _place_anchors(out, inp, bg):
    picked = _choose_anchor_colors(bg)
    if picked is None:
        return None
    warm_color, cool_color = picked

    dir_pool = [c for c in _COLORS if c != int(bg) and c not in (_WARM_COLORS | _COOL_COLORS)]
    if not dir_pool:
        return None
    dir_color = int(random.choice(dir_pool))

    cool_edge = random.choice(["top", "bottom"])
    anchor_specs = [
        {"color": warm_color, "edge": "left", "direction": "left", "axis": "h"},
        {"color": warm_color, "edge": "right", "direction": "right", "axis": "h"},
        {"color": cool_color, "edge": cool_edge, "direction": _EDGE_TO_DIRECTION[cool_edge], "axis": "v"},
    ]

    anchors = []
    for spec in anchor_specs:
        candidates = _anchor_candidates(out, spec["edge"], bg)
        if not candidates:
            return None
        base = random.choice(candidates)
        anchor = {
            "color": int(spec["color"]),
            "edge": spec["edge"],
            "direction": spec["direction"],
            "axis": spec["axis"],
            "r": int(base["r"]),
            "c": int(base["c"]),
            "tr": int(base["tr"]),
            "tc": int(base["tc"]),
        }
        anchors.append(anchor)
        _put_anchor(out, anchor, dir_color)
        _put_anchor(inp, anchor, dir_color)
    return anchors, dir_color


def _sprite_signature_sequence(anchors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    signatures = [(int(a["color"]), a["direction"]) for a in anchors]
    out = signatures[:]
    while len(out) < n_sprites:
        out.append(random.choice(signatures))
    random.shuffle(out)
    return out


def _place_aligned_sprites(out, anchors, sprite_signatures, dir_color, bg):
    by_sig = {(int(a["color"]), a["direction"]): a for a in anchors}
    used_body_colors = set()
    used_shapes = set()
    sprites = []

    for connector_color, direction in sprite_signatures:
        anchor = by_sig[(int(connector_color), direction)]
        placed = None
        for _ in range(_SPRITE_PLACE_RETRIES):
            body = _body_color(bg, connector_color, dir_color, used_body_colors)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, connector_color, direction, bg)
            if sprite is None:
                continue
            if sprite["body_signature"] in used_shapes:
                continue
            positions = _aligned_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            connector_pos = positions[0]
            _place(out, sprite, connector_pos[0], connector_pos[1], bg, solved=True, anchor_color=connector_color)
            placed = (body, sprite, anchor, connector_pos)
            break
        if placed is None:
            return None
        body, sprite, anchor, connector_pos = placed
        used_body_colors.add(body)
        used_shapes.add(sprite["body_signature"])
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
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
