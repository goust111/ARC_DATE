# Variant of 581f7754
# Dimension: D2 — 匹配条件：滑动轴决定规则 (match_cond)
# Change: 锚点在上/下边缘（非角落）→水平滑动对齐列；锚点在左/右边缘（或角落）→垂直滑动对齐行... → 标记位置决定轴

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D2 variant
# ==================================
# a) Concrete behavior and trace
#    - Objects remain the same high-level types: isolated border anchors plus interior multi-color sprites.
#    - Each sprite still matches the anchor that shares its minority marker color.
#    - The changed part is only the movement-axis decision:
#      old rule: anchor on top/bottom => horizontal move; anchor on left/right => vertical move.
#      new rule: marker above/below the sprite body => vertical move; marker left/right of the body => horizontal move.
#    - Small trace:
#      old: a red anchor on the top border forced a red-marked sprite to move horizontally until marker column = anchor column.
#      new: the same top-border red anchor only supplies the target row/column coordinate; if the sprite's red marker sits above the body,
#      the sprite now moves vertically until marker row = anchor row. The anchor is still needed, but it no longer chooses the axis.
#
# b) Complexity calibration
#    - To make the rule visible, markers are restricted to one of four exterior side slots relative to the body: top / bottom / left / right.
#    - Generator forces both anchor edge families (top-bottom and left-right) to appear in each sample.
#    - Some sprites use the marker on the same edge as their anchor (this flips the original axis), while others use an orthogonal side
#      with the same anchor (this preserves the original axis). The mixture is deliberate: anchor position alone is no longer sufficient.
#    - Sprites still reuse anchor colors, so the task is not reduced to one-off matching.
#
# c) Hidden assumptions in the original main()
#    - Anchor detection remains valid: anchors are still isolated border singletons.
#    - Marker-color detection by minority count remains valid because each generated sprite still has one marker pixel.
#    - The original translation code assumed the axis was derivable from anchor coordinates alone; that is now false and must be replaced.
#    - Original singleton distractors depended on anchor-chosen axis. Under the new rule a lone pixel has no body-relative side, so the
#      variant removes these distractors instead of inventing an arbitrary fallback.
#
# d) Placement legality
#    - Output-first generation is preserved.
#    - Solved output places every sprite so either marker row = anchor row (vertical movers) or marker column = anchor column (horizontal movers).
#    - Input is created by scattering each sprite along the orthogonal axis, guaranteeing a non-zero move and preserving non-overlap checks.
#    - Sprites remain at least one cell away from borders and other foreground, so solver component extraction is stable.
#
# e) Visual observability
#    - Marker side is always visibly external to the body; no hollow/interior markers are used in this variant.
#    - Because anchors appear on both border families while axis is decided by sprite structure, a human can see examples where a top-border
#      anchor leads to vertical motion and a left-border anchor leads to horizontal motion, which is the key contrast from the original task.
#
# f) Visual cleanliness
#    - Objects stay sparse with 8-neighborhood gaps, so no component-merging or layered clutter occurs.
#    - Marker/body colors remain simple two-color sprites.
#
# g) Color reasoning
#    - Colors only encode sprite-anchor pairing. No ordering, remapping, or color-id arithmetic is introduced.
#
# h) Post-change solver assumptions audited
#    - main() now assumes every movable sprite has at least one non-marker body pixel and its marker lies outside the body's bbox on exactly
#      one cardinal side. generate_input() enforces this by constructing the marker from explicit side pools.
#    - main() ignores single-color interior components. generate_input() produces none besides border anchors.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-color sprites to align their marker pixels with matching isolated border anchors.

    Input structure:
    - The background is the most frequent color.
    - Interior sprites are 8-connected two-color components with one minority-color marker pixel attached to a larger body.
    - Border anchors are isolated non-background singletons, and each marker color matches one anchor color.

    Rule:
    - Pair every sprite with the border anchor that has the same color as its marker.
    - If the marker sits above or below the sprite body, slide the whole sprite vertically so the marker row matches the anchor row.
    - If the marker sits to the left or right of the sprite body, slide the whole sprite horizontally so the marker column matches the anchor column.
    - Source cells are cleared first; shifted pixels that leave the grid are clipped, and existing non-background cells are not overwritten.

    Key insight: The anchor still tells you which color to match and which row or column to hit, but the sprite's own marker side determines the move axis.
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
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        body = {(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker)}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in body:
                    return (int(r), int(c))
        return pts[0]

    def marker_axis(coords: np.ndarray, marker: int, mp: tuple[int, int]) -> str | None:
        body = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker)]
        if not body:
            return None
        r0 = min(r for r, _c in body)
        c0 = min(c for _r, c in body)
        r1 = max(r for r, _c in body)
        c1 = max(c for _r, c in body)
        mr, mc = mp
        if mr < r0 or mr > r1:
            return "v"
        if mc < c0 or mc > c1:
            return "h"

        attached = [(rr, cc) for rr, cc in neighbors4(mr, mc) if (int(rr), int(cc)) in set(body)]
        if not attached:
            return None
        rows = {rr for rr, _cc in attached}
        cols = {cc for _rr, cc in attached}
        if all(rr > mr for rr in rows) or all(rr < mr for rr in rows):
            return "v"
        if all(cc > mc for cc in cols) or all(cc < mc for cc in cols):
            return "h"
        return None

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None:
            continue
        if int(marker_col) not in anchors_by_color:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None or mp in anchor_cells:
            continue

        axis = marker_axis(coords, int(marker_col), mp)
        if axis is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        dr = 0
        dc = 0
        if axis == "h":
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
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
_GEN_RETRY_LIMIT = 300

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
    return "tb" if edge in ("top", "bottom") else "lr"


def _orthogonal_sides(anchor):
    return _LR_SIDES if _edge_family(anchor["edge"]) == "tb" else _TB_SIDES


def _marker_side_options(anchor, mode):
    if mode == "same_edge":
        return [anchor["edge"]]
    return list(_orthogonal_sides(anchor))


def _make_sprite(area, body_color, marker_color, bg, marker_side):
    body = _make_body(area)
    pools = _marker_cells_by_side(body)
    options = pools.get(marker_side, [])
    if not options:
        return None
    marker = random.choice(options)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "marker_side": marker_side,
        "move_axis": _SIDE_TO_AXIS[marker_side],
    }


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _visible_sprite_cells(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    cells = []
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v == int(bg):
                continue
            cells.append((top + r, left + c, v))
    return cells, top, left, h, w


def _can_place(canvas, sprite, marker_r, marker_c, bg, *, allow_border):
    H, W = canvas.shape
    cells, top, left, h, w = _visible_sprite_cells(canvas, sprite, marker_r, marker_c, bg)
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


def _scatter_positions(canvas, sprite, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if sprite["move_axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg, allow_border=False)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg, allow_border=False)
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
        anchor = {"color": int(color), "r": int(r), "c": int(c), "edge": edge}
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

    for marker, mode in zip(marker_colors, modes):
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)

        side_options = _marker_side_options(anchor, mode)
        random.shuffle(side_options)
        sprite = None
        for marker_side in side_options:
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg, marker_side)
            if sprite is not None:
                positions = _aligned_positions(out, sprite, anchor, bg)
                if positions:
                    marker_pos = random.choice(positions)
                    _place(out, sprite, marker_pos[0], marker_pos[1], bg)
                    sprites.append((sprite, marker_pos))
                    break
        else:
            return None
    return sprites


def _scatter_sprites(inp, sprites, bg):
    random.shuffle(sprites)
    for sprite, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, marker_pos, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D2 input")
