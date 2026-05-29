# Variant of 581f7754
# Dimension: D17 — 符号编码：锚点颜色附加信息 (symbol_instructions)
# Change: 锚点颜色仅用于配对匹配，不编码其他信息... → 图例区域定义颜色配对规则

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D17 variant
# a) Concrete behavior:
#    The base task matched each sprite's minority marker color directly to an isolated border anchor
#    of the same color. The variant inserts a compact legend into one grid corner. The legend is a
#    4-column framed block whose inner rows read as [anchor color, marker color]. Border anchors keep
#    their original colors, while sprites now carry the corresponding marker color from the legend's
#    right column. Example: if the legend row is [2, 6], then a sprite whose minority pixel is 6 must
#    slide toward the border anchor colored 2. The slide axis is unchanged: top/bottom anchors enforce
#    column alignment, left/right anchors enforce row alignment.
#    A concrete trace:
#    - Original: anchor 2 on the top edge, sprite marker also 2. Solver finds marker color 2, picks
#      anchor 2, then shifts the sprite horizontally until the marker column matches the anchor column.
#    - Variant: legend row says 2 -> 6, top-edge anchor is still 2, sprite marker is now 6. Solver
#      must first decode the legend, invert it to 6 -> 2, and only then do the same horizontal slide.
#    - Difference: the movement geometry is identical, but color lookup is indirect and therefore the
#      legend must be excluded from both anchor detection and movable-sprite extraction.
# b) Complexity calibration:
#    The legend remains intentionally tiny: 2-3 mapping rows, one-to-one, no branching and no counts.
#    This is enough to make the change visible while avoiding combinatorial ambiguity. I force the
#    mapping to be a derangement on the anchor colors, so direct color equality is never valid.
# c) Hidden assumptions in the original main():
#    - "anchor color == marker color" no longer holds; every line that used marker_col as the anchor
#      lookup key must now go through marker_to_anchor.
#    - "all non-bg connected components are either sprites or anchors" no longer holds because the
#      corner legend is also a non-bg multicolor component.
#    - "any isolated border pixel of a relevant color is an anchor" remains valid, but the legend must
#      not create fake anchors. A solid framed legend touching the corner keeps its border pixels
#      non-isolated, so anchor detection stays clean.
#    - "single-color singleton distractors use anchor colors" no longer holds; in this variant they
#      use marker colors because marker colors are the symbolic codes.
# d) Placement legality:
#    The generator pastes the legend first so later placement code naturally treats it as occupied.
#    Anchors still require an isolated 8-neighborhood. Sprites keep a one-cell moat from all existing
#    non-bg cells, so they cannot merge with anchors or the legend. The output-first workflow is kept:
#    place legend + anchors + solved sprites in the output canvas, then scatter the sprites along the
#    inverse axis into the input canvas and validate with main().
# e) Visual observability:
#    The legend is always present, always in a corner, and always nontrivial because every pair swaps
#    colors. Human solvers can see that the markers use colors drawn from the legend's right column,
#    while the border anchors use the left column. The slide axis remains easy to infer from edge
#    placement, so the only new concept to infer is the legend lookup.
# f) Visual cleanliness:
#    The legend is a compact 4-column framed box with no overlap and no moving parts. Body colors avoid
#    all anchor/marker colors, which makes the marker pixel visually salient instead of blending into
#    other semantic colors. The existing sprite spacing rules are reused unchanged.
# g) Color-rule inferability:
#    The number of mapped colors is small (2-3), repeated across several sprites per grid, and the
#    mapping is spatially documented by the legend rows. No rule depends on numeric color ids or hidden
#    ordering; everything needed for the color transformation is visible in the corner legend itself.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite so its marker reaches the border anchor named by the corner legend.

    Input structure:
    - The background is still the most frequent color.
    - One corner contains a compact framed legend whose inner rows encode [anchor color, marker color].
    - Elsewhere, isolated border pixels are anchors and interior multi-color sprites carry the marker colors from the legend.

    Rule:
    - Decode the legend into anchor-color -> marker-color pairs, then invert it so a sprite marker can identify its anchor.
    - Move each sprite (and eligible singleton distractor) along the same axis as in the base task: top/bottom anchors align columns, left/right anchors align rows.
    - Keep the legend and anchors fixed; clip pixels that move out of bounds and never overwrite existing non-background cells.

    Key insight: The edge position still chooses the translation axis, but color matching is no longer direct — the corner legend translates marker colors into anchor colors first.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    # --- STEP 0) Find and decode the corner legend. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    corners = {(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))}

    legend_comp = None
    for comp in comps:
        cells = {(int(r), int(c)) for r, c in comp["coords"]}
        if any(cell in corners for cell in cells):
            if legend_comp is None or int(comp["coords"].shape[0]) > int(legend_comp["coords"].shape[0]):
                legend_comp = comp

    assert legend_comp is not None

    legend_coords = legend_comp["coords"]
    legend_cells = {(int(r), int(c)) for r, c in legend_coords}
    r0 = int(legend_coords[:, 0].min())
    c0 = int(legend_coords[:, 1].min())
    r1 = int(legend_coords[:, 0].max())
    c1 = int(legend_coords[:, 1].max())
    legend = g[r0 : r1 + 1, c0 : c1 + 1]
    frame = int(legend[0, 0])

    anchor_to_marker: dict[int, int] = {}
    for rr in range(1, int(legend.shape[0]) - 1):
        if int(legend.shape[1]) < 3:
            continue
        anchor_col = int(legend[int(rr), 1])
        marker_col = int(legend[int(rr), 2])
        if anchor_col in {int(bg), int(frame)} or marker_col in {int(bg), int(frame)}:
            continue
        anchor_to_marker[int(anchor_col)] = int(marker_col)

    marker_to_anchor = {int(marker): int(anchor) for anchor, marker in anchor_to_marker.items()}
    assert marker_to_anchor

    # --- STEP 1) Detect isolated border anchors outside the legend. ---
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c) or (int(r), int(c)) in legend_cells:
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
        cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []

    # --- STEP 2/3) Inspect all non-legend components and resolve their target anchors via the legend. ---
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue
        if all((int(r), int(c)) in legend_cells for r, c in coords):
            continue

        marker_col = component_marker_color(coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in marker_to_anchor or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in marker_to_anchor:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        anchor_col = int(marker_to_anchor[int(marker_col)])
        if int(anchor_col) not in anchors_by_color:
            continue
        ar, ac = anchors_by_color[int(anchor_col)]

        # --- STEP 4) Compute the translation from marker to anchor axis alignment. ---
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 5) Clear moved components, then paste them at their translated locations. ---
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
_ACTIVE_COLORS = tuple(range(1, 10))
_LEGEND_FRAME_COLOR = 0
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
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_DISTRACTOR_PROB = 0.15
_DISTRACTOR_COUNT_RANGE = (1, 4)

_LEGEND_WIDTH = 4

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


def _scatter_positions(canvas, sprite, anchor, marker_pos, bg):
    H, W = canvas.shape
    mr, mc = marker_pos
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


def _body_color(bg, reserved, used):
    pool = [c for c in _ACTIVE_COLORS if c not in reserved and c not in used and c != int(bg)]
    if not pool:
        pool = [c for c in _ACTIVE_COLORS if c not in reserved and c != int(bg)]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _deranged_mapping(anchor_colors):
    if len(anchor_colors) == 2:
        return {int(anchor_colors[0]): int(anchor_colors[1]), int(anchor_colors[1]): int(anchor_colors[0])}
    perm = anchor_colors[:]
    while True:
        random.shuffle(perm)
        if all(int(a) != int(b) for a, b in zip(anchor_colors, perm)):
            return {int(a): int(b) for a, b in zip(anchor_colors, perm)}


def _build_legend(anchor_to_marker):
    legend = np.full((len(anchor_to_marker) + 2, _LEGEND_WIDTH), int(_LEGEND_FRAME_COLOR), dtype=int)
    for i, (anchor_col, marker_col) in enumerate(anchor_to_marker.items(), start=1):
        legend[int(i), 1] = int(anchor_col)
        legend[int(i), 2] = int(marker_col)
    return legend


def _legend_origins(H, W, legend):
    h, w = legend.shape
    return [
        (0, 0),
        (0, W - w),
        (H - h, 0),
        (H - h, W - w),
    ]


def _add_distractors(out, inp, anchors, anchor_to_marker, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return
    anchor = random.choice(anchors)
    marker_color = int(anchor_to_marker[int(anchor["color"])])
    sprite = {"patch": np.array([[marker_color]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        aligned = _aligned_positions(out, sprite, anchor, bg)
        random.shuffle(aligned)
        placed = False
        for marker_pos in aligned:
            scattered = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
            if not scattered:
                continue
            in_pos = random.choice(scattered)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg)
            _place(inp, sprite, in_pos[0], in_pos[1], bg)
            placed = True
            break
        if not placed:
            return


def _gen_attempt():
    bg = random.randint(*_BG_COLOR_RANGE)
    anchor_count = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _ACTIVE_COLORS if c != int(bg)], anchor_count)
    anchor_to_marker = _deranged_mapping(anchor_colors)

    out, inp = _new_canvas_pair(bg, anchor_to_marker)
    placed = _place_anchors(out, inp, bg, anchor_colors)
    if placed is None:
        return None
    anchors = placed

    sprite_anchor_colors = _sprite_anchor_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, anchor_to_marker, sprite_anchor_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, sprites, bg):
        return None

    _add_distractors(out, inp, anchors, anchor_to_marker, bg)
    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def _new_canvas_pair(bg, anchor_to_marker):
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    legend = _build_legend(anchor_to_marker)
    r0, c0 = random.choice(_legend_origins(H, W, legend))
    paste(out, legend, r0, c0)
    paste(inp, legend, r0, c0)
    return out, inp


def _place_anchors(out, inp, bg, anchor_colors):
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
    return anchors


def _sprite_anchor_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    sprite_anchor_colors = anchor_colors[:]
    while len(sprite_anchor_colors) < n_sprites:
        sprite_anchor_colors.append(random.choice(anchor_colors))
    random.shuffle(sprite_anchor_colors)
    return sprite_anchor_colors


def _place_aligned_sprites(out, anchors, anchor_to_marker, sprite_anchor_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    reserved = set(map(int, anchor_to_marker.keys())) | {int(_LEGEND_FRAME_COLOR)}
    used_body = set()
    sprites = []
    for anchor_color in sprite_anchor_colors:
        anchor = by_color[int(anchor_color)]
        marker = int(anchor_to_marker[int(anchor_color)])
        body = _body_color(bg, reserved, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
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


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D17 input")
