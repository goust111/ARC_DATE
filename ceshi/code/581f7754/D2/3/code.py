# Variant of 581f7754
# Dimension: D2 — 匹配条件：滑动轴决定规则 (match_cond)
# Change: 锚点在上/下边缘（非角落）→水平滑动对齐列；锚点在左/右边缘（或角落）→垂直滑动对齐行... → 双轴对齐（对角滑动）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D2 variant:
# - Concrete behavior:
#   Each border anchor now specifies one exact interior target cell: the cell one step inward
#   from that anchor. A top anchor at (0,c) targets (1,c), a bottom anchor at (H-1,c) targets
#   (H-2,c), a left anchor at (r,0) targets (r,1), and a right anchor at (r,W-1) targets
#   (r,W-2). The whole sprite translates so its marker lands on that target, so both row and
#   column generally change together.
# - Trace of the changed step:
#   If a marker of color 4 starts at (6,3) and its anchor is on the top edge at (0,8), the
#   original task only matched the column and moved to (6,8). The variant instead moves to the
#   inward-neighbor cell (1,8), so the translation is (-5,+5). This is the exact point where
#   the solver logic changes: the destination is no longer “same row or same column”, but a
#   single cell derived from the anchor.
# - Complexity calibration:
#   A dual-axis target gives each anchor color only one landing cell. Reusing the same marker
#   color for multiple sprites would force them to stack, so the generator now uses one sprite
#   per anchor color and disables same-color singleton distractors. To keep the diagonal rule
#   visually obvious, input placements are forced to differ from the target on both axes, with
#   a preference for shifts of at least 2 cells in each direction.
# - Implicit-assumption audit of the original code:
#   The old `main()` assumed one of `dr` or `dc` stayed zero because the anchor selected an
#   axis. That assumption is invalid here; both components of the translation must be computed.
#   The old generator also relied on a free axis for settled placements and scattered sprites
#   only along the inverse axis. Those steps must be replaced by a fixed solved marker position
#   and a 2D scatter. Finally, the old marker generator could place the marker deep inside the
#   patch; with an inward-neighbor target that would push the sprite across the border, so the
#   variant constrains the marker to the side of the sprite that faces its anchor.
# - Placement legality:
#   Solved sprites are placed on an anchor-free output canvas first, because in the final output
#   each marker is intentionally adjacent to its anchor. Input sprites are placed on a canvas
#   that already contains the anchors, so they remain isolated from anchors and from each other.
#   All solved non-background cells are still kept strictly inside the border.
# - Visual observability and neatness:
#   Every generated grid uses 2-3 distinct anchor/sprite pairs. The output shows each marker
#   tucked just inside the matching border pixel, and the input shows the same components moved
#   away in both row and column. This makes the diagonal nature of the motion readable from a
#   small number of examples without relying on color-id tricks.
# - Color inference:
#   Colors only indicate anchor-marker pairing. No ordering, ranking, or arithmetic over color
#   ids is used anywhere in the rule.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate interior sprites so their marker pixels end up immediately inside the matching border anchors.

    Input structure:
    - The background is the modal color of the grid.
    - Interior connected components are mostly two-color sprites: one dominant body color plus one minority marker color.
    - Isolated non-background border pixels serve as anchors, and a sprite's marker color identifies its matching anchor.

    Rule:
    - For each sprite, find the anchor with the same color as its marker.
    - Move the entire sprite in both row and column so the marker lands on the inward-neighbor cell of that anchor.
    - Keep anchors fixed, clear the source sprite positions, clip translated pixels that leave the grid, and never overwrite existing non-background cells.

    Key insight: A border anchor no longer chooses a slide axis; it defines one exact destination cell just inside the border, so the marker must be aligned in both coordinates at once.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def anchor_target(r: int, c: int) -> tuple[int, int]:
        tr = int(r)
        tc = int(c)
        if tr == 0:
            tr = 1
        elif tr == H - 1:
            tr = H - 2
        if tc == 0:
            tc = 1
        elif tc == W - 1:
            tc = W - 2
        return tr, tc

    # --- STEP 1) Detect anchors: isolated border pixels (non-bg). ---
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

    # --- STEP 2) Extract all foreground connected components as candidate sprites. ---
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

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        # --- STEP 3) Decide marker color for this component. ---
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
        tr, tc = anchor_target(int(ar), int(ac))

        # --- STEP 4) Compute translation (dr,dc) to the anchor-adjacent interior target. ---
        dr = int(tr - mp[0])
        dc = int(tc - mp[1])
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


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 300
_SPRITE_BUILD_RETRIES = 30
_STRONG_SHIFT_MIN = 2

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


def _marker_cell(body, side):
    r0, c0, r1, c1 = _bbox(body)
    if side == "top":
        pool = [(r - 1, c) for r, c in body if r == r0]
    elif side == "bottom":
        pool = [(r + 1, c) for r, c in body if r == r1]
    elif side == "left":
        pool = [(r, c - 1) for r, c in body if c == c0]
    else:
        pool = [(r, c + 1) for r, c in body if c == c1]
    pool = [p for p in pool if p not in body]
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg, side):
    body = _make_body(area)
    marker = _marker_cell(body, side)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {
        "patch": patch,
        "offset": (marker[0] - r0, marker[1] - c0),
        "area": int((patch != int(bg)).sum()),
    }


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
    return [(side, r, c) for side, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _anchor_target(anchor, shape):
    H, W = shape
    if anchor["side"] == "top":
        return 1, int(anchor["c"])
    if anchor["side"] == "bottom":
        return int(H - 2), int(anchor["c"])
    if anchor["side"] == "left":
        return int(anchor["r"]), 1
    if anchor["side"] == "right":
        return int(anchor["r"]), int(W - 2)
    raise ValueError(f"unknown side {anchor['side']}")


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


def _choose_anchors(shape, bg):
    H, W = shape
    anchor_canvas = np.full((H, W), int(bg), dtype=int)
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(anchor_canvas, bg)
        if not candidates:
            return None
        side, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "side": side}
        anchors.append(anchor)
        _put_anchor(anchor_canvas, r, c, color)
    return anchor_canvas, anchors


def _put_anchors(canvas, anchors):
    for anchor in anchors:
        _put_anchor(canvas, anchor["r"], anchor["c"], anchor["color"])


def _place_aligned_sprites(out, anchors, bg):
    used_body = set()
    sprites = []
    for anchor in anchors:
        target = _anchor_target(anchor, out.shape)
        built = None
        for _ in range(_SPRITE_BUILD_RETRIES):
            body = _body_color(bg, anchor["color"], used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, anchor["color"], bg, anchor["side"])
            if _can_place(out, sprite, target[0], target[1], bg):
                built = (body, sprite)
                break
        if built is None:
            return None
        body, sprite = built
        used_body.add(body)
        _place(out, sprite, target[0], target[1], bg)
        sprites.append({"sprite": sprite, "anchor": anchor, "target": target})
    return sprites


def _scatter_positions(canvas, sprite, target, bg):
    H, W = canvas.shape
    candidates = []
    strong = []
    tr, tc = target
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if r == tr or c == tc:
                continue
            if not _can_place(canvas, sprite, r, c, bg):
                continue
            candidates.append((r, c))
            if abs(r - tr) >= _STRONG_SHIFT_MIN and abs(c - tc) >= _STRONG_SHIFT_MIN:
                strong.append((r, c))
    return strong if strong else candidates


def _scatter_sprites(inp, sprites, bg):
    ordered = sorted(sprites, key=lambda item: item["sprite"]["area"], reverse=True)
    placements = []
    for item in ordered:
        positions = _scatter_positions(inp, item["sprite"], item["target"], bg)
        if not positions:
            return None
        in_pos = random.choice(positions)
        _place(inp, item["sprite"], in_pos[0], in_pos[1], bg)
        placements.append((item, in_pos))
    return placements


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()

    chosen = _choose_anchors(out.shape, bg)
    if chosen is None:
        return None
    anchor_canvas, anchors = chosen

    sprites = _place_aligned_sprites(out, anchors, bg)
    if sprites is None:
        return None

    inp[:, :] = anchor_canvas
    placements = _scatter_sprites(inp, sprites, bg)
    if placements is None:
        return None

    _put_anchors(out, anchors)

    if not _generated_grid_is_valid(inp, out, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D2 input")
