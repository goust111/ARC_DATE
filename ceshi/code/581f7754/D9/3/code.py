# Variant of 581f7754
# Dimension: D9 — 精灵内部标记识别规则 (meta)
# Change: 标记色为精灵内部出现次数最少的颜色（少数色），仅1个像素... → 标记为精灵内部被背景色包围的孤立像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D9 variant
# ==================================
# a) Concrete behavior + trace
#    - Original sprite logic: a component usually had one 1-cell minority color, so the
#      solver could identify the controller pixel by counting colors alone.
#    - Variant sprite logic here: each component has one "tip" cell with exactly three
#      4-neighbors outside the component; that tip color matches a border anchor.
#      The same anchor color is repeated on extra body cells, so color frequency no longer
#      isolates the moving pixel. A separate 1-cell decoy color is embedded in the body.
#    - Example trace:
#      Original counts: {body=8:6, marker=4:1} -> old solver picks color 4 directly.
#      Variant counts: {body=8:6, anchor=4:3, decoy=7:1}. Old minority-color logic would
#      pick color 7 and fail or move incorrectly. New logic scores each cell by how many of
#      its 4-neighbors are background / outside the component; the exposed tip has score 3,
#      every other sprite cell has score <= 2, so the anchor-color tip is selected.
# b) Complexity calibration
#    - The change is visually observable because the anchor color appears multiple times
#      inside the sprite, but only one occurrence sits on a protruding tip.
#    - A single unrelated decoy pixel prevents the instances from collapsing back into the
#      original "least frequent color" task.
#    - Anchor count stays small (1-3) and sprite count stays moderate (3-5), preserving
#      repeated evidence without creating too many special cases.
# c) Implicit-assumption audit of the original main()
#    - `component_marker_color(coords)` depended on "marker color is globally least frequent".
#      That is false now and must be removed.
#    - `marker_pixel(coords, marker)` assumed the correct color was already known. Now the
#      solver must identify the pixel before reading its color.
#    - Singleton anchor components were harmless before because the minority-color rule often
#      skipped them. In the new logic, a singleton has maximal exposure 4, so anchors must be
#      explicitly excluded from the movable components list.
# d) Placement legality
#    - The generator still uses isolated interior placement with an 8-neighbor gap, so
#      connected components never merge accidentally.
#    - Every generated tip is fully inside the grid before motion, so its 4-neighbor exposure
#      is measured against component topology rather than against the canvas boundary.
#    - Output-first generation is preserved: solved sprites are placed first, then scattered
#      along the inverse axis, and the final input is oracle-checked with `main(input)`.
# e) Visual observability
#    - Humans can see that the repeated anchor-colored cells do not all behave equally; the
#      controlling one is the cell sticking out into background.
#    - The decoy singleton color is visible but never the moving cue, so the examples provide
#      evidence against a pure frequency rule.
# f) Visual cleanliness
#    - Distractor singleton mini-sprites from the original generator are removed to keep the
#      topology cue readable.
#    - Sprites use a thick base body plus one protruding tip, which is visually easy to parse.
# g) Color-inference safety
#    - No sorting or movement uses raw color ids. Color only serves as the anchor-match key
#      after the marker pixel has been identified structurally.
#    - The decoy color is chosen from non-anchor colors when possible, so the color cue on the
#      border remains unambiguous.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites so their most exposed tip-pixel lands on the matching border anchor.

    Input structure:
    - The background is the most common color.
    - Interior sprites are multi-color connected components; each contains one pixel whose 4-neighborhood touches background / component-exterior more than any other sprite pixel.
    - Isolated non-background border singletons provide the anchor colors and the slide axis.

    Rule:
    - For each sprite, find the uniquely most exposed pixel, then match that pixel's color to the border anchor of the same color.
    - A top/bottom-edge anchor makes the sprite slide horizontally until the marker column matches the anchor column; a left/right-edge anchor makes it slide vertically until the marker row matches the anchor row.
    - Source cells are cleared first, moved pixels clip at the grid boundary, and settled non-background cells are not overwritten.

    Key insight:
    - The controlling pixel is not the rarest color anymore; it is the component cell with the strongest 4-neighbor exposure to empty space, so local topology determines which color drives the motion.
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

    def exposed_marker_pixel(coords: np.ndarray) -> tuple[int, int] | None:
        if coords.size == 0:
            return None
        pts = [(int(r), int(c)) for r, c in coords]
        comp_set = set(pts)
        scored: list[tuple[int, int, int]] = []
        for r, c in pts:
            exposure = 0
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) not in comp_set:
                    exposure += 1
            scored.append((int(exposure), int(r), int(c)))

        best = max(score for score, _r, _c in scored)
        candidates = [(r, c) for score, r, c in scored if score == best]
        if len(candidates) == 1:
            return candidates[0]

        anchor_colored = [(r, c) for r, c in candidates if int(g[r, c]) in anchors_by_color]
        if len(anchor_colored) == 1:
            return anchor_colored[0]
        if anchor_colored:
            candidates = anchor_colored

        return min(candidates)

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        if int(coords.shape[0]) == 1:
            r, c = map(int, coords[0])
            if (r, c) in anchor_cells:
                continue
            marker_pos = (int(r), int(c))
        else:
            marker_pos = exposed_marker_pixel(coords)
            if marker_pos is None:
                continue

        marker_col = int(g[marker_pos[0], marker_pos[1]])
        if marker_col not in anchors_by_color:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - marker_pos[1])
        else:
            dr = int(ar - marker_pos[0])

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
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_GEN_RETRY_LIMIT = 400

_RECT_H_RANGE = (2, 4)
_RECT_W_RANGE = (3, 5)
_FRAME_H_RANGE = (3, 5)
_FRAME_W_RANGE = (3, 5)


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


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _flip(cells):
    cells = set(cells)
    if random.random() < 0.5:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < 0.5:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _filled_body():
    h = random.randint(*_RECT_H_RANGE)
    w = random.randint(*_RECT_W_RANGE)
    return {(r, c) for r in range(h) for c in range(w)}


def _frame_body():
    h = random.randint(*_FRAME_H_RANGE)
    w = random.randint(*_FRAME_W_RANGE)
    return {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }


def _notched_body():
    cells = _filled_body()
    r0, c0, r1, c1 = _bbox(cells)
    removable = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    cells.remove(random.choice(sorted(removable)))
    return cells


def _base_body():
    x = random.random()
    if x < 0.45:
        body = _filled_body()
    elif x < 0.75:
        body = _frame_body()
    else:
        body = _notched_body()
    return _flip(body)


def _tip_candidates(body):
    body = set(body)
    r0, c0, r1, c1 = _bbox(body)
    out = []
    for r, c in sorted(body):
        if r == r0 and c0 < c < c1 and (r - 1, c) not in body:
            out.append(((r - 1, c), (r, c)))
        if r == r1 and c0 < c < c1 and (r + 1, c) not in body:
            out.append(((r + 1, c), (r, c)))
        if c == c0 and r0 < r < r1 and (r, c - 1) not in body:
            out.append(((r, c - 1), (r, c)))
        if c == c1 and r0 < r < r1 and (r, c + 1) not in body:
            out.append(((r, c + 1), (r, c)))
    return out


def _cell_exposure(cells, pos):
    cell_set = set(cells)
    r, c = pos
    return sum((r + dr, c + dc) not in cell_set for dr, dc in _NEIGHBORS4)


def _color_pool(bg, forbidden=()):
    blocked = {int(bg)} | {int(x) for x in forbidden}
    return [c for c in _COLORS if c not in blocked]


def _build_sprite(marker_color, bg, anchor_colors, used_body_colors):
    for _ in range(80):
        raw_body = _base_body()
        tip_options = _tip_candidates(raw_body)
        if not tip_options:
            continue
        tip, attach = random.choice(tip_options)
        raw_comp = set(raw_body) | {tip}
        r0, c0, _, _ = _bbox(raw_comp)
        body = {(r - r0, c - c0) for r, c in raw_body}
        tip = (tip[0] - r0, tip[1] - c0)
        attach = (attach[0] - r0, attach[1] - c0)
        comp = set(body) | {tip}

        if _cell_exposure(comp, tip) != 3:
            continue
        if max(_cell_exposure(comp, cell) for cell in body) >= 3:
            continue

        marker_pool = [attach] + [cell for cell in sorted(body) if cell != attach]
        if len(marker_pool) < 2:
            continue
        marker_body_cells = {attach, random.choice(marker_pool[1:])}

        decoy_candidates = [cell for cell in sorted(body) if cell not in marker_body_cells]
        if not decoy_candidates:
            continue
        min_exposure = min(_cell_exposure(comp, cell) for cell in decoy_candidates)
        decoy_pool = [cell for cell in decoy_candidates if _cell_exposure(comp, cell) == min_exposure]
        decoy = random.choice(decoy_pool)

        body_pool = _color_pool(bg, forbidden=set(anchor_colors))
        if not body_pool:
            return None
        preferred_body = [c for c in body_pool if c not in used_body_colors and c != int(marker_color)]
        body_color = random.choice(preferred_body or [c for c in body_pool if c != int(marker_color)])

        decoy_pool_colors = _color_pool(bg, forbidden=set(anchor_colors) | {body_color, marker_color})
        if not decoy_pool_colors:
            decoy_pool_colors = _color_pool(bg, forbidden={body_color, marker_color})
        if not decoy_pool_colors:
            continue
        decoy_color = random.choice(decoy_pool_colors)

        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        for r, c in marker_body_cells | {tip}:
            patch[r - r0, c - c0] = int(marker_color)
        patch[decoy[0] - r0, decoy[1] - c0] = int(decoy_color)

        return {
            "patch": patch,
            "offset": (tip[0] - r0, tip[1] - c0),
            "body_color": int(body_color),
            "marker_color": int(marker_color),
            "decoy_color": int(decoy_color),
            "marker_count": int(sum(int(v) == int(marker_color) for v in patch.ravel())),
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
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(out, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body_colors = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        sprite = _build_sprite(marker, bg, set(by_color), used_body_colors)
        if sprite is None:
            return None
        used_body_colors.add(int(sprite["body_color"]))
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


def _gen_attempt(return_meta=False):
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

    if return_meta:
        return inp, out, {"anchors": anchors, "sprites": sprites, "bg": bg, "marker_colors": marker_colors}
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D9 input")
