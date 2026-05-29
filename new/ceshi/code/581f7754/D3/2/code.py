# Variant of 581f7754
# Dimension: D3 — 源对象：精灵结构 (src)
# Change: 多色精灵（8-连通分量），包含1个主体色（4-13像素）和1个标记色（1像素，最少数色），标记与主体4-相邻... → 空心精灵（背景色洞）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste

# Design analysis for the D3 variant
# a) Concrete behavior + trace:
#    - Each movable source object is now a compound "hollow sprite":
#      a single-color closed frame in the interior plus one singleton marker pixel
#      trapped inside the frame's cavity.
#    - The marker is intentionally separated from the body by at least one bg cell,
#      so body and marker are no longer one 8-connected component.
#    - Example: a 5x5 ring with color 6 and a color-2 marker at its center is
#      scattered in the input. In the output, the whole ring+marker translate
#      together so the center marker lines up with the border anchor of color 2.
#      The original solver would inspect one multi-color component and use the
#      minority pixel; the variant solver must instead detect a closed body,
#      recover its enclosed cavity, and find the singleton marker inside it.
# b) Complexity calibration:
#    - If the cavity were always 1x1, the old minority-color logic would still
#      solve the task. To make the variant real, the generator enforces larger
#      frames (outer size 5-7) so a marker can sit strictly inside the hole
#      without touching the body, forcing enclosure-based reasoning.
#    - Sprite count is reduced to 2-4 and the minimum grid size is raised to 12
#      so these larger hollow objects remain legible and placeable.
# c) Implicit-assumption audit of the original main():
#    - `find_components(... monochromatic=False, connectivity=8)` assumed marker
#      and body belong to one component. This fails once the marker is detached
#      inside a larger hole.
#    - `component_marker_color()` assumed the marker is the least frequent color
#      inside that component. After separation, the body component is monochrome
#      and the marker becomes its own singleton, so the old frequency heuristic
#      no longer identifies the sprite.
#    - `marker_pixel()` preferred a marker touching the body. In the variant the
#      correct marker should be selected by enclosure, not adjacency.
# d) Placement legality:
#    - Generator placement reserves the full sprite bounding box, not only the
#      visible cells. This prevents another object from being accidentally placed
#      inside the hollow cavity.
#    - Frames stay at least one bg cell away from the border and from each other;
#      anchors remain isolated border singletons.
# e) Visual observability:
#    - Across examples, every movable object visibly becomes a ring/frame with a
#      point inside it, and the point is the one that aligns with a border anchor.
#      This makes the enclosure relation visible to a human from a few samples.
# f) Visual cleanliness:
#    - Hollow sprites are larger but fewer. Their holes stay empty except for the
#      enclosed marker, so the grid does not become cluttered or ambiguous.
# g) Color inferability:
#    - Colors are only used for direct matching between an enclosed marker and a
#      same-colored anchor. No ordering or hidden color-id arithmetic is used.
# h) Variant metric used in validation:
#    - Every generated sprite must have at least one enclosed bg cell besides the
#      marker and the marker must be non-8-adjacent to the body. That guarantees
#      the hollow-sprite behavior is actually present.
# i) New solver assumptions and generator guarantees:
#    - main() assumes each hollow body encloses exactly one singleton marker whose
#      color matches an anchor; the generator constructs exactly that.
#    - main() assumes anchors stay isolated on the border; the generator keeps an
#      8-neighborhood gap around anchors.
#    - main() moves any leftover non-anchor singleton anchor-color pixels like the
#      original distractor rule, but the generator does not rely on them.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide hollow sprites so the pixel sealed inside each cavity meets the matching border anchor.

    Input structure:
    - The background is the most common color.
    - Each source sprite is a monochrome hollow body in the interior, plus one isolated marker pixel trapped inside its hole.
    - Border anchors are isolated non-bg singletons on the outer edge.

    Rule:
    - Pair each enclosed marker with the border anchor of the same color.
    - If that anchor is on the top or bottom edge, slide the whole hollow sprite horizontally until the marker reaches the anchor column.
    - If the anchor is on the left or right edge (or in a corner), slide the whole hollow sprite vertically until the marker reaches the anchor row.

    Key insight:
    - The marker is not identified by minority-color adjacency anymore; it is the unique non-bg pixel enclosed by a closed hollow body, and that enclosed pixel controls the translation.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    def enclosed_open_cells(body_coords: np.ndarray) -> set[tuple[int, int]]:
        body = {(int(r), int(c)) for r, c in body_coords}
        r0 = min(r for r, _c in body)
        c0 = min(c for _r, c in body)
        r1 = max(r for r, _c in body)
        c1 = max(c for _r, c in body)
        hh = int(r1 - r0 + 3)
        ww = int(c1 - c0 + 3)
        blocked = np.zeros((hh, ww), dtype=bool)
        for r, c in body:
            blocked[int(r - r0 + 1), int(c - c0 + 1)] = True
        seen = np.zeros_like(blocked, dtype=bool)
        stack = [(0, 0)]
        seen[0, 0] = True
        while stack:
            rr, cc = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < hh and 0 <= nc < ww):
                    continue
                if seen[nr, nc] or blocked[nr, nc]:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        enclosed = set()
        for rr in range(1, hh - 1):
            for cc in range(1, ww - 1):
                if blocked[rr, cc] or seen[rr, cc]:
                    continue
                enclosed.add((int(r0 + rr - 1), int(c0 + cc - 1)))
        return enclosed

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

    # --- STEP 2) Split the foreground into monochrome pieces. ---
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    singleton_by_pos: dict[tuple[int, int], int] = {}
    for comp in comps:
        coords = comp["coords"]
        if int(coords.shape[0]) != 1:
            continue
        r, c = map(int, coords[0])
        if (r, c) in anchor_cells:
            continue
        singleton_by_pos[(int(r), int(c))] = int(comp["color"])

    moves: list[tuple[np.ndarray, int, int]] = []
    claimed_markers: set[tuple[int, int]] = set()

    # --- STEP 3) Recover each hollow sprite by pairing a closed body with its enclosed singleton marker. ---
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        if any(is_border(int(r), int(c)) for r, c in coords):
            continue
        enclosed = enclosed_open_cells(coords)
        if not enclosed:
            continue

        enclosed_markers = [
            ((int(r), int(c)), int(singleton_by_pos[(int(r), int(c))]))
            for (r, c) in enclosed
            if (int(r), int(c)) in singleton_by_pos
            and int(singleton_by_pos[(int(r), int(c))]) in anchors_by_color
        ]
        if len(enclosed_markers) != 1:
            continue

        marker_pos, marker_col = enclosed_markers[0]
        ar, ac = anchors_by_color[int(marker_col)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - marker_pos[1])
        else:
            dr = int(ar - marker_pos[0])

        sprite_coords = np.vstack([coords, np.array([[marker_pos[0], marker_pos[1]]], dtype=int)])
        moves.append((sprite_coords, int(dr), int(dc)))
        claimed_markers.add(marker_pos)

    # --- STEP 4) Preserve the original singleton-distractor behavior for any unclaimed anchor-color singletons. ---
    for comp in comps:
        coords = comp["coords"]
        if int(coords.shape[0]) != 1:
            continue
        r, c = map(int, coords[0])
        pos = (int(r), int(c))
        if pos in anchor_cells or pos in claimed_markers:
            continue
        color = int(comp["color"])
        if color not in anchors_by_color:
            continue
        ar, ac = anchors_by_color[color]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - c)
        else:
            dr = int(ar - r)
        moves.append((coords.copy(), int(dr), int(dc)))

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


# --------------- generator (o5x) ---------------

# CFG: palette and canvas bounds.
_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (12, 22)
_GRID_WIDTH_RANGE = (12, 22)

# CFG: placement isolation and anchor location.
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

# CFG: anchor, sprite, and retry counts.
_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (2, 4)
_GEN_RETRY_LIMIT = 500

# CFG: hollow-body geometry.
_FRAME_HEIGHT_RANGE = (5, 7)
_FRAME_WIDTH_RANGE = (5, 7)
_MAX_TEETH = 2
_FLIP_PROB = 0.5


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


def _enclosed_open_cells_from_set(body):
    body = {(int(r), int(c)) for r, c in body}
    r0, c0, r1, c1 = _bbox(body)
    hh = int(r1 - r0 + 3)
    ww = int(c1 - c0 + 3)
    blocked = np.zeros((hh, ww), dtype=bool)
    for r, c in body:
        blocked[int(r - r0 + 1), int(c - c0 + 1)] = True
    seen = np.zeros_like(blocked, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        rr, cc = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < hh and 0 <= nc < ww):
                continue
            if seen[nr, nc] or blocked[nr, nc]:
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    enclosed = set()
    for rr in range(1, hh - 1):
        for cc in range(1, ww - 1):
            if blocked[rr, cc] or seen[rr, cc]:
                continue
            enclosed.add((int(r0 + rr - 1), int(c0 + cc - 1)))
    return enclosed


def _make_hollow_body():
    h = random.randint(*_FRAME_HEIGHT_RANGE)
    w = random.randint(*_FRAME_WIDTH_RANGE)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    side_specs = [
        ("top", 1, lambda pos: (1, pos), range(2, w - 2)),
        ("bottom", 1, lambda pos: (h - 2, pos), range(2, w - 2)),
        ("left", 1, lambda pos: (pos, 1), range(2, h - 2)),
        ("right", 1, lambda pos: (pos, w - 2), range(2, h - 2)),
    ]
    n_teeth = random.randint(0, _MAX_TEETH)
    for _ in range(n_teeth):
        valid = [(build, positions) for _name, _depth, build, positions in side_specs if list(positions)]
        if not valid:
            break
        build, positions = random.choice(valid)
        pos = random.choice(list(positions))
        cells.add(build(int(pos)))
    body = _flip(cells)
    enclosed = _enclosed_open_cells_from_set(body)
    if not enclosed:
        return _make_hollow_body()
    return body


def _marker_cell(body):
    body_set = {(int(r), int(c)) for r, c in body}
    enclosed = _enclosed_open_cells_from_set(body_set)
    deep = []
    for r, c in enclosed:
        touches_body = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (int(r + dr), int(c + dc)) in body_set:
                    touches_body = True
                    break
            if touches_body:
                break
        if not touches_body:
            deep.append((int(r), int(c)))
    if not deep:
        return None
    return random.choice(deep)


def _make_sprite(body_color, marker_color, bg):
    for _ in range(40):
        body = _make_hollow_body()
        marker = _marker_cell(body)
        if marker is None:
            continue
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[int(r - r0), int(c - c0)] = int(body_color)
        patch[int(marker[0] - r0), int(marker[1] - c0)] = int(marker_color)
        enclosed = _enclosed_open_cells_from_set(body)
        return {
            "patch": patch,
            "offset": (int(marker[0] - r0), int(marker[1] - c0)),
            "body_area": len(body),
            "hole_area": len(enclosed),
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
    view = canvas[top : top + h, left : left + w]
    if bool((view != int(bg)).any()):
        return False
    mask = patch != int(bg)
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
        sprite = _make_sprite(body, marker, bg)
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
    sprites = _place_aligned_sprites(out, anchors, marker_colors, bg)
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
    raise RuntimeError("failed to generate a valid 581f7754 D3 input")
