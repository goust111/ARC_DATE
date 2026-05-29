# Combo variant of 581f7754 (index 22)
# combo_spec: D3/3+D12/0
# Name: 对角标记+矩形精灵
#
# Design analysis:
# - concretization:
#   - Each movable multicolor object is now a solid rectangle plus one marker pixel placed just outside one
#     rectangle corner on the diagonal: for a 2x3 body, examples are `m.../ .###/ .###` after normalization,
#     or the same idea at the other three corners.
#   - The marker is still the least-frequent color and still matches a same-colored isolated border anchor,
#     but it no longer shares an edge with the body. The whole sprite remains one 8-connected component only.
# - behavior trace:
#   - Original task: irregular bodies with a side-attached marker were moved by matching marker color to a
#     border singleton, then sliding along the axis implied by the anchor edge.
#   - Fused task: removing the unique marker from a component must leave a filled rectangle, and the marker
#     must sit at one of the four exterior diagonal corner cells of that rectangle. Motion after recognition is
#     unchanged: top/bottom anchors align columns, left/right anchors align rows.
#   - Difference: shape regularity is higher, so the visible challenge shifts from reading irregular outlines to
#     noticing the 8-connected corner attachment and using the rectangle corner as the marker reference point.
# - complexity calibration:
#   - Rectangles reduce body-shape variety, so the generator compensates with varied rectangle sizes, all four
#     corner-marker placements, mixed anchor axes, and repeated anchor colors across 3-5 sprites.
#   - Body colors are chosen away from anchor colors when possible, making marker-anchor equality the repeated
#     cue instead of hidden color ordering.
# - implicit assumptions review:
#   - `find_components(..., connectivity=8)` is still required; under 4-connectivity every sprite would split
#     into a rectangle body and an isolated marker pixel.
#   - Least-frequency alone is not a sufficient fused description, so the solver now verifies that the non-marker
#     cells form a solid rectangle and that the marker touches that rectangle only diagonally at a corner.
#   - Translation, clearing, clipping, and collision handling stay valid because the transform is still pure
#     axis-aligned sliding of whole components.
# - placement legality:
#   - The generator places anchors as isolated non-corner border pixels and keeps each sprite at least one
#     background cell away from borders, anchors, and other sprites in both output-first and scattered states.
#   - Marker positions are restricted to the four exterior diagonal corner cells of the rectangle, so they are
#     guaranteed to be 8-adjacent but not 4-adjacent to the body.
# - visual observability:
#   - Across a few examples a human can see two stable cues: the marker has the same color as a border anchor,
#     and it sits on a corner offset rather than on an edge. The rectangle body makes that corner relation easy
#     to read directly from pixels.
# - cleanliness:
#   - Optional singleton distractors are removed in this combo variant so every non-anchor moving object obeys
#     the same rectangle-plus-corner-marker schema and the scenes stay sparse.
# - inference:
#   - No invisible numeric property is used. The only color relation is visible equality between marker and
#     anchor, and the only shape relation is visible rectangle geometry plus diagonal contact.
# - new solver assumptions and generator guarantees:
#   - `main()` assumes every intended multicolor sprite has exactly one marker pixel and a one-color filled
#     rectangle body; `_generated_grid_is_valid()` checks that every generated multicolor component satisfies
#     those assumptions before accepting a sample.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, neighbors8, paste


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rectangle_sprite_info(
    grid: np.ndarray, coords: np.ndarray
) -> dict[str, object] | None:
    vals = [int(grid[int(r), int(c)]) for r, c in coords]
    counts = Counter(vals)
    if len(counts) < 2:
        return None

    marker_color, freq = min(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
    if int(freq) != 1:
        return None

    marker = next(
        (int(r), int(c))
        for r, c in coords
        if int(grid[int(r), int(c)]) == int(marker_color)
    )
    body = [
        (int(r), int(c))
        for r, c in coords
        if int(grid[int(r), int(c)]) != int(marker_color)
    ]
    if not body:
        return None

    body_colors = {int(grid[r, c]) for r, c in body}
    if len(body_colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(body)
    if len(body) != (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
        return None

    body_set = set(body)
    if any((r, c) not in body_set for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
        return None

    corners = {
        (int(r0 - 1), int(c0 - 1)): "nw",
        (int(r0 - 1), int(c1 + 1)): "ne",
        (int(r1 + 1), int(c0 - 1)): "sw",
        (int(r1 + 1), int(c1 + 1)): "se",
    }
    marker_corner = corners.get(marker)
    if marker_corner is None:
        return None

    if any((int(rr), int(cc)) in body_set for rr, cc in neighbors4(*marker)):
        return None

    touch8 = [(int(rr), int(cc)) for rr, cc in neighbors8(*marker) if (int(rr), int(cc)) in body_set]
    if len(touch8) != 1:
        return None

    return {
        "marker_color": int(marker_color),
        "marker": marker,
        "body_color": int(next(iter(body_colors))),
        "body_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "body_area": int(len(body)),
        "corner": marker_corner,
    }


def _is_movable_singleton(
    grid: np.ndarray,
    coords: np.ndarray,
    bg: int,
    anchors_by_color: dict[int, tuple[int, int]],
    anchor_cells: set[tuple[int, int]],
) -> dict[str, object] | None:
    if int(coords.shape[0]) != 1:
        return None
    r, c = map(int, coords[0])
    color = int(grid[r, c])
    if color == int(bg) or color not in anchors_by_color or (r, c) in anchor_cells:
        return None
    return {"marker_color": int(color), "marker": (int(r), int(c))}


def _find_anchors(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = grid.shape

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            value = int(grid[r, c])
            if value == int(bg):
                continue
            if r in (0, H - 1) and c in (0, W - 1):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and value not in anchors_by_color:
                anchors_by_color[int(value)] = (int(r), int(c))
    return anchors_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide rectangle-bodied sprites so their diagonal corner markers line up with matching border anchors.

    Input structure:
    - The background is the most common color.
    - Each movable multicolor sprite is one 8-connected component made of a solid rectangle body plus one 1-pixel marker color.
    - The marker sits just outside a rectangle corner, touching the body only diagonally, while isolated non-background border singletons act as anchors.

    Rule:
    - Pair every sprite with the border anchor sharing its marker color.
    - Anchors on the top or bottom edge force a horizontal slide so the marker reaches the anchor column; anchors on the left or right edge force a vertical slide so the marker reaches the anchor row.
    - Pixels shifted beyond the grid are clipped, and already occupied non-background destination cells are left unchanged.

    Key insight:
    - The important object cue is not just the rare color: if you remove that pixel, the rest of the component is a filled rectangle, and the rare pixel belongs to it only through an exterior diagonal corner contact, so 8-connectivity is essential.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    anchors_by_color = _find_anchors(g, bg)
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        info = _rectangle_sprite_info(g, coords)
        if info is None:
            info = _is_movable_singleton(g, coords, bg, anchors_by_color, anchor_cells)
        if info is None:
            continue

        marker_color = int(info["marker_color"])
        if marker_color not in anchors_by_color:
            continue

        marker_r, marker_c = map(int, info["marker"])
        anchor_r, anchor_c = anchors_by_color[marker_color]
        horizontal = int(anchor_r) in (0, int(H - 1))

        dr = 0
        dc = 0
        if horizontal:
            dc = int(anchor_c - marker_c)
        else:
            dr = int(anchor_r - marker_r)

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

_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_GEN_RETRY_LIMIT = 400
_SPRITE_BUILD_RETRY_LIMIT = 80

_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (2, 7)
_BODY_AREA_RANGE = (4, 13)
_MARKER_CORNERS = ("nw", "ne", "sw", "se")


def _inclusive_range(bounds: tuple[int, int]) -> range:
    return range(int(bounds[0]), int(bounds[1]) + 1)


_RECTANGLE_DIMS = [
    (int(h), int(w))
    for h in _inclusive_range(_BODY_HEIGHT_RANGE)
    for w in _inclusive_range(_BODY_WIDTH_RANGE)
    if _BODY_AREA_RANGE[0] <= int(h) * int(w) <= _BODY_AREA_RANGE[1]
]


def _top_left(sprite: dict[str, object], marker_r: int, marker_c: int) -> tuple[int, int]:
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas: np.ndarray, sprite: dict[str, object], marker_r: int, marker_c: int, bg: int) -> bool:
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


def _place(canvas: np.ndarray, sprite: dict[str, object], marker_r: int, marker_c: int, bg: int) -> None:
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _isolated_bg_cell(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _anchor_candidates(canvas: np.ndarray, bg: int) -> list[tuple[str, int, int]]:
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


def _put_anchor(canvas: np.ndarray, r: int, c: int, color: int) -> None:
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _make_sprite(body_color: int, marker_color: int, bg: int) -> dict[str, object]:
    height, width = random.choice(_RECTANGLE_DIMS)
    body = {(r, c) for r in range(height) for c in range(width)}
    corner = random.choice(_MARKER_CORNERS)
    marker = {
        "nw": (-1, -1),
        "ne": (-1, width),
        "sw": (height, -1),
        "se": (height, width),
    }[corner]

    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    patch[int(marker[0] - r0), int(marker[1] - c0)] = int(marker_color)
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
        "corner": corner,
        "body_shape": (int(height), int(width)),
    }


def _aligned_positions(
    canvas: np.ndarray, sprite: dict[str, object], anchor: dict[str, int], bg: int
) -> list[tuple[int, int]]:
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


def _scatter_positions(
    canvas: np.ndarray,
    sprite: dict[str, object],
    anchor: dict[str, int],
    marker: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    marker_r, marker_c = marker
    if anchor["axis"] == "h":
        return [
            (marker_r, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != marker_c and _can_place(canvas, sprite, marker_r, c, bg)
        ]
    return [
        (r, marker_c)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != marker_r and _can_place(canvas, sprite, r, marker_c, bg)
    ]


def _body_color(bg: int, anchor_colors: list[int], used: set[int]) -> int:
    blocked = {int(bg), *map(int, anchor_colors)}
    pool = [c for c in _COLORS if c not in blocked and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in blocked]
    return int(random.choice(pool))


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair() -> tuple[int, np.ndarray, np.ndarray]:
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, out, inp


def _place_anchors(
    out: np.ndarray, inp: np.ndarray, bg: int
) -> tuple[list[int], list[dict[str, int]]] | None:
    anchor_colors = random.sample(
        [c for c in _COLORS if c != bg],
        random.randint(*_ANCHOR_COUNT_RANGE),
    )
    required_axes = ["h", "v"]
    if len(anchor_colors) > 2:
        required_axes.append(random.choice(("h", "v")))
    random.shuffle(required_axes)

    anchors: list[dict[str, int]] = []
    for color, axis_needed in zip(anchor_colors, required_axes):
        candidates = [cand for cand in _anchor_candidates(out, bg) if cand[0] == axis_needed]
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors: list[int]) -> list[int]:
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(
    out: np.ndarray,
    anchors: list[dict[str, int]],
    marker_colors: list[int],
    anchor_colors: list[int],
    bg: int,
) -> list[tuple[dict[str, object], dict[str, int], tuple[int, int]]] | None:
    by_color = {a["color"]: a for a in anchors}
    used_body: set[int] = set()
    sprites: list[tuple[dict[str, object], dict[str, int], tuple[int, int]]] = []

    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body_color = _body_color(bg, anchor_colors, used_body)

        placed = False
        for _ in range(_SPRITE_BUILD_RETRY_LIMIT):
            sprite = _make_sprite(body_color, marker, bg)
            positions = _aligned_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            marker_pos = random.choice(positions)
            _place(out, sprite, marker_pos[0], marker_pos[1], bg)
            sprites.append((sprite, anchor, marker_pos))
            used_body.add(int(body_color))
            placed = True
            break
        if not placed:
            return None

    return sprites


def _scatter_sprites(
    inp: np.ndarray,
    sprites: list[tuple[dict[str, object], dict[str, int], tuple[int, int]]],
    bg: int,
) -> bool:
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _variant_stats(grid: np.ndarray, bg: int) -> dict[str, object]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    multi = 0
    rect_diag = 0
    bad = 0
    corner_counts = Counter()
    shape_counts = Counter()

    for comp in comps:
        coords = comp["coords"]
        info = _rectangle_sprite_info(grid, coords)
        if info is None:
            vals = {int(grid[int(r), int(c)]) for r, c in coords}
            if len(vals) > 1:
                bad += 1
            continue
        multi += 1
        rect_diag += 1
        corner_counts[str(info["corner"])] += 1
        r0, c0, r1, c1 = info["body_bbox"]
        shape_counts[(int(r1 - r0 + 1), int(c1 - c0 + 1))] += 1

    return {
        "multi": int(multi),
        "rect_diag": int(rect_diag),
        "bad": int(bad),
        "corners": dict(corner_counts),
        "shapes": dict(shape_counts),
    }


def _generated_grid_is_valid(inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    if not _bg_is_mode(inp, bg):
        return False
    anchors = _find_anchors(inp, bg)
    axes = {
        "h" if int(r) in (0, int(inp.shape[0] - 1)) else "v"
        for r, _c in anchors.values()
    }
    if len(anchors) < 2 or axes != {"h", "v"}:
        return False
    stats = _variant_stats(inp, bg)
    if stats["multi"] == 0 or stats["rect_diag"] != stats["multi"] or stats["bad"] != 0:
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt() -> np.ndarray | None:
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(out, anchors, marker_colors, anchor_colors, bg)
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
