# Combo variant of 581f7754 (index 16)
# combo_spec: D12/2+D14/1
# Name: 线段精灵+滑到边界

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior and traced example:
#    The original task used chunky irregular sprites and moved them only until the marker
#    landed on the anchor row/column. This combo changes both pieces at once.
#    Example: suppose a top-edge anchor of color 4 sits at column 3, and a matching
#    sprite is now a thin 5-cell horizontal line with a 4-colored marker attached beside
#    its right endpoint. If the marker currently sits at column 9, the old task would
#    shift the whole component left so the marker finishes at column 3. The combo still
#    uses "top edge => horizontal motion" and still uses "anchor is left of marker" to
#    choose a leftward move, but it does not stop at column 3. Instead it keeps sliding
#    until some sprite cell reaches the left border. The visible difference from the
#    original is therefore twofold: bodies are skinny line segments, and solved sprites
#    are flush against a border rather than merely aligned to the anchor projection.
# b) Complexity calibration:
#    The generator keeps 2-3 anchor colors and 4-5 sprites so marker colors repeat and
#    color-matching remains necessary. Body orientation is limited to horizontal/vertical
#    line segments and lengths 2..7, which is visually distinct from the old blocky
#    bodies but still simple enough for humans to parse quickly.
# c) Hidden assumptions checked against the old solver:
#    - Anchor detection via isolated border singletons still works unchanged.
#    - Marker-color matching still works because each sprite is a 2-color component with
#      one unique marker pixel.
#    - The old `dc = ac - marker_c` / `dr = ar - marker_r` assumption is invalid now:
#      anchor coordinates no longer give the destination, only the axis and the travel
#      direction do. Translation must be computed from component extrema and the chosen
#      boundary.
#    - The old generator assumed solved sprites stayed one cell away from all borders.
#      That must be relaxed on exactly one side, otherwise the gravity stop state could
#      not be generated.
#    - A new solver assumption is that anchor row/column is not exactly equal to the
#      marker row/column along the motion axis, because otherwise direction is undefined.
#      The generator guarantees strict inequality when scattering input sprites.
# d) Placement legality:
#    Output is built first with boundary-touching settled sprites. `_can_place` allows
#    contact on only the intended stopping side and still enforces non-overlap plus an
#    empty 8-neighborhood from anchors and other sprites. Input sprites are then scattered
#    back into the interior on the correct side of their anchor so the solver recovers the
#    intended left/right/up/down direction.
# e) Visual observability:
#    A human looking at a few examples can spot that marker colors still match anchors,
#    but the destination is no longer the anchor line because outputs visibly pile up
#    against borders. The thin line bodies make the D12 change obvious at a glance.
# f) Visual cleanliness:
#    Distractor singletons are removed. Every movable object follows the same grammar:
#    one straight line body plus one adjacent endpoint marker. This keeps the grid legible
#    even when several sprites share a marker color.
# g) Color inference:
#    Color ids are not used numerically. Colors only encode anchor-marker pairing; the
#    new behavior is driven by visible geometry and relative position.
# h) Generator guarantees for new solver assumptions:
#    - Every movable component has one least-frequent marker pixel.
#    - The remaining cells form a 1-pixel-wide line segment of length 2..7.
#    - Anchors are isolated non-corner border pixels.
#    - Settled destinations do not overlap, so the solver's "do not overwrite non-bg"
#      paste rule reproduces the constructed output exactly.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide thin marker-bearing line sprites toward their matching anchors until they hit a grid edge.

    Input structure:
    - The background is the most common color.
    - Interior sprites are 2-color connected components: a 1-pixel-wide horizontal or vertical line body plus one marker pixel attached beside an endpoint.
    - Border anchors are isolated non-background singletons whose color matches sprite markers.

    Rule:
    - Match each sprite to the anchor of the same marker color.
    - A top/bottom anchor selects horizontal motion; a left/right anchor selects vertical motion.
    - The anchor only determines which way to move along that axis: if it lies left/up of the marker, slide toward that side; if it lies right/down, slide toward that side. Keep moving until the component touches the corresponding grid boundary.

    Key insight:
    Separate anchor matching from stopping position: color chooses the anchor, anchor position chooses the travel axis and direction, and the boundary itself is the final stop.
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

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def parse_line_sprite(coords: np.ndarray) -> dict[str, object] | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) != 2:
            return None

        marker_col, marker_count = min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
        if int(marker_count) != 1:
            return None

        marker_pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker_col)]
        if len(marker_pts) != 1:
            return None
        marker = marker_pts[0]

        body = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) != int(marker_col)]
        if not (2 <= len(body) <= 7):
            return None

        body_rows = sorted({int(r) for r, _c in body})
        body_cols = sorted({int(c) for _r, c in body})
        endpoints: set[tuple[int, int]]
        if len(body_rows) == 1:
            row = int(body_rows[0])
            if len(body_cols) != len(body) or body_cols != list(range(int(body_cols[0]), int(body_cols[-1]) + 1)):
                return None
            endpoints = {(int(row), int(body_cols[0])), (int(row), int(body_cols[-1]))}
        elif len(body_cols) == 1:
            col = int(body_cols[0])
            if len(body_rows) != len(body) or body_rows != list(range(int(body_rows[0]), int(body_rows[-1]) + 1)):
                return None
            endpoints = {(int(body_rows[0]), int(col)), (int(body_rows[-1]), int(col))}
        else:
            return None

        if not any(abs(int(marker[0]) - int(er)) + abs(int(marker[1]) - int(ec)) == 1 for er, ec in endpoints):
            return None

        return {
            "marker_color": int(marker_col),
            "marker": (int(marker[0]), int(marker[1])),
            "rows": [int(r) for r, _c in coords],
            "cols": [int(c) for _r, c in coords],
        }

    moves: list[tuple[np.ndarray, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        parsed = parse_line_sprite(coords)
        if parsed is None:
            continue

        marker_col = int(parsed["marker_color"])
        if marker_col not in anchors_by_color:
            continue

        mp = parsed["marker"]
        ar, ac = anchors_by_color[int(marker_col)]
        rows = parsed["rows"]
        cols = parsed["cols"]

        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            if int(ac) < int(mp[1]):
                dc = -min(cols)
            elif int(ac) > int(mp[1]):
                dc = int(W - 1 - max(cols))
            else:
                continue
        else:
            if int(ar) < int(mp[0]):
                dr = -min(rows)
            elif int(ar) > int(mp[0]):
                dr = int(H - 1 - max(rows))
            else:
                continue

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


# --------------- generator (combo variant) ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (12, 22)
_GRID_WIDTH_RANGE = (12, 22)

_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (4, 5)
_LINE_LENGTH_RANGE = (2, 7)
_GEN_RETRY_LIMIT = 500
_SPRITE_RETRY_LIMIT = 80


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _top_left(sprite: dict, marker_r: int, marker_c: int) -> tuple[int, int]:
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int) -> bool:
    occupied = {
        (int(top) + int(r), int(left) + int(c))
        for r, c in np.argwhere(mask)
    }
    H, W = canvas.shape
    for r, c in occupied:
        for rr in range(int(r) - _ISOLATION_RADIUS, int(r) + _ISOLATION_RADIUS + 1):
            for cc in range(int(c) - _ISOLATION_RADIUS, int(c) + _ISOLATION_RADIUS + 1):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (int(rr), int(cc)) in occupied:
                    continue
                if int(canvas[rr, cc]) != int(bg):
                    return True
    return False


def _can_place(
    canvas: np.ndarray,
    sprite: dict,
    marker_r: int,
    marker_c: int,
    bg: int,
    *,
    allow_top: bool = False,
    allow_bottom: bool = False,
    allow_left: bool = False,
    allow_right: bool = False,
) -> bool:
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    if not allow_top and top < _INNER_BORDER_GAP:
        return False
    if not allow_left and left < _INNER_BORDER_GAP:
        return False
    if not allow_bottom and top + h > H - _INNER_BORDER_GAP:
        return False
    if not allow_right and left + w > W - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _place(canvas: np.ndarray, sprite: dict, marker_r: int, marker_c: int, bg: int) -> None:
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _isolated_bg_cell(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    r0, r1 = max(0, int(r) - _ISOLATION_RADIUS), min(H, int(r) + _ISOLATION_RADIUS + 1)
    c0, c1 = max(0, int(c) - _ISOLATION_RADIUS), min(W, int(c) + _ISOLATION_RADIUS + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas: np.ndarray, r: int, c: int, bg: int) -> bool:
    H, W = canvas.shape
    if not (int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))):
        return False
    if int(r) in (0, int(H - 1)) and int(c) in (0, int(W - 1)):
        return False
    return _isolated_bg_cell(canvas, int(r), int(c), int(bg))


def _anchor_candidates(canvas: np.ndarray, bg: int, axis: str | None = None) -> list[tuple[str, int, int]]:
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    out: list[tuple[str, int, int]] = []
    if axis in (None, "h"):
        for c in cols:
            out.append(("h", 0, int(c)))
            out.append(("h", H - 1, int(c)))
    if axis in (None, "v"):
        for r in rows:
            out.append(("v", int(r), 0))
            out.append(("v", int(r), W - 1))
    random.shuffle(out)
    return [(ax, r, c) for ax, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas: np.ndarray, r: int, c: int, color: int) -> None:
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _body_color(bg: int, marker: int, used: set[int]) -> int:
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return int(random.choice(pool))


def _make_line_sprite(body_axis: str, length: int, body_color: int, marker_color: int, bg: int) -> dict:
    if body_axis == "h":
        body = {(0, int(c)) for c in range(int(length))}
        endpoint_c = 0 if random.random() < 0.5 else int(length - 1)
        marker = (-1 if random.random() < 0.5 else 1, int(endpoint_c))
    else:
        body = {(int(r), 0) for r in range(int(length))}
        endpoint_r = 0 if random.random() < 0.5 else int(length - 1)
        marker = (int(endpoint_r), -1 if random.random() < 0.5 else 1)

    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = int(body_color)
    patch[int(marker[0] - r0), int(marker[1] - c0)] = int(marker_color)
    return {
        "patch": patch,
        "offset": (int(marker[0] - r0), int(marker[1] - c0)),
        "body_axis": body_axis,
        "length": int(length),
    }


def _settled_positions(canvas: np.ndarray, sprite: dict, anchor: dict, bg: int) -> list[dict]:
    H, W = canvas.shape
    patch = sprite["patch"]
    off_r, off_c = sprite["offset"]
    h, w = patch.shape
    positions: list[dict] = []
    if anchor["axis"] == "h":
        left_marker_c = int(off_c)
        right_marker_c = int(W - (w - off_c))
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            if _can_place(canvas, sprite, r, left_marker_c, bg, allow_left=True):
                positions.append({"marker": (int(r), int(left_marker_c)), "direction": "left"})
            if _can_place(canvas, sprite, r, right_marker_c, bg, allow_right=True):
                positions.append({"marker": (int(r), int(right_marker_c)), "direction": "right"})
        return positions

    top_marker_r = int(off_r)
    bottom_marker_r = int(H - (h - off_r))
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        if _can_place(canvas, sprite, top_marker_r, c, bg, allow_top=True):
            positions.append({"marker": (int(top_marker_r), int(c)), "direction": "up"})
        if _can_place(canvas, sprite, bottom_marker_r, c, bg, allow_bottom=True):
            positions.append({"marker": (int(bottom_marker_r), int(c)), "direction": "down"})
    return positions


def _scatter_positions(canvas: np.ndarray, sprite: dict, anchor: dict, settled: dict, bg: int) -> list[tuple[int, int]]:
    H, W = canvas.shape
    mr, mc = settled["marker"]
    direction = settled["direction"]
    out: list[tuple[int, int]] = []
    if anchor["axis"] == "h":
        if direction == "left":
            cols = range(max(_INNER_BORDER_GAP, int(anchor["c"]) + 1), W - _INNER_BORDER_GAP)
        else:
            cols = range(_INNER_BORDER_GAP, min(W - _INNER_BORDER_GAP, int(anchor["c"])))
        for c in cols:
            if _can_place(canvas, sprite, int(mr), int(c), bg):
                out.append((int(mr), int(c)))
        return out

    if direction == "up":
        rows = range(max(_INNER_BORDER_GAP, int(anchor["r"]) + 1), H - _INNER_BORDER_GAP)
    else:
        rows = range(_INNER_BORDER_GAP, min(H - _INNER_BORDER_GAP, int(anchor["r"])))
    for r in rows:
        if _can_place(canvas, sprite, int(r), int(mc), bg):
            out.append((int(r), int(mc)))
    return out


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas_pair() -> tuple[int, np.ndarray, np.ndarray]:
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return int(bg), out, inp


def _place_anchors(out: np.ndarray, inp: np.ndarray, bg: int) -> tuple[list[int], list[dict]] | None:
    count = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _COLORS if c != int(bg)], count)
    axis_plan = ["h", "v"][:count]
    while len(axis_plan) < count:
        axis_plan.append(random.choice(["h", "v"]))
    random.shuffle(axis_plan)

    anchors: list[dict] = []
    for color, preferred_axis in zip(anchor_colors, axis_plan):
        candidates = _anchor_candidates(out, bg, axis=preferred_axis)
        if not candidates:
            candidates = _anchor_candidates(out, bg, axis=None)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors


def _marker_color_sequence(anchor_colors: list[int], total_sprites: int) -> list[int]:
    marker_colors = list(anchor_colors)
    while len(marker_colors) < total_sprites:
        marker_colors.append(int(random.choice(anchor_colors)))
    random.shuffle(marker_colors)
    return marker_colors[:total_sprites]


def _sprite_body_axes(total_sprites: int) -> list[str]:
    axes = ["h", "v"][:total_sprites]
    while len(axes) < total_sprites:
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)
    return axes[:total_sprites]


def _place_settled_sprites(out: np.ndarray, anchors: list[dict], marker_colors: list[int], bg: int) -> list[tuple[dict, dict, dict]] | None:
    by_color = {a["color"]: a for a in anchors}
    used_body: set[int] = set()
    direction_counts: Counter[str] = Counter()
    sprites: list[tuple[dict, dict, dict]] = []
    body_axes = _sprite_body_axes(len(marker_colors))
    for marker, body_axis in zip(marker_colors, body_axes):
        anchor = by_color[int(marker)]
        chosen: tuple[dict, int, dict] | None = None
        for _ in range(_SPRITE_RETRY_LIMIT):
            body = _body_color(bg, int(marker), used_body)
            sprite = _make_line_sprite(body_axis, random.randint(*_LINE_LENGTH_RANGE), body, int(marker), bg)
            positions = _settled_positions(out, sprite, anchor, bg)
            if not positions:
                continue
            least_used = min(direction_counts[pos["direction"]] for pos in positions)
            balanced_positions = [pos for pos in positions if direction_counts[pos["direction"]] == least_used]
            chosen = (sprite, body, random.choice(balanced_positions))
            break
        if chosen is None:
            return None
        sprite, body, settled = chosen
        used_body.add(int(body))
        direction_counts[str(settled["direction"])] += 1
        _place(out, sprite, settled["marker"][0], settled["marker"][1], bg)
        sprites.append((sprite, anchor, settled))
    return sprites


def _scatter_sprites(inp: np.ndarray, sprites: list[tuple[dict, dict, dict]], bg: int) -> bool:
    random.shuffle(sprites)
    for sprite, anchor, settled in sprites:
        positions = _scatter_positions(inp, sprite, anchor, settled, bg)
        if not positions:
            return False
        marker_r, marker_c = random.choice(positions)
        _place(inp, sprite, marker_r, marker_c, bg)
    return True


def _generated_grid_is_valid(inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt() -> np.ndarray | None:
    bg, out, inp = _new_canvas_pair()
    placed = _place_anchors(out, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors = placed

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = _marker_color_sequence(anchor_colors, total_sprites)
    sprites = _place_settled_sprites(out, anchors, marker_colors, bg)
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
