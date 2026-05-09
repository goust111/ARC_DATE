# Variant of 581f7754
# Dimension: D8 — 多步处理：精灵移动顺序 (multi_step)
# Change: 所有精灵同时移动（先统一清除原位置，再统一粘贴到新位置），无先后顺序依赖
#         → 分轮次移动（先水平后垂直），第二轮必须读取第一轮留下的中间态

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for the D8 variant
# ----------------------------------
# spec:
#   core_change:
#     original: all movable components are cleared together, then all are pasted together.
#     variant:  move horizontal-anchor sprites first, then move vertical-anchor sprites.
#   concrete_visual_effect:
#     - A horizontal sprite can try to land on cells still occupied by a vertical sprite
#       during round 1.
#     - Those target cells are blocked in round 1, so the horizontal sprite keeps holes.
#     - In round 2 the vertical sprite leaves, but the blocked horizontal pixels are not
#       replayed. The final output therefore differs from the simultaneous version.
#   traced_example:
#     - Suppose a top-edge anchor wants sprite H to slide sideways into column 8.
#     - Suppose a left-edge anchor wants sprite V to slide vertically, and V currently
#       occupies some cells inside H's future resting patch.
#     - Original solver:
#         clear H and V first -> H can paste into V's old cells -> no hole.
#     - Variant solver:
#         round 1 clears only H -> V still blocks those cells -> H loses pixels there;
#         round 2 moves V away -> the holes remain visible.
#   complexity_calibration:
#     - Every generated grid contains both axis types and at least one forced H-vs-V
#       interaction, so the order rule is observable rather than accidental.
#     - The interacting overlap is kept small (typically 2-4 cells) and avoids H's marker
#       cell, so the sprite remains recognizable.
#   implicit_assumption_audit:
#     - The original "clear all movers first" logic is no longer valid because it assumes
#       original positions never matter after move planning.
#     - Move detection itself can still be done from the input grid because no sprite moves
#       before its axis round starts, and anchors remain fixed.
#     - Round 2 may safely use the original coords for vertical sprites because only
#       horizontal sprites moved in round 1.
#   placement_legality:
#     - Input sprites are still fully inside the interior, isolated by at least one bg cell,
#       and anchors remain isolated border singletons.
#     - Solved placements are planned on a clean auxiliary canvas so aligned destinations stay
#       legible even though the actual final output may have order-induced holes.
#   observability_and_cleanliness:
#     - The generator rejects samples where sequential and simultaneous outputs are identical.
#     - It also requires at least one horizontal mover and one vertical mover, preventing
#       degeneration back to the original one-round behavior.


def _detect_moves(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
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
        S = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[dict] = []
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
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "axis": "h" if horizontal else "v",
                "marker_color": int(marker_col),
            }
        )

    return g, bg, moves


def _apply_move_batch(state: np.ndarray, source: np.ndarray, moves: list[dict], bg: int) -> None:
    H, W = state.shape
    for move in moves:
        coords = move["coords"]
        state[coords[:, 0], coords[:, 1]] = int(bg)

    for move in moves:
        dr = int(move["dr"])
        dc = int(move["dc"])
        for r, c in move["coords"]:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < H and 0 <= cc < W and int(state[rr, cc]) == int(bg):
                state[rr, cc] = int(source[int(r), int(c)])


def _simultaneous_main(grid: np.ndarray) -> np.ndarray:
    g, bg, moves = _detect_moves(grid)
    out = g.copy()
    _apply_move_batch(out, g, moves, bg)
    return out


def _order_effect_size(grid: np.ndarray) -> int:
    seq = main(grid)
    sim = _simultaneous_main(grid)
    return int(np.count_nonzero(seq != sim))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward border anchors, but do it in two axis-based rounds.

    Input structure:
    - Background = most frequent color.
    - Interior multi-color sprites each contain a minority-color marker pixel.
    - Isolated border singletons with matching colors serve as anchors and also say which axis that sprite uses.

    Rule:
    - Match each sprite to the anchor sharing its marker color.
    - Anchors on the top/bottom edges trigger horizontal sliding; anchors on the left/right edges trigger vertical sliding.
    - Apply the movement in two stages: first all horizontal sliders, then all vertical sliders. The second round starts from the first round's result, so blocking from still-unmoved sprites matters.

    Key insight: You must separate move planning from move execution: anchor position still picks the axis, but the final picture depends on the fixed round order because horizontal moves happen before vertical ones vacate their old cells.
    """
    g, bg, moves = _detect_moves(grid)
    out = g.copy()
    horizontal = [move for move in moves if move["axis"] == "h"]
    vertical = [move for move in moves if move["axis"] == "v"]
    _apply_move_batch(out, g, horizontal, bg)
    _apply_move_batch(out, g, vertical, bg)
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
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 400

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

_INTERACTION_MIN_OVERLAP = 2
_INTERACTION_MAX_OVERLAP = 4

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
        removable = [(r, c) for r, c in cells if not (r in (0, h - 1) and c in (0, w - 1))]
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
        return [(r, anchor["c"]) for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP) if _can_place(canvas, sprite, r, anchor["c"], bg)]
    return [(anchor["r"], c) for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP) if _can_place(canvas, sprite, anchor["r"], c, bg)]


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [(mr, c) for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP) if c != mc and _can_place(canvas, sprite, mr, c, bg)]
    return [(r, mc) for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP) if r != mr and _can_place(canvas, sprite, r, mc, bg)]


def _sprite_cells(sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    patch = sprite["patch"]
    h, w = patch.shape
    cells = {}
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v != int(bg):
                cells[(top + r, left + c)] = v
    return cells


def _interaction_body_overlap(h_sprite, h_solved, h_marker_color, v_sprite, v_input, bg):
    h_cells = _sprite_cells(h_sprite, h_solved[0], h_solved[1], bg)
    v_cells = _sprite_cells(v_sprite, v_input[0], v_input[1], bg)
    overlap = set(h_cells) & set(v_cells)
    marker_coord = (int(h_solved[0]), int(h_solved[1]))
    if marker_coord in overlap:
        return []
    return [cell for cell in overlap if int(h_cells[cell]) != int(h_marker_color)]


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
    n_anchors = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _COLORS if c != bg], n_anchors)
    axes = ["h", "v"]
    if n_anchors == 3:
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)

    anchors = []
    for color, needed_axis in zip(anchor_colors, axes):
        candidates = [cand for cand in _anchor_candidates(out, bg) if cand[0] == needed_axis]
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(out, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchors


def _far_positions(anchor, solved, positions):
    if anchor["axis"] == "h":
        strong = [pos for pos in positions if abs(int(pos[1]) - int(solved[1])) >= 2]
    else:
        strong = [pos for pos in positions if abs(int(pos[0]) - int(solved[0])) >= 2]
    return strong or positions


def _place_interacting_pair(out, inp, anchors, bg, used_body):
    h_anchors = [a for a in anchors if a["axis"] == "h"]
    v_anchors = [a for a in anchors if a["axis"] == "v"]
    if not h_anchors or not v_anchors:
        return None

    pair_choices = [(h_anchor, v_anchor) for h_anchor in h_anchors for v_anchor in v_anchors]
    random.shuffle(pair_choices)

    for h_anchor, v_anchor in pair_choices:
        for _ in range(90):
            h_body = _body_color(bg, h_anchor["color"], used_body)
            v_body = _body_color(bg, v_anchor["color"], used_body | {h_body})
            h_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), h_body, h_anchor["color"], bg)
            v_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), v_body, v_anchor["color"], bg)

            v_solved_positions = _aligned_positions(out, v_sprite, v_anchor, bg)
            random.shuffle(v_solved_positions)
            for v_solved in v_solved_positions:
                out_after_v = out.copy()
                _place(out_after_v, v_sprite, v_solved[0], v_solved[1], bg)
                h_solved_positions = _aligned_positions(out_after_v, h_sprite, h_anchor, bg)
                random.shuffle(h_solved_positions)

                v_input_positions = _far_positions(v_anchor, v_solved, _scatter_positions(inp, v_sprite, v_anchor, v_solved, bg))
                random.shuffle(v_input_positions)
                for v_input in v_input_positions:
                    temp_in = inp.copy()
                    _place(temp_in, v_sprite, v_input[0], v_input[1], bg)

                    for h_solved in h_solved_positions:
                        overlap = _interaction_body_overlap(h_sprite, h_solved, h_anchor["color"], v_sprite, v_input, bg)
                        if not (_INTERACTION_MIN_OVERLAP <= len(overlap) <= _INTERACTION_MAX_OVERLAP):
                            continue

                        h_input_positions = _far_positions(h_anchor, h_solved, _scatter_positions(temp_in, h_sprite, h_anchor, h_solved, bg))
                        random.shuffle(h_input_positions)
                        for h_input in h_input_positions:
                            _place(out, v_sprite, v_solved[0], v_solved[1], bg)
                            _place(out, h_sprite, h_solved[0], h_solved[1], bg)
                            _place(inp, v_sprite, v_input[0], v_input[1], bg)
                            _place(inp, h_sprite, h_input[0], h_input[1], bg)
                            used_body.update({h_body, v_body})
                            return [
                                {"sprite": h_sprite, "anchor": h_anchor, "solved": h_solved, "input": h_input},
                                {"sprite": v_sprite, "anchor": v_anchor, "solved": v_solved, "input": v_input},
                            ]
    return None


def _place_extra_sprites(out, inp, anchors, bg, used_body, n_extra):
    sprites = []
    anchor_by_color = {anchor["color"]: anchor for anchor in anchors}
    marker_colors = [random.choice(list(anchor_by_color)) for _ in range(n_extra)]
    for marker in marker_colors:
        anchor = anchor_by_color[int(marker)]
        placed = False
        for _ in range(80):
            body = _body_color(bg, marker, used_body)
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            solved_positions = _aligned_positions(out, sprite, anchor, bg)
            random.shuffle(solved_positions)
            for solved in solved_positions:
                input_positions = _far_positions(anchor, solved, _scatter_positions(inp, sprite, anchor, solved, bg))
                if not input_positions:
                    continue
                inp_pos = random.choice(input_positions)
                _place(out, sprite, solved[0], solved[1], bg)
                _place(inp, sprite, inp_pos[0], inp_pos[1], bg)
                used_body.add(body)
                sprites.append({"sprite": sprite, "anchor": anchor, "solved": solved, "input": inp_pos})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return sprites


def _move_axis_counts(inp):
    _g, _bg, moves = _detect_moves(inp)
    horizontal = sum(1 for move in moves if move["axis"] == "h")
    vertical = sum(1 for move in moves if move["axis"] == "v")
    return horizontal, vertical


def _generated_grid_is_valid(inp, bg):
    if not _bg_is_mode(inp, bg):
        return False
    try:
        horizontal, vertical = _move_axis_counts(inp)
        if horizontal == 0 or vertical == 0:
            return False
        if _order_effect_size(inp) < _INTERACTION_MIN_OVERLAP:
            return False
        main(inp)
    except Exception:
        return False
    return True


def _gen_attempt():
    bg, out, inp = _new_canvas_pair()
    anchors = _place_anchors(out, inp, bg)
    if anchors is None:
        return None

    used_body = set()
    pair = _place_interacting_pair(out, inp, anchors, bg, used_body)
    if pair is None:
        return None

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    extras = _place_extra_sprites(out, inp, anchors, bg, used_body, total_sprites - len(pair))
    if extras is None:
        return None

    if not _generated_grid_is_valid(inp, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D8 input")
