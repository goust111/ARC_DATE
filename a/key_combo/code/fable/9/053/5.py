from __future__ import annotations

import random

import numpy as np


WALL = 5
BG = 0
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _component_bfs(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg or color == WALL:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": pts, "bbox": _bbox(pts)})
    return comps


def _touch_side(box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each moving rectangle must touch exactly one border"
    return sides[0]


def _direction_for_side(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    assert bg != WALL, "background must differ from grey walls"
    walls = g == WALL
    rects = []
    for comp in _component_bfs(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "each mover must be a solid rectangle"
        side = _touch_side(comp["bbox"], g.shape)
        dr, dc = _direction_for_side(side)
        rects.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "side": side,
                "dr": dr,
                "dc": dc,
            }
        )
    assert rects, "expected at least one edge-touching rectangle"
    return bg, walls, rects


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _solve(grid: np.ndarray, *, return_states: bool = False):
    g = np.asarray(grid, dtype=int)
    bg, walls, rects = _parse_scene(g)
    h, w = g.shape
    states = []
    global_claim = np.zeros((h, w), dtype=bool)

    for rect in rects:
        claim = np.zeros((h, w), dtype=bool)
        r0, c0 = rect["r0"], rect["c0"]
        rr = slice(r0, r0 + rect["h"])
        cc = slice(c0, c0 + rect["w"])
        claim[rr, cc] = True
        global_claim[rr, cc] = True
        states.append({**rect, "claim": claim, "active": True})

    for _step in range(h + w + 5):
        active = [state for state in states if state["active"]]
        if not active:
            break

        candidates: dict[int, tuple[int, int, int, int]] = {}
        blocked: set[int] = set()
        for state in active:
            nr0 = state["r0"] + state["dr"]
            nc0 = state["c0"] + state["dc"]
            nr1 = nr0 + state["h"] - 1
            nc1 = nc0 + state["w"] - 1
            color = int(state["color"])
            if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
                blocked.add(color)
                continue
            if np.any(walls[nr0 : nr1 + 1, nc0 : nc1 + 1]):
                blocked.add(color)
                continue
            occupied = global_claim[nr0 : nr1 + 1, nc0 : nc1 + 1]
            own = state["claim"][nr0 : nr1 + 1, nc0 : nc1 + 1]
            if np.any(occupied & ~own):
                blocked.add(color)
                continue
            candidates[color] = (nr0, nc0, nr1, nc1)

        cand_items = [(state, candidates[state["color"]]) for state in active if state["color"] in candidates]
        for i in range(len(cand_items)):
            state_i, box_i = cand_items[i]
            if state_i["color"] in blocked:
                continue
            for j in range(i + 1, len(cand_items)):
                state_j, box_j = cand_items[j]
                if _boxes_overlap(box_i, box_j):
                    blocked.add(int(state_i["color"]))
                    blocked.add(int(state_j["color"]))

        movers = []
        for state in active:
            if state["color"] in candidates and state["color"] not in blocked:
                movers.append(state)
            else:
                state["active"] = False

        if not movers:
            break

        for state in movers:
            nr0, nc0, nr1, nc1 = candidates[state["color"]]
            state["r0"] = nr0
            state["c0"] = nc0
            state["claim"][nr0 : nr1 + 1, nc0 : nc1 + 1] = True
            global_claim[nr0 : nr1 + 1, nc0 : nc1 + 1] = True

    out = np.full((h, w), bg, dtype=int)
    out[walls] = WALL
    for state in states:
        out[state["r0"] : state["r0"] + state["h"], state["c0"] : state["c0"] + state["w"]] = state["color"]

    if return_states:
        return out, states
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide every edge-touching rectangle synchronously to its opposite side, with walls, shared channels, and earlier arrivals blocking later ones.

    Input structure:
    - The dominant background contains solid colored rectangles, each touching exactly one outer border and therefore pointing toward the opposite border.
    - Grey cells are fixed wall segments inside the arena.
    - Horizontal and vertical slide corridors may cross, and multiple rectangles may move in opposite directions along the same corridor.

    Rule:
    - All rectangles start moving at the same speed, one cell per tick, in the direction implied by the border they touch.
    - A rectangle stops forever when its next step would leave the grid or overlap a grey wall, a cell already claimed by another rectangle, or another rectangle's simultaneous next position.
    - Earlier arrivals permanently claim the cells they have occupied, so in an unobstructed head-on corridor two opposite rectangles split the gap in half, while at a crossing the first rectangle through the overlap blocks the later one.

    Key insight:
    - The puzzle is defined by synchronous motion history, not by independently sliding each rectangle to a static target: midway meetings, wall-shortened races, and crossing priority all come from the same shared-time simulation.
    """
    return _solve(np.asarray(grid, dtype=int))


def _make_object(side: str, band0: int, band_size: int, depth: int, color: int, h: int, w: int) -> dict:
    if side == "left":
        return {"color": color, "r0": band0, "c0": 0, "h": band_size, "w": depth, "side": side}
    if side == "right":
        return {"color": color, "r0": band0, "c0": w - depth, "h": band_size, "w": depth, "side": side}
    if side == "top":
        return {"color": color, "r0": 0, "c0": band0, "h": depth, "w": band_size, "side": side}
    if side == "bottom":
        return {"color": color, "r0": h - depth, "c0": band0, "h": depth, "w": band_size, "side": side}
    raise ValueError(side)


def _render(shape: tuple[int, int], objects: list[dict], walls: np.ndarray) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[walls] = WALL
    for obj in objects:
        r0, c0 = obj["r0"], obj["c0"]
        grid[r0 : r0 + obj["h"], c0 : c0 + obj["w"]] = obj["color"]
    return grid


def _interval_candidates(limit: int, size: int, used: list[tuple[int, int]], *, margin: int = 1, gap: int = 1) -> list[int]:
    starts = []
    for start in range(margin, limit - size - margin + 1):
        end = start + size - 1
        ok = True
        for u0, u1 in used:
            if not (end + gap < u0 or u1 + gap < start):
                ok = False
                break
        if ok:
            starts.append(start)
    return starts


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(np.asarray(grid, dtype=int) == int(color))
    assert pts.size > 0
    return _bbox(pts)


def _side_for_color(grid: np.ndarray, color: int) -> str:
    return _touch_side(_bbox_for_color(grid, color), grid.shape)


def _travel_distance(inp: np.ndarray, out: np.ndarray, color: int) -> int:
    ib = _bbox_for_color(inp, color)
    ob = _bbox_for_color(out, color)
    side = _touch_side(ib, inp.shape)
    if side == "left":
        return int(ob[1] - ib[1])
    if side == "right":
        return int(ib[1] - ob[1])
    if side == "top":
        return int(ob[0] - ib[0])
    return int(ib[0] - ob[0])


def _channel_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if (a[0], a[2]) == (b[0], b[2]):
        left, right = sorted((a, b), key=lambda box: box[1])
        return int(right[1] - left[3] - 1)
    assert (a[1], a[3]) == (b[1], b[3])
    top, bottom = sorted((a, b), key=lambda box: box[0])
    return int(bottom[0] - top[2] - 1)


def _touches_opposite(box: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> bool:
    h, w = shape
    if side == "left":
        return box[3] == w - 1
    if side == "right":
        return box[1] == 0
    if side == "top":
        return box[2] == h - 1
    return box[0] == 0


def _erase_color(grid: np.ndarray, color: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == int(color)] = BG
    return out


def _erase_walls(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == WALL] = BG
    return out


def _transpose_if_needed(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return grid.T.copy()
    return grid


def _generate_meet(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(12, 20)
    w = rng.randint(14, 24)
    band = rng.randint(1, 3)
    left_depth = rng.randint(1, 3)
    right_depth = rng.randint(1, 3)
    if w - left_depth - right_depth < 5:
        return None
    row_options = _interval_candidates(h, band, [])
    if not row_options:
        return None
    row0 = rng.choice(row_options)
    colors = rng.sample(RECT_COLORS, 3)
    objects = [
        _make_object("left", row0, band, left_depth, colors[0], h, w),
        _make_object("right", row0, band, right_depth, colors[1], h, w),
    ]
    free_colors: list[int] = []
    if rng.random() < 0.7:
        free_band = rng.randint(1, 3)
        extra_rows = _interval_candidates(h, free_band, [(row0, row0 + band - 1)])
        if extra_rows:
            extra_row0 = rng.choice(extra_rows)
            free_side = rng.choice(("left", "right"))
            free_depth = rng.randint(1, 3)
            objects.append(_make_object(free_side, extra_row0, free_band, free_depth, colors[2], h, w))
            free_colors.append(colors[2])
    walls = np.zeros((h, w), dtype=bool)
    grid = _render((h, w), objects, walls)
    return _transpose_if_needed(grid, rng), {"type": "meet", "pair": (colors[0], colors[1]), "free": tuple(free_colors)}


def _generate_wall(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(12, 20)
    w = rng.randint(15, 24)
    band = rng.randint(1, 3)
    left_depth = rng.randint(1, 3)
    right_depth = rng.randint(1, 3)
    gap = w - left_depth - right_depth
    if gap < 6:
        return None
    row_options = _interval_candidates(h, band, [])
    if not row_options:
        return None
    row0 = rng.choice(row_options)
    colors = rng.sample(RECT_COLORS, 3)
    objects = [
        _make_object("left", row0, band, left_depth, colors[0], h, w),
        _make_object("right", row0, band, right_depth, colors[1], h, w),
    ]
    blocked_color = colors[0] if rng.random() < 0.5 else colors[1]
    free_colors: list[int] = []
    if blocked_color == colors[0]:
        stop_dist = rng.randint(1, max(1, gap // 2 - 1))
        wall_col = left_depth + stop_dist
    else:
        stop_dist = rng.randint(1, max(1, gap // 2 - 1))
        wall_col = w - right_depth - 1 - stop_dist
    walls = np.zeros((h, w), dtype=bool)
    wall_r0 = max(1, row0 - 1)
    wall_r1 = min(h - 2, row0 + band)
    walls[wall_r0 : wall_r1 + 1, wall_col] = True
    if rng.random() < 0.6:
        free_band = rng.randint(1, 3)
        extra_rows = _interval_candidates(h, free_band, [(row0, row0 + band - 1)])
        if extra_rows:
            extra_row0 = rng.choice(extra_rows)
            free_side = rng.choice(("left", "right"))
            free_depth = rng.randint(1, 3)
            objects.append(_make_object(free_side, extra_row0, free_band, free_depth, colors[2], h, w))
            free_colors.append(colors[2])
    grid = _render((h, w), objects, walls)
    return _transpose_if_needed(grid, rng), {
        "type": "wall",
        "pair": (colors[0], colors[1]),
        "blocked": blocked_color,
        "free": tuple(free_colors),
    }


def _cross_time(obj: dict, other: dict, shape: tuple[int, int]) -> int:
    h, w = shape
    if obj["side"] == "left":
        return int(other["c0"] - obj["w"] + 1)
    if obj["side"] == "right":
        return int((w - obj["w"]) - (other["c0"] + other["w"] - 1))
    if obj["side"] == "top":
        return int(other["r0"] - obj["h"] + 1)
    return int((h - obj["h"]) - (other["r0"] + other["h"] - 1))


def _generate_cross(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    h_band = rng.randint(1, 3)
    h_depth = rng.randint(1, 3)
    v_band = rng.randint(1, 3)
    v_depth = rng.randint(1, 3)
    row_options = _interval_candidates(h, h_band, [], margin=max(2, v_depth + 1))
    col_options = _interval_candidates(w, v_band, [], margin=max(2, h_depth + 1))
    if not row_options or not col_options:
        return None
    row0 = rng.choice(row_options)
    col0 = rng.choice(col_options)
    colors = rng.sample(RECT_COLORS, 2)
    h_side = rng.choice(("left", "right"))
    v_side = rng.choice(("top", "bottom"))
    h_obj = _make_object(h_side, row0, h_band, h_depth, colors[0], h, w)
    v_obj = _make_object(v_side, col0, v_band, v_depth, colors[1], h, w)
    t_h = _cross_time(h_obj, v_obj, (h, w))
    t_v = _cross_time(v_obj, h_obj, (h, w))
    if min(t_h, t_v) < 1 or t_h == t_v:
        return None
    winner = colors[0] if t_h < t_v else colors[1]
    loser = colors[1] if winner == colors[0] else colors[0]
    walls = np.zeros((h, w), dtype=bool)
    grid = _render((h, w), [h_obj, v_obj], walls)
    return grid, {"type": "cross", "winner": winner, "loser": loser, "horizontal": colors[0], "vertical": colors[1]}


def _validate_meet(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    a, b = meta["pair"]
    in_gap = _channel_gap(_bbox_for_color(inp, a), _bbox_for_color(inp, b))
    out_gap = _channel_gap(_bbox_for_color(out, a), _bbox_for_color(out, b))
    assert out_gap == in_gap % 2
    for color in meta["free"]:
        side = _side_for_color(inp, color)
        assert _touches_opposite(_bbox_for_color(out, color), side, inp.shape)


def _validate_wall(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    no_wall_out = main(_erase_walls(inp))
    assert not np.array_equal(no_wall_out, out)
    blocked = meta["blocked"]
    assert _travel_distance(inp, out, blocked) < _travel_distance(inp, no_wall_out, blocked)
    a, b = meta["pair"]
    assert _channel_gap(_bbox_for_color(out, a), _bbox_for_color(out, b)) == 0
    for color in meta["free"]:
        side = _side_for_color(inp, color)
        assert _touches_opposite(_bbox_for_color(out, color), side, inp.shape)


def _validate_cross(inp: np.ndarray, out: np.ndarray, meta: dict) -> None:
    winner = meta["winner"]
    loser = meta["loser"]
    loser_without_winner = main(_erase_color(inp, winner))
    assert _travel_distance(inp, loser_without_winner, loser) > _travel_distance(inp, out, loser)

    h_color = meta["horizontal"]
    v_color = meta["vertical"]
    h_box = _bbox_for_color(inp, h_color)
    v_box = _bbox_for_color(inp, v_color)
    overlap = (h_box[0], v_box[1], h_box[2], v_box[3])
    loser_box = _bbox_for_color(out, loser)
    loser_side = _side_for_color(inp, loser)
    if loser_side == "left":
        assert loser_box[3] == overlap[1] - 1
    elif loser_side == "right":
        assert loser_box[1] == overlap[3] + 1
    elif loser_side == "top":
        assert loser_box[2] == overlap[0] - 1
    else:
        assert loser_box[0] == overlap[2] + 1
    winner_side = _side_for_color(inp, winner)
    assert _touches_opposite(_bbox_for_color(out, winner), winner_side, inp.shape)


def _validate_generated(inp: np.ndarray, meta: dict) -> bool:
    try:
        out = main(inp)
        assert inp.shape == out.shape
        assert not np.array_equal(inp, out)
        assert _bg_color(inp) == BG
        assert _bg_color(out) == BG
        in_colors = set(np.unique(inp).tolist())
        out_colors = set(np.unique(out).tolist())
        assert in_colors == out_colors
        for color in out_colors - {BG, WALL}:
            ib = _bbox_for_color(inp, int(color))
            ob = _bbox_for_color(out, int(color))
            assert (ib[2] - ib[0], ib[3] - ib[1]) == (ob[2] - ob[0], ob[3] - ob[1])
        if meta["type"] == "meet":
            _validate_meet(inp, out, meta)
        elif meta["type"] == "wall":
            _validate_wall(inp, out, meta)
        else:
            _validate_cross(inp, out, meta)
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    generators = (
        (0.30, _generate_meet),
        (0.35, _generate_wall),
        (0.35, _generate_cross),
    )
    weights = [item[0] for item in generators]
    funcs = [item[1] for item in generators]
    for _ in range(MAX_GEN_ATTEMPTS):
        gen = rng.choices(funcs, weights=weights, k=1)[0]
        built = gen(rng)
        if built is None:
            continue
        grid, meta = built
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid synchronized-sliding input")
