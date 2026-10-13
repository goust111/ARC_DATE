import random
from collections import defaultdict, deque

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 7)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
SIDE_ORDER = ("up", "down", "left", "right")
GEN_RETRIES = 200


BASE_SHAPES = {
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ],
}


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_orientations(cells):
    cur = list(cells)
    seen = set()
    out = []
    for _ in range(4):
        for reflected in (False, True):
            pts = [(r, -c) if reflected else (r, c) for r, c in cur]
            norm = _normalize(pts)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        cur = [(c, -r) for r, c in cur]
    return out


SHAPES_BY_AREA = {
    area: [ori for base in bases for ori in _shape_orientations(base)]
    for area, bases in BASE_SHAPES.items()
}


def _bbox(coords):
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_coords(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _divider_lines(g, axis):
    h, w = g.shape
    if axis == 0:
        threshold = max(3, w // 2)
        return [r for r in range(h) if int(np.count_nonzero(g[r] == WALL)) >= threshold]
    threshold = max(3, h // 2)
    return [c for c in range(w) if int(np.count_nonzero(g[:, c] == WALL)) >= threshold]


def _parse_layout(g):
    div_rows = _divider_lines(g, axis=0)
    div_cols = _divider_lines(g, axis=1)
    assert len(div_rows) >= 2 and len(div_cols) >= 2
    assert div_rows[0] == 0 and div_rows[-1] == g.shape[0] - 1
    assert div_cols[0] == 0 and div_cols[-1] == g.shape[1] - 1

    rooms = {}
    for rr in range(len(div_rows) - 1):
        for cc in range(len(div_cols) - 1):
            r0 = div_rows[rr] + 1
            r1 = div_rows[rr + 1] - 1
            c0 = div_cols[cc] + 1
            c1 = div_cols[cc + 1] - 1
            assert r0 <= r1 and c0 <= c1
            rooms[(rr, cc)] = (r0, r1, c0, c1)
    return div_rows, div_cols, rooms


def _wall_sides_with_gaps(g, bounds, div_rows, div_cols):
    r0, r1, c0, c1 = bounds
    top = div_rows[div_rows.index(r0 - 1)]
    bottom = div_rows[div_rows.index(r1 + 1)]
    left = div_cols[div_cols.index(c0 - 1)]
    right = div_cols[div_cols.index(c1 + 1)]
    gaps = []
    if np.any(g[top, c0 : c1 + 1] == BG):
        gaps.append("up")
    if np.any(g[bottom, c0 : c1 + 1] == BG):
        gaps.append("down")
    if np.any(g[r0 : r1 + 1, left] == BG):
        gaps.append("left")
    if np.any(g[r0 : r1 + 1, right] == BG):
        gaps.append("right")
    return gaps


def _room_object(g, bounds):
    r0, r1, c0, c1 = bounds
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    mask = (sub != BG) & (sub != WALL)
    if not np.any(mask):
        return None
    comps = _component_coords(mask)
    assert len(comps) == 1
    coords = comps[0].copy()
    coords[:, 0] += r0
    coords[:, 1] += c0
    colors = {int(g[r, c]) for r, c in coords}
    assert len(colors) == 1
    return {
        "coords": coords,
        "color": colors.pop(),
        "size": int(coords.shape[0]),
    }


def _neighbor(room_id, side, nrows, ncols):
    r, c = room_id
    dr, dc = DIRS[side]
    nr, nc = r + dr, c + dc
    assert 0 <= nr < nrows and 0 <= nc < ncols
    return nr, nc


def _translated_coords(coords, target_bounds, side):
    r0, r1, c0, c1 = target_bounds
    rr0, cc0, rr1, cc1 = _bbox(coords)
    if side == "right":
        shift_r, shift_c = 0, c1 - cc1
    elif side == "left":
        shift_r, shift_c = 0, c0 - cc0
    elif side == "down":
        shift_r, shift_c = r1 - rr1, 0
    else:
        shift_r, shift_c = r0 - rr0, 0
    moved = coords.copy()
    moved[:, 0] += shift_r
    moved[:, 1] += shift_c
    return moved


def _winner_color(items, fallback):
    if not items:
        return fallback
    max_size = max(item["size"] for item in items)
    winners = [item for item in items if item["size"] == max_size]
    if len(winners) == 1:
        return winners[0]["color"]
    return fallback


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each room's monochrome object toward its unique wall gap.

    Occupied rooms move according to the side of their only gap. If the destination
    room is empty, the object relocates there and stops flush against the far wall.
    If multiple objects enter the same empty room, a unique larger one recolors the
    smaller arrivals; equal-largest arrivals keep their own colors. If the destination
    room already contains an object, the mover acts as dye: the resident stays in its
    room and is recolored by the unique largest incoming color, while the moving copy
    disappears. This also handles paired occupied rooms sharing the same doorway.
    """
    g = np.asarray(grid, dtype=int)
    div_rows, div_cols, rooms = _parse_layout(g)
    nrows = len(div_rows) - 1
    ncols = len(div_cols) - 1

    objects = {}
    exits = {}
    plans = {}
    for room_id, bounds in rooms.items():
        obj = _room_object(g, bounds)
        if obj is None:
            continue
        gap_sides = _wall_sides_with_gaps(g, bounds, div_rows, div_cols)
        assert len(gap_sides) == 1
        side = gap_sides[0]
        target = _neighbor(room_id, side, nrows, ncols)
        objects[room_id] = obj
        exits[room_id] = side
        plans[room_id] = {"target": target, "side": side, "obj": obj}

    occupied = set(objects)
    incoming_to_occupied = defaultdict(list)
    incoming_to_empty = defaultdict(list)

    for room_id, plan in plans.items():
        target = plan["target"]
        if target in occupied:
            incoming_to_occupied[target].append(plan["obj"])
        else:
            incoming_to_empty[target].append((room_id, plan))

    out = g.copy()
    out[(out != BG) & (out != WALL)] = BG

    for room_id, obj in objects.items():
        if room_id not in incoming_to_occupied:
            continue
        color = _winner_color(incoming_to_occupied[room_id], obj["color"])
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = color

    for target, movers in incoming_to_empty.items():
        # If a room receives incoming dye from an occupied neighbor, it stays put and does not move.
        live = [plan for source, plan in movers if source not in incoming_to_occupied]
        if not live:
            continue
        sizes = [plan["obj"]["size"] for plan in live]
        max_size = max(sizes)
        winner_color = None
        if sizes.count(max_size) == 1:
            winner_color = live[sizes.index(max_size)]["obj"]["color"]
        for plan in live:
            obj = plan["obj"]
            color = obj["color"]
            if winner_color is not None and obj["size"] < max_size:
                color = winner_color
            moved = _translated_coords(obj["coords"], rooms[target], plan["side"])
            out[moved[:, 0], moved[:, 1]] = color

    return out


def _pick_shape(rng, area, max_h=None, max_w=None):
    choices = []
    for shape in SHAPES_BY_AREA[area]:
        h = 1 + max(r for r, _ in shape)
        w = 1 + max(c for _, c in shape)
        if max_h is not None and h > max_h:
            continue
        if max_w is not None and w > max_w:
            continue
        choices.append(shape)
    assert choices
    return rng.choice(choices)


def _place_shape(rng, grid, bounds, shape, color):
    r0, r1, c0, c1 = bounds
    shape_h = 1 + max(r for r, _ in shape)
    shape_w = 1 + max(c for _, c in shape)
    top = rng.randint(r0 + 1, r1 - shape_h)
    left = rng.randint(c0 + 1, c1 - shape_w)
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _build_empty_grid(rng):
    room_rows = rng.randint(3, 4)
    room_cols = rng.randint(3, 4)
    max_h = 7 if room_rows == 3 else 6
    max_w = 7 if room_cols == 3 else 6
    heights = [rng.randint(5, max_h) for _ in range(room_rows)]
    widths = [rng.randint(5, max_w) for _ in range(room_cols)]
    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    g = np.full((h, w), BG, dtype=int)

    r = 0
    row_lines = [0]
    g[0, :] = WALL
    for hh in heights:
        r += hh + 1
        row_lines.append(r)
        g[r, :] = WALL

    c = 0
    col_lines = [0]
    g[:, 0] = WALL
    for ww in widths:
        c += ww + 1
        col_lines.append(c)
        g[:, c] = WALL

    rooms = {}
    cur_r = 1
    for rr, hh in enumerate(heights):
        cur_c = 1
        for cc, ww in enumerate(widths):
            rooms[(rr, cc)] = (cur_r, cur_r + hh - 1, cur_c, cur_c + ww - 1)
            cur_c += ww + 1
        cur_r += hh + 1
    return g, row_lines, col_lines, rooms


def _carve_gap(rng, grid, bounds_a, bounds_b):
    ar0, ar1, ac0, ac1 = bounds_a
    br0, br1, bc0, bc1 = bounds_b
    if ar0 == br0 and ar1 == br1:
        wall_col = ac1 + 1 if ac1 < bc0 else bc1 + 1
        gap_r = rng.randint(ar0, ar1)
        grid[gap_r, wall_col] = BG
        return
    wall_row = ar1 + 1 if ar1 < br0 else br1 + 1
    gap_c = rng.randint(ac0, ac1)
    grid[wall_row, gap_c] = BG


def _adjacent_pairs(room_ids):
    out = []
    room_set = set(room_ids)
    for r, c in room_ids:
        for side in ("right", "down"):
            dr, dc = DIRS[side]
            nb = (r + dr, c + dc)
            if nb in room_set:
                out.append(((r, c), nb))
    return out


def _hub_candidates(room_rows, room_cols):
    cand = []
    for r in range(room_rows):
        for c in range(1, room_cols - 1):
            cand.append(((r, c - 1), (r, c), (r, c + 1), "horizontal"))
    for r in range(1, room_rows - 1):
        for c in range(room_cols):
            cand.append(((r - 1, c), (r, c), (r + 1, c), "vertical"))
    return cand


def _find_non_overlapping_scenarios(rng, room_rows, room_cols):
    room_ids = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    hubs = _hub_candidates(room_rows, room_cols)
    rng.shuffle(hubs)
    pairs = _adjacent_pairs(room_ids)
    for left, center, right, axis in hubs:
        used = {left, center, right}
        pair_choices = [pair for pair in pairs if pair[0] not in used and pair[1] not in used]
        rng.shuffle(pair_choices)
        if not pair_choices:
            continue
        pair = pair_choices[0]
        used_pair = used | set(pair)
        simple_pool = [p for p in pairs if p[0] not in used_pair and p[1] not in used_pair]
        rng.shuffle(simple_pool)
        simple_count = rng.choice((0, 1, 2))
        simples = []
        for p in simple_pool:
            if len(simples) == simple_count:
                break
            if p[0] in used_pair or p[1] in used_pair:
                continue
            if any(p[0] in set(q) or p[1] in set(q) for q in simples):
                continue
            simples.append(p)
        return {
            "hub": (left, center, right, axis),
            "pair": pair,
            "simples": simples,
        }
    return None


def _distinct_colors(rng, count):
    return rng.sample(PALETTE, count)


def _gen_attempt(rng):
    g, row_lines, col_lines, rooms = _build_empty_grid(rng)
    room_rows = len(row_lines) - 1
    room_cols = len(col_lines) - 1
    plan = _find_non_overlapping_scenarios(rng, room_rows, room_cols)
    if plan is None:
        return None

    hub_left, hub_center, hub_right, hub_axis = plan["hub"]
    hub_equal = rng.random() < 0.5
    if hub_equal:
        area_a = area_b = rng.choice((2, 3, 4))
    else:
        area_a, area_b = rng.sample((2, 3, 4), 2)

    pair_a, pair_b = plan["pair"]
    pair_colors = _distinct_colors(rng, 2)
    hub_colors = _distinct_colors(rng, 2)

    if hub_axis == "horizontal":
        room_w = rooms[hub_center][3] - rooms[hub_center][2] + 1
        shape_a = _pick_shape(rng, area_a, max_w=room_w - 2)
        shape_b = _pick_shape(rng, area_b, max_w=room_w - 2)
        wa = 1 + max(c for _, c in shape_a)
        wb = 1 + max(c for _, c in shape_b)
        if wa + wb > room_w:
            return None
    else:
        room_h = rooms[hub_center][1] - rooms[hub_center][0] + 1
        shape_a = _pick_shape(rng, area_a, max_h=room_h - 2)
        shape_b = _pick_shape(rng, area_b, max_h=room_h - 2)
        ha = 1 + max(r for r, _ in shape_a)
        hb = 1 + max(r for r, _ in shape_b)
        if ha + hb > room_h:
            return None

    _carve_gap(rng, g, rooms[hub_left], rooms[hub_center])
    _carve_gap(rng, g, rooms[hub_right], rooms[hub_center])
    _place_shape(rng, g, rooms[hub_left], shape_a, hub_colors[0])
    _place_shape(rng, g, rooms[hub_right], shape_b, hub_colors[1])

    _carve_gap(rng, g, rooms[pair_a], rooms[pair_b])
    pair_shape_a = _pick_shape(rng, rng.choice((2, 3, 4)))
    pair_shape_b = _pick_shape(rng, rng.choice((2, 3, 4)))
    _place_shape(rng, g, rooms[pair_a], pair_shape_a, pair_colors[0])
    _place_shape(rng, g, rooms[pair_b], pair_shape_b, pair_colors[1])

    for a, b in plan["simples"]:
        source, target = (a, b) if rng.random() < 0.5 else (b, a)
        _carve_gap(rng, g, rooms[source], rooms[target])
        color = rng.choice(PALETTE)
        area = rng.choice((2, 3, 4))
        shape = _pick_shape(rng, area)
        _place_shape(rng, g, rooms[source], shape, color)

    # Prevent accidental grey noise from changing the parser's assumptions.
    parsed_rows, parsed_cols, parsed_rooms = _parse_layout(g)
    if len(parsed_rows) != len(row_lines) or len(parsed_cols) != len(col_lines):
        return None
    occupied = 0
    for room_id, bounds in parsed_rooms.items():
        obj = _room_object(g, bounds)
        if obj is None:
            continue
        occupied += 1
        gap_sides = _wall_sides_with_gaps(g, bounds, parsed_rows, parsed_cols)
        if len(gap_sides) != 1:
            return None
    if occupied < 4:
        return None
    out = main(g)
    if np.array_equal(g, out):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-slide puzzle")
