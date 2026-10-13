from collections import Counter, deque

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, find_components, get_bg, paste
import random


def _distances_from(sources: np.ndarray, passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        rr = int(r)
        cc = int(c)
        assert passable[rr, cc]
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _framed_square_rooms(g: np.ndarray, wall: int, sizes: set[int]) -> list[tuple[int, int, int]]:
    h, w = g.shape
    rooms: list[tuple[int, int, int]] = []
    for n in sorted(sizes):
        assert n > 1
        for r in range(h - n + 1):
            for c in range(w - n + 1):
                if np.any(g[r : r + n, c : c + n] == wall):
                    continue

                side_wall_counts: list[int] = []
                if r > 0:
                    side_wall_counts.append(int(np.count_nonzero(g[r - 1, c : c + n] == wall)))
                if r + n < h:
                    side_wall_counts.append(int(np.count_nonzero(g[r + n, c : c + n] == wall)))
                if c > 0:
                    side_wall_counts.append(int(np.count_nonzero(g[r : r + n, c - 1] == wall)))
                if c + n < w:
                    side_wall_counts.append(int(np.count_nonzero(g[r : r + n, c + n] == wall)))

                if len(side_wall_counts) >= 3 and all(count >= n - 1 for count in side_wall_counts):
                    rooms.append((r, c, n))
    return rooms


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            area = 0
            while q:
                rr, cc = q.popleft()
                area += 1
                r0 = min(r0, rr)
                c0 = min(c0, cc)
                r1 = max(r1, rr)
                c1 = max(c1, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append((r0, c0, r1, c1, area))
    return components


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _middle_fill_bounded_by_side_rooms(
    bbox: tuple[int, int, int, int],
    rooms: list[tuple[int, int, int]],
) -> bool:
    r0, c0, r1, c1 = bbox
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    above: list[tuple[int, int]] = []
    below: list[tuple[int, int]] = []

    for r, c, n in rooms:
        rr1 = r + n - 1
        cc1 = c + n - 1
        row_overlap = _overlap_len(r0, r1, r, rr1)
        col_overlap = _overlap_len(c0, c1, c, cc1)
        if row_overlap > 0 and 1 <= c0 - cc1 <= 2:
            left.append((r, rr1))
        if row_overlap > 0 and 1 <= c - c1 <= 2:
            right.append((r, rr1))
        if col_overlap > 0 and 1 <= r0 - rr1 <= 2:
            above.append((c, cc1))
        if col_overlap > 0 and 1 <= r - r1 <= 2:
            below.append((c, cc1))

    def spans_contain(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]], lo: int, hi: int) -> bool:
        if not intervals_a or not intervals_b:
            return False
        a0 = min(start for start, _end in intervals_a)
        a1 = max(end for _start, end in intervals_a)
        b0 = min(start for start, _end in intervals_b)
        b1 = max(end for _start, end in intervals_b)
        return lo >= a0 and hi <= a1 and lo >= b0 and hi <= b1

    return spans_contain(left, right, r0, r1) or spans_contain(above, below, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the maze corridor and framed rooms lying on shortest paths between the two colored endpoint components.

    Input structure
    - The grid has a dominant background, one wall color forming thin orthogonal frames, and one endpoint/fill color.
    - The endpoint/fill color forms exactly two 4-connected solid rectangular components; their larger side lengths define the framed room sizes.

    Rule
    - Treat wall pixels as blocked and every other pixel as passable.
    - Find every passable pixel that lies on a shortest 4-connected path between the two endpoint components.
    - Also fill any framed square room of an endpoint-derived size that intersects that shortest-path mask.

    Key insight
    - Shortest paths identify the route through open connector corridors, while the wall-bordered square rooms need to be expanded from the route centerline to their full interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    counts = Counter(map(int, g.flat))

    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 2
    assert counts[bg] > max(counts[color] for color in non_bg)
    non_bg_counts = {color: int(np.count_nonzero(g == color)) for color in non_bg}
    wall_count = max(non_bg_counts.values())
    wall_candidates = [color for color, count in non_bg_counts.items() if count == wall_count]
    assert len(wall_candidates) == 1
    wall = wall_candidates[0]
    paint_candidates = [color for color in non_bg if color != wall]
    assert len(paint_candidates) == 1
    paint = paint_candidates[0]
    assert non_bg_counts[wall] > non_bg_counts[paint]

    endpoints = find_components(g, bg=bg, colors=[paint], connectivity=4)
    assert len(endpoints) == 2
    endpoint_coords = [component["coords"] for component in endpoints]

    room_sizes: set[int] = set()
    for coords in endpoint_coords:
        r0, c0, r1, c1 = bbox_coords(coords)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        assert coords.shape[0] == bh * bw
        room_sizes.add(max(bh, bw))

    passable = g != wall
    dist0 = _distances_from(endpoint_coords[0], passable)
    dist1 = _distances_from(endpoint_coords[1], passable)
    endpoint_distances = [int(dist0[r, c]) for r, c in endpoint_coords[1]]
    reachable_distances = [d for d in endpoint_distances if d >= 0]
    assert len(reachable_distances) == len(endpoint_distances)
    shortest = min(reachable_distances)

    route = (dist0 >= 0) & (dist1 >= 0) & (dist0 + dist1 == shortest)
    assert np.any(route)

    fill = route.copy()
    rooms = _framed_square_rooms(g, wall, room_sizes)
    assert rooms
    for r, c, n in rooms:
        if np.any(route[r : r + n, c : c + n]):
            fill[r : r + n, c : c + n] = True

    room_mask = np.zeros_like(fill, dtype=bool)
    for r, c, n in rooms:
        room_mask[r : r + n, c : c + n] = True
    open_fill = fill & (g == bg) & ~room_mask
    for r0, c0, r1, c1, area in _component_bboxes(open_fill):
        if r1 > r0 and c1 > c0 and area >= 20:
            assert _middle_fill_bounded_by_side_rooms(
                (r0, c0, r1, c1),
                rooms,
            ), "middle fill must stay within the facing room-frame spans"

    H, W = g.shape
    for r, c, n in rooms:
        has_doorway = False
        if r > 0:
            has_doorway = has_doorway or np.any(g[r - 1, c : c + n] != wall)
        if r + n < H:
            has_doorway = has_doorway or np.any(g[r + n, c : c + n] != wall)
        if c > 0:
            has_doorway = has_doorway or np.any(g[r : r + n, c - 1] != wall)
        if c + n < W:
            has_doorway = has_doorway or np.any(g[r : r + n, c + n] != wall)
        if has_doorway:
            assert np.any(fill[r : r + n, c : c + n]), "rooms with doorways must be on the shortest path"

    out = g.copy()
    out[fill & (g == bg)] = paint
    return out

# --------------- generator (attempt 2 refined middle-fill) ---------------

BG = 0
WALL = 8
PAINT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Canvas bounds used by the observed evaluation examples.
ROW_MIN = 22
ROW_MAX = 29
COL_MIN = 24
COL_MAX = 28

# Room-grid geometry and room-count filters.
ROOM_SIZE_RANGE = range(3, 6)
ROOM_GRID_RANGE = range(1, 8)
BLOCK_ROOM_MIN = 3
BLOCK_ROOM_MAX = 14
TOTAL_ROOM_MIN = 7
TOTAL_ROOM_MAX = 22
GAP_RANGE = range(3, 8)
ORIENTATIONS = ("vertical", "horizontal")

# Coarse layout and final route validation thresholds.
LAYOUT_DISTANCE_MIN = 36
LAYOUT_DISTANCE_FLOORS = [36, 42, 48, 54, 60]
SHORTEST_DISTANCE_MIN = 36
SHORTEST_DISTANCE_MAX = 72
TURN_MIN = 8
TURN_MAX = 21
ON_ROUTE_ROOM_MIN = 6
ON_ROUTE_ROOM_MAX = 14
OFF_ROUTE_ROOM_MIN = 1
OFF_ROUTE_ROOM_MAX = 8

# Connector sampling: offset connectors create the observed middle fill region.
OFFSET_CONNECTOR_PROB = 0.60
CONNECTOR_CHOICE_TRIES = 300
CONNECTOR_DELTA_MAX = 12
CONNECTOR_DELTA_MIN_CAP = 5

# Endpoint room selection favors far rooms, with a minority near-exit branch.
NEAR_EXIT_ENDPOINT_PROB = 0.30
NEAR_EXIT_STEPS = [-1, 0, 1]

# Full-example rejection retries.
GENERATE_ATTEMPTS = 300

_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_LAYOUTS = None
_LAST_META = None


def _block_shape(rows, cols, size):
    return rows * (size + 1) + 1, cols * (size + 1) + 1


def _room_topleft(room_r, room_c, size):
    return 1 + room_r * (size + 1), 1 + room_c * (size + 1)


def _layout_distance_estimate(r1, c1, s1, r2, c2, s2, gap):
    return (r1 + c1 - 2) * (s1 + 1) + (r2 + c2 - 2) * (s2 + 1) + s1 + s2 + gap


def _layout_fits(orientation, bh1, bw1, bh2, bw2, gap):
    if orientation == "vertical":
        return bh1 + gap + bh2 <= ROW_MAX and max(bw1, bw2) <= COL_MAX
    return bw1 + gap + bw2 <= COL_MAX and max(bh1, bh2) <= ROW_MAX


def _build_layouts():
    """Enumerate all layout parameter tuples before random sampling."""
    layouts = []
    for orientation in ORIENTATIONS:
        for s1 in ROOM_SIZE_RANGE:
            for s2 in ROOM_SIZE_RANGE:
                for r1 in ROOM_GRID_RANGE:
                    for c1 in ROOM_GRID_RANGE:
                        n1 = r1 * c1
                        if not BLOCK_ROOM_MIN <= n1 <= BLOCK_ROOM_MAX:
                            continue
                        bh1, bw1 = _block_shape(r1, c1, s1)
                        for r2 in ROOM_GRID_RANGE:
                            for c2 in ROOM_GRID_RANGE:
                                n2 = r2 * c2
                                total_rooms = n1 + n2
                                valid_block2 = BLOCK_ROOM_MIN <= n2 <= BLOCK_ROOM_MAX
                                valid_total = TOTAL_ROOM_MIN <= total_rooms <= TOTAL_ROOM_MAX
                                if not (valid_block2 and valid_total):
                                    continue
                                bh2, bw2 = _block_shape(r2, c2, s2)
                                for gap in GAP_RANGE:
                                    approx = _layout_distance_estimate(r1, c1, s1, r2, c2, s2, gap)
                                    if approx < LAYOUT_DISTANCE_MIN:
                                        continue
                                    if _layout_fits(orientation, bh1, bw1, bh2, bw2, gap):
                                        layouts.append((orientation, s1, s2, r1, c1, r2, c2, gap, approx))
    return layouts


def _layouts():
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = _build_layouts()
    return _LAYOUTS


def _choose_layout():
    layouts = _layouts()
    lo = random.choice(LAYOUT_DISTANCE_FLOORS)
    candidates = [layout for layout in layouts if layout[-1] >= lo]
    if not candidates:
        candidates = layouts
    return random.choice(candidates)


def _side_positions(count, size):
    positions = []
    for index in range(count):
        for offset in range(size):
            positions.append((1 + index * (size + 1) + offset, index, offset))
    return positions


def _connector_delta(span, offset_connector):
    if not offset_connector:
        return 0
    max_delta = min(CONNECTOR_DELTA_MAX, span - 1)
    min_delta = min(CONNECTOR_DELTA_MIN_CAP, max_delta)
    if min_delta <= 1:
        return None
    return random.choice((-1, 1)) * random.randint(min_delta, max_delta)


def _side_connector_choice(span_lo, span_hi, size1, side_positions1, size2, side_positions2, offset_connector):
    """Pick compatible starts and exit offsets along the shared span."""
    for _ in range(CONNECTOR_CHOICE_TRIES):
        span = random.randint(span_lo, span_hi)
        local1, index1, offset1 = random.choice(side_positions1)
        local2, index2, offset2 = random.choice(side_positions2)
        delta = _connector_delta(span, offset_connector)
        if delta is None:
            continue

        shift = local1 + delta - local2
        lo = max(0, -shift)
        hi = min(span - size1, span - size2 - shift)
        if lo > hi:
            continue
        start1 = random.randint(lo, hi)
        start2 = start1 + shift
        exit1 = start1 + local1
        exit2 = start2 + local2
        common_lo = max(start1 + 1, start2 + 1)
        common_hi = min(start1 + size1 - 2, start2 + size2 - 2)
        if min(exit1, exit2) < common_lo or max(exit1, exit2) > common_hi:
            continue
        return span, start1, start2, exit1, exit2, index1, offset1, index2, offset2
    return None


def _make_block(rows, cols, size, exit_side, exit_index=None, exit_offset=None):
    h, w = _block_shape(rows, cols, size)
    block = np.full((h, w), BG, dtype=int)
    block[:: size + 1, :] = WALL
    block[:, :: size + 1] = WALL

    for i in range(1, rows):
        wr = i * (size + 1)
        for j in range(cols):
            c0 = 1 + j * (size + 1)
            block[wr, c0 + random.randrange(size)] = BG
    for j in range(1, cols):
        wc = j * (size + 1)
        for i in range(rows):
            r0 = 1 + i * (size + 1)
            block[r0 + random.randrange(size), wc] = BG

    if exit_side == "bottom":
        side_index = random.randrange(cols) if exit_index is None else int(exit_index)
        offset = random.randrange(size) if exit_offset is None else int(exit_offset)
        room = (rows - 1, side_index)
        rr = h - 1
        cc = 1 + room[1] * (size + 1) + offset
    elif exit_side == "top":
        side_index = random.randrange(cols) if exit_index is None else int(exit_index)
        offset = random.randrange(size) if exit_offset is None else int(exit_offset)
        room = (0, side_index)
        rr = 0
        cc = 1 + room[1] * (size + 1) + offset
    elif exit_side == "right":
        side_index = random.randrange(rows) if exit_index is None else int(exit_index)
        offset = random.randrange(size) if exit_offset is None else int(exit_offset)
        room = (side_index, cols - 1)
        rr = 1 + room[0] * (size + 1) + offset
        cc = w - 1
    else:
        side_index = random.randrange(rows) if exit_index is None else int(exit_index)
        offset = random.randrange(size) if exit_offset is None else int(exit_offset)
        room = (side_index, 0)
        rr = 1 + room[0] * (size + 1) + offset
        cc = 0
    block[rr, cc] = BG
    return block, (rr, cc), room


def _choose_endpoint_axis(length, exit_index):
    if length == 1:
        return 0
    if random.random() < NEAR_EXIT_ENDPOINT_PROB:
        return min(length - 1, max(0, exit_index + random.choice(NEAR_EXIT_STEPS)))
    far = max(abs(i - exit_index) for i in range(length))
    return random.choice([i for i in range(length) if abs(i - exit_index) == far])


def _choose_endpoint_room(rows, cols, exit_room, exit_side):
    er, ec = exit_room
    if exit_side == "bottom":
        return (0 if rows > 1 else er), _choose_endpoint_axis(cols, ec)
    if exit_side == "top":
        return (rows - 1 if rows > 1 else er), _choose_endpoint_axis(cols, ec)
    if exit_side == "right":
        return _choose_endpoint_axis(rows, er), (0 if cols > 1 else ec)
    return _choose_endpoint_axis(rows, er), (cols - 1 if cols > 1 else ec)


def _room_interiors(rows, cols, size, origin_r, origin_c):
    rooms = []
    for rr in range(rows):
        for cc in range(cols):
            lr, lc = _room_topleft(rr, cc, size)
            rooms.append((origin_r + lr, origin_c + lc, size))
    return rooms


def _paint_endpoint(grid, origin_r, origin_c, room, size, paint):
    lr, lc = _room_topleft(*room, size)
    grid[origin_r + lr : origin_r + lr + size, origin_c + lc : origin_c + lc + size] = paint


def _place_vertical(s1, s2, r1, c1, r2, c2, gap, paint, offset_connector):
    bh1, bw1 = _block_shape(r1, c1, s1)
    bh2, bw2 = _block_shape(r2, c2, s2)
    hmin = bh1 + gap + bh2
    h = random.randint(max(ROW_MIN, hmin), ROW_MAX)
    choice = _side_connector_choice(
        max(COL_MIN, bw1, bw2),
        COL_MAX,
        bw1,
        _side_positions(c1, s1),
        bw2,
        _side_positions(c2, s2),
        offset_connector,
    )
    if choice is None:
        return None
    w, c_1, c_2, exit_col1, exit_col2, index1, offset1, index2, offset2 = choice
    b1, exit1, exit_room1 = _make_block(r1, c1, s1, "bottom", index1, offset1)
    b2, exit2, exit_room2 = _make_block(r2, c2, s2, "top", index2, offset2)
    top = random.randint(0, h - hmin)
    r_1 = top
    r_2 = top + bh1 + gap
    grid = np.full((h, w), BG, dtype=int)
    paste(grid, b1, r_1, c_1)
    paste(grid, b2, r_2, c_2)
    if exit_col1 == exit_col2:
        draw_line(grid, r_1 + exit1[0], exit_col1, end_r=r_2 + exit2[0], end_c=exit_col1, color=BG)

    ep1 = _choose_endpoint_room(r1, c1, exit_room1, "bottom")
    ep2 = _choose_endpoint_room(r2, c2, exit_room2, "top")
    rooms = _room_interiors(r1, c1, s1, r_1, c_1) + _room_interiors(r2, c2, s2, r_2, c_2)
    _paint_endpoint(grid, r_1, c_1, ep1, s1, paint)
    _paint_endpoint(grid, r_2, c_2, ep2, s2, paint)
    meta = {
        "orientation": "vertical",
        "gap": gap,
        "connector_delta": abs(exit_col2 - exit_col1),
        "blocks": [{"rows": r1, "cols": c1, "size": s1}, {"rows": r2, "cols": c2, "size": s2}],
    }
    return grid, rooms, meta


def _place_horizontal(s1, s2, r1, c1, r2, c2, gap, paint, offset_connector):
    bh1, bw1 = _block_shape(r1, c1, s1)
    bh2, bw2 = _block_shape(r2, c2, s2)
    wmin = bw1 + gap + bw2
    w = random.randint(max(COL_MIN, wmin), COL_MAX)
    choice = _side_connector_choice(
        max(ROW_MIN, bh1, bh2),
        ROW_MAX,
        bh1,
        _side_positions(r1, s1),
        bh2,
        _side_positions(r2, s2),
        offset_connector,
    )
    if choice is None:
        return None
    h, r_1, r_2, exit_row1, exit_row2, index1, offset1, index2, offset2 = choice
    b1, exit1, exit_room1 = _make_block(r1, c1, s1, "right", index1, offset1)
    b2, exit2, exit_room2 = _make_block(r2, c2, s2, "left", index2, offset2)
    left = random.randint(0, w - wmin)
    c_1 = left
    c_2 = left + bw1 + gap
    grid = np.full((h, w), BG, dtype=int)
    paste(grid, b1, r_1, c_1)
    paste(grid, b2, r_2, c_2)
    if exit_row1 == exit_row2:
        draw_line(grid, exit_row1, c_1 + exit1[1], end_r=exit_row1, end_c=c_2 + exit2[1], color=BG)

    ep1 = _choose_endpoint_room(r1, c1, exit_room1, "right")
    ep2 = _choose_endpoint_room(r2, c2, exit_room2, "left")
    rooms = _room_interiors(r1, c1, s1, r_1, c_1) + _room_interiors(r2, c2, s2, r_2, c_2)
    _paint_endpoint(grid, r_1, c_1, ep1, s1, paint)
    _paint_endpoint(grid, r_2, c_2, ep2, s2, paint)
    meta = {
        "orientation": "horizontal",
        "gap": gap,
        "connector_delta": abs(exit_row2 - exit_row1),
        "blocks": [{"rows": r1, "cols": c1, "size": s1}, {"rows": r2, "cols": c2, "size": s2}],
    }
    return grid, rooms, meta


def _place_layout(layout, paint, offset_connector):
    orientation, s1, s2, r1, c1, r2, c2, gap, _ = layout
    if orientation == "vertical":
        return _place_vertical(s1, s2, r1, c1, r2, c2, gap, paint, offset_connector)
    return _place_horizontal(s1, s2, r1, c1, r2, c2, gap, paint, offset_connector)


def _bfs_path_stats(grid, paint):
    """Return one shortest path's distance and turn count between paint components."""
    comps = find_components(grid, bg=BG, colors=[paint], connectivity=4)
    if len(comps) != 2:
        return None
    passable = grid != WALL
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    parent = np.full((h, w, 2), -2, dtype=int)
    q = deque()
    for r, c in comps[0]["coords"]:
        rr = int(r)
        cc = int(c)
        if dist[rr, cc] < 0:
            dist[rr, cc] = 0
            parent[rr, cc] = (-1, -1)
            q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                parent[nr, nc] = (r, c)
                q.append((nr, nc))

    targets = [(int(r), int(c)) for r, c in comps[1]["coords"] if dist[int(r), int(c)] >= 0]
    if len(targets) != len(comps[1]["coords"]):
        return None
    target = min(targets, key=lambda rc: int(dist[rc]))
    shortest = int(dist[target])
    path = []
    cur = target
    while cur != (-1, -1):
        path.append(cur)
        pr, pc = parent[cur]
        cur = (int(pr), int(pc))
    path.reverse()

    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    return shortest, turns


def _shortest_route_mask(grid, comps):
    """Return the all-shortest-path mask and distance between endpoint components."""
    passable = grid != WALL
    dist0 = _distances_from(comps[0]["coords"], passable)
    dist1 = _distances_from(comps[1]["coords"], passable)
    target_dists = [int(dist0[int(r), int(c)]) for r, c in comps[1]["coords"]]
    if any(d < 0 for d in target_dists):
        return None
    shortest = min(target_dists)
    route = (dist0 >= 0) & (dist1 >= 0) & (dist0 + dist1 == shortest)
    if not np.any(route):
        return None
    return route, shortest


def _endpoint_room_sizes(comps):
    sizes = set()
    for coords in [comps[0]["coords"], comps[1]["coords"]]:
        r0, c0, r1, c1 = bbox_coords(coords)
        sizes.add(max(r1 - r0 + 1, c1 - c0 + 1))
    return sizes


def _fill_route_rooms(grid, route, comps):
    fill = route.copy()
    for r, c, n in _framed_square_rooms(grid, WALL, _endpoint_room_sizes(comps)):
        if np.any(route[r : r + n, c : c + n]):
            fill[r : r + n, c : c + n] = True
    return fill


def _route_room_set(fill, rooms):
    on_route_rooms = set()
    for r, c, n in rooms:
        if np.any(fill[r : r + n, c : c + n]):
            on_route_rooms.add((r, c, n))
    return on_route_rooms


def _room_boundary_cells(r, c, n):
    for cc in range(c, c + n):
        yield r - 1, cc
    for cc in range(c, c + n):
        yield r + n, cc
    for rr in range(r, r + n):
        yield rr, c - 1
    for rr in range(r, r + n):
        yield rr, c + n


def _seal_off_route_rooms(grid, rooms, on_route_rooms):
    h, w = grid.shape
    for r, c, n in rooms:
        if (r, c, n) in on_route_rooms:
            continue
        for rr, cc in _room_boundary_cells(r, c, n):
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != WALL:
                grid[rr, cc] = WALL


def _path_stats_valid(shortest, turns):
    return SHORTEST_DISTANCE_MIN <= shortest <= SHORTEST_DISTANCE_MAX and TURN_MIN <= turns <= TURN_MAX


def _room_counts_valid(on_route, off_route):
    return ON_ROUTE_ROOM_MIN <= on_route <= ON_ROUTE_ROOM_MAX and OFF_ROUTE_ROOM_MIN <= off_route <= OFF_ROUTE_ROOM_MAX


def _solver_accepts(grid):
    try:
        main(grid)
    except Exception:
        return False
    return True


def _gen_attempt():
    global _LAST_META
    layout = _choose_layout()
    paint = random.choice(PAINT_COLORS)
    placed = _place_layout(layout, paint, random.random() < OFFSET_CONNECTOR_PROB)
    if placed is None:
        return None

    grid, rooms, meta = placed
    comps = find_components(grid, bg=BG, colors=[paint], connectivity=4)
    if len(comps) != 2:
        return None

    route_info = _shortest_route_mask(grid, comps)
    if route_info is None:
        return None
    route, shortest = route_info
    on_route_rooms = _route_room_set(_fill_route_rooms(grid, route, comps), rooms)

    path = _bfs_path_stats(grid, paint)
    if path is None:
        return None
    shortest_p, turns = path
    if not _path_stats_valid(shortest_p, turns):
        return None

    on_route = len(on_route_rooms)
    off_route = len(rooms) - on_route
    if not _room_counts_valid(on_route, off_route):
        return None

    _seal_off_route_rooms(grid, rooms, on_route_rooms)
    if not _solver_accepts(grid):
        return None

    meta["shortest_distance"] = shortest
    meta["turns_on_single_path"] = turns
    meta["rooms_on_route"] = on_route
    meta["rooms_off_route"] = off_route
    _LAST_META = meta
    return grid


def generate_input():
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 3ad05f52 input")
