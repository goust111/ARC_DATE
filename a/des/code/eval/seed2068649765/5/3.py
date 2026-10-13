import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
AZURE = 8
LOCK_COLORS = (1, 2, 3, 6, 7, 9)
SCENARIOS = ("chain", "branch", "blocked")
GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_wall_or_oob(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return r < 0 or r >= h or c < 0 or c >= w or int(grid[r, c]) == GREY


def _looks_like_door(grid: np.ndarray, r: int, c: int) -> bool:
    color = int(grid[r, c])
    if color in (BLACK, GREY, AZURE):
        return False

    up = _is_wall_or_oob(grid, r - 1, c)
    down = _is_wall_or_oob(grid, r + 1, c)
    left = _is_wall_or_oob(grid, r, c - 1)
    right = _is_wall_or_oob(grid, r, c + 1)

    vertical = up and down and not (left and right)
    horizontal = left and right and not (up and down)
    return vertical or horizontal


def _label_rooms(grid: np.ndarray, door_mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    room_mask = (grid != GREY) & ~door_mask
    room_ids = np.full(grid.shape, -1, dtype=int)
    rooms: list[np.ndarray] = []
    h, w = grid.shape

    for comp in _components(room_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        rid = len(rooms)
        rooms.append(comp)
        room_ids[comp[:, 0], comp[:, 1]] = rid

    return rooms, room_ids


def _adjacent_room_ids(room_ids: np.ndarray, r: int, c: int) -> list[int]:
    h, w = room_ids.shape
    ids = {
        int(room_ids[nr, nc])
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < h and 0 <= nc < w and int(room_ids[nr, nc]) >= 0
    }
    return sorted(ids)


def _parse_maze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    candidates = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _looks_like_door(g, r, c):
                candidates[r, c] = True

    door_mask = candidates.copy()
    for _ in range(3):
        rooms, room_ids = _label_rooms(g, door_mask)
        refined = np.zeros_like(door_mask)
        for r, c in np.argwhere(candidates):
            room_neighbors = _adjacent_room_ids(room_ids, int(r), int(c))
            color = int(g[r, c])
            if color == YELLOW:
                if len(room_neighbors) == 1 and (r in (0, h - 1) or c in (0, w - 1)):
                    refined[r, c] = True
            elif len(room_neighbors) == 2:
                refined[r, c] = True
        if np.array_equal(refined, door_mask):
            break
        door_mask = refined

    rooms, room_ids = _label_rooms(g, door_mask)
    room_keys: list[set[int]] = []
    for room in rooms:
        keys = {
            int(g[r, c])
            for r, c in room
            if int(g[r, c]) not in (BLACK,)
        }
        room_keys.append(keys)

    doors: list[tuple[int, int, int]] = []
    entrance_rooms: list[int] = []
    for r, c in np.argwhere(door_mask):
        room_neighbors = _adjacent_room_ids(room_ids, int(r), int(c))
        color = int(g[r, c])
        if color == YELLOW:
            if len(room_neighbors) == 1:
                entrance_rooms.append(room_neighbors[0])
        elif len(room_neighbors) == 2 and room_neighbors[0] != room_neighbors[1]:
            doors.append((room_neighbors[0], room_neighbors[1], color))

    assert len(entrance_rooms) == 1, "expected a unique entrance"
    assert rooms, "expected at least one room"

    return {
        "grid": g,
        "rooms": rooms,
        "room_ids": room_ids,
        "room_keys": room_keys,
        "doors": doors,
        "entrance_room": entrance_rooms[0],
        "door_mask": door_mask,
    }


def _reachable_rooms(parsed: dict) -> tuple[set[int], set[int]]:
    reached = {int(parsed["entrance_room"])}
    keys = set(parsed["room_keys"][parsed["entrance_room"]])

    changed = True
    while changed:
        changed = False
        for a, b, color in parsed["doors"]:
            if color not in keys:
                continue
            if a in reached and b not in reached:
                reached.add(b)
                keys |= parsed["room_keys"][b]
                changed = True
            elif b in reached and a not in reached:
                reached.add(a)
                keys |= parsed["room_keys"][a]
                changed = True
    return reached, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every room reachable from the yellow entrance under color-key door rules.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_maze(g)
    reached, _ = _reachable_rooms(parsed)

    out = g.copy()
    for rid in reached:
        room = parsed["rooms"][rid]
        room_rows = room[:, 0]
        room_cols = room[:, 1]
        black_mask = g[room_rows, room_cols] == BLACK
        out[room_rows[black_mask], room_cols[black_mask]] = AZURE
    return out


def _build_layout(rng: random.Random) -> dict | None:
    for _ in range(LAYOUT_ATTEMPTS):
        room_rows, room_cols = rng.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
        heights = [rng.randint(3, 6) for _ in range(room_rows)]
        widths = [rng.randint(3, 6) for _ in range(room_cols)]
        h = sum(heights) + room_rows + 1
        w = sum(widths) + room_cols + 1
        if not (10 <= h <= 24 and 10 <= w <= 24):
            continue

        row_bounds: list[tuple[int, int]] = []
        col_bounds: list[tuple[int, int]] = []

        cur = 1
        for height in heights:
            row_bounds.append((cur, cur + height - 1))
            cur += height + 1

        cur = 1
        for width in widths:
            col_bounds.append((cur, cur + width - 1))
            cur += width + 1

        rooms = []
        room_by_rc: dict[tuple[int, int], int] = {}
        for rr, (r0, r1) in enumerate(row_bounds):
            for cc, (c0, c1) in enumerate(col_bounds):
                rid = rr * room_cols + cc
                room_by_rc[(rr, cc)] = rid
                rooms.append(
                    {
                        "id": rid,
                        "row": rr,
                        "col": cc,
                        "r0": r0,
                        "r1": r1,
                        "c0": c0,
                        "c1": c1,
                    }
                )

        adj: dict[int, list[int]] = {room["id"]: [] for room in rooms}
        adj_pairs: list[tuple[int, int]] = []
        for rr in range(room_rows):
            for cc in range(room_cols):
                rid = room_by_rc[(rr, cc)]
                if cc + 1 < room_cols:
                    nid = room_by_rc[(rr, cc + 1)]
                    adj[rid].append(nid)
                    adj[nid].append(rid)
                    adj_pairs.append((rid, nid))
                if rr + 1 < room_rows:
                    nid = room_by_rc[(rr + 1, cc)]
                    adj[rid].append(nid)
                    adj[nid].append(rid)
                    adj_pairs.append((rid, nid))

        boundary_rooms = [
            room["id"]
            for room in rooms
            if room["row"] in (0, room_rows - 1) or room["col"] in (0, room_cols - 1)
        ]

        return {
            "shape": (h, w),
            "room_rows": room_rows,
            "room_cols": room_cols,
            "row_bounds": row_bounds,
            "col_bounds": col_bounds,
            "wall_rows": [0] + [r1 + 1 for _r0, r1 in row_bounds],
            "wall_cols": [0] + [c1 + 1 for _c0, c1 in col_bounds],
            "rooms": rooms,
            "adj": adj,
            "adj_pairs": adj_pairs,
            "boundary_rooms": boundary_rooms,
        }
    return None


def _random_simple_path(
    adj: dict[int, list[int]],
    start: int,
    length: int,
    rng: random.Random,
) -> list[int] | None:
    path = [start]
    used = {start}

    def dfs(cur: int) -> bool:
        if len(path) == length:
            return True
        nbrs = [nbr for nbr in adj[cur] if nbr not in used]
        rng.shuffle(nbrs)
        for nbr in nbrs:
            used.add(nbr)
            path.append(nbr)
            if dfs(nbr):
                return True
            path.pop()
            used.remove(nbr)
        return False

    return path if dfs(start) else None


def _frontier_edges(
    adj_pairs: list[tuple[int, int]],
    reached: set[int],
    used_doors: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    out = []
    for a, b in adj_pairs:
        if _edge_key(a, b) in used_doors:
            continue
        if (a in reached) ^ (b in reached):
            out.append((a, b))
    return out


def _plan_scenario(layout: dict, entrance: int, scenario: str, rng: random.Random) -> dict | None:
    room_count = len(layout["rooms"])
    room_keys = {rid: set() for rid in range(room_count)}
    doors: dict[tuple[int, int], int] = {}

    if scenario == "chain":
        max_len = min(room_count - 1, 6)
        lengths = list(range(max_len, 2, -1))
        rng.shuffle(lengths)
        path = None
        for length in lengths:
            path = _random_simple_path(layout["adj"], entrance, length, rng)
            if path is not None:
                break
        if path is None:
            return None

        chain_colors = rng.sample(LOCK_COLORS, len(path) - 1)
        reached = set(path)
        room_keys[path[0]].add(chain_colors[0])
        for idx, color in enumerate(chain_colors):
            doors[_edge_key(path[idx], path[idx + 1])] = color
            if idx + 1 < len(chain_colors):
                room_keys[path[idx + 1]].add(chain_colors[idx + 1])

    elif scenario == "branch":
        neighbors = list(layout["adj"][entrance])
        if len(neighbors) < 2:
            return None
        n1, n2 = rng.sample(neighbors, 2)
        c1, c2, c3 = rng.sample(LOCK_COLORS, 3)
        reached = {entrance, n1, n2}
        room_keys[entrance].update({c1, c2})
        doors[_edge_key(entrance, n1)] = c1
        doors[_edge_key(entrance, n2)] = c2

        second_level = []
        for src in (n1, n2):
            for nbr in layout["adj"][src]:
                if nbr not in reached:
                    second_level.append((src, nbr))
        if second_level and rng.random() < 0.65:
            src, nbr = rng.choice(second_level)
            doors[_edge_key(src, nbr)] = c3
            room_keys[src].add(c3)
            reached.add(nbr)

    else:
        reached = {entrance}
        if rng.random() < 0.5:
            room_keys[entrance].add(rng.choice(LOCK_COLORS))

    reachable_palette = sorted({color for rid in reached for color in room_keys[rid]})
    blocked_choices = [color for color in LOCK_COLORS if color not in reachable_palette]
    blocked_color = rng.choice(blocked_choices) if blocked_choices else rng.choice(LOCK_COLORS)

    unreached = [rid for rid in range(room_count) if rid not in reached]
    if unreached:
        room_keys[rng.choice(unreached)].add(blocked_color)

    frontier = _frontier_edges(layout["adj_pairs"], reached, doors)
    if scenario == "blocked":
        if not frontier:
            return None
        rng.shuffle(frontier)
        for a, b in frontier[: rng.randint(1, min(2, len(frontier)))]:
            doors[_edge_key(a, b)] = blocked_color
    elif frontier and rng.random() < 0.8:
        a, b = rng.choice(frontier)
        doors[_edge_key(a, b)] = blocked_color

    for a, b in layout["adj_pairs"]:
        key = _edge_key(a, b)
        if key in doors or rng.random() >= 0.28:
            continue
        a_reached = a in reached
        b_reached = b in reached
        if a_reached ^ b_reached:
            doors[key] = blocked_color
        elif a_reached and b_reached:
            pool = reachable_palette or [blocked_color]
            doors[key] = rng.choice(pool)
        else:
            doors[key] = rng.choice(LOCK_COLORS)

    marker_colors = {rid: set(colors) for rid, colors in room_keys.items()}
    for rid in range(room_count):
        if len(marker_colors[rid]) >= 3:
            continue
        if rid in reached:
            pool = [color for color in reachable_palette if color not in marker_colors[rid]]
        else:
            pool = [color for color in LOCK_COLORS if color not in marker_colors[rid]]
        rng.shuffle(pool)
        while pool and len(marker_colors[rid]) < 3 and rng.random() < 0.35:
            marker_colors[rid].add(pool.pop())
            if rng.random() < 0.7:
                break

    return {
        "entrance": entrance,
        "doors": doors,
        "markers": marker_colors,
        "reached": reached,
        "reachable_palette": set(reachable_palette),
        "blocked_color": blocked_color,
    }


def _shared_wall_position(layout: dict, a: int, b: int, rng: random.Random) -> tuple[int, int]:
    room_a = layout["rooms"][a]
    room_b = layout["rooms"][b]
    if room_a["row"] == room_b["row"]:
        left, right = (room_a, room_b) if room_a["col"] < room_b["col"] else (room_b, room_a)
        col = left["c1"] + 1
        row = rng.randint(max(left["r0"], right["r0"]), min(left["r1"], right["r1"]))
        return row, col

    top, bottom = (room_a, room_b) if room_a["row"] < room_b["row"] else (room_b, room_a)
    row = top["r1"] + 1
    col = rng.randint(max(top["c0"], bottom["c0"]), min(top["c1"], bottom["c1"]))
    return row, col


def _entrance_position(layout: dict, room_id: int, rng: random.Random) -> tuple[int, int]:
    room = layout["rooms"][room_id]
    sides = []
    if room["row"] == 0:
        sides.append("top")
    if room["row"] == layout["room_rows"] - 1:
        sides.append("bottom")
    if room["col"] == 0:
        sides.append("left")
    if room["col"] == layout["room_cols"] - 1:
        sides.append("right")

    side = rng.choice(sides)
    if side == "top":
        return room["r0"] - 1, rng.randint(room["c0"], room["c1"])
    if side == "bottom":
        return room["r1"] + 1, rng.randint(room["c0"], room["c1"])
    if side == "left":
        return rng.randint(room["r0"], room["r1"]), room["c0"] - 1
    return rng.randint(room["r0"], room["r1"]), room["c1"] + 1


def _marker_candidates(grid: np.ndarray, room: dict) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r in range(room["r0"], room["r1"] + 1):
        for c in range(room["c0"], room["c1"] + 1):
            if int(grid[r, c]) != BLACK:
                continue
            if any(
                0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY
                for nr, nc in _neighbors4(r, c)
            ):
                candidates.append((r, c))
    return candidates


def _render_layout(layout: dict, plan: dict, rng: random.Random) -> np.ndarray | None:
    h, w = layout["shape"]
    grid = np.full((h, w), BLACK, dtype=int)
    grid[layout["wall_rows"], :] = GREY
    grid[:, layout["wall_cols"]] = GREY

    for (a, b), color in plan["doors"].items():
        r, c = _shared_wall_position(layout, a, b, rng)
        grid[r, c] = color

    er, ec = _entrance_position(layout, plan["entrance"], rng)
    grid[er, ec] = YELLOW

    for rid, colors in plan["markers"].items():
        color_list = list(colors)
        rng.shuffle(color_list)
        candidates = _marker_candidates(grid, layout["rooms"][rid])
        rng.shuffle(candidates)
        if len(color_list) > len(candidates):
            return None
        for color, (r, c) in zip(color_list, candidates):
            grid[r, c] = color

    return grid


def _validate_generated(grid: np.ndarray, layout: dict, plan: dict, scenario: str) -> bool:
    try:
        parsed = _parse_maze(grid)
    except AssertionError:
        return False

    if int(np.count_nonzero(grid == YELLOW)) != 1:
        return False
    if len(parsed["rooms"]) != len(layout["rooms"]):
        return False

    planned_to_actual: dict[int, int] = {}
    for room in layout["rooms"]:
        rid = int(parsed["room_ids"][room["r0"], room["c0"]])
        if rid < 0:
            return False
        planned_to_actual[room["id"]] = rid

    actual_reached, actual_keys = _reachable_rooms(parsed)
    expected_reached = {planned_to_actual[rid] for rid in plan["reached"]}
    if actual_reached != expected_reached:
        return False

    if scenario == "blocked" and len(actual_reached) != 1:
        return False
    if scenario == "chain" and len(actual_reached) < 3:
        return False
    if scenario == "branch":
        actual_entrance = planned_to_actual[plan["entrance"]]
        open_neighbors = 0
        for a, b, color in parsed["doors"]:
            if color not in actual_keys:
                continue
            if a == actual_entrance and b in actual_reached:
                open_neighbors += 1
            elif b == actual_entrance and a in actual_reached:
                open_neighbors += 1
        if open_neighbors < 2:
            return False

    expected = grid.copy()
    for rid in actual_reached:
        room = parsed["rooms"][rid]
        rows = room[:, 0]
        cols = room[:, 1]
        mask = grid[rows, cols] == BLACK
        expected[rows[mask], cols[mask]] = AZURE

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    """
    Generate random door-and-key mazes with chain, branch, and blocked variants.
    """
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        layout = _build_layout(rng)
        if layout is None:
            continue

        scenario = rng.choice(SCENARIOS)
        entrance = rng.choice(layout["boundary_rooms"])
        plan = _plan_scenario(layout, entrance, scenario, rng)
        if plan is None:
            continue

        grid = _render_layout(layout, plan, rng)
        if grid is None:
            continue
        if _validate_generated(grid, layout, plan, scenario):
            return grid

    raise RuntimeError("failed to generate a valid maze input")
