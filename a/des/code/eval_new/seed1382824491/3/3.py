import random
from collections import defaultdict, deque

import numpy as np


FLOOR = 0
START = 4
GREY = 5
END = 9
WALL_COLORS = (1, 3, 7)
KEY_COLORS = (2, 6, 8)
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            idx = len(components)
            labels[r, c] = idx
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            components.append(cells)
    return labels, components


def _key_border_mask() -> np.ndarray:
    mask = np.ones((4, 4), dtype=bool)
    mask[1:3, 1:3] = False
    return mask


KEY_BORDER_MASK = _key_border_mask()


def _find_keys(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    keys: list[dict] = []
    for r in range(h - 3):
        for c in range(w - 3):
            patch = g[r : r + 4, c : c + 4]
            if not bool(np.all(patch[KEY_BORDER_MASK] == GREY)):
                continue
            inner = patch[1:3, 1:3]
            col = int(inner[0, 0])
            if col in (FLOOR, START, GREY, END):
                continue
            if not bool(np.all(inner == col)):
                continue
            cells = {(r + dr, c + dc) for dr in range(4) for dc in range(4)}
            keys.append({"color": col, "top": r, "left": c, "cells": cells})
    return keys


def _infer_wall_and_lock_vertical(g: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    a = int(g[r - 1, c])
    b = int(g[r + 1, c])
    if a == b:
        return None
    score_a = int(r - 2 >= 0 and int(g[r - 2, c]) == a) + int(r + 2 < g.shape[0] and int(g[r + 2, c]) == a)
    score_b = int(r - 2 >= 0 and int(g[r - 2, c]) == b) + int(r + 2 < g.shape[0] and int(g[r + 2, c]) == b)
    if score_a == score_b:
        return None
    return (a, b) if score_a > score_b else (b, a)


def _infer_wall_and_lock_horizontal(g: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    a = int(g[r, c - 1])
    b = int(g[r, c + 1])
    if a == b:
        return None
    score_a = int(c - 2 >= 0 and int(g[r, c - 2]) == a) + int(c + 2 < g.shape[1] and int(g[r, c + 2]) == a)
    score_b = int(c - 2 >= 0 and int(g[r, c - 2]) == b) + int(c + 2 < g.shape[1] and int(g[r, c + 2]) == b)
    if score_a == score_b:
        return None
    return (a, b) if score_a > score_b else (b, a)


def _detect_doors(g: np.ndarray, floorish: np.ndarray) -> list[dict]:
    h, w = g.shape
    doors: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != FLOOR:
                continue

            vertical = bool(floorish[r, c - 1] and floorish[r, c + 1] and not floorish[r - 1, c] and not floorish[r + 1, c])
            horizontal = bool(floorish[r - 1, c] and floorish[r + 1, c] and not floorish[r, c - 1] and not floorish[r, c + 1])

            if vertical == horizontal:
                continue

            if vertical:
                info = _infer_wall_and_lock_vertical(g, r, c)
                if info is None:
                    continue
                wall_color, lock_color = info
                doors.append(
                    {
                        "pos": (r, c),
                        "orientation": "vertical",
                        "wall_color": wall_color,
                        "lock_color": lock_color,
                        "side_cells": ((r, c - 1), (r, c + 1)),
                    }
                )
            else:
                info = _infer_wall_and_lock_horizontal(g, r, c)
                if info is None:
                    continue
                wall_color, lock_color = info
                doors.append(
                    {
                        "pos": (r, c),
                        "orientation": "horizontal",
                        "wall_color": wall_color,
                        "lock_color": lock_color,
                        "side_cells": ((r - 1, c), (r + 1, c)),
                    }
                )
    return doors


def _solve_room_path(grid: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray, list[list[tuple[int, int]]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    keys = _find_keys(g)
    key_colors = {int(key["color"]) for key in keys}

    key_mask = np.zeros(g.shape, dtype=bool)
    for key in keys:
        for r, c in key["cells"]:
            key_mask[r, c] = True

    floorish = (g == FLOOR) | (g == START) | (g == END) | key_mask
    doors = _detect_doors(g, floorish)

    room_mask = floorish.copy()
    for door in doors:
        r, c = door["pos"]
        room_mask[r, c] = False

    room_ids, room_cells = _label_components(room_mask)

    start_pos = tuple(map(int, np.argwhere(g == START)[0]))
    end_pos = tuple(map(int, np.argwhere(g == END)[0]))
    start_room = int(room_ids[start_pos])
    end_room = int(room_ids[end_pos])

    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for door_idx, door in enumerate(doors):
        (a_r, a_c), (b_r, b_c) = door["side_cells"]
        ra = int(room_ids[a_r, a_c])
        rb = int(room_ids[b_r, b_c])
        if ra == -1 or rb == -1 or ra == rb:
            continue
        door["rooms"] = (ra, rb)
        if int(door["lock_color"]) in key_colors:
            graph[ra].append((rb, door_idx))
            graph[rb].append((ra, door_idx))

    prev_room = {start_room: None}
    prev_door: dict[int, int] = {}
    q = deque([start_room])
    while q and end_room not in prev_room:
        room = q.popleft()
        for nxt, door_idx in graph.get(room, []):
            if nxt in prev_room:
                continue
            prev_room[nxt] = room
            prev_door[nxt] = door_idx
            q.append(nxt)

    room_path: list[int] = []
    door_path: list[int] = []
    if end_room in prev_room:
        cur = end_room
        while cur is not None:
            room_path.append(cur)
            if cur != start_room:
                door_path.append(prev_door[cur])
            cur = prev_room[cur]
        room_path.reverse()
        door_path.reverse()

    return keys, doors, room_ids, room_cells, room_path, door_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock door gaps by matching grey-framed key plaques to wall-lock colors, then mark the unique room-route from the yellow start to the maroon goal.

    Input structure:
    - Black cells are room floors, enclosed by one-pixel-thick colored walls that partition the grid into rectangular rooms.
    - Some internal walls contain a one-cell doorway gap; one of the wall cells touching that gap is replaced by a differently colored lock pixel.
    - A yellow start pixel and a maroon goal pixel lie on room floors.
    - Grey-framed plaques sit on floors; each plaque is a 4x4 grey border with a solid 2x2 colored center that gives one available key color.

    Rule:
    - Read every plaque center color as a usable key color.
    - A doorway is traversable exactly when its lock pixel color matches one of those key colors; otherwise that wall gap stays closed logically.
    - Treat each room as a node, keep only traversable doorways as edges, and find the unique route from the start room to the goal room.
    - Paint every floor cell of each room on that route yellow, paint the traversed doorway-gap cells yellow, erase all key plaques back to black floor, and keep the start and goal markers unchanged.

    Key insight: The puzzle is solved at the room-graph level, not by local pixel motion: key colors first determine which door gaps really connect rooms, and only then does the unique unlocked room-route become visible.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    keys, doors, _room_ids, room_cells, room_path, door_path = _solve_room_path(g)

    for room_id in room_path:
        for r, c in room_cells[room_id]:
            out[r, c] = START

    for door_idx in door_path:
        r, c = doors[door_idx]["pos"]
        out[r, c] = START

    for key in keys:
        for r, c in key["cells"]:
            out[r, c] = FLOOR

    start_pos = tuple(map(int, np.argwhere(g == START)[0]))
    end_pos = tuple(map(int, np.argwhere(g == END)[0]))
    out[start_pos] = START
    out[end_pos] = END
    return out


def _room_id(rr: int, cc: int, cols: int) -> int:
    return rr * cols + cc


def _grid_neighbors(rr: int, cc: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    if rr > 0:
        out.append((rr - 1, cc))
    if rr + 1 < rows:
        out.append((rr + 1, cc))
    if cc > 0:
        out.append((rr, cc - 1))
    if cc + 1 < cols:
        out.append((rr, cc + 1))
    return out


def _make_spanning_tree(rows: int, cols: int, rng: random.Random) -> list[tuple[int, int]]:
    start = (rng.randrange(rows), rng.randrange(cols))
    seen = {start}
    frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for nxt in _grid_neighbors(start[0], start[1], rows, cols):
        frontier.append((start, nxt))

    edges: list[tuple[int, int]] = []
    while frontier:
        idx = rng.randrange(len(frontier))
        (ar, ac), (br, bc) = frontier.pop(idx)
        if (br, bc) in seen:
            continue
        seen.add((br, bc))
        edges.append((_room_id(ar, ac, cols), _room_id(br, bc, cols)))
        for nxt in _grid_neighbors(br, bc, rows, cols):
            if nxt not in seen:
                frontier.append(((br, bc), nxt))
    return edges


def _tree_path(num_rooms: int, edges: list[tuple[int, int]], start: int, end: int) -> list[int]:
    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    prev = {start: None}
    q = deque([start])
    while q and end not in prev:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return []

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _all_pair_paths(num_rooms: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    paths: list[list[int]] = []
    for a in range(num_rooms):
        for b in range(a + 1, num_rooms):
            path = _tree_path(num_rooms, edges, a, b)
            if path:
                paths.append(path)
    return paths


def _sample_room_sizes(count: int, rng: random.Random) -> list[int] | None:
    for _ in range(100):
        sizes = [rng.randint(6, 8) for _ in range(count)]
        if sum(sizes) + count + 1 <= 30:
            return sizes
    return None


def _build_room_layout(rows: int, cols: int, heights: list[int], widths: list[int]) -> tuple[dict[int, tuple[int, int, int, int]], list[int], list[int], int, int]:
    wall_rows = [0]
    cur = 1
    room_boxes: dict[int, tuple[int, int, int, int]] = {}
    for rr, h in enumerate(heights):
        r0 = cur
        r1 = r0 + h - 1
        wall_rows.append(r1 + 1)
        cur = r1 + 2
        room_boxes[_room_id(rr, 0, cols)] = (r0, 0, r1, 0)

    wall_cols = [0]
    cur = 1
    for w in widths:
        c0 = cur
        c1 = c0 + w - 1
        wall_cols.append(c1 + 1)
        cur = c1 + 2

    for rr in range(rows):
        r0 = wall_rows[rr] + 1
        r1 = wall_rows[rr + 1] - 1
        for cc in range(cols):
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            room_boxes[_room_id(rr, cc, cols)] = (r0, c0, r1, c1)

    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    return room_boxes, wall_rows, wall_cols, height, width


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _free_positions_in_room(room: tuple[int, int, int, int], size: tuple[int, int], margin: int, occupied: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    sh, sw = size
    top_min = r0 + margin
    left_min = c0 + margin
    top_max = r1 - sh + 1 - margin
    left_max = c1 - sw + 1 - margin
    out: list[tuple[int, int]] = []
    if top_min > top_max or left_min > left_max:
        return out
    for r in range(top_min, top_max + 1):
        for c in range(left_min, left_max + 1):
            if bool(occupied[r : r + sh, c : c + sw].any()):
                continue
            out.append((r, c))
    return out


def _draw_key(grid: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr in range(4):
        for dc in range(4):
            rr = top + dr
            cc = left + dc
            if dr in (0, 3) or dc in (0, 3):
                grid[rr, cc] = GREY
            else:
                grid[rr, cc] = color
            cells.append((rr, cc))
    return cells


def _choose_start_end_path(num_rooms: int, edges: list[tuple[int, int]], rng: random.Random) -> list[int] | None:
    candidates = [path for path in _all_pair_paths(num_rooms, edges) if len(path) >= 3 and len(path) < num_rooms]
    if not candidates:
        candidates = [path for path in _all_pair_paths(num_rooms, edges) if len(path) >= 2]
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    top = candidates[: max(1, min(6, len(candidates)))]
    return list(rng.choice(top))


def _expected_output(
    inp: np.ndarray,
    room_boxes: dict[int, tuple[int, int, int, int]],
    path_rooms: list[int],
    path_doors: list[tuple[int, int]],
    key_cells: list[tuple[int, int]],
    start_pos: tuple[int, int],
    end_pos: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    for room_id in path_rooms:
        r0, c0, r1, c1 = room_boxes[room_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = START
    for r, c in path_doors:
        out[r, c] = START
    for r, c in key_cells:
        out[r, c] = FLOOR
    out[start_pos] = START
    out[end_pos] = END
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.choice([2, 3])
    cols = rng.choice([2, 3])
    heights = _sample_room_sizes(rows, rng)
    widths = _sample_room_sizes(cols, rng)
    if heights is None or widths is None:
        return None

    room_boxes, wall_rows, wall_cols, h, w = _build_room_layout(rows, cols, heights, widths)
    num_rooms = rows * cols

    tree_edges = _make_spanning_tree(rows, cols, rng)
    room_path = _choose_start_end_path(num_rooms, tree_edges, rng)
    if room_path is None:
        return None

    path_edge_keys = {_edge_key(room_path[i], room_path[i + 1]) for i in range(len(room_path) - 1)}
    off_path_rooms = [room_id for room_id in range(num_rooms) if room_id not in set(room_path)]
    if not off_path_rooms:
        return None

    num_keys = rng.randint(1, min(2, len(KEY_COLORS), len(off_path_rooms)))
    chosen_keys = list(rng.sample(list(KEY_COLORS), num_keys))
    closed_lock_colors = [c for c in KEY_COLORS if c not in chosen_keys]
    if not closed_lock_colors:
        return None

    edge_status: dict[tuple[int, int], bool] = {}
    for a, b in tree_edges:
        key = _edge_key(a, b)
        if key in path_edge_keys:
            edge_status[key] = True
        else:
            edge_status[key] = rng.random() < 0.5

    non_path_edges = [key for key in edge_status if key not in path_edge_keys]
    if non_path_edges and not any(not edge_status[key] for key in non_path_edges):
        edge_status[rng.choice(non_path_edges)] = False

    open_edges = [key for key, open_ in edge_status.items() if open_]
    open_lock_assignment: dict[tuple[int, int], int] = {}
    seed_colors = chosen_keys[:]
    rng.shuffle(seed_colors)
    for i, key in enumerate(open_edges):
        if i < len(seed_colors):
            open_lock_assignment[key] = seed_colors[i]
        else:
            open_lock_assignment[key] = rng.choice(chosen_keys)

    closed_lock_assignment = {key: rng.choice(closed_lock_colors) for key, open_ in edge_status.items() if not open_}

    grid = np.full((h, w), FLOOR, dtype=int)

    top_color, bottom_color, left_color, right_color = rng.choices(WALL_COLORS, k=4)
    grid[0, :] = top_color
    grid[-1, :] = bottom_color
    grid[:, 0] = left_color
    grid[:, -1] = right_color

    h_wall_colors = [rng.choice(WALL_COLORS) for _ in range(rows - 1)]
    v_wall_colors = [rng.choice(WALL_COLORS) for _ in range(cols - 1)]
    for idx, wall_r in enumerate(wall_rows[1:-1]):
        grid[wall_r, :] = h_wall_colors[idx]
    for idx, wall_c in enumerate(wall_cols[1:-1]):
        grid[:, wall_c] = v_wall_colors[idx]

    tree_edge_set = {_edge_key(a, b) for a, b in tree_edges}
    door_positions: dict[tuple[int, int], tuple[int, int]] = {}
    occupied = np.zeros((h, w), dtype=bool)
    path_door_positions: list[tuple[int, int]] = []

    for rr in range(rows):
        for cc in range(cols):
            room_id = _room_id(rr, cc, cols)
            r0, c0, r1, c1 = room_boxes[room_id]
            occupied[r0 : r1 + 1, c0 : c1 + 1] = False

            if cc + 1 < cols:
                other = _room_id(rr, cc + 1, cols)
                edge = _edge_key(room_id, other)
                if edge in tree_edge_set:
                    wall_c = wall_cols[cc + 1]
                    door_r = rng.randint(r0 + 2, r1 - 2)
                    grid[door_r, wall_c] = FLOOR
                    lock_color = open_lock_assignment[edge] if edge_status[edge] else closed_lock_assignment[edge]
                    if rng.random() < 0.5:
                        grid[door_r - 1, wall_c] = lock_color
                    else:
                        grid[door_r + 1, wall_c] = lock_color
                    door_positions[edge] = (door_r, wall_c)

            if rr + 1 < rows:
                other = _room_id(rr + 1, cc, cols)
                edge = _edge_key(room_id, other)
                if edge in tree_edge_set:
                    wall_r = wall_rows[rr + 1]
                    door_c = rng.randint(c0 + 2, c1 - 2)
                    grid[wall_r, door_c] = FLOOR
                    lock_color = open_lock_assignment[edge] if edge_status[edge] else closed_lock_assignment[edge]
                    if rng.random() < 0.5:
                        grid[wall_r, door_c - 1] = lock_color
                    else:
                        grid[wall_r, door_c + 1] = lock_color
                    door_positions[edge] = (wall_r, door_c)

    key_cells: list[tuple[int, int]] = []
    key_rooms = rng.sample(off_path_rooms, num_keys)
    for room_id, key_color in zip(key_rooms, chosen_keys, strict=False):
        positions = _free_positions_in_room(room_boxes[room_id], (4, 4), margin=1, occupied=occupied)
        if not positions:
            return None
        top, left = rng.choice(positions)
        cells = _draw_key(grid, top, left, key_color)
        key_cells.extend(cells)
        for r, c in cells:
            occupied[r, c] = True

    start_room = room_path[0]
    end_room = room_path[-1]
    for room_id, color in ((start_room, START), (end_room, END)):
        positions = _free_positions_in_room(room_boxes[room_id], (1, 1), margin=0, occupied=occupied)
        if not positions:
            return None
        r, c = rng.choice(positions)
        grid[r, c] = color
        occupied[r, c] = True
        if color == START:
            start_pos = (r, c)
        else:
            end_pos = (r, c)

    for i in range(len(room_path) - 1):
        edge = _edge_key(room_path[i], room_path[i + 1])
        path_door_positions.append(door_positions[edge])

    expected = _expected_output(grid, room_boxes, room_path, path_door_positions, key_cells, start_pos, end_pos)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    if len(_find_keys(grid)) != num_keys:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = random.Random()
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
