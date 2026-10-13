from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Node = tuple[int, int]
Edge = frozenset[Node]

ROOM_H = 5
ROOM_W = 5
MAX_GENERATION_ATTEMPTS = 500
_GEN_COUNTER = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_straight_run(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return min(h, w) == 1 and int(coords.shape[0]) == max(h, w)


def _marker_near_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(coords)
    return min(r0, c0, h - 1 - r1, w - 1 - c1) <= 1


def _identify_roles(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    assert len(colors) == 3, "expected background plus wall, endpoints, and marker"

    components = {color: _component_coords(g == color) for color in colors}

    endpoint_candidates = []
    for color in colors:
        comps = components[color]
        if len(comps) != 2:
            continue
        if all(int(comp.shape[0]) >= 1 for comp in comps):
            endpoint_candidates.append(color)
    assert len(endpoint_candidates) == 1, "expected exactly one color with two endpoint components"
    endpoint_color = endpoint_candidates[0]

    remaining = [color for color in colors if color != endpoint_color]
    marker_candidates = []
    for color in remaining:
        comps = components[color]
        if len(comps) != 1:
            continue
        comp = comps[0]
        if 1 <= int(comp.shape[0]) <= 3 and _is_straight_run(comp) and _marker_near_border(comp, g.shape):
            marker_candidates.append(color)
    assert len(marker_candidates) == 1, "expected a unique marker run"
    marker_color = marker_candidates[0]
    wall_color = next(color for color in remaining if color != marker_color)

    marker = components[marker_color][0]
    for r, c in marker:
        for nr in range(int(r) - 1, int(r) + 2):
            for nc in range(int(c) - 1, int(c) + 2):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    assert int(g[nr, nc]) != wall_color, "marker run must stay detached from walls"

    return {
        "bg": bg,
        "wall": wall_color,
        "endpoint": endpoint_color,
        "marker": marker_color,
        "endpoint_components": components[endpoint_color],
        "marker_coords": marker,
        "key_width": int(marker.shape[0]),
    }


def _collect_doors(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: list[dict] = []
    candidate_vertical = {
        c
        for c in range(1, w - 1)
        if int(np.count_nonzero(g[:, c] == wall)) >= max(4, h // 2)
    }
    candidate_horizontal = {
        r
        for r in range(1, h - 1)
        if int(np.count_nonzero(g[r, :] == wall)) >= max(4, w // 2)
    }

    for c in sorted(candidate_vertical):
        r = 0
        while r < h:
            if int(g[r, c]) != bg:
                r += 1
                continue
            r0 = r
            while r < h and int(g[r, c]) == bg:
                r += 1
            r1 = r - 1
            width = r1 - r0 + 1
            if not (1 <= width <= 3 and r0 > 0 and r1 + 1 < h):
                continue
            if int(g[r0 - 1, c]) != wall or int(g[r1 + 1, c]) != wall:
                continue
            if all(int(g[rr, c - 1]) != wall and int(g[rr, c + 1]) != wall for rr in range(r0, r1 + 1)):
                doors.append(
                    {
                        "orientation": "vertical",
                        "width": width,
                        "coords": [(rr, c) for rr in range(r0, r1 + 1)],
                    }
                )

    for r in sorted(candidate_horizontal):
        c = 0
        while c < w:
            if int(g[r, c]) != bg:
                c += 1
                continue
            c0 = c
            while c < w and int(g[r, c]) == bg:
                c += 1
            c1 = c - 1
            width = c1 - c0 + 1
            if not (1 <= width <= 3 and c0 > 0 and c1 + 1 < w):
                continue
            if int(g[r, c0 - 1]) != wall or int(g[r, c1 + 1]) != wall:
                continue
            if all(int(g[r - 1, cc]) != wall and int(g[r + 1, cc]) != wall for cc in range(c0, c1 + 1)):
                doors.append(
                    {
                        "orientation": "horizontal",
                        "width": width,
                        "coords": [(r, cc) for cc in range(c0, c1 + 1)],
                    }
                )

    assert doors, "expected at least one door gap"
    return doors


def _bfs_counts(passable: np.ndarray, starts: list[Coord]) -> tuple[np.ndarray, np.ndarray]:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    count = np.zeros((h, w), dtype=int)
    q = deque()
    for r, c in starts:
        rr = int(r)
        cc = int(c)
        if not passable[rr, cc] or dist[rr, cc] == 0:
            continue
        dist[rr, cc] = 0
        count[rr, cc] = 1
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w and passable[nr, nc]):
                continue
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                count[nr, nc] = count[r, c]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                count[nr, nc] = min(2, int(count[nr, nc]) + int(count[r, c]))
    return dist, count


def _unique_shortest_path(passable: np.ndarray, starts: list[Coord], goals: list[Coord]) -> tuple[list[Coord], int]:
    dist, count = _bfs_counts(passable, starts)
    reachable = [(int(dist[r, c]), (int(r), int(c))) for r, c in goals if dist[int(r), int(c)] >= 0]
    assert reachable, "endpoints must connect after filtering doors"
    shortest = min(d for d, _ in reachable)

    total = 0
    end_coord: Coord | None = None
    for r, c in goals:
        rr = int(r)
        cc = int(c)
        if int(dist[rr, cc]) != shortest:
            continue
        total = min(2, total + int(count[rr, cc]))
        if count[rr, cc] > 0 and end_coord is None:
            end_coord = (rr, cc)
    assert total == 1 and end_coord is not None, "expected a unique shortest path"

    start_set = {(int(r), int(c)) for r, c in starts}
    path = [end_coord]
    cur = end_coord
    while cur not in start_set:
        r, c = cur
        preds = []
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]:
                if int(dist[nr, nc]) == int(dist[r, c]) - 1 and int(count[nr, nc]) > 0:
                    preds.append((nr, nc))
        assert len(preds) == 1, "path backtracking must stay unique"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path, shortest


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    roles = _identify_roles(g)
    bg = int(roles["bg"])
    wall = int(roles["wall"])
    endpoint_color = int(roles["endpoint"])
    doors = _collect_doors(g, bg, wall)

    all_door_mask = np.zeros_like(g, dtype=bool)
    blocked_wrong_doors = np.zeros_like(g, dtype=bool)
    for door in doors:
        coords = door["coords"]
        width = int(door["width"])
        for r, c in coords:
            all_door_mask[int(r), int(c)] = True
            if width != int(roles["key_width"]):
                blocked_wrong_doors[int(r), int(c)] = True

    passable_bg = (g == bg) | (g == endpoint_color)
    passable_filtered = passable_bg & ~blocked_wrong_doors

    start_comp, goal_comp = roles["endpoint_components"]
    starts = [(int(r), int(c)) for r, c in start_comp]
    goals = [(int(r), int(c)) for r, c in goal_comp]
    path, filtered_len = _unique_shortest_path(passable_filtered, starts, goals)

    dist_all, _ = _bfs_counts(passable_bg, starts)
    all_reachable = [int(dist_all[r, c]) for r, c in goals if dist_all[int(r), int(c)] >= 0]
    assert all_reachable, "endpoints must connect in the original maze"
    all_len = min(all_reachable)

    return {
        **roles,
        "doors": doors,
        "door_mask": all_door_mask,
        "wrong_door_mask": blocked_wrong_doors,
        "path": path,
        "filtered_len": int(filtered_len),
        "all_len": int(all_len),
        "passable_filtered": passable_filtered,
        "passable_all": passable_bg,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the detached marker run, use that length as the key width, and paint the unique shortest route that only crosses matching-width doors.

    Input structure:
    - The grid has a dominant background, one wall color forming thin orthogonal partitions, two same-color endpoint blocks, and one detached marker run of a third color.
    - Door gaps appear as short 1-to-3-cell openings inside horizontal or vertical wall segments.
    - The marker run sits near the border in empty space and does not touch the wall structure.

    Rule:
    - Measure the marker-run length k.
    - Reinterpret every door whose width is not exactly k as blocked wall, while doors of width k stay traversable.
    - Find the unique shortest 4-connected path between the two endpoint components under that rewritten connectivity.
    - Recolor only the background cells on that path with the endpoint color; walls, door gaps, endpoints, and the marker remain unchanged.

    Key insight:
    - The marker run is the key that rewrites which openings count as doors, so the visually shorter route through a wrong-width gap is a decoy until connectivity is filtered first.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)

    out = g.copy()
    endpoint_color = int(info["endpoint"])
    for r, c in info["path"]:
        rr = int(r)
        cc = int(c)
        if out[rr, cc] == int(info["bg"]) and not info["door_mask"][rr, cc]:
            out[rr, cc] = endpoint_color
    return out


def _room_mask() -> np.ndarray:
    mask = np.zeros((ROOM_H, ROOM_W), dtype=bool)
    cr = ROOM_H // 2
    cc = ROOM_W // 2
    mask[cr, :] = True
    mask[:, cc] = True
    mask[max(0, cr - 1) : min(ROOM_H, cr + 2), 0] = True
    mask[max(0, cr - 1) : min(ROOM_H, cr + 2), ROOM_W - 1] = True
    mask[0, max(0, cc - 1) : min(ROOM_W, cc + 2)] = True
    mask[ROOM_H - 1, max(0, cc - 1) : min(ROOM_W, cc + 2)] = True
    return mask


def _edge(a: Node, b: Node) -> Edge:
    return frozenset((a, b))


def _grid_neighbors(node: Node, rows: int, cols: int) -> list[Node]:
    r, c = node
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    random.shuffle(out)
    return out


def _turn_count(path: list[Node]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _enumerate_paths(rows: int, cols: int, start: Node, goal: Node, max_len: int, limit: int = 4000) -> list[list[Node]]:
    paths: list[list[Node]] = []
    goal_r, goal_c = goal

    def dfs(node: Node, path: list[Node], seen: set[Node]) -> None:
        if len(paths) >= limit:
            return
        used = len(path) - 1
        rem = abs(node[0] - goal_r) + abs(node[1] - goal_c)
        if used + rem > max_len:
            return
        if node == goal:
            paths.append(path.copy())
            return
        for nxt in _grid_neighbors(node, rows, cols):
            if nxt in seen:
                continue
            path.append(nxt)
            seen.add(nxt)
            dfs(nxt, path, seen)
            seen.remove(nxt)
            path.pop()

    dfs(start, [start], {start})
    return paths


def _door_span(center: int, width: int) -> tuple[int, int]:
    if width == 1:
        return center, center
    if width == 2:
        if random.choice([False, True]):
            return center - 1, center
        return center, center + 1
    return center - 1, center + 1


def _render_maze(
    rows: int,
    cols: int,
    wall_color: int,
    endpoint_color: int,
    edge_widths: dict[Edge, int],
    start: Node,
    goal: Node,
) -> np.ndarray:
    maze_h = rows * (ROOM_H + 1) + 1
    maze_w = cols * (ROOM_W + 1) + 1
    maze = np.full((maze_h, maze_w), wall_color, dtype=int)
    room_mask = _room_mask()

    for rr in range(rows):
        for cc in range(cols):
            r0 = 1 + rr * (ROOM_H + 1)
            c0 = 1 + cc * (ROOM_W + 1)
            room = maze[r0 : r0 + ROOM_H, c0 : c0 + ROOM_W]
            room[room_mask] = 0

    for rr in range(rows):
        for cc in range(cols):
            node = (rr, cc)
            r0 = 1 + rr * (ROOM_H + 1)
            c0 = 1 + cc * (ROOM_W + 1)
            center_r = r0 + ROOM_H // 2
            center_c = c0 + ROOM_W // 2

            if cc + 1 < cols:
                width = edge_widths.get(_edge(node, (rr, cc + 1)))
                if width is not None:
                    c_wall = c0 + ROOM_W
                    r1, r2 = _door_span(center_r, int(width))
                    maze[r1 : r2 + 1, c_wall] = 0
            if rr + 1 < rows:
                width = edge_widths.get(_edge(node, (rr + 1, cc)))
                if width is not None:
                    r_wall = r0 + ROOM_H
                    c1, c2 = _door_span(center_c, int(width))
                    maze[r_wall, c1 : c2 + 1] = 0

            if node == start or node == goal:
                maze[center_r, center_c] = endpoint_color

    return maze


def _place_marker(grid: np.ndarray, bg_color: int, marker_color: int, key_width: int, side: str) -> None:
    h, w = grid.shape
    if side in {"top", "bottom"}:
        row = 0 if side == "top" else h - 1
        candidates = []
        for c0 in range(w - key_width + 1):
            coords = [(row, c0 + dc) for dc in range(key_width)]
            if all(int(grid[r, c]) == int(bg_color) for r, c in coords):
                ok = True
                for r, c in coords:
                    for nr in range(max(0, r - 1), min(h, r + 2)):
                        for nc in range(max(0, c - 1), min(w, c + 2)):
                            if int(grid[nr, nc]) != int(bg_color):
                                ok = False
                if ok:
                    candidates.append(coords)
        if candidates:
            for r, c in random.choice(candidates):
                grid[r, c] = marker_color
            return
    else:
        col = 0 if side == "left" else w - 1
        candidates = []
        for r0 in range(h - key_width + 1):
            coords = [(r0 + dr, col) for dr in range(key_width)]
            if all(int(grid[r, c]) == int(bg_color) for r, c in coords):
                ok = True
                for r, c in coords:
                    for nr in range(max(0, r - 1), min(h, r + 2)):
                        for nc in range(max(0, c - 1), min(w, c + 2)):
                            if int(grid[nr, nc]) != int(bg_color):
                                ok = False
                if ok:
                    candidates.append(coords)
        if candidates:
            for r, c in random.choice(candidates):
                grid[r, c] = marker_color
            return
    raise RuntimeError("failed to place detached marker")


def _build_case(key_width: int) -> np.ndarray | None:
    rows, cols = random.choice([(3, 4), (4, 3), (4, 4)])
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    start, goal = random.sample(nodes, 2)
    if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 3:
        return None

    shortest = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    paths = _enumerate_paths(rows, cols, start, goal, shortest + 4)
    if not paths:
        return None

    by_len: dict[int, list[list[Node]]] = {}
    for path in paths:
        by_len.setdefault(len(path) - 1, []).append(path)

    if shortest not in by_len:
        return None
    decoy_path = random.choice(by_len[shortest])
    decoy_edges = {_edge(decoy_path[i], decoy_path[i + 1]) for i in range(len(decoy_path) - 1)}

    valid_candidates = []
    for length, group in by_len.items():
        if length < shortest + 2:
            continue
        for path in group:
            edges = {_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}
            if decoy_edges - edges and _turn_count(path) >= 1:
                valid_candidates.append(path)
    if not valid_candidates:
        return None
    valid_path = random.choice(valid_candidates)
    valid_edges = {_edge(valid_path[i], valid_path[i + 1]) for i in range(len(valid_path) - 1)}

    edge_widths: dict[Edge, int] = {}
    for edge in valid_edges:
        edge_widths[edge] = key_width

    other_widths = [w for w in (1, 2, 3) if w != key_width]
    for edge in decoy_edges - valid_edges:
        edge_widths[edge] = random.choice(other_widths)

    for r in range(rows):
        for c in range(cols):
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= rows or nc >= cols:
                    continue
                edge = _edge((r, c), (nr, nc))
                if edge in edge_widths:
                    continue
                if random.random() < 0.35:
                    edge_widths[edge] = random.choice(other_widths)

    palette = random.sample(range(10), 4)
    bg_color, wall_color, endpoint_color, marker_color = palette
    maze = _render_maze(rows, cols, wall_color, endpoint_color, edge_widths, start, goal)

    maze_h, maze_w = maze.shape
    marker_side = random.choice(["top", "bottom", "left", "right"])
    top = 2 if marker_side == "top" else 1
    bottom = 2 if marker_side == "bottom" else 1
    left = 2 if marker_side == "left" else 1
    right = 2 if marker_side == "right" else 1

    extra_h = 30 - (maze_h + top + bottom)
    extra_w = 30 - (maze_w + left + right)
    if extra_h < 0 or extra_w < 0:
        return None
    add_top = random.randint(0, extra_h)
    add_bottom = random.randint(0, extra_h - add_top)
    add_left = random.randint(0, extra_w)
    add_right = random.randint(0, extra_w - add_left)
    top += add_top
    bottom += add_bottom
    left += add_left
    right += add_right

    height = maze_h + top + bottom
    width = maze_w + left + right
    grid = np.full((height, width), bg_color, dtype=int)
    grid[top : top + maze_h, left : left + maze_w] = np.where(maze == 0, bg_color, maze)
    _place_marker(grid, bg_color, marker_color, key_width, marker_side)

    return grid


def _validate_generated_case(grid: np.ndarray, expected_key_width: int) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if int(info["key_width"]) != int(expected_key_width):
        return False
    if int(info["filtered_len"]) <= 4:
        return False
    if int(info["all_len"]) >= int(info["filtered_len"]):
        return False

    door_widths = [int(door["width"]) for door in info["doors"]]
    if expected_key_width not in door_widths:
        return False
    if not any(width != expected_key_width for width in door_widths):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    path_cells = 0
    for r, c in info["path"]:
        if int(grid[int(r), int(c)]) == int(info["bg"]) and not info["door_mask"][int(r), int(c)]:
            path_cells += 1
            if int(out[int(r), int(c)]) != int(info["endpoint"]):
                return False
        if info["door_mask"][int(r), int(c)] and int(out[int(r), int(c)]) != int(grid[int(r), int(c)]):
            return False
    if path_cells < 3:
        return False

    for r, c in info["marker_coords"]:
        if int(out[int(r), int(c)]) != int(grid[int(r), int(c)]):
            return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GENERATION_ATTEMPTS):
        _GEN_COUNTER += 1
        key_width = ((_GEN_COUNTER - 1) % 3) + 1
        grid = _build_case(key_width)
        if grid is not None and _validate_generated_case(grid, key_width):
            return grid
    raise RuntimeError("failed to generate a valid door-key maze input")
