import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
PATH_COLORS = {BLUE, RED, GREY}

SOURCE_COLORS = [3, 4, 6]
CONTAINER_COLORS = [7, 8, 9]

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _turn_right(direction):
    dr, dc = direction
    return dc, -dr


def _turn_left(direction):
    dr, dc = direction
    return -dc, dr


def _component_cells(grid: np.ndarray, start, visited: np.ndarray):
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    visited[start] = True
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if visited[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return color, cells


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_source(grid: np.ndarray, color: int, cells) -> bool:
    if len(cells) != 4:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _is_container(grid: np.ndarray, color: int, cells):
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    border = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border.add((r, c))
    if set(cells) != border:
        return None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not bool(np.all(interior == BLACK)):
        return None
    interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return interior_cells


def _find_objects(grid: np.ndarray):
    h, w = grid.shape
    network_mask = np.isin(grid, list(PATH_COLORS))
    visited = np.zeros((h, w), dtype=bool)
    sources = []
    containers = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == BLACK or color in PATH_COLORS:
                continue

            color, cells = _component_cells(grid, (r, c), visited)
            adj_network = set()
            adj_pairs = []
            for cr, cc in cells:
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not network_mask[nr, nc]:
                        continue
                    adj_network.add((nr, nc))
                    adj_pairs.append(((cr, cc), (nr, nc)))

            if _is_solid_source(grid, color, cells) and len(adj_network) == 1:
                leaf = next(iter(adj_network))
                incoming_dirs = {
                    (leaf[0] - sr, leaf[1] - sc)
                    for (sr, sc), target in adj_pairs
                    if target == leaf
                }
                if len(incoming_dirs) == 1:
                    sources.append(
                        {
                            "color": color,
                            "leaf": leaf,
                            "incoming": next(iter(incoming_dirs)),
                        }
                    )
                continue

            interior = _is_container(grid, color, cells)
            if interior is not None and len(adj_network) == 1:
                containers.append(
                    {
                        "color": color,
                        "leaf": next(iter(adj_network)),
                        "interior": interior,
                    }
                )

    return sources, containers


def _network_directions(grid: np.ndarray):
    h, w = grid.shape
    network_mask = np.isin(grid, list(PATH_COLORS))
    directions = {}
    for r in range(h):
        for c in range(w):
            if not network_mask[r, c]:
                continue
            dirs = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and network_mask[nr, nc]:
                    dirs.append((dr, dc))
            directions[(r, c)] = dirs
    return directions


def _next_direction(grid: np.ndarray, directions, pos, incoming):
    dirs = directions.get(pos, [])
    if not dirs:
        return None

    if incoming is None:
        return dirs[0] if len(dirs) == 1 else None

    back = (-incoming[0], -incoming[1])
    if len(dirs) == 1:
        return dirs[0] if dirs[0] != back else None

    if len(dirs) == 2:
        forward = [d for d in dirs if d != back]
        return forward[0] if len(forward) == 1 else None

    marker = int(grid[pos])
    desired = incoming
    if marker == RED:
        desired = _turn_right(incoming)
    elif marker == BLUE:
        desired = _turn_left(incoming)
    return desired if desired in dirs else None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sources, containers = _find_objects(g)
    if not sources or not containers:
        return out

    directions = _network_directions(g)
    container_by_leaf = {tuple(obj["leaf"]): obj for obj in containers}

    active = [
        {
            "pos": tuple(src["leaf"]),
            "incoming": tuple(src["incoming"]),
            "color": int(src["color"]),
        }
        for src in sources
    ]

    for _ in range(g.size * 4 + 10):
        if not active:
            break

        intents = []
        for signal in active:
            pos = signal["pos"]
            container = container_by_leaf.get(pos)
            if container is not None:
                for r, c in container["interior"]:
                    out[r, c] = signal["color"]
                continue

            direction = _next_direction(g, directions, pos, signal["incoming"])
            if direction is None:
                continue

            intents.append(
                {
                    "from": pos,
                    "to": (pos[0] + direction[0], pos[1] + direction[1]),
                    "incoming": direction,
                    "color": signal["color"],
                }
            )

        if not intents:
            break

        collided = set()
        by_dest = {}
        for idx, intent in enumerate(intents):
            by_dest.setdefault(intent["to"], []).append(idx)
        for ids in by_dest.values():
            if len(ids) > 1:
                collided.update(ids)

        edge_map = {(intent["from"], intent["to"]): idx for idx, intent in enumerate(intents)}
        for idx, intent in enumerate(intents):
            other = edge_map.get((intent["to"], intent["from"]))
            if other is not None and other != idx:
                collided.add(idx)
                collided.add(other)

        next_active = []
        for idx, intent in enumerate(intents):
            if idx in collided:
                continue
            next_active.append(
                {
                    "pos": intent["to"],
                    "incoming": intent["incoming"],
                    "color": intent["color"],
                }
            )
        active = next_active

    return out


def _draw_source(canvas: np.ndarray, top: int, left: int, color: int):
    canvas[top : top + 2, left : left + 2] = color


def _draw_container(canvas: np.ndarray, top: int, left: int, color: int):
    canvas[top : top + 4, left : left + 4] = color
    canvas[top + 1 : top + 3, left + 1 : left + 3] = BLACK


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    points = np.argwhere(patch != BLACK)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    r0 = max(0, int(r0) - 1)
    c0 = max(0, int(c0) - 1)
    r1 = min(patch.shape[0] - 1, int(r1) + 1)
    c1 = min(patch.shape[1] - 1, int(c1) + 1)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(patch, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _make_turn_patch(rng: random.Random, source_color: int) -> np.ndarray:
    canvas = np.zeros((24, 24), dtype=int)
    src_left = 2
    pre_len = rng.randint(2, 5)
    len_e = rng.randint(2, 5)
    len_n = rng.randint(2, 5)
    len_s = rng.randint(2, 5)
    node_r = 10
    node_c = src_left + 2 + pre_len

    marker = rng.choice([GREY, RED, BLUE])
    selected = {GREY: "E", RED: "S", BLUE: "N"}[marker]
    present = {"E": False, "N": False, "S": False}

    if selected == "E":
        present["E"] = True
        if rng.random() < 0.5:
            present["N"] = True
            present["S"] = rng.random() < 0.5
        else:
            present["S"] = True
            present["N"] = rng.random() < 0.5
    elif selected == "N":
        present["N"] = True
        present["E"] = True
        present["S"] = rng.random() < 0.6
    else:
        present["S"] = True
        present["E"] = True
        present["N"] = rng.random() < 0.6

    _draw_source(canvas, node_r - 1, src_left, source_color)
    canvas[node_r, src_left + 2 : node_c + 1] = GREY
    canvas[node_r, node_c] = marker

    if present["N"]:
        canvas[node_r - len_n : node_r, node_c] = GREY
    if present["S"]:
        canvas[node_r + 1 : node_r + len_s + 1, node_c] = GREY
    if present["E"]:
        canvas[node_r, node_c + 1 : node_c + len_e + 1] = GREY

    extra_dirs = [d for d in ("N", "E", "S") if present[d] and d != selected]
    container_flags = {selected: True}
    for direction in extra_dirs:
        container_flags[direction] = rng.random() < 0.6
    if extra_dirs and not any(container_flags[d] for d in extra_dirs) and rng.random() < 0.5:
        container_flags[rng.choice(extra_dirs)] = True

    if present["N"] and container_flags.get("N", False):
        _draw_container(canvas, node_r - len_n - 4, node_c - 1, rng.choice(CONTAINER_COLORS))
    if present["S"] and container_flags.get("S", False):
        _draw_container(canvas, node_r + len_s + 1, node_c - 1, rng.choice(CONTAINER_COLORS))
    if present["E"] and container_flags.get("E", False):
        _draw_container(canvas, node_r - 1, node_c + len_e + 1, rng.choice(CONTAINER_COLORS))

    return _crop_patch(canvas)


def _make_collision_patch(rng: random.Random, left_color: int, right_color: int) -> np.ndarray:
    canvas = np.zeros((10, 24), dtype=int)
    row = 4
    src_left = 2
    run_len = rng.randint(6, 10)
    left_leaf = src_left + 2
    right_leaf = left_leaf + run_len

    _draw_source(canvas, row - 1, src_left, left_color)
    _draw_source(canvas, row - 1, right_leaf + 1, right_color)
    canvas[row, left_leaf : right_leaf + 1] = GREY

    return _crop_patch(canvas)


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    for _ in range(200):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            continue

        mask = patch != BLACK
        if bool(np.any(grid[top : top + ph, left : left + pw][mask] != BLACK)):
            continue

        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        occupied[r0:r1, c0:c1] = True
        return True

    return False


def _gen_attempt(rng: random.Random):
    collision = rng.random() < 0.5
    turn_count = 1 if collision else (2 if rng.random() < 0.6 else 1)
    total_sources = turn_count + (2 if collision else 0)

    source_colors = SOURCE_COLORS[:]
    rng.shuffle(source_colors)
    source_colors = source_colors[:total_sources]

    patches = []
    for i in range(turn_count):
        patches.append(_random_transform(_make_turn_patch(rng, source_colors[i]), rng))
    if collision:
        patches.append(
            _random_transform(
                _make_collision_patch(rng, source_colors[-2], source_colors[-1]),
                rng,
            )
        )

    rng.shuffle(patches)
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    area = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches)
    min_side = max(max_h + 4, max_w + 4, int(area ** 0.5) + 7, 22)
    if min_side > 34:
        return None

    grid_h = rng.randint(min_side, min(34, min_side + 6))
    grid_w = rng.randint(min_side, min(34, min_side + 6))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for patch in sorted(patches, key=lambda p: p.shape[0] * p.shape[1], reverse=True):
        if not _place_patch(grid, occupied, patch, rng):
            return None

    sources, containers = _find_objects(grid)
    if len(sources) != total_sources or not containers:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
