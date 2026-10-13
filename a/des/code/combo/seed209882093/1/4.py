import random
from collections.abc import Iterable

import numpy as np


BG = 0
PATH = 5
WALL = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
ROOM_H_RANGE = (7, 11)
ROOM_W_RANGE = (7, 11)
GEN_RETRIES = 400
ROOM_PLACEMENT_TRIES = 80
ROOM_CONTENT_TRIES = 80

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _is_rectangular_room(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(patch[0, :] == WALL)):
        return False
    if not bool(np.all(patch[-1, :] == WALL)):
        return False
    if not bool(np.all(patch[:, 0] == WALL)):
        return False
    if not bool(np.all(patch[:, -1] == WALL)):
        return False
    if bool(np.any(patch[1:-1, 1:-1] == WALL)):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each gray path with its room's seed color, then place one matching
    marker just outside the wall that the far path endpoint touches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    room_components = _components(g == WALL)
    assert room_components, "expected at least one room"

    for room in room_components:
        box = _bbox(room)
        assert _is_rectangular_room(g, box), "wall component must be a hollow rectangle"
        r0, c0, r1, c1 = box

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((interior != BG) & (interior != PATH))
        assert seed_coords.shape == (1, 2), "each room must contain exactly one seed"
        seed_r, seed_c = int(seed_coords[0, 0] + r0 + 1), int(seed_coords[0, 1] + c0 + 1)
        seed_color = int(g[seed_r, seed_c])
        assert seed_color not in {BG, PATH, WALL}

        path_components = _components(interior == PATH)
        assert len(path_components) == 1, "each room must contain exactly one path"
        path_local = path_components[0]
        path_cells = [(int(r + r0 + 1), int(c + c0 + 1)) for r, c in path_local]
        path_set = set(path_cells)

        degrees = {
            cell: sum((nr, nc) in path_set for nr, nc in _neighbors4(cell[0], cell[1]))
            for cell in path_cells
        }
        assert max(degrees.values()) <= 2, "path must not branch"

        if len(path_cells) == 1:
            endpoints = [path_cells[0]]
        else:
            endpoints = [cell for cell, degree in degrees.items() if degree == 1]
            assert len(endpoints) == 2, "path must have exactly two endpoints"

        seed_adjacent = [
            cell
            for cell in endpoints
            if abs(cell[0] - seed_r) + abs(cell[1] - seed_c) == 1
        ]
        assert seed_adjacent, "seed must touch one path endpoint"
        start = seed_adjacent[0]
        end = start if len(endpoints) == 1 else endpoints[1] if endpoints[0] == start else endpoints[0]

        wall_dirs = []
        for dr, dc in DIR4:
            wr, wc = end[0] + dr, end[1] + dc
            if 0 <= wr < g.shape[0] and 0 <= wc < g.shape[1] and int(g[wr, wc]) == WALL:
                wall_dirs.append((dr, dc))
        assert len(wall_dirs) == 1, "far endpoint must touch exactly one wall cell"
        dr, dc = wall_dirs[0]

        for pr, pc in path_cells:
            out[pr, pc] = seed_color

        mark_r = end[0] + 2 * dr
        mark_c = end[1] + 2 * dc
        assert 0 <= mark_r < g.shape[0] and 0 <= mark_c < g.shape[1]
        out[mark_r, mark_c] = seed_color

    return out


def _choose_room_count(h: int, w: int, rng: random.Random) -> int:
    max_rooms = 1
    if h >= 22 and w >= 22:
        max_rooms = 2
    if h >= 25 and w >= 25:
        max_rooms = 3
    if h >= 28 and w >= 28:
        max_rooms = 4
    options = list(range(1, max_rooms + 1))
    weights = [2, 4, 4, 3][:max_rooms]
    return int(rng.choices(options, weights=weights, k=1)[0])


def _random_spanning_tree(
    allowed: set[tuple[int, int]],
    root: tuple[int, int],
    rng: random.Random,
) -> dict[tuple[int, int], tuple[int, int] | None]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    visited = {root}
    stack = [root]
    while stack:
        cell = stack[-1]
        neighbors = [
            nb
            for nb in _neighbors4(cell[0], cell[1])
            if nb in allowed and nb not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        nxt = rng.choice(neighbors)
        visited.add(nxt)
        parent[nxt] = cell
        stack.append(nxt)
    assert len(parent) == len(allowed)
    return parent


def _chain_to_root(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    path = []
    cur: tuple[int, int] | None = start
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path


def _seed_candidates(
    start: tuple[int, int],
    path_set: set[tuple[int, int]],
    core_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for cell in _neighbors4(start[0], start[1]):
        if cell not in core_set or cell in path_set:
            continue
        adj_path = sum(nb in path_set for nb in _neighbors4(cell[0], cell[1]))
        if adj_path == 1:
            candidates.append(cell)
    return candidates


def _build_room(rh: int, rw: int, seed_color: int, rng: random.Random) -> np.ndarray | None:
    ih, iw = rh - 2, rw - 2
    core = [(r, c) for r in range(1, ih - 1) for c in range(1, iw - 1)]
    core_set = set(core)
    assert core

    for _ in range(ROOM_CONTENT_TRIES):
        side = rng.choice(("top", "bottom", "left", "right"))
        if side == "top":
            exit_cell = (0, rng.randint(1, iw - 2))
        elif side == "bottom":
            exit_cell = (ih - 1, rng.randint(1, iw - 2))
        elif side == "left":
            exit_cell = (rng.randint(1, ih - 2), 0)
        else:
            exit_cell = (rng.randint(1, ih - 2), iw - 1)

        allowed = set(core_set)
        allowed.add(exit_cell)
        parent = _random_spanning_tree(allowed, exit_cell, rng)

        candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        for start in core:
            path = _chain_to_root(parent, start)
            if len(path) < 2:
                continue
            path_set = set(path)
            seeds = _seed_candidates(start, path_set, core_set)
            if seeds:
                candidates.append((path, seeds))

        if not candidates:
            continue

        if rng.random() < 0.55:
            max_len = max(len(path) for path, _ in candidates)
            long_candidates = [item for item in candidates if len(item[0]) >= max_len - 1]
            if long_candidates:
                candidates = long_candidates

        path, seeds = rng.choice(candidates)
        seed_cell = rng.choice(seeds)

        room = np.full((rh, rw), BG, dtype=int)
        room[0, :] = WALL
        room[-1, :] = WALL
        room[:, 0] = WALL
        room[:, -1] = WALL
        for pr, pc in path:
            room[pr + 1, pc + 1] = PATH
        room[seed_cell[0] + 1, seed_cell[1] + 1] = seed_color
        return room

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    room_count = _choose_room_count(h, w, rng)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    unique_colors = rng.random() < 0.75 and room_count <= len(SEED_COLORS)
    if unique_colors:
        room_colors = rng.sample(SEED_COLORS, k=room_count)
    else:
        room_colors = [rng.choice(SEED_COLORS) for _ in range(room_count)]

    for i in range(room_count):
        placed = False
        for _ in range(ROOM_PLACEMENT_TRIES):
            rh = rng.randint(*ROOM_H_RANGE)
            rw = rng.randint(*ROOM_W_RANGE)
            if h - rh < 3 or w - rw < 3:
                continue

            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1

            if bool(np.any(blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2])):
                continue

            room = _build_room(rh, rw, room_colors[i], rng)
            if room is None:
                continue

            grid[r0 : r1 + 1, c0 : c1 + 1] = room
            blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if bool(np.array_equal(out, grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
