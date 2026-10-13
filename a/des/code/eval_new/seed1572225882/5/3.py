import random
from collections import deque

import numpy as np


BG = 0
WALL = 9
CHAIN = (2, 7, 4, 3, 8)
CHAIN_INDEX = {color: i for i, color in enumerate(CHAIN)}


def _wall_lines(grid, wall):
    h, w = grid.shape
    row_threshold = max(2, w // 2)
    col_threshold = max(2, h // 2)
    rows = [r for r in range(h) if np.count_nonzero(grid[r, :] == wall) >= row_threshold]
    cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == wall) >= col_threshold]
    return rows, cols


def _rooms_from_walls(grid, wall):
    h_lines, v_lines = _wall_lines(grid, wall)
    rooms = []
    lookup = {}
    for ri in range(len(h_lines) - 1):
        r0, r1 = h_lines[ri] + 1, h_lines[ri + 1] - 1
        if r0 > r1:
            continue
        for ci in range(len(v_lines) - 1):
            c0, c1 = v_lines[ci] + 1, v_lines[ci + 1] - 1
            if c0 > c1:
                continue
            idx = len(rooms)
            rooms.append((r0, r1, c0, c1, ri, ci))
            lookup[(ri, ci)] = idx
    return rooms, lookup, h_lines, v_lines


def _find_seeds(grid, rooms, bg, wall):
    seeds = {}
    for idx, (r0, r1, c0, c1, _, _) in enumerate(rooms):
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(x) for x in np.unique(patch) if int(x) not in (bg, wall)]
        if colors:
            seeds[idx] = colors[0]
    return seeds


def _room_graph(grid, rooms, lookup, h_lines, v_lines, wall):
    adjacency = {i: set() for i in range(len(rooms))}
    doors = []

    for idx, (_, _, _, _, ri, ci) in enumerate(rooms):
        right = lookup.get((ri, ci + 1))
        if right is not None:
            wall_col = v_lines[ci + 1]
            r0, r1, _, _, _, _ = rooms[idx]
            gap = [(r, wall_col) for r in range(r0, r1 + 1) if grid[r, wall_col] != wall]
            if gap:
                adjacency[idx].add(right)
                adjacency[right].add(idx)
                doors.append((idx, right, gap))

        down = lookup.get((ri + 1, ci))
        if down is not None:
            wall_row = h_lines[ri + 1]
            _, _, c0, c1, _, _ = rooms[idx]
            gap = [(wall_row, c) for c in range(c0, c1 + 1) if grid[wall_row, c] != wall]
            if gap:
                adjacency[idx].add(down)
                adjacency[down].add(idx)
                doors.append((idx, down, gap))

    return adjacency, doors


def _propagate_colors(adjacency, seeds):
    room_colors = dict(seeds)
    best = {}

    for start, color in seeds.items():
        if color not in CHAIN_INDEX:
            continue
        start_index = CHAIN_INDEX[color]
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            room, dist = queue.popleft()
            chain_pos = start_index + dist
            if chain_pos >= len(CHAIN):
                continue
            if room not in seeds:
                old = best.get(room)
                if old is None or chain_pos < old:
                    best[room] = chain_pos
            if chain_pos + 1 >= len(CHAIN):
                continue
            for nxt in adjacency.get(room, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))

    for room, chain_pos in best.items():
        if room not in room_colors:
            room_colors[room] = CHAIN[chain_pos]
    return room_colors


def _stronger_color(a, b):
    ai = CHAIN_INDEX.get(a)
    bi = CHAIN_INDEX.get(b)
    if ai is None:
        return b if bi is not None else a
    if bi is None:
        return a
    return a if ai <= bi else b


def main(grid: np.ndarray) -> np.ndarray:
    """Fill rectangular rooms by propagating seed colors through door gaps with one-step fading.

    Input structure:
    The most common color is the background, and maroon wall lines form a grid of rectangular
    rooms.  Internal wall segments have one single-cell opening that connects neighboring
    rooms.  Some room interiors contain one colored seed pixel from the ordered visible chain
    red, orange, yellow, green, azure.

    Rule:
    A seeded room is filled with its own seed color.  Across each door into another room, the
    color advances one step later in the chain.  Propagation continues over the room graph until
    the chain ends.  If several sources can reach the same initially empty room, the room takes
    the earliest, strongest chain color among the candidates.  Walls are preserved.

    Key insight:
    The wall openings define the room graph, and the number of crossed openings gives the
    fading distance in the fixed color chain.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    if grid.ndim != 2 or grid.size == 0:
        return out

    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    wall = WALL if np.any(grid == WALL) else bg
    if wall == bg:
        return out

    rooms, lookup, h_lines, v_lines = _rooms_from_walls(grid, wall)
    if not rooms:
        return out

    seeds = _find_seeds(grid, rooms, bg, wall)
    adjacency, doors = _room_graph(grid, rooms, lookup, h_lines, v_lines, wall)
    room_colors = _propagate_colors(adjacency, seeds)

    for idx, (r0, r1, c0, c1, _, _) in enumerate(rooms):
        color = room_colors.get(idx)
        if color is None:
            continue
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = color

    for left, right, cells in doors:
        lc = room_colors.get(left)
        rc = room_colors.get(right)
        if lc is None and rc is None:
            continue
        color = rc if lc is None else lc if rc is None else _stronger_color(lc, rc)
        for r, c in cells:
            if out[r, c] == bg:
                out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    heights = [random.randint(4, 6) for _ in range(room_rows)]
    widths = [random.randint(4, 6) for _ in range(room_cols)]

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    grid = np.full((h, w), BG, dtype=int)

    h_lines = [0]
    pos = 0
    for height in heights:
        pos += height + 1
        h_lines.append(pos)

    v_lines = [0]
    pos = 0
    for width in widths:
        pos += width + 1
        v_lines.append(pos)

    grid[h_lines, :] = WALL
    grid[:, v_lines] = WALL

    for ri in range(1, room_rows):
        wall_row = h_lines[ri]
        for ci in range(room_cols):
            c0, c1 = v_lines[ci] + 1, v_lines[ci + 1] - 1
            grid[wall_row, random.randint(c0, c1)] = BG

    for ci in range(1, room_cols):
        wall_col = v_lines[ci]
        for ri in range(room_rows):
            r0, r1 = h_lines[ri] + 1, h_lines[ri + 1] - 1
            grid[random.randint(r0, r1), wall_col] = BG

    rooms = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols)]
    max_seeds = min(len(CHAIN) - 1, len(rooms) - 1)
    if max_seeds <= 1:
        seed_count = 1
    else:
        seed_count = random.randint(1, min(3, max_seeds))
    seed_rooms = random.sample(rooms, seed_count)
    seed_colors = random.sample(CHAIN[:-1], seed_count)

    for (ri, ci), color in zip(seed_rooms, seed_colors):
        r0, r1 = h_lines[ri] + 1, h_lines[ri + 1] - 1
        c0, c1 = v_lines[ci] + 1, v_lines[ci + 1] - 1
        grid[random.randint(r0, r1), random.randint(c0, c1)] = color

    return grid
