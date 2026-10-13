from collections import deque
import random

import numpy as np


BACKGROUND = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _wall_lines(grid, axis):
    length = grid.shape[1 - axis]
    counts = np.sum(grid == WALL, axis=1 if axis == 0 else 0)
    cutoff = max(3, int(np.ceil(length * 0.55)))
    return [int(i) for i, count in enumerate(counts) if count >= cutoff]


def _room_color(patch):
    vals = patch[(patch != BACKGROUND) & (patch != WALL)]
    if vals.size == 0:
        return None
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill empty rooms by propagating colors through one-pixel doors.

    Input structure:
    A black background grid is divided by single-pixel grey wall rows and
    columns into rectangular rooms.  A room interior is either empty black or
    uniformly filled with one non-black, non-grey color.  Adjacent rooms may
    have exactly one missing wall pixel on their shared wall segment, forming a
    door while the rest of the wall grid remains grey.

    Rule:
    Treat each rectangle between consecutive wall lines as one room and each
    missing wall pixel between neighboring rooms as an undirected connection.
    Starting from all rooms that already contain a color, breadth-first
    propagate that color through door connections into empty rooms.  Once an
    empty room receives a color, it can pass that same color onward through its
    other doors.  Fill only room interiors; grey walls and black door pixels
    stay in their original locations.

    Key insight:
    The wall gap does not get colored; it identifies the next room in a door
    graph whose colors are inherited from the initially colored source rooms.
    """
    grid = np.asarray(grid)
    h_lines = _wall_lines(grid, 0)
    v_lines = _wall_lines(grid, 1)
    if len(h_lines) < 2 or len(v_lines) < 2:
        return grid.copy()

    row_bands = [(h_lines[i] + 1, h_lines[i + 1]) for i in range(len(h_lines) - 1)]
    col_bands = [(v_lines[i] + 1, v_lines[i + 1]) for i in range(len(v_lines) - 1)]

    rooms = []
    room_at = {}
    for ri, (r0, r1) in enumerate(row_bands):
        if r0 >= r1:
            continue
        for ci, (c0, c1) in enumerate(col_bands):
            if c0 >= c1:
                continue
            room_at[(ri, ci)] = len(rooms)
            rooms.append((r0, r1, c0, c1))

    colors = []
    for r0, r1, c0, c1 in rooms:
        colors.append(_room_color(grid[r0:r1, c0:c1]))

    adj = [set() for _ in rooms]
    for ri, (r0, r1) in enumerate(row_bands):
        if r0 >= r1:
            continue
        for ci in range(len(col_bands) - 1):
            left = room_at.get((ri, ci))
            right = room_at.get((ri, ci + 1))
            if left is None or right is None:
                continue
            wall_col = v_lines[ci + 1]
            if np.any(grid[r0:r1, wall_col] != WALL):
                adj[left].add(right)
                adj[right].add(left)

    for ri in range(len(row_bands) - 1):
        wall_row = h_lines[ri + 1]
        for ci, (c0, c1) in enumerate(col_bands):
            if c0 >= c1:
                continue
            top = room_at.get((ri, ci))
            bottom = room_at.get((ri + 1, ci))
            if top is None or bottom is None:
                continue
            if np.any(grid[wall_row, c0:c1] != WALL):
                adj[top].add(bottom)
                adj[bottom].add(top)

    assigned = list(colors)
    queue = deque(idx for idx, color in enumerate(assigned) if color is not None)
    while queue:
        cur = queue.popleft()
        for nxt in sorted(adj[cur]):
            if assigned[nxt] is None:
                assigned[nxt] = assigned[cur]
                queue.append(nxt)

    out = grid.copy()
    for idx, color in enumerate(assigned):
        if color is None:
            continue
        r0, r1, c0, c1 = rooms[idx]
        out[r0:r1, c0:c1] = color
    return out


def _sizes(count):
    while True:
        vals = [random.randint(2, 5) for _ in range(count)]
        if sum(vals) + count + 1 <= 30:
            return vals


def _line_positions(sizes):
    lines = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        lines.append(pos)
    return lines


def _neighbors(cell, rows, cols):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def generate_input() -> np.ndarray:
    room_rows = random.randint(2, 5)
    room_cols = random.randint(2, 5)
    row_sizes = _sizes(room_rows)
    col_sizes = _sizes(room_cols)
    h_lines = _line_positions(row_sizes)
    v_lines = _line_positions(col_sizes)

    height = h_lines[-1] + 1
    width = v_lines[-1] + 1
    grid = np.zeros((height, width), dtype=int)
    for r in h_lines:
        grid[r, :] = WALL
    for c in v_lines:
        grid[:, c] = WALL

    cells = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    max_roots = min(3, max(1, len(cells) // 2))
    root_count = random.randint(1, max_roots)
    roots = random.sample(cells, root_count)

    assigned = set(roots)
    frontier = []
    for root in roots:
        for nb in _neighbors(root, room_rows, room_cols):
            frontier.append((root, nb))

    edges = []
    while len(assigned) < len(cells):
        frontier = [(parent, child) for parent, child in frontier if child not in assigned]
        parent, child = random.choice(frontier)
        assigned.add(child)
        edges.append((parent, child))
        for nb in _neighbors(child, room_rows, room_cols):
            if nb not in assigned:
                frontier.append((child, nb))

    for (r0, c0), (r1, c1) in edges:
        if r0 == r1:
            wall_col = v_lines[max(c0, c1)]
            door_row = random.randint(h_lines[r0] + 1, h_lines[r0 + 1] - 1)
            grid[door_row, wall_col] = BACKGROUND
        else:
            wall_row = h_lines[max(r0, r1)]
            door_col = random.randint(v_lines[c0] + 1, v_lines[c0 + 1] - 1)
            grid[wall_row, door_col] = BACKGROUND

    colors = random.sample(PALETTE, root_count)
    for (room_r, room_c), color in zip(roots, colors):
        r0 = h_lines[room_r] + 1
        r1 = h_lines[room_r + 1]
        c0 = v_lines[room_c] + 1
        c1 = v_lines[room_c + 1]
        grid[r0:r1, c0:c1] = color

    return grid
