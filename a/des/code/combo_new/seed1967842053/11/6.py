from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

FILL_COLORS = (RED, GREEN, YELLOW, MAGENTA, ORANGE)


def _find_doors(grid):
    h, w = grid.shape
    wall = grid == BLUE
    vertical = np.zeros((h, w), dtype=bool)
    horizontal = np.zeros((h, w), dtype=bool)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            up = wall[r - 1, c]
            down = wall[r + 1, c]
            left = wall[r, c - 1]
            right = wall[r, c + 1]
            if up and down and not left and not right:
                vertical[r, c] = True
            if left and right and not up and not down:
                horizontal[r, c] = True

    return vertical, horizontal


def _component_labels(open_mask):
    h, w = open_mask.shape
    labels = np.full((h, w), -1, dtype=int)
    cells_by_label = []

    for sr in range(h):
        for sc in range(w):
            if not open_mask[sr, sc] or labels[sr, sc] != -1:
                continue

            label = len(cells_by_label)
            q = deque([(sr, sc)])
            labels[sr, sc] = label
            cells = []

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and open_mask[nr, nc]
                        and labels[nr, nc] == -1
                    ):
                        labels[nr, nc] = label
                        q.append((nr, nc))

            cells_by_label.append(cells)

    return labels, cells_by_label


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each walled room by BFS distance from the red-marked start room.

    Input structure:
    The grid has black background, one-pixel blue wall segments forming
    rectangular rooms, one black one-pixel door gap in each shared internal
    wall, solid outer blue walls, and exactly one red pixel inside the start
    room.

    Rule:
    Treat each door gap as a passage between the two rooms on opposite sides of
    that gap. Starting with the room containing the red marker, run breadth
    first search over the room graph. Fill all non-wall, non-door pixels of each
    visited room with red, green, yellow, magenta, then orange by BFS depth,
    cycling that color sequence for deeper layers. Blue walls and black door
    gaps are left unchanged, and the original red marker is absorbed into the
    red fill of its room.

    Key insight:
    Door gaps must first be recognized as graph edges, not fillable floor; once
    the room adjacency graph is reconstructed, the output is just graph-depth
    coloring of the rectangular room components.
    """
    grid = np.asarray(grid)
    out = grid.copy()

    vertical_doors, horizontal_doors = _find_doors(grid)
    door_mask = vertical_doors | horizontal_doors
    open_mask = (grid != BLUE) & ~door_mask
    labels, rooms = _component_labels(open_mask)

    starts = np.argwhere(grid == RED)
    if len(starts) == 0:
        return out

    sr, sc = starts[0]
    start_label = labels[sr, sc]
    if start_label < 0:
        return out

    adjacency = [set() for _ in rooms]

    for r, c in np.argwhere(vertical_doors):
        left = labels[r, c - 1]
        right = labels[r, c + 1]
        if left >= 0 and right >= 0 and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    for r, c in np.argwhere(horizontal_doors):
        up = labels[r - 1, c]
        down = labels[r + 1, c]
        if up >= 0 and down >= 0 and up != down:
            adjacency[up].add(down)
            adjacency[down].add(up)

    depths = [-1] * len(rooms)
    depths[start_label] = 0
    q = deque([start_label])
    while q:
        room = q.popleft()
        for nxt in sorted(adjacency[room]):
            if depths[nxt] == -1:
                depths[nxt] = depths[room] + 1
                q.append(nxt)

    for label, depth in enumerate(depths):
        if depth < 0:
            continue
        color = FILL_COLORS[depth % len(FILL_COLORS)]
        for r, c in rooms[label]:
            out[r, c] = color

    return out


def _random_partition_count():
    if random.random() < 0.35:
        return random.randint(4, 5)
    return random.randint(2, 5)


def _make_room_sizes(count):
    return [random.randint(2, 5) for _ in range(count)]


def _sample_layout():
    while True:
        room_rows = _random_partition_count()
        room_cols = _random_partition_count()
        heights = _make_room_sizes(room_rows)
        widths = _make_room_sizes(room_cols)
        h = sum(heights) + room_rows + 1
        w = sum(widths) + room_cols + 1
        if h <= 30 and w <= 30:
            return heights, widths


def _ranges_from_sizes(sizes):
    ranges = []
    walls = [0]
    pos = 1
    for size in sizes:
        start = pos
        end = start + size
        ranges.append((start, end))
        walls.append(end)
        pos = end + 1
    return ranges, walls


def generate_input() -> np.ndarray:
    heights, widths = _sample_layout()
    row_ranges, wall_rows = _ranges_from_sizes(heights)
    col_ranges, wall_cols = _ranges_from_sizes(widths)

    h = wall_rows[-1] + 1
    w = wall_cols[-1] + 1
    grid = np.zeros((h, w), dtype=int)

    for r in wall_rows:
        grid[r, :] = BLUE
    for c in wall_cols:
        grid[:, c] = BLUE

    for i in range(len(row_ranges) - 1):
        wall_r = row_ranges[i][1]
        for c0, c1 in col_ranges:
            grid[wall_r, random.randint(c0, c1 - 1)] = BLACK

    for j in range(len(col_ranges) - 1):
        wall_c = col_ranges[j][1]
        for r0, r1 in row_ranges:
            grid[random.randint(r0, r1 - 1), wall_c] = BLACK

    rr = random.randrange(len(row_ranges))
    cc = random.randrange(len(col_ranges))
    r0, r1 = row_ranges[rr]
    c0, c1 = col_ranges[cc]
    grid[random.randint(r0, r1 - 1), random.randint(c0, c1 - 1)] = RED

    return grid
