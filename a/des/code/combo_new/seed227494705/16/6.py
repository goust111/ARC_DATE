from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _neighbors(r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def main(grid: np.ndarray) -> np.ndarray:
    """Mark the unique open maze route from the yellow start to the yellow end.

    Input structure: Blue walls enclose one-cell channels from the left edge to
    the right edge.  Red and green door pixels sit in channels, and two yellow
    edge pixels are the start and end.
    Rule: Green doors are passable and remain green; red doors and blue walls are
    blocked.  Find the route from start to end through black channel cells and
    green doors, then recolor every black cell on that route orange while keeping
    yellow endpoints, green doors, red doors, and walls unchanged.
    Key insight: Door color controls passability, while vertical connectors are
    the only way to switch lanes around closed red doors.
    """
    h, w = grid.shape
    yellows = list(map(tuple, np.argwhere(grid == YELLOW)))
    start = min(yellows, key=lambda p: p[1])
    end = max(yellows, key=lambda p: p[1])
    passable = {BLACK, GREEN, YELLOW}
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nr, nc in _neighbors(*cur):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in parent and int(grid[nr, nc]) in passable:
                parent[(nr, nc)] = cur
                q.append((nr, nc))
    out = grid.copy()
    cur = end
    while cur is not None:
        r, c = cur
        if int(grid[r, c]) == BLACK:
            out[r, c] = ORANGE
        cur = parent.get(cur)
    return out


def generate_input() -> np.ndarray:
    lanes = random.randint(3, 5)
    h = 2 * lanes + 1
    w = random.randint(15, 26)
    grid = np.full((h, w), BLUE, dtype=int)
    rows = [1 + 2 * i for i in range(lanes)]
    for r in rows:
        grid[r, :] = BLACK
    start_lane = 0
    end_lane = lanes - 1
    connectors = sorted(random.sample(range(3, w - 3), lanes - 1))
    path_cells = set()
    cur_row = rows[start_lane]
    cur_col = 0
    for next_row, col in zip(rows[1:], connectors):
        for c in range(min(cur_col, col), max(cur_col, col) + 1):
            path_cells.add((cur_row, c))
        for r in range(min(cur_row, next_row), max(cur_row, next_row) + 1):
            grid[r, col] = BLACK
            path_cells.add((r, col))
        cur_row, cur_col = next_row, col
    for c in range(cur_col, w):
        path_cells.add((cur_row, c))
    grid[rows[start_lane], 0] = YELLOW
    grid[rows[end_lane], w - 1] = YELLOW
    for r in rows:
        for c in random.sample(range(2, w - 2), random.randint(1, 3)):
            if (r, c) in path_cells and grid[r, c] == BLACK:
                grid[r, c] = GREEN
            elif grid[r, c] == BLACK:
                grid[r, c] = RED
    return grid
