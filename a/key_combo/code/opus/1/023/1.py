import random
from collections import deque

import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    out = []
    for r, c in zip(*np.where(grid == color)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    walls = sorted(_components(grid, 5), key=lambda comp: min(c for _, c in comp))
    if len(walls) < 2:
        return out
    left_wall, right_wall = walls[0], walls[-1]
    lr0, lr1 = min(r for r, _ in left_wall), max(r for r, _ in left_wall)
    lc0, lc1 = min(c for _, c in left_wall), max(c for _, c in left_wall)
    rr0, rr1 = min(r for r, _ in right_wall), max(r for r, _ in right_wall)
    rc0, rc1 = min(c for _, c in right_wall), max(c for _, c in right_wall)

    seed_cells = [(r, c, int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 3) & (grid != 5)))]
    if len(seed_cells) < 2:
        return out
    seed_cells.sort(key=lambda x: x[1])
    left_color, right_color = seed_cells[0][2], seed_cells[-1][2]

    for r in range(lr0 + 1, lr1):
        for c in range(lc0 + 1, lc1 + 1):
            if out[r, c] == 0:
                out[r, c] = left_color
    for r in range(rr0 + 1, rr1):
        for c in range(rc0, rc1):
            if out[r, c] == 0:
                out[r, c] = right_color

    gap0, gap1 = lc1 + 1, rc0 - 1
    row0, row1 = max(lr0 + 1, rr0 + 1), min(lr1 - 1, rr1 - 1)
    if gap0 > gap1 or row0 > row1:
        return out

    gap_len = gap1 - gap0 + 1
    left_limit = gap0 + gap_len // 2 - 1
    right_start = gap0 + (gap_len + 1) // 2
    mid_col = gap0 + gap_len // 2 if gap_len % 2 else None

    for r in range(row0, row1 + 1):
        obstacles = sorted(c for c in range(gap0, gap1 + 1) if grid[r, c] == 3)
        for c in range(gap0, gap1 + 1):
            if grid[r, c] == 3:
                continue
            blue_allowed = not any(o < c for o in obstacles)
            red_allowed = not any(o > c for o in obstacles)
            if blue_allowed and not red_allowed:
                out[r, c] = left_color
            elif red_allowed and not blue_allowed:
                out[r, c] = right_color
            elif not blue_allowed and not red_allowed:
                out[r, c] = 0
            else:
                if mid_col is not None and c == mid_col:
                    out[r, c] = 0
                elif c <= left_limit:
                    out[r, c] = left_color
                elif c >= right_start:
                    out[r, c] = right_color
    return out


def _draw_left_c(grid, r0, c0, height, width):
    grid[r0, c0:c0 + width] = 5
    grid[r0 + height - 1, c0:c0 + width] = 5
    grid[r0:r0 + height, c0] = 5


def _draw_right_c(grid, r0, c0, height, width):
    grid[r0, c0:c0 + width] = 5
    grid[r0 + height - 1, c0:c0 + width] = 5
    grid[r0:r0 + height, c0 + width - 1] = 5


def generate_input() -> np.ndarray:
    h = random.randint(16, 20)
    w = random.randint(18, 23)
    grid = np.zeros((h, w), dtype=int)
    height = random.randint(5, 8)
    width = random.randint(3, 4)
    r0 = random.randint(2, h - height - 2)
    gap = random.randint(4, min(8, w - 2 * width - 4))
    lc0 = random.randint(1, max(1, w - (2 * width + gap) - 1))
    rc0 = lc0 + width + gap
    _draw_left_c(grid, r0, lc0, height, width)
    _draw_right_c(grid, r0, rc0, height, width)
    grid[r0 + height // 2, lc0 + 1] = 1
    grid[r0 + height // 2, rc0 + width - 2] = 2
    gap_cols = list(range(lc0 + width, rc0))
    rows = list(range(r0 + 1, r0 + height - 1))
    random.shuffle(rows)
    for r in rows[: random.randint(1, 2)]:
        c = random.choice(gap_cols[1:-1] if len(gap_cols) > 2 else gap_cols)
        grid[r, c] = 3
        if random.random() < 0.35 and c + 1 in gap_cols:
            grid[r, c + 1] = 3
    return grid
