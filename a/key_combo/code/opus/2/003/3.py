import random
from collections import deque
import numpy as np


def _legend(grid):
    h, w = grid.shape
    mapping = {}
    cells = set()
    for r in range(max(0, h - 6), h):
        for c in range(max(0, w - 6), w - 1):
            a, b = int(grid[r, c]), int(grid[r, c + 1])
            if a not in (0, 5) and b not in (0, 5) and a != b:
                mapping[a] = b
                cells.add((r, c))
                cells.add((r, c + 1))
    return mapping, cells


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    mapping, legend_cells = _legend(grid)
    h, w = grid.shape
    for color, new_color in mapping.items():
        for sr, sc in zip(*np.where(grid == color)):
            if (int(sr), int(sc)) in legend_cells:
                continue
            seen = {(int(sr), int(sc))}
            q = deque([(int(sr), int(sc))])
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen and grid[rr, cc] == 0:
                        seen.add((rr, cc))
                        q.append((rr, cc))
            if len(seen) >= 6:
                for r, c in seen:
                    if (r, c) not in legend_cells:
                        out[r, c] = new_color
            else:
                out[int(sr), int(sc)] = 0
    return out


def generate_input() -> np.ndarray:
    h, w = 16, 18
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = grid[:, 13] = 5
    grid[0, :14] = grid[-1, :14] = 5
    for c in random.sample([3, 6, 9], 2):
        grid[:, c] = 5
    for r in random.sample([4, 8, 11], 2):
        grid[r, :14] = 5
    grid[:, 14] = 5
    seed_colors = random.sample([1, 2, 3, 6], random.randint(3, 4))
    out_colors = random.sample([4, 7, 8, 9], len(seed_colors))
    rooms = []
    seen = np.zeros_like(grid, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, 13):
            if grid[r, c] == 0 and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < 14 and grid[xx, yy] == 0 and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                if 3 <= len(comp) <= 25:
                    rooms.append(comp)
    random.shuffle(rooms)
    for color, room in zip(seed_colors, rooms):
        r, c = random.choice(room)
        grid[r, c] = color
    start = h - len(seed_colors) - 1
    for i, (a, b) in enumerate(zip(seed_colors, out_colors)):
        grid[start + i, 15] = a
        grid[start + i, 16] = b
    return grid
