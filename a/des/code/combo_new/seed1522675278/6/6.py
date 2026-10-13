import random
from collections import deque

import numpy as np


BG = 0
ARROW = 9


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for comp in _components(grid == ARROW):
        if len(comp) < 3:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if min(cols) == 0:
            row = sorted(rows)[1]
            pos = (row, 2)
            step = (0, 1)
        elif max(cols) == grid.shape[1] - 1:
            row = sorted(rows)[1]
            pos = (row, grid.shape[1] - 3)
            step = (0, -1)
        elif min(rows) == 0:
            col = sorted(cols)[1]
            pos = (2, col)
            step = (1, 0)
        else:
            col = sorted(cols)[1]
            pos = (grid.shape[0] - 3, col)
            step = (-1, 0)
        spray = None
        r, c = pos
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if grid[r, c] != BG:
                if spray is None:
                    spray = int(grid[r, c])
                elif grid[r, c] != spray:
                    break
            elif spray is not None:
                out[r, c] = spray
            r += step[0]
            c += step[1]
        for r, c in comp:
            out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    side = random.choice(["l", "r", "t", "b"])
    lines = random.sample(range(2, (h if side in "lr" else w) - 2), random.randint(2, 4))
    for line in lines:
        if side == "l":
            grid[line, 0] = ARROW
            grid[line - 1, 1] = ARROW
            grid[line + 1, 1] = ARROW
            seed = random.choice([1, 2, 3, 4, 6, 7, 8])
            end = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8] if c != seed])
            s = random.randint(3, w // 2)
            e = random.randint(s + 2, w - 2)
            grid[line, s] = seed
            grid[line, e] = end
        elif side == "r":
            grid[line, w - 1] = ARROW
            grid[line - 1, w - 2] = ARROW
            grid[line + 1, w - 2] = ARROW
            seed = random.choice([1, 2, 3, 4, 6, 7, 8])
            end = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8] if c != seed])
            s = random.randint(w // 2, w - 4)
            e = random.randint(1, s - 2)
            grid[line, s] = seed
            grid[line, e] = end
        elif side == "t":
            grid[0, line] = ARROW
            grid[1, line - 1] = ARROW
            grid[1, line + 1] = ARROW
            seed = random.choice([1, 2, 3, 4, 6, 7, 8])
            end = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8] if c != seed])
            s = random.randint(3, h // 2)
            e = random.randint(s + 2, h - 2)
            grid[s, line] = seed
            grid[e, line] = end
        else:
            grid[h - 1, line] = ARROW
            grid[h - 2, line - 1] = ARROW
            grid[h - 2, line + 1] = ARROW
            seed = random.choice([1, 2, 3, 4, 6, 7, 8])
            end = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8] if c != seed])
            s = random.randint(h // 2, h - 4)
            e = random.randint(1, s - 2)
            grid[s, line] = seed
            grid[e, line] = end
    return grid
