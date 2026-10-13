import random
from collections import defaultdict, deque

import numpy as np


BG = 0


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
    out = np.zeros_like(grid)
    row_arrows = defaultdict(list)
    col_arrows = defaultdict(list)
    used = set()
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            if len(comp) != 3:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if min(cols) == 0:
                row_arrows[sorted(rows)[1]].append("L")
                used |= set(comp)
            elif max(cols) == grid.shape[1] - 1:
                row_arrows[sorted(rows)[1]].append("R")
                used |= set(comp)
            elif min(rows) == 0:
                col_arrows[sorted(cols)[1]].append("T")
                used |= set(comp)
            elif max(rows) == grid.shape[0] - 1:
                col_arrows[sorted(cols)[1]].append("B")
                used |= set(comp)
    for r in range(grid.shape[0]):
        vals = [grid[r, c] for c in range(grid.shape[1]) if grid[r, c] != BG and (r, c) not in used]
        if not vals:
            continue
        arrows = row_arrows.get(r, [])
        if len(arrows) >= 2:
            k = (len(vals) + 1) // 2
            out[r, :k] = vals[:k]
            out[r, -len(vals[k:]) :] = vals[k:]
        elif arrows == ["L"]:
            out[r, -len(vals) :] = vals
        elif arrows == ["R"]:
            out[r, : len(vals)] = vals
        else:
            for c, v in enumerate(vals):
                out[r, c] = v
    for c in range(grid.shape[1]):
        vals = [out[r, c] if out[r, c] != BG else grid[r, c] for r in range(grid.shape[0]) if ((out[r, c] if out[r, c] != BG else grid[r, c]) != BG and (r, c) not in used)]
        arrows = col_arrows.get(c, [])
        if arrows:
            for r in range(grid.shape[0]):
                out[r, c] = BG
            if len(arrows) >= 2:
                k = (len(vals) + 1) // 2
                for i, v in enumerate(vals[:k]):
                    out[i, c] = v
                for i, v in enumerate(vals[k:]):
                    out[grid.shape[0] - len(vals[k:]) + i, c] = v
            elif arrows == ["T"]:
                for i, v in enumerate(vals):
                    out[grid.shape[0] - len(vals) + i, c] = v
            else:
                for i, v in enumerate(vals):
                    out[i, c] = v
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5)):
        r = random.randint(2, h - 3)
        c = random.randint(2, w - 3)
        grid[r, c] = color
    for _ in range(random.randint(2, 4)):
        r = random.randint(2, h - 3)
        grid[r, 0] = 9
        grid[r - 1, 1] = 9
        grid[r + 1, 1] = 9
        if random.random() < 0.3:
            grid[r, w - 1] = 9
            grid[r - 1, w - 2] = 9
            grid[r + 1, w - 2] = 9
    for _ in range(random.randint(1, 3)):
        c = random.randint(2, w - 3)
        grid[0, c] = 8
        grid[1, c - 1] = 8
        grid[1, c + 1] = 8
    return grid
