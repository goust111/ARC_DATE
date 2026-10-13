import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    seed = np.asarray(grid)
    n, m = seed.shape
    out = np.zeros((n * n, m * m), dtype=int)
    for r in range(n):
        for c in range(m):
            block = out[r * n:(r + 1) * n, c * m:(c + 1) * m]
            if seed[r, c] == 2:
                block[:, :] = seed
            elif seed[r, c] == 5:
                block[:, :] = 5
    for comp in _components(out == 0):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        area = len(comp)
        rect_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        color = 3 if area == rect_area else 4
        for r, c in comp:
            out[r, c] = color
    return out


def _red_shape():
    shapes = [
        [(1, 1), (1, 2), (2, 1), (3, 1), (3, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (3, 3)],
        [(1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3)],
        [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0)],
    ]
    return random.choice(shapes)


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = np.zeros((5, 5), dtype=int)
        for r, c in _red_shape():
            grid[r, c] = 2
        candidates = [(r, c) for r in range(5) for c in range(5) if grid[r, c] == 0]
        random.shuffle(candidates)
        placed = 0
        for r, c in candidates:
            if all(grid[r + dr, c + dc] != 2 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < 5 and 0 <= c + dc < 5):
                grid[r, c] = 5
                placed += 1
                if placed == random.randint(1, 3):
                    break
        out = main(grid)
        if 3 in out and 4 in out:
            return grid
    return grid
