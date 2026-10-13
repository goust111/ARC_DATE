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
    grid = np.asarray(grid)
    h, w = grid.shape
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, [1, 2, 3]))]
    valid = [(r, c, color) for r, c, color in markers if np.any(grid[:r, c] == 5)]
    rays = [[set() for _ in range(w)] for _ in range(h)]
    for r, c, color in valid:
        for rr in range(r + 1, h):
            rays[rr][c].add(color)
    out = grid.copy()
    collision = np.zeros((h, w), bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                continue
            if len(rays[r][c]) == 1:
                out[r, c] = next(iter(rays[r][c]))
            elif len(rays[r][c]) > 1:
                out[r, c] = 4
                collision[r, c] = True
    for comp in _components(collision):
        area = len(comp)
        color = 7 if area == 1 else 6 if area <= 3 else 9
        for r, c in comp:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(15, 18)
    grid = np.zeros((h, w), dtype=int)
    top_h = random.randint(4, 6)
    covered_cols = list(range(3, w - 3))
    for c in covered_cols:
        bottom = random.randint(1, top_h - 1)
        for r in range(bottom + 1):
            if abs(c - w // 2) <= top_h - r + random.randint(0, 1):
                grid[r, c] = 5
        grid[bottom, c] = 5
    colors = [1, 2, 3]
    cols = random.sample(covered_cols, 3)
    for c in cols[:2]:
        rs = sorted(random.sample(range(top_h + 1, h - 3), 2))
        cs = random.sample(colors, 2)
        grid[rs[0], c] = cs[0]
        grid[rs[1], c] = cs[1]
    c = cols[2]
    grid[random.randint(top_h + 1, h - 3), c] = random.choice(colors)
    for _ in range(random.randint(1, 3)):
        c = random.choice([0, 1, w - 2, w - 1])
        r = random.randint(top_h + 2, h - 2)
        grid[r, c] = random.choice(colors)
    return grid
