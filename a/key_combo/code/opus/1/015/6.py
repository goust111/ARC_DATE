import random
from collections import deque
import numpy as np


def _component(grid, color):
    pts = np.argwhere(grid == color)
    return [(int(r), int(c)) for r, c in pts]


def main(grid: np.ndarray) -> np.ndarray:
    block = None
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            vals = grid[r:r + 2, c:c + 2]
            if np.all(vals != 0) and 7 not in vals:
                block = vals
                break
        if block is not None:
            break
    n = len(set(int(v) for v in block.ravel() if v != 0))
    sprite_pts = _component(grid, 7)
    rows = [r for r, _ in sprite_pts]
    cols = [c for _, c in sprite_pts]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    tile = grid[r0:r1 + 1, c0:c1 + 1]
    th, tw = tile.shape
    out = np.zeros((th, tw * n), dtype=int)
    for i in range(n):
        piece = tile if i % 2 == 0 else tile[:, ::-1]
        out[:, i * tw:(i + 1) * tw] = piece
    return out


def generate_input() -> np.ndarray:
    shapes = [
        [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (1, 3)],
    ]
    grid = np.zeros((12, 12), dtype=int)
    shape = random.choice(shapes)
    sr, sc = random.randint(1, 6), random.randint(1, 4)
    for dr, dc in shape:
        grid[sr + dr, sc + dc] = 7
    n = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 8, 9], n)
    vals = colors + random.choices(colors, k=4 - n)
    random.shuffle(vals)
    for _ in range(50):
        br, bc = random.randint(0, 10), random.randint(7, 10)
        if np.all(grid[br:br + 2, bc:bc + 2] == 0):
            grid[br:br + 2, bc:bc + 2] = np.array(vals).reshape(2, 2)
            return grid
    return generate_input()
