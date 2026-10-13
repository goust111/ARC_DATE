import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    frames = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r:r + 3, c:c + 3]
            if block[1, 1] == 0 and np.all(block[[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]] == 1):
                frames.append((r, c))
    active = set()
    for r, c in frames:
        for rr, cc in ((0, 0), (0, 2), (2, 0), (2, 2)):
            out[r + rr, c + cc] = 0
        for rr, cc in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
            out[r + rr, c + cc] = 2
            active.add((r + rr, c + cc))

    generation = 1
    current = active
    while current:
        by_row = {}
        for r, c in current:
            by_row.setdefault(r, set()).add(c)
        children = set()
        for r, cols in by_row.items():
            if r + 2 >= h:
                continue
            cols = sorted(cols)
            for c1, c2 in zip(cols, cols[1:]):
                if c2 == c1 + 2 and out[r + 2, c1 + 1] == 0:
                    children.add((r + 2, c1 + 1))
        color = 7 if generation % 2 == 1 else 6
        for r, c in children:
            out[r, c] = color
        current = children
        generation += 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(15, 18)
    w = random.randint(13, 16)
    g = np.zeros((h, w), dtype=int)
    row = random.randint(1, 3)
    cols = random.sample(range(1, w - 3, 4), random.randint(2, 3))
    potential = []
    for c in cols:
        g[row, c:c + 3] = 1
        g[row + 2, c:c + 3] = 1
        g[row + 1, c] = 1
        g[row + 1, c + 2] = 1
        potential.append((row + 3, c + 1))
    if potential:
        r, c = random.choice(potential)
        if r < h:
            g[r, c] = 4
    for _ in range(random.randint(3, 7)):
        for _attempt in range(100):
            r = random.randint(row + 4, h - 2)
            c = random.randint(1, w - 2)
            if g[r, c] == 0 and all(g[max(0, r - 1):r + 2, max(0, c - 1):c + 2].flatten() != 4):
                g[r, c] = 4
                break
    return g
