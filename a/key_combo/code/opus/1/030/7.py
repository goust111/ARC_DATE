import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    out = g.copy()
    added = {c: set() for c in range(w)}
    for r, c in zip(*np.where(g == 4)):
        d_top = r
        d_bot = h - 1 - r
        if d_top <= d_bot:
            out[r - 1, c] = 2
            added[c].add(2)
        if d_bot <= d_top:
            out[r + 1, c] = 1
            added[c].add(1)
    for c in range(w):
        if len(added[c]) > 1:
            out[1:h - 1, c] = 0
    out[0, :] = 2
    out[h - 1, :] = 1
    return out


def generate_input() -> np.ndarray:
    h = random.choice([13, 15, 17])
    w = random.choice([9, 11, 13])
    g = np.zeros((h, w), dtype=int)
    g[0, :] = 2
    g[h - 1, :] = 1
    mid = h // 2

    cols = list(range(w))
    random.shuffle(cols)
    up_cols = cols[:2]
    down_cols = cols[2:4]
    mixed_cols = cols[4:6]
    rest = cols[6:]

    for c in up_cols:
        for r in random.sample(range(1, mid), random.randint(1, 3)):
            g[r, c] = 4
    for c in down_cols:
        for r in random.sample(range(mid + 1, h - 1), random.randint(1, 3)):
            g[r, c] = 4
    for c in mixed_cols:
        g[random.choice(range(1, mid)), c] = 4
        g[random.choice(range(mid + 1, h - 1)), c] = 4
        if random.random() < 0.5:
            g[random.choice(range(1, h - 1)), c] = 4

    target = random.randint(8, 15)
    current = int(np.sum(g == 4))
    for c in rest:
        if current >= target:
            break
        choices = [r for r in range(1, h - 1) if r != mid and g[r, c] == 0]
        for r in random.sample(choices, min(len(choices), random.randint(0, 2), target - current)):
            g[r, c] = 4
            current += 1
    return g
