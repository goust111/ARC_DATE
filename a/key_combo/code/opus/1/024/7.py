import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    res = grid.copy()
    for c in range(w):
        lane = grid[:h - 1, c]
        non_blue = lane[lane != 1]
        base = int(non_blue[0]) if non_blue.size else 0
        if non_blue.size and not np.all(non_blue == base):
            continue
        blues = set(np.where(lane == 1)[0].tolist())
        if blues == {h - 2}:
            res[h - 3, c] = 1
        elif blues == {h - 3, h - 2}:
            res[h - 3, c] = base
            res[h - 2, c] = base
            res[0, c] = 2
            res[1, c] = 2
    return res


def generate_input() -> np.ndarray:
    h, w = 10, 8
    g = np.zeros((h, w), dtype=int)
    g[-1, :] = 2
    cols = list(range(w))
    random.shuffle(cols)
    uniform_marked = cols[:3]
    mixed_marked = cols[3:5]
    rest = cols[5:]
    for i, c in enumerate(uniform_marked):
        base = random.choice([0, 3, 4, 7])
        g[:h - 1, c] = base
        if i == 0:
            g[h - 2, c] = 1
        else:
            g[h - 3:h - 1, c] = 1
    for c in mixed_marked:
        vals = [random.choice([0, 3, 4, 7]) for _ in range(h - 1)]
        vals[0] = 0
        vals[1] = 3 if vals[0] != 3 else 4
        g[:h - 1, c] = vals
        if random.random() < 0.5:
            g[h - 2, c] = 1
        else:
            g[h - 3:h - 1, c] = 1
    for c in rest:
        if random.random() < 0.5:
            g[:h - 1, c] = random.choice([0, 3, 4, 7])
        else:
            g[:h - 1, c] = [random.choice([0, 3, 4, 7]) for _ in range(h - 1)]
    return g
