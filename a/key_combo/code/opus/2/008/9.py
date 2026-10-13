import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sources = []
    for br in range(5):
        for bc in range(5):
            if g[br * 3 + 1, bc * 3 + 1] == 5:
                sources.append((br, bc))
    counts = np.zeros((5, 5), dtype=int)
    for sr, sc in sources:
        for i in range(5):
            counts[sr, i] += 1
            if i != sr:
                counts[i, sc] += 1
    out = np.zeros((15, 15), dtype=int)
    for br in range(5):
        for bc in range(5):
            if counts[br, bc] == 1:
                out[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3] = 1
            elif counts[br, bc] >= 2:
                out[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3] = 2
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    n = random.randint(2, 3)
    while True:
        sources = random.sample([(r, c) for r in range(5) for c in range(5)], n)
        if any(r1 != r2 and c1 != c2 for i, (r1, c1) in enumerate(sources) for r2, c2 in sources[i + 1:]):
            break
    for br, bc in sources:
        g[br * 3 + 1, bc * 3 + 1] = 5
    for _ in range(random.randint(4, 10)):
        br, bc = random.randrange(5), random.randrange(5)
        pos = random.choice([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)])
        rr, cc = br * 3 + pos[0], bc * 3 + pos[1]
        if g[rr, cc] == 0:
            g[rr, cc] = 7
    return g
