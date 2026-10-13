import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each path using the palette entry indexed by its gate-crossing count."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    palette = [int(c) for c in g[-1] if int(c) not in (0, 5)]
    gaps = set()
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1]):
            if g[r, c] != 0:
                continue
            if c > 0 and c + 1 < g.shape[1] and g[r, c - 1] == g[r, c + 1] == 5:
                gaps.add((r, c))
            if r > 0 and r + 1 < g.shape[0] and g[r - 1, c] == g[r + 1, c] == 5:
                gaps.add((r, c))
    for color in sorted(set(map(int, g.ravel())) - {0, 5, *palette}):
        cnt = sum((r, c) in gaps for r, c in zip(*np.where(g == color)))
        out[g == color] = palette[max(0, min(cnt - 1, len(palette) - 1))]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    g[-1, 2 : 2 + len(palette)] = palette
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        r = random.randint(2, g.shape[0] - 4)
        c0 = random.randint(1, 4)
        c1 = random.randint(c0 + 6, g.shape[1] - 3)
        g[r, c0:c1] = color
        count = random.randint(1, min(4, len(palette)))
        xs = sorted(random.sample(range(c0 + 1, c1 - 1), count))
        for x in xs:
            g[r, x] = color
            g[r, x - 1] = 5
            g[r, x + 1] = 5
            g[r, x] = color
            if r > 1:
                g[r - 1, x] = 0
    return g
