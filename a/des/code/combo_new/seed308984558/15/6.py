import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse the color order of tightly nested concentric rectangles."""
    g = np.array(grid, dtype=int)
    out = np.zeros_like(g)
    layers = []
    for color in sorted(set(map(int, g.ravel())) - {0}):
        rs, cs = np.where(g == color)
        layers.append((len(rs), color, int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())))
    layers.sort(reverse=True)
    rev = [layer[1] for layer in layers[::-1]]
    for new_color, (_, _, r0, r1, c0, c1) in zip(rev, layers):
        out[r0 : r1 + 1, c0 : c1 + 1] = new_color
    return out


def generate_input() -> np.ndarray:
    n = random.choice([13, 15, 17, 19])
    g = np.zeros((n, n), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(3, 5))
    for i, color in enumerate(colors):
        r0 = c0 = i
        r1 = c1 = n - 1 - i
        g[r0 : r1 + 1, c0 : c1 + 1] = color
    return g
