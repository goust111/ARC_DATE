import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Separate nested visible rectangles into full single-layer panels ordered by the right legend."""
    g = np.array(grid, dtype=int)
    order = [int(v) for v in g[:, -1] if int(v) != 0]
    core = g[:, :-1]
    panels = []
    for color in order:
        rs, cs = np.where(core == color)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        panel = np.zeros_like(core)
        panel[r0 : r1 + 1, c0 : c1 + 1] = color
        panels.append(panel)
    return np.hstack(panels)


def generate_input() -> np.ndarray:
    n = random.randint(7, 10)
    g = np.zeros((n, n + 1), dtype=int)
    max_layers = 3 if n >= 9 else 2
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, max_layers))
    bounds = []
    r0, c0, r1, c1 = 1, 1, n - 2, n - 3
    for color in colors:
        if r0 > r1 or c0 > c1:
            break
        bounds.append((r0, c0, r1, c1, color))
        g[r0 : r1 + 1, c0 : c1 + 1] = color
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    for i, color in enumerate(colors[: len(bounds)]):
        g[i, -1] = color
    return g
