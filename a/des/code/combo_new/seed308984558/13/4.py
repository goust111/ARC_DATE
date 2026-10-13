from __future__ import annotations
import random
import numpy as np
from arc_utils import find_components

BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color segment pair with a straight link, then thicken the shortest completed paths.

    Input structure:
    - Black background with exactly two disjoint segments of each color.
    - A color's segments share a row or a column and have a clear connection corridor.

    Rule:
    - Draw the missing straight connection between each color's two segments.
    - Count total pixels per completed path.
    - For the shortest path or tied shortest paths, expand each path pixel to a 3x3 same-color square, filling only background cells.
    - Leave longer paths one-pixel wide.

    Key insight:
    - Every color is first completed to a full path; only then can path lengths be compared.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    by = {}
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        by.setdefault(int(comp['color']), []).append(comp)
    lengths = {}
    for color, pair in by.items():
        if len(pair) < 2:
            continue
        pair = pair[:2]
        a,b = pair
        ar0,ac0,ar1,ac1 = map(int, a['bbox']); br0,bc0,br1,bc1 = map(int, b['bbox'])
        if ar0 == br0:
            out[ar0, min(ac0,bc0):max(ac1,bc1)+1] = color
        else:
            out[min(ar0,br0):max(ar1,br1)+1, ac0] = color
        lengths[color] = int(np.sum(out == color))
    shortest = min(lengths.values())
    grown = out.copy()
    for color, length in lengths.items():
        if length != shortest: continue
        for r,c in np.argwhere(out == color):
            r0,r1 = max(0,r-1), min(out.shape[0], r+2)
            c0,c1 = max(0,c-1), min(out.shape[1], c+2)
            mask = grown[r0:r1,c0:c1] == BG
            grown[r0:r1,c0:c1][mask] = color
    return grown


def generate_input() -> np.ndarray:
    g = np.zeros((18,24), dtype=int)
    for color in random.sample([1,2,3,4,6,7,8,9], random.randint(2,4)):
        if random.choice([True, False]):
            r = random.randint(2,15); c1 = random.randint(1,8); c2 = random.randint(12,18); l = random.randint(2,4)
            g[r, c1:c1+l] = color; g[r, c2:c2+l] = color
        else:
            c = random.randint(2,21); r1 = random.randint(1,6); r2 = random.randint(10,14); l = random.randint(2,4)
            g[r1:r1+l, c] = color; g[r2:r2+l, c] = color
    return g
