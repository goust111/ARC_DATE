from __future__ import annotations
import random
import numpy as np

BG, GREY = 0, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a KxK rectangle from every seed toward its quadrant corner, turning overlaps of different colors grey.

    Input structure:
    - Even-sized black square grid with colored singleton seeds.
    - Each seed lies in one of the four quadrants.

    Rule:
    - Count total seeds of each color to get K for that color.
    - Grow a KxK solid rectangle from each seed toward its quadrant corner, with the seed at the corner nearest the center.
    - Clip rectangles to the grid edges.
    - Any cell painted by more than one color becomes grey.

    Key insight:
    - Color frequency controls rectangle size, while quadrant controls growth direction.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    h,w = g.shape
    counts = {int(c): int(np.sum(g == c)) for c in np.unique(g) if c != BG}
    cover = np.zeros(g.shape, dtype=int)
    owner = np.zeros(g.shape, dtype=int)
    for r,c in np.argwhere(g != BG):
        color = int(g[r,c]); k = counts[color]
        if r < h//2 and c < w//2: r0,r1,c0,c1 = max(0,r-k+1), r+1, max(0,c-k+1), c+1
        elif r < h//2 and c >= w//2: r0,r1,c0,c1 = max(0,r-k+1), r+1, c, min(w,c+k)
        elif r >= h//2 and c < w//2: r0,r1,c0,c1 = r, min(h,r+k), max(0,c-k+1), c+1
        else: r0,r1,c0,c1 = r, min(h,r+k), c, min(w,c+k)
        cover[r0:r1,c0:c1] += 1
        mask = owner[r0:r1,c0:c1] == 0
        owner[r0:r1,c0:c1][mask] = color
        owner[r0:r1,c0:c1][~mask & (owner[r0:r1,c0:c1] != color)] = GREY
    out[(cover > 0) & (owner != GREY)] = owner[(cover > 0) & (owner != GREY)]
    out[owner == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    n = random.choice([12,14,16,18])
    g = np.zeros((n,n), dtype=int)
    colors = random.sample([1,2,3,4,6,7,8,9], random.randint(2,4))
    for color in colors:
        for _ in range(random.randint(1,3)):
            quad = random.choice([(0,0),(0,1),(1,0),(1,1)])
            rr = random.randint(1, n//2-2) + quad[0]*(n//2)
            cc = random.randint(1, n//2-2) + quad[1]*(n//2)
            g[rr,cc] = color
    return g
