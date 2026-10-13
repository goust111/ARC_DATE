import random

import numpy as np


BG = 0
TRACK = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the first H track cells from each seed, where H is the number of background holes in that track.

    Input structure:
    - A black background contains straight grey tracks with background gaps along the same row or column.
    - One colored seed sits just outside one endpoint of each track.
    - Tracks are otherwise isolated.

    Rule:
    - Count the number of background gap runs along each track axis.
    - Starting from the seed and moving onto the track, recolor the first H grey cells with the seed color, skipping the gap cells themselves.
    - Leave tracks with zero holes unchanged.

    Key insight:
    - The holes are not blockers to fill; they are the counter that decides how far the seed color propagates along the track.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] in (BG, TRACK):
                continue
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == TRACK:
                    cells = []
                    x, y = rr, cc
                    while 0 <= x < g.shape[0] and 0 <= y < g.shape[1] and g[x, y] in (TRACK, BG):
                        cells.append((x, y))
                        x += dr
                        y += dc
                    holes = 0
                    prev = TRACK
                    for x, y in cells:
                        cur = g[x, y]
                        if cur == BG and prev == TRACK:
                            holes += 1
                        prev = cur
                    painted = 0
                    for x, y in cells:
                        if g[x, y] == TRACK and painted < holes:
                            out[x, y] = g[r, c]
                            painted += 1
                    break
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(16, 24), random.randint(16, 24)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4)):
        if random.choice([True, False]):
            r = random.randint(1, h - 2)
            c0 = random.randint(2, w - 8)
            length = random.randint(5, 9)
            g[r, c0 : c0 + length] = TRACK
            for _ in range(random.randint(0, 2)):
                g[r, random.randint(c0 + 1, c0 + length - 2)] = BG
            g[r, c0 - 1] = color
        else:
            c = random.randint(1, w - 2)
            r0 = random.randint(2, h - 8)
            length = random.randint(5, 9)
            g[r0 : r0 + length, c] = TRACK
            for _ in range(random.randint(0, 2)):
                g[random.randint(r0 + 1, r0 + length - 2), c] = BG
            g[r0 - 1, c] = color
    return g
