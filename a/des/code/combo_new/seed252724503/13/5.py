import random

import numpy as np

BLACK = 0
YELLOW = 4


def _first_pos(coords):
    return min(coords, key=lambda p: (p[0], p[1]))


def _draw_path(out: np.ndarray, a, b, corner, color: int):
    ar, ac = a
    br, bc = b
    cr, cc = corner
    step = 1 if cc >= ac else -1
    for c in range(ac, cc + step, step):
        out[ar, c] = color
    step = 1 if br >= ar else -1
    for r in range(ar, br + step, step):
        out[r, cc] = color
    step = 1 if bc >= cc else -1
    for c in range(cc, bc + step, step):
        out[br, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored point pair through the yellow-selected L-corner.

    Input structure:
    - Black background with several foreground colors, each appearing as exactly two singleton points.
    - Each color pair defines two possible axis-aligned L-path corners.
    - Exactly one of those two corners contains a yellow marker pixel.

    Rule:
    - Pair points by color and find which candidate corner contains yellow.
    - Draw the L-shaped path from the first point to that corner and then to the second point using the pair color.
    - Process pairs in first-appearance order so later paths can overwrite earlier crossings.

    Key insight:
    - The yellow pixel disambiguates which of the two possible L turns is correct for each same-color pair.
    """
    out = grid.copy()
    colors = []
    for color in (int(v) for v in np.unique(grid) if v not in (BLACK, YELLOW)):
        coords = list(zip(*np.where(grid == color)))
        colors.append((_first_pos(coords), color, coords))
    colors.sort()
    for _pos, color, coords in colors:
        a, b = sorted(coords)
        corners = [(a[0], b[1]), (b[0], a[1])]
        corner = corners[0] if grid[corners[0]] == YELLOW else corners[1]
        _draw_path(out, a, b, corner, color)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 5, 6, 7, 8, 9]
    for _ in range(400):
        h = rng.randint(14, 22)
        w = rng.randint(14, 22)
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            for _ in range(100):
                a = (rng.randint(1, h - 2), rng.randint(1, w - 2))
                b = (rng.randint(1, h - 2), rng.randint(1, w - 2))
                if a[0] == b[0] or a[1] == b[1]:
                    continue
                corners = [(a[0], b[1]), (b[0], a[1])]
                if len({a, b, corners[0], corners[1]}) < 4:
                    continue
                if any(p in occupied for p in (a, b, corners[0], corners[1])):
                    continue
                turn = rng.choice(corners)
                grid[a] = color
                grid[b] = color
                grid[turn] = YELLOW
                occupied.update((a, b, corners[0], corners[1]))
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 13/5")
