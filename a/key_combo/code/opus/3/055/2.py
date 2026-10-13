import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the framed interior, rotate it from the magenta corner marker, and compress colors by top row."""
    border = np.argwhere((grid == 5) | (grid == 6))
    r0, c0 = border.min(axis=0)
    r1, c1 = border.max(axis=0)
    mr, mc = np.argwhere(grid == 6)[0]
    if (mr, mc) == (r0, c0):
        k = 0
    elif (mr, mc) == (r0, c1):
        k = 3
    elif (mr, mc) == (r1, c1):
        k = 2
    else:
        k = 1
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    rotated = np.rot90(interior, k=k)
    items = []
    for color in sorted(int(x) for x in np.unique(rotated) if x != 0):
        rows = np.where(rotated == color)[0]
        items.append((int(rows.min()), color))
    items.sort()
    out = np.zeros((5, 3), dtype=int)
    for i, (_, color) in enumerate(items):
        out[4 - i, :] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        grid = np.zeros((14, 14), dtype=int)
        r0, c0, side = 2, 2, 9
        r1, c1 = r0 + side - 1, c0 + side - 1
        grid[r0:r1 + 1, c0] = 5
        grid[r0:r1 + 1, c1] = 5
        grid[r0, c0:c1 + 1] = 5
        grid[r1, c0:c1 + 1] = 5
        corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        grid[random.choice(corners)] = 6
        colors = random.sample([1, 2, 3, 7], random.randint(3, 4))
        inner = np.zeros((side - 2, side - 2), dtype=int)
        occupied = np.zeros_like(inner, dtype=bool)
        ok = True
        for color in colors:
            for _ in range(100):
                length = random.randint(2, 5)
                horiz = random.choice([True, False])
                if horiz:
                    r = random.randrange(inner.shape[0])
                    c = random.randrange(inner.shape[1] - length + 1)
                    cells = [(r, c + i) for i in range(length)]
                else:
                    r = random.randrange(inner.shape[0] - length + 1)
                    c = random.randrange(inner.shape[1])
                    cells = [(r + i, c) for i in range(length)]
                if all(not occupied[p] for p in cells):
                    for p in cells:
                        occupied[p] = True
                        inner[p] = color
                    break
            else:
                ok = False
        if not ok:
            continue
        unique = True
        for k in range(4):
            rows = []
            rot = np.rot90(inner, k=k)
            for color in colors:
                rows.append(int(np.where(rot == color)[0].min()))
            unique = unique and len(rows) == len(set(rows))
        if unique:
            grid[r0 + 1:r1, c0 + 1:c1] = inner
            return grid
