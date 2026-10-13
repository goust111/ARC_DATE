import random

import numpy as np


def _center(tile, size):
    out = np.zeros((size, size), dtype=int)
    h, w = tile.shape
    r0 = (size - h) // 2
    c0 = (size - w) // 2
    out[r0:r0 + h, c0:c0 + w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    reds = np.argwhere(grid == 2)
    if len(reds) == 0:
        return grid.copy()
    r1, c1 = reds.min(axis=0)
    r2, c2 = reds.max(axis=0)
    marker = np.argwhere(grid == 1)
    if len(marker):
        rm, cm = map(int, marker[0])
        if r1 <= rm <= r2:
            if cm > c2:
                c2 = cm
            elif cm < c1:
                c1 = cm
        elif c1 <= cm <= c2:
            if rm > r2:
                r2 = rm
            elif rm < r1:
                r1 = rm
    H, W = int(r2 - r1 + 1), int(c2 - c1 + 1)
    tile = np.full((H, W), 4, dtype=int)
    tile[0, :] = 2
    tile[-1, :] = 2
    tile[:, 0] = 2
    tile[:, -1] = 2
    rots = [tile, np.rot90(tile, -1), np.rot90(tile, -2), np.rot90(tile, -3)]
    size = max(H, W)
    q = [_center(t, size) for t in rots]
    return np.block([[q[0], q[1]], [q[3], q[2]]])


def generate_input() -> np.ndarray:
    n = random.randint(13, 18)
    grid = np.zeros((n, n), dtype=int)
    rh = random.randint(4, 6)
    rw = random.randint(4, 6)
    r1 = random.randint(3, n - rh - 3)
    c1 = random.randint(3, n - rw - 3)
    r2, c2 = r1 + rh - 1, c1 + rw - 1
    grid[r1:r2 + 1, c1:c2 + 1] = 4
    grid[r1, c1:c2 + 1] = 2
    grid[r2, c1:c2 + 1] = 2
    grid[r1:r2 + 1, c1] = 2
    grid[r1:r2 + 1, c2] = 2
    dirs = []
    if c2 + 2 < n:
        dirs.append(("right", random.randint(r1, r2), random.randint(c2 + 2, n - 1)))
    if c1 - 2 >= 0:
        dirs.append(("left", random.randint(r1, r2), random.randint(0, c1 - 2)))
    if r2 + 2 < n:
        dirs.append(("down", random.randint(r2 + 2, n - 1), random.randint(c1, c2)))
    if r1 - 2 >= 0:
        dirs.append(("up", random.randint(0, r1 - 2), random.randint(c1, c2)))
    _, rm, cm = random.choice(dirs)
    grid[rm, cm] = 1
    return grid
