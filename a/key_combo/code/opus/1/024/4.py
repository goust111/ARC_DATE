import random
import numpy as np


SYMMETRIC = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]]),
]
ASYMMETRIC = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]]),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]]),
    np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]]),
]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    items = []
    for color in [int(c) for c in np.unique(grid) if c != 0]:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0)
        if r2 - r1 > 2 or c2 - c1 > 2:
            continue
        patch = np.zeros((3, 3), dtype=int)
        patch[:r2 - r1 + 1, :c2 - c1 + 1] = grid[r1:r2 + 1, c1:c2 + 1]
        if not np.array_equal(patch, patch.T):
            items.append((c1, r1, color))
    items.sort()
    out = np.zeros((5, 3), dtype=int)
    for i, (_, _, color) in enumerate(items[:5]):
        out[4 - i, :] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6], random.randint(5, 6))
    non_count = random.randint(2, min(4, len(colors) - 1))
    flags = [False] * non_count + [True] * (len(colors) - non_count)
    random.shuffle(flags)
    positions = []
    used = set()
    for color, sym in zip(colors, flags):
        for _ in range(300):
            r = random.randrange(0, 13)
            c = random.randrange(0, 13)
            box = {(rr, cc) for rr in range(r - 1, r + 4) for cc in range(c - 1, c + 4)}
            if not (box & used) and c not in [p[1] for p in positions]:
                break
        positions.append((r, c))
        used |= box
        pat = random.choice(SYMMETRIC if sym else ASYMMETRIC)
        for rr in range(3):
            for cc in range(3):
                if pat[rr, cc]:
                    g[r + rr, c + cc] = color
    return g
