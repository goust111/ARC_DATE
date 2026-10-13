import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the sprite with alternating mirrors, then extend each gray control pixel toward its nearest vertical edge bar."""
    t = np.array(grid, dtype=int)
    middle = np.hstack([t, np.fliplr(t)])
    mirrored = np.flipud(middle)
    out = np.vstack([mirrored, middle, mirrored]).copy()
    controls = [tuple(p) for p in np.argwhere(out == 5)]
    h, _ = out.shape
    for r, c in controls:
        up = None
        for rr in range(r - 1, -1, -1):
            if out[rr, c] in (1, 2):
                up = (r - rr, int(out[rr, c]))
                break
        down = None
        for rr in range(r + 1, h):
            if out[rr, c] in (1, 2):
                down = (rr - r, int(out[rr, c]))
                break
        if up is None and down is None:
            continue
        if down is None or (up is not None and up[0] < down[0]):
            nr, color = r - 1, up[1]
        else:
            nr, color = r + 1, down[1]
        if out[nr, c] == 0:
            out[nr, c] = color
    return out


def generate_input() -> np.ndarray:
    h = random.choice([5, 6])
    w = random.choice([5, 6])
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = 2
    grid[-1, :] = 1
    rows = [2] if h == 5 else [2, 3]
    cols = random.sample(range(1, w - 1), random.choice([2, 3]))
    for i, c in enumerate(cols):
        grid[rows[i % len(rows)], c] = 5
    return grid
