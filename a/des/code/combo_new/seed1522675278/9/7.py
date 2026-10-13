import random
import numpy as np


BG = 0
YELLOW = 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve yellow path crossings by giving the crossing pixel to the horizontal path.

    Input structure:
    - Colored one-pixel-wide paths run across the grid.
    - Wherever a horizontal path and a vertical path cross, the crossing pixel is yellow.

    Rule:
    - For each yellow crossing, inspect the nearest non-yellow colored pixels on the left
      and right along the same row.
    - Use that horizontal path color for the crossing pixel.
    - Leave all non-yellow path pixels unchanged.

    Key insight:
    - Yellow only marks ambiguous crossings; the hidden winning color is recovered by
      reconstructing which path is horizontal at that location.
    """
    out = grid.copy()
    ys = np.argwhere(grid == YELLOW)
    for r, c in ys:
        lc = c - 1
        while lc >= 0 and grid[r, lc] == YELLOW:
            lc -= 1
        rc = c + 1
        while rc < grid.shape[1] and grid[r, rc] == YELLOW:
            rc += 1
        if lc >= 0 and grid[r, lc] != BG:
            out[r, c] = grid[r, lc]
        elif rc < grid.shape[1] and grid[r, rc] != BG:
            out[r, c] = grid[r, rc]
    return out


def generate_input() -> np.ndarray:
    H = random.randint(15, 24)
    W = random.randint(15, 24)
    grid = np.zeros((H, W), dtype=int)
    pool = [1, 2, 3, 6, 7, 8, 9]
    hcount = random.randint(2, 3)
    hcolors = random.sample(pool, hcount)
    remain = [c for c in pool if c not in hcolors]
    vcount = random.randint(2, min(3, len(remain)))
    vcolors = random.sample(remain, vcount)
    rows = random.sample(range(2, H - 2), len(hcolors))
    cols = random.sample(range(2, W - 2), len(vcolors))
    for r, color in zip(rows, hcolors):
        grid[r, :] = color
    for c, color in zip(cols, vcolors):
        mask = grid[:, c] != BG
        grid[:, c] = color
        grid[mask, c] = YELLOW
    return grid
