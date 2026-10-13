import random
import numpy as np


BG = 0
GREY = 5
RED = 2


def _components(passable):
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Fill enclosed grey regions using red seed counts as palette indices.

    Input structure: Grey closed boundaries enclose black interiors containing
    one or more red seed pixels. The bottom row starts with a compact color bar
    that excludes black, grey, and red.
    Rule: Count red seeds in each enclosed region. Use the kth bottom palette
    color for a region with k seeds, filling only black interior cells while
    preserving the red seeds, grey boundaries, and palette.
    Key insight: The ordered bottom color strip is a lookup table indexed by
    seed count; seed positions are distractors.
    """
    out = grid.copy()
    h, w = grid.shape
    palette = [int(grid[h - 1, c]) for c in range(w) if int(grid[h - 1, c]) not in (BG, GREY, RED)]
    passable = (grid != GREY)
    passable[h - 1, :] = False
    for comp in _components(passable):
        if any(r in (0, h - 2) or c in (0, w - 1) for r, c in comp):
            continue
        k = sum(grid[r, c] == RED for r, c in comp)
        if 1 <= k <= len(palette):
            color = palette[k - 1]
            for r, c in comp:
                if out[r, c] == BG:
                    out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 24, 24
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample([1, 3, 4, 6, 7, 8, 9], 6)
    grid[h - 1, :6] = palette
    frames = [(2, 2), (2, 10), (10, 3), (11, 13)]
    for idx, (r0, c0) in enumerate(frames[: random.randint(2, 4)]):
        rh, rw = random.randint(5, 7), random.randint(5, 8)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        k = random.randint(1, 6)
        spots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in random.sample(spots, k):
            grid[r, c] = RED
    return grid
