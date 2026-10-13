import random
import numpy as np
from collections import Counter


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow axis-aligned rays from each seed and resolve color conflicts in favor of the rarer seed color.

    Input structure:
    - Colored seed pixels sit on a black background.
    - Grey wall segments block growth.

    Rule:
    - Cast rays up, down, left, and right from every seed through black cells until a
      wall, another seed, or the boundary is reached.
    - If multiple rays cover a cell, keep the common color if they agree; otherwise use
      the globally rarer seed color among the contenders.
    - Preserve all walls and seed pixels.

    Key insight:
    - The local crossing rule depends on a global statistic: seed-color frequency across
      the whole input, not on traversal order.
    """
    out = grid.copy()
    seeds = [(r, c, int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != GREY))]
    freq = Counter(color for _, _, color in seeds)
    cover = {}
    for r, c, color in seeds:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BG:
                cover.setdefault((nr, nc), []).append(color)
                nr += dr
                nc += dc
    for pos, colors in cover.items():
        uniq = set(colors)
        if len(uniq) == 1:
            out[pos] = colors[0]
        else:
            out[pos] = min(uniq, key=lambda c: (freq[c], c))
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    for _ in range(random.randint(3, 5)):
        if random.random() < 0.5:
            r = random.randint(2, 21)
            c0 = random.randint(2, 15)
            l = random.randint(3, 6)
            grid[r, c0 : c0 + l] = GREY
        else:
            c = random.randint(2, 21)
            r0 = random.randint(2, 15)
            l = random.randint(3, 6)
            grid[r0 : r0 + l, c] = GREY
    colors = [1, 2, 3, 4, 6]
    for _ in range(random.randint(4, 7)):
        for _ in range(200):
            r = random.randint(1, 22)
            c = random.randint(1, 22)
            if grid[r, c] == BG and all(grid[r + dr, c + dc] == BG for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]):
                grid[r, c] = random.choice(colors)
                break
    return grid
