import random
from typing import List, Tuple

import numpy as np


def _random_connected_shape(height: int, width: int, area: int) -> List[Tuple[int, int]]:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            continue
        cells.add(random.choice(candidates))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Split the input at the full gray separator row into aligned upper and
        lower panels carrying blue and red foreground shapes.
    Analyze:
        Compute the panelwise background intersection as ray seeds and the
        panelwise foreground union as colored blockers.
    Transform:
        Paint blockers with blue, red, or yellow overlap colors, then extend
        horizontal green rays from every seed until blockers truncate them.
    Return:
        Return the panel-sized working grid after removing the separator row and
        combining the obstacle and ray effects.
    """
    grid = np.asarray(grid, dtype=int)
    sep = next(r for r in range(grid.shape[0]) if np.all(grid[r] == 5))
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    if top.shape != bottom.shape:
        raise ValueError("panels must have the same shape")

    h, w = top.shape
    out = np.zeros((h, w), dtype=int)
    top_fg = top != 0
    bottom_fg = bottom != 0
    both_fg = top_fg & bottom_fg
    only_top = top_fg & ~bottom_fg
    only_bottom = bottom_fg & ~top_fg

    out[only_top] = 1
    out[only_bottom] = 2
    out[both_fg] = 4

    obstacle = top_fg | bottom_fg
    seeds = ~obstacle
    for r, c in np.argwhere(seeds):
        cc = c
        while cc >= 0 and not obstacle[r, cc]:
            out[r, cc] = 3
            cc -= 1
        cc = c + 1
        while cc < w and not obstacle[r, cc]:
            out[r, cc] = 3
            cc += 1
    return out


def generate_input() -> np.ndarray:
    for _ in range(5000):
        panel_h = random.randint(9, 11)
        panel_w = random.randint(9, 11)
        top = np.zeros((panel_h, panel_w), dtype=int)
        bottom = np.zeros((panel_h, panel_w), dtype=int)

        blue = _random_connected_shape(panel_h - 2, panel_w - 2, random.randint(10, 18))
        red = _random_connected_shape(panel_h - 2, panel_w - 2, random.randint(10, 18))
        blue_r = random.randint(0, 1)
        blue_c = random.randint(0, 1)
        red_r = random.randint(0, 1)
        red_c = random.randint(0, 1)
        for r, c in blue:
            top[blue_r + r, blue_c + c] = 1
        for r, c in red:
            bottom[red_r + r, red_c + c] = 2

        bg_intersection = (top == 0) & (bottom == 0)
        if int(bg_intersection.sum()) < 5 or len({r for r, _ in np.argwhere(bg_intersection)}) < 3:
            continue

        sep = np.full((1, panel_w), 5, dtype=int)
        return np.vstack([top, sep, bottom])
    raise RuntimeError("failed to generate a valid task-7 input")
