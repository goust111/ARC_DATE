from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
GEN_ATTEMPTS = 300

DIRS = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    return sum(0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1] and mask[r + dr, c + dc] for dr, dc in DIRS.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray route segment by tracing from colored endpoints and applying color-controlled turn/straight behavior at colored crossings.

    Input structure:
    - The background is black.
    - Gray one-pixel paths form a network, with colored edge endpoints and colored degree-4 crossings.
    - Non-crossing route pixels remain gray in the input.

    Rule:
    - Start from every colored endpoint on the grid edge with that endpoint's color as the current color.
    - Follow the path forward, painting traversed gray pixels with the current color.
    - At a colored crossing, go straight if the crossing color equals the current color; otherwise turn right and update the current color to the crossing color.
    - Keep all endpoint and crossing pixels in their original colors.

    Key insight:
    - The colored crossing pixels act as dynamic route controls: they do not just label paths, they change both the continuation direction and, after a turn, the color that future segments should use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    path_mask = g != BG
    out = g.copy()

    endpoints = []
    for r in range(h):
        for c in range(w):
            if g[r, c] in (BG, GRAY):
                continue
            deg = _degree(path_mask, r, c)
            if deg == 1 and (r in (0, h - 1) or c in (0, w - 1)):
                endpoints.append((r, c))

    for start in endpoints:
        current_color = int(g[start])
        prev = start
        neighbors = [(start[0] + dr, start[1] + dc, d) for d, (dr, dc) in DIRS.items() if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w and path_mask[start[0] + dr, start[1] + dc]]
        assert len(neighbors) == 1
        r, c, direction = neighbors[0]
        steps = 0
        while steps < h * w * 2:
            steps += 1
            if g[r, c] == GRAY:
                out[r, c] = current_color
                dr, dc = DIRS[direction]
                r, c = r + dr, c + dc
                continue
            deg = _degree(path_mask, r, c)
            if deg == 1:
                break
            if int(g[r, c]) != current_color:
                current_color = int(g[r, c])
                direction = (direction + 1) % 4
            dr, dc = DIRS[direction]
            r, c = r + dr, c + dc

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(15, 25)
        w = random.randint(15, 25)
        grid = np.full((h, w), BG, dtype=int)
        mid_r = random.randint(4, h - 5)
        mid_c = random.randint(4, w - 5)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
        left_c, right_c = 0, w - 1
        top_r, bottom_r = 0, h - 1

        grid[mid_r, left_c:right_c + 1] = GRAY
        grid[top_r:bottom_r + 1, mid_c] = GRAY
        grid[mid_r, left_c] = colors[0]
        grid[mid_r, right_c] = colors[1]
        grid[top_r, mid_c] = colors[2]
        grid[bottom_r, mid_c] = colors[3]
        grid[mid_r, mid_c] = random.choice(colors[:5])
        return grid
    raise RuntimeError("failed to generate a valid instance")
