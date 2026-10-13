import random

import numpy as np


BLACK = 0
GREY = 5
ANCHOR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color anchors and eject grey obstacle rays perpendicular to each link.

    Input structure:
    - Every non-grey non-black color appears exactly twice as two anchor pixels.
    - Each anchor pair is aligned either horizontally or vertically.
    - Grey obstacle pixels may lie on the straight segment between aligned anchors.

    Rule:
    - Draw the straight colored segment between each same-color anchor pair, leaving
      obstacle cells grey.
    - For every obstacle on that segment, emit grey rays in the two directions
      perpendicular to the segment until the grid boundary or until just before a
      colored line cell.
    - Keep anchors and existing obstacles unchanged.

    Key insight:
    - Obstacle behavior is inherited from its hosting connection: the link orientation
      determines a 90-degree-rotated ray direction, so pairing anchors correctly is the
      prerequisite to understanding the obstacle effect.
    """
    out = grid.copy()
    pairs = []
    for color in [int(v) for v in np.unique(grid) if v not in (BLACK, GREY)]:
        pts = np.argwhere(grid == color)
        if len(pts) == 2:
            pairs.append((color, tuple(pts[0]), tuple(pts[1])))
    for color, a, b in pairs:
        if a[0] == b[0]:
            r = a[0]
            c0, c1 = sorted([a[1], b[1]])
            for c in range(c0, c1 + 1):
                if out[r, c] == BLACK:
                    out[r, c] = color
        else:
            c = a[1]
            r0, r1 = sorted([a[0], b[0]])
            for r in range(r0, r1 + 1):
                if out[r, c] == BLACK:
                    out[r, c] = color
    for color, a, b in pairs:
        if a[0] == b[0]:
            r = a[0]
            c0, c1 = sorted([a[1], b[1]])
            obstacles = [(r, c) for c in range(c0, c1 + 1) if grid[r, c] == GREY]
            dirs = [(-1, 0), (1, 0)]
        else:
            c = a[1]
            r0, r1 = sorted([a[0], b[0]])
            obstacles = [(r, c) for r in range(r0, r1 + 1) if grid[r, c] == GREY]
            dirs = [(0, -1), (0, 1)]
        for r, c in obstacles:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if out[nr, nc] not in (BLACK, GREY):
                        break
                    out[nr, nc] = GREY
                    nr += dr
                    nc += dc
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample(ANCHOR_COLORS, random.randint(3, 5)):
        if random.random() < 0.5:
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - 6)
            c1 = random.randint(c0 + 4, w - 2)
            grid[r, c0] = color
            grid[r, c1] = color
            for c in random.sample(range(c0 + 1, c1), random.randint(1, min(3, c1 - c0 - 1))):
                grid[r, c] = GREY
        else:
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - 6)
            r1 = random.randint(r0 + 4, h - 2)
            grid[r0, c] = color
            grid[r1, c] = color
            for r in random.sample(range(r0 + 1, r1), random.randint(1, min(3, r1 - r0 - 1))):
                grid[r, c] = GREY
    return grid
