import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
GREY = 5
SEED_COLORS = [1, 3, 4, 6, 7, 8, 9]


def _rectangles(grid):
    h, w = grid.shape
    rects = []
    grey = np.argwhere(grid == GREY)
    rows = sorted(set(int(r) for r, _ in grey))
    cols = sorted(set(int(c) for _, c in grey))
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0 + 2:
                continue
            for c0 in cols:
                for c1 in cols:
                    if c1 <= c0 + 2:
                        continue
                    if np.all(grid[r0, c0:c1 + 1] == GREY) and np.all(grid[r1, c0:c1 + 1] == GREY):
                        if np.all(grid[r0:r1 + 1, c0] == GREY) and np.all(grid[r0:r1 + 1, c1] == GREY):
                            interior = grid[r0 + 1:r1, c0 + 1:c1]
                            seeds = [(r0 + 1 + int(r), c0 + 1 + int(c), int(interior[r, c]))
                                     for r, c in zip(*np.where((interior != BG) & (interior != GREY)))]
                            if len(seeds) == 1:
                                rects.append((r0, r1, c0, c1, seeds[0]))
    # Generated rectangles are disjoint; remove duplicate large candidates.
    unique = []
    seen = set()
    for item in rects:
        key = item[:4]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """Grow cross branches from seeds by the area of their grey rectangle.

    Input structure:
    Grey rectangular frames enclose black interiors, each containing exactly one
    non-grey seed pixel.

    Rule:
    For every rectangle, compute its interior area.  The seed grows that many pixels
    up, down, left, and right, crossing grey borders if needed and clipping at the
    grid edge.  Different-colored branch crossings become red; same-colored crossings
    keep their color.  Frames and seed pixels remain.

    Key insight:
    The hidden scalar controlling branch length is the room's interior area, and
    branch overlaps use a special red mixing rule.
    """
    out = np.array(grid, copy=True)
    claims = defaultdict(list)
    for r0, r1, c0, c1, (sr, sc, color) in _rectangles(grid):
        length = (r1 - r0 - 1) * (c1 - c0 - 1)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, length + 1):
                r, c = sr + dr * step, sc + dc * step
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    break
                claims[(r, c)].append(color)
    seed_and_grey = {(r, c) for r, c in zip(*np.where((grid == GREY) | ((grid != BG) & (grid != GREY))))}
    for (r, c), colors in claims.items():
        if (r, c) in seed_and_grey:
            continue
        unique = set(colors)
        out[r, c] = colors[0] if len(unique) == 1 else RED
    return out


def _draw_rect(grid, r0, c0, ih, iw, color):
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    if r1 >= grid.shape[0] - 1 or c1 >= grid.shape[1] - 1:
        return False
    if np.any(grid[r0:r1 + 1, c0:c1 + 1] != BG):
        return False
    grid[r0, c0:c1 + 1] = GREY
    grid[r1, c0:c1 + 1] = GREY
    grid[r0:r1 + 1, c0] = GREY
    grid[r0:r1 + 1, c1] = GREY
    sr = random.randint(r0 + 1, r1 - 1)
    sc = random.randint(c0 + 1, c1 - 1)
    grid[sr, sc] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(18, 29)
        w = random.randint(18, 29)
        grid = np.zeros((h, w), dtype=int)
        colors = random.sample(SEED_COLORS, random.randint(2, 4))
        slots = [(2, 2), (2, w // 2), (h // 2, 2), (h // 2, w // 2)]
        random.shuffle(slots)
        ok = True
        for color, (r0, c0) in zip(colors, slots):
            if not _draw_rect(grid, r0, c0, random.randint(2, 4), random.randint(2, 4), color):
                ok = False
                break
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid seed-cross puzzle")
