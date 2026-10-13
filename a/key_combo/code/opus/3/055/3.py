import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Use each segment's top row for layer order and its pixel count for concentric square thickness."""
    segments = []
    for color in sorted(int(x) for x in np.unique(grid) if x != 0):
        coords = np.argwhere(grid == color)
        segments.append((int(coords[:, 0].min()), int(coords.shape[0]), color))
    segments.sort()
    inner_to_outer = list(reversed(segments))
    _, size, color = inner_to_outer[0]
    out = np.full((size, size), color, dtype=int)
    for _, thickness, color in inner_to_outer[1:]:
        n = out.shape[0] + 2 * thickness
        wrapped = np.full((n, n), color, dtype=int)
        wrapped[thickness:thickness + out.shape[0], thickness:thickness + out.shape[1]] = out
        out = wrapped
    return out


def generate_input() -> np.ndarray:
    while True:
        grid = np.zeros((18, 12), dtype=int)
        n = random.randint(3, 5)
        lengths = [random.randint(1, 4) for _ in range(n)]
        if 2 * sum(lengths) - n > 15:
            continue
        colors = random.sample([1, 2, 3, 7, 6], n)
        rows = random.sample(range(18), n)
        occupied = np.zeros_like(grid, dtype=bool)
        ok = True
        for row, length, color in zip(rows, lengths, colors):
            for _ in range(100):
                horiz = random.choice([True, False])
                if horiz or length == 1:
                    r = row
                    c = random.randrange(12 - length + 1)
                    cells = [(r, c + i) for i in range(length)]
                else:
                    r = row
                    if r + length > 18:
                        continue
                    c = random.randrange(12)
                    cells = [(r + i, c) for i in range(length)]
                blocked = False
                for cr, cc in cells:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < 18 and 0 <= nc < 12 and occupied[nr, nc]:
                                blocked = True
                if not blocked:
                    for p in cells:
                        occupied[p] = True
                        grid[p] = color
                    break
            else:
                ok = False
        if ok:
            return grid
