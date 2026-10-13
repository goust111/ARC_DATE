import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Compress away gray separator lines and expand colored pixels into diamonds.

    Input structure: Full gray rows and columns partition a black grid, with
    colored singleton pixels placed in the remaining cells.
    Rule: Before deleting separators, each colored pixel counts adjacent gray
    neighbors to obtain its Manhattan diamond radius. After deleting all complete
    gray rows and columns, each pixel draws its radius diamond at the compressed
    coordinate; larger radii win overlaps, with reading order as the tie break.
    Key insight: The deleted separators still encode the expansion radius, so the
    radius must be read before coordinate compression changes the grid.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    gray_rows = {r for r in range(h) if np.all(grid[r, :] == 5)}
    gray_cols = {c for c in range(w) if np.all(grid[:, c] == 5)}
    row_map = {r: i for i, r in enumerate(r for r in range(h) if r not in gray_rows)}
    col_map = {c: i for i, c in enumerate(c for c in range(w) if c not in gray_cols)}

    pieces = []
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (0, 5):
                continue
            radius = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 5:
                    radius += 1
            pieces.append((row_map[r], col_map[c], radius, val))

    out = np.zeros((len(row_map), len(col_map)), dtype=int)
    for r, c, radius, val in sorted(pieces, key=lambda x: (-x[2], x[0], x[1])):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == 0:
                        out[rr, cc] = val
    return out


def generate_input() -> np.ndarray:
    h = random.choice([13, 14, 15])
    w = random.choice([13, 14, 15])
    row_lines = sorted(random.sample(range(3, h - 3), 2))
    col_lines = sorted(random.sample(range(3, w - 3), 2))
    grid = np.zeros((h, w), dtype=int)
    grid[row_lines, :] = 5
    grid[:, col_lines] = 5

    def radius_at(r, c):
        return sum(0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr, c + dc] == 5
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                  if grid[r, c] == 0 and radius_at(r, c) <= 2]
    zeros = [p for p in candidates if radius_at(*p) == 0]
    twos = [p for p in candidates if radius_at(*p) >= 2]
    chosen = []
    if zeros:
        chosen.append(random.choice(zeros))
    if twos:
        chosen.append(random.choice(twos))
    random.shuffle(candidates)
    for p in candidates:
        if p not in chosen and all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 2 for q in chosen):
            chosen.append(p)
        if len(chosen) >= random.randint(5, 7):
            break
    colors = [1, 2, 3, 4, 7]
    for r, c in chosen:
        grid[r, c] = random.choice(colors)
    return grid
