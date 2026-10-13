import random
import numpy as np


def _orbit(n, r, c):
    return {(r, c), (n - 1 - r, c), (r, n - 1 - c), (n - 1 - r, n - 1 - c)}


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    n = grid.shape[0]
    gray = [(int(r), int(c)) for r, c in np.argwhere(grid == 5)]
    filled = {}
    for r, c in gray:
        colors = [int(grid[a, b]) for a, b in _orbit(n, r, c) if grid[a, b] not in (0, 5)]
        if colors:
            color = colors[0]
            out[r, c] = color
            filled[(r, c)] = color
    for (r, c), color in filled.items():
        if color == 2:
            nbrs = ((-1, -1), (-1, 1), (1, -1), (1, 1))
            mark = 4
        else:
            nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
            mark = 7
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and out[nr, nc] == 0:
                out[nr, nc] = mark
    return out


def generate_input() -> np.ndarray:
    n = random.choice([9, 11, 13])
    grid = np.zeros((n, n), dtype=int)
    orbits = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            orb = tuple(sorted(_orbit(n, r, c)))
            if len(orb) > 1 and orb not in orbits:
                orbits.append(orb)
    random.shuffle(orbits)
    chosen = orbits[:random.randint(5, 8)]
    gray_counts = {1: 0, 2: 0}
    for orb in chosen:
        color = random.choice([1, 2])
        for r, c in orb:
            grid[r, c] = color
        if gray_counts[color] < 2 or random.random() < 0.55:
            r, c = random.choice(orb)
            grid[r, c] = 5
            gray_counts[color] += 1
    if gray_counts[1] < 2 or gray_counts[2] < 2:
        return generate_input()
    return grid
