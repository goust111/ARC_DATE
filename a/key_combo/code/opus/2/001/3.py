import random
from math import gcd

import numpy as np


def _detect_k(grid):
    pts = np.argwhere(grid == 3)
    rows = sorted(set(int(r) for r, _ in pts))
    cols = sorted(set(int(c) for _, c in pts))
    runs = []
    for vals in (rows, cols):
        start = prev = vals[0]
        for v in vals[1:]:
            if v == prev + 1:
                prev = v
            else:
                runs.append(prev - start + 1)
                start = prev = v
        runs.append(prev - start + 1)
    k = runs[0]
    for run in runs[1:]:
        k = gcd(k, run)
    return max(1, k)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    k = _detect_k(grid)
    h, w = grid.shape
    small = np.zeros((h // k, w // k), dtype=int)
    for r in range(small.shape[0]):
        for c in range(small.shape[1]):
            if np.any(grid[r * k : (r + 1) * k, c * k : (c + 1) * k] == 3):
                small[r, c] = 1
    sh, sw = small.shape
    for r in range(sh):
        for c in range(sw):
            if small[r, c] == 1 and small[sh - 1 - r, sw - 1 - c] == 0:
                rr, cc = (sh - 1 - r) * k, (sw - 1 - c) * k
                out[rr : rr + k, cc : cc + k] = 2
    return out


def generate_input() -> np.ndarray:
    k = random.choice([2, 3, 4])
    sh = random.choice([4, 5, 6])
    sw = random.choice([4, 5, 6])
    if sh * k > 30 or sw * k > 30:
        sh = sw = 5
    small = np.zeros((sh, sw), dtype=int)
    pairs = []
    for r in range(sh):
        for c in range(sw):
            mr, mc = sh - 1 - r, sw - 1 - c
            if (r, c) <= (mr, mc):
                pairs.append(((r, c), (mr, mc)))
    random.shuffle(pairs)
    for a, b in pairs[: random.randint(4, min(9, len(pairs)))]:
        small[a] = 1
        small[b] = 1
    ones = [(r, c) for r in range(sh) for c in range(sw) if small[r, c] == 1 and (r, c) != (sh - 1 - r, sw - 1 - c)]
    random.shuffle(ones)
    for r, c in ones[: random.choice([1, 2])]:
        small[r, c] = 0
    grid = np.zeros((sh * k, sw * k), dtype=int)
    for r in range(sh):
        for c in range(sw):
            if small[r, c]:
                grid[r * k : (r + 1) * k, c * k : (c + 1) * k] = 3
    return grid
