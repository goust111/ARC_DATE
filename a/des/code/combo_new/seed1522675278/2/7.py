import random

import numpy as np


BOARD = [0, 5]
SEEDS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow parity-restricted square rings around each colored seed, with ring depth determined by nearest-seed spacing.

    Input structure:
    - The background is a one-cell chessboard of black and grey.
    - Colored seeds replace some board cells and are well separated.
    - Each seed keeps the parity class of its original board square.

    Rule:
    - Around each seed, draw square rings of increasing Chebyshev radius.
    - Only ring cells with the same parity as the seed are painted.
    - The maximum radius is half the nearest-seed gap, rounded down.

    Key insight:
    - The board parity masks the ring, while nearest-neighbor distance caps its depth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    seeds = [(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] not in BOARD]
    for r, c, color in seeds:
        others = [max(abs(r - rr), abs(c - cc)) for rr, cc, _ in seeds if (rr, cc) != (r, c)]
        radius = max(1, (min(others) - 1) // 2) if others else 2
        parity = (r + c) % 2
        for k in range(1, radius + 1):
            for rr in range(r - k, r + k + 1):
                for cc in range(c - k, c + k + 1):
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and max(abs(rr - r), abs(cc - c)) == k and (rr + cc) % 2 == parity:
                        out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(17, 23)
    g = np.fromfunction(lambda r, c: np.where((r + c) % 2 == 0, BOARD[0], BOARD[1]), (h, w), dtype=int)
    pts = []
    for color in random.sample(SEEDS, random.randint(2, 4)):
        for _ in range(100):
            r, c = random.randint(3, h - 4), random.randint(3, w - 4)
            if all(max(abs(r - rr), abs(c - cc)) >= 6 for rr, cc in pts):
                g[r, c] = color
                pts.append((r, c))
                break
    return g.astype(int)
