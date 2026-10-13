from __future__ import annotations

import random

import numpy as np


def _clockwise(tile: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(tile, -turns)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a binary mask by stamping successive clockwise rotations of the tile on foreground cells.

    Input structure:
    - A full grey vertical divider separates a square mask on the left from a square tile on the right.
    - The mask uses black and one foreground color.
    - The tile is a black-backed one-color asymmetric pattern.

    Rule:
    - Scan mask cells in row-major order and number only foreground cells.
    - Background mask cells produce all-black output blocks.
    - The kth foreground cell receives the tile rotated clockwise by k-1 quarter turns.

    Key insight:
    The same mask controls both block placement and the rotation counter, but the counter advances only on foreground cells.
    """
    g = np.asarray(grid, dtype=int)
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == 5)]
    if len(sep_cols) != 1:
        raise ValueError("expected one grey separator column")
    sep = sep_cols[0]
    n = sep
    m = g.shape[1] - sep - 1
    if g.shape[0] < max(n, m):
        raise ValueError("input is too short for the declared square panels")
    mask = g[:n, :sep]
    tile = g[:m, sep + 1 :]
    out = np.zeros((n * m, n * m), dtype=int)
    count = 0
    for r in range(n):
        for c in range(n):
            if int(mask[r, c]) == 0:
                continue
            rotated = _clockwise(tile, count % 4)
            out[r * m : (r + 1) * m, c * m : (c + 1) * m] = rotated
            count += 1
    return out


def _distinct_rotations(tile: np.ndarray) -> bool:
    rots = [_clockwise(tile, k) for k in range(4)]
    return all(not np.array_equal(rots[i], rots[j]) for i in range(4) for j in range(i + 1, 4))


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        n = random.choice([3, 4])
        m = random.choice([3, 4])
        a, b = random.sample([c for c in range(1, 10) if c != 5], 2)
        mask = np.zeros((n, n), dtype=int)
        foreground_count = random.randint(3, n * n - 1)
        for r, c in random.sample([(r, c) for r in range(n) for c in range(n)], foreground_count):
            mask[r, c] = a
        tile = np.zeros((m, m), dtype=int)
        if m == 3:
            coords = [(0, 0), (1, 0), (1, 2), (2, 1)]
        else:
            coords = [(0, 0), (0, 2), (1, 0), (2, 3), (3, 1)]
            if random.random() < 0.5:
                coords.append((2, 1))
        for r, c in coords:
            tile[r, c] = b
        if not _distinct_rotations(tile):
            continue
        height = max(n, m)
        g = np.zeros((height, n + 1 + m), dtype=int)
        g[:, n] = 5
        g[:n, :n] = mask
        g[:m, n + 1 :] = tile
        return g
    raise RuntimeError("failed to generate puzzle input")
