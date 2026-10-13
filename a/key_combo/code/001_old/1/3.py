from __future__ import annotations

import random

import numpy as np


def _ring_coords(n: int, radius: int) -> list[tuple[int, int]]:
    c = n // 2
    r0, r1 = c - radius, c + radius
    c0, c1 = c - radius, c + radius
    coords: list[tuple[int, int]] = []
    coords.extend((r0, cc) for cc in range(c0, c1 + 1))
    coords.extend((rr, c1) for rr in range(r0 + 1, r1 + 1))
    coords.extend((r1, cc) for cc in range(c1 - 1, c0 - 1, -1))
    coords.extend((rr, c0) for rr in range(r1 - 1, r0, -1))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each concentric square ring by the number of non-black colors visible on that ring.

    Input structure:
    - An odd square grid has black background and scattered colored pixels on concentric square rings.
    - Each ring may use one to three non-black colors.
    - The center cell is independent of the rings.

    Rule:
    - For every ring around the center, count its distinct non-black colors.
    - Move all cells on that ring clockwise by that count, carrying black cells along with colored cells.
    - Leave the center cell unchanged.

    Key insight:
    The ring decomposition supplies the independent paths, and each path's color count supplies its own cyclic shift distance.
    """
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    if n != m or n % 2 == 0:
        raise ValueError("expected an odd square grid")
    out = g.copy()
    for radius in range(1, n // 2 + 1):
        coords = _ring_coords(n, radius)
        values = [int(g[r, c]) for r, c in coords]
        colors = {v for v in values if v != 0}
        if not colors:
            continue
        shift = len(colors) % len(coords)
        for i, (r, c) in enumerate(coords):
            nr, nc = coords[(i + shift) % len(coords)]
            out[nr, nc] = values[i]
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        n = random.choice([7, 9, 11])
        g = np.zeros((n, n), dtype=int)
        for radius in range(1, n // 2 + 1):
            coords = _ring_coords(n, radius)
            k_colors = random.randint(1, 3)
            colors = random.sample(range(1, 10), k_colors)
            fill_count = random.randint(max(1, len(coords) // 5), max(1, (len(coords) * 3) // 5))
            for r, c in random.sample(coords, fill_count):
                g[r, c] = random.choice(colors)
            values = [int(g[r, c]) for r, c in coords]
            shift = len({v for v in values if v}) % len(coords)
            if shift and values == values[-shift:] + values[:-shift]:
                rr, cc = random.choice(coords)
                g[rr, cc] = random.choice(colors)
        if random.random() < 0.5:
            g[n // 2, n // 2] = random.choice(range(0, 10))
        return g
    raise RuntimeError("failed to generate puzzle input")
