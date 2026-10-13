import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Select the quadrant with the magenta marker, rotate it counterclockwise, and take the green foreground union with its diagonal quadrant."""
    h, w = grid.shape
    sep_row = next(r for r in range(h) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != 0)
    sep_col = next(c for c in range(w) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) == int(grid[sep_row, 0]))
    qh, qw = sep_row, sep_col
    quads = {
        "tl": grid[:sep_row, :sep_col].copy(),
        "tr": grid[:sep_row, sep_col + 1 :].copy(),
        "bl": grid[sep_row + 1 :, :sep_col].copy(),
        "br": grid[sep_row + 1 :, sep_col + 1 :].copy(),
    }
    mr, mc = map(int, np.argwhere(grid == 6)[0])
    if mr < sep_row and mc < sep_col:
        src, diag = "tl", "br"
    elif mr < sep_row and mc > sep_col:
        src, diag = "tr", "bl"
    elif mr > sep_row and mc < sep_col:
        src, diag = "bl", "tr"
    else:
        src, diag = "br", "tl"
    source = quads[src]
    source[source == 6] = 0
    rotated = np.rot90(source, 1)
    target = quads[diag]
    return np.where((rotated != 0) | (target != 0), 3, 0).reshape(qh, qw)


def _random_mask(rng: random.Random, n: int, color: int, density: float) -> np.ndarray:
    mask = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            if rng.random() < density:
                mask[r, c] = color
    return mask


def generate_input() -> np.ndarray:
    rng = random.Random()
    while True:
        q = rng.choice([5, 6])
        size = 2 * q + 1
        grid = np.zeros((size, size), dtype=int)
        grid[q, :] = 5
        grid[:, q] = 5
        quads = {}
        for name, color in zip(("tl", "tr", "bl", "br"), rng.sample([1, 2, 3, 4], k=4)):
            quads[name] = _random_mask(rng, q, color, rng.uniform(0.22, 0.38))
        marker_quad = rng.choice(["tl", "tr", "bl", "br"])
        marker_positions = np.argwhere(quads[marker_quad] == 0)
        if marker_positions.size == 0:
            continue
        mr, mc = map(int, marker_positions[rng.randrange(len(marker_positions))])
        quads[marker_quad][mr, mc] = 6

        diag = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[marker_quad]
        src_no_marker = quads[marker_quad].copy()
        src_no_marker[src_no_marker == 6] = 0
        if np.array_equal(np.rot90(src_no_marker, 1) != 0, quads[diag] != 0):
            continue

        grid[:q, :q] = quads["tl"]
        grid[:q, q + 1 :] = quads["tr"]
        grid[q + 1 :, :q] = quads["bl"]
        grid[q + 1 :, q + 1 :] = quads["br"]
        return grid
