import random
from collections import Counter

import numpy as np


QUADS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _split_axis(grid):
    h, w = grid.shape
    if len(set(grid[0, :])) == 1 and grid[0, 0] != 0:
        return grid[1:, :], "vertical"
    if len(set(grid[-1, :])) == 1 and grid[-1, 0] != 0:
        return grid[:-1, :], "vertical"
    if len(set(grid[:, 0])) == 1 and grid[0, 0] != 0:
        return grid[:, 1:], "horizontal"
    if len(set(grid[:, -1])) == 1 and grid[0, -1] != 0:
        return grid[:, :-1], "horizontal"
    raise ValueError("axis indicator not found")


def _unfold_body(body):
    block_rows = body.shape[0] // 2
    block_cols = body.shape[1] // 2
    blocks = []
    for br in range(block_rows - 1, -1, -1):
        for bc in range(block_cols - 1, -1, -1):
            blocks.append(body[2 * br:2 * br + 2, 2 * bc:2 * bc + 2])
    layers = len(blocks)
    q = 2 * layers
    out = np.zeros((2 * q, 2 * q), dtype=int)
    origins = {(0, 0): (0, 0), (0, 1): (0, q), (1, 0): (q, 0), (1, 1): (q, q)}
    for k, block in enumerate(blocks):
        vals = {(0, 0): int(block[0, 0]), (0, 1): int(block[0, 1]),
                (1, 0): int(block[1, 0]), (1, 1): int(block[1, 1])}
        lo, hi = k, q - 1 - k
        for quad, val in vals.items():
            r0, c0 = origins[quad]
            out[r0 + lo:r0 + hi + 1, c0 + lo] = val
            out[r0 + lo:r0 + hi + 1, c0 + hi] = val
            out[r0 + lo, c0 + lo:c0 + hi + 1] = val
            out[r0 + hi, c0 + lo:c0 + hi + 1] = val
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Unfold a 2x2 block strip, then mirror the rarest outer-color quadrant.

    Input structure: A 2x2-block strip has an extra uniform edge row or column
    that acts as the reflection-axis indicator.
    Rule: Remove the indicator, read body blocks in reverse row-major order, and
    use each block's four pixels as the next nested layer color of the four
    output quadrants. The quadrant whose outer color is least frequent in the
    unfolded output is reflected across the indicated vertical or horizontal
    axis and copied over its opposite quadrant.
    Key insight: The source quadrant is selected by global rarity after unfold,
    while the indicator edge independently decides the mirror direction.
    """
    grid = np.asarray(grid, dtype=int)
    body, axis = _split_axis(grid)
    out = _unfold_body(body)
    q = out.shape[0] // 2
    outer_colors = {(0, 0): int(out[0, 0]), (0, 1): int(out[0, q]),
                    (1, 0): int(out[q, 0]), (1, 1): int(out[q, q])}
    counts = Counter(int(x) for x in out.ravel())
    source = min(QUADS, key=lambda quad: (counts[outer_colors[quad]], quad[0], quad[1]))
    sr, sc = source[0] * q, source[1] * q
    patch = out[sr:sr + q, sc:sc + q].copy()
    if axis == "vertical":
        target = (1 - source[0], source[1])
        patch = np.flipud(patch)
    else:
        target = (source[0], 1 - source[1])
        patch = np.fliplr(patch)
    tr, tc = target[0] * q, target[1] * q
    out[tr:tr + q, tc:tc + q] = patch
    return out


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 6, 7]
    for _ in range(100):
        body = np.zeros((4, 6), dtype=int)
        common = random.sample(palette[:4], 3)
        for r in range(4):
            for c in range(6):
                body[r, c] = random.choice(common)
        rare = random.choice([6, 7])
        body[-2, -2] = rare
        body[-2, -1] = common[0]
        body[-1, -2] = common[1]
        body[-1, -1] = common[2]
        if random.random() < 0.5:
            body = body.T
        raw = _unfold_body(body)
        q = raw.shape[0] // 2
        outer = [int(raw[0, 0]), int(raw[0, q]), int(raw[q, 0]), int(raw[q, q])]
        counts = Counter(int(x) for x in raw.ravel())
        values = [counts[c] for c in outer]
        if values.count(min(values)) != 1:
            continue
        indicator_color = random.choice(common + [rare])
        if random.random() < 0.5:
            row = np.full((1, body.shape[1]), indicator_color, dtype=int)
            return np.vstack([row, body]) if random.random() < 0.5 else np.vstack([body, row])
        col = np.full((body.shape[0], 1), indicator_color, dtype=int)
        return np.hstack([col, body]) if random.random() < 0.5 else np.hstack([body, col])
    body = np.array([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4],
                     [1, 2, 1, 2, 6, 1], [3, 4, 3, 4, 2, 3]], dtype=int)
    return np.vstack([np.full((1, 6), 1, dtype=int), body])
