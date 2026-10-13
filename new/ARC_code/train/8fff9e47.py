from __future__ import annotations

import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a 4-cell-wide strip of 2x2 blocks into four nested corner-fill patterns.

    Input structure
    The input is an even-sized rectangle whose shorter dimension is four pixels.
    It can be partitioned exactly into 2x2 blocks, giving one block for each layer
    of the output quadrants.

    Rule
    Read the 2x2 blocks in reverse row-major order: bottom-right block first,
    moving left, then moving upward. For each block, send its top-left,
    top-right, bottom-left, and bottom-right pixels to the layer sequences of the
    upper-left, upper-right, lower-left, and lower-right output quadrants.

    Key insight
    Each input 2x2 block supplies the four colors for one concentric layer, and
    the layer order is the block order from bottom-right back to top-left.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "input must split into 2x2 blocks"
    assert min(h, w) == 4, "input must have one dimension of four pixels"

    n = max(h, w)
    blocks = []
    for br in range(h // 2 - 1, -1, -1):
        for bc in range(w // 2 - 1, -1, -1):
            block = g[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2]
            blocks.append(block.reshape(4))

    layers = np.asarray(blocks, dtype=int)
    assert layers.shape == (n, 4), "expected one 2x2 block per output layer"
    ul_seq, ur_seq, ll_seq, lr_seq = (layers[:, i] for i in range(4))

    out = np.empty((2 * n, 2 * n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = ul_seq[min(r, c)]
            out[r, n + c] = ur_seq[min(r, n - 1 - c)]
            out[n + r, c] = ll_seq[min(n - 1 - r, c)]
            out[n + r, n + c] = lr_seq[min(n - 1 - r, n - 1 - c)]

    return out

# --------------- generator (att2) ---------------
# Canvas/layout bounds.
LAYER_COUNTS, FIXED_SIDE = [4, 6, 8, 10, 12], 4
STRIP_ORIENTATIONS = ["vertical", "horizontal"]
SQUARE_ORIENTATIONS = ["vertical", "horizontal", "square"]
# Color sampling bounds.
COLOR_DOMAIN, COLOR_COUNT_RANGE = range(10), (8, 10)
# Generation retry limit.
MAX_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(LAYER_COUNTS)
    orientations = SQUARE_ORIENTATIONS if n == FIXED_SIDE else STRIP_ORIENTATIONS
    orientation = random.choice(orientations)
    h, w = (n, FIXED_SIDE) if orientation == "vertical" else (FIXED_SIDE, n)

    color_count = random.randint(*COLOR_COUNT_RANGE)
    palette = random.sample(COLOR_DOMAIN, color_count)
    total = h * w
    flat = palette[:] + random.choices(palette, k=total - color_count)
    random.shuffle(flat)
    grid = np.array(flat, dtype=int).reshape(h, w)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != (2 * n, 2 * n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
