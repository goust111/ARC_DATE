from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each input pixel into a miniature same-color mask of the whole grid.

    Input structure
    - The input is a non-empty two-dimensional grid of foreground colors.
    - Black is reserved for the empty/background pixels introduced in the expanded output.
    - The output has one input-shaped block for each input pixel.

    Rule
    - For every input pixel with color c, replace that pixel by a block the same shape as the input.
    - Inside the block, write c exactly where the original input also has color c.
    - Write black everywhere else in the block.

    Key insight
    Each pixel selects the global mask of its own color; the result is the blockwise self-mask of the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all(g != 0), "input uses foreground colors; black is output background"

    h, w = g.shape
    masks = {int(color): np.where(g == int(color), int(color), 0) for color in np.unique(g)}

    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = masks[int(g[r, c])]
    return out

# --------------- generator (att2) ---------------
# Sampling bounds and retry budget.
GRID_SIZES = (3, 4, 5)
COLOR_COUNTS = (2, 3)
FOREGROUND_COLORS = np.arange(1, 10)
MIN_CELLS_PER_COLOR = 2
GEN_RETRY_LIMIT = 300


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    h, w = inp.shape
    masks = {int(color): np.where(inp == int(color), int(color), 0) for color in np.unique(inp)}
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = masks[int(inp[r, c])]
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(GRID_SIZES))
    k = int(np.random.choice(COLOR_COUNTS))
    colors = np.random.choice(FOREGROUND_COLORS, size=k, replace=False).astype(int)

    flat = []
    for color in colors:
        flat.extend([int(color)] * MIN_CELLS_PER_COLOR)
    flat.extend(np.random.choice(colors, size=n * n - len(flat)).astype(int).tolist())
    np.random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(n, n)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid 973e499e input")
