from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the grid across its horizontal midline by reversing the row order.

    Input structure:
    - A non-empty 2D grid of colored pixels.

    Rule:
    - Preserve every row exactly as-is.
    - Place input row r at output row H - 1 - r.

    Key insight:
    - Only the vertical order of rows changes; colors and left-to-right row
      contents are unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a non-empty grid"
    return np.flipud(g).copy()

# --------------- generator (o5x) ---------------
# Sampling configuration: canvas sizes, palette, and retry budget.
_GRID_SIZES = [5, 7]
_PALETTE = np.array([1, 2, 3, 4, 7, 8], dtype=int)
_GEN_RETRY_LIMIT = 300

def _ensure_all_palette_colors(grid: np.ndarray) -> None:
    """Mutate grid so every palette color appears at least once."""
    n = int(grid.shape[0])
    present = set(int(x) for x in np.unique(grid))
    missing = [int(c) for c in _PALETTE if int(c) not in present]
    if missing:
        cells = np.random.permutation(n * n)[: len(missing)]
        for color, idx in zip(missing, cells):
            grid[int(idx) // n, int(idx) % n] = color


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(_GRID_SIZES))
    grid = np.random.choice(_PALETTE, size=(n, n)).astype(int)
    _ensure_all_palette_colors(grid)

    if np.array_equal(grid, np.flipud(grid)):
        return None

    out = np.flipud(grid).copy()
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 68b16354")
