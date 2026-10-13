from __future__ import annotations

import numpy as np

from arc_utils import Color, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stagger the two horizontal input regions onto a larger maroon separator canvas.

    Input structure:
    - The input is a rectangular grid with an even number of rows.
    - The top half and bottom half are the two regions to preserve.
    - Maroon is absent from the input and is used only as the new separator color.

    Rule:
    - Create an output with one extra row and one extra column, filled with maroon.
    - Copy the top half into the upper-left corner.
    - Copy the bottom half one row below the top region and one column to the right.

    Key insight:
    The output is not a recoloring of either region; it inserts an L-shaped maroon
    gap by diagonally offsetting the lower half from the upper half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    assert H > 0 and W > 0 and H % 2 == 0, "input must have two equal-height regions"
    assert not bool(np.any(g == Color.MAROON)), "maroon is reserved for the inserted separator"

    half = H // 2
    out = np.full((H + 1, W + 1), Color.MAROON, dtype=int)
    paste(out, g[:half, :], 0, 0)
    paste(out, g[half:, :], half + 1, 1)
    return out

# --------------- generator (att1) ---------------

# Canvas bounds, color counts, palette, and retry budget.
HEIGHT_OPTIONS = (2, 4, 6, 8, 10)
WIDTH_RANGE, COLOR_COUNT_RANGE = (1, 9), (2, 5)
NON_MAROON_COLORS = tuple(c for c in Color.ALL_COLORS if c != Color.MAROON)
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = rng.choice(HEIGHT_OPTIONS)
    W = rng.integers(*WIDTH_RANGE)
    n_colors = rng.integers(*COLOR_COUNT_RANGE)
    colors = rng.choice(NON_MAROON_COLORS, size=n_colors, replace=False)
    inp = np.array(rng.choice(colors, size=(H, W)), dtype=int)
    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
