import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the full-width colored top band into the bottom rows while leaving the original band and middle background unchanged.

    Input structure
    - The background is the most common color.
    - All foreground pixels form a contiguous band of complete rows starting at the top edge.
    - Every row below that top band is background.

    Rule
    - Copy the top band into the same-width bottom rows, reversing the row order vertically.
    - Preserve the original top band and the background gap between the two bands.

    Key insight
    - The output is a vertical edge reflection of the input's top colored row band, not a recoloring or resizing.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    row_has_fg = np.any(g != bg, axis=1)
    assert row_has_fg.size > 0
    assert row_has_fg[0]

    top_h = 0
    for has_fg in row_has_fg:
        if not has_fg:
            break
        top_h += 1

    assert top_h > 0
    assert top_h * 2 <= g.shape[0]
    assert not np.any(row_has_fg[top_h:])

    top_band = g[:top_h, :]
    assert np.all(top_band != bg)
    assert np.all(g[top_h:, :] == bg)

    out = g.copy()
    out[-top_h:, :] = top_band[::-1, :]
    return out

# --------------- generator (o5x) ---------------

# Canvas/top-band bounds and row-color sampling.
GRID_ROWS, GRID_COLS, BAND_HEIGHT = (8, 14), (3, 8), (2, 4)
FOREGROUND_COLORS, ROW_COLOR_COUNT = range(1, 10), 2
# Retry budget.
MAX_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    band_h = random.randint(BAND_HEIGHT[0], min(BAND_HEIGHT[1], rows // 2))

    colors = random.sample(FOREGROUND_COLORS, ROW_COLOR_COUNT)
    row_colors = colors[:] + [random.choice(colors) for _ in range(band_h - ROW_COLOR_COUNT)]
    random.shuffle(row_colors)

    grid = np.zeros((rows, cols), dtype=int)
    grid[:band_h, :] = np.array(row_colors)[:, None]

    expected = grid.copy()
    expected[-band_h:, :] = grid[:band_h, :][::-1, :]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
