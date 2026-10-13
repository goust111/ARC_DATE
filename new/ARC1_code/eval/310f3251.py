from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the input as a 3-by-3 wallpaper after adding a red up-left shadow of each foreground pixel.

    Input structure:
    - A background-colored grid with exactly one non-background foreground color.
    - Red is absent from the input, and all foreground pixels are below the top row.

    Rule:
    - In one tile, paint red at the cell one row above and one column left of each
      foreground pixel, wrapping the column around the left edge of the tile.
    - Original foreground pixels remain on top of any red shadow pixels.
    - Tile the marked grid three times vertically and three times horizontally.

    Key insight:
    - The output is a periodic repetition of a locally annotated copy of the
      input, not a scale-up of the input pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    assert foreground_colors[0] != Color.RED, "red must be the added shadow color"

    foreground = g != bg
    coords = np.argwhere(foreground)
    assert coords.size > 0, "expected at least one foreground pixel"
    assert np.all(coords[:, 0] > 0), "foreground pixels must have an up-left target row"

    width = g.shape[1]
    tile = np.full_like(g, bg)
    for r, c in coords:
        tile[int(r) - 1, (int(c) - 1) % width] = Color.RED

    tile[foreground] = g[foreground]
    return np.tile(tile, (3, 3))

# --------------- generator (o5x) ---------------
# Grid and foreground sampling.
GRID_SIZE_RANGE, FOREGROUND_COUNT_RANGE, FOREGROUND_ROW_MIN = (2, 6), (1, 3), 1
FOREGROUND_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
# Rendering and retry limits.
BACKGROUND_COLOR, SHADOW_COLOR, TILE_REPEATS, MAX_ATTEMPTS = 0, 2, (3, 3), 300


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    fg = inp != BACKGROUND_COLOR
    coords = np.argwhere(fg)
    _, width = inp.shape
    tile = np.zeros_like(inp)
    tile[coords[:, 0] - 1, (coords[:, 1] - 1) % width] = SHADOW_COLOR
    tile[fg] = inp[fg]
    return np.tile(tile, TILE_REPEATS)


def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.randint(*GRID_SIZE_RANGE))
    color = int(np.random.choice(FOREGROUND_COLORS))
    count = int(np.random.randint(*FOREGROUND_COUNT_RANGE))

    inp = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    cells = [(r, c) for r in range(FOREGROUND_ROW_MIN, size) for c in range(size)]
    chosen = np.random.choice(len(cells), size=count, replace=False)
    for idx in chosen:
        r, c = cells[int(idx)]
        inp[r, c] = color

    out = _gen_expected_output(inp)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
