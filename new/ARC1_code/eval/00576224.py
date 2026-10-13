from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a 6x6 pattern by tiling a 2x2 grid, with the middle tile row mirrored left-to-right.

    Input structure:
    - A 2x2 color tile; no color has a separate background role.

    Rule:
    - Make a 3-by-3 array of 2x2 tiles.
    - Use the input tile unchanged in the top and bottom tile rows.
    - Use the left-right mirror of the input tile in the middle tile row.

    Key insight:
    - The output is not pixel scaling: it is repeated whole-tile placement with a horizontal reflection in the center band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (2, 2), "expected a 2x2 input tile"

    normal_band = np.tile(g, (1, 3))
    mirrored_band = np.tile(np.fliplr(g), (1, 3))
    return np.vstack((normal_band, mirrored_band, normal_band))

# --------------- generator (o5x) ---------------
# Tile and color bounds.
TILE_SHAPE = (2, 2)
COLOR_COUNTS = (3, 4)
COLOR_POOL = np.arange(1, 10)
# Output construction and retry policy.
BAND_TILE_REPEAT = (1, 3)
MAX_GENERATION_ATTEMPTS = 300

def _sample_tile_values() -> np.ndarray:
    n_colors = int(np.random.choice(COLOR_COUNTS))
    colors = np.random.choice(COLOR_POOL, size=n_colors, replace=False)
    if n_colors == 3:
        values = np.array([*colors, int(np.random.choice(colors))], dtype=int)
    else:
        values = colors.astype(int)
    np.random.shuffle(values)
    return values

def _expected_output(tile: np.ndarray) -> np.ndarray:
    return np.vstack((
        np.tile(tile, BAND_TILE_REPEAT),
        np.tile(np.fliplr(tile), BAND_TILE_REPEAT),
        np.tile(tile, BAND_TILE_REPEAT),
    ))

def _gen_attempt() -> np.ndarray | None:
    tile = _sample_tile_values().reshape(TILE_SHAPE)
    if np.array_equal(tile, np.fliplr(tile)):
        return None
    return tile

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None and np.array_equal(main(inp), _expected_output(inp)):
            return inp
    raise RuntimeError("failed to generate a valid input")
