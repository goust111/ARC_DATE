from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Create a four-quadrant mirror pattern from the whole input grid.

    Input structure:
    - The entire rectangular input grid is the seed tile.

    Rule:
    - Place the input tile in the upper-left quadrant.
    - Place its left-to-right reflection in the upper-right quadrant.
    - Reflect that full top strip top-to-bottom to make the lower half.

    Key insight:
    - The output is a 2x2 reflection tiling, not a pixelwise scale-up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a non-empty grid"

    top = np.concatenate((g, np.fliplr(g)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)

# --------------- generator (o5x) ---------------
# Tile shape bounds.
TILE_SIZE_MIN = 2
TILE_SIZE_MAX_EXCLUSIVE = 6
# Palette sampling.
COLOR_VALUES = np.arange(10, dtype=int)
COLOR_COUNT_MIN = 3
COLOR_COUNT_MAX_EXCLUSIVE = 5
# Retry limit.
MAX_GENERATION_ATTEMPTS = 300

def _gen_expected_output(tile: np.ndarray) -> np.ndarray:
    top = np.concatenate((tile, np.fliplr(tile)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(TILE_SIZE_MIN, TILE_SIZE_MAX_EXCLUSIVE))
    width = int(rng.integers(TILE_SIZE_MIN, TILE_SIZE_MAX_EXCLUSIVE))
    color_count = int(rng.integers(COLOR_COUNT_MIN, COLOR_COUNT_MAX_EXCLUSIVE))
    palette = rng.choice(COLOR_VALUES, size=color_count, replace=False)

    tile = rng.choice(palette, size=(height, width), replace=True).astype(int)

    # These anchors make the seed non-symmetric in both axes by construction.
    tile[0, 0] = palette[0]
    tile[0, width - 1] = palette[1]
    tile[height - 1, 0] = palette[2]

    if color_count == COLOR_COUNT_MAX_EXCLUSIVE - 1:
        anchor_positions = {(0, 0), (0, width - 1), (height - 1, 0)}
        free_positions = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in anchor_positions
        ]
        r, c = free_positions[int(rng.integers(0, len(free_positions)))]
        tile[r, c] = palette[3]

    if np.array_equal(tile, np.fliplr(tile)) or np.array_equal(tile, np.flipud(tile)):
        return None

    expected = _gen_expected_output(tile)
    if not np.array_equal(main(tile), expected):
        return None
    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
