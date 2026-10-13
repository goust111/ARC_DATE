from __future__ import annotations

import numpy as np


BACKGROUND = 0
BAND_HEIGHT = 3
INPUT_HEIGHT = 15
INPUT_WIDTH = 10
PAIR_MAP = {
    8: 2,
    3: 1,
    5: 4,
}
SOURCE_COLORS = tuple(PAIR_MAP)
ROW_COUNT_MIN = 3
ROW_COUNT_MAX = 6
MIN_COLOR_OCCURRENCES = 2
MAX_ATTEMPTS = 400


def _odd_copy(tile: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(tile)
    out = mirrored.copy()
    for src, dst in PAIR_MAP.items():
        out[mirrored == src] = dst
    return out


def _find_band(grid: np.ndarray) -> tuple[int, np.ndarray]:
    active_rows = np.any(grid != BACKGROUND, axis=1)
    rows = np.flatnonzero(active_rows)
    assert rows.size > 0, "grid must contain a foreground row band"

    r0 = int(rows[0])
    r1 = int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1)), "foreground rows must be contiguous"
    assert r1 - r0 + 1 == BAND_HEIGHT, "foreground band must be exactly three rows tall"
    assert np.all(grid[:r0] == BACKGROUND), "rows above the band must be background"
    assert np.all(grid[r1 + 1 :] == BACKGROUND), "rows below the band must be background"
    assert r0 % BAND_HEIGHT == 0, "band must align to full-height tiling slots"

    tile = grid[r0 : r1 + 1, :].copy()
    colors = sorted(int(color) for color in np.unique(tile) if color != BACKGROUND)
    assert 2 <= len(colors) <= 3, "band must use two or three paired source colors"
    assert all(color in PAIR_MAP for color in colors), "every source color must have a paired mapping"
    assert not np.array_equal(tile, np.fliplr(tile)), "band must not be left-right symmetric"
    return r0, tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the 3-row foreground band vertically, alternating plain copies with mirrored paired-color copies.

    Input structure:
    - The input is a black-background grid containing exactly one contiguous full-width foreground band.
    - The band is exactly three rows tall and contains two or three source colors from {teal, green, grey}.
    - Rows outside the band are entirely black, and the band pattern is not left-right symmetric.

    Rule:
    - Identify the unique three-row foreground band and treat it as the tile anchored at its original row slot.
    - Repeat that band upward and downward in steps of three rows until the whole canvas is filled.
    - Keep even-numbered copies unchanged, but for odd-numbered copies first mirror the band left-to-right and then recolor teal->red, green->blue, grey->yellow.

    Key insight:
    - The sparse input marks one phase of a row-band tiling, and every odd offset copy combines two mandatory operations: horizontal mirroring and paired recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.shape[0] % BAND_HEIGHT == 0, "grid height must be divisible by the band height"

    r0, tile = _find_band(g)
    odd_tile = _odd_copy(tile)

    out = np.empty_like(g)
    for start in range(0, g.shape[0], BAND_HEIGHT):
        copy_index = (start - r0) // BAND_HEIGHT
        out[start : start + BAND_HEIGHT, :] = tile if copy_index % 2 == 0 else odd_tile
    return out


def _band_start_choices(height: int) -> list[int]:
    return [start for start in range(BAND_HEIGHT, height - BAND_HEIGHT, BAND_HEIGHT)]


def _random_mask(rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((BAND_HEIGHT, INPUT_WIDTH), dtype=bool)
    for row in range(BAND_HEIGHT):
        count = int(rng.integers(ROW_COUNT_MIN, ROW_COUNT_MAX + 1))
        cols = rng.choice(INPUT_WIDTH, size=count, replace=False)
        mask[row, cols] = True
    return mask


def _fill_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    chosen_colors: tuple[int, ...],
) -> np.ndarray:
    tile = np.zeros(mask.shape, dtype=int)
    positions = list(map(tuple, np.argwhere(mask)))

    color_pool = []
    for color in chosen_colors:
        color_pool.extend([color] * MIN_COLOR_OCCURRENCES)
    extra = len(positions) - len(color_pool)
    if extra < 0:
        return tile
    if extra:
        color_pool.extend(rng.choice(chosen_colors, size=extra, replace=True).tolist())
    rng.shuffle(color_pool)

    for (r, c), color in zip(positions, color_pool):
        tile[r, c] = int(color)
    return tile


def _tile_valid(tile: np.ndarray, chosen_colors: tuple[int, ...]) -> bool:
    if tile.shape != (BAND_HEIGHT, INPUT_WIDTH):
        return False
    if np.any(np.all(tile == BACKGROUND, axis=1)):
        return False

    mask = tile != BACKGROUND
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(tile, np.fliplr(tile)):
        return False
    if np.count_nonzero(mask[:, : INPUT_WIDTH // 2]) == 0:
        return False
    if np.count_nonzero(mask[:, INPUT_WIDTH // 2 :]) == 0:
        return False

    colors = [int(color) for color in np.unique(tile) if color != BACKGROUND]
    if sorted(colors) != sorted(chosen_colors):
        return False

    for color in chosen_colors:
        if int(np.count_nonzero(tile == color)) < MIN_COLOR_OCCURRENCES:
            return False

    odd_tile = _odd_copy(tile)
    if np.array_equal(tile, odd_tile):
        return False
    if np.array_equal(np.fliplr(tile), odd_tile):
        return False
    return True


def _build_tile(rng: np.random.Generator) -> np.ndarray | None:
    color_count = 3 if float(rng.random()) < 0.65 else 2
    chosen = tuple(int(color) for color in rng.choice(SOURCE_COLORS, size=color_count, replace=False))

    for _ in range(MAX_ATTEMPTS):
        mask = _random_mask(rng)
        tile = _fill_mask(rng, mask, chosen)
        if _tile_valid(tile, chosen):
            return tile
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile = _build_tile(rng)
    if tile is None:
        return None

    start = int(rng.choice(_band_start_choices(INPUT_HEIGHT)))
    grid = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=int)
    grid[start : start + BAND_HEIGHT, :] = tile

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None

    for block_start in range(0, INPUT_HEIGHT, BAND_HEIGHT):
        copy_index = (block_start - start) // BAND_HEIGHT
        block = out[block_start : block_start + BAND_HEIGHT, :]
        expected = tile if copy_index % 2 == 0 else _odd_copy(tile)
        if not np.array_equal(block, expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
