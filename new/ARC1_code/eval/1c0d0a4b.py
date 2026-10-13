from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected at least one occupied row or column"
    runs = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each separated 3x3 tile by its background-cell complement, colored red.

    Input structure:
    - The background is the dominant empty color.
    - One foreground color occupies cells inside 3x3 tile regions.
    - Tile regions are separated from each other by full-background rows and columns.

    Rule:
    - Keep the same grid shape and the same separator/background cells.
    - Inside each 3x3 tile, erase every foreground pixel.
    - Paint each originally background pixel inside the tile red.

    Key insight:
    - The task is not about the connected teal objects globally; it is a local
      occupancy inversion performed independently inside every separated 3x3 tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    foreground_colors = set(map(int, np.unique(g[g != bg])))
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    occupied_rows = [r for r in range(g.shape[0]) if np.any(g[r, :] != bg)]
    occupied_cols = [c for c in range(g.shape[1]) if np.any(g[:, c] != bg)]
    row_runs = _runs(occupied_rows)
    col_runs = _runs(occupied_cols)
    assert all(r1 - r0 + 1 == 3 for r0, r1 in row_runs), "expected 3-cell tile row bands"
    assert all(c1 - c0 + 1 == 3 for c0, c1 in col_runs), "expected 3-cell tile column bands"

    out = np.full_like(g, bg)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (3, 3), "expected a 3x3 tile"
            out[r0 : r1 + 1, c0 : c1 + 1] = np.where(tile == bg, Color.RED, bg)

    return out

# --------------- generator (o5x) ---------------

# Canvas/tile geometry.
TILE_SIZE = 3
TILE_STRIDE = 4
TILE_MARGIN = 1
# Palette.
BG_COLOR = 0
INPUT_COLOR = 8
OUTPUT_COLOR = 2
# Layout, occupancy, and retry sampling knobs.
BAND_SHAPE_GROUPS = (((1, 3), (3, 1)), ((2, 2),), ((2, 3), (3, 2)), ((3, 3),))
MIN_FG_PER_TILE = 4
MAX_FG_PER_TILE = 7
MAX_GENERATION_ATTEMPTS = 300


def _seed_tile_masks(row_bands: int, col_bands: int) -> list[list[np.ndarray]]:
    """Return masks seeded so every local row and column has foreground."""
    tile_masks = [[np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool) for _ in range(col_bands)] for _ in range(row_bands)]
    for band_r in range(row_bands):
        for local_r in range(TILE_SIZE):
            band_c = int(np.random.randint(col_bands))
            local_c = int(np.random.randint(TILE_SIZE))
            tile_masks[band_r][band_c][local_r, local_c] = True
    for band_c in range(col_bands):
        for local_c in range(TILE_SIZE):
            band_r = int(np.random.randint(row_bands))
            local_r = int(np.random.randint(TILE_SIZE))
            tile_masks[band_r][band_c][local_r, local_c] = True
    return tile_masks


def _render_grids(row_bands: int, col_bands: int, tile_masks) -> tuple[np.ndarray, np.ndarray]:
    """Render masks as the input grid and oracle output used by smoke checks."""
    grid_shape = (TILE_STRIDE * row_bands + TILE_MARGIN, TILE_STRIDE * col_bands + TILE_MARGIN)
    grid = np.zeros(grid_shape, dtype=int)
    expected = np.zeros_like(grid)
    for band_r in range(row_bands):
        for band_c in range(col_bands):
            mask = tile_masks[band_r][band_c]
            current_count = int(mask.sum())
            target_count = int(np.random.randint(max(MIN_FG_PER_TILE, current_count), MAX_FG_PER_TILE + 1))
            if target_count > current_count:
                chosen = np.random.choice(np.flatnonzero(~mask.ravel()), size=target_count - current_count, replace=False)
                mask.ravel()[chosen] = True
            r0 = TILE_MARGIN + TILE_STRIDE * band_r
            c0 = TILE_MARGIN + TILE_STRIDE * band_c
            tile = np.s_[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            grid[tile] = np.where(mask, INPUT_COLOR, BG_COLOR)
            expected[tile] = np.where(mask, BG_COLOR, OUTPUT_COLOR)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    group = BAND_SHAPE_GROUPS[np.random.randint(len(BAND_SHAPE_GROUPS))]
    row_bands, col_bands = group[np.random.randint(len(group))]
    tile_masks = _seed_tile_masks(row_bands, col_bands)
    grid, expected = _render_grids(row_bands, col_bands, tile_masks)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
