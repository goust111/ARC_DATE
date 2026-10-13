from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SEED_COLORS = (RED, GREEN, BLUE)
DIVIDER_COLORS = (4, 5, 6, 7, 8, 9)
GEN_RETRIES = 300


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color == BLACK:
            continue
        if np.all(grid[r] == color):
            rows.append(int(r))
    return rows


def _band_slices(grid: np.ndarray, separators: list[int]) -> list[tuple[int, int, int]]:
    assert separators, "expected at least one horizontal divider"
    starts = [0] + [r + 1 for r in separators]
    stops = separators + [int(grid.shape[0])]
    bands: list[tuple[int, int, int]] = []
    for idx, (top, bottom) in enumerate(zip(starts, stops, strict=False)):
        if top >= bottom:
            continue
        if idx == 0:
            fill_color = int(grid[separators[0], 0])
        else:
            fill_color = int(grid[separators[idx - 1], 0])
        bands.append((int(top), int(bottom), fill_color))
    assert bands, "expected at least one non-empty band"
    return bands


def _seed_for_band(grid: np.ndarray, top: int, bottom: int) -> tuple[int, int, int]:
    coords = np.argwhere(grid[top:bottom, :] != BLACK)
    assert coords.shape[0] == 1, "each band must contain exactly one seed pixel"
    r_local, c = map(int, coords[0])
    r = top + r_local
    color = int(grid[r, c])
    assert color in SEED_COLORS, "band seed must be red, green, or blue"
    return r, c, color


def _pattern_mask(seed_color: int, top: int, bottom: int, width: int) -> np.ndarray:
    height = bottom - top
    if seed_color == RED:
        mask = np.zeros((height, width), dtype=bool)
        mask[::2, :] = True
        return mask
    if seed_color == GREEN:
        cols = (np.arange(width) % 2) == 0
        return np.broadcast_to(cols, (height, width)).copy()

    rows = np.arange(top, bottom)[:, None]
    cols = np.arange(width)[None, :]
    return ((rows + cols) % 2) == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separator-defined horizontal band with a seed-selected pattern in the adjacent divider color.

    Input structure:
    - The background is black.
    - One or more full-width horizontal divider rows split the grid into horizontal bands.
    - Each divider row is monochrome, and each band contains exactly one non-black seed pixel.
    - Seed pixels use only red, green, or blue, while divider colors come from other colors.

    Rule:
    - Identify the unique seed inside each band.
    - Red seeds produce horizontal stripes from the band's top row, green seeds produce
      vertical stripes from the grid's left edge, and blue seeds produce a checkerboard
      using global row+column parity.
    - The fill color is taken from the divider immediately above the band, except for the
      top band, which uses the divider immediately below it.
    - Only originally black pixels inside the band may be filled; divider rows stay fixed
      and the seed pixel itself is removed to black.

    Key insight:
    - The seed color encodes geometry only, while the visible paint color comes from a
      separate nearby object: the adjacent horizontal divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    separators = _separator_rows(g)
    bands = _band_slices(g, separators)

    out = g.copy()
    width = int(g.shape[1])
    for top, bottom, fill_color in bands:
        seed_r, seed_c, seed_color = _seed_for_band(g, top, bottom)
        out[seed_r, seed_c] = BLACK

        mask = _pattern_mask(seed_color, top, bottom, width)
        band_out = out[top:bottom, :]
        eligible = (g[top:bottom, :] == BLACK) & mask
        band_out[eligible] = fill_color

    return out


def _seed_sequence(count: int) -> list[int]:
    seq: list[int] = []
    while len(seq) < count:
        seq.extend(random.sample(SEED_COLORS, len(SEED_COLORS)))
    return seq[:count]


def _build_candidate() -> np.ndarray:
    band_count = random.randint(3, 5)
    band_heights = [random.randint(1, 5) for _ in range(band_count)]
    height = sum(band_heights) + (band_count - 1)
    width = random.randint(6, 24)

    grid = np.zeros((height, width), dtype=int)
    divider_colors = random.sample(DIVIDER_COLORS, band_count - 1)
    seed_colors = _seed_sequence(band_count)

    separator_rows: list[int] = []
    cursor = 0
    for idx, band_height in enumerate(band_heights[:-1]):
        cursor += band_height
        separator_rows.append(cursor)
        grid[cursor, :] = divider_colors[idx]
        cursor += 1

    starts = [0] + [r + 1 for r in separator_rows]
    stops = separator_rows + [height]
    for seed_color, top, bottom in zip(seed_colors, starts, stops, strict=False):
        seed_r = random.randint(top, bottom - 1)
        seed_c = random.randint(0, width - 1)
        grid[seed_r, seed_c] = seed_color

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        separators = _separator_rows(grid)
        if len(separators) < 2:
            return False
        bands = _band_slices(grid, separators)
        seen_seed_colors = set()
        for top, bottom, _fill_color in bands:
            _r, _c, color = _seed_for_band(grid, top, bottom)
            seen_seed_colors.add(int(color))
        if seen_seed_colors != set(SEED_COLORS):
            return False
        out = main(grid)
    except AssertionError:
        return False

    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
