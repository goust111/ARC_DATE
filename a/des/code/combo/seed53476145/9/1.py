from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COLORS = (2, 1, 3, 4)  # red, blue, green, yellow
PARITY_ORDER = ((0, 0), (0, 1), (1, 1), (1, 0))
BACKGROUND_COLORS = (BLACK, GREY)


def _checkerboard(height: int, width: int) -> np.ndarray:
    row_tiles = np.arange(height, dtype=int)[:, None] // 2
    col_tiles = np.arange(width, dtype=int)[None, :] // 2
    return np.where((row_tiles + col_tiles) % 2 == 0, BLACK, GREY).astype(int)


def _find_legend(top_row: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    non_bg = np.flatnonzero(~np.isin(top_row, BACKGROUND_COLORS))
    assert len(non_bg) == 4, f"expected four legend pixels, got {len(non_bg)}"
    start = int(non_bg[0])
    assert np.array_equal(non_bg, np.arange(start, start + 4)), "legend must be contiguous"
    return start, tuple(int(v) for v in top_row[start : start + 4])


def _class_to_color(legend: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    return {parity: int(color) for parity, color in zip(PARITY_ORDER, legend)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each singleton marker according to the parity class of its containing 2x2 board cell.

    Input structure:
    - A black/grey checkerboard made of 2x2 tiles, with the top-left tile black.
    - The top row contains a contiguous 4-pixel legend whose colors define the
      recoloring for tile-parity classes.
    - Every other non-background colored cell is a singleton marker placed at
      the center position of some 2x2 tile.

    Rule:
    - Let the tile row/column be r//2 and c//2.
    - Use the legend order:
      even/even, even/odd, odd/odd, odd/even.
    - Recolor each marker to the legend color for its tile class.
    - Leave the checkerboard and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 4 and g.shape[1] >= 4

    legend_start, legend = _find_legend(g[0])
    mapping = _class_to_color(legend)

    out = g.copy()
    marker_mask = ~np.isin(g, BACKGROUND_COLORS)
    marker_mask[0, legend_start : legend_start + 4] = False

    center_rows = (np.arange(g.shape[0]) % 2 == 1)[:, None]
    center_cols = (np.arange(g.shape[1]) % 2 == 1)[None, :]
    marker_mask &= center_rows & center_cols

    for r, c in np.argwhere(marker_mask):
        parity = ((int(r) // 2) % 2, (int(c) // 2) % 2)
        out[int(r), int(c)] = mapping[parity]

    return out


# --------------- generator ---------------

MIN_TILE_ROWS = 3
MAX_TILE_ROWS = 7
MIN_TILE_COLS = 4
MAX_TILE_COLS = 9
MAX_MARKERS = 12
MAX_GENERATION_ATTEMPTS = 300
SOURCE_COLOR_POOL = (1, 2, 3, 4, 6, 8, 9)


def _tile_center(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 2 * tile_r + 1, 2 * tile_c + 1


def _expected_color(tile_r: int, tile_c: int) -> int:
    parity = (tile_r % 2, tile_c % 2)
    idx = PARITY_ORDER.index(parity)
    return int(LEGEND_COLORS[idx])


def _all_tiles(tile_rows: int, tile_cols: int) -> list[tuple[int, int]]:
    return [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols)]


def _tiles_by_class(tile_rows: int, tile_cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups = {parity: [] for parity in PARITY_ORDER}
    for tile in _all_tiles(tile_rows, tile_cols):
        groups[(tile[0] % 2, tile[1] % 2)].append(tile)
    for parity in PARITY_ORDER:
        assert groups[parity], f"missing tiles for parity class {parity}"
    return groups


def _build_expected(grid: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for tile_r, tile_c in markers:
        rr, cc = _tile_center(tile_r, tile_c)
        out[rr, cc] = _expected_color(tile_r, tile_c)
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(MIN_TILE_ROWS, MAX_TILE_ROWS)
    tile_cols = random.randint(MIN_TILE_COLS, MAX_TILE_COLS)
    height, width = 2 * tile_rows, 2 * tile_cols

    grid = _checkerboard(height, width)
    legend_start = random.randint(0, width - 4)
    grid[0, legend_start : legend_start + 4] = LEGEND_COLORS

    by_class = _tiles_by_class(tile_rows, tile_cols)
    required_tiles = [random.choice(by_class[parity]) for parity in PARITY_ORDER]
    used = set(required_tiles)

    total_tiles = tile_rows * tile_cols
    marker_count = random.randint(4, min(MAX_MARKERS, total_tiles))
    extras_needed = marker_count - len(required_tiles)

    remaining = [tile for tile in _all_tiles(tile_rows, tile_cols) if tile not in used]
    extra_tiles = random.sample(remaining, extras_needed) if extras_needed else []
    marker_tiles = required_tiles + extra_tiles

    for tile_r, tile_c in marker_tiles:
        target = _expected_color(tile_r, tile_c)
        source_options = [color for color in SOURCE_COLOR_POOL if int(color) != int(target)]
        marker_color = int(random.choice(source_options))
        rr, cc = _tile_center(tile_r, tile_c)
        grid[rr, cc] = marker_color

    expected = _build_expected(grid, marker_tiles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-parity sample")
