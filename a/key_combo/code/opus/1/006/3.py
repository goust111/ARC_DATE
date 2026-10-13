from __future__ import annotations

import random

import numpy as np


GRID_SIZE = 11
BLACK = 0
BLUE = 1
MARK = 5
OUTER_BASE_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
OUTER_TILE_COORDS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
MAX_GENERATION_ATTEMPTS = 300


def _bands_from_separators(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        bands.append((start, sep))
        start = sep + 1
    bands.append((start, limit))
    return bands


def _tile_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected an 11x11 grid"

    row_seps = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == BLACK)]
    col_seps = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)]
    assert row_seps == [3, 7] and col_seps == [3, 7], "expected 3x3 tiles split by black rows and columns"

    row_bands = _bands_from_separators(row_seps, g.shape[0])
    col_bands = _bands_from_separators(col_seps, g.shape[1])
    assert len(row_bands) == 3 and len(col_bands) == 3
    assert all(r1 - r0 == 3 for r0, r1 in row_bands)
    assert all(c1 - c0 == 3 for c0, c1 in col_bands)
    return row_bands, col_bands


def _center_and_palette(grid: np.ndarray) -> tuple[tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands = _tile_layout(g)

    cr0, cr1 = row_bands[1]
    cc0, cc1 = col_bands[1]
    center_tile = g[cr0:cr1, cc0:cc1]
    non_black = np.argwhere(center_tile != BLACK)
    assert non_black.shape[0] == 1, "center tile must contain exactly one non-black marker"
    mr, mc = map(int, non_black[0])
    assert int(center_tile[mr, mc]) == MARK, "center marker must be gray"
    center = (cr0 + mr, cc0 + mc)

    tile_info: list[tuple[int, int]] = []
    for tr, tc in OUTER_TILE_COORDS:
        r0, r1 = row_bands[tr]
        c0, c1 = col_bands[tc]
        tile = g[r0:r1, c0:c1]
        blue_count = int(np.count_nonzero(tile == BLUE))
        assert 1 <= blue_count <= 8, "outer tiles must have between 1 and 8 blue cells"
        base_colors = sorted(int(color) for color in set(np.unique(tile)) - {BLACK, BLUE})
        assert len(base_colors) == 1, "each outer tile must expose exactly one base color"
        tile_info.append((blue_count, base_colors[0]))

    counts = sorted(count for count, _ in tile_info)
    assert counts == list(range(1, 9)), "blue counts must cover 1..8 exactly once"
    tile_info.sort(key=lambda item: item[0])
    palette = [base_color for _, base_color in tile_info]
    return center, palette


def _rings(shape: tuple[int, int], center: tuple[int, int], palette: list[int]) -> np.ndarray:
    rows = np.arange(shape[0], dtype=int)[:, None]
    cols = np.arange(shape[1], dtype=int)[None, :]
    dist = np.abs(rows - int(center[0])) + np.abs(cols - int(center[1]))
    out = np.empty(shape, dtype=int)
    out[dist == 0] = MARK
    if palette:
        out[dist > 0] = np.take(np.asarray(palette, dtype=int), (dist[dist > 0] - 1) % len(palette))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count blue pixels in the outer tiles to order their base colors, then paint Manhattan rings from the center marker with that dynamic palette.

    Input structure:
    - The input is an 11x11 grid split into a 3x3 array of 3x3 tiles by full black separator rows and columns.
    - The center tile contains exactly one gray marker pixel and is otherwise black.
    - Each of the eight outer tiles contains blue foreground pixels plus one visible non-blue, non-black base color; black cells may also appear inside a tile.

    Rule:
    - Count the number of blue pixels in each outer tile, excluding the center tile.
    - Sort the eight outer tiles by that blue count from 1 through 8 and collect their base colors in that order to form an eight-color palette.
    - Produce an 11x11 output grid with no preserved separators: the center marker stays gray, and every other cell is colored by its Manhattan distance from the marker using the dynamic palette cyclically.

    Key insight:
    - The tile array is only an encoding device: blue-count ranking extracts an ordered color palette, and the output discards the tiles entirely in favor of distance rings around the marker.
    """
    center, palette = _center_and_palette(np.asarray(grid, dtype=int))
    return _rings(np.asarray(grid, dtype=int).shape, center, palette)


def _all_tile_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3)]


def _sample_marker_local() -> tuple[int, int]:
    all_cells = _all_tile_cells()
    off_center = [cell for cell in all_cells if cell != (1, 1)]
    if random.random() < 0.7:
        return random.choice(off_center)
    return random.choice(all_cells)


def _sample_black_count(blue_count: int, force_holes: bool) -> int:
    max_black = min(3, 8 - blue_count)
    if max_black <= 0:
        return 0
    if force_holes:
        return random.randint(1, max_black)
    if random.random() < 0.35:
        return random.randint(1, max_black)
    return 0


def _make_outer_tile(base_color: int, blue_count: int, force_holes: bool) -> np.ndarray:
    tile = np.full((3, 3), int(base_color), dtype=int)
    cells = _all_tile_cells()
    blue_cells = random.sample(cells, blue_count)
    black_count = _sample_black_count(blue_count, force_holes)
    remaining = [cell for cell in cells if cell not in blue_cells]
    black_cells = random.sample(remaining, black_count)
    for r, c in blue_cells:
        tile[r, c] = BLUE
    for r, c in black_cells:
        tile[r, c] = BLACK
    return tile


def _expected_from_plan(center: tuple[int, int], palette: list[int]) -> np.ndarray:
    return _rings((GRID_SIZE, GRID_SIZE), center, palette)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[3, :] = BLACK
    grid[7, :] = BLACK
    grid[:, 3] = BLACK
    grid[:, 7] = BLACK

    marker_local = _sample_marker_local()
    center_global = (4 + marker_local[0], 4 + marker_local[1])
    grid[center_global] = MARK

    palette = random.sample(list(OUTER_BASE_COLORS), len(OUTER_BASE_COLORS))
    counts_by_position = random.sample(list(range(1, 9)), len(OUTER_TILE_COORDS))

    hole_candidates = [i for i, count in enumerate(counts_by_position) if count <= 7]
    forced_hole_tiles: set[int] = set()
    if hole_candidates and random.random() < 0.6:
        forced_count = random.randint(1, min(len(hole_candidates), 4))
        forced_hole_tiles = set(random.sample(hole_candidates, forced_count))

    bands = [(0, 3), (4, 7), (8, 11)]
    for idx, ((tr, tc), blue_count) in enumerate(zip(OUTER_TILE_COORDS, counts_by_position)):
        base_color = palette[blue_count - 1]
        tile = _make_outer_tile(base_color, blue_count, idx in forced_hole_tiles)
        r0, r1 = bands[tr]
        c0, c1 = bands[tc]
        grid[r0:r1, c0:c1] = tile

    expected = _expected_from_plan(center_global, palette)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
