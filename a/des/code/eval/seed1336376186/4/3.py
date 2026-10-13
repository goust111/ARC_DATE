from __future__ import annotations

import random

import numpy as np


SEP_COLOR = 8
TOWER_COLOR = 2
COLOR_COUNT = 10
MAX_GENERATION_ATTEMPTS = 400
GRID_LIMIT = 30


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _equal_spans(bands: list[tuple[int, int]]) -> bool:
    if not bands:
        return False
    heights = {b1 - b0 + 1 for b0, b1 in bands}
    return len(heights) == 1


def _find_separator_color(
    grid: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_map: dict[int, list[int]] = {}
    col_map: dict[int, list[int]] = {}

    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            row_map.setdefault(int(grid[r, 0]), []).append(r)
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_map.setdefault(int(grid[0, c]), []).append(c)

    candidates = sorted(set(row_map) & set(col_map))
    valid = []
    for color in candidates:
        row_bands = _bands(h, row_map[color])
        col_bands = _bands(w, col_map[color])
        if _equal_spans(row_bands) and _equal_spans(col_bands):
            valid.append((color, row_map[color], col_map[color], row_bands, col_bands))

    assert len(valid) == 1, "expected exactly one separator lattice color"
    return valid[0]


def _tower_candidates(cell: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = cell.shape
    out: list[tuple[int, int, int]] = []
    for c in range(w):
        rows = np.flatnonzero(cell[:, c] != bg)
        if rows.size == 0:
            continue
        if rows[-1] != h - 1:
            continue
        if not np.array_equal(rows, np.arange(rows[0], h)):
            continue
        colors = np.unique(cell[rows, c])
        if colors.size != 1:
            continue
        out.append((c, int(colors[0]), int(rows.size)))
    return out


def _extract_cell(cell: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    tower_info = _tower_candidates(cell, bg)
    assert len(tower_info) == 1, "each region must contain exactly one signal tower"
    tower_col, _tower_color, tower_height = tower_info[0]

    mask = cell != bg
    mask[:, tower_col] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "each region must contain one non-empty pattern tile"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    tile = cell[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(tile != bg), "pattern tile must be a filled rectangle"
    assert c1 < tower_col or c0 > tower_col, "tower and tile must occupy distinct columns"
    return tower_height, tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Within each separator-defined row of regions, use signal-tower heights as
    a permutation index for the tiles, then stack the tiles vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    _sep, sep_rows, sep_cols, row_bands, col_bands = _find_separator_color(g)
    assert sep_rows and sep_cols
    region_rows = len(row_bands)
    region_cols = len(col_bands)
    assert region_rows >= 1 and region_cols >= 1

    per_row: list[list[tuple[int, np.ndarray]]] = []
    tile_shape: tuple[int, int] | None = None
    for rb in row_bands:
        row_items: list[tuple[int, np.ndarray]] = []
        for cb in col_bands:
            r0, r1 = rb
            c0, c1 = cb
            tower_height, tile = _extract_cell(g[r0 : r1 + 1, c0 : c1 + 1], bg)
            if tile_shape is None:
                tile_shape = tuple(map(int, tile.shape))
            assert tile.shape == tile_shape, "all pattern tiles must share one size"
            row_items.append((tower_height, tile))
        heights = sorted(height for height, _tile in row_items)
        assert heights == list(range(1, region_cols + 1)), "tower heights must rank each row"
        per_row.append(sorted(row_items, key=lambda item: item[0]))

    assert tile_shape is not None
    tile_h, tile_w = tile_shape
    out_h = region_cols * tile_h
    out_w = region_rows * tile_w + max(0, region_rows - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for out_col_block, row_items in enumerate(per_row):
        c0 = out_col_block * (tile_w + 1)
        for rank, (_height, tile) in enumerate(row_items):
            r0 = rank * tile_h
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    return out


def _random_tile(tile_h: int, tile_w: int, palette: list[int]) -> np.ndarray:
    tile = np.random.choice(palette, size=(tile_h, tile_w)).astype(int)
    if np.unique(tile).size == 1:
        flat = tile.reshape(-1)
        idx = random.randrange(flat.size)
        choices = [c for c in palette if c != int(flat[idx])]
        flat[idx] = random.choice(choices)
    return tile


def _uniform_rows_and_cols(grid: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    h, w = grid.shape
    row_map: dict[int, list[int]] = {}
    col_map: dict[int, list[int]] = {}
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            row_map.setdefault(int(grid[r, 0]), []).append(r)
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_map.setdefault(int(grid[0, c]), []).append(c)
    return row_map, col_map


def _build_expected_output(
    tiles: list[list[np.ndarray]],
    heights: list[list[int]],
    bg: int,
) -> np.ndarray:
    region_rows = len(tiles)
    region_cols = len(tiles[0])
    tile_h, tile_w = tiles[0][0].shape
    out = np.full(
        (region_cols * tile_h, region_rows * tile_w + max(0, region_rows - 1)),
        bg,
        dtype=int,
    )
    for rr in range(region_rows):
        ordered = sorted(
            zip(heights[rr], tiles[rr]),
            key=lambda item: item[0],
        )
        c0 = rr * (tile_w + 1)
        for i, (_height, tile) in enumerate(ordered):
            r0 = i * tile_h
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return out


def _sample_params() -> tuple[int, int, int, int] | None:
    region_rows = random.randint(2, 4)
    region_cols = random.randint(2, 4)
    tile_h = random.randint(2, 4)
    tile_w = random.randint(2, 4)

    cell_h = tile_h + region_cols
    cell_w = tile_w + 1
    grid_h = region_rows * cell_h + (region_rows - 1)
    grid_w = region_cols * cell_w + (region_cols - 1)
    if grid_h > GRID_LIMIT or grid_w > GRID_LIMIT:
        return None
    return region_rows, region_cols, tile_h, tile_w


def _candidate_colors(bg: int) -> list[int]:
    return [c for c in range(COLOR_COUNT) if c not in {bg, SEP_COLOR, TOWER_COLOR}]


def _enough_unique_tiles(tile_h: int, tile_w: int, palette_size: int, need: int) -> bool:
    area = tile_h * tile_w
    return palette_size ** area - palette_size >= need


def _gen_attempt() -> np.ndarray | None:
    params = _sample_params()
    if params is None:
        return None
    region_rows, region_cols, tile_h, tile_w = params
    cell_h = tile_h + region_cols
    cell_w = tile_w + 1
    grid_h = region_rows * cell_h + (region_rows - 1)
    grid_w = region_cols * cell_w + (region_cols - 1)

    bg = random.choice([c for c in range(COLOR_COUNT) if c not in {SEP_COLOR, TOWER_COLOR}])
    palette = _candidate_colors(bg)
    total_tiles = region_rows * region_cols
    valid_palette_sizes = [
        size
        for size in range(2, len(palette) + 1)
        if _enough_unique_tiles(tile_h, tile_w, size, total_tiles)
    ]
    if not valid_palette_sizes:
        return None
    palette_size = random.choice(valid_palette_sizes)
    tile_palette = random.sample(palette, palette_size)

    tile_bank: list[np.ndarray] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(tile_bank) < total_tiles and attempts < 5000:
        attempts += 1
        tile = _random_tile(tile_h, tile_w, tile_palette)
        key = tile.tobytes()
        if key in seen:
            continue
        seen.add(key)
        tile_bank.append(tile)
    if len(tile_bank) != total_tiles:
        return None
    random.shuffle(tile_bank)

    heights = [random.sample(list(range(1, region_cols + 1)), region_cols) for _ in range(region_rows)]
    tiles: list[list[np.ndarray]] = []
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for rr in range(region_rows - 1):
        sep_r = (rr + 1) * cell_h + rr
        grid[sep_r, :] = SEP_COLOR
    for cc in range(region_cols - 1):
        sep_c = (cc + 1) * cell_w + cc
        grid[:, sep_c] = SEP_COLOR

    idx = 0
    for rr in range(region_rows):
        row_tiles: list[np.ndarray] = []
        r0 = rr * (cell_h + 1)
        for cc in range(region_cols):
            c0 = cc * (cell_w + 1)
            tile = tile_bank[idx]
            idx += 1
            row_tiles.append(tile)

            tower_on_left = random.choice([False, True])
            tower_col = c0 if tower_on_left else c0 + cell_w - 1
            tile_c0 = c0 + 1 if tower_on_left else c0
            grid[r0 : r0 + tile_h, tile_c0 : tile_c0 + tile_w] = tile

            tower_height = heights[rr][cc]
            tower_r0 = r0 + cell_h - tower_height
            grid[tower_r0 : r0 + cell_h, tower_col] = TOWER_COLOR
        tiles.append(row_tiles)

    expected = _build_expected_output(tiles, heights, bg)

    row_map, col_map = _uniform_rows_and_cols(grid)
    if row_map != {SEP_COLOR: [(i + 1) * cell_h + i for i in range(region_rows - 1)]}:
        return None
    if col_map != {SEP_COLOR: [(i + 1) * cell_w + i for i in range(region_cols - 1)]}:
        return None

    counts = np.bincount(grid.ravel(), minlength=COLOR_COUNT)
    if counts[bg] <= max(int(counts[c]) for c in range(COLOR_COUNT) if c != bg):
        return None
    if len(np.unique(grid)) < 4:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
