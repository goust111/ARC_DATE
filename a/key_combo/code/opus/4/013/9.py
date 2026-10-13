from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
DIVIDER = 9
NON_SPECIAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_ATTEMPTS = 400


def _most_common_excluding(grid: np.ndarray, excluded: set[int]) -> int:
    vals = [int(v) for v in grid.flat if int(v) not in excluded]
    assert vals, "expected at least one non-excluded color"
    return Counter(vals).most_common(1)[0][0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_divider_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = set()
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            candidates.add(int(grid[r, 0]))
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            candidates.add(int(grid[0, c]))
    assert len(candidates) == 1, f"expected one divider color, found {sorted(candidates)}"
    return next(iter(candidates))


def _bands(size: int, line_indices: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(line_indices):
        assert start < idx, "divider lines must leave a nonempty region"
        spans.append((start, idx))
        start = idx + 1
    assert start < size, "divider lines must leave a nonempty trailing region"
    spans.append((start, size))
    return spans


def _guide_spec(tile: np.ndarray, bg: int, divider: int) -> tuple[str, int]:
    n = tile.shape[0]
    row_guides = [
        r
        for r in range(n)
        if int(tile[r, 0]) not in {bg, divider} and np.all(tile[r, :] == tile[r, 0])
    ]
    col_guides = [
        c
        for c in range(n)
        if int(tile[0, c]) not in {bg, divider} and np.all(tile[:, c] == tile[0, c])
    ]
    assert len(row_guides) + len(col_guides) == 1, "expected exactly one guide row or column"
    if row_guides:
        return "row", int(row_guides[0])
    return "col", int(col_guides[0])


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_color(g)
    bg = _most_common_excluding(g, {divider})

    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == divider)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == divider)]
    row_bands = _bands(g.shape[0], row_lines)
    col_bands = _bands(g.shape[1], col_lines)

    tile_region: tuple[int, int, np.ndarray] | None = None
    for block_r, (r0, r1) in enumerate(row_bands):
        for block_c, (c0, c1) in enumerate(col_bands):
            patch = g[r0:r1, c0:c1]
            fg_mask = (patch != bg) & (patch != divider)
            if fg_mask.any():
                assert tile_region is None, "expected exactly one non-background region"
                tile_region = (block_r, block_c, patch)

    assert tile_region is not None, "expected one region containing the tile"
    region_r, region_c, region_patch = tile_region
    tile_mask = (region_patch != bg) & (region_patch != divider)
    tile_box = _bbox(tile_mask)
    assert tile_box is not None
    tile = _crop_bbox(region_patch, tile_box)
    assert tile.shape[0] == tile.shape[1] and tile.shape[0] >= 1, "tile must be square"

    orientation, guide_index = _guide_spec(tile, bg, divider)
    if orientation == "row":
        assert guide_index < len(row_bands), "guide row index must fit partition row count"
    else:
        assert guide_index < len(col_bands), "guide column index must fit partition column count"

    return {
        "bg": int(bg),
        "divider": int(divider),
        "rows": int(len(row_bands)),
        "cols": int(len(col_bands)),
        "tile": tile,
        "orientation": orientation,
        "guide_index": int(guide_index),
        "tile_region": (int(region_r), int(region_c)),
    }


def _render_output(rows: int, cols: int, tile: np.ndarray, orientation: str, guide_index: int) -> np.ndarray:
    n = int(tile.shape[0])
    out = np.zeros((rows * n, cols * n), dtype=int)
    if orientation == "row":
        out[guide_index * n : (guide_index + 1) * n, :] = np.tile(tile, (1, cols))
    else:
        out[:, guide_index * n : (guide_index + 1) * n] = np.tile(tile, (rows, 1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count divider-defined regions to size the canvas, then tile the extracted square into the guide-selected output band.

    Input structure:
    - A non-black background is partitioned by one-color full rows and full columns into an `R x C` grid of rectangular regions.
    - Exactly one region contains a square multicolor tile; the other regions are pure background.
    - Inside that tile, exactly one full row or one full column is a solid non-background guide line.

    Rule:
    - Identify the divider color from the full rows and columns, then count region rows `R` and region columns `C`.
    - Locate the unique non-background region and crop its square tile from the surrounding background.
    - Read the tile's unique solid guide row or guide column and keep its 0-indexed position `g`.
    - Build a black output canvas of `R x C` tile blocks, each block having the tile's own size.
    - If the guide is a row, copy the tile into every block across output block row `g`; if the guide is a column, copy it into every block down output block column `g`.

    Key insight:
    - The macro partition count and the micro guide-line index are both essential: one fixes the enlarged canvas dimensions, and the other selects the only active tile band inside that canvas.
    """
    meta = _parse_scene(grid)
    return _render_output(
        int(meta["rows"]),
        int(meta["cols"]),
        np.asarray(meta["tile"], dtype=int),
        str(meta["orientation"]),
        int(meta["guide_index"]),
    )


def _guide_from_tile(tile: np.ndarray) -> tuple[str, int] | None:
    try:
        return _guide_spec(tile, bg=-1, divider=-2)
    except AssertionError:
        return None


def _sample_tile(
    n: int,
    orientation: str,
    guide_index: int,
    palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    assert len(palette) >= 3
    guide_color = rng.choice(palette)
    fill_colors = [c for c in palette if c != guide_color]
    for _ in range(300):
        tile = np.array(
            [[rng.choice(fill_colors) for _ in range(n)] for _ in range(n)],
            dtype=int,
        )
        if orientation == "row":
            tile[guide_index, :] = guide_color
        else:
            tile[:, guide_index] = guide_color
        if len(set(map(int, tile.flat))) < 3:
            continue
        if _guide_from_tile(tile) != (orientation, guide_index):
            continue
        return tile
    raise RuntimeError("failed to build a tile with a unique guide")


def _rand_partition_sizes(
    count: int,
    minimum: int,
    maximum: int,
    forced_index: int,
    forced_minimum: int,
    rng: random.Random,
) -> list[int]:
    sizes = [rng.randint(minimum, maximum) for _ in range(count)]
    sizes[forced_index] = rng.randint(forced_minimum, maximum)
    return sizes


def _build_input(
    rows: int,
    cols: int,
    bg: int,
    tile: np.ndarray,
    tile_region: tuple[int, int],
    row_sizes: list[int],
    col_sizes: list[int],
    rng: random.Random,
) -> np.ndarray:
    height = sum(row_sizes) + rows - 1
    width = sum(col_sizes) + cols - 1
    grid = np.full((height, width), bg, dtype=int)

    r_cursor = 0
    for r in range(rows - 1):
        r_cursor += row_sizes[r]
        grid[r_cursor, :] = DIVIDER
        r_cursor += 1

    c_cursor = 0
    for c in range(cols - 1):
        c_cursor += col_sizes[c]
        grid[:, c_cursor] = DIVIDER
        c_cursor += 1

    target_r, target_c = tile_region
    region_top = sum(row_sizes[:target_r]) + target_r
    region_left = sum(col_sizes[:target_c]) + target_c
    region_h = row_sizes[target_r]
    region_w = col_sizes[target_c]
    tile_h, tile_w = tile.shape
    assert tile_h <= region_h and tile_w <= region_w

    offset_r = rng.randint(0, region_h - tile_h)
    offset_c = rng.randint(0, region_w - tile_w)
    grid[region_top + offset_r : region_top + offset_r + tile_h, region_left + offset_c : region_left + offset_c + tile_w] = tile
    return grid


def _expected_output_from_parts(rows: int, cols: int, tile: np.ndarray, orientation: str, guide_index: int) -> np.ndarray:
    return _render_output(rows, cols, tile, orientation, guide_index)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    orientation = rng.choice(["row", "col"])
    min_n = max(3, rows if orientation == "row" else cols)
    n = rng.randint(min_n, 4)
    guide_index = rng.randrange(rows if orientation == "row" else cols)

    bg = rng.choice(NON_SPECIAL_COLORS)
    palette = [c for c in NON_SPECIAL_COLORS if c != bg]
    rng.shuffle(palette)
    tile_palette = palette[: rng.randint(3, min(4, len(palette)))]
    tile = _sample_tile(n, orientation, guide_index, tile_palette, rng)

    tile_region = (rng.randrange(rows), rng.randrange(cols))
    row_sizes = _rand_partition_sizes(rows, 3, 6, tile_region[0], n, rng)
    col_sizes = _rand_partition_sizes(cols, 3, 6, tile_region[1], n, rng)
    grid = _build_input(rows, cols, bg, tile, tile_region, row_sizes, col_sizes, rng)

    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None

    expected = _expected_output_from_parts(rows, cols, tile, orientation, guide_index)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    try:
        meta = _parse_scene(grid)
        actual = main(grid)
    except AssertionError:
        return None

    if tuple(map(int, meta["tile_region"])) != tuple(map(int, tile_region)):
        return None
    if int(meta["rows"]) != rows or int(meta["cols"]) != cols:
        return None
    if str(meta["orientation"]) != orientation or int(meta["guide_index"]) != guide_index:
        return None
    if not np.array_equal(np.asarray(meta["tile"], dtype=int), tile):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
