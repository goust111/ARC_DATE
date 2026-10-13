from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
GREY = 5
MAROON = 9
CONTROL_COLORS = (1, 2, 3, 4, 6, 7, 8)
TILE_SIZES = (5, 7, 9)
SECTORS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
MAX_GENERATION_ATTEMPTS = 100


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if bool(flag) == value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _partition_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_is_separator = np.all(grid == GREY, axis=1)
    col_is_separator = np.all(grid == GREY, axis=0)

    row_spans = _runs(row_is_separator, False)
    col_spans = _runs(col_is_separator, False)
    row_separators = _runs(row_is_separator, True)
    col_separators = _runs(col_is_separator, True)

    assert len(row_spans) == 3 and len(col_spans) == 3, "expected a 3x3 partition"
    assert len(row_separators) == 2 and len(col_separators) == 2, "expected two separator lines each way"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all tiles must share one size"
    return row_spans, col_spans


def _tile_features(tile: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
    grey_coords = np.argwhere(tile == GREY)
    assert grey_coords.shape[0] == 9, "each tile must contain one 3x3 grey block"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (3, 3), "grey block must be 3x3"
    assert np.all(tile[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey block must be solid"

    control_coords = np.argwhere((tile != BACKGROUND) & (tile != GREY))
    assert control_coords.shape[0] == 1, "each tile must contain exactly one control pixel"
    cr, cc = map(int, control_coords[0])
    control_color = int(tile[cr, cc])
    return (int(r0), int(c0), int(r1), int(c1)), (cr, cc), control_color


def _relative_index(coord: tuple[int, int], grey_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = coord
    r0, c0, r1, c1 = grey_bbox
    row_idx = 0 if r < r0 else 1 if r <= r1 else 2
    col_idx = 0 if c < c0 else 1 if c <= c1 else 2
    assert (row_idx, col_idx) != (1, 1), "control pixel cannot lie inside the grey block"
    return row_idx, col_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one cell inside the outlier tile's grey block based on the tile's control-pixel direction.

    Input structure:
    - The grid is split by full grey separator rows and columns into a 3x3 array of equal-sized tiles.
    - Each tile contains a central solid 3x3 grey block on a black background plus exactly one non-grey control pixel outside that block.
    - Across the nine tiles, eight control pixels share one color and one tile has a different control color.

    Rule:
    - Compare the tiles by their control-pixel color and select the unique outlier tile.
    - In that tile, locate the control pixel relative to the 3x3 grey block: above / aligned / below chooses the block's top / middle / bottom row, and left / aligned / right chooses the left / middle / right column.
    - Recolor the selected grey cell maroon and return only the tile's 3x3 grey block as the output.

    Key insight:
    - The task combines a cross-tile comparison to find the anomalous tile with an intra-tile directional readout to choose which grey cell gets recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _partition_spans(g)

    tiles: list[tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int]] = []
    colors: list[int] = []
    for rr in row_spans:
        for cc in col_spans:
            r0, r1 = rr
            c0, c1 = cc
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            grey_bbox, control_coord, control_color = _tile_features(tile)
            tiles.append((tile, grey_bbox, control_coord, control_color))
            colors.append(control_color)

    color_counts = Counter(colors)
    outlier_colors = [color for color, count in color_counts.items() if count == 1]
    common_colors = [color for color, count in color_counts.items() if count == 8]
    assert len(outlier_colors) == 1 and len(common_colors) == 1, "expected one outlier color and one common color"
    outlier_color = outlier_colors[0]
    outlier_index = colors.index(outlier_color)

    tile, grey_bbox, control_coord, _ = tiles[outlier_index]
    row_idx, col_idx = _relative_index(control_coord, grey_bbox)
    gr0, gc0, gr1, gc1 = grey_bbox
    out = tile[gr0 : gr1 + 1, gc0 : gc1 + 1].copy()
    out[row_idx, col_idx] = MAROON
    return out


def _grey_window(tile_size: int) -> tuple[int, int]:
    start = (tile_size - 3) // 2
    return start, start + 2


def _sector_positions(tile_size: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    start, end = _grey_window(tile_size)
    sectors = {sector: [] for sector in SECTORS}
    for r in range(tile_size):
        for c in range(tile_size):
            if start <= r <= end and start <= c <= end:
                continue
            row_idx = 0 if r < start else 1 if r <= end else 2
            col_idx = 0 if c < start else 1 if c <= end else 2
            sectors[(row_idx, col_idx)].append((r, c))
    return sectors


def _build_grid(
    tile_size: int,
    controls: list[tuple[int, int]],
    control_colors: list[int],
) -> np.ndarray:
    grid_size = tile_size * 3 + 2
    grid = np.zeros((grid_size, grid_size), dtype=int)

    sep1 = tile_size
    sep2 = tile_size * 2 + 1
    grid[sep1, :] = GREY
    grid[sep2, :] = GREY
    grid[:, sep1] = GREY
    grid[:, sep2] = GREY

    grey_start, grey_end = _grey_window(tile_size)
    for idx, ((local_r, local_c), color) in enumerate(zip(controls, control_colors, strict=True)):
        tile_r = idx // 3
        tile_c = idx % 3
        base_r = tile_r * (tile_size + 1)
        base_c = tile_c * (tile_size + 1)
        grid[
            base_r + grey_start : base_r + grey_end + 1,
            base_c + grey_start : base_c + grey_end + 1,
        ] = GREY
        grid[base_r + local_r, base_c + local_c] = color

    return grid


def _expected_output(sector: tuple[int, int]) -> np.ndarray:
    out = np.full((3, 3), GREY, dtype=int)
    out[sector[0], sector[1]] = MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_size = int(rng.choice(np.array(TILE_SIZES, dtype=int)))
    sector_map = _sector_positions(tile_size)

    sectors = list(SECTORS)
    extra_sector = sectors[int(rng.integers(len(sectors)))]
    assigned_sectors = sectors + [extra_sector]
    rng.shuffle(assigned_sectors)

    controls = [
        sector_map[sector][int(rng.integers(len(sector_map[sector])))]
        for sector in assigned_sectors
    ]

    color_pair = rng.choice(np.array(CONTROL_COLORS, dtype=int), size=2, replace=False)
    common_color, outlier_color = map(int, color_pair)
    outlier_index = int(rng.integers(9))
    control_colors = [common_color] * 9
    control_colors[outlier_index] = outlier_color

    grid = _build_grid(tile_size, controls, control_colors)
    expected = _expected_output(assigned_sectors[outlier_index])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
