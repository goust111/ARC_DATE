from __future__ import annotations

import itertools

import numpy as np


TILE_SIZE = 3
TILE_COUNT = 4
GRID_WIDTH = TILE_SIZE * TILE_COUNT
SPRITE_ROWS = 3
GAP_ROW = 3
BAR_TOP = 4
BAR_HEIGHT = 9
BAR_BOTTOM = BAR_TOP + BAR_HEIGHT

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SPRITE_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
RNG = np.random.default_rng()
MAX_ATTEMPTS = 5000


def _tile_slice(index: int) -> tuple[slice, slice]:
    c0 = TILE_SIZE * index
    return slice(0, SPRITE_ROWS), slice(c0, c0 + TILE_SIZE)


def _tile_center_col(index: int) -> int:
    return TILE_SIZE * index + 1


def _foreground_color(tile: np.ndarray) -> int:
    colors = sorted(set(map(int, tile.flatten())) - {BLACK})
    assert len(colors) == 1, "each sprite tile must use exactly one non-black color"
    return int(colors[0])


def _tile_count_and_color(tile: np.ndarray) -> tuple[int, int]:
    color = _foreground_color(tile)
    count = int(np.count_nonzero(tile != BLACK))
    assert 1 <= count <= BAR_HEIGHT, "sprite foreground count must encode a valid bar depth"
    return count, color


def _validate_input_structure(grid: np.ndarray) -> None:
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = grid.shape
    assert width == GRID_WIDTH, "expected four adjacent 3x3 tiles across the top"
    assert BAR_BOTTOM <= height <= 30, "grid height must fit the full bar section"
    assert np.all(grid[GAP_ROW, :] == BLACK), "row 3 must be the separating empty row"

    bar_cols = {_tile_center_col(i) for i in range(TILE_COUNT)}

    for index in range(TILE_COUNT):
        tile = grid[_tile_slice(index)]
        count, color = _tile_count_and_color(tile)
        assert color in SPRITE_COLORS, "sprite colors must come from the allowed palette"
        assert count > 0

    mid = grid[BAR_TOP:BAR_BOTTOM, :]
    for col in range(width):
        if col in bar_cols:
            assert np.all(mid[:, col] == RED), "each bar must be a solid red vertical segment"
        else:
            assert np.all(mid[:, col] == BLACK), "no extra foreground may appear beside the bars"

    if height > BAR_BOTTOM:
        assert np.all(grid[BAR_BOTTOM:, :] == BLACK), "extra rows below the bars must be empty"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the top part of each red bar to the color encoded by its 3x3 sprite's foreground count.

    Input structure:
    - The grid has black background, four adjacent 3x3 sprite tiles in rows 0-2, and an empty separator row 3.
    - Each sprite tile uses exactly one non-black color from a fixed visible palette and contains at least one foreground pixel.
    - Starting at row 4, the center column below each tile contains a 9-pixel-tall red vertical bar; all other cells in that band are black.

    Rule:
    - Partition the top strip into four 3x3 tiles from left to right.
    - For each tile, count its non-black pixels and read the tile's single foreground color.
    - On the aligned red bar below that tile, recolor the first n cells from the top to the tile's color, where n is the foreground count.
    - Leave the remaining cells of that 9-pixel bar red, and preserve the sprite strip, separator row, and any extra blank rows.

    Key insight:
    - The sprite shape itself is irrelevant except for how many pixels it contains: the count is the only information that transfers to the bar, where it becomes a vertical recolor depth.
    """
    g = np.asarray(grid, dtype=int)
    _validate_input_structure(g)

    out = g.copy()
    for index in range(TILE_COUNT):
        tile = g[_tile_slice(index)]
        count, color = _tile_count_and_color(tile)
        col = _tile_center_col(index)
        out[BAR_TOP : BAR_TOP + count, col] = color
        out[BAR_TOP + count : BAR_BOTTOM, col] = RED
    return out


def _mask_from_count(count: int) -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    chosen = RNG.choice(TILE_SIZE * TILE_SIZE, size=count, replace=False)
    mask.flat[chosen] = True
    return mask


def _bbox_area(mask: np.ndarray) -> int:
    rows, cols = np.where(mask)
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        stack = [(rr, cc)]
        seen[rr, cc] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _has_feature_inversion(counts: list[int], values: list[int]) -> bool:
    for i, j in itertools.combinations(range(TILE_COUNT), 2):
        if counts[i] < counts[j] and values[i] > values[j]:
            return True
        if counts[j] < counts[i] and values[j] > values[i]:
            return True
    return False


def _counts_for_sample() -> list[int]:
    low = int(RNG.choice([1, 2]))
    high = int(RNG.choice([8, 9]))
    remaining_pool = [n for n in range(1, BAR_HEIGHT + 1) if n not in {low, high}]
    middle = RNG.choice(remaining_pool, size=2, replace=False).tolist()
    counts = [low, high, *map(int, middle)]
    RNG.shuffle(counts)
    return [int(v) for v in counts]


def _tile_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full(mask.shape, BLACK, dtype=int)
    tile[mask] = int(color)
    return tile


def _build_expected(grid: np.ndarray, counts: list[int], colors: list[int]) -> np.ndarray:
    out = grid.copy()
    for index, (count, color) in enumerate(zip(counts, colors)):
        col = _tile_center_col(index)
        out[BAR_TOP : BAR_TOP + count, col] = int(color)
        out[BAR_TOP + count : BAR_BOTTOM, col] = RED
    return out


def _semantic_checks(grid: np.ndarray, counts: list[int], colors: list[int], masks: list[np.ndarray]) -> bool:
    if sorted(set(counts)) != sorted(counts):
        return False
    if min(counts) > 2 or max(counts) < 8:
        return False
    if counts == sorted(counts) or counts == sorted(counts, reverse=True):
        return False
    if len(set(colors)) != TILE_COUNT:
        return False
    if sum(mask[1, 1] for mask in masks) in {0, TILE_COUNT}:
        return False
    if len({_component_count(mask) for mask in masks}) < 2:
        return False

    top_row_sums = [int(mask[0, :].sum()) for mask in masks]
    bbox_areas = [_bbox_area(mask) for mask in masks]
    if not _has_feature_inversion(counts, top_row_sums):
        return False
    if not _has_feature_inversion(counts, bbox_areas):
        return False

    try:
        _validate_input_structure(grid)
    except AssertionError:
        return False
    return True


def _attempt() -> np.ndarray | None:
    counts = _counts_for_sample()
    colors = RNG.permutation(np.array(SPRITE_COLORS, dtype=int)).tolist()
    masks = [_mask_from_count(count) for count in counts]

    extra_rows = int(RNG.integers(0, 4))
    grid = np.full((BAR_BOTTOM + extra_rows, GRID_WIDTH), BLACK, dtype=int)
    grid[BAR_TOP:BAR_BOTTOM, [_tile_center_col(i) for i in range(TILE_COUNT)]] = RED

    for index, (mask, color) in enumerate(zip(masks, colors)):
        grid[_tile_slice(index)] = _tile_from_mask(mask, int(color))

    if not _semantic_checks(grid, counts, colors, masks):
        return None

    expected = _build_expected(grid, counts, colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 9")
