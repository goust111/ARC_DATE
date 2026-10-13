from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER_SIZE = 2
GRID_MIN = 10
GRID_MAX = 18
EDGE_MARGIN = 3
MAX_GENERATION_ATTEMPTS = 300
NON_BLACK_COLORS = tuple(range(1, 10))


def _find_marker_block(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette ids"

    coords = np.argwhere(g != BACKGROUND)
    assert len(coords) == 4, "expected exactly four non-background cells"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1) == int(r0) + 1 and int(c1) == int(c0) + 1, "marker must be a 2x2 block"

    block = g[int(r0) : int(r0) + MARKER_SIZE, int(c0) : int(c0) + MARKER_SIZE]
    assert np.all(block != BACKGROUND), "marker block must be fully colored"
    assert len({int(v) for v in block.flat}) == 4, "marker colors must be distinct"
    return int(r0), int(c0), block.copy()


def _region_mask(shape: tuple[int, int], r0: int, c0: int) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (
        ((rows < r0) & (cols < c0))
        | ((rows < r0) & (cols > c0 + 1))
        | ((rows > r0 + 1) & (cols < c0))
        | ((rows > r0 + 1) & (cols > c0 + 1))
    )


def _global_tile(block: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % 2
    cols = np.arange(shape[1])[None, :] % 2
    return block[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the four diagonal regions around a 2x2 marker block with the marker's own periodic tile.

    Input structure:
    - A black background contains exactly one 2x2 marker block.
    - The marker block's four cells are non-black and pairwise distinct.
    - Every non-marker cell is black.

    Rule:
    - Locate the marker block and keep it unchanged.
    - Define the four diagonal regions strictly outside the marker's rows and columns.
    - Use the diagonally opposite marker cell to select each region:
      bottom-right -> top-left, bottom-left -> top-right, top-right -> bottom-left, top-left -> bottom-right.
    - Fill every cell in those regions with the original 2x2 marker block, repeated without rotation
      using the global coordinate phase `(row mod 2, col mod 2)`.
    - Leave the cross-shaped band sharing the marker's two rows or two columns black.

    Key insight:
    - The marker block is both the positional legend for the diagonal regions and the exact 2x2 tile
      whose absolute parity pattern must be repeated across those regions.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, block = _find_marker_block(g)
    assert np.count_nonzero(g[:r0, :]) == 0
    assert np.count_nonzero(g[r0 + 2 :, :]) == 0
    assert np.count_nonzero(g[r0 : r0 + 2, :c0]) == 0
    assert np.count_nonzero(g[r0 : r0 + 2, c0 + 2 :]) == 0

    out = np.zeros_like(g)
    tiled = _global_tile(block, g.shape)
    rows = np.arange(g.shape[0])[:, None]
    cols = np.arange(g.shape[1])[None, :]
    region_specs = (
        (int(block[1, 1]), (rows < r0) & (cols < c0)),
        (int(block[1, 0]), (rows < r0) & (cols > c0 + 1)),
        (int(block[0, 1]), (rows > r0 + 1) & (cols < c0)),
        (int(block[0, 0]), (rows > r0 + 1) & (cols > c0 + 1)),
    )
    for _opposite_color, mask in region_specs:
        out[mask] = tiled[mask]
    out[r0 : r0 + MARKER_SIZE, c0 : c0 + MARKER_SIZE] = block
    return out


def _fill_with_marker_relative_phase(grid: np.ndarray) -> np.ndarray:
    r0, c0, block = _find_marker_block(grid)
    rows = (np.arange(grid.shape[0])[:, None] - r0) % 2
    cols = (np.arange(grid.shape[1])[None, :] - c0) % 2
    out = np.zeros_like(grid)
    mask = _region_mask(grid.shape, r0, c0)
    tiled = block[rows, cols]
    out[mask] = tiled[mask]
    out[r0 : r0 + 2, c0 : c0 + 2] = block
    return out


def _fill_with_local_region_phase(grid: np.ndarray) -> np.ndarray:
    r0, c0, block = _find_marker_block(grid)
    h, w = grid.shape
    out = np.zeros_like(grid)
    out[r0 : r0 + 2, c0 : c0 + 2] = block

    def paint(rs: int, re: int, cs: int, ce: int, row_anchor: int, col_anchor: int) -> None:
        if rs >= re or cs >= ce:
            return
        rows = (np.arange(rs, re)[:, None] - row_anchor) % 2
        cols = (np.arange(cs, ce)[None, :] - col_anchor) % 2
        out[rs:re, cs:ce] = block[rows, cols]

    paint(0, r0, 0, c0, 0, 0)
    paint(0, r0, c0 + 2, w, 0, c0 + 2)
    paint(r0 + 2, h, 0, c0, r0 + 2, 0)
    paint(r0 + 2, h, c0 + 2, w, r0 + 2, c0 + 2)
    return out


def _fill_with_solid_opposites(grid: np.ndarray) -> np.ndarray:
    r0, c0, block = _find_marker_block(grid)
    out = np.zeros_like(grid)
    out[:r0, :c0] = int(block[1, 1])
    out[:r0, c0 + 2 :] = int(block[1, 0])
    out[r0 + 2 :, :c0] = int(block[0, 1])
    out[r0 + 2 :, c0 + 2 :] = int(block[0, 0])
    out[r0 : r0 + 2, c0 : c0 + 2] = block
    return out


def _region_slices(r0: int, c0: int) -> tuple[tuple[slice, slice], ...]:
    return (
        (slice(0, r0), slice(0, c0)),
        (slice(0, r0), slice(c0 + 2, None)),
        (slice(r0 + 2, None), slice(0, c0)),
        (slice(r0 + 2, None), slice(c0 + 2, None)),
    )


def _is_informative_example(grid: np.ndarray) -> bool:
    r0, c0, block = _find_marker_block(grid)
    if min(r0, c0, grid.shape[0] - (r0 + 2), grid.shape[1] - (c0 + 2)) < EDGE_MARGIN:
        return False

    out = main(grid)
    colors = {int(v) for v in block.flat}
    for rs, cs in _region_slices(r0, c0):
        region = out[rs, cs]
        if min(region.shape) < 2:
            return False
        if {int(v) for v in np.unique(region)} != colors:
            return False

    wrong_variants = (
        _fill_with_marker_relative_phase(grid),
        _fill_with_local_region_phase(grid),
        _fill_with_solid_opposites(grid),
    )
    return all(not np.array_equal(out, wrong) for wrong in wrong_variants)


def _sample_dimension_and_position() -> tuple[int, int, int, int]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    valid_positions = [
        (r0, c0)
        for r0 in range(EDGE_MARGIN, h - EDGE_MARGIN - 1)
        for c0 in range(EDGE_MARGIN, w - EDGE_MARGIN - 1)
        if min(r0, c0, h - (r0 + 2), w - (c0 + 2)) >= EDGE_MARGIN
    ]
    assert valid_positions, "grid size must admit a marker with visible diagonal regions"
    r0, c0 = random.choice(valid_positions)
    return h, w, r0, c0


def _generate_candidate() -> np.ndarray:
    h, w, r0, c0 = _sample_dimension_and_position()
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(NON_BLACK_COLORS, 4)
    block = np.array(colors, dtype=int).reshape(2, 2)
    grid[r0 : r0 + 2, c0 : c0 + 2] = block
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if _is_informative_example(grid):
            return grid
    raise RuntimeError("failed to generate a non-degenerate input")
