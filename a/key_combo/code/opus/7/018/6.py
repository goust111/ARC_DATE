import random
from typing import Sequence

import numpy as np


BG = 0
COLUMNS = 4
BLOCK_SIZES = (3, 4)
MAX_GENERATION_ATTEMPTS = 300
COLOR_POOL = tuple(range(1, 10))


def _compress_partitioned_grid(grid: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    assert w % COLUMNS == 0, "width must be split into 4 equal block columns"
    block_size = w // COLUMNS
    assert block_size in BLOCK_SIZES, "block size must be 3 or 4"
    assert h % block_size == 0, "height must be a multiple of the block size"

    rows = h // block_size
    assert rows >= 3, "need at least three compressed rows"

    compressed = np.zeros((rows, COLUMNS), dtype=int)
    rel_positions: list[tuple[int, int]] = []

    for r_block in range(rows):
        for c_block in range(COLUMNS):
            r0 = r_block * block_size
            c0 = c_block * block_size
            block = g[r0 : r0 + block_size, c0 : c0 + block_size]
            coords = np.argwhere(block != BG)
            assert coords.shape[0] == 1, "each block must contain exactly one foreground pixel"
            rr, cc = map(int, coords[0])
            compressed[r_block, c_block] = int(block[rr, cc])
            if r_block == 0:
                rel_positions.append((rr, cc))
            else:
                assert rel_positions[c_block] == (rr, cc), "block-internal positions must be column-fixed"

    return block_size, compressed, rel_positions


def _validate_shift_chain(compressed: np.ndarray) -> None:
    for i in range(compressed.shape[0] - 1):
        assert np.array_equal(np.roll(compressed[i], -1), compressed[i + 1]), (
            "each compressed row must be the previous row shifted left by one"
        )


def _expand_prediction(
    predicted_row: Sequence[int], rel_positions: Sequence[tuple[int, int]], block_size: int
) -> np.ndarray:
    out = np.zeros((block_size, COLUMNS * block_size), dtype=int)
    for col, color in enumerate(predicted_row):
        rr, cc = rel_positions[col]
        out[rr, col * block_size + cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a sparse partitioned grid, predict the next cyclic row, and re-expand it as one block-row.

    Input structure:
    - The background is black.
    - The width consists of 4 equal block columns, with block size 3 or 4.
    - Each block contains exactly one non-background pixel, and within each block column that pixel keeps the same block-internal coordinate across all block rows.
    - After replacing every block by its lone pixel color, the compressed rows form a sequence where each next row is a one-step cyclic left shift of the previous row.

    Rule:
    - Infer the block size from the 4-column partition and compress the grid to one color per block.
    - Record the unique block-internal foreground coordinate for each of the 4 block columns.
    - Predict the next compressed row by cyclically left-shifting the last compressed row by one position.
    - Produce a single block-row output of height equal to the block size; in block column j, place the predicted color at that column's recorded block-internal coordinate and leave all other cells black.

    Key insight:
    - The cyclic pattern is hidden at the coarse block level, so the grid must be compressed before the left-shift rule becomes visible, then the predicted coarse row must be unfolded back to block scale.
    """
    block_size, compressed, rel_positions = _compress_partitioned_grid(grid)
    _validate_shift_chain(compressed)
    predicted = np.roll(compressed[-1], -1)
    return _expand_prediction(predicted, rel_positions, block_size)


def _sample_row_count(block_size: int) -> int:
    if block_size == 3:
        return random.choice((4, 5, 6))
    return random.choice((3, 4, 5, 6))


def _sample_positions(block_size: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(block_size) for c in range(block_size)]
    for _ in range(100):
        picks = random.sample(cells, COLUMNS)
        rows = {r for r, _ in picks}
        cols = {c for _, c in picks}
        if len(rows) >= 2 and len(cols) >= 2:
            return picks
    raise RuntimeError("failed to sample non-degenerate block positions")


def _build_input(
    block_size: int,
    row_count: int,
    base_colors: Sequence[int],
    rel_positions: Sequence[tuple[int, int]],
    start_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    compressed = np.zeros((row_count, COLUMNS), dtype=int)
    grid = np.zeros((row_count * block_size, COLUMNS * block_size), dtype=int)

    base = np.asarray(base_colors, dtype=int)
    for r_block in range(row_count):
        row = np.roll(base, -(start_shift + r_block))
        compressed[r_block] = row
        for c_block, color in enumerate(row):
            rr, cc = rel_positions[c_block]
            grid[r_block * block_size + rr, c_block * block_size + cc] = int(color)

    return grid, compressed


def _validate_generated_case(
    grid: np.ndarray,
    compressed_expected: np.ndarray,
    rel_positions_expected: Sequence[tuple[int, int]],
) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(np.count_nonzero(grid == BG)) > int(np.count_nonzero(grid != BG)), "black must stay dominant"

    block_size, compressed_actual, rel_positions_actual = _compress_partitioned_grid(grid)
    assert np.array_equal(compressed_actual, compressed_expected)
    assert list(rel_positions_actual) == list(rel_positions_expected)
    _validate_shift_chain(compressed_actual)

    # Anti-degeneracy checks for the generated family.
    assert len({tuple(row) for row in compressed_actual.tolist()}) >= 3, "need multiple visible cycle steps"
    assert len(set(map(int, compressed_actual[0].tolist()))) == COLUMNS, "use four distinct visible colors"
    assert len(set(rel_positions_actual)) == COLUMNS, "each block column should carry its own visible coordinate"
    assert len({r for r, _ in rel_positions_actual}) >= 2, "coordinates should span multiple block rows"
    assert len({c for _, c in rel_positions_actual}) >= 2, "coordinates should span multiple block cols"

    predicted = np.roll(compressed_actual[-1], -1)
    assert not np.array_equal(predicted, compressed_actual[-1]), "prediction must change the final row"

    out = main(grid)
    expected_out = _expand_prediction(predicted, rel_positions_actual, block_size)
    assert np.array_equal(out, expected_out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        block_size = random.choice(BLOCK_SIZES)
        row_count = _sample_row_count(block_size)
        base_colors = random.sample(COLOR_POOL, COLUMNS)
        rel_positions = _sample_positions(block_size)
        start_shift = random.randrange(COLUMNS)

        grid, compressed = _build_input(
            block_size=block_size,
            row_count=row_count,
            base_colors=base_colors,
            rel_positions=rel_positions,
            start_shift=start_shift,
        )
        try:
            _validate_generated_case(grid, compressed, rel_positions)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
