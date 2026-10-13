from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRID_SIZE = 4
BLOCK_SIZE = 4
OUTPUT_SIZE = GRID_SIZE * BLOCK_SIZE
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

QUADRANT_CELLS = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 2), (0, 3), (1, 2), (1, 3)),
    ((2, 0), (2, 1), (3, 0), (3, 1)),
    ((2, 2), (2, 3), (3, 2), (3, 3)),
)


def _quadrant_turns(r: int, c: int) -> int:
    if r < 2 and c < 2:
        return 2
    if r < 2 and c >= 2:
        return -1
    if r >= 2 and c < 2:
        return 1
    return 0


def _rot(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, turns)


def _stamp_output(grid: np.ndarray, default_turns: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    stamp_mask = g == BLACK
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            color = int(g[r, c])
            if color == BLACK:
                continue
            turns = default_turns if default_turns is not None else _quadrant_turns(r, c)
            block = _rot(stamp_mask, turns)
            rs = slice(r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE)
            cs = slice(c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE)
            out[rs, cs][block] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 4x4 pattern by stamping the whole-grid black mask into each colored cell's block with a quadrant-dependent rotation.

    Input structure:
    - The input is a 4x4 grid with black background cells.
    - Non-black pixels may use multiple colors and can appear in any of the four 2x2 quadrants.

    Rule:
    - Take the black-cell positions of the entire 4x4 input as a 4x4 stamp template.
    - Produce a 16x16 output split into 4x4 blocks; each input pixel controls the block at the same coarse position.
    - For a non-black input pixel, stamp the template in that block using the input pixel's color:
      upper-left quadrant pixels use the template rotated 180 degrees, upper-right use 90 degrees clockwise,
      lower-left use 90 degrees counterclockwise, and lower-right use the unrotated template.
    - For a black input pixel, leave the corresponding 4x4 output block completely black.

    Key insight:
    - The input serves simultaneously as the placement map, the complement-derived stamp shape, and the source of each block's rotation through the pixel's quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 4x4 input grid"
    assert np.any(g == BLACK), "expected at least one black cell for the stamp mask"
    assert np.any(g != BLACK), "expected at least one colored cell to stamp"
    return _stamp_output(g)


def _all_quadrants_used(grid: np.ndarray) -> bool:
    return all(any(int(grid[r, c]) != BLACK for r, c in cells) for cells in QUADRANT_CELLS)


def _rotations_are_distinct(mask: np.ndarray) -> bool:
    signatures = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(signatures) == 4


def _sample_foreground_cells() -> list[tuple[int, int]]:
    counts = [1, 1, 1, 1]
    for _ in range(random.randint(1, 4)):
        choices = [idx for idx, count in enumerate(counts) if count < 4]
        counts[random.choice(choices)] += 1

    cells: list[tuple[int, int]] = []
    for quadrant_cells, count in zip(QUADRANT_CELLS, counts):
        cells.extend(random.sample(list(quadrant_cells), count))
    random.shuffle(cells)
    return cells


def _assign_colors(cell_count: int) -> list[int]:
    distinct = random.randint(2, min(4, cell_count))
    palette = random.sample(PALETTE, distinct)
    colors = palette[:]
    colors.extend(random.choice(palette) for _ in range(cell_count - distinct))
    random.shuffle(colors)
    return colors


def _reconstruct_input(output: np.ndarray) -> np.ndarray:
    out = np.asarray(output, dtype=int)
    assert out.shape == (OUTPUT_SIZE, OUTPUT_SIZE)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    recovered_stamp: np.ndarray | None = None

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            block = out[r * BLOCK_SIZE : (r + 1) * BLOCK_SIZE, c * BLOCK_SIZE : (c + 1) * BLOCK_SIZE]
            colors = [int(color) for color in np.unique(block) if int(color) != BLACK]
            if not colors:
                continue
            assert len(colors) == 1, "a stamped block must use exactly one non-black color"
            grid[r, c] = colors[0]

            block_mask = block != BLACK
            stamp_mask = _rot(block_mask, -_quadrant_turns(r, c))
            if recovered_stamp is None:
                recovered_stamp = stamp_mask
            else:
                assert np.array_equal(recovered_stamp, stamp_mask), "all blocks must encode the same base stamp"

    assert recovered_stamp is not None, "expected at least one stamped block"
    assert np.array_equal(recovered_stamp, grid == BLACK), "output must uniquely encode the input black mask"
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIZE, GRID_SIZE):
        return False
    fg_mask = g != BLACK
    stamp_mask = g == BLACK
    if not np.any(fg_mask) or not np.any(stamp_mask):
        return False
    if not _all_quadrants_used(g):
        return False
    if len(np.unique(g[fg_mask])) < 2:
        return False
    if not _rotations_are_distinct(stamp_mask):
        return False
    if any(np.array_equal(stamp_mask, np.rot90(fg_mask, k)) for k in range(4)):
        return False

    actual = main(g)
    if any(np.array_equal(actual, _stamp_output(g, default_turns=k)) for k in (0, 1, 2, 3)):
        return False

    try:
        recovered = _reconstruct_input(actual)
    except AssertionError:
        return False
    return np.array_equal(recovered, g)


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    cells = _sample_foreground_cells()
    colors = _assign_colors(len(cells))
    for (r, c), color in zip(cells, colors):
        grid[r, c] = color

    if _validate_generated_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for quadrant-complement stamping")
