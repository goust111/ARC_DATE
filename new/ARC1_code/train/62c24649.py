from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the grid by reflecting it rightward and then downward.

    Input structure
    The input is a small rectangular color grid.

    Rule
    Form the top half of the output by placing the original grid next to its
    left-right mirror. Form the bottom half by reflecting that top half
    top-to-bottom.

    Key insight
    The output is not a nearest-neighbor scale-up; it is a four-quadrant mirror
    tile whose upper-left quadrant is the original input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0, "input grid must be nonempty"

    top = np.concatenate((g, np.fliplr(g)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)

# --------------- generator (o5x) ---------------

# Input shape.
GEN_GRID_SHAPE = (3, 3)
# Palette bounds and color universe; 0 is always included.
MIN_PALETTE_SIZE = 3
MAX_PALETTE_SIZE = 5
BASE_COLOR = 0
NONZERO_COLORS = np.arange(1, 10)
# Retry ceiling for oracle-checked generation attempts.
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _gen_distinct_palette() -> list[int]:
    color_count = int(np.random.randint(MIN_PALETTE_SIZE, MAX_PALETTE_SIZE + 1))
    nonzero = np.random.choice(NONZERO_COLORS, size=color_count - 1, replace=False)
    palette = [BASE_COLOR] + [int(c) for c in nonzero]
    np.random.shuffle(palette)
    return palette


def _gen_force_asymmetry(grid: np.ndarray, palette: list[int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    a, b = np.random.choice(np.array(palette, dtype=int), size=2, replace=False)

    lr_pairs = [((r, c), (r, w - 1 - c)) for r in range(h) for c in range(w // 2)]
    ud_pairs = [((r, c), (h - 1 - r, c)) for r in range(h // 2) for c in range(w)]
    pair_options = [
        (lr_pair, ud_pair)
        for lr_pair in lr_pairs
        for ud_pair in ud_pairs
        if set(lr_pair).isdisjoint(set(ud_pair))
    ]
    lr_pair, ud_pair = pair_options[int(np.random.randint(len(pair_options)))]

    (r0, c0), (r1, c1) = lr_pair
    grid[r0, c0] = a
    grid[r1, c1] = b
    protected = {(r0, c0), (r1, c1)}

    (r0, c0), (r1, c1) = ud_pair
    grid[r0, c0] = a
    grid[r1, c1] = b
    protected.update({(r0, c0), (r1, c1)})
    return protected


def _fill_unprotected_cells(grid: np.ndarray, palette: list[int], protected: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    present = {int(grid[r, c]) for r, c in protected}
    required = [c for c in palette if c not in present]
    coords = [(r, c) for r in range(h) for c in range(w) if (r, c) not in protected]
    np.random.shuffle(coords)
    if len(required) > len(coords):
        return False
    for color, (r, c) in zip(required, coords):
        grid[r, c] = color
    for r, c in coords[len(required) :]:
        grid[r, c] = int(np.random.choice(np.array(palette, dtype=int)))
    return True


def _gen_attempt() -> np.ndarray | None:
    palette = _gen_distinct_palette()
    grid = np.zeros(GEN_GRID_SHAPE, dtype=int)

    protected = _gen_force_asymmetry(grid, palette)
    if not _fill_unprotected_cells(grid, palette, protected):
        return None
    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid 62c24649 input")
