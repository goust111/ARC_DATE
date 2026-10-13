from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
INPUT_COLORS = (1, 2, 3, 4)
OUTPUT_PALETTE = (1, 2, 3)  # blue, red, green
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _compress_blocks(grid: np.ndarray, bg: int) -> np.ndarray:
    height, width = map(int, grid.shape)
    assert height % 3 == 0 and width % 3 == 0, "grid must split into a 3x3 partition"

    block_h = height // 3
    block_w = width // 3
    compressed = np.empty((3, 3), dtype=int)

    for br in range(3):
        for bc in range(3):
            block = grid[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ]
            foreground = block[block != bg]
            assert foreground.size == 1, "each block must contain exactly one non-background pixel"
            compressed[br, bc] = int(foreground[0])

    return compressed


def _diversity_rank_colors(compressed: np.ndarray) -> dict[int, int]:
    diversities = [len(set(map(int, compressed[:, c]))) for c in range(3)]
    assert set(diversities) == {1, 2, 3}, "column diversities must be exactly 1, 2, 3"
    ordered_cols = sorted(range(3), key=lambda c: diversities[c])
    return {col: OUTPUT_PALETTE[rank] for rank, col in enumerate(ordered_cols)}


def _expected_output(grid: np.ndarray, compressed: np.ndarray, bg: int) -> np.ndarray:
    block_w = grid.shape[1] // 3
    rank_colors = _diversity_rank_colors(compressed)
    out = grid.copy()
    for col, color in rank_colors.items():
        c0 = col * block_w
        c1 = (col + 1) * block_w
        band = out[:, c0:c1]
        band[band != bg] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each original column band by the diversity rank of its compressed 3x3 column.

    Input structure:
    - The input is a black-background grid whose height and width are divisible by 3.
    - The grid is implicitly partitioned into a 3x3 layout of equal rectangular blocks.
    - Each block contains exactly one non-background pixel colored from a small foreground palette.

    Rule:
    - Compress the 3x3 partition by reading the unique non-background color from each block.
    - For each compressed column, count how many distinct colors appear in its three cells.
    - Rank the three columns from lowest to highest diversity.
    - Recolor all non-background pixels in the corresponding original 3-column-wide band:
      lowest diversity -> blue(1), middle -> red(2), highest -> green(3).
    - Preserve all background cells.

    Key insight:
    - The compressed 3x3 grid is only an intermediate metadata layer; its column-wise color diversity
      determines how to repaint the original larger grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    compressed = _compress_blocks(g, bg)
    return _expected_output(g, compressed, bg)


def _sample_diversity1_column(rng: np.random.Generator) -> np.ndarray:
    color = int(rng.choice((2, 3, 4)))
    return np.full(3, color, dtype=int)


def _sample_diversity2_column(rng: np.random.Generator) -> np.ndarray:
    colors = rng.choice(INPUT_COLORS, size=2, replace=False)
    repeated = int(rng.integers(0, 2))
    single = 1 - repeated
    column = np.array([colors[repeated], colors[repeated], colors[single]], dtype=int)
    rng.shuffle(column)
    return column


def _sample_diversity3_column(rng: np.random.Generator) -> np.ndarray:
    column = rng.choice(INPUT_COLORS, size=3, replace=False).astype(int)
    rng.shuffle(column)
    return column


def _build_compressed_grid(rng: np.random.Generator) -> np.ndarray:
    compressed = np.empty((3, 3), dtype=int)
    physical_cols = rng.permutation(3)
    samplers = (
        _sample_diversity1_column,
        _sample_diversity2_column,
        _sample_diversity3_column,
    )

    for target_col, sampler in zip(physical_cols, samplers):
        compressed[:, int(target_col)] = sampler(rng)

    return compressed


def _render_input(rng: np.random.Generator, compressed: np.ndarray) -> np.ndarray:
    block_h = int(rng.integers(2, 9))
    block_w = int(rng.integers(2, 8))
    grid = np.full((3 * block_h, 3 * block_w), BACKGROUND, dtype=int)

    for br, bc in itertools.product(range(3), range(3)):
        rr = br * block_h + int(rng.integers(0, block_h))
        cc = bc * block_w + int(rng.integers(0, block_w))
        grid[rr, cc] = int(compressed[br, bc])

    return grid


def _valid_example(inp: np.ndarray, compressed: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False

    expected = _expected_output(inp, compressed, BACKGROUND)
    if not np.array_equal(actual, expected):
        return False

    changed = int(np.count_nonzero(actual != inp))
    if changed < 3:
        return False

    block_w = inp.shape[1] // 3
    rank_colors = _diversity_rank_colors(compressed)
    # Keep every band visually meaningful: at least one pixel in the band changes.
    for col, out_color in rank_colors.items():
        band_colors = set(map(int, compressed[:, col]))
        if band_colors == {out_color}:
            return False
        c0 = col * block_w
        c1 = (col + 1) * block_w
        if not np.any(inp[:, c0:c1] != actual[:, c0:c1]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    compressed = _build_compressed_grid(rng)
    inp = _render_input(rng, compressed)
    if _valid_example(inp, compressed):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
