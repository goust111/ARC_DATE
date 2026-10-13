from __future__ import annotations

from collections import Counter
import random

import numpy as np


SEP_COLOR = 5
OUTPUT_BG = 0
OUTPUT_DIFF = 3
PALETTE = (1, 2, 4, 6, 7, 8, 9)
BLOCK_H_RANGE = (2, 5)
BLOCK_W_RANGE = (2, 5)
RETRY_LIMIT = 400


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _separator_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g[0, 0]) == SEP_COLOR, "expected grey outer frame"
    assert np.all(g[0, :] == SEP_COLOR) and np.all(g[-1, :] == SEP_COLOR), "expected full horizontal border"
    assert np.all(g[:, 0] == SEP_COLOR) and np.all(g[:, -1] == SEP_COLOR), "expected full vertical border"

    full_sep_rows = np.all(g == SEP_COLOR, axis=1)
    full_sep_cols = np.all(g == SEP_COLOR, axis=0)
    row_runs = _runs([r for r in range(g.shape[0]) if not full_sep_rows[r]])
    col_runs = _runs([c for c in range(g.shape[1]) if not full_sep_cols[c]])

    assert len(row_runs) == 4, f"expected 4 row bands, got {row_runs}"
    assert len(col_runs) == 3, f"expected 3 column bands, got {col_runs}"
    assert int(full_sep_rows.sum()) == 5, "expected 5 separator rows"
    assert int(full_sep_cols.sum()) == 4, "expected 4 separator columns"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1, "expected equal-height blocks"
    assert len(col_widths) == 1, "expected equal-width blocks"
    return row_runs, col_runs


def _strict_majority(block: np.ndarray) -> int:
    vals = list(map(int, block.ravel()))
    assert SEP_COLOR not in vals, "separator color must not appear inside a block"
    counts = Counter(vals).most_common()
    assert len(counts) >= 2, "each block must contain a majority and a minority color"
    assert counts[0][1] > counts[1][1], f"majority must be unique, got {counts}"
    return int(counts[0][0])


def _majority_grid(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_runs, col_runs = _separator_bands(grid)
    majority = np.zeros((4, 3), dtype=int)
    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            majority[br, bc] = _strict_majority(block)
    return majority, row_runs, col_runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each lattice block to its majority color, xor the upper and lower halves, and keep only the disagreements in green.

    Input structure:
    - A grey separator lattice with outer frame divides the interior into 4 row bands and 3 column bands of equal-sized blocks.
    - Each block is filled with two non-grey colors, and exactly one of them is the unique majority within that block.
    - The top 2x3 blocks and bottom 2x3 blocks are compared by matching columns and relative row positions.

    Rule:
    - Detect the separator rows and columns, then isolate the 12 interior blocks.
    - Replace each block by its unique majority color to obtain a 4x3 coarse grid.
    - Compare row 0 with row 2 and row 1 with row 3 cell by cell; output green where the paired majority colors differ, otherwise black.

    Key insight:
    - The pixel-level block contents are noise; only each block's majority color matters, and the final answer is the xor of the upper and lower majority summaries.
    """
    g = np.asarray(grid, dtype=int)
    majority, _row_runs, _col_runs = _majority_grid(g)
    top = majority[:2, :]
    bottom = majority[2:, :]
    out = np.full((2, 3), OUTPUT_BG, dtype=int)
    out[top != bottom] = OUTPUT_DIFF
    return out


def _sample_diff_mask() -> np.ndarray:
    styles = (
        [1],
        [2, 3, 4],
        [4, 5],
    )
    diff_count = random.choice(random.choice(styles))
    flat = np.array([True] * diff_count + [False] * (6 - diff_count), dtype=bool)
    np.random.shuffle(flat)
    mask = flat.reshape(2, 3)
    assert 0 < int(mask.sum()) < 6
    return mask


def _sample_majority_layers(colors: tuple[int, int], diff_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = colors
    for _ in range(200):
        top_bits = np.random.randint(0, 2, size=(2, 3), dtype=int)
        bottom_bits = top_bits ^ diff_mask.astype(int)
        if len(set(map(int, top_bits.ravel()))) < 2:
            continue
        if len(set(map(int, bottom_bits.ravel()))) < 2:
            continue
        top = np.where(top_bits == 0, a, b)
        bottom = np.where(bottom_bits == 0, a, b)
        return top, bottom
    raise RuntimeError("failed to sample majority layers")


def _minority_count(area: int) -> int:
    max_minority = (area - 1) // 2
    assert max_minority >= 1, "block must be large enough for a strict majority"
    choices = list(range(1, max_minority + 1))
    weights = [2 if count == max_minority else 1 for count in choices]
    return random.choices(choices, weights=weights, k=1)[0]


def _render_block(
    majority_color: int,
    block_h: int,
    block_w: int,
    noise_choices: tuple[int, ...],
) -> np.ndarray:
    area = block_h * block_w
    minority_cells = _minority_count(area)
    minority_color = random.choice([c for c in noise_choices if c != majority_color])
    block = np.full((block_h, block_w), majority_color, dtype=int)
    cells = random.sample(range(area), minority_cells)
    for idx in cells:
        block[idx // block_w, idx % block_w] = minority_color
    return block


def _pair_blocks(
    top_majority: int,
    bottom_majority: int,
    block_h: int,
    block_w: int,
    noise_choices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        top_block = _render_block(top_majority, block_h, block_w, noise_choices)
        bottom_block = _render_block(bottom_majority, block_h, block_w, noise_choices)
        if top_majority == bottom_majority and np.array_equal(top_block, bottom_block):
            continue
        return top_block, bottom_block
    raise RuntimeError("failed to build a non-degenerate paired block")


def _render_grid(
    top_majority: np.ndarray,
    bottom_majority: np.ndarray,
    block_h: int,
    block_w: int,
    noise_choices: tuple[int, ...],
) -> np.ndarray:
    height = 4 * block_h + 5
    width = 3 * block_w + 4
    grid = np.full((height, width), SEP_COLOR, dtype=int)
    for rel_r in range(2):
        for c in range(3):
            top_block, bottom_block = _pair_blocks(
                int(top_majority[rel_r, c]),
                int(bottom_majority[rel_r, c]),
                block_h,
                block_w,
                noise_choices,
            )
            top_r0 = 1 + rel_r * (block_h + 1)
            bot_r0 = 1 + (rel_r + 2) * (block_h + 1)
            c0 = 1 + c * (block_w + 1)
            grid[top_r0 : top_r0 + block_h, c0 : c0 + block_w] = top_block
            grid[bot_r0 : bot_r0 + block_h, c0 : c0 + block_w] = bottom_block
    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    majority, row_runs, col_runs = _majority_grid(grid)
    assert row_runs[0][1] - row_runs[0][0] + 1 in range(BLOCK_H_RANGE[0], BLOCK_H_RANGE[1] + 1)
    assert col_runs[0][1] - col_runs[0][0] + 1 in range(BLOCK_W_RANGE[0], BLOCK_W_RANGE[1] + 1)

    block_patterns = {}
    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(block)))
            assert SEP_COLOR not in colors and OUTPUT_BG not in colors and OUTPUT_DIFF not in colors
            assert len(colors) == 2, f"expected exactly two visible colors per block, got {colors}"
            block_patterns[(br, bc)] = block

    majority_colors = set(map(int, majority.ravel()))
    assert len(majority_colors) == 2, f"expected exactly two majority colors, got {majority_colors}"

    out = main(grid)
    green_count = int(np.count_nonzero(out == OUTPUT_DIFF))
    assert out.shape == (2, 3)
    assert 0 < green_count < 6, "xor output must contain both disagreements and agreements"

    same_majority_nonidentical = False
    for rel_r in range(2):
        for c in range(3):
            top_block = block_patterns[(rel_r, c)]
            bottom_block = block_patterns[(rel_r + 2, c)]
            if majority[rel_r, c] == majority[rel_r + 2, c] and not np.array_equal(top_block, bottom_block):
                same_majority_nonidentical = True
    assert same_majority_nonidentical, "need at least one same-majority pair whose raw blocks still differ"


def _gen_attempt() -> np.ndarray | None:
    block_h = random.randint(*BLOCK_H_RANGE)
    block_w = random.randint(*BLOCK_W_RANGE)
    majority_pair = tuple(random.sample(PALETTE, 2))

    extra_candidates = [c for c in PALETTE if c not in majority_pair]
    extra_noise_count = random.randint(0, min(2, len(extra_candidates)))
    noise_choices = tuple(majority_pair) + tuple(random.sample(extra_candidates, extra_noise_count))

    diff_mask = _sample_diff_mask()
    top_majority, bottom_majority = _sample_majority_layers(majority_pair, diff_mask)
    grid = _render_grid(top_majority, bottom_majority, block_h, block_w, noise_choices)

    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None

    expected = np.full((2, 3), OUTPUT_BG, dtype=int)
    expected[diff_mask] = OUTPUT_DIFF
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the majority-xor lattice task")
