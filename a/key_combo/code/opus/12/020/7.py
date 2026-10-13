from __future__ import annotations

import random

import numpy as np


BG = 0
GRID_H = 10
BLOCK_W = 6
NUM_BLOCKS = 4
TOTAL_W = BLOCK_W * NUM_BLOCKS
COLORS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _fill_matching_endpoint_rows(block: np.ndarray) -> np.ndarray:
    """Fill a row when its two boundary markers match and are non-background."""
    out = np.array(block, dtype=int, copy=True)
    for r in range(out.shape[0]):
        left = int(out[r, 0])
        right = int(out[r, -1])
        if left != BG and left == right:
            out[r, :] = left
    return out


def _find_smallest_mirror_period(grid: np.ndarray) -> int:
    """Return the smallest block width whose copies are either equal or left-right mirrored."""
    g = np.asarray(grid, dtype=int)
    _, width = g.shape
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        base = g[:, :period]
        base_flip = np.fliplr(base)
        valid = True
        for start in range(0, width, period):
            block = g[:, start : start + period]
            if not (np.array_equal(block, base) or np.array_equal(block, base_flip)):
                valid = False
                break
        if valid:
            return period
    raise AssertionError("expected a valid mirrored horizontal period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the smallest horizontally mirrored period block, then fill rows whose boundary markers match.

    Input structure:
    - A black-background grid is tiled by equal-width column blocks.
    - Every block matches the leftmost block either directly or after a left-right flip.
    - Within the true source block, some rows carry colored marker pixels only at the left and right boundaries.

    Rule:
    - Scan block widths from smallest to largest and find the first width whose blocks all equal the first block or its left-right mirror.
    - Crop the leftmost block with that minimal width.
    - For each cropped row, compare the leftmost and rightmost pixels.
    - If the two endpoint colors are equal and non-black, fill the whole row with that color; otherwise leave the row unchanged.

    Key insight:
    - Mirrored repetitions swap which boundary marker belongs on each side, so the endpoint pairing is only trustworthy after recovering the smallest source block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    period = _find_smallest_mirror_period(g)
    block = g[:, :period]
    return _fill_matching_endpoint_rows(block)


def _build_full_grid(base_block: np.ndarray) -> np.ndarray:
    pieces = []
    for idx in range(NUM_BLOCKS):
        pieces.append(base_block if idx % 2 == 0 else np.fliplr(base_block))
    return np.concatenate(pieces, axis=1)


def _sample_base_block() -> tuple[np.ndarray, set[int], set[int]]:
    block = np.zeros((GRID_H, BLOCK_W), dtype=int)

    marked_count = random.randint(4, 8)
    marked_rows = random.sample(range(GRID_H), marked_count)
    match_count = random.randint(2, marked_count - 1)
    match_rows = set(random.sample(marked_rows, match_count))
    mismatch_rows = set(marked_rows) - match_rows

    for r in match_rows:
        color = random.choice(COLORS)
        block[r, 0] = color
        block[r, -1] = color

    for r in mismatch_rows:
        left, right = random.sample(COLORS, 2)
        block[r, 0] = left
        block[r, -1] = right

    return block, match_rows, mismatch_rows


def _validate_semantics(full_grid: np.ndarray, base_block: np.ndarray, match_rows: set[int], mismatch_rows: set[int]) -> bool:
    if len(match_rows) < 2 or len(mismatch_rows) < 1:
        return False

    if _find_smallest_mirror_period(full_grid) != BLOCK_W:
        return False

    expected = _fill_matching_endpoint_rows(base_block)
    if not np.array_equal(main(full_grid), expected):
        return False
    if np.array_equal(expected, base_block):
        return False

    # Reject cases where the crop step becomes unnecessary because every marked row
    # would make the same decision even if one compared the outermost wide-grid edges.
    wrong_rows = [
        r
        for r in mismatch_rows
        if int(full_grid[r, 0]) == int(full_grid[r, -1]) != BG
    ]
    if not wrong_rows:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        base_block, match_rows, mismatch_rows = _sample_base_block()
        full_grid = _build_full_grid(base_block)
        if _validate_semantics(full_grid, base_block, match_rows, mismatch_rows):
            return full_grid
    raise RuntimeError("failed to generate a valid input")
