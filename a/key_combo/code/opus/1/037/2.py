from __future__ import annotations

import random

import numpy as np


SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
BLOCK_H_CHOICES = (3, 4)
BLOCK_W_CHOICES = (4, 5)
ROW_CHOICES = (2, 3, 4)
COL_CHOICES = (3, 4, 5)
MAX_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            runs.append((start, int(idx) - 1))
            start = None
    if start is not None:
        runs.append((start, int(len(mask)) - 1))
    return runs


def _lattice_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full_sep_rows = np.all(grid == SEPARATOR, axis=1)
    full_sep_cols = np.all(grid == SEPARATOR, axis=0)
    assert full_sep_rows[0] and full_sep_rows[-1], "expected separator border rows"
    assert full_sep_cols[0] and full_sep_cols[-1], "expected separator border cols"

    row_runs = _runs(~full_sep_rows)
    col_runs = _runs(~full_sep_cols)
    assert row_runs and col_runs, "expected at least one block row and block col"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1, "expected equal-height row bands"
    assert len(col_widths) == 1, "expected equal-width col bands"
    return row_runs, col_runs


def _majority_and_minority_count(block: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(block, return_counts=True)
    assert not np.any(colors == SEPARATOR), "block should not contain separator pixels"

    order = np.argsort(-counts, kind="stable")
    top = int(order[0])
    if len(order) > 1:
        assert int(counts[top]) > int(counts[int(order[1])]), "expected a unique majority color"

    majority = int(colors[top])
    majority_count = int(counts[top])
    minority_count = int(block.size) - majority_count
    return majority, minority_count


def _compressed_and_minority_counts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_runs, col_runs = _lattice_runs(grid)
    compressed = np.empty((len(row_runs), len(col_runs)), dtype=int)
    minority_counts = np.empty_like(compressed)

    for i, (r0, r1) in enumerate(row_runs):
        for j, (c0, c1) in enumerate(col_runs):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            majority, minority_count = _majority_and_minority_count(block)
            compressed[i, j] = majority
            minority_counts[i, j] = minority_count

    return compressed, minority_counts


def _bordered_shifted_output(compressed: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    rows, cols = map(int, compressed.shape)
    shifted = np.empty_like(compressed)
    for i in range(rows):
        shifted[i] = np.roll(compressed[i], int(shifts[i]) % cols)

    out = np.full((rows + 2, cols + 2), SEPARATOR, dtype=int)
    out[1:-1, 1:-1] = shifted
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each separator-delimited block to its majority color, then cyclically right-shift each compressed row by the minority count of its leftmost block.

    Input structure:
    - Gray 5 full rows and columns, including the outer border, divide the canvas into equal-sized rectangular blocks.
    - Every block is fully filled with non-gray colors and has one unique majority color over its internal pixels.
    - The minority pixels discarded from the leftmost block of each lattice row encode that row's later shift amount.

    Rule:
    - Detect the full gray separator rows and columns to recover the block lattice.
    - Replace every block by its unique majority color to form an R x C compressed grid, while also counting each block's minority pixels.
    - For each row i, cyclically shift compressed row i to the right by the minority-pixel count of block (i, 0).
    - Return the shifted compressed grid with a one-cell gray border.

    Key insight:
    - Majority voting produces both the visible compressed image and the hidden control signal: the number of minority pixels lost in the leftmost block of each row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    compressed, minority_counts = _compressed_and_minority_counts(g)
    return _bordered_shifted_output(compressed, minority_counts[:, 0])


def _majority_minimum(area: int) -> int:
    return (3 * area + 4) // 5


def _max_minority_count(area: int) -> int:
    return area - _majority_minimum(area)


def _choose_geometry() -> tuple[int, int, int, int, int] | None:
    for _ in range(50):
        block_h = random.choice(BLOCK_H_CHOICES)
        block_w = random.choice(BLOCK_W_CHOICES)
        max_minority = _max_minority_count(block_h * block_w)

        col_choices = [
            cols
            for cols in COL_CHOICES
            if cols * block_w + cols + 1 <= 30 and cols - 1 <= max_minority
        ]
        row_choices = [
            rows for rows in ROW_CHOICES if rows * block_h + rows + 1 <= 30
        ]
        if row_choices and col_choices:
            return (
                random.choice(row_choices),
                random.choice(col_choices),
                block_h,
                block_w,
                max_minority,
            )
    return None


def _choose_left_shifts(rows: int, cols: int) -> np.ndarray:
    choices = list(range(1, cols))
    while True:
        shifts = np.array([random.choice(choices) for _ in range(rows)], dtype=int)
        if rows > 1 and len(choices) > 1 and len(set(map(int, shifts))) < 2:
            continue
        if np.array_equal(shifts % cols, (-shifts) % cols):
            continue
        return shifts


def _choose_count_grid(rows: int, cols: int, max_minority: int) -> np.ndarray | None:
    counts = np.empty((rows, cols), dtype=int)
    left_shifts = _choose_left_shifts(rows, cols)
    counts[:, 0] = left_shifts
    all_choices = list(range(1, max_minority + 1))

    for i in range(rows):
        second_choices = [m for m in all_choices if m % cols != int(left_shifts[i]) % cols]
        if not second_choices:
            return None
        counts[i, 1] = random.choice(second_choices)
        for j in range(2, cols):
            counts[i, j] = random.choice(all_choices)
    return counts


def _choose_majority_grid(rows: int, cols: int) -> np.ndarray:
    majorities = np.empty((rows, cols), dtype=int)
    for i in range(rows):
        majorities[i] = np.array(random.sample(PALETTE, cols), dtype=int)
    return majorities


def _choose_minority_colors(majorities: np.ndarray) -> np.ndarray:
    rows, cols = map(int, majorities.shape)
    minority_colors = np.empty_like(majorities)
    for i in range(rows):
        row_colors = list(map(int, majorities[i]))
        for j in range(cols):
            majority = int(majorities[i, j])
            choices = [color for color in PALETTE if color != majority]
            shared_row_choices = [color for color in row_colors if color != majority]
            if shared_row_choices and random.random() < 0.6:
                choices = shared_row_choices + choices
            minority_colors[i, j] = random.choice(choices)
    return minority_colors


def _minority_mask(height: int, width: int, count: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.random() < 0.5:
        chosen = random.sample(range(height * width), count)
        mask[np.array(chosen) // width, np.array(chosen) % width] = True
        return mask

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < count:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if frontier:
            cells.add(random.choice(tuple(frontier)))
        else:
            remaining = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if (r, c) not in cells
            ]
            cells.add(random.choice(remaining))

    for r, c in cells:
        mask[r, c] = True
    return mask


def _render_block(
    height: int, width: int, majority: int, minority: int, minority_count: int
) -> np.ndarray:
    block = np.full((height, width), int(majority), dtype=int)
    minority_mask = _minority_mask(height, width, minority_count)
    block[minority_mask] = int(minority)
    return block


def _render_input(
    majorities: np.ndarray,
    minority_counts: np.ndarray,
    minority_colors: np.ndarray,
    block_h: int,
    block_w: int,
) -> np.ndarray:
    rows, cols = map(int, majorities.shape)
    height = rows * block_h + rows + 1
    width = cols * block_w + cols + 1
    grid = np.full((height, width), SEPARATOR, dtype=int)

    for i in range(rows):
        for j in range(cols):
            r0 = 1 + i * (block_h + 1)
            c0 = 1 + j * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = _render_block(
                block_h,
                block_w,
                int(majorities[i, j]),
                int(minority_colors[i, j]),
                int(minority_counts[i, j]),
            )

    return grid


def _validate_generated_instance(
    grid: np.ndarray,
    expected_majorities: np.ndarray,
    expected_counts: np.ndarray,
    expected_output: np.ndarray,
) -> bool:
    parsed_majorities, parsed_counts = _compressed_and_minority_counts(grid)
    if not np.array_equal(parsed_majorities, expected_majorities):
        return False
    if not np.array_equal(parsed_counts, expected_counts):
        return False
    if not np.array_equal(main(grid), expected_output):
        return False

    rows, cols = map(int, expected_majorities.shape)
    if np.array_equal(expected_output, _bordered_shifted_output(expected_majorities, np.zeros(rows, dtype=int))):
        return False
    if np.array_equal(expected_output, _bordered_shifted_output(expected_majorities, -expected_counts[:, 0])):
        return False
    if np.array_equal(expected_output, _bordered_shifted_output(expected_majorities, expected_counts[:, 1])):
        return False
    if np.array_equal(expected_output, _bordered_shifted_output(expected_majorities, expected_counts.sum(axis=1))):
        return False
    if np.array_equal(
        expected_output,
        _bordered_shifted_output(expected_majorities, np.full(rows, int(expected_counts[0, 0]), dtype=int)),
    ):
        return False

    for row in expected_majorities:
        if len(set(map(int, row))) != cols:
            return False
    if not np.all((1 <= expected_counts[:, 0]) & (expected_counts[:, 0] <= cols - 1)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    geometry = _choose_geometry()
    if geometry is None:
        return None
    rows, cols, block_h, block_w, max_minority = geometry

    majorities = _choose_majority_grid(rows, cols)
    minority_counts = _choose_count_grid(rows, cols, max_minority)
    if minority_counts is None:
        return None
    minority_colors = _choose_minority_colors(majorities)

    grid = _render_input(majorities, minority_counts, minority_colors, block_h, block_w)
    expected = _bordered_shifted_output(majorities, minority_counts[:, 0])
    if not _validate_generated_instance(grid, majorities, minority_counts, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-compress roll puzzle")
