from __future__ import annotations

import numpy as np


FRAME_COLOR = 2
BLANK_COLOR = 0
FILL_COLOR = 4

ROW_STRIDE = 2
COL_STRIDE = 4
CENTER_ROW_OFFSET = 1
CENTER_COL_OFFSET = 1
GUTTER_COL_OFFSET = 3

MACRO_ROW_OPTIONS = (3, 5, 7, 9)
MIN_MACRO_COLS = 3
MAX_MACRO_COLS = 6
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _macro_axes(shape: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    assert h >= 7 and w >= 11, "expected at least a 3x3 macro-grid"
    assert h % 2 == 1 and (w + 1) % 4 == 0, "expected 2r+1 by 4c-1 geometry"

    n_rows = (h - 1) // ROW_STRIDE
    n_cols = (w + 1) // COL_STRIDE
    assert n_rows % 2 == 1, "macro-row count must be odd to define a vertical palindrome axis"

    center_rows = np.arange(CENTER_ROW_OFFSET, h, ROW_STRIDE)
    center_cols = np.arange(CENTER_COL_OFFSET, w, COL_STRIDE)
    gutter_cols = np.arange(GUTTER_COL_OFFSET, w, COL_STRIDE)
    return n_rows, n_cols, center_rows, center_cols, gutter_cols


def _extract_center_matrix(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    n_rows, n_cols, center_rows, center_cols, gutter_cols = _macro_axes(grid.shape)
    frame = int(grid[0, 0])
    assert frame != BLANK_COLOR, "macro-cell frame must differ from blank"

    if gutter_cols.size:
        assert np.all(grid[:, gutter_cols] == BLANK_COLOR), "expected solid blank gutter columns"

    variable_mask = np.zeros(grid.shape, dtype=bool)
    variable_mask[np.ix_(center_rows, center_cols)] = True

    frame_mask = np.ones(grid.shape, dtype=bool)
    frame_mask[:, gutter_cols] = False
    frame_mask &= ~variable_mask
    assert np.all(grid[frame_mask] == frame), "all non-center, non-gutter cells should be the frame color"

    centers = grid[np.ix_(center_rows, center_cols)].copy()
    assert np.all((centers == BLANK_COLOR) | (centers == FILL_COLOR)), "center pixels should be blank or filled"
    return centers, center_rows, center_cols, frame, n_rows, n_cols


def _complete_palindromic_columns(centers: np.ndarray) -> np.ndarray:
    completed = centers.copy()
    n_rows, n_cols = completed.shape
    for c in range(n_cols):
        for top in range(n_rows // 2):
            bottom = n_rows - 1 - top
            a = int(completed[top, c])
            b = int(completed[bottom, c])
            if a == FILL_COLOR and b == BLANK_COLOR:
                completed[bottom, c] = FILL_COLOR
            elif a == BLANK_COLOR and b == FILL_COLOR:
                completed[top, c] = FILL_COLOR
    return completed


def _grid_from_centers(centers: np.ndarray) -> np.ndarray:
    n_rows, n_cols = map(int, centers.shape)
    h = ROW_STRIDE * n_rows + 1
    w = COL_STRIDE * n_cols - 1
    grid = np.full((h, w), FRAME_COLOR, dtype=int)
    gutter_cols = np.arange(GUTTER_COL_OFFSET, w, COL_STRIDE)
    center_rows = np.arange(CENTER_ROW_OFFSET, h, ROW_STRIDE)
    center_cols = np.arange(CENTER_COL_OFFSET, w, COL_STRIDE)
    grid[:, gutter_cols] = BLANK_COLOR
    grid[np.ix_(center_rows, center_cols)] = np.where(centers, FILL_COLOR, BLANK_COLOR)
    return grid


def _palindromic_column_from_count(n_rows: int, count: int, rng: np.random.Generator) -> np.ndarray:
    half = n_rows // 2
    col = np.zeros(n_rows, dtype=bool)
    pair_count = count // 2
    if pair_count:
        chosen = rng.choice(half, size=pair_count, replace=False)
        col[chosen] = True
        col[n_rows - 1 - chosen] = True
    if count % 2 == 1:
        col[half] = True
    return col


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if np.array_equal(perm, np.arange(n)):
        i, j = rng.choice(n, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _damage_columns(completed: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    n_rows, n_cols = completed.shape
    half = n_rows // 2
    observed = completed.copy()

    candidate_cols = [c for c in range(n_cols) if np.any(completed[:half, c] & completed[:half, c][::-1][::-1])]
    if not candidate_cols:
        return None

    for _ in range(120):
        observed[:] = completed
        total_erased = 0
        order_changed = False

        shuffled = rng.permutation(n_cols)
        chosen_cols = shuffled[: int(rng.integers(1, min(n_cols, 4) + 1))]
        for c in chosen_cols:
            pair_rows = [r for r in range(half) if completed[r, c] and completed[n_rows - 1 - r, c]]
            if not pair_rows:
                continue
            max_erasures = min(2, len(pair_rows))
            weights = np.array([0.28, 0.42, 0.30][: max_erasures + 1], dtype=float)
            weights /= weights.sum()
            erase_count = int(rng.choice(np.arange(max_erasures + 1), p=weights))
            if erase_count == 0:
                continue
            erase_rows = rng.choice(pair_rows, size=erase_count, replace=False)
            erase_top_side = bool(rng.integers(0, 2))
            for r in np.atleast_1d(erase_rows):
                rr = int(r) if erase_top_side else int(n_rows - 1 - r)
                observed[rr, c] = False
                total_erased += 1

        if total_erased == 0:
            continue

        target_counts = completed.sum(axis=0)
        observed_counts = observed.sum(axis=0)
        target_order = np.argsort(-target_counts, kind="stable")
        observed_order = np.argsort(-observed_counts, kind="stable")
        order_changed = not np.array_equal(observed_order, target_order)

        if order_changed:
            return observed
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the erased center pixels by vertical-palindrome completion, then sort macro-columns by filled-center count.

    Input structure:
    - The grid is a rectangular macro-grid of red 3x3 cells stacked with shared horizontal borders and separated by black vertical gutter columns.
    - Every macro-cell differs only at its center pixel, which is either black or yellow.
    - Within each macro-column, the full center-pixel pattern should be vertically palindromic, but some yellow centers on one side of the axis were erased to black.

    Rule:
    - Read the yellow/black center pixels as a matrix whose rows are macro-rows and whose columns are macro-columns.
    - For each column, compare every center with its vertical mirror partner; if one side is yellow and the other side is black, restore the black cell to yellow.
    - Count yellow centers in each repaired column and reorder whole center columns from highest count to lowest count.
    - Leave the red frames and black gutter columns unchanged.

    Key insight:
    - The visible damage only affects center pixels and is recoverable because each macro-column is meant to be a vertical palindrome; the repaired counts, not the damaged ones, determine the final column order.
    """
    g = np.asarray(grid, dtype=int)
    centers, center_rows, center_cols, _frame, _n_rows, _n_cols = _extract_center_matrix(g)
    completed = _complete_palindromic_columns(centers)
    counts = (completed == FILL_COLOR).sum(axis=0)
    assert len(set(map(int, counts.tolist()))) == completed.shape[1], "repaired column counts must be unique"

    order = np.argsort(-counts, kind="stable")
    out = g.copy()
    out[np.ix_(center_rows, center_cols)] = completed[:, order]
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows = int(rng.choice(MACRO_ROW_OPTIONS))
    max_cols = min(MAX_MACRO_COLS, n_rows + 1, 7)
    n_cols = int(rng.integers(MIN_MACRO_COLS, max_cols + 1))

    top_count = int(rng.integers(n_cols - 1, n_rows + 1))
    sorted_counts = np.arange(top_count, top_count - n_cols, -1, dtype=int)

    completed_sorted = np.column_stack(
        [_palindromic_column_from_count(n_rows, int(count), rng) for count in sorted_counts]
    )
    perm = _non_identity_permutation(n_cols, rng)
    completed_input = completed_sorted[:, perm]

    damaged_input = _damage_columns(completed_input, rng)
    if damaged_input is None:
        return None

    inp = _grid_from_centers(damaged_input)
    expected = _grid_from_centers(completed_sorted)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
