from __future__ import annotations

import numpy as np


def _run_starts_and_lengths(lines: np.ndarray) -> tuple[list[int], np.ndarray]:
    starts = [0]
    lengths: list[int] = []
    run_start = 0
    for idx in range(1, lines.shape[0]):
        if not np.array_equal(lines[idx], lines[idx - 1]):
            lengths.append(idx - run_start)
            starts.append(idx)
            run_start = idx
    lengths.append(lines.shape[0] - run_start)
    return starts, np.array(lengths, dtype=int)


def _collapse_base_and_counts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_starts, row_repeats = _run_starts_and_lengths(grid)
    col_starts, col_repeats = _run_starts_and_lengths(grid.T)
    base = grid[np.ix_(row_starts, col_starts)].copy()
    return base, row_repeats, col_repeats


def _stretch(base: np.ndarray, row_repeats: np.ndarray, col_repeats: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(base, row_repeats, axis=0), col_repeats, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover a stretched square base grid, swap row/column repeat counts, and stretch it again.

    Input structure
    - The input is a solid color grid with no background.
    - It comes from a square base matrix whose rows and columns were expanded into adjacent runs of identical rows and columns.
    - The row-run counts and column-run counts are position-wise data that may differ.

    Rule
    - Collapse each maximal run of identical neighboring rows and each maximal run of identical neighboring columns to recover the square base matrix.
    - Record the row repeat-count sequence `r` and the column repeat-count sequence `c`.
    - Re-expand the same base matrix using swapped counts: row `i` repeats `c[i]` times and column `j` repeats `r[j]` times.

    Key insight
    - The duplicate runs are not disposable noise: their run lengths are the payload, and the answer is the same base matrix stretched with the two ordered count sequences exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0

    base, row_repeats, col_repeats = _collapse_base_and_counts(g)
    assert base.shape[0] == base.shape[1], "the recovered base must be square"
    assert len(row_repeats) == len(col_repeats) == base.shape[0], "row/col run counts must align"

    return _stretch(base, col_repeats, row_repeats)


MAX_GENERATION_ATTEMPTS = 500
BASE_ATTEMPTS = 300
SEQUENCE_ATTEMPTS = 300
BASE_SIDE_RANGE = (2, 5)
REPEAT_MAX = 6
PALETTE = np.arange(1, 10)


def _sample_sequence(
    rng: np.random.Generator,
    length: int,
    min_repeat: int,
    max_repeat: int,
    *,
    target_sum: int | None = None,
    require_singleton: bool = False,
) -> np.ndarray | None:
    for _ in range(SEQUENCE_ATTEMPTS):
        if target_sum is None:
            total = int(rng.integers(length * min_repeat, length * max_repeat + 1))
        else:
            total = int(target_sum)
        if total < length * min_repeat or total > length * max_repeat:
            return None

        remaining = total
        values: list[int] = []
        valid = True
        for idx in range(length):
            slots_left = length - idx - 1
            lo = max(min_repeat, remaining - slots_left * max_repeat)
            hi = min(max_repeat, remaining - slots_left * min_repeat)
            if lo > hi:
                valid = False
                break
            value = int(rng.integers(lo, hi + 1))
            values.append(value)
            remaining -= value
        if not valid or remaining != 0:
            continue

        seq = np.array(values, dtype=int)
        if len(set(seq.tolist())) == 1:
            continue
        if int(seq.max()) == 1:
            continue
        if require_singleton and 1 not in seq:
            continue
        return seq
    return None


def _all_rows_unique(base: np.ndarray) -> bool:
    return len({tuple(map(int, row)) for row in base}) == base.shape[0]


def _all_cols_unique(base: np.ndarray) -> bool:
    return len({tuple(map(int, base[:, c])) for c in range(base.shape[1])}) == base.shape[1]


def _sample_base(rng: np.random.Generator, side: int) -> np.ndarray | None:
    for _ in range(BASE_ATTEMPTS):
        color_count = int(rng.integers(3, min(9, side * side) + 1))
        colors = rng.choice(PALETTE, size=color_count, replace=False)
        base = rng.choice(colors, size=(side, side), replace=True).astype(int)

        for pos, color in zip(rng.permutation(side * side)[:color_count], colors):
            base.flat[int(pos)] = int(color)

        if np.array_equal(base, base.T):
            continue
        if len(np.unique(base)) < 3:
            continue
        if not _all_rows_unique(base) or not _all_cols_unique(base):
            continue
        if any(len(set(map(int, row))) == 1 for row in base):
            continue
        if any(len(set(map(int, base[:, c]))) == 1 for c in range(side)):
            continue
        return base
    return None


def _sample_repeat_pair(rng: np.random.Generator, side: int) -> tuple[np.ndarray, np.ndarray] | None:
    singleton_mode = int(rng.integers(3))
    same_total = bool(rng.integers(2))

    if singleton_mode == 0:
        row_min = col_min = 2
        row_singleton = col_singleton = False
    elif singleton_mode == 1:
        if bool(rng.integers(2)):
            row_min, col_min = 1, 2
            row_singleton, col_singleton = True, False
        else:
            row_min, col_min = 2, 1
            row_singleton, col_singleton = False, True
    else:
        row_min = col_min = 1
        row_singleton = col_singleton = True

    row_repeats = _sample_sequence(
        rng,
        side,
        row_min,
        REPEAT_MAX,
        require_singleton=row_singleton,
    )
    if row_repeats is None:
        return None

    col_repeats = _sample_sequence(
        rng,
        side,
        col_min,
        REPEAT_MAX,
        target_sum=int(row_repeats.sum()) if same_total else None,
        require_singleton=col_singleton,
    )
    if col_repeats is None:
        return None

    if np.array_equal(row_repeats, col_repeats):
        return None
    if same_total and int(row_repeats.sum()) != int(col_repeats.sum()):
        return None
    if not same_total and int(row_repeats.sum()) == int(col_repeats.sum()):
        return None
    return row_repeats, col_repeats


def _validate_instance(
    inp: np.ndarray,
    base: np.ndarray,
    row_repeats: np.ndarray,
    col_repeats: np.ndarray,
) -> bool:
    recovered_base, recovered_rows, recovered_cols = _collapse_base_and_counts(inp)
    if not np.array_equal(recovered_base, base):
        return False
    if not np.array_equal(recovered_rows, row_repeats):
        return False
    if not np.array_equal(recovered_cols, col_repeats):
        return False

    expected = _stretch(base, col_repeats, row_repeats)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(actual, inp):
        return False
    if np.array_equal(actual, base):
        return False
    if actual.shape == inp.T.shape and np.array_equal(actual, inp.T):
        return False

    row_starts, _ = _run_starts_and_lengths(inp)
    col_starts, _ = _run_starts_and_lengths(inp.T)
    row_only = inp[row_starts, :]
    col_only = inp[:, col_starts]
    if np.array_equal(actual, row_only) or np.array_equal(actual, col_only):
        return False

    if max(inp.shape) > 30 or max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    side = int(rng.integers(BASE_SIDE_RANGE[0], BASE_SIDE_RANGE[1] + 1))

    repeats = _sample_repeat_pair(rng, side)
    if repeats is None:
        return None
    row_repeats, col_repeats = repeats

    base = _sample_base(rng, side)
    if base is None:
        return None

    inp = _stretch(base, row_repeats, col_repeats)
    if not _validate_instance(inp, base, row_repeats, col_repeats):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
