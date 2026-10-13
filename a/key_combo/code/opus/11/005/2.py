from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
SEED_SIZE_RANGE = (3, 5)
SEED_FOREGROUND_RANGE = (3, 7)
INPUT_SIZE_RANGE = (12, 20)
RUN_LENGTH_RANGE = (2, 4)
GENERATION_ATTEMPTS = 1000


def _run_start_indices(matrix: np.ndarray) -> list[int]:
    starts = [0]
    for idx in range(1, matrix.shape[0]):
        if not np.array_equal(matrix[idx], matrix[idx - 1]):
            starts.append(idx)
    return starts


def _dedupe_rows(grid: np.ndarray) -> np.ndarray:
    return grid[_run_start_indices(grid)].copy()


def _dedupe_cols(grid: np.ndarray) -> np.ndarray:
    return grid[:, _run_start_indices(grid.T)].copy()


def _recover_seed(grid: np.ndarray) -> np.ndarray:
    return _dedupe_cols(_dedupe_rows(grid))


def _stamp_seed(seed: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    h, w = map(int, seed.shape)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(seed[r, c]) != BACKGROUND:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = seed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the stretched seed and expand it by stamping the seed into its own foreground-marked blocks.

    Input structure:
    - The input is a 2-D grid formed by stretching a smaller seed into adjacent runs of identical rows and columns.
    - The hidden seed is a 3x3 to 5x5 two-color pattern: black background plus one non-black foreground color.
    - The visible grid therefore contains consecutive duplicate rows and consecutive duplicate columns.

    Rule:
    - Keep the first row from each maximal run of identical adjacent rows.
    - On the result, keep the first column from each maximal run of identical adjacent columns; this recovers the seed.
    - Create an output made of seed-shaped blocks arranged in a seed_rows by seed_cols layout.
    - For each foreground pixel of the recovered seed, copy the whole seed into the corresponding block; for each background pixel, leave the block black.

    Key insight:
    - The compressed seed has a double role: it is both the placement mask and the template being stamped, so the self-similar expansion cannot be solved until the duplicate runs have been removed correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "grid must be non-empty"

    seed = _recover_seed(g)
    assert seed.shape[0] <= g.shape[0] and seed.shape[1] <= g.shape[1]
    return _stamp_seed(seed)


def _all_run_patterns(parts: int) -> list[tuple[int, ...]]:
    lo, hi = RUN_LENGTH_RANGE
    size_lo, size_hi = INPUT_SIZE_RANGE
    candidates: list[tuple[int, ...]] = []
    for runs in itertools.product(range(lo, hi + 1), repeat=parts):
        total = sum(runs)
        if size_lo <= total <= size_hi:
            candidates.append(tuple(int(x) for x in runs))
    return candidates


def _choose_runs(parts: int, rng: np.random.Generator) -> tuple[int, ...]:
    candidates = _all_run_patterns(parts)
    assert candidates, f"no valid run pattern for {parts} parts"
    varied = [runs for runs in candidates if len(set(runs)) > 1]
    equal = [runs for runs in candidates if len(set(runs)) == 1]

    roll = float(rng.random())
    if varied and roll < 0.75:
        pool = varied
    elif equal and roll < 0.9:
        pool = equal
    else:
        pool = candidates
    return pool[int(rng.integers(len(pool)))]


def _adjacent_rows_distinct(grid: np.ndarray) -> bool:
    return all(not np.array_equal(grid[r], grid[r + 1]) for r in range(grid.shape[0] - 1))


def _adjacent_cols_distinct(grid: np.ndarray) -> bool:
    return all(not np.array_equal(grid[:, c], grid[:, c + 1]) for c in range(grid.shape[1] - 1))


def _seed_is_valid(seed: np.ndarray) -> bool:
    h, w = map(int, seed.shape)
    colors = sorted(set(map(int, seed.flat)))
    if BACKGROUND not in colors or len(colors) != 2:
        return False

    foreground = [color for color in colors if color != BACKGROUND]
    if len(foreground) != 1:
        return False

    fg_positions = np.argwhere(seed != BACKGROUND)
    fg_count = int(len(fg_positions))
    min_fg, max_fg = SEED_FOREGROUND_RANGE
    if not (min_fg <= fg_count <= min(max_fg, h * w - 1)):
        return False

    # Make both axes matter in the final stamp.
    if len(set(map(int, fg_positions[:, 0]))) < 2:
        return False
    if len(set(map(int, fg_positions[:, 1]))) < 2:
        return False

    if not _adjacent_rows_distinct(seed):
        return False
    if not _adjacent_cols_distinct(seed):
        return False

    # Avoid a solid foreground bbox, which makes the self-similar copy visually less informative.
    rows = fg_positions[:, 0]
    cols = fg_positions[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    bbox = seed[r0 : r1 + 1, c0 : c1 + 1]
    if np.all(bbox != BACKGROUND):
        return False

    return True


def _random_seed(rng: np.random.Generator) -> np.ndarray | None:
    size_lo, size_hi = SEED_SIZE_RANGE
    h = int(rng.integers(size_lo, size_hi + 1))
    w = int(rng.integers(size_lo, size_hi + 1))
    fg = int(rng.choice(FOREGROUND_COLORS))

    min_fg, max_fg = SEED_FOREGROUND_RANGE
    fg_count = int(rng.integers(min_fg, min(max_fg, h * w - 1) + 1))
    flat_indices = rng.choice(h * w, size=fg_count, replace=False)

    seed = np.zeros((h, w), dtype=int)
    seed.flat[flat_indices] = fg
    if not _seed_is_valid(seed):
        return None
    return seed


def _stretch_seed(seed: np.ndarray, row_runs: tuple[int, ...], col_runs: tuple[int, ...]) -> np.ndarray:
    stretched = np.repeat(seed, row_runs, axis=0)
    stretched = np.repeat(stretched, col_runs, axis=1)
    return stretched.astype(int, copy=False)


def _semantic_checks(seed: np.ndarray, row_runs: tuple[int, ...], col_runs: tuple[int, ...], inp: np.ndarray) -> bool:
    if not (INPUT_SIZE_RANGE[0] <= inp.shape[0] <= INPUT_SIZE_RANGE[1]):
        return False
    if not (INPUT_SIZE_RANGE[0] <= inp.shape[1] <= INPUT_SIZE_RANGE[1]):
        return False

    recovered = _recover_seed(inp)
    if not np.array_equal(recovered, seed):
        return False

    row_only = _dedupe_rows(inp)
    col_only = _dedupe_cols(inp)
    if row_only.shape == seed.shape:
        return False
    if col_only.shape == seed.shape:
        return False

    expected = _stamp_seed(seed)
    if not np.array_equal(main(inp), expected):
        return False

    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False
    if np.count_nonzero(expected) <= np.count_nonzero(seed):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        seed = _random_seed(rng)
        if seed is None:
            continue

        row_runs = _choose_runs(seed.shape[0], rng)
        col_runs = _choose_runs(seed.shape[1], rng)

        # 3x3 seeds necessarily use 4x repeats on both axes to stay within the input-size range.
        # Keep some of them for coverage, but do not let that simplest family dominate.
        if seed.shape == (3, 3) and float(rng.random()) < 0.7:
            continue

        inp = _stretch_seed(seed, row_runs, col_runs)

        if _semantic_checks(seed, row_runs, col_runs, inp):
            return inp
    raise RuntimeError("failed to generate a valid input")
