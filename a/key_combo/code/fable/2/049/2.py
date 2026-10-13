import random

import numpy as np


def _compress_adjacent_rows(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0

    representatives = [g[0].copy()]
    counts = [1]
    for r in range(1, g.shape[0]):
        if np.array_equal(g[r], representatives[-1]):
            counts[-1] += 1
        else:
            representatives.append(g[r].copy())
            counts.append(1)
    return np.vstack(representatives), counts


def _compress_adjacent_cols(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] > 0

    representatives = [g[:, 0].copy()]
    counts = [1]
    for c in range(1, g.shape[1]):
        if np.array_equal(g[:, c], representatives[-1]):
            counts[-1] += 1
        else:
            representatives.append(g[:, c].copy())
            counts.append(1)
    return np.column_stack(representatives), counts


def _stretch(core: np.ndarray, row_repeats: list[int], col_repeats: list[int]) -> np.ndarray:
    g = np.asarray(core, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == len(row_repeats)
    assert g.shape[1] == len(col_repeats)
    assert all(k >= 1 for k in row_repeats)
    assert all(k >= 1 for k in col_repeats)
    return np.repeat(np.repeat(g, row_repeats, axis=0), col_repeats, axis=1)


def _decompose(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    row_compressed, row_repeats = _compress_adjacent_rows(g)
    core, col_repeats = _compress_adjacent_cols(row_compressed)

    n_rows, n_cols = core.shape
    assert n_rows == n_cols, "the compressed answer matrix must be square"
    assert len(row_repeats) == len(col_repeats) == n_rows
    assert n_rows >= 2, "a nontrivial rhythm swap needs at least a 2x2 answer matrix"

    for i in range(n_rows - 1):
        assert not np.array_equal(core[i], core[i + 1]), "adjacent answer rows must stay distinguishable"
        assert not np.array_equal(core[:, i], core[:, i + 1]), "adjacent answer cols must stay distinguishable"

    rebuilt = _stretch(core, row_repeats, col_repeats)
    assert np.array_equal(rebuilt, g), "input must be exactly a row/col-stretched square color matrix"
    return core, row_repeats, col_repeats


def _row_signatures(core: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(map(int, row)) for row in np.asarray(core, dtype=int)}


def _col_signatures(core: np.ndarray) -> set[tuple[int, ...]]:
    g = np.asarray(core, dtype=int)
    return {tuple(map(int, g[:, c])) for c in range(g.shape[1])}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the row and column stretch rhythms of a stretched block mosaic while keeping the compressed color matrix fixed.

    Input structure:
    - The grid has no background; every cell belongs to a filled rectangular color block.
    - It is formed from an n x n color matrix whose i-th answer row is repeated r_i times and whose j-th answer column is repeated c_j times.
    - Adjacent answer rows and adjacent answer columns are not identical, so maximal runs of equal full rows and equal full columns reveal the two stretch rhythms.

    Rule:
    - Compress maximal runs of identical adjacent rows to recover the row rhythm r and a row-compressed grid.
    - Compress maximal runs of identical adjacent columns in that row-compressed grid to recover the square color matrix and the column rhythm c.
    - Re-expand that same color matrix with the rhythms swapped: repeat answer row i exactly c_i times and answer column j exactly r_j times.

    Key insight:
    - The visible mosaic carries two independent channels of information: the color matrix and the row/column run lengths; the output keeps the color matrix in place and only exchanges the two run-length channels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    core, row_repeats, col_repeats = _decompose(g)
    return _stretch(core, col_repeats, row_repeats)


GEN_ATTEMPTS = 400
N_RANGE = (2, 6)
PALETTE_COLORS = tuple(range(1, 10))
PALETTE_SIZE_RANGE = (2, 5)
REPEAT_MIN = 1
REPEAT_MAX = 5


def _sample_repeats(n: int) -> list[int]:
    upper = min(REPEAT_MAX, 30 // n)
    assert upper >= REPEAT_MIN
    while True:
        seq = [random.randint(REPEAT_MIN, upper) for _ in range(n)]
        if len(set(seq)) >= 2:
            return seq


def _sample_core(n: int) -> np.ndarray | None:
    palette_size = random.randint(PALETTE_SIZE_RANGE[0], min(PALETTE_SIZE_RANGE[1], n * n))
    palette = random.sample(PALETTE_COLORS, palette_size)
    core = np.array([[random.choice(palette) for _ in range(n)] for _ in range(n)], dtype=int)

    if len(set(map(int, core.flat))) < 2:
        return None
    if np.array_equal(core, core.T):
        return None
    if len(_row_signatures(core)) != n:
        return None
    if len(_col_signatures(core)) != n:
        return None
    if any(len(set(map(int, core[r]))) < 2 for r in range(n)):
        return None
    if any(len(set(map(int, core[:, c]))) < 2 for c in range(n)):
        return None
    return core


def _valid_example(core: np.ndarray, row_repeats: list[int], col_repeats: list[int], grid: np.ndarray, out: np.ndarray) -> bool:
    n = core.shape[0]
    if core.shape != (n, n):
        return False
    if len(row_repeats) != n or len(col_repeats) != n:
        return False
    if len(set(row_repeats)) < 2 or len(set(col_repeats)) < 2:
        return False
    if row_repeats == col_repeats:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False
    if out.shape == grid.T.shape and np.array_equal(out, grid.T):
        return False

    row_compressed, rr = _compress_adjacent_rows(grid)
    recovered, cc = _compress_adjacent_cols(row_compressed)
    if rr != row_repeats or cc != col_repeats:
        return False
    if not np.array_equal(recovered, core):
        return False
    if not np.array_equal(main(grid), out):
        return False
    if not np.array_equal(main(out), grid):
        return False
    if len(_row_signatures(core)) != n or len(_col_signatures(core)) != n:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*N_RANGE)
    core = _sample_core(n)
    if core is None:
        return None

    row_repeats = _sample_repeats(n)
    col_repeats = _sample_repeats(n)
    if row_repeats == col_repeats:
        return None

    grid = _stretch(core, row_repeats, col_repeats)
    out = _stretch(core, col_repeats, row_repeats)

    if not _valid_example(core, row_repeats, col_repeats, grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rhythm-swap ARC input")
