import numpy as np


DIVIDER_COLOR = 6
BACKGROUND = 0
PALETTE = np.array([1, 2, 3, 4, 5, 7], dtype=int)
GENERATION_ATTEMPTS = 400


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            keep.append(i)
    return keep


def _compress_duplicate_runs(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(grid)
    col_starts = _run_start_indices(grid.T)
    return grid[np.ix_(row_starts, col_starts)].copy()


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == DIVIDER_COLOR, axis=1))
    assert rows.size == 1, f"expected one horizontal divider row, got {rows.tolist()}"
    return int(rows[0])


def _trim_trailing_zeros(header: np.ndarray) -> np.ndarray:
    end = header.size
    while end > 0 and int(header[end - 1]) == BACKGROUND:
        end -= 1
    return header[:end].copy()


def _diagonal_table(header: np.ndarray) -> np.ndarray:
    idx = np.maximum.outer(np.arange(header.size), np.arange(header.size))
    return header[idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden header by compressing the upper stretched matrix, then use it to fill the blank lower square with a diagonal-index table.

    Input structure:
    - The grid has black background below a single full-width horizontal magenta divider.
    - Above the divider, every cell belongs to a stretched color matrix with no black gaps.
    - The stretched matrix comes from repeating the rows and columns of a smaller diagonal-index matrix.

    Rule:
    - Keep the first row from each maximal run of identical neighboring rows in the upper matrix, and do the same for columns.
    - Read the compressed matrix's top row as the header, removing any trailing black cells.
    - Repeat that active header once to length 2K, where K is the trimmed header length, and fill the lower square by color `header[max(r,c)]`.
    - Leave the upper area and the divider unchanged.

    Key insight:
    - The instruction sequence is not directly visible in the input; it must be decoded by collapsing the redundant row and column runs first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_r = _find_divider_row(g)
    assert 0 < divider_r < g.shape[0] - 1

    upper = g[:divider_r]
    lower = g[divider_r + 1 :]
    assert upper.size > 0 and lower.size > 0
    assert np.all(upper != BACKGROUND), "upper area must be fully occupied"
    assert np.all(lower == BACKGROUND), "lower area must start blank"

    compressed = _compress_duplicate_runs(upper)
    assert compressed.shape[0] == compressed.shape[1], "compressed source must be square"
    header = _trim_trailing_zeros(compressed[0])
    assert header.size > 0
    assert np.array_equal(compressed[:, 0][: header.size], header)

    out_side = header.size * 2
    assert lower.shape == (out_side, out_side), f"expected blank target square of shape {(out_side, out_side)}, got {lower.shape}"

    doubled = np.resize(header, out_side)
    filled = _diagonal_table(doubled)

    out = g.copy()
    out[divider_r + 1 :, :] = filled
    return out


def _sample_header(rng: np.random.Generator, size: int) -> np.ndarray:
    distinct = int(rng.integers(2, min(5, size) + 1))
    palette = rng.choice(PALETTE, size=distinct, replace=False)
    header = np.empty(size, dtype=int)
    header[0] = int(rng.choice(palette))
    used = {int(header[0])}
    for i in range(1, size):
        choices = [int(c) for c in palette if int(c) != int(header[i - 1])]
        if size - i == len(set(map(int, palette))) - len(used):
            choices = [c for c in choices if c not in used]
        header[i] = int(rng.choice(choices))
        used.add(int(header[i]))
    if len(set(map(int, header))) < 2:
        return _sample_header(rng, size)
    return header


def _sample_row_runs(rng: np.random.Generator, size: int) -> np.ndarray:
    runs = rng.integers(2, 4, size=size)
    if np.all(runs == runs[0]):
        idx = int(rng.integers(size))
        runs[idx] = 5 - int(runs[idx])
    return runs.astype(int)


def _build_stretched_source(header: np.ndarray, row_runs: np.ndarray) -> np.ndarray:
    base = _diagonal_table(header)
    col_runs = np.full(header.size, 2, dtype=int)
    return np.repeat(np.repeat(base, row_runs, axis=0), col_runs, axis=1)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    source_size = int(rng.integers(3, 6))
    header = _sample_header(rng, source_size)
    row_runs = _sample_row_runs(rng, source_size)
    stretched = _build_stretched_source(header, row_runs)

    lower_side = source_size * 2
    if stretched.shape[1] != lower_side:
        return None

    grid = np.zeros((stretched.shape[0] + 1 + lower_side, lower_side), dtype=int)
    grid[: stretched.shape[0], :] = stretched
    grid[stretched.shape[0], :] = DIVIDER_COLOR

    out = main(grid)
    expected = np.vstack([stretched, np.full((1, lower_side), DIVIDER_COLOR, dtype=int), _diagonal_table(np.resize(header, lower_side))])
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
