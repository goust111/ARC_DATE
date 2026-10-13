import numpy as np


MAX_RETRIES = 300
N_RANGE = (4, 9)
MARGIN_RANGE = (0, 4)
MATRIX_COLOR_COUNT_RANGE = (3, 6)
SHIFT_STYLE_COUNT = 3
MATRIX_STYLE_COUNT = 3


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "expected at least one non-background cell"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _decode_left_control(block: np.ndarray, bg: int) -> np.ndarray:
    n_rows, n_cols = map(int, block.shape)
    assert n_rows == n_cols, "left control must be square"
    fg = block[block != bg]
    assert fg.size > 0, "left control must contain foreground cells"
    assert len(np.unique(fg)) == 1, "left control must be monochrome except for gaps"

    shifts = np.zeros(n_rows, dtype=int)
    for r in range(n_rows):
        gaps = np.flatnonzero(block[r] == bg)
        assert len(gaps) == 1, "each matrix row needs exactly one left-control gap"
        shifts[r] = int(gaps[0])
    return shifts


def _decode_top_control(block: np.ndarray, bg: int) -> np.ndarray:
    n_rows, n_cols = map(int, block.shape)
    assert n_rows == n_cols, "top control must be square"
    fg = block[block != bg]
    assert fg.size > 0, "top control must contain foreground cells"
    assert len(np.unique(fg)) == 1, "top control must be monochrome except for gaps"

    shifts = np.zeros(n_cols, dtype=int)
    for c in range(n_cols):
        gaps = np.flatnonzero(block[:, c] == bg)
        assert len(gaps) == 1, "each matrix column needs exactly one top-control gap"
        shifts[c] = int(gaps[0])
    return shifts


def _apply_shifts(matrix: np.ndarray, row_shifts: np.ndarray, col_shifts: np.ndarray) -> np.ndarray:
    n = int(matrix.shape[0])
    assert matrix.shape == (n, n)
    assert row_shifts.shape == (n,)
    assert col_shifts.shape == (n,)

    shifted = matrix.copy()
    for r, amount in enumerate(row_shifts):
        shifted[r] = np.roll(shifted[r], int(amount) % n)

    out = shifted.copy()
    for c, amount in enumerate(col_shifts):
        out[:, c] = np.roll(shifted[:, c], int(amount) % n)
    return out


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bg = _get_bg(grid)
    r0, c0, r1, c1 = _non_bg_bbox(grid, bg)
    content = np.asarray(grid[r0:r1, c0:c1], dtype=int)
    h, w = map(int, content.shape)

    assert h == w and h % 2 == 0, "content bbox must be an even square"
    n = h // 2

    top_left = content[:n, :n]
    top_control = content[:n, n:]
    left_control = content[n:, :n]
    matrix = content[n:, n:]

    assert np.all(top_left == bg), "top-left quadrant must stay background"
    assert matrix.shape == (n, n)
    assert np.all(matrix != bg), "matrix region must not contain the background color"
    assert len(np.unique(matrix)) >= 2, "matrix should contain multiple colors"
    return left_control, top_control, matrix, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode row and column cyclic shifts from two control blocks and apply them to the
    interior matrix. The left control block stores one background gap per matrix row;
    the gap's column index is that row's rightward shift. The top control block stores
    one background gap per matrix column; the gap's row index is that column's
    downward shift. The output keeps only the transformed matrix.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    left_control, top_control, matrix, bg = _extract_parts(g)
    row_shifts = _decode_left_control(left_control, bg)
    col_shifts = _decode_top_control(top_control, bg)
    return _apply_shifts(matrix, row_shifts, col_shifts)


def _segment_labels(rng: np.random.Generator, n: int, parts: int) -> np.ndarray:
    assert 1 < parts <= n
    cuts = sorted(rng.choice(np.arange(1, n), size=parts - 1, replace=False).tolist())
    bounds = [0] + cuts + [n]
    labels = np.zeros(n, dtype=int)
    for idx in range(parts):
        labels[bounds[idx] : bounds[idx + 1]] = idx
    return labels


def _build_matrix(rng: np.random.Generator, n: int, palette: np.ndarray) -> np.ndarray:
    k = int(len(palette))
    style = int(rng.integers(SHIFT_STYLE_COUNT))

    if style == 0:
        row_codes = rng.integers(0, k, size=n)
        col_codes = rng.integers(0, k, size=n)
        matrix = palette[(row_codes[:, None] + col_codes[None, :]) % k]
    elif style == 1:
        row_parts = int(rng.integers(2, min(4, n) + 1))
        col_parts = int(rng.integers(2, min(4, n) + 1))
        row_labels = _segment_labels(rng, n, row_parts)
        col_labels = _segment_labels(rng, n, col_parts)
        offset = int(rng.integers(0, k))
        matrix = palette[(2 * row_labels[:, None] + col_labels[None, :] + offset) % k]
    else:
        row_step = int(rng.integers(1, k + 1))
        col_step = int(rng.integers(1, k + 1))
        offset = int(rng.integers(0, k))
        rr = np.arange(n, dtype=int)[:, None]
        cc = np.arange(n, dtype=int)[None, :]
        matrix = palette[(offset + row_step * rr + col_step * cc) % k]

    accent_count = int(rng.integers(1, max(2, n // 2) + 1))
    for _ in range(accent_count):
        if bool(rng.integers(2)):
            r = int(rng.integers(0, n))
            matrix[r, :] = np.roll(matrix[r, :], int(rng.integers(0, n)))
        else:
            c = int(rng.integers(0, n))
            matrix[:, c] = np.roll(matrix[:, c], int(rng.integers(0, n)))

    if len(np.unique(matrix)) < min(3, k):
        return _build_matrix(rng, n, palette)
    if any(len(np.unique(matrix[r])) == 1 for r in range(n)):
        return _build_matrix(rng, n, palette)
    if any(len(np.unique(matrix[:, c])) == 1 for c in range(n)):
        return _build_matrix(rng, n, palette)
    return matrix


def _build_shift_vector(rng: np.random.Generator, n: int) -> np.ndarray:
    style = int(rng.integers(MATRIX_STYLE_COUNT))
    if style == 0:
        shifts = rng.integers(0, n, size=n)
    elif style == 1:
        shifts = np.zeros(n, dtype=int)
        active = int(rng.integers(max(2, n // 2), n + 1))
        rows = rng.choice(n, size=active, replace=False)
        shifts[rows] = rng.integers(1, n, size=active)
    else:
        choices = rng.choice(np.arange(n), size=int(rng.integers(2, min(4, n) + 1)), replace=False)
        shifts = rng.choice(choices, size=n, replace=True)

    if np.all(shifts == 0):
        idx = int(rng.integers(0, n))
        shifts[idx] = int(rng.integers(1, n))
    return shifts.astype(int)


def _assemble_input(
    bg: int,
    left_color: int,
    top_color: int,
    matrix: np.ndarray,
    row_shifts: np.ndarray,
    col_shifts: np.ndarray,
    margins: tuple[int, int, int, int],
) -> np.ndarray:
    n = int(matrix.shape[0])
    top_margin, bottom_margin, left_margin, right_margin = map(int, margins)
    grid = np.full((top_margin + 2 * n + bottom_margin, left_margin + 2 * n + right_margin), bg, dtype=int)

    r0 = top_margin
    c0 = left_margin
    grid[r0 : r0 + n, c0 + n : c0 + 2 * n] = top_color
    grid[r0 + n : r0 + 2 * n, c0 : c0 + n] = left_color
    grid[r0 + n : r0 + 2 * n, c0 + n : c0 + 2 * n] = matrix

    for r, amount in enumerate(row_shifts):
        grid[r0 + n + r, c0 + int(amount)] = bg
    for c, amount in enumerate(col_shifts):
        grid[r0 + int(amount), c0 + n + c] = bg

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(*N_RANGE))
    colors = rng.permutation(10)
    bg = int(colors[0])
    left_color = int(colors[1])
    top_color = int(colors[2])
    palette_size = int(rng.integers(*MATRIX_COLOR_COUNT_RANGE))
    palette = np.asarray(colors[3 : 3 + palette_size], dtype=int)
    if len(palette) < 3:
        return None

    matrix = _build_matrix(rng, n, palette)
    row_shifts = _build_shift_vector(rng, n)
    col_shifts = _build_shift_vector(rng, n)

    if len(np.unique(row_shifts)) == 1 and len(np.unique(col_shifts)) == 1:
        return None
    if np.all(row_shifts == 0) or np.all(col_shifts == 0):
        return None

    transformed = _apply_shifts(matrix, row_shifts, col_shifts)
    if np.array_equal(matrix, transformed):
        return None

    margins = tuple(int(rng.integers(*MARGIN_RANGE)) for _ in range(4))
    grid = _assemble_input(bg, left_color, top_color, matrix, row_shifts, col_shifts, margins)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, transformed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
