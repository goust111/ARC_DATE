import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

MASK_HEIGHT = 5
GAP_ROW_MIN = 1
WIDTH_RANGE = (9, 13)
ACTIVE_COUNT_CHOICES = (2, 3, 4)
GEN_RETRIES = 400


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == YELLOW, axis=1))
    assert rows.size == 1, "expected exactly one yellow separator row"
    return int(rows[0])


def _wall_rows(grid: np.ndarray) -> list[int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.any(grid[r] == GRAY)]
    assert rows, "expected a gray wall band"
    assert rows == list(range(rows[0], rows[-1] + 1)), "wall rows must form one contiguous band"
    return rows


def _wall_span(row: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(row == GRAY)
    assert 2 <= cols.size <= 4, "each wall row must have 2..4 gray pixels"
    return int(cols[0]), int(cols[-1])


def _parse_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    wall_rows = _wall_rows(g)
    mask_h = len(wall_rows)

    assert sep >= mask_h, "top mask must fit above separator"
    top = g[sep - mask_h : sep]
    bottom = g[sep + 1 : sep + 1 + mask_h]
    assert top.shape == bottom.shape == (mask_h, g.shape[1]), "masks must have equal shape"
    assert np.all(np.isin(top, (BLACK, RED))), "top mask must be binary red/black"
    assert np.all(np.isin(bottom, (BLACK, RED))), "bottom mask must be binary red/black"

    gap = g[sep + 1 + mask_h : wall_rows[0]]
    assert gap.shape[0] >= GAP_ROW_MIN, "expected at least one gap row before the walls"
    assert not np.any(gap == GRAY), "gap rows cannot contain wall pixels"
    assert all(g.shape[1] == top.shape[1] for _ in wall_rows)
    return top, bottom, wall_rows


def _active_rows(top: np.ndarray, bottom: np.ndarray) -> list[int]:
    return [int(r) for r in range(top.shape[0]) if np.any(top[r] != bottom[r])]


def _solve_from_parts(top: np.ndarray, bottom: np.ndarray, wall_band: np.ndarray) -> np.ndarray:
    active = _active_rows(top, bottom)
    assert active, "expected at least one active xor row"
    assert len(active) <= wall_band.shape[0], "not enough wall rows for active rows"

    out = np.zeros_like(top)
    for order, row_idx in enumerate(active):
        left, right = _wall_span(wall_band[order])
        out[row_idx, left : right + 1] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the XOR-selected mask rows using scanline spans defined by ordered wall rows.

    Input structure:
    - A full yellow separator row splits two same-sized red/black binary masks.
    - Below the lower mask, after at least one blank gap row, sits a contiguous band of gray wall rows.
    - Each wall row contains 2..4 gray pixels whose leftmost and rightmost columns define a horizontal span.

    Rule:
    - Compare the top and bottom masks row by row; a row is active iff the two masks differ anywhere on that row.
    - Enumerate active rows from top to bottom, then pair the k-th active mask row with the k-th wall row.
    - In the output, paint green across the paired wall row's leftmost-to-rightmost gray span on that active row.
    - Non-active rows stay black, and the output size matches one mask.

    Key insight:
    - The XOR result is only a row selector; the visible fill width comes from the wall band, and the pairing is by activation order rather than by row index.
    """
    top, bottom, wall_rows = _parse_layout(grid)
    wall_band = np.asarray(grid, dtype=int)[wall_rows]
    return _solve_from_parts(top, bottom, wall_band)


def _random_mask_row(width: int, rng: np.random.Generator) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    style = int(rng.integers(0, 4))
    if style == 0:
        return row
    if style == 1:
        seg_len = int(rng.integers(1, max(2, width // 2 + 1)))
        start = int(rng.integers(0, width - seg_len + 1))
        row[start : start + seg_len] = RED
        return row
    if style == 2 and width >= 5:
        cut1 = int(rng.integers(1, max(2, width // 3 + 1)))
        cut2 = int(rng.integers(cut1 + 1, width))
        row[max(0, cut1 - 1) : cut1 + 1] = RED
        row[cut2 : min(width, cut2 + 2)] = RED
        return row

    count = int(rng.integers(1, max(2, width // 2 + 1)))
    cols = rng.choice(width, size=count, replace=False)
    row[np.asarray(cols, dtype=int)] = RED
    return row


def _flip_some_cells(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = row.copy()
    width = row.size
    flip_count = int(rng.integers(1, min(4, width) + 1))
    cols = rng.choice(width, size=flip_count, replace=False)
    cols = np.asarray(cols, dtype=int)
    out[cols] = np.where(out[cols] == RED, BLACK, RED)
    assert np.any(out != row)
    return out


def _choose_active_rows(rng: np.random.Generator) -> list[int]:
    for _ in range(100):
        count = int(rng.choice(ACTIVE_COUNT_CHOICES))
        rows = sorted(int(v) for v in rng.choice(MASK_HEIGHT, size=count, replace=False))
        if rows != list(range(count)):
            return rows
    raise RuntimeError("failed to choose a non-prefix active-row set")


def _choose_intervals(width: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    candidates = [
        (left, right)
        for left in range(width - 1)
        for right in range(left + 1, width)
        if right - left + 1 >= 3
    ]
    rng.shuffle(candidates)
    return [(int(left), int(right)) for left, right in candidates[:MASK_HEIGHT]]


def _wall_cols_for_interval(left: int, right: int, rng: np.random.Generator) -> list[int]:
    span_cols = list(range(left, right + 1))
    max_count = min(4, len(span_cols))
    count = int(rng.integers(2, max_count + 1))
    chosen = {left, right}
    interior = span_cols[1:-1]
    if interior and count > 2:
        extra = rng.choice(interior, size=count - 2, replace=False)
        chosen.update(int(v) for v in np.asarray(extra, dtype=int).tolist())
    return sorted(chosen)


def _build_masks(width: int, active_rows: list[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros((MASK_HEIGHT, width), dtype=int)
    bottom = np.zeros((MASK_HEIGHT, width), dtype=int)
    active_set = set(active_rows)

    for r in range(MASK_HEIGHT):
        base = _random_mask_row(width, rng)
        if r in active_set:
            if rng.random() < 0.5:
                bottom[r] = base
                top[r] = _flip_some_cells(base, rng)
            else:
                top[r] = base
                bottom[r] = _flip_some_cells(base, rng)
        else:
            top[r] = base
            bottom[r] = base.copy()

    return top, bottom


def _build_grid(
    top: np.ndarray,
    bottom: np.ndarray,
    intervals: list[tuple[int, int]],
    rng: np.random.Generator,
    gap_rows: int,
) -> np.ndarray:
    width = top.shape[1]
    total_rows = MASK_HEIGHT + 1 + MASK_HEIGHT + gap_rows + MASK_HEIGHT
    grid = np.zeros((total_rows, width), dtype=int)

    grid[:MASK_HEIGHT] = top
    grid[MASK_HEIGHT] = YELLOW
    grid[MASK_HEIGHT + 1 : 2 * MASK_HEIGHT + 1] = bottom

    wall_start = 2 * MASK_HEIGHT + 1 + gap_rows
    for idx, (left, right) in enumerate(intervals):
        cols = _wall_cols_for_interval(left, right, rng)
        grid[wall_start + idx, cols] = GRAY

    return grid


def _wrong_row_index_output(top: np.ndarray, bottom: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(top)
    for row_idx in _active_rows(top, bottom):
        left, right = intervals[row_idx]
        out[row_idx, left : right + 1] = GREEN
    return out


def _validate_generated(grid: np.ndarray) -> None:
    top, bottom, wall_rows = _parse_layout(grid)
    wall_band = np.asarray(grid, dtype=int)[wall_rows]
    active = _active_rows(top, bottom)
    assert 2 <= len(active) <= 4, "generator must cover active-row branch cases"
    assert active != list(range(len(active))), "ordered pairing must differ from row-index pairing"

    spans = [_wall_span(row) for row in wall_band]
    assert len(set(spans[: len(active)])) == len(active), "used wall spans must be distinct"
    assert len(set(spans)) >= len(active) + 1, "wall band should contain unused alternative spans"

    out = _solve_from_parts(top, bottom, wall_band)
    assert out.shape == top.shape

    for order, row_idx in enumerate(active):
        left, right = spans[order]
        assert np.all(out[row_idx, left : right + 1] == GREEN)
        if left > 0:
            assert np.all(out[row_idx, :left] == BLACK)
        if right + 1 < out.shape[1]:
            assert np.all(out[row_idx, right + 1 :] == BLACK)

    inactive = sorted(set(range(MASK_HEIGHT)) - set(active))
    assert inactive, "xor step must matter"
    for row_idx in inactive:
        assert np.all(out[row_idx] == BLACK)

    wrong = _wrong_row_index_output(top, bottom, spans)
    assert not np.array_equal(out, wrong), "ordered pairing must be necessary"

    all_rows = np.zeros_like(out)
    for r, (left, right) in enumerate(spans):
        all_rows[r, left : right + 1] = GREEN
    assert not np.array_equal(out, all_rows), "xor row selection must be necessary"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    active_rows = _choose_active_rows(rng)
    top, bottom = _build_masks(width, active_rows, rng)
    intervals = _choose_intervals(width, rng)
    gap_rows = int(rng.integers(GAP_ROW_MIN, GAP_ROW_MIN + 2))
    grid = _build_grid(top, bottom, intervals, rng, gap_rows)

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid XOR-wall-fill input")
