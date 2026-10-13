from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

HEIGHT_RANGE = (13, 18)  # upper bound exclusive
WIDTH_RANGE = (13, 18)   # upper bound exclusive
VALID_COUNT_CHOICES = (2, 3)
ROW_MIX_PATTERNS = (
    (RED, BLUE),
    (BLUE, RED),
    (RED, BLACK),
    (BLACK, RED),
)
COL_MIX_PATTERNS = (
    (BLUE, RED),
    (RED, BLUE),
    (BLUE, BLACK),
    (BLACK, BLUE),
)
MAX_GENERATION_ATTEMPTS = 500
SPACED_SAMPLE_ATTEMPTS = 200


def _valid_horizontal_rows(grid: np.ndarray) -> list[int]:
    height, width = map(int, grid.shape)
    return [
        r
        for r in range(1, height - 1)
        if int(grid[r, 0]) == RED
        and int(grid[r, width - 1]) == RED
        and np.all(grid[r, 1 : width - 1] == BLACK)
    ]


def _valid_vertical_cols(grid: np.ndarray) -> list[int]:
    height, width = map(int, grid.shape)
    return [
        c
        for c in range(1, width - 1)
        if int(grid[0, c]) == BLUE
        and int(grid[height - 1, c]) == BLUE
        and np.all(grid[1 : height - 1, c] == BLACK)
    ]


def _expected_output(grid: np.ndarray, valid_rows: list[int], valid_cols: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    height, width = map(int, out.shape)

    if valid_rows:
        out[np.ix_(valid_rows, np.arange(1, width - 1, dtype=int))] = RED
    if valid_cols:
        out[np.ix_(np.arange(1, height - 1, dtype=int), valid_cols)] = BLUE
    if valid_rows and valid_cols:
        out[np.ix_(valid_rows, valid_cols)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only those edge-to-edge corridors whose opposite boundary endpoints have the required matching color, and mark row/column crossings yellow.

    Input structure:
    - A black-background grid has red and blue endpoint pixels scattered on the four borders.
    - Interior rows can have left/right border endpoints, interior columns can have top/bottom border endpoints.
    - Some interior cells may be grey blockers.

    Rule:
    - An interior row is a valid horizontal corridor only when its left and right border pixels are both red and every interior cell on that row is black.
    - An interior column is a valid vertical corridor only when its top and bottom border pixels are both blue and every interior cell on that column is black.
    - Fill valid horizontal corridors red, fill valid vertical corridors blue, and recolor cells where both kinds of corridors pass through the same interior cell to yellow.
    - Preserve every original border endpoint and every original grey blocker.

    Key insight:
    - Corridor existence is gated by two visible conditions at once: the path must be unobstructed and the opposite boundary endpoints must match the orientation-specific color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "grid must have an interior"

    valid_rows = _valid_horizontal_rows(g)
    valid_cols = _valid_vertical_cols(g)
    return _expected_output(g, valid_rows, valid_cols)


def _sample_spaced_indices(
    rng: np.random.Generator, start: int, stop: int, count: int
) -> list[int]:
    span = stop - start + 1
    compressed_span = span - (count - 1)
    if compressed_span < count:
        raise RuntimeError("not enough room for spaced indices")

    base = sorted(int(v) for v in rng.choice(compressed_span, size=count, replace=False))
    return [start + base_idx + offset for offset, base_idx in enumerate(base)]


def _paint_row_endpoints(grid: np.ndarray, rows: list[int], left: int, right: int) -> None:
    width = int(grid.shape[1])
    for r in rows:
        grid[r, 0] = left
        grid[r, width - 1] = right


def _paint_col_endpoints(grid: np.ndarray, cols: list[int], top: int, bottom: int) -> None:
    height = int(grid.shape[0])
    for c in cols:
        grid[0, c] = top
        grid[height - 1, c] = bottom


def _edge_color_counts(grid: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.count_nonzero(grid[1:-1, 0] != BLACK)),
        int(np.count_nonzero(grid[1:-1, -1] != BLACK)),
        int(np.count_nonzero(grid[0, 1:-1] != BLACK)),
        int(np.count_nonzero(grid[-1, 1:-1] != BLACK)),
    )


def _build_candidate(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))
    grid = np.zeros((height, width), dtype=int)

    n_valid_rows = int(rng.choice(VALID_COUNT_CHOICES))
    n_valid_cols = int(rng.choice(VALID_COUNT_CHOICES))

    row_roles = _sample_spaced_indices(rng, 1, height - 2, n_valid_rows + 3)
    col_roles = _sample_spaced_indices(rng, 1, width - 2, n_valid_cols + 3)
    row_roles = [int(v) for v in rng.permutation(row_roles)]
    col_roles = [int(v) for v in rng.permutation(col_roles)]

    valid_rows = sorted(row_roles[:n_valid_rows])
    blocked_row = int(row_roles[n_valid_rows])
    blueblue_row = int(row_roles[n_valid_rows + 1])
    mixed_row = int(row_roles[n_valid_rows + 2])

    valid_cols = sorted(col_roles[:n_valid_cols])
    blocked_col = int(col_roles[n_valid_cols])
    redred_col = int(col_roles[n_valid_cols + 1])
    mixed_col = int(col_roles[n_valid_cols + 2])

    _paint_row_endpoints(grid, valid_rows, RED, RED)
    _paint_row_endpoints(grid, [blocked_row], RED, RED)
    _paint_row_endpoints(grid, [blueblue_row], BLUE, BLUE)
    row_mix = tuple(int(v) for v in ROW_MIX_PATTERNS[int(rng.integers(len(ROW_MIX_PATTERNS)))])
    _paint_row_endpoints(grid, [mixed_row], row_mix[0], row_mix[1])

    _paint_col_endpoints(grid, valid_cols, BLUE, BLUE)
    _paint_col_endpoints(grid, [blocked_col], BLUE, BLUE)
    _paint_col_endpoints(grid, [redred_col], RED, RED)
    col_mix = tuple(int(v) for v in COL_MIX_PATTERNS[int(rng.integers(len(COL_MIX_PATTERNS)))])
    _paint_col_endpoints(grid, [mixed_col], col_mix[0], col_mix[1])

    grid[blocked_row, blocked_col] = GREY

    protected_rows = set(valid_rows) | {blueblue_row, mixed_row}
    protected_cols = set(valid_cols) | {redred_col, mixed_col}
    blocker_rows = [r for r in range(1, height - 1) if r not in protected_rows]
    blocker_cols = [c for c in range(1, width - 1) if c not in protected_cols]
    extra_candidates = [
        (r, c)
        for r in blocker_rows
        for c in blocker_cols
        if (r, c) != (blocked_row, blocked_col)
    ]
    if extra_candidates:
        n_extra = int(rng.integers(0, min(4, len(extra_candidates)) + 1))
        if n_extra:
            picks = rng.choice(len(extra_candidates), size=n_extra, replace=False)
            for idx in np.atleast_1d(picks):
                r, c = extra_candidates[int(idx)]
                grid[r, c] = GREY

    meta = {
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "blocked_row": blocked_row,
        "blocked_col": blocked_col,
        "blueblue_row": blueblue_row,
        "redred_col": redred_col,
        "mixed_row": mixed_row,
        "mixed_col": mixed_col,
        "row_mix": row_mix,
        "col_mix": col_mix,
    }
    return grid, meta


def _row_is_clear(grid: np.ndarray, row: int) -> bool:
    return bool(np.all(grid[row, 1:-1] == BLACK))


def _col_is_clear(grid: np.ndarray, col: int) -> bool:
    return bool(np.all(grid[1:-1, col] == BLACK))


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    valid_rows = _valid_horizontal_rows(grid)
    valid_cols = _valid_vertical_cols(grid)

    if valid_rows != meta["valid_rows"] or valid_cols != meta["valid_cols"]:
        return False
    if len(valid_rows) < 2 or len(valid_cols) < 2:
        return False

    edge_counts = _edge_color_counts(grid)
    if not all(3 <= count <= 6 for count in edge_counts):
        return False

    blocked_row = int(meta["blocked_row"])
    blocked_col = int(meta["blocked_col"])
    blueblue_row = int(meta["blueblue_row"])
    redred_col = int(meta["redred_col"])
    mixed_row = int(meta["mixed_row"])
    mixed_col = int(meta["mixed_col"])

    if not (
        int(grid[blocked_row, 0]) == RED
        and int(grid[blocked_row, -1]) == RED
        and np.any(grid[blocked_row, 1:-1] == GREY)
        and blocked_row not in valid_rows
    ):
        return False
    if not (
        int(grid[0, blocked_col]) == BLUE
        and int(grid[-1, blocked_col]) == BLUE
        and np.any(grid[1:-1, blocked_col] == GREY)
        and blocked_col not in valid_cols
    ):
        return False
    if not (
        int(grid[blueblue_row, 0]) == BLUE
        and int(grid[blueblue_row, -1]) == BLUE
        and _row_is_clear(grid, blueblue_row)
        and blueblue_row not in valid_rows
    ):
        return False
    if not (
        int(grid[0, redred_col]) == RED
        and int(grid[-1, redred_col]) == RED
        and _col_is_clear(grid, redred_col)
        and redred_col not in valid_cols
    ):
        return False
    if not (
        tuple(int(v) for v in (grid[mixed_row, 0], grid[mixed_row, -1])) == meta["row_mix"]
        and _row_is_clear(grid, mixed_row)
        and mixed_row not in valid_rows
    ):
        return False
    if not (
        tuple(int(v) for v in (grid[0, mixed_col], grid[-1, mixed_col])) == meta["col_mix"]
        and _col_is_clear(grid, mixed_col)
        and mixed_col not in valid_cols
    ):
        return False

    if int(np.count_nonzero(grid == GREY)) < 1:
        return False

    expected = _expected_output(grid, valid_rows, valid_cols)
    if expected[valid_rows[0], valid_cols[0]] != YELLOW:
        return False
    if expected[blocked_row, blocked_col] != GREY:
        return False
    if expected[blocked_row, valid_cols[0]] != BLUE:
        return False
    if expected[valid_rows[0], blocked_col] != RED:
        return False
    if not np.array_equal(expected[:, 0], grid[:, 0]):
        return False
    if not np.array_equal(expected[:, -1], grid[:, -1]):
        return False
    if not np.array_equal(expected[0, :], grid[0, :]):
        return False
    if not np.array_equal(expected[-1, :], grid[-1, :]):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, meta = _build_candidate(rng)
        if _semantic_checks(grid, meta):
            return grid
    raise RuntimeError(f"failed to generate a valid corridor-crossing input after {MAX_GENERATION_ATTEMPTS} attempts")
