from __future__ import annotations

import numpy as np


BG = 0
VERTICAL_BLUE = 1
HORIZONTAL_RED = 2
INTERFERENCE_YELLOW = 4
ALT_GREY = 5

_ALLOWED_INPUT_COLORS = {BG, VERTICAL_BLUE, HORIZONTAL_RED}
_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0

_GEN_HEIGHT_RANGE = (13, 21)  # numpy upper bound is exclusive
_GEN_WIDTH_RANGE = (13, 21)
_GEN_BAND_SIZE_PLAN = ((2, 2), (2, 3), (3, 2), (3, 3))
_GEN_RELATION_PLAN = ("mismatch", "match", "mismatch")
_GEN_MARGIN = 3
_GEN_MAX_ATTEMPTS = 200


def _is_contiguous_run(indices: np.ndarray, allowed_lengths: set[int]) -> bool:
    if indices.size not in allowed_lengths:
        return False
    return bool(np.all(indices == np.arange(indices[0], indices[0] + indices.size)))


def _detect_bands(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)

    horizontal_rows = np.flatnonzero(np.all(grid == HORIZONTAL_RED, axis=1))
    assert _is_contiguous_run(horizontal_rows, {2, 3}), "expected one 2-3 row full-width red band"

    vertical_cols = np.flatnonzero(np.all(grid != BG, axis=0))
    assert _is_contiguous_run(vertical_cols, {2, 3}), "expected one 2-3 column full-height vertical band"

    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[horizontal_rows] = True
    col_mask[vertical_cols] = True

    non_band_rows = np.flatnonzero(~row_mask)
    non_band_cols = np.flatnonzero(~col_mask)

    assert horizontal_rows.size * vertical_cols.size >= 4, "intersection must cover at least four cells"
    assert np.all(grid[np.ix_(horizontal_rows, vertical_cols)] == HORIZONTAL_RED), "overlap must be red in input"
    assert np.all(grid[np.ix_(horizontal_rows, non_band_cols)] == HORIZONTAL_RED), "horizontal band must be solid red"
    assert np.all(grid[np.ix_(non_band_rows, vertical_cols)] == VERTICAL_BLUE), "vertical band must be blue outside overlap"
    assert np.all(grid[np.ix_(non_band_rows, non_band_cols)] == BG), "background must stay black"

    return horizontal_rows, vertical_cols


def _expected_output(
    shape: tuple[int, int],
    horizontal_rows: np.ndarray,
    vertical_cols: np.ndarray,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)

    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[horizontal_rows] = True
    col_mask[vertical_cols] = True

    non_band_rows = np.flatnonzero(~row_mask)
    non_band_cols = np.flatnonzero(~col_mask)

    if non_band_cols.size:
        horizontal_pattern = np.where(non_band_cols % 2 == 0, HORIZONTAL_RED, ALT_GREY)
        out[np.ix_(horizontal_rows, non_band_cols)] = horizontal_pattern

    if non_band_rows.size:
        vertical_pattern = np.where(non_band_rows % 2 == 0, VERTICAL_BLUE, ALT_GREY)[:, None]
        out[np.ix_(non_band_rows, vertical_cols)] = vertical_pattern

    cross_rows, cross_cols = np.meshgrid(horizontal_rows, vertical_cols, indexing="ij")
    out[np.ix_(horizontal_rows, vertical_cols)] = np.where(
        (cross_rows % 2) == (cross_cols % 2),
        INTERFERENCE_YELLOW,
        BG,
    )
    return out


def _wrong_local_origin_output(
    shape: tuple[int, int],
    horizontal_rows: np.ndarray,
    vertical_cols: np.ndarray,
) -> np.ndarray:
    """Shortcut oracle: restart parity at the overlap's top-left corner."""
    h, w = shape
    out = np.full((h, w), BG, dtype=int)

    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[horizontal_rows] = True
    col_mask[vertical_cols] = True

    non_band_rows = np.flatnonzero(~row_mask)
    non_band_cols = np.flatnonzero(~col_mask)
    out[np.ix_(horizontal_rows, non_band_cols)] = np.where(non_band_cols % 2 == 0, HORIZONTAL_RED, ALT_GREY)
    out[np.ix_(non_band_rows, vertical_cols)] = np.where(non_band_rows % 2 == 0, VERTICAL_BLUE, ALT_GREY)[:, None]

    top = int(horizontal_rows[0])
    left = int(vertical_cols[0])
    rr, cc = np.meshgrid(horizontal_rows, vertical_cols, indexing="ij")
    out[np.ix_(horizontal_rows, vertical_cols)] = np.where(
        ((rr - top) % 2) == ((cc - left) % 2),
        INTERFERENCE_YELLOW,
        BG,
    )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a red horizontal band and a blue vertical band into alternating stripes whose overlap becomes a parity interference checkerboard.

    Input structure:
    - The grid has black background.
    - One red band spans the full width and occupies 2-3 consecutive rows.
    - One blue band spans the full height and occupies 2-3 consecutive columns, but the overlap is red because the horizontal band covers it.

    Rule:
    - Detect the full-width red band and the full-height non-background band.
    - Recolor the horizontal band's non-overlap cells by column parity: red on even columns and grey on odd columns.
    - Recolor the vertical band's non-overlap cells by row parity: blue on even rows and grey on odd rows.
    - In the overlap rectangle, paint yellow where row and column parities match and black where they differ; leave the rest unchanged.

    Key insight:
    - The crossing area is not inherited from either band alone; it combines two independent 1D parity signals into a 2D interference pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= _ALLOWED_INPUT_COLORS, "unexpected input colors"

    horizontal_rows, vertical_cols = _detect_bands(g)
    return _expected_output(g.shape, horizontal_rows, vertical_cols)


def _pick_start(length: int, band_size: int, parity: int) -> int | None:
    max_start = length - band_size - _GEN_MARGIN
    candidates = [
        start
        for start in range(_GEN_MARGIN, max_start + 1)
        if start % 2 == parity
    ]
    if not candidates:
        return None
    return int(candidates[int(_GEN_RNG.integers(0, len(candidates)))])


def _build_input(
    height: int,
    width: int,
    band_top: int,
    band_height: int,
    band_left: int,
    band_width: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[:, band_left : band_left + band_width] = VERTICAL_BLUE
    grid[band_top : band_top + band_height, :] = HORIZONTAL_RED
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    expected: np.ndarray,
    band_top: int,
    band_height: int,
    band_left: int,
    band_width: int,
    relation: str,
) -> bool:
    horizontal_rows, vertical_cols = _detect_bands(grid)
    target_rows = np.arange(band_top, band_top + band_height)
    target_cols = np.arange(band_left, band_left + band_width)
    if not np.array_equal(horizontal_rows, target_rows):
        return False
    if not np.array_equal(vertical_cols, target_cols):
        return False
    if not np.array_equal(_expected_output(grid.shape, horizontal_rows, vertical_cols), expected):
        return False

    wrong = _wrong_local_origin_output(grid.shape, horizontal_rows, vertical_cols)
    if relation == "mismatch" and np.array_equal(wrong, expected):
        return False
    if relation == "match" and not np.array_equal(wrong, expected):
        return False

    if int(np.count_nonzero(expected == INTERFERENCE_YELLOW)) == 0:
        return False
    if int(np.count_nonzero(expected == ALT_GREY)) == 0:
        return False
    if not np.any(expected[np.ix_(horizontal_rows, vertical_cols)] == BG):
        return False
    if not np.any(expected[np.ix_(horizontal_rows, vertical_cols)] == INTERFERENCE_YELLOW):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    band_height, band_width = _GEN_BAND_SIZE_PLAN[call_index % len(_GEN_BAND_SIZE_PLAN)]
    relation = _GEN_RELATION_PLAN[call_index % len(_GEN_RELATION_PLAN)]

    for _ in range(_GEN_MAX_ATTEMPTS):
        height = int(_GEN_RNG.integers(*_GEN_HEIGHT_RANGE))
        width = int(_GEN_RNG.integers(*_GEN_WIDTH_RANGE))

        row_parity = int(_GEN_RNG.integers(0, 2))
        if relation == "match":
            col_parity = row_parity
        else:
            col_parity = 1 - row_parity

        band_top = _pick_start(height, band_height, row_parity)
        band_left = _pick_start(width, band_width, col_parity)
        if band_top is None or band_left is None:
            continue

        grid = _build_input(height, width, band_top, band_height, band_left, band_width)
        horizontal_rows = np.arange(band_top, band_top + band_height)
        vertical_cols = np.arange(band_left, band_left + band_width)
        expected = _expected_output(grid.shape, horizontal_rows, vertical_cols)

        if _validate_generated_case(
            grid,
            expected,
            band_top,
            band_height,
            band_left,
            band_width,
            relation,
        ):
            return grid

    raise RuntimeError("failed to generate a valid crossing-band interference input")
