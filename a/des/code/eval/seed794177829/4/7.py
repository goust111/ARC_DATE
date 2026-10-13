import numpy as np


BG = 0
MARKER = 9
STRIPE_COLORS = tuple(range(1, 9))
CORNERS = ("tl", "tr", "bl", "br")

MIN_SIZE = 8
MAX_SIZE = 18
MIN_HORIZONTAL = 2
MAX_HORIZONTAL = 4
MIN_VERTICAL = 2
MAX_VERTICAL = 4
MAX_TOTAL_STRIPES = 8
GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _corner_slices(shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = shape
    return {
        "tl": (slice(0, 2), slice(0, 2)),
        "tr": (slice(0, 2), slice(w - 2, w)),
        "bl": (slice(h - 2, h), slice(0, 2)),
        "br": (slice(h - 2, h), slice(w - 2, w)),
    }


def _find_start_corner(grid: np.ndarray) -> str:
    matches = []
    for name, (rs, cs) in _corner_slices(grid.shape).items():
        if np.all(grid[rs, cs] == MARKER):
            matches.append(name)
    assert len(matches) == 1, "expected exactly one 2x2 maroon corner marker"
    return matches[0]


def _ordered_indices(values: list[int], reverse: bool) -> dict[int, int]:
    ordered = sorted(values, reverse=reverse)
    return {value: idx for idx, value in enumerate(ordered)}


def _row_color(grid: np.ndarray, row: int) -> int:
    values = grid[row]
    colors = np.unique(values[values != MARKER])
    assert len(colors) == 1 and int(colors[0]) != BG
    return int(colors[0])


def _col_color(grid: np.ndarray, col: int, horizontal_rows: list[int]) -> int:
    mask = np.ones(grid.shape[0], dtype=bool)
    mask[horizontal_rows] = False
    values = grid[mask, col]
    colors = np.unique(values[(values != BG) & (values != MARKER)])
    assert len(colors) == 1
    return int(colors[0])


def _paint_weave(
    grid: np.ndarray,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
    corner: str,
) -> np.ndarray:
    row_phase = _ordered_indices(horizontal_rows, reverse=corner in ("bl", "br"))
    col_phase = _ordered_indices(vertical_cols, reverse=corner in ("tr", "br"))

    out = grid.copy()
    for row in horizontal_rows:
        for col in vertical_cols:
            if (row_phase[row] + col_phase[col]) % 2 == 0:
                out[row, col] = col_colors[col]
            else:
                out[row, col] = row_colors[row]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Redraw woven crossings from horizontal-on-top input stripes."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 4
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    corner = _find_start_corner(g)

    horizontal_rows = np.flatnonzero(np.count_nonzero(g != BG, axis=1) == w).tolist()
    vertical_cols = np.flatnonzero(np.count_nonzero(g != BG, axis=0) == h).tolist()
    assert horizontal_rows, "expected at least one horizontal stripe"
    assert vertical_cols, "expected at least one vertical stripe"

    row_colors = {row: _row_color(g, row) for row in horizontal_rows}
    col_colors = {col: _col_color(g, col, horizontal_rows) for col in vertical_cols}

    return _paint_weave(g, horizontal_rows, vertical_cols, row_colors, col_colors, corner)


def _sample_case() -> tuple[np.ndarray, np.ndarray]:
    h = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))
    w = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))

    max_h = min(MAX_HORIZONTAL, h - 4, MAX_TOTAL_STRIPES - MIN_VERTICAL)
    h_count = int(_RNG.integers(MIN_HORIZONTAL, max_h + 1))

    max_v = min(MAX_VERTICAL, w - 4, MAX_TOTAL_STRIPES - h_count)
    v_count = int(_RNG.integers(MIN_VERTICAL, max_v + 1))

    colors = _RNG.choice(np.array(STRIPE_COLORS, dtype=int), size=h_count + v_count, replace=False)
    row_colors_list = [int(color) for color in colors[:h_count]]
    col_colors_list = [int(color) for color in colors[h_count:]]

    row_pool = np.arange(2, h - 2, dtype=int)
    col_pool = np.arange(2, w - 2, dtype=int)
    horizontal_rows = sorted(int(v) for v in _RNG.choice(row_pool, size=h_count, replace=False))
    vertical_cols = sorted(int(v) for v in _RNG.choice(col_pool, size=v_count, replace=False))

    grid = np.full((h, w), BG, dtype=int)
    for col, color in zip(vertical_cols, col_colors_list):
        grid[:, col] = color
    for row, color in zip(horizontal_rows, row_colors_list):
        grid[row, :] = color

    corner = CORNERS[int(_RNG.integers(0, len(CORNERS)))]
    rs, cs = _corner_slices(grid.shape)[corner]
    grid[rs, cs] = MARKER

    row_colors = dict(zip(horizontal_rows, row_colors_list))
    col_colors = dict(zip(vertical_cols, col_colors_list))
    expected = _paint_weave(grid, horizontal_rows, vertical_cols, row_colors, col_colors, corner)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid, expected = _sample_case()
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError(f"failed to generate a valid weave puzzle after {GENERATION_ATTEMPTS} attempts")
