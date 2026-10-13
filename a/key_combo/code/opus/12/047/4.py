import itertools
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


INPUT_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
GEN_ATTEMPTS = 400
MIN_GAP = 3
SIZE_MIN = 10
SIZE_MAX = 18
RNG = np.random.default_rng()


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values))
    color, count = counts.most_common(1)[0]
    assert color != Color.BLACK
    assert list(counts.values()).count(count) == 1
    return int(color)


def _find_full_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = list(map(int, np.flatnonzero(np.all(grid != Color.BLACK, axis=1))))
    cols = list(map(int, np.flatnonzero(np.all(grid != Color.BLACK, axis=0))))
    assert rows and cols
    return rows, cols


def _line_spacing_ok(positions: list[int], limit: int) -> bool:
    if any(p <= 0 or p >= limit - 1 for p in positions):
        return False
    return all(b - a >= MIN_GAP for a, b in zip(positions, positions[1:]))


def _bar_colors(grid: np.ndarray, rows: list[int], cols: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    row_colors = {r: _majority_color(grid[r, :]) for r in rows}
    col_colors = {c: _majority_color(grid[:, c]) for c in cols}
    colors = list(row_colors.values()) + list(col_colors.values())
    assert len(colors) == len(set(colors))
    return row_colors, col_colors


def _render_output(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape

    if width < height:
        ring_picker = lambda r, c: row_colors[r]
        center_picker = lambda r, c: col_colors[c]
    elif width > height:
        ring_picker = lambda r, c: col_colors[c]
        center_picker = lambda r, c: row_colors[r]
    else:
        ring_picker = lambda r, c: Color.YELLOW
        center_picker = lambda r, c: row_colors[r]

    for r in rows:
        for c in cols:
            out[r - 1 : r + 2, c - 1 : c + 2] = ring_picker(r, c)
            out[r, c] = center_picker(r, c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a 3x3 hollow marker at every full-row/full-column crossing, with colors determined by bar-length comparison.

    Input structure:
    - The grid is a black background containing 2-3 full horizontal bars and 2-3 full vertical bars.
    - Each bar is one cell thick, spans its entire row or column, and uses a color unique among all bars.
    - Bar crossings lie away from the border and are spaced so their 3x3 neighborhoods do not overlap.

    Rule:
    - Find every row that is entirely non-black and every column that is entirely non-black.
    - Infer each row-bar color and column-bar color from the dominant color along that full line.
    - At every row/column crossing, paint the surrounding 8 cells of the 3x3 neighborhood with the shorter bar's color and paint the center with the longer bar's color.
    - If the grid is square, treat the bar lengths as tied: paint the 8-cell ring yellow and keep the center in the row-bar color.

    Key insight:
    - The local marker color is not fixed by color identity; it is determined by comparing the global lengths of the crossing full row and full column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == Color.BLACK)

    rows, cols = _find_full_rows_cols(g)
    assert _line_spacing_ok(rows, g.shape[0])
    assert _line_spacing_ok(cols, g.shape[1])
    row_colors, col_colors = _bar_colors(g, rows, cols)
    return _render_output(g, rows, cols, row_colors, col_colors)


def _sample_spaced_positions(limit: int, count: int) -> list[int]:
    choices = [
        combo
        for combo in itertools.combinations(range(1, limit - 1), count)
        if all(b - a >= MIN_GAP for a, b in zip(combo, combo[1:]))
    ]
    assert choices
    idx = int(RNG.integers(len(choices)))
    return list(map(int, choices[idx]))


def _pick_dimensions(branch: str) -> tuple[int, int]:
    if branch == "tie":
        side = int(RNG.integers(11, SIZE_MAX + 1))
        return side, side
    if branch == "row_short":
        height = int(RNG.integers(12, SIZE_MAX + 1))
        delta = int(RNG.integers(2, min(6, height - SIZE_MIN + 1)))
        width = height - delta
        return height, width
    height = int(RNG.integers(SIZE_MIN, SIZE_MAX - 1))
    delta = int(RNG.integers(2, min(6, SIZE_MAX - height + 1)))
    width = height + delta
    return height, width


def _make_owner_mask(n_rows: int, n_cols: int) -> np.ndarray:
    mask = RNG.integers(0, 2, size=(n_rows, n_cols), dtype=int).astype(bool)
    if mask.all():
        mask[0, 0] = False
    if (~mask).all():
        mask[0, 0] = True
    return mask


def _build_input(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    row_colors: list[int],
    col_colors: list[int],
    row_priority: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r, color in zip(rows, row_colors):
        grid[r, :] = int(color)
    for c, color in zip(cols, col_colors):
        grid[:, c] = int(color)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            grid[r, c] = int(row_colors[i] if row_priority[i, j] else col_colors[j])
    return grid


def _expected_from_blueprint(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    row_map = {r: int(color) for r, color in zip(rows, row_colors)}
    col_map = {c: int(color) for c, color in zip(cols, col_colors)}
    return _render_output(grid, rows, cols, row_map, col_map)


def _valid_instance(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    row_colors: list[int],
    col_colors: list[int],
    expected: np.ndarray,
) -> bool:
    found_rows, found_cols = _find_full_rows_cols(grid)
    if found_rows != rows or found_cols != cols:
        return False
    found_row_colors, found_col_colors = _bar_colors(grid, found_rows, found_cols)
    if [found_row_colors[r] for r in rows] != row_colors:
        return False
    if [found_col_colors[c] for c in cols] != col_colors:
        return False
    centers = grid[np.ix_(rows, cols)]
    expected_centers = expected[np.ix_(rows, cols)]
    if np.array_equal(centers, expected_centers):
        return False
    if np.array_equal(grid, expected):
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    branch = str(RNG.choice(["row_short", "col_short", "tie"]))
    height, width = _pick_dimensions(branch)
    n_rows = int(RNG.integers(2, 4))
    n_cols = int(RNG.integers(2, 4))

    rows = _sample_spaced_positions(height, n_rows)
    cols = _sample_spaced_positions(width, n_cols)

    colors = list(map(int, RNG.choice(INPUT_COLORS, size=n_rows + n_cols, replace=False)))
    row_colors = colors[:n_rows]
    col_colors = colors[n_rows:]

    row_priority = _make_owner_mask(n_rows, n_cols)
    grid = _build_input(height, width, rows, cols, row_colors, col_colors, row_priority)
    expected = _expected_from_blueprint(grid, rows, cols, row_colors, col_colors)

    if not _valid_instance(grid, rows, cols, row_colors, col_colors, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid full-row/full-column crossing input")
