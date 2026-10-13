import random
from collections import Counter

import numpy as np


BLACK = 0
AZURE = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 400


def _corner_positions(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]


def _find_origin(grid: np.ndarray) -> tuple[int, int]:
    corners = [(r, c) for r, c in _corner_positions(*grid.shape) if int(grid[r, c]) == AZURE]
    assert len(corners) == 1, f"expected exactly one azure corner, got {corners}"
    return corners[0]


def _distance_from_origin(origin: tuple[int, int], row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if origin == (0, 0):
        return row, col
    if origin == (0, width - 1):
        return row, width - 1 - col
    if origin == (height - 1, 0):
        return height - 1 - row, col
    assert origin == (height - 1, width - 1)
    return height - 1 - row, width - 1 - col


def _line_color_from_row(grid: np.ndarray, row: int, vertical_cols: set[int]) -> int:
    primary = [
        int(grid[row, col])
        for col in range(grid.shape[1])
        if col not in vertical_cols and int(grid[row, col]) not in (BLACK, AZURE)
    ]
    if primary:
        colors = sorted(set(primary))
        assert len(colors) == 1, f"ambiguous horizontal color on row {row}: {colors}"
        return colors[0]

    counts = Counter(
        int(value)
        for value in grid[row]
        if int(value) not in (BLACK, AZURE)
    )
    assert counts, f"could not recover a horizontal color on row {row}"
    return counts.most_common(1)[0][0]


def _line_color_from_col(grid: np.ndarray, col: int, horizontal_rows: set[int]) -> int:
    primary = [
        int(grid[row, col])
        for row in range(grid.shape[0])
        if row not in horizontal_rows and int(grid[row, col]) not in (BLACK, AZURE)
    ]
    if primary:
        colors = sorted(set(primary))
        assert len(colors) == 1, f"ambiguous vertical color on col {col}: {colors}"
        return colors[0]

    counts = Counter(
        int(grid[row, col])
        for row in range(grid.shape[0])
        if int(grid[row, col]) not in (BLACK, AZURE)
    )
    assert counts, f"could not recover a vertical color on col {col}"
    return counts.most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint the crossings of full-width and full-height colored paths into an alternating woven order.

    Input structure:
    - The background is black.
    - Several single-cell-wide horizontal paths span entire rows from the left edge to the right edge.
    - Several single-cell-wide vertical paths span entire columns from the top edge to the bottom edge.
    - Each path has its own color; at crossings, the input shows either the horizontal path color or the vertical path color.
    - Exactly one corner cell is azure and acts as the start marker.

    Rule:
    - Detect horizontal path rows as the rows with no black cells, and vertical path columns as the columns with no black cells.
    - Recover each path's own color from its non-crossing cells.
    - List every crossing between a detected horizontal row and vertical column, and sort the crossings by distance from the azure corner using row-distance plus column-distance; ties are broken by smaller row-distance.
    - Recolor the 1st, 3rd, 5th, ... crossings with the horizontal path color, and recolor the 2nd, 4th, 6th, ... crossings with the vertical path color.
    - Leave all non-crossing path pixels unchanged and keep the azure corner marker in the output.

    Key insight:
    - Even though crossings may hide one strand, the full-row and full-column supports reveal the underlying horizontal and vertical paths; the azure corner then defines the weave order that alternates which strand appears on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    origin = _find_origin(g)

    horizontal_rows = [int(row) for row in np.flatnonzero(np.all(g != BLACK, axis=1))]
    vertical_cols = [int(col) for col in np.flatnonzero(np.all(g != BLACK, axis=0))]
    assert horizontal_rows, "expected at least one horizontal path"
    assert vertical_cols, "expected at least one vertical path"

    hset = set(horizontal_rows)
    vset = set(vertical_cols)
    row_colors = {row: _line_color_from_row(g, row, vset) for row in horizontal_rows}
    col_colors = {col: _line_color_from_col(g, col, hset) for col in vertical_cols}

    crossings: list[tuple[int, int, int, int]] = []
    for row in horizontal_rows:
        for col in vertical_cols:
            row_dist, col_dist = _distance_from_origin(origin, row, col, height, width)
            crossings.append((row_dist + col_dist, row_dist, row, col))
    crossings.sort()

    out = g.copy()
    for index, (_total_dist, _row_dist, row, col) in enumerate(crossings, start=1):
        out[row, col] = row_colors[row] if index % 2 == 1 else col_colors[col]

    out[origin] = AZURE
    return out


def _sample_indices(limit: int, count: int, include: set[int], exclude: set[int]) -> list[int] | None:
    chosen = set(include)
    pool = [idx for idx in range(limit) if idx not in chosen and idx not in exclude]
    needed = count - len(chosen)
    if needed < 0 or needed > len(pool):
        return None
    chosen.update(random.sample(pool, needed))
    return sorted(chosen)


def _visibility_mask(rows: int, cols: int) -> np.ndarray:
    mask = np.random.randint(0, 2, size=(rows, cols), dtype=int).astype(bool)
    if rows * cols > 1 and (mask.all() or (~mask).all()):
        flat = mask.reshape(-1)
        flip_index = random.randrange(flat.size)
        flat[flip_index] = not flat[flip_index]
    return mask


def _build_grid(
    height: int,
    width: int,
    horizontal_rows: list[int],
    vertical_cols: list[int],
    horizontal_colors: list[int],
    vertical_colors: list[int],
    origin: tuple[int, int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)

    for row, color in zip(horizontal_rows, horizontal_colors):
        grid[row, :] = color
    for col, color in zip(vertical_cols, vertical_colors):
        grid[:, col] = color

    mask = _visibility_mask(len(horizontal_rows), len(vertical_cols))
    for row_index, row in enumerate(horizontal_rows):
        for col_index, col in enumerate(vertical_cols):
            if mask[row_index, col_index]:
                grid[row, col] = horizontal_colors[row_index]

    grid[origin] = AZURE
    return grid


def _validate_grid(
    grid: np.ndarray,
    horizontal_rows: list[int],
    vertical_cols: list[int],
) -> None:
    assert np.count_nonzero(grid == AZURE) == 1
    detected_rows = [int(row) for row in np.flatnonzero(np.all(grid != BLACK, axis=1))]
    detected_cols = [int(col) for col in np.flatnonzero(np.all(grid != BLACK, axis=0))]
    assert detected_rows == horizontal_rows, (detected_rows, horizontal_rows)
    assert detected_cols == vertical_cols, (detected_cols, vertical_cols)

    colors = [int(color) for color in grid.flatten() if int(color) not in (BLACK, AZURE)]
    assert len(set(colors)) >= len(horizontal_rows) + len(vertical_cols)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(8, 18)
    width = random.randint(8, 18)

    candidates = [
        (h_count, v_count)
        for h_count in range(1, 5)
        for v_count in range(1, 5)
        if h_count * v_count >= 4
        and h_count + v_count <= 6
        and height - h_count >= 3
        and width - v_count >= 3
    ]
    if not candidates:
        return None
    h_count, v_count = random.choice(candidates)

    origin = random.choice(_corner_positions(height, width))
    origin_row, origin_col = origin
    mode = random.choice(["none", "horizontal", "vertical", "both"])

    include_rows = {origin_row} if mode in ("horizontal", "both") else set()
    include_cols = {origin_col} if mode in ("vertical", "both") else set()
    exclude_rows = {origin_row} if mode in ("none", "vertical") else set()
    exclude_cols = {origin_col} if mode in ("none", "horizontal") else set()

    horizontal_rows = _sample_indices(height, h_count, include_rows, exclude_rows)
    vertical_cols = _sample_indices(width, v_count, include_cols, exclude_cols)
    if horizontal_rows is None or vertical_cols is None:
        return None

    colors = random.sample(PALETTE, h_count + v_count)
    horizontal_colors = colors[:h_count]
    vertical_colors = colors[h_count:]

    grid = _build_grid(
        height,
        width,
        horizontal_rows,
        vertical_cols,
        horizontal_colors,
        vertical_colors,
        origin,
    )
    _validate_grid(grid, horizontal_rows, vertical_cols)

    try:
        output = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, output):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid weave input after {MAX_GENERATION_ATTEMPTS} attempts")
