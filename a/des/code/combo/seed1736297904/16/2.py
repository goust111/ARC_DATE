import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_MIN = 11
GRID_MAX = 21
PATH_COUNT_MIN = 2
PATH_COUNT_MAX = 5
MIN_LANE_GAP = 2
GENERATION_ATTEMPTS = 300
SPACING_ATTEMPTS = 200


def _ray_color(grid: np.ndarray, row: int, col: int, dr: int, dc: int) -> int | None:
    height, width = grid.shape
    r, c = row + dr, col + dc
    while 0 <= r < height and 0 <= c < width:
        value = int(grid[r, c])
        if value == GREY:
            r += dr
            c += dc
            continue
        if value == BG:
            return None
        return value
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for row, col in np.argwhere(g == GREY):
        row = int(row)
        col = int(col)

        left = _ray_color(g, row, col, 0, -1)
        right = _ray_color(g, row, col, 0, 1)
        up = _ray_color(g, row, col, -1, 0)
        down = _ray_color(g, row, col, 1, 0)

        horizontal = left is not None and left == right
        vertical = up is not None and up == down
        assert horizontal ^ vertical, "each grey cell must have exactly one straight-through color"

        winner = int(left if horizontal else up)
        side_colors = {color for color in (left, right, up, down) if color is not None and color != winner}
        assert side_colors, "each grey cell must also be touched by a second color"
        out[row, col] = winner

    return out


def _sample_spaced(rng: np.random.Generator, low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = np.arange(low, high + 1, dtype=int)
    if len(values) < count:
        return None

    for _ in range(SPACING_ATTEMPTS):
        picks = sorted(int(v) for v in rng.choice(values, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_horizontal_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    max_paths = min(PATH_COUNT_MAX, (height - 3) // 2)
    if max_paths < PATH_COUNT_MIN or width < 4:
        return None

    path_count = int(rng.integers(PATH_COUNT_MIN, max_paths + 1))
    rows = _sample_spaced(rng, 2, height - 3, path_count, MIN_LANE_GAP)
    if rows is None:
        return None

    bend_col = int(rng.integers(1, width - 2))
    colors = [int(v) for v in rng.choice(PALETTE, size=path_count + 1, replace=False)]
    row_colors = colors[:-1]
    snake_color = colors[-1]

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full_like(grid, BG)

    for row, color in zip(rows, row_colors):
        grid[row, :] = color
        expected[row, :] = color

    current_col = bend_col + int(rng.integers(0, 2))
    previous_row = 0
    for row, color in zip(rows, row_colors):
        grid[previous_row:row, current_col] = snake_color
        expected[previous_row:row, current_col] = snake_color
        grid[row, bend_col : bend_col + 2] = GREY
        expected[row, bend_col : bend_col + 2] = color
        current_col = bend_col if current_col == bend_col + 1 else bend_col + 1
        previous_row = row + 1

    grid[previous_row:, current_col] = snake_color
    expected[previous_row:, current_col] = snake_color
    return grid, expected


def _build_vertical_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    max_paths = min(PATH_COUNT_MAX, (width - 3) // 2)
    if max_paths < PATH_COUNT_MIN or height < 4:
        return None

    path_count = int(rng.integers(PATH_COUNT_MIN, max_paths + 1))
    cols = _sample_spaced(rng, 2, width - 3, path_count, MIN_LANE_GAP)
    if cols is None:
        return None

    bend_row = int(rng.integers(1, height - 2))
    colors = [int(v) for v in rng.choice(PALETTE, size=path_count + 1, replace=False)]
    col_colors = colors[:-1]
    snake_color = colors[-1]

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full_like(grid, BG)

    for col, color in zip(cols, col_colors):
        grid[:, col] = color
        expected[:, col] = color

    current_row = bend_row + int(rng.integers(0, 2))
    previous_col = 0
    for col, color in zip(cols, col_colors):
        grid[current_row, previous_col:col] = snake_color
        expected[current_row, previous_col:col] = snake_color
        grid[bend_row : bend_row + 2, col] = GREY
        expected[bend_row : bend_row + 2, col] = color
        current_row = bend_row if current_row == bend_row + 1 else bend_row + 1
        previous_col = col + 1

    grid[current_row, previous_col:] = snake_color
    expected[current_row, previous_col:] = snake_color
    return grid, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    builder = _build_horizontal_case if int(rng.integers(0, 2)) == 0 else _build_vertical_case
    built = builder(rng)
    if built is None:
        return None

    grid, expected = built
    if np.count_nonzero(grid == GREY) < 4:
        return None
    if np.any(expected == GREY):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-overlap puzzle")
