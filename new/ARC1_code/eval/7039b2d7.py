import numpy as np

from arc_utils import colors_present, get_bg


def _count_background_bands(separator_lines: np.ndarray) -> int:
    bands = 0
    in_band = False
    for is_separator in separator_lines:
        if is_separator:
            in_band = False
        elif not in_band:
            bands += 1
            in_band = True
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize a foreground line partition as a plain background-colored grid whose shape is the number of row and column regions.

    Input structure
    The input has one background color and one foreground color; the foreground
    pixels are exactly the union of complete horizontal and vertical separator
    lines, possibly including a line on the border.
    Rule
    Count the contiguous runs of non-separator rows and non-separator columns.
    Return a uniform grid of the background color with those run counts as its
    height and width.
    Key insight
    The output does not preserve the separator lines themselves; it records only
    how many background regions the line partition creates in each direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude=[bg])
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    full_rows = np.all(g == fg, axis=1)
    full_cols = np.all(g == fg, axis=0)
    assert np.any(full_rows)
    assert np.any(full_cols)

    foreground_mask = g == fg
    line_mask = full_rows[:, None] | full_cols[None, :]
    assert np.array_equal(foreground_mask, line_mask)

    out_h = _count_background_bands(full_rows)
    out_w = _count_background_bands(full_cols)
    assert out_h > 0
    assert out_w > 0
    return np.full((out_h, out_w), bg, dtype=int)

# --------------- generator (o5x) ---------------

# Canvas/output bounds.
GRID_MIN_SIZE, GRID_MAX_SIZE_EXCLUSIVE = 10, 28
OUTPUT_HEIGHT_RANGE = (2, 6)
OUTPUT_WIDTH_RANGE = (3, 6)
ARC_COLOR_IDS = np.arange(10)

# Optional border separator sampling by output dimension.
ROW_BORDER_EXTRA_BY_HEIGHT = {2: ([0, 1, 2], [0.75, 0.15, 0.10]), 3: ([0, 1, 2], [0.80, 0.15, 0.05]), 4: ([0, 1], [0.85, 0.15])}
COL_BORDER_EXTRA_BY_WIDTH = {3: ([1, 2], [0.75, 0.25]), 4: ([0, 1], [0.75, 0.25])}
BORDER_START_PROB = 0.5

# Retry budget.
GENERATION_RETRY_LIMIT = 300


def _gen_positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(np.random.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def _gen_border_flags(extra_count: int) -> tuple[bool, bool]:
    if extra_count == 0:
        return False, False
    if extra_count == 2:
        return True, True
    return (True, False) if np.random.random() < BORDER_START_PROB else (False, True)


def _gen_border_extra_count(dimension: int, choices_by_dimension: dict[int, tuple[list[int], list[float]]]) -> int:
    choices = choices_by_dimension.get(dimension)
    if choices is None:
        return 0
    values, probabilities = choices
    return int(np.random.choice(values, p=probabilities))


def _gen_separator_plan() -> tuple[int, int, int, int, tuple[bool, bool], tuple[bool, bool]]:
    out_h = int(np.random.randint(*OUTPUT_HEIGHT_RANGE))
    out_w = int(np.random.randint(*OUTPUT_WIDTH_RANGE))

    row_extra = _gen_border_extra_count(out_h, ROW_BORDER_EXTRA_BY_HEIGHT)
    col_extra = _gen_border_extra_count(out_w, COL_BORDER_EXTRA_BY_WIDTH)
    row_borders = _gen_border_flags(row_extra)
    col_borders = _gen_border_flags(col_extra)
    row_seps = out_h - 1 + row_extra
    col_seps = out_w - 1 + col_extra
    return out_h, out_w, row_seps, col_seps, row_borders, col_borders


def _gen_min_square_size(row_seps: int, col_seps: int, out_h: int, out_w: int) -> int:
    min_size = max(GRID_MIN_SIZE, row_seps + out_h, col_seps + out_w)
    for size in range(min_size, GRID_MAX_SIZE_EXCLUSIVE):
        fg_area = row_seps * size + col_seps * size - row_seps * col_seps
        if size * size - fg_area > fg_area:
            return size
    return GRID_MAX_SIZE_EXCLUSIVE


def _gen_separator_positions(total: int, band_sizes: list[int], start_border: bool, end_border: bool) -> list[int]:
    pos = 0
    separators = []
    if start_border:
        separators.append(pos)
        pos += 1
    for i, size in enumerate(band_sizes):
        pos += size
        if i < len(band_sizes) - 1:
            separators.append(pos)
            pos += 1
    if end_border:
        separators.append(total - 1)
    return separators


def _gen_grid_layout(
    out_h: int,
    out_w: int,
    row_seps: int,
    col_seps: int,
    row_borders: tuple[bool, bool],
    col_borders: tuple[bool, bool],
) -> tuple[int, list[int], list[int]]:
    """Choose square size and separator indices for both axes."""
    size_min = _gen_min_square_size(row_seps, col_seps, out_h, out_w)
    size = int(np.random.randint(size_min, GRID_MAX_SIZE_EXCLUSIVE))
    row_bands = _gen_positive_composition(size - row_seps, out_h)
    col_bands = _gen_positive_composition(size - col_seps, out_w)
    row_lines = _gen_separator_positions(size, row_bands, *row_borders)
    col_lines = _gen_separator_positions(size, col_bands, *col_borders)
    return size, row_lines, col_lines


def _render_separator_grid(size: int, row_lines: list[int], col_lines: list[int]) -> tuple[np.ndarray, int]:
    bg, fg = np.random.choice(ARC_COLOR_IDS, size=2, replace=False).astype(int)
    inp = np.full((size, size), int(bg), dtype=int)
    inp[row_lines, :] = int(fg)
    inp[:, col_lines] = int(fg)
    return inp, int(bg)


def _gen_attempt() -> np.ndarray | None:
    out_h, out_w, row_seps, col_seps, row_borders, col_borders = _gen_separator_plan()
    size, row_lines, col_lines = _gen_grid_layout(out_h, out_w, row_seps, col_seps, row_borders, col_borders)
    inp, bg = _render_separator_grid(size, row_lines, col_lines)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
