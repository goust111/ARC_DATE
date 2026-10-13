import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, paste
import random


BAR_ORDER = (Color.GREEN, Color.RED)
OUT_SHAPE = (2, 3)


def _assert_horizontal_bars(g: np.ndarray, color: int) -> int:
    comps = components_mask(g == int(color), connectivity=4)
    for coords in comps:
        rows = np.unique(coords[:, 0])
        cols = np.sort(coords[:, 1])
        assert rows.size == 1, "expected each foreground object to be a horizontal bar"
        assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "expected contiguous bars"
    return len(comps)


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize green and red horizontal bars as a row-major 2x3 count strip.

    Input structure
    The input has a dominant background color and foreground objects consisting
    only of separated green and red one-cell-thick horizontal bars.

    Rule
    Count the green bars, then the red bars. Create a black 2x3 grid and fill
    one cell per bar in row-major order, placing all green cells first and all
    red cells after them.

    Key insight
    The output discards bar positions and lengths; it keeps only the number of
    bars of each of the two foreground colors, packed into a fixed six-cell
    summary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg not in BAR_ORDER, "background must differ from the bar colors"
    assert set(colors_present(g, exclude={bg})) == set(BAR_ORDER), "expected exactly green and red bars"

    counts = [_assert_horizontal_bars(g, color) for color in BAR_ORDER]
    total = sum(counts)
    assert 0 < total <= OUT_SHAPE[0] * OUT_SHAPE[1], "bar count must fit the 2x3 summary"

    out = np.full(OUT_SHAPE, Color.BLACK, dtype=int)
    slot = 0
    for color, count in zip(BAR_ORDER, counts):
        for _ in range(count):
            out[slot // OUT_SHAPE[1], slot % OUT_SHAPE[1]] = int(color)
            slot += 1
    return out

# --------------- generator (o5x) ---------------

# Canvas palette, size bounds, and border padding.
_GEN_BG_COLORS = (Color.BLUE, Color.GREY, Color.TEAL, Color.MAROON, Color.ORANGE, Color.YELLOW, Color.PINK)
_GEN_ROWS_RANGE = (10, 15)
_GEN_COLS_RANGE = (10, 19)
_GEN_BORDER_PAD = 1

# Bar count, length, output, and retry bounds.
_GEN_GREEN_RANGE = (2, 4)
_GEN_RED_RANGE = (1, 4)
_GEN_TOTAL_RANGE = (4, 6)
_GEN_LENGTH_RANGE = (2, 8)
_GEN_OUT_SHAPE = (2, 3)
_GEN_RETRY_LIMIT = 300
_GEN_COUNT_CHOICES = tuple(
    (green_count, red_count)
    for green_count in range(_GEN_GREEN_RANGE[0], _GEN_GREEN_RANGE[1] + 1)
    for red_count in range(_GEN_RED_RANGE[0], _GEN_RED_RANGE[1] + 1)
    if _GEN_TOTAL_RANGE[0] <= green_count + red_count <= _GEN_TOTAL_RANGE[1]
)


def _gen_expected_output(green_count, red_count):
    out = np.full(_GEN_OUT_SHAPE, Color.BLACK, dtype=int)
    flat = out.ravel()
    flat[:green_count] = Color.GREEN
    flat[green_count : green_count + red_count] = Color.RED
    return out


def _gen_bar_slots(grid, bg, color, length):
    """Return top-left locations that avoid overlap and same-color contact."""
    rows, cols = grid.shape
    slots = []
    for r in range(_GEN_BORDER_PAD, rows - _GEN_BORDER_PAD):
        for c in range(cols - length + 1):
            if not np.all(grid[r, c : c + length] == bg):
                continue
            if c > 0 and grid[r, c - 1] == color:
                continue
            if c + length < cols and grid[r, c + length] == color:
                continue
            if r > 0 and np.any(grid[r - 1, c : c + length] == color):
                continue
            if r + 1 < rows and np.any(grid[r + 1, c : c + length] == color):
                continue
            slots.append((r, c))
    return slots


def _gen_place_bars(grid, bg, bars):
    _, cols = grid.shape
    for color in bars:
        length = random.randint(_GEN_LENGTH_RANGE[0], min(_GEN_LENGTH_RANGE[1], cols))
        slots = _gen_bar_slots(grid, bg, color, length)
        if not slots:
            return False
        r, c = random.choice(slots)
        paste(grid, np.full((1, length), color, dtype=int), r, c)
    return True


def _gen_is_valid_grid(grid, bg, green_count, red_count):
    colors, counts = np.unique(grid, return_counts=True)
    if int(colors[int(np.argmax(counts))]) != int(bg):
        return False
    if set(map(int, colors)) != {int(bg), int(Color.GREEN), int(Color.RED)}:
        return False
    expected = _gen_expected_output(green_count, red_count)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    rows = random.randint(*_GEN_ROWS_RANGE)
    cols = random.randint(*_GEN_COLS_RANGE)
    bg = random.choice(_GEN_BG_COLORS)
    green_count, red_count = random.choice(_GEN_COUNT_CHOICES)

    grid = np.full((rows, cols), bg, dtype=int)
    bars = [Color.GREEN] * green_count + [Color.RED] * red_count
    random.shuffle(bars)

    if not _gen_place_bars(grid, bg, bars):
        return None
    if not _gen_is_valid_grid(grid, bg, green_count, red_count):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5289ad53 input")
