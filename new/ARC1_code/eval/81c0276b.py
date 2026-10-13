from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste
import random


def _spans(length: int, bars: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for bar in sorted(map(int, bars)):
        assert start <= bar < length, "bar index out of bounds"
        if start < bar:
            spans.append((start, bar))
        start = bar + 1
    if start < length:
        spans.append((start, length))
    return spans


def _separator_color(g: np.ndarray, bg: int) -> tuple[int, np.ndarray, np.ndarray]:
    candidates = []
    for color in colors_present(g, exclude={bg}):
        row_bars = np.flatnonzero(np.all(g == color, axis=1))
        col_bars = np.flatnonzero(np.all(g == color, axis=0))
        if len(row_bars) and len(col_bars):
            candidates.append((int(color), row_bars, col_bars))

    assert len(candidates) == 1, "expected one gridline color with full rows and columns"
    return candidates[0]


def _single_cell_color(cell: np.ndarray, bg: int) -> int | None:
    mask = cell != bg
    if not np.any(mask):
        return None

    colors = colors_present(cell, exclude={bg})
    assert len(colors) == 1, "each occupied table cell must contain one sprite color"

    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    sprite_h = r1 - r0 + 1
    sprite_w = c1 - c0 + 1
    assert int(mask.sum()) == sprite_h * sprite_w, "cell sprite must be a solid rectangle"
    assert (sprite_h, sprite_w) == (2, 2), "cell sprites are 2 by 2 squares"

    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize a gridded table of colored square sprites as count bars ordered by increasing count.

    Input structure
    The input is a table on a dominant background. A unique non-background color
    forms full horizontal and vertical gridline bars. Each occupied table cell
    contains one solid 2 by 2 square sprite, and some sprites may use the same
    color as the gridline.

    Rule
    Ignore empty cells and ignore sprites whose color is the gridline color. For
    every other sprite color, count how many table cells contain that color.
    Create one output row per counted color, sorted by increasing count; each row
    is a left-aligned horizontal bar of that color with length equal to its
    count, padded with background to the maximum count.

    Key insight
    The large grid is only a frequency table: the separator color is not part of
    the histogram, even when it appears as a cell sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    separator, row_bars, col_bars = _separator_color(g, bg)

    counts: Counter[int] = Counter()
    for r0, r1 in _spans(g.shape[0], row_bars):
        for c0, c1 in _spans(g.shape[1], col_bars):
            color = _single_cell_color(g[r0:r1, c0:c1], bg)
            if color is not None and color != separator:
                counts[color] += 1

    assert counts, "expected at least one non-separator sprite color"
    assert len(set(counts.values())) == len(counts), "sprite color counts must be unique"

    ordered = sorted(counts.items(), key=lambda item: item[1])
    out = np.full((len(ordered), max(counts.values())), bg, dtype=int)
    for r, (color, count) in enumerate(ordered):
        out[r, :count] = color
    return out

# --------------- generator (o5x) ---------------
# Histogram, palette, and retry bounds.
N_COLORS_RANGE = (2, 4)
MAX_COUNT_RANGE = (2, 4)
COUNT_SAMPLE_START = 1
COLOR_POOL = range(1, 10)
# Fixed grid geometry and sprite placement.
TABLE_ROWS = TABLE_COLS = (3, 4)
HEIGHT_BY_TABLE_ROWS = {3: 16, 4: 19}
WIDTHS_BY_TABLE_COLS = {3: [13, 16, 17], 4: [18, 21]}
ROW_BARS = np.array([4, 9, 14])
BAR_START, BAR_SPACING = 4, 5
MIN_USABLE_SPAN = 3
SPRITE_OFFSET = (1, 1)
SPRITE = np.ones((2, 2), dtype=int)
MAX_ATTEMPTS = 300

def _sample_case() -> tuple[int, list[int], list[int], np.ndarray]:
    """Choose colors/counts and render the matching histogram output."""
    n_colors = random.randint(*N_COLORS_RANGE)
    max_count = random.randint(max(MAX_COUNT_RANGE[0], n_colors), MAX_COUNT_RANGE[1])
    counts = sorted(random.sample(range(COUNT_SAMPLE_START, max_count), n_colors - 1) + [max_count])
    palette = random.sample(COLOR_POOL, n_colors + 1)
    separator, counted_colors = palette[0], palette[1:]
    out = np.zeros((n_colors, max_count), dtype=int)
    for r, (color, count) in enumerate(zip(counted_colors, counts)):
        out[r, :count] = color
    return separator, counted_colors, counts, out

def _build_grid(needed: int, separator: int) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Create the separator grid and return usable cell origins plus expected count."""
    layouts = [(r, c) for r in TABLE_ROWS for c in TABLE_COLS if r * c > needed]
    table_rows, table_cols = random.choice(layouts)
    height, width = HEIGHT_BY_TABLE_ROWS[table_rows], random.choice(WIDTHS_BY_TABLE_COLS[table_cols])
    col_bars = np.arange(BAR_START, width, BAR_SPACING)
    grid = np.zeros((height, width), dtype=int)
    grid[ROW_BARS, :] = separator
    grid[:, col_bars] = separator
    usable = [
        (r0, c0)
        for r0, r1 in _spans(height, ROW_BARS)
        if r1 - r0 >= MIN_USABLE_SPAN
        for c0, c1 in _spans(width, col_bars)
        if c1 - c0 >= MIN_USABLE_SPAN
    ]
    return grid, usable, table_rows * table_cols

def _fill_cells(grid, usable, counted_colors, counts, separator) -> None:
    """Place counted sprites, then fill remaining usable cells with separator sprites."""
    cell_colors = [color for color, count in zip(counted_colors, counts) for _ in range(count)]
    cell_colors.extend([separator] * (len(usable) - len(cell_colors)))
    random.shuffle(cell_colors)
    dr, dc = SPRITE_OFFSET
    for (r0, c0), color in zip(usable, cell_colors):
        paste(grid, SPRITE * color, r0 + dr, c0 + dc)

def _gen_attempt() -> np.ndarray | None:
    separator, counted_colors, counts, out = _sample_case()
    inp, usable, expected_cells = _build_grid(sum(counts), separator)
    if len(usable) != expected_cells:
        return None
    _fill_cells(inp, usable, counted_colors, counts, separator)
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate valid input")
