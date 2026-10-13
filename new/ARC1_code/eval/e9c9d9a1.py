import numpy as np

from arc_utils import Color, colors_present, get_bg


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1, *lines, limit]
    regions = [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:])]
    assert all(r0 <= r1 for r0, r1 in regions)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Fill a green-lined table by coloring its four corner cells and its fully internal cells.

    Input structure
    - The input contains a black background and one foreground separator color forming complete horizontal and vertical grid lines.
    - Those separator lines divide the canvas into rectangular background cells.
    Rule
    - Preserve the separator lines, fill the top-left/top-right/bottom-left/bottom-right cells with red/yellow/blue/teal, and fill every cell that is neither in an outer row band nor an outer column band with orange.
    - Leave all other edge cells black.
    Key insight
    - The green scaffold is a table; output colors depend only on each cell's position in that table: corner, strict interior, or remaining edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    line_color = fg_colors[0]
    assert line_color == Color.GREEN

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == line_color)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == line_color)]
    assert full_rows and full_cols

    line_mask = g == line_color
    expected_line_mask = np.zeros(g.shape, dtype=bool)
    expected_line_mask[full_rows, :] = True
    expected_line_mask[:, full_cols] = True
    assert np.array_equal(line_mask, expected_line_mask)

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) >= 2 and len(col_regions) >= 2

    out = g.copy()
    last_row = len(row_regions) - 1
    last_col = len(col_regions) - 1
    for row_i, (r0, r1) in enumerate(row_regions):
        for col_i, (c0, c1) in enumerate(col_regions):
            fill = None
            if row_i == 0 and col_i == 0:
                fill = Color.RED
            elif row_i == 0 and col_i == last_col:
                fill = Color.YELLOW
            elif row_i == last_row and col_i == 0:
                fill = Color.BLUE
            elif row_i == last_row and col_i == last_col:
                fill = Color.TEAL
            elif 0 < row_i < last_row and 0 < col_i < last_col:
                fill = Color.ORANGE

            if fill is not None:
                region = g[r0 : r1 + 1, c0 : c1 + 1]
                assert np.all(region == bg)
                out[r0 : r1 + 1, c0 : c1 + 1] = fill

    return out

# --------------- generator (o5x) ---------------

# Canvas, separator counts, band widths, and retry budget.
GRID_HEIGHT_RANGE = (12, 20)  # rng-exclusive high
GRID_WIDTH_RANGE = (12, 17)
LINE_COUNT_RANGE = (2, 5)
BAND_WIDTH_RANGE = (1, 6)  # inclusive
GENERATION_RETRIES = 300


def _gen_partition_bands(total: int, line_count: int, rng: np.random.Generator) -> list[int]:
    band_count = line_count + 1
    remaining = total - line_count
    widths = []
    min_band, max_band = BAND_WIDTH_RANGE
    for i in range(band_count):
        left = band_count - i - 1
        lo = max(min_band, remaining - max_band * left)
        hi = min(max_band, remaining - min_band * left)
        width = int(rng.integers(lo, hi + 1))
        widths.append(width)
        remaining -= width
    rng.shuffle(widths)
    return widths


def _gen_layout(rng: np.random.Generator) -> tuple[int, int, int, int]:
    num_hlines = int(rng.integers(*LINE_COUNT_RANGE))
    num_vlines = int(rng.integers(*LINE_COUNT_RANGE))
    choices = []
    for height in range(*GRID_HEIGHT_RANGE):
        for width in range(*GRID_WIDTH_RANGE):
            green = num_hlines * width + num_vlines * height - num_hlines * num_vlines
            if height * width - green > green:
                choices.append((height, width))
    height, width = choices[int(rng.integers(0, len(choices)))]
    return height, width, num_hlines, num_vlines


def _gen_line_positions(bands: list[int]) -> list[int]:
    """Return separator indices implied by shuffled cell band widths."""
    positions = []
    pos = bands[0]
    for band_width in bands[1:]:
        positions.append(pos)
        pos += 1 + band_width
    return positions


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height, width, num_hlines, num_vlines = _gen_layout(rng)

    row_bands = _gen_partition_bands(height, num_hlines, rng)
    col_bands = _gen_partition_bands(width, num_vlines, rng)

    grid = np.zeros((height, width), dtype=int)
    grid[_gen_line_positions(row_bands), :] = Color.GREEN
    grid[:, _gen_line_positions(col_bands)] = Color.GREEN

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e9c9d9a1 input")
