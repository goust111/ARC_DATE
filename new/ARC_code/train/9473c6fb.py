import numpy as np

from arc_utils import Color, bbox_coords, coords_where, get_bg, in_bounds, neighbors4


_CYCLE = (Color.RED, Color.TEAL, Color.GREY)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the sparse foreground pixels in scan order along their longer spread, cycling red, teal, and grey.

    Input structure:
    - The dominant color is the background.
    - Every foreground object is a single isolated 4-connected pixel.
    - The foreground pixel set has a rectangular bbox whose row or column extent defines the scan direction.

    Rule:
    - Keep the background unchanged.
    - If the foreground bbox is at least as tall as it is wide, order foreground pixels row-major; otherwise order them column-major.
    - Recolor the ordered pixels by repeating red, teal, grey from the first pixel onward.

    Key insight:
    - The original foreground colors are only markers; the output color depends on each marker's ordinal position in the dominant scan direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg not in _CYCLE, "background must be distinct from the output color cycle"

    fg = g != bg
    coords = coords_where(fg)
    assert len(coords) > 0, "expected at least one foreground marker"

    coords_list = [(int(r), int(c)) for r, c in coords]
    for r, c in coords_list:
        assert all(
            not in_bounds(g.shape, nr, nc) or int(g[nr, nc]) == bg
            for nr, nc in neighbors4(r, c)
        ), "foreground markers must be isolated 4-connected pixels"

    r0, c0, r1, c1 = bbox_coords(coords)
    row_extent = int(r1 - r0 + 1)
    col_extent = int(c1 - c0 + 1)
    if row_extent >= col_extent:
        ordered = sorted(coords_list)
    else:
        ordered = sorted(coords_list, key=lambda rc: (rc[1], rc[0]))

    out = g.copy()
    for i, (r, c) in enumerate(ordered):
        out[r, c] = _CYCLE[i % len(_CYCLE)]

    return out

# --------------- generator (att2) ---------------

import random

# Canvas and marker density bounds.
HEIGHT_RANGE = range(6, 10)
WIDTH_RANGE = range(8, 13)
MARKER_COUNT_RANGE = (6, 10)
DENSITY_RANGE = (0.09, 0.15)
# Bbox span and palette constraints.
MIN_CROSS_SPAN = 3
PALETTE_SIZE_RANGE = (3, 4)
BACKGROUND_COLORS = [0, 1, 3, 4, 6, 7, 9]
PREFERRED_BACKGROUND = 7
PREFERRED_BACKGROUND_RATE = 0.6
OUTPUT_CYCLE = (2, 8, 5)
# Generation retry limit.
MAX_ATTEMPTS = 300


def _gen_value_sequence(scan_offsets, span):
    values = []

    def can_place(i, value):
        return i == 0 or scan_offsets[i] - scan_offsets[i - 1] != 1 or abs(value - values[-1]) >= 2

    def search(i):
        if i == len(scan_offsets):
            return 0 in values and span - 1 in values
        choices = list(range(span))
        random.shuffle(choices)
        for value in choices:
            if can_place(i, value):
                values.append(value)
                if search(i + 1):
                    return True
                values.pop()
        return False

    return values if search(0) else None


def _candidate_layouts(marker_count):
    choices = []
    for height in HEIGHT_RANGE:
        for width in WIDTH_RANGE:
            density = marker_count / (height * width)
            if not (DENSITY_RANGE[0] <= density <= DENSITY_RANGE[1]):
                continue
            directions = (["row"] if height >= marker_count else []) + (["col"] if width >= marker_count else [])
            if directions:
                choices.append((height, width, directions))
    return choices


def _scan_offsets(span, marker_count):
    offsets = [0, span - 1]
    offsets.extend(random.sample(range(1, span - 1), marker_count - 2))
    offsets.sort()
    return offsets


def _row_major_coords(height, width, marker_count):
    row_span = random.randint(marker_count, height)
    max_col_span = min(width, row_span, marker_count)
    if max_col_span < MIN_CROSS_SPAN:
        return None
    col_span = random.randint(MIN_CROSS_SPAN, max_col_span)
    r0 = random.randint(0, height - row_span)
    c0 = random.randint(0, width - col_span)

    rows = _scan_offsets(row_span, marker_count)
    cols = _gen_value_sequence(rows, col_span)
    return None if cols is None else [(r0 + row, c0 + col) for row, col in zip(rows, cols)]


def _col_major_coords(height, width, marker_count):
    col_span = random.randint(marker_count, width)
    max_row_span = min(height, col_span - 1, marker_count)
    if max_row_span < MIN_CROSS_SPAN:
        return None
    row_span = random.randint(MIN_CROSS_SPAN, max_row_span)
    r0 = random.randint(0, height - row_span)
    c0 = random.randint(0, width - col_span)

    cols = _scan_offsets(col_span, marker_count)
    rows = _gen_value_sequence(cols, row_span)
    return None if rows is None else [(r0 + row, c0 + col) for row, col in zip(rows, cols)]


def _choose_background():
    if random.random() < PREFERRED_BACKGROUND_RATE:
        return PREFERRED_BACKGROUND
    return random.choice([c for c in BACKGROUND_COLORS if c != PREFERRED_BACKGROUND])


def _paint_markers(height, width, coords):
    bg = _choose_background()
    grid = np.full((height, width), bg, dtype=int)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)
    marker_colors = palette + [random.choice(palette) for _ in range(len(coords) - palette_size)]
    random.shuffle(marker_colors)
    for (r, c), color in zip(coords, marker_colors):
        grid[r, c] = color
    return grid


def _expected_output(grid, coords, direction):
    expected = grid.copy()
    ordered = sorted(coords) if direction == "row" else sorted(coords, key=lambda rc: (rc[1], rc[0]))
    for i, (r, c) in enumerate(ordered):
        expected[r, c] = OUTPUT_CYCLE[i % len(OUTPUT_CYCLE)]
    return expected


def _gen_attempt():
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    choices = _candidate_layouts(marker_count)
    if not choices:
        return None

    height, width, directions = random.choice(choices)
    direction = random.choice(directions)
    if direction == "row":
        coords = _row_major_coords(height, width, marker_count)
    else:
        coords = _col_major_coords(height, width, marker_count)
    if coords is None:
        return None

    grid = _paint_markers(height, width, coords)
    try:
        if np.array_equal(main(grid), _expected_output(grid, coords, direction)):
            return grid
    except AssertionError:
        pass
    return None


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9473c6fb input")
