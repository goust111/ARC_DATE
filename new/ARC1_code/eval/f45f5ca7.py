import numpy as np

from arc_utils import Color, get_bg


LANE_COLUMN = {
    Color.TEAL: 1,
    Color.RED: 2,
    Color.YELLOW: 3,
    Color.GREEN: 4,
}


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored left-edge pixel into its color-specific lane on the same row.

    Input structure
    The grid has a dominant background color, and every foreground cell lies in
    the leftmost column.  The foreground colors are teal, red, yellow, and
    green.

    Rule
    Erase the leftmost-column foreground cells, then repaint each one in the
    same row at its fixed color lane: teal in column 1, red in column 2, yellow
    in column 3, and green in column 4.

    Key insight
    The color is preserved and acts as the horizontal lane code, so all
    occurrences of a color move to the same output column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fg = g != bg
    coords = np.argwhere(fg)
    assert len(coords) > 0
    assert np.all(coords[:, 1] == 0)
    assert set(map(int, g[fg])) == set(LANE_COLUMN)
    assert g.shape[1] > max(LANE_COLUMN.values())

    out = np.full_like(g, bg)
    for r, c in coords:
        color = int(g[r, c])
        out[int(r), LANE_COLUMN[color]] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas, occupancy, and retry bounds.
_GRID_SHAPE_F45F5CA7 = (10, 10)
_SOURCE_COLUMN_F45F5CA7 = 0
_TOTAL_FG_RANGE_F45F5CA7 = (7, 9)
_PER_COLOR_COUNT_RANGE_F45F5CA7 = range(1, 4)
_MAX_GENERATION_ATTEMPTS_F45F5CA7 = 300
# Palette and fixed output lanes.
_GEN_COLORS_F45F5CA7 = (8, 2, 4, 3)
_GEN_LANES_F45F5CA7 = {8: 1, 2: 2, 4: 3, 3: 4}


def _gen_color_counts_f45f5ca7(total):
    options = []
    for a in _PER_COLOR_COUNT_RANGE_F45F5CA7:
        for b in _PER_COLOR_COUNT_RANGE_F45F5CA7:
            for c in _PER_COLOR_COUNT_RANGE_F45F5CA7:
                d = total - a - b - c
                if d in _PER_COLOR_COUNT_RANGE_F45F5CA7:
                    options.append((a, b, c, d))
    return options[np.random.randint(len(options))]


def _gen_color_sequence_f45f5ca7(total):
    counts = _gen_color_counts_f45f5ca7(total)
    colors = np.repeat(_GEN_COLORS_F45F5CA7, counts).astype(int)
    np.random.shuffle(colors)
    return colors


def _gen_render_pair_f45f5ca7(rows, colors):
    grid = np.zeros(_GRID_SHAPE_F45F5CA7, dtype=int)
    expected = np.zeros_like(grid)
    grid[rows, _SOURCE_COLUMN_F45F5CA7] = colors
    lane_columns = [_GEN_LANES_F45F5CA7[int(color)] for color in colors]
    expected[rows, lane_columns] = colors
    return grid, expected


def _gen_attempt():
    total = int(np.random.randint(*_TOTAL_FG_RANGE_F45F5CA7))
    rows = np.random.choice(_GRID_SHAPE_F45F5CA7[0], size=total, replace=False)
    colors = _gen_color_sequence_f45f5ca7(total)
    grid, expected = _gen_render_pair_f45f5ca7(rows, colors)

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input():
    for _ in range(_MAX_GENERATION_ATTEMPTS_F45F5CA7):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f45f5ca7 input")
