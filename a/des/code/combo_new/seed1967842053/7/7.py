import random

import numpy as np


def _mode_in_scan_order(values):
    counts = {}
    order = []
    for value in np.asarray(values).ravel().tolist():
        value = int(value)
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    return max(order, key=lambda value: counts[value])


def main(grid: np.ndarray) -> np.ndarray:
    """Recover legend-marked horizontal bands and mix crossings as checkerboards.

    Input structure:
    The grid has a normal drawing area plus a one-cell-wide legend column at the
    far right. In the drawing area, complete rows are horizontal color bands and
    complete columns are vertical color bands. Vertical bands cover the crossing
    cells in the input, while the legend column stores the true horizontal band
    color for each band row and the background color for all other rows.

    Rule:
    Remove the legend column. For every row whose legend entry is not background,
    recover that row's horizontal band color from the legend. For every full
    uniform non-background column in the drawing area, treat it as a vertical
    band. At each horizontal/vertical crossing, replace the crossing cell and
    the three cells down/right of it with a 2x2 checkerboard: horizontal color
    on the top-left and bottom-right, vertical color on the top-right and
    bottom-left.

    Key insight:
    The visible crossing color is misleading because it only shows the vertical
    band; the right-side legend is the only source for the hidden horizontal
    color needed to build each two-color checkerboard.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.shape[1] < 2:
        return grid.copy()

    legend = grid[:, -1]
    background = _mode_in_scan_order(legend)
    drawing = grid[:, :-1]
    out = drawing.copy()

    horizontal_bands = [
        (row, int(color))
        for row, color in enumerate(legend.tolist())
        if int(color) != background
    ]

    vertical_bands = []
    for col in range(drawing.shape[1]):
        column = drawing[:, col]
        color = int(column[0])
        if color != background and np.all(column == color):
            vertical_bands.append((col, color))

    height, width = out.shape
    for row, h_color in horizontal_bands:
        for col, v_color in vertical_bands:
            if row + 1 >= height or col + 1 >= width:
                continue
            out[row, col] = h_color
            out[row, col + 1] = v_color
            out[row + 1, col] = v_color
            out[row + 1, col + 1] = h_color

    return out


def _spaced_positions(length, count, gap):
    for _ in range(300):
        picked = sorted(random.sample(range(length), count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked

    picked = []
    start = random.randrange(gap)
    for pos in range(start, length, gap):
        picked.append(pos)
        if len(picked) == count:
            return picked
    return list(range(0, count * gap, gap))


def generate_input() -> np.ndarray:
    rows = random.randint(7, 22)
    drawing_cols = random.randint(7, 22)
    gap = random.choice([2, 3])

    row_slots = rows - 1
    col_slots = drawing_cols - 1
    max_h = min(3, max(1, (row_slots + gap - 1) // gap))
    max_v = min(3, max(1, (col_slots + gap - 1) // gap))
    h_count = random.randint(1, max_h)
    v_count = random.randint(1, max_v)

    h_rows = _spaced_positions(row_slots, h_count, gap)
    v_cols = _spaced_positions(col_slots, v_count, gap)

    colors = list(range(10))
    random.shuffle(colors)
    background = colors[0]
    h_colors = colors[1 : 1 + h_count]
    v_colors = colors[1 + h_count : 1 + h_count + v_count]

    grid = np.full((rows, drawing_cols + 1), background, dtype=int)

    for row, color in zip(h_rows, h_colors):
        grid[row, :-1] = color
        grid[row, -1] = color

    for col, color in zip(v_cols, v_colors):
        grid[:, col] = color

    return grid
