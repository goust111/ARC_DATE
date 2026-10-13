import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GRAY = 5


def _bbox(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("expected marked panel pixels")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    return int(r0), int(r1), int(c0), int(c1)


def _panel_lines(panel, color):
    row_lines = [i for i in range(panel.shape[0]) if np.all(panel[i, :] == color)]
    col_lines = [j for j in range(panel.shape[1]) if np.all(panel[:, j] == color)]
    if len(row_lines) < 2 or len(col_lines) < 2:
        raise ValueError("panel grid lines are ambiguous")
    return row_lines, col_lines


def _intervals(lines):
    return [(lines[i] + 1, lines[i + 1]) for i in range(len(lines) - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the lit window mask to the matching canvas cells.

    Input structure:
    The input is a black grid with a blue-bordered window panel on the left and
    a gray-bordered canvas panel on the right, separated by black space.  Both
    panels are rectangular lattices with the same number of rows and columns.
    Window interiors are either yellow, meaning lit, or black, meaning dark.
    Canvas interiors contain colored patterns on black backgrounds.

    Rule:
    Crop the gray canvas panel for the output.  For every lattice position,
    keep the canvas interior unchanged when the matching window cell is yellow.
    When the matching window cell is dark, replace only that canvas interior
    with black while leaving all gray borders intact.

    Key insight:
    The blue window panel is a spatial on/off mask for the corresponding cells
    of the gray canvas panel.
    """
    grid = np.asarray(grid)
    br0, br1, bc0, bc1 = _bbox(grid == BLUE)
    gr0, gr1, gc0, gc1 = _bbox(grid == GRAY)

    window = grid[br0:br1, bc0:bc1]
    canvas = grid[gr0:gr1, gc0:gc1].copy()

    w_rows, w_cols = _panel_lines(window, BLUE)
    c_rows, c_cols = _panel_lines(canvas, GRAY)
    w_row_spans = _intervals(w_rows)
    w_col_spans = _intervals(w_cols)
    c_row_spans = _intervals(c_rows)
    c_col_spans = _intervals(c_cols)

    rows = min(len(w_row_spans), len(c_row_spans))
    cols = min(len(w_col_spans), len(c_col_spans))
    for r in range(rows):
        wy0, wy1 = w_row_spans[r]
        cy0, cy1 = c_row_spans[r]
        for c in range(cols):
            wx0, wx1 = w_col_spans[c]
            cx0, cx1 = c_col_spans[c]
            lit = np.any(window[wy0:wy1, wx0:wx1] == YELLOW)
            if not lit:
                canvas[cy0:cy1, cx0:cx1] = BLACK
    return canvas


def _draw_lattice(grid, top, left, rows, cols, cell_h, cell_w, color):
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    for r in range(rows + 1):
        y = top + r * (cell_h + 1)
        grid[y, left:left + width] = color
    for c in range(cols + 1):
        x = left + c * (cell_w + 1)
        grid[top:top + height, x] = color
    return height, width


def _cell_origin(top, left, r, c, cell_h, cell_w):
    return top + r * (cell_h + 1) + 1, left + c * (cell_w + 1) + 1


def _paint_pattern(grid, y, x, h, w, colors, style):
    color_a = colors[0]
    color_b = colors[1]
    if style == 0:
        grid[y:y + h, x + w // 2] = color_a
        grid[y + h // 2, x:x + w] = color_b
    elif style == 1:
        n = min(h, w)
        for k in range(n):
            grid[y + k, x + k] = color_a
            grid[y + k, x + w - 1 - k] = color_b
    elif style == 2:
        grid[y, x:x + w] = color_a
        grid[y:y + h, x] = color_a
        grid[y + h - 1, x:x + w] = color_b
    elif style == 3:
        grid[y:y + h, x + w - 1] = color_a
        grid[y + h - 1, x:x + w] = color_b
    elif style == 4:
        grid[y:y + h, x:x + w] = color_a
        if h > 2 and w > 2:
            grid[y + 1:y + h - 1, x + 1:x + w - 1] = BLACK
            grid[y + h // 2, x + w // 2] = color_b
    else:
        grid[y, x] = color_a
        grid[y + h - 1, x + w - 1] = color_b
        grid[y + h // 2, x:x + w] = color_a


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    while True:
        rows = int(rng.integers(2, 5))
        cols = int(rng.integers(2, 5))
        max_cell_w = max(2, (26 - (cols + 1) * 2) // (2 * cols))
        cell_h = int(rng.integers(2, 6))
        cell_w = int(rng.integers(2, min(6, max_cell_w + 1)))
        panel_h = rows * cell_h + rows + 1
        panel_w = cols * cell_w + cols + 1
        top = int(rng.integers(0, 4))
        left = int(rng.integers(0, 4))
        bottom = int(rng.integers(0, 4))
        right = int(rng.integers(0, 4))
        gap = 1
        height = top + panel_h + bottom
        width = left + panel_w + gap + panel_w + right
        if height <= 30 and width <= 30:
            break

    grid = np.zeros((height, width), dtype=int)
    canvas_left = left + panel_w + gap
    _draw_lattice(grid, top, left, rows, cols, cell_h, cell_w, BLUE)
    _draw_lattice(grid, top, canvas_left, rows, cols, cell_h, cell_w, GRAY)

    mask = rng.integers(0, 2, size=(rows, cols), dtype=int)
    while np.all(mask == 0) or np.all(mask == 1):
        mask = rng.integers(0, 2, size=(rows, cols), dtype=int)

    palette = np.array([2, 3, 6, 7, 8, 9])
    for r in range(rows):
        for c in range(cols):
            wy, wx = _cell_origin(top, left, r, c, cell_h, cell_w)
            if mask[r, c]:
                grid[wy:wy + cell_h, wx:wx + cell_w] = YELLOW

            cy, cx = _cell_origin(top, canvas_left, r, c, cell_h, cell_w)
            pair = rng.choice(palette, size=2, replace=False)
            style = int((r * cols + c + rng.integers(0, 6)) % 6)
            _paint_pattern(grid, cy, cx, cell_h, cell_w, pair, style)

    return grid
