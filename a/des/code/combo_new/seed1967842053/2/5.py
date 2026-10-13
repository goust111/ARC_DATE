import numpy as np


GRAY = 5
BACKGROUND = 0


def _framework_bounds(grid):
    pts = np.argwhere(grid == GRAY)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _instruction_colors(grid):
    colors = []
    for color in np.unique(grid):
        color = int(color)
        if color not in (BACKGROUND, GRAY):
            colors.append(color)
    return colors


def _extract_thread_colors(grid, h_count, v_count):
    horizontal = []
    vertical = []
    singles = []

    for color in _instruction_colors(grid):
        pts = np.argwhere(grid == color)
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        if len(rows) == 1 and len(cols) > 1:
            horizontal.append((int(rows[0]), color))
        elif len(cols) == 1 and len(rows) > 1:
            vertical.append((int(cols[0]), color))
        elif len(rows) == 1 and len(cols) == 1:
            singles.append((int(rows[0]), int(cols[0]), color))

    if singles:
        missing_h = h_count - len(horizontal)
        missing_v = v_count - len(vertical)
        singles.sort()
        if missing_h > 0 and missing_v <= 0:
            horizontal.extend((r, color) for r, _c, color in singles)
        elif missing_v > 0 and missing_h <= 0:
            vertical.extend((c, color) for _r, c, color in singles)
        else:
            horizontal.extend((r, color) for r, _c, color in singles[:missing_h])
            vertical.extend((c, color) for _r, c, color in singles[missing_h:])

    horizontal.sort(key=lambda item: item[0])
    vertical.sort(key=lambda item: item[0])
    return [color for _pos, color in horizontal], [color for _pos, color in vertical]


def main(grid: np.ndarray) -> np.ndarray:
    """Weave the instruction thread colors into the lower gray lattice frame.

    Input structure:
    The background is black. Above the frame are isolated colored horizontal
    and vertical line segments, with each segment using a distinct non-gray
    color. Below them is one gray rectangular lattice whose full gray rows and
    columns delimit one-pixel cells.

    Rule:
    Crop out the gray lattice frame. Sort upper horizontal segments from top to
    bottom and assign them to frame cell rows. Sort upper vertical segments from
    left to right and assign them to frame cell columns. Keep the gray lattice
    pixels unchanged. For each cell at row i and column j, write the horizontal
    color when i + j is even, otherwise write the vertical color.

    Key insight:
    The colored segments only define row and column thread palettes; the visible
    color at every crossing is determined by the checkerboard parity of the
    frame cell coordinates.
    """
    grid = np.asarray(grid)
    r0, c0, r1, c1 = _framework_bounds(grid)
    frame = grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    row_lines = [r for r in range(frame.shape[0]) if np.all(frame[r, :] == GRAY)]
    col_lines = [c for c in range(frame.shape[1]) if np.all(frame[:, c] == GRAY)]
    h_count = len(row_lines) - 1
    v_count = len(col_lines) - 1

    h_colors, v_colors = _extract_thread_colors(grid, h_count, v_count)

    for i in range(h_count):
        cell_r = (row_lines[i] + row_lines[i + 1]) // 2
        for j in range(v_count):
            cell_c = (col_lines[j] + col_lines[j + 1]) // 2
            frame[cell_r, cell_c] = h_colors[i] if (i + j) % 2 == 0 else v_colors[j]

    return frame


def _can_place(area, pixels):
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    r0 = max(0, min(rows) - 1)
    r1 = min(area.shape[0] - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(area.shape[1] - 1, max(cols) + 1)
    return not np.any(area[r0 : r1 + 1, c0 : c1 + 1])


def _place_segment(area, length, horizontal, color, rng, used_rows, used_cols):
    height, width = area.shape
    for _ in range(400):
        if horizontal:
            row = int(rng.integers(0, height))
            if row in used_rows:
                continue
            col = int(rng.integers(0, width - length + 1))
            pixels = [(row, c) for c in range(col, col + length)]
        else:
            col = int(rng.integers(0, width))
            if col in used_cols:
                continue
            row = int(rng.integers(0, height - length + 1))
            pixels = [(r, col) for r in range(row, row + length)]

        if _can_place(area, pixels):
            for r, c in pixels:
                area[r, c] = color
            if horizontal:
                used_rows.add(row)
            else:
                used_cols.add(col)
            return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    pairs = [(h, v) for h in range(2, 6) for v in range(2, 6) if h + v <= 8]
    h_count, v_count = pairs[int(rng.integers(0, len(pairs)))]

    frame_h = 2 * h_count + 1
    frame_w = 2 * v_count + 1
    min_top_h = max(8, h_count + 4)
    max_top_h = min(16, 30 - frame_h - 1)
    top_h = int(rng.integers(min_top_h, max_top_h + 1))
    sep_h = 1
    max_bottom = 30 - top_h - sep_h - frame_h
    bottom_margin = int(rng.integers(0, min(3, max_bottom + 1)))

    min_width = max(12, frame_w + 2, h_count + 5, v_count + 5)
    width = int(rng.integers(min_width, 31))
    frame_col = int(rng.integers(0, width - frame_w + 1))
    frame_top = top_h + sep_h

    grid = np.zeros((frame_top + frame_h + bottom_margin, width), dtype=int)
    top = grid[:top_h, :]

    colors = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
    selected = rng.choice(colors, size=h_count + v_count, replace=False)
    tasks = []
    for idx in range(h_count):
        tasks.append((True, v_count, int(selected[idx])))
    for idx in range(v_count):
        tasks.append((False, h_count, int(selected[h_count + idx])))
    rng.shuffle(tasks)

    used_rows = set()
    used_cols = set()
    for horizontal, length, color in tasks:
        placed = _place_segment(top, length, horizontal, color, rng, used_rows, used_cols)
        if not placed:
            return generate_input()

    for rr in range(frame_h):
        for cc in range(frame_w):
            if rr % 2 == 0 or cc % 2 == 0:
                grid[frame_top + rr, frame_col + cc] = GRAY

    return grid
