import random
from collections import Counter

import numpy as np


_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

_GEN_RETRIES = 400
_GRID_COUNT_RANGE = (2, 4)
_CELL_SIZE_RANGE = (5, 8)
_FRAME_COLOR_POOL = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if row_lines and col_lines:
            candidates.append((color, len(row_lines) + len(col_lines)))
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0][0]


def _intervals_between_lines(line_positions: list[int], size: int) -> list[tuple[int, int]]:
    line_set = set(map(int, line_positions))
    intervals: list[tuple[int, int]] = []
    start = None
    for idx in range(size):
        if idx in line_set:
            if start is not None:
                intervals.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        intervals.append((start, size))
    return intervals


def _regular_square_regions(grid: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_lines = [r for r in range(h) if np.all(grid[r, :] == sep)]
    col_lines = [c for c in range(w) if np.all(grid[:, c] == sep)]
    row_intervals = _intervals_between_lines(row_lines, h)
    col_intervals = _intervals_between_lines(col_lines, w)
    assert row_intervals and col_intervals, "expected separator-partitioned regions"
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "regions must be equal-sized"
    assert next(iter(row_sizes)) == next(iter(col_sizes)), "regions must be square"
    return row_intervals, col_intervals


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    assert rows.size > 0, "expected a non-empty object"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _gap_direction(cell: np.ndarray, bg: int) -> tuple[str, tuple[int, int, int, int], int]:
    mask = cell != bg
    rr0, cc0, rr1, cc1 = _bbox(mask)
    obj = mask[rr0 : rr1 + 1, cc0 : cc1 + 1]
    height, width = obj.shape
    assert height >= 3 and width >= 3, "frame must have an interior"
    assert np.count_nonzero(obj[1:-1, 1:-1]) == 0, "frame interior must be empty"

    side_deficits = {
        "up": width - int(np.count_nonzero(obj[0, :])),
        "down": width - int(np.count_nonzero(obj[-1, :])),
        "left": height - int(np.count_nonzero(obj[:, 0])),
        "right": height - int(np.count_nonzero(obj[:, -1])),
    }
    gap_sides = [name for name, deficit in side_deficits.items() if deficit == 1]
    assert len(gap_sides) == 1, "expected a unique one-pixel gap side"
    gap_side = gap_sides[0]
    for name, deficit in side_deficits.items():
        if name != gap_side:
            assert deficit == 0, "non-gap sides must be intact"

    colors = set(map(int, np.unique(cell[mask])))
    assert len(colors) == 1, "each region should contain a monochrome frame"
    return gap_side, (rr0, cc0, rr1, cc1), next(iter(colors))


def _separator_neighbors(start: int, end: int, limit: int) -> tuple[int | None, int | None]:
    top_or_left = start - 1 if start > 0 else None
    bottom_or_right = end if end < limit else None
    return top_or_left, bottom_or_right


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-partitioned region grid, each frame's one-pixel gap points to the
    neighboring region whose separator wall should be opened.

    Input structure:
    - Background is the dominant color.
    - A separator color forms full horizontal and vertical one-pixel lines that divide
      the canvas into equal square regions.
    - Every region contains one monochrome square frame with exactly one single-pixel
      gap on one of its four sides.

    Rule:
    - Detect the gap side for each region.
    - Remove the separator segment between that region and the neighboring region on
      the gap side.
    - Also remove the entire frame side opposite the gap, turning the frame into a U.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _find_separator_color(g, bg)
    row_intervals, col_intervals = _regular_square_regions(g, sep)
    out = g.copy()

    for r0, r1 in row_intervals:
        top_sep, bottom_sep = _separator_neighbors(r0, r1, g.shape[0])
        for c0, c1 in col_intervals:
            left_sep, right_sep = _separator_neighbors(c0, c1, g.shape[1])
            cell = g[r0:r1, c0:c1]
            gap_side, (fr0, fc0, fr1, fc1), _color = _gap_direction(cell, bg)

            if gap_side == "up" and top_sep is not None:
                out[top_sep, c0:c1] = bg
            elif gap_side == "down" and bottom_sep is not None:
                out[bottom_sep, c0:c1] = bg
            elif gap_side == "left" and left_sep is not None:
                out[r0:r1, left_sep] = bg
            elif gap_side == "right" and right_sep is not None:
                out[r0:r1, right_sep] = bg

            opp = _OPPOSITE[gap_side]
            if opp == "up":
                out[r0 + fr0, c0 + fc0 : c0 + fc1 + 1] = bg
            elif opp == "down":
                out[r0 + fr1, c0 + fc0 : c0 + fc1 + 1] = bg
            elif opp == "left":
                out[r0 + fr0 : r0 + fr1 + 1, c0 + fc0] = bg
            else:
                out[r0 + fr0 : r0 + fr1 + 1, c0 + fc1] = bg

    return out


def _valid_dirs(row: int, col: int, rows: int, cols: int) -> list[str]:
    dirs: list[str] = []
    if row > 0:
        dirs.append("up")
    if row + 1 < rows:
        dirs.append("down")
    if col > 0:
        dirs.append("left")
    if col + 1 < cols:
        dirs.append("right")
    return dirs


def _balanced_directions(rows: int, cols: int) -> list[list[str]]:
    counts = Counter({name: 0 for name in _DIRS})
    order = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(order)
    assigned = [[None for _ in range(cols)] for _ in range(rows)]
    for r, c in order:
        valid = _valid_dirs(r, c, rows, cols)
        best_score = min(counts[name] for name in valid)
        options = [name for name in valid if counts[name] == best_score]
        choice = random.choice(options)
        assigned[r][c] = choice
        counts[choice] += 1
    assert all(all(cell is not None for cell in row) for row in assigned)
    return assigned


def _draw_frame(cell: np.ndarray, bg: int, color: int, gap_side: str) -> None:
    h, w = cell.shape
    side = random.randint(3, min(h, w) - 2)
    top = random.randint(1, h - side - 1)
    left = random.randint(1, w - side - 1)
    bottom = top + side - 1
    right = left + side - 1

    cell[top, left : right + 1] = color
    cell[bottom, left : right + 1] = color
    cell[top : bottom + 1, left] = color
    cell[top : bottom + 1, right] = color

    gap_idx = random.randint(1, side - 2)
    if gap_side == "up":
        cell[top, left + gap_idx] = bg
    elif gap_side == "down":
        cell[bottom, left + gap_idx] = bg
    elif gap_side == "left":
        cell[top + gap_idx, left] = bg
    else:
        cell[top + gap_idx, right] = bg


def _gen_attempt() -> np.ndarray | None:
    region_rows = random.randint(*_GRID_COUNT_RANGE)
    region_cols = random.randint(*_GRID_COUNT_RANGE)
    cell_size = random.randint(*_CELL_SIZE_RANGE)

    palette = list(_FRAME_COLOR_POOL)
    bg = random.choice(palette)
    palette.remove(bg)
    sep = random.choice(palette)
    palette.remove(sep)
    frame_palette_size = random.randint(1, min(3, len(palette)))
    frame_colors = random.sample(palette, frame_palette_size)

    height = region_rows * cell_size + region_rows + 1
    width = region_cols * cell_size + region_cols + 1
    grid = np.full((height, width), bg, dtype=int)

    row_lines = [i * (cell_size + 1) for i in range(region_rows + 1)]
    col_lines = [i * (cell_size + 1) for i in range(region_cols + 1)]
    for r in row_lines:
        grid[r, :] = sep
    for c in col_lines:
        grid[:, c] = sep

    directions = _balanced_directions(region_rows, region_cols)
    row_intervals, col_intervals = _regular_square_regions(grid, sep)

    for r_index, (r0, r1) in enumerate(row_intervals):
        for c_index, (c0, c1) in enumerate(col_intervals):
            color = random.choice(frame_colors)
            cell = grid[r0:r1, c0:c1]
            _draw_frame(cell, bg, color, directions[r_index][c_index])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len({directions[r][c] for r in range(region_rows) for c in range(region_cols)}) < min(4, region_rows * region_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
