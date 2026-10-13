import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 22
MIN_GAP = 3
MAX_HORIZONTAL_PATHS = 3
MAX_VERTICAL_PATHS = 3
MAX_EXTRA_PATHS = 2
GENERATION_RETRIES = 400


def _find_crossings(grid: np.ndarray) -> list[tuple[int, int, tuple[tuple[int, int], tuple[int, int]]]]:
    """Return each crossing center and the two pixels that belong to the lower path."""
    h, w = grid.shape
    crossings = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center == BLACK:
                continue

            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])

            if BLACK in (up, down, left, right):
                continue
            if up != down or left != right or up == left:
                continue

            if center == up:
                lower_neighbors = ((r, c - 1), (r, c + 1))
            elif center == left:
                lower_neighbors = ((r - 1, c), (r + 1, c))
            else:
                continue

            crossings.append((r, c, lower_neighbors))
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split over-under crossings by keeping the upper path intact and marking the lower path breakpoints in grey.

    Valid crossings are axis-aligned plus-shapes whose vertical arm is one color,
    horizontal arm is a different color, and the center color indicates which arm is on top.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for _r, _c, lower_neighbors in _find_crossings(g):
        for nr, nc in lower_neighbors:
            out[nr, nc] = GREY

    return out


def _max_spaced_count(low: int, high: int, gap: int) -> int:
    if low > high:
        return 0
    return 1 + (high - low) // gap


def _sample_spaced_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    if low + (count - 1) * gap > high:
        return None

    positions = []
    start = low
    for idx in range(count):
        max_pos = high - (count - idx - 1) * gap
        pos = random.randint(start, max_pos)
        positions.append(pos)
        start = pos + gap
    return positions


def _paint_horizontal(grid: np.ndarray, row: int, color: int) -> None:
    grid[row, :] = color


def _paint_vertical(grid: np.ndarray, col: int, color: int) -> None:
    grid[:, col] = color


def _corner_cells(kind: str, h: int, w: int, rows: list[int], cols: list[int]) -> list[tuple[int, int]]:
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    if kind == "tl":
        r = random.randint(1, min_row - 2)
        c = random.randint(1, min_col - 2)
        cells = [(r, cc) for cc in range(0, c + 1)]
        cells.extend((rr, c) for rr in range(0, r))
        return cells

    if kind == "tr":
        r = random.randint(1, min_row - 2)
        c = random.randint(max_col + 2, w - 2)
        cells = [(rr, c) for rr in range(0, r + 1)]
        cells.extend((r, cc) for cc in range(c + 1, w))
        return cells

    if kind == "bl":
        r = random.randint(max_row + 2, h - 2)
        c = random.randint(1, min_col - 2)
        cells = [(r, cc) for cc in range(0, c + 1)]
        cells.extend((rr, c) for rr in range(r + 1, h))
        return cells

    if kind == "br":
        r = random.randint(max_row + 2, h - 2)
        c = random.randint(max_col + 2, w - 2)
        cells = [(rr, c) for rr in range(r, h)]
        cells.extend((r, cc) for cc in range(c + 1, w))
        return cells

    raise ValueError(f"unknown corner kind: {kind}")


def _feasible_corners(h: int, w: int, rows: list[int], cols: list[int]) -> list[str]:
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    corners = []
    if min_row >= 3 and min_col >= 3:
        corners.append("tl")
    if min_row >= 3 and max_col <= w - 4:
        corners.append("tr")
    if max_row <= h - 4 and min_col >= 3:
        corners.append("bl")
    if max_row <= h - 4 and max_col <= w - 4:
        corners.append("br")
    return corners


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert grid.ndim == 2
    assert not np.any(grid == GREY), "grey is reserved for the output"

    crossings = _find_crossings(grid)
    assert crossings, "generator must create at least one crossing"
    crossing_cells = {(r, c) for r, c, _ in crossings}

    h, w = grid.shape
    for r, c in np.argwhere(grid != BLACK):
        r = int(r)
        c = int(c)
        degree = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
                degree += 1

        if (r, c) in crossing_cells:
            assert degree == 4
        elif degree == 1:
            assert r in (0, h - 1) or c in (0, w - 1)
        else:
            assert degree == 2, f"unexpected local structure at {(r, c)}"

    out = main(grid)
    changed = np.argwhere(out != grid)
    assert len(changed) == 2 * len(crossings)
    assert int(np.count_nonzero(out == GREY)) == 2 * len(crossings)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)

    row_low, row_high = 2, h - 3
    col_low, col_high = 2, w - 3
    max_h = min(MAX_HORIZONTAL_PATHS, _max_spaced_count(row_low, row_high, MIN_GAP))
    max_v = min(MAX_VERTICAL_PATHS, _max_spaced_count(col_low, col_high, MIN_GAP))
    if max_h == 0 or max_v == 0:
        return None

    n_h = random.randint(1, max_h)
    n_v = random.randint(1, max_v)

    if n_h + n_v > len(PATH_COLORS):
        return None

    rows = _sample_spaced_positions(row_low, row_high, n_h, MIN_GAP)
    cols = _sample_spaced_positions(col_low, col_high, n_v, MIN_GAP)
    if rows is None or cols is None:
        return None

    colors = list(PATH_COLORS)
    random.shuffle(colors)
    h_colors = colors[:n_h]
    v_colors = colors[n_h : n_h + n_v]
    next_color = n_h + n_v

    grid = np.zeros((h, w), dtype=int)
    for row, color in zip(rows, h_colors):
        _paint_horizontal(grid, row, color)
    for col, color in zip(cols, v_colors):
        _paint_vertical(grid, col, color)

    top_is_horizontal = [random.random() < 0.5 for _ in range(n_h * n_v)]
    if len(top_is_horizontal) > 1 and (all(top_is_horizontal) or not any(top_is_horizontal)):
        flip_idx = random.randrange(len(top_is_horizontal))
        top_is_horizontal[flip_idx] = not top_is_horizontal[flip_idx]

    idx = 0
    for row, h_color in zip(rows, h_colors):
        for col, v_color in zip(cols, v_colors):
            grid[row, col] = h_color if top_is_horizontal[idx] else v_color
            idx += 1

    corner_kinds = _feasible_corners(h, w, rows, cols)
    random.shuffle(corner_kinds)
    max_extra = min(MAX_EXTRA_PATHS, len(corner_kinds), len(PATH_COLORS) - next_color)
    n_extra = random.randint(0, max_extra)

    for kind in corner_kinds[:n_extra]:
        color = colors[next_color]
        next_color += 1
        cells = _corner_cells(kind, h, w, rows, cols)
        for r, c in cells:
            if grid[r, c] != BLACK:
                return None
            grid[r, c] = color

    _validate_generated_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-path input")
