import numpy as np

from arc_utils import draw_line, get_bg
import random
from itertools import product


def _paint_cell(out: np.ndarray, bg: int, color: int, r: int, c: int) -> None:
    current = int(out[r, c])
    assert current in (bg, color)
    out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each top-edge color downward, turning horizontally only when it reaches its matching endpoint pixel.

    Input structure
    A single contiguous run of non-background pixels lies on the top edge; any other
    foreground pixel has a color from that run and acts as that color's endpoint.

    Rule
    For each top-edge source pixel, draw a vertical segment down its column to the
    row of the same-colored endpoint, then draw the horizontal segment to that
    endpoint. If the source color has no separate endpoint, draw the vertical
    segment straight down to the bottom edge.

    Key insight
    The top run gives the source column for each color, while each matching
    off-edge pixel gives the row and final column where that color's path turns.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    bg = get_bg(g)

    top_cols = np.flatnonzero(g[0] != bg)
    assert top_cols.size > 0
    assert np.all(np.diff(top_cols) == 1)

    sources = [(int(c), int(g[0, c])) for c in top_cols]
    source_colors = [color for _, color in sources]
    assert len(set(source_colors)) == len(source_colors)
    assert set(map(int, np.unique(g[g != bg]))) == set(source_colors)

    out = g.copy()
    for src_col, color in sources:
        color_coords = np.argwhere(g == color)
        endpoints = [
            (int(r), int(c))
            for r, c in color_coords
            if not (int(r) == 0 and int(c) == src_col)
        ]
        assert len(endpoints) <= 1

        if endpoints:
            end_row, end_col = endpoints[0]
            assert end_row > 0
        else:
            end_row, end_col = h - 1, src_col

        for r in range(end_row + 1):
            _paint_cell(out, bg, color, r, src_col)

        if endpoints:
            c0, c1 = sorted((src_col, end_col))
            for c in range(c0, c1 + 1):
                _paint_cell(out, bg, color, end_row, c)

    return out

# --------------- generator (o5x) ---------------

# Canvas/source placement.
GRID_SIZE_MIN, GRID_SIZE_MAX = 11, 15
SRC_COUNT = 3
SRC_CENTER_OFFSET = 2
SRC_START_JITTER = (-1, 0, 0, 1)
SRC_START_MIN_COL, SRC_START_RIGHT_PADDING = 2, 5

# Endpoints/path lengths.
ENDPOINT_ROW_MIN, ENDPOINT_ROW_BOTTOM_PADDING = 2, 2
PATH_LENGTH_MIN, PATH_LENGTH_MAX = 8, 16
ENDPOINT_COUNT_CHOICES = (2, 3)
LEFT_SIDE, RIGHT_SIDE = "L", "R"
SIDE_PATTERNS_BY_ENDPOINT_COUNT = {
    3: ((LEFT_SIDE, LEFT_SIDE, RIGHT_SIDE), (LEFT_SIDE, RIGHT_SIDE, RIGHT_SIDE)),
    2: ((None, RIGHT_SIDE, RIGHT_SIDE), (LEFT_SIDE, None, RIGHT_SIDE), (LEFT_SIDE, LEFT_SIDE, None)),
}

# Color/retry policy.
COLOR_VALUES = tuple(range(1, 10))
RETRY_LIMIT = 300


def _gen_wall_col(n: int, side: str) -> int:
    return 0 if side == LEFT_SIDE else n - 1


def _gen_path_cells(n: int, src_col: int, side: str | None, end_row: int | None) -> set[tuple[int, int]]:
    if side is None:
        return {(r, src_col) for r in range(n)}

    assert end_row is not None
    wall_col = _gen_wall_col(n, side)
    cells = {(r, src_col) for r in range(end_row + 1)}
    c0, c1 = sorted((src_col, wall_col))
    cells.update((end_row, c) for c in range(c0, c1 + 1))
    return cells


def _gen_endpoint_row_candidates(n: int, src_col: int, side: str) -> list[int]:
    wall_col = _gen_wall_col(n, side)
    dist = abs(src_col - wall_col)
    lo = max(ENDPOINT_ROW_MIN, PATH_LENGTH_MIN - 1 - dist)
    hi = min(n - ENDPOINT_ROW_BOTTOM_PADDING, PATH_LENGTH_MAX - 1 - dist)
    return list(range(lo, hi + 1))


def _gen_sides_cross_ok(sides: tuple[str | None, ...], rows: tuple[int | None, ...]) -> bool:
    for i, side in enumerate(sides):
        if side is None:
            continue
        crossed = range(i) if side == LEFT_SIDE else range(i + 1, SRC_COUNT)
        for j in crossed:
            if sides[j] is None or rows[j] is None or rows[i] is None:
                return False
            if rows[i] <= rows[j]:
                return False
    return True


def _gen_paths_disjoint(
    n: int,
    src_cols: tuple[int, int, int],
    sides: tuple[str | None, ...],
    rows: tuple[int | None, ...],
) -> bool:
    occupied = set()
    for src_col, side, row in zip(src_cols, sides, rows):
        cells = _gen_path_cells(n, src_col, side, row)
        if occupied & cells:
            return False
        occupied.update(cells)
    return True


def _gen_valid_rows(
    n: int,
    src_cols: tuple[int, int, int],
    sides: tuple[str | None, ...],
) -> list[tuple[int | None, ...]]:
    candidate_lists = []
    endpoint_indices = []
    for i, side in enumerate(sides):
        if side is None:
            continue
        rows = _gen_endpoint_row_candidates(n, src_cols[i], side)
        if not rows:
            return []
        endpoint_indices.append(i)
        candidate_lists.append(rows)

    valid = []
    for picked in product(*candidate_lists):
        rows = [None] * SRC_COUNT
        for i, row in zip(endpoint_indices, picked):
            rows[i] = row
        row_tuple = tuple(rows)
        if not _gen_sides_cross_ok(sides, row_tuple):
            continue
        if not _gen_paths_disjoint(n, src_cols, sides, row_tuple):
            continue
        valid.append(row_tuple)
    return valid


def _gen_build_pair(
    n: int,
    src_cols: tuple[int, int, int],
    colors: tuple[int, int, int],
    sides: tuple[str | None, ...],
    rows: tuple[int | None, ...],
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((n, n), dtype=int)
    for src_col, color, side, row in zip(src_cols, colors, sides, rows):
        if side is None:
            draw_line(out, 0, src_col, end_r=n - 1, end_c=src_col, color=color)
        else:
            assert row is not None
            wall_col = _gen_wall_col(n, side)
            draw_line(out, 0, src_col, end_r=row, end_c=src_col, color=color)
            draw_line(out, row, src_col, end_r=row, end_c=wall_col, color=color)

    inp = np.zeros((n, n), dtype=int)
    for src_col, color in zip(src_cols, colors):
        inp[0, src_col] = color
    for color, side, row in zip(colors, sides, rows):
        if side is None:
            continue
        wall_col = _gen_wall_col(n, side)
        assert row is not None
        inp[row, wall_col] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    base_start = n // 2 - SRC_CENTER_OFFSET
    start_col = min(
        max(base_start + random.choice(SRC_START_JITTER), SRC_START_MIN_COL),
        n - SRC_START_RIGHT_PADDING,
    )
    src_cols = tuple(start_col + i for i in range(SRC_COUNT))
    colors = tuple(random.sample(COLOR_VALUES, SRC_COUNT))

    endpoint_count = random.choice(ENDPOINT_COUNT_CHOICES)
    configs = []
    for sides in SIDE_PATTERNS_BY_ENDPOINT_COUNT[endpoint_count]:
        for rows in _gen_valid_rows(n, src_cols, sides):
            configs.append((sides, rows))
    if not configs:
        return None

    sides, rows = random.choice(configs)
    inp, out = _gen_build_pair(n, src_cols, colors, sides, rows)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
