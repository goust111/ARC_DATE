import random

import numpy as np


BLUE = 1
GREEN = 3
BLACK = 0

MIN_SEGMENTS = 2
MAX_SEGMENTS = 3
MIN_SEGMENT_WIDTH = 2
MIN_BAR_HEIGHT = 3
MAX_BAR_HEIGHT = 6
MIN_WIDTH = 9
MAX_WIDTH = 16
MAX_HEIGHT = 14
MAX_ATTEMPTS = 300


def _runs_of_value(row: np.ndarray, value: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, cell in enumerate(row.tolist()):
        if int(cell) == value:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _parse_mask(top_row: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    segments = _runs_of_value(top_row, BLUE)
    assert MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS, "expected 2 or 3 blue mask segments"
    assert all(c1 - c0 + 1 >= MIN_SEGMENT_WIDTH for c0, c1 in segments), "mask segments must be at least 2 cells wide"

    gap_cols: list[int] = []
    for idx in range(len(segments) - 1):
        prev_end = segments[idx][1]
        next_start = segments[idx + 1][0]
        gap = list(range(prev_end + 1, next_start))
        assert len(gap) == 1, "each internal gap must be exactly one column wide"
        gap_col = gap[0]
        assert 0 < gap_col < len(top_row) - 1, "gap columns cannot lie on the left or right border"
        gap_cols.append(gap_col)
    assert len(gap_cols) in (1, 2), "expected one or two gap columns"
    assert set(map(int, np.unique(top_row))).issubset({BLACK, BLUE}), "top mask must use only black and blue"
    return segments, gap_cols


def _read_bar_height(grid: np.ndarray) -> int:
    nonzero_rows = np.flatnonzero(grid[1:, 0] == GREEN)
    assert nonzero_rows.size > 0, "expected a green marker bar below the top row"
    height = int(nonzero_rows[-1]) + 1
    assert MIN_BAR_HEIGHT <= height <= MAX_BAR_HEIGHT, "marker bar height out of range"
    assert np.all(grid[1 : height + 1, 0] == GREEN), "marker bar must be contiguous from row 1"
    assert np.all(grid[height + 1 :, 0] == BLACK), "column 0 below the marker bar must be background"
    return height


def _validate_input(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 2 <= h <= 30 and 2 <= w <= 30, "grid must respect ARC size bounds"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, GREEN}), "input must use only black, blue, and green"

    segments, gap_cols = _parse_mask(g[0])
    bar_height = _read_bar_height(g)
    assert bar_height + 1 <= h, "marker bar must fit in the grid"

    lower = g[1:, :]
    allowed = np.zeros_like(lower, dtype=bool)
    allowed[:bar_height, 0] = True
    assert np.all(lower[~allowed] == BLACK), "all lower foreground must belong to the first-column marker bar"
    assert np.all(lower[allowed] == GREEN), "the lower marker bar must be entirely green"
    return segments, gap_cols, bar_height


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the mask-defined column spans with checkerboards and mark each single-column gap with a green bar.

    Input structure:
    - The top row is a blue-and-black mask with 2 or 3 contiguous blue segments.
    - Consecutive blue segments are separated by single black gap columns, and the only lower foreground is a green bar in column 0.
    - The green bar starts at row 1, has height 3 to 6, and all other cells are black background.

    Rule:
    - Read the blue segments from the top-row mask and the green bar height from column 0.
    - Clear the canvas to black, then keep the original top row unchanged.
    - For each blue segment, fill rows 1..H in that column span with a blue checkerboard starting at the segment's upper-left cell; for each mask gap column, draw a solid green bar over rows 1..H; finally restore the original first-column marker bar.

    Key insight:
    - The top mask encodes two separate targets at once: blue runs define checkerboard rectangles, while the single black separators become vertical green bars over the row range given by the first-column marker height.
    """
    g = np.asarray(grid, dtype=int)
    segments, gap_cols, bar_height = _validate_input(g)

    out = np.full_like(g, BLACK)
    out[0] = g[0]

    for c0, c1 in segments:
        for r in range(1, bar_height + 1):
            for c in range(c0, c1 + 1):
                if (r - 1 + c - c0) % 2 == 0:
                    out[r, c] = BLUE

    for c in gap_cols:
        out[1 : bar_height + 1, c] = GREEN

    out[1 : bar_height + 1, 0] = GREEN
    return out


def _random_partition(total: int, buckets: int) -> list[int]:
    parts = [0] * buckets
    for _ in range(total):
        parts[random.randrange(buckets)] += 1
    return parts


def _build_mask(width: int, num_segments: int, touch_left: bool, touch_right: bool) -> np.ndarray:
    base_left = 0 if touch_left else 1
    base_right = 0 if touch_right else 1
    gap_count = num_segments - 1
    min_used = base_left + base_right + gap_count + num_segments * MIN_SEGMENT_WIDTH
    assert min_used <= width

    extra = width - min_used
    bucket_names = [f"segment_{idx}" for idx in range(num_segments)]
    if not touch_left:
        bucket_names.append("left_margin")
    if not touch_right:
        bucket_names.append("right_margin")
    extras = _random_partition(extra, len(bucket_names))
    extra_by_name = dict(zip(bucket_names, extras))

    leading = base_left + extra_by_name.get("left_margin", 0)
    trailing = base_right + extra_by_name.get("right_margin", 0)
    widths = [MIN_SEGMENT_WIDTH + extra_by_name[f"segment_{idx}"] for idx in range(num_segments)]

    top_row = np.zeros(width, dtype=int)
    col = leading
    for idx, seg_width in enumerate(widths):
        top_row[col : col + seg_width] = BLUE
        col += seg_width
        if idx < num_segments - 1:
            col += 1
    assert col + trailing == width
    return top_row


def _gen_attempt() -> np.ndarray:
    num_segments = random.choice([2, 3])
    touch_left = random.random() < 0.5
    touch_right = random.random() < 0.5
    min_width = (
        (0 if touch_left else 1)
        + (0 if touch_right else 1)
        + (num_segments - 1)
        + num_segments * MIN_SEGMENT_WIDTH
    )
    width = random.randint(max(MIN_WIDTH, min_width), min(MAX_WIDTH, min_width + 6))
    top_row = _build_mask(width, num_segments, touch_left, touch_right)

    bar_height = random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT)
    extra_rows = random.randint(0, max(0, MAX_HEIGHT - (bar_height + 1)))
    height = bar_height + 1 + extra_rows

    grid = np.zeros((height, width), dtype=int)
    grid[0] = top_row
    grid[1 : bar_height + 1, 0] = GREEN

    try:
        segments, gap_cols, parsed_height = _validate_input(grid)
    except AssertionError:
        return None

    if parsed_height != bar_height:
        return None
    if len(segments) != num_segments or len(gap_cols) != num_segments - 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out[1 : bar_height + 1] == BLUE) == 0:
        return None
    if any(np.count_nonzero(out[1 : bar_height + 1, c] == GREEN) != bar_height for c in gap_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
