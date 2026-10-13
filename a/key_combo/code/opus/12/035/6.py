from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LENGTH_TO_COLOR = {
    2: RED,
    3: GREEN,
    4: YELLOW,
    5: GREY,
}

VALID_LENGTHS = tuple(sorted(LENGTH_TO_COLOR))
BOX_PAIRS = [
    (11, 14),
    (14, 11),
    (11, 19),
    (19, 11),
    (17, 19),
    (19, 17),
]
LINE_COUNT_CHOICES = (2, 3)
PADDING_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def _has_adjacent_pair(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == BLUE) & (values[1:] == BLUE)))


def _allowed_lengths(span: int) -> list[int]:
    return [length for length in VALID_LENGTHS if (span + 1) % (length + 1) == 0]


def _gap_positions(span: int, length: int) -> list[int]:
    segments = (span + 1) // (length + 1)
    return [(i + 1) * length + i for i in range(segments - 1)]


def _effective_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if _has_adjacent_pair(grid[r])]


def _effective_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if _has_adjacent_pair(grid[:, c])]


def _adjacent_run_lengths_row(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    left = 0
    cc = c - 1
    while cc >= 0 and grid[r, cc] == BLUE:
        left += 1
        cc -= 1

    right = 0
    cc = c + 1
    while cc < grid.shape[1] and grid[r, cc] == BLUE:
        right += 1
        cc += 1
    return left, right


def _adjacent_run_lengths_col(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    up = 0
    rr = r - 1
    while rr >= 0 and grid[rr, c] == BLUE:
        up += 1
        rr -= 1

    down = 0
    rr = r + 1
    while rr < grid.shape[0] and grid[rr, c] == BLUE:
        down += 1
        rr += 1
    return up, down


def _collect_row_gaps(grid: np.ndarray, line_rows: list[int], line_cols: set[int]) -> list[tuple[int, int, int]]:
    gaps: list[tuple[int, int, int]] = []
    for r in line_rows:
        for c in range(1, grid.shape[1] - 1):
            if c in line_cols or grid[r, c] != BLACK:
                continue
            if grid[r, c - 1] != BLUE or grid[r, c + 1] != BLUE:
                continue
            left, right = _adjacent_run_lengths_row(grid, r, c)
            assert left == right, "row gap must be flanked by equal-length blue segments"
            assert left in LENGTH_TO_COLOR, "row gap segment length must be 2-5"
            gaps.append((r, c, left))
    return gaps


def _collect_col_gaps(grid: np.ndarray, line_rows: set[int], line_cols: list[int]) -> list[tuple[int, int, int]]:
    gaps: list[tuple[int, int, int]] = []
    for c in line_cols:
        for r in range(1, grid.shape[0] - 1):
            if r in line_rows or grid[r, c] != BLACK:
                continue
            if grid[r - 1, c] != BLUE or grid[r + 1, c] != BLUE:
                continue
            up, down = _adjacent_run_lengths_col(grid, r, c)
            assert up == down, "column gap must be flanked by equal-length blue segments"
            assert up in LENGTH_TO_COLOR, "column gap segment length must be 2-5"
            gaps.append((r, c, up))
    return gaps


def _find_gaps(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    line_rows = _effective_rows(grid)
    line_cols = _effective_cols(grid)
    assert line_rows, "expected at least one effective horizontal line"
    assert line_cols, "expected at least one effective vertical line"

    row_gaps = _collect_row_gaps(grid, line_rows, set(line_cols))
    col_gaps = _collect_col_gaps(grid, set(line_rows), line_cols)
    assert row_gaps or col_gaps, "expected at least one one-cell gap"
    return line_rows, line_cols, row_gaps, col_gaps


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each one-cell grid-line gap with the color encoded by the adjacent blue segment length.

    Input structure:
    - A black background contains an incomplete blue rectilinear grid made from horizontal and vertical lines.
    - Effective horizontal lines are exactly the rows containing a horizontally adjacent blue pair; effective vertical lines are exactly the columns containing a vertically adjacent blue pair.
    - Along each effective line, blue runs of length 2-5 are separated by one-cell black gaps, and every true gap lies on only one orientation rather than at a line intersection.

    Rule:
    - Keep all existing blue pixels and all non-gap black pixels unchanged.
    - Find every one-cell black gap on each effective row and column.
    - Measure the length L of either immediately adjacent blue run along that line; the two sides are equal in valid inputs.
    - Recolor the gap by the fixed code L=2->red(2), L=3->green(3), L=4->yellow(4), L=5->grey(5).

    Key insight:
    - The missing cells are detected by line-adjacency, but their fill color is not constant: it is determined by counting the length of the neighboring intact blue segment on that same row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE}, "input must contain only black and blue"

    _line_rows, _line_cols, row_gaps, col_gaps = _find_gaps(g)

    out = g.copy()
    for r, c, length in row_gaps:
        out[r, c] = LENGTH_TO_COLOR[length]
    for r, c, length in col_gaps:
        out[r, c] = LENGTH_TO_COLOR[length]
    return out


def _choose_line_lengths(
    row_allowed: list[int],
    row_count: int,
    col_allowed: list[int],
    col_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for row_lengths in itertools.product(row_allowed, repeat=row_count):
        for col_lengths in itertools.product(col_allowed, repeat=col_count):
            if set(row_lengths) | set(col_lengths) != set(VALID_LENGTHS):
                continue
            score = len(set(row_lengths)) + len(set(col_lengths))
            candidates.append((row_lengths, col_lengths, score))
    if not candidates:
        return None
    best_score = max(score for _rows, _cols, score in candidates)
    finalists = [(rows, cols) for rows, cols, score in candidates if score == best_score]
    return random.choice(finalists)


def _choose_marked_positions(span: int, count: int, forbidden: set[int]) -> list[int] | None:
    if count == 2:
        return [0, span - 1]

    middle_candidates = [
        pos
        for pos in range(2, span - 2)
        if pos not in forbidden and pos - 0 >= 2 and (span - 1) - pos >= 2
    ]
    if not middle_candidates:
        return None
    return [0, random.choice(middle_candidates), span - 1]


def _place_complete_grid(
    box_h: int,
    box_w: int,
    row_positions: list[int],
    row_lengths: tuple[int, ...],
    col_positions: list[int],
    col_lengths: tuple[int, ...],
) -> tuple[np.ndarray, set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    patch = np.zeros((box_h, box_w), dtype=int)
    for r in row_positions:
        patch[r, :] = BLUE
    for c in col_positions:
        patch[:, c] = BLUE

    row_gap_info: set[tuple[int, int, int]] = set()
    for r, length in zip(row_positions, row_lengths):
        for c in _gap_positions(box_w, length):
            assert c not in col_positions, "row gap cannot be placed on a vertical line intersection"
            patch[r, c] = BLACK
            row_gap_info.add((r, c, length))

    col_gap_info: set[tuple[int, int, int]] = set()
    for c, length in zip(col_positions, col_lengths):
        for r in _gap_positions(box_h, length):
            assert r not in row_positions, "column gap cannot be placed on a horizontal line intersection"
            patch[r, c] = BLACK
            col_gap_info.add((r, c, length))

    return patch, row_gap_info, col_gap_info


def _build_expected_output(
    grid: np.ndarray,
    row_gaps: set[tuple[int, int, int]],
    col_gaps: set[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c, length in row_gaps | col_gaps:
        out[r, c] = LENGTH_TO_COLOR[length]
    return out


def _validate_candidate(
    grid: np.ndarray,
    expected: np.ndarray,
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    row_positions: list[int],
    row_lengths: tuple[int, ...],
    col_positions: list[int],
    col_lengths: tuple[int, ...],
) -> bool:
    if set(map(int, np.unique(grid))) != {BLACK, BLUE}:
        return False

    want_rows = [top + r for r in row_positions]
    want_cols = [left + c for c in col_positions]
    line_rows, line_cols, row_gaps, col_gaps = _find_gaps(grid)
    if line_rows != want_rows or line_cols != want_cols:
        return False

    want_row_gaps = {
        (top + r, left + c, length)
        for r, length in zip(row_positions, row_lengths)
        for c in _gap_positions(box_w, length)
    }
    want_col_gaps = {
        (top + r, left + c, length)
        for c, length in zip(col_positions, col_lengths)
        for r in _gap_positions(box_h, length)
    }
    if set(row_gaps) != want_row_gaps or set(col_gaps) != want_col_gaps:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    box_h, box_w = random.choice(BOX_PAIRS)
    row_allowed = _allowed_lengths(box_w)
    col_allowed = _allowed_lengths(box_h)
    row_count = random.choice(LINE_COUNT_CHOICES)
    col_count = random.choice(LINE_COUNT_CHOICES)

    chosen = _choose_line_lengths(row_allowed, row_count, col_allowed, col_count)
    if chosen is None:
        return None
    row_lengths, col_lengths = chosen

    forbidden_cols = set().union(*[set(_gap_positions(box_w, length)) for length in row_lengths])
    forbidden_rows = set().union(*[set(_gap_positions(box_h, length)) for length in col_lengths])

    row_positions = _choose_marked_positions(box_h, row_count, forbidden_rows)
    col_positions = _choose_marked_positions(box_w, col_count, forbidden_cols)
    if row_positions is None or col_positions is None:
        return None

    top = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)
    height = top + box_h + bottom
    width = left + box_w + right
    if height > 30 or width > 30:
        return None

    patch, row_gap_info, col_gap_info = _place_complete_grid(
        box_h, box_w, row_positions, row_lengths, col_positions, col_lengths
    )

    grid = np.zeros((height, width), dtype=int)
    grid[top : top + box_h, left : left + box_w] = patch
    expected = _build_expected_output(
        grid,
        {(top + r, left + c, length) for r, c, length in row_gap_info},
        {(top + r, left + c, length) for r, c, length in col_gap_info},
    )

    if not _validate_candidate(
        grid,
        expected,
        top,
        left,
        box_h,
        box_w,
        row_positions,
        row_lengths,
        col_positions,
        col_lengths,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid line-length gap-encoding input")
