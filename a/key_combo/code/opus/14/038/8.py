import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PATTERN_COLORS = (BLUE, RED, GREEN, PINK, TEAL, MAROON)
MAX_GENERATION_ATTEMPTS = 500


def _paint_rect(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    rh = random.randint(1, max(1, h // 2 + 1))
    rw = random.randint(1, max(1, w // 2 + 1))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    grid[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_bar(grid: np.ndarray, color: int, horizontal: bool) -> None:
    h, w = grid.shape
    if horizontal:
        r = random.randrange(h)
        c0 = random.randint(0, w - 2)
        c1 = random.randint(c0 + 1, w - 1)
        grid[r, c0 : c1 + 1] = color
    else:
        c = random.randrange(w)
        r0 = random.randint(0, h - 2)
        r1 = random.randint(r0 + 1, h - 1)
        grid[r0 : r1 + 1, c] = color


def _paint_lshape(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    arm_h = random.randint(2, h)
    arm_w = random.randint(2, w)
    r0 = random.randint(0, h - arm_h)
    c0 = random.randint(0, w - arm_w)
    orientation = random.choice(("tl", "tr", "bl", "br"))
    if orientation == "tl":
        grid[r0 : r0 + arm_h, c0] = color
        grid[r0, c0 : c0 + arm_w] = color
    elif orientation == "tr":
        grid[r0 : r0 + arm_h, c0 + arm_w - 1] = color
        grid[r0, c0 : c0 + arm_w] = color
    elif orientation == "bl":
        grid[r0 : r0 + arm_h, c0] = color
        grid[r0 + arm_h - 1, c0 : c0 + arm_w] = color
    else:
        grid[r0 : r0 + arm_h, c0 + arm_w - 1] = color
        grid[r0 + arm_h - 1, c0 : c0 + arm_w] = color


def _paint_diag(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    r0 = random.randrange(h)
    c0 = random.randrange(w)
    dr = random.choice((-1, 1))
    dc = random.choice((-1, 1))
    length = random.randint(max(2, min(h, w) // 2), min(h, w))
    r, c = r0, c0
    for _ in range(length):
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color
        r += dr
        c += dc


def _paint_dots(grid: np.ndarray, colors: Iterable[int]) -> None:
    h, w = grid.shape
    for color in colors:
        r = random.randrange(h)
        c = random.randrange(w)
        grid[r, c] = color


def _rows_of_rotation(pattern: np.ndarray) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    for k in range(4):
        rotated = np.rot90(pattern, k)
        rows.extend(tuple(int(v) for v in row) for row in rotated)
    return rows


def _pattern_is_informative(pattern: np.ndarray) -> bool:
    h, w = pattern.shape
    if h != w:
        return False
    non_black = int(np.count_nonzero(pattern))
    if non_black < h + 3 or non_black > h * h - 2:
        return False
    used_colors = set(int(v) for v in np.unique(pattern)) - {BLACK}
    if len(used_colors) < 3:
        return False
    for row in pattern:
        if len(set(int(v) for v in row)) < 2:
            return False
        if not np.any(row != BLACK):
            return False
    for col in pattern.T:
        if len(set(int(v) for v in col)) < 2:
            return False
        if not np.any(col != BLACK):
            return False
    rotated_rows = _rows_of_rotation(pattern)
    if len(rotated_rows) != len(set(rotated_rows)):
        return False
    if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in range(1, 4)):
        return False
    return True


def _generate_pattern(size: int) -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pattern = np.zeros((size, size), dtype=int)
        palette = tuple(random.sample(PATTERN_COLORS, k=random.randint(3, 5)))
        motif_count = random.randint(5, 8)
        for _ in range(motif_count):
            color = random.choice(palette)
            kind = random.choice(("rect", "hbar", "vbar", "lshape", "diag"))
            if kind == "rect":
                _paint_rect(pattern, color)
            elif kind == "hbar":
                _paint_bar(pattern, color, horizontal=True)
            elif kind == "vbar":
                _paint_bar(pattern, color, horizontal=False)
            elif kind == "lshape":
                _paint_lshape(pattern, color)
            else:
                _paint_diag(pattern, color)
        _paint_dots(pattern, palette)
        for _ in range(random.randint(1, max(2, size // 2))):
            r = random.randrange(size)
            c = random.randrange(size)
            pattern[r, c] = random.choice((BLACK,) + palette)
        if _pattern_is_informative(pattern):
            return pattern
    raise RuntimeError("failed to generate a non-degenerate square pattern")


def _find_divider_column(grid: np.ndarray) -> int:
    full_grey = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(full_grey) == 1, "expected exactly one full-height grey divider"
    return int(full_grey[0])


def _count_full_width_bars(panel: np.ndarray, color: int) -> int:
    count = 0
    for row in panel:
        values = set(int(v) for v in row)
        assert len(values) == 1, "each right-panel row must be uniform"
        value = next(iter(values))
        assert value in {BLACK, YELLOW, ORANGE}, "right panel may contain only black, yellow, or orange"
        if value == color:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the left square pattern by a bar-coded amount and extract one row.

    Input structure:
    - The grid contains a single full-height grey divider column that splits the canvas.
    - Left of the divider is an HxH square pattern on black, with H equal to the grid height.
    - Right of the divider is a stack of full-width horizontal rows, each either black, yellow, or orange.

    Rule:
    - Count yellow rows Y and orange rows O in the right panel.
    - Rotate the left HxH pattern counterclockwise by Y mod 4 quarter-turns.
    - Return only row O of the rotated pattern as a 1xH output grid.

    Key insight:
    - The right-side bars are an external two-parameter code: yellow chooses the rotation, and orange chooses which row survives after that rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    divider = _find_divider_column(g)
    assert divider == h, "left side must be an HxH square with H equal to grid height"
    assert divider + 1 < w, "right panel must exist"

    pattern = g[:, :divider]
    panel = g[:, divider + 1 :]

    yellow_count = _count_full_width_bars(panel, YELLOW)
    orange_count = _count_full_width_bars(panel, ORANGE)

    assert yellow_count <= 3, "yellow bar count must stay in the 0..3 range"
    assert orange_count < h, "orange bar count must be a valid row index"

    rotated = np.rot90(pattern, yellow_count % 4)
    return rotated[orange_count : orange_count + 1, :].copy()


def _build_grid(pattern: np.ndarray, yellow_count: int, orange_count: int) -> np.ndarray:
    size = pattern.shape[0]
    grid = np.zeros((size, 2 * size + 1), dtype=int)
    grid[:, :size] = pattern
    grid[:, size] = GREY

    rows = list(range(size))
    chosen_rows = random.sample(rows, yellow_count + orange_count)
    yellow_rows = set(chosen_rows[:yellow_count])
    orange_rows = set(chosen_rows[yellow_count:])
    for r in yellow_rows:
        grid[r, size + 1 :] = YELLOW
    for r in orange_rows:
        grid[r, size + 1 :] = ORANGE
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    h, _ = grid.shape
    divider = _find_divider_column(grid)
    pattern = grid[:, :divider]
    panel = grid[:, divider + 1 :]
    yellow_count = _count_full_width_bars(panel, YELLOW)
    orange_count = _count_full_width_bars(panel, ORANGE)
    expected = np.rot90(pattern, yellow_count % 4)[orange_count : orange_count + 1, :]
    if not np.array_equal(out, expected):
        return False

    rotated_rows = _rows_of_rotation(pattern)
    if len(rotated_rows) != len(set(rotated_rows)):
        return False

    baseline = tuple(int(v) for v in out[0])
    alternatives = 0
    for turns in range(4):
        rotated = np.rot90(pattern, turns)
        for row_idx in range(h):
            candidate = tuple(int(v) for v in rotated[row_idx])
            if candidate == baseline:
                if turns != yellow_count % 4 or row_idx != orange_count:
                    return False
            alternatives += 1
    if alternatives != 4 * h:
        return False
    return _pattern_is_informative(pattern)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = random.randint(5, 10)
        yellow_count = random.randint(0, 3)
        max_orange = min(size - 1, size - yellow_count)
        if max_orange < 0:
            continue
        orange_count = random.randint(0, max_orange)

        pattern = _generate_pattern(size)
        grid = _build_grid(pattern, yellow_count, orange_count)

        out = main(grid)
        if len(set(int(v) for v in out[0])) < 2:
            continue
        if not _validate_semantics(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
