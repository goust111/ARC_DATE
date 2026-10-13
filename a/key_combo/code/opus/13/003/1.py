from __future__ import annotations

import random

import numpy as np


BG = 0
SHELF = 5
COLORS = (1, 2, 3)
MAX_ATTEMPTS = 400


def _settled_row(shelf_mask: np.ndarray, start_row: int, col: int) -> int:
    """Return the resting row for one marker falling straight down in its column."""
    h = int(shelf_mask.shape[0])
    below = np.flatnonzero(shelf_mask[int(start_row) + 1 :, int(col)])
    if below.size == 0:
        return h - 1
    return int(start_row + below[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered color pairs onto shelves, then fill the rectangles defined by their landed corners.

    Input structure:
    - Black background with 3-4 grey horizontal shelf segments on distinct rows.
    - Three foreground colors each appear exactly twice as isolated single pixels.
    - Each foreground pixel has a clear vertical drop path to either a shelf in its column or the grid bottom.

    Rule:
    - Move every foreground pixel straight down until the next cell below is a grey shelf cell or the bottom edge.
    - For each color, use its two landed positions as opposite corners of an axis-aligned bounding box.
    - Fill that entire box with the color, then restore all grey shelf cells on top of the fill.

    Key insight:
    - The original marker positions are only launch points; the output geometry is determined by the post-gravity landing rows, not the input rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    shelf_mask = g == SHELF
    out = np.full(g.shape, BG, dtype=int)

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, SHELF)):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        landed = [(_settled_row(shelf_mask, int(r), int(c)), int(c)) for r, c in coords]
        rows = [p[0] for p in landed]
        cols = [p[1] for p in landed]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    out[shelf_mask] = SHELF
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_spaced_rows(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    choices = list(range(int(lo), int(hi) + 1))
    for _ in range(80):
        rows = sorted(random.sample(choices, count))
        if all(rows[i + 1] - rows[i] >= int(min_gap) for i in range(len(rows) - 1)):
            return rows
    return None


def _support_segment(left: int, right: int, side: str) -> tuple[int, int]:
    """
    Build one shelf segment inside a color window while excluding the opposite marker column.
    """
    if side == "left":
        end = random.randint(left + 2, right - 1)
        return left, end
    start = random.randint(left + 1, right - 2)
    return start, right


def _layout_windows() -> tuple[int, list[tuple[int, int]]] | None:
    widths = [random.randint(4, 5) for _ in range(3)]
    gaps = [random.randint(1, 2) for _ in range(2)]
    needed = sum(widths) + sum(gaps)
    if needed > 20:
        return None

    w = random.randint(max(14, needed), 20)
    left_margin = random.randint(0, w - needed)

    windows: list[tuple[int, int]] = []
    cur = left_margin
    for i, width in enumerate(widths):
        windows.append((cur, cur + width - 1))
        cur += width
        if i < 2:
            cur += gaps[i]
    return w, windows


def _build_attempt() -> np.ndarray | None:
    layout = _layout_windows()
    if layout is None:
        return None
    w, windows = layout

    num_shelves = random.choice((3, 4))
    h = random.randint(13, 18)
    shelf_rows = _sample_spaced_rows(num_shelves, 3, h - 2, 2)
    if shelf_rows is None:
        return None

    base_rows = shelf_rows[:3]
    extra_row = shelf_rows[3] if num_shelves == 4 else None
    random.shuffle(base_rows)

    grid = np.zeros((h, w), dtype=int)
    support_rows_by_col: dict[int, int] = {}

    extra_color = random.randrange(3) if extra_row is not None else None
    base_sides = [random.choice(("left", "right")) for _ in COLORS]

    for idx, (left, right) in enumerate(windows):
        base_side = base_sides[idx]
        a, b = _support_segment(left, right, base_side)
        row = base_rows[idx]
        grid[row, a : b + 1] = SHELF
        support_rows_by_col[left if base_side == "left" else right] = row

    if extra_row is not None:
        extra_left, extra_right = windows[int(extra_color)]
        other_side = "right" if base_sides[int(extra_color)] == "left" else "left"
        a, b = _support_segment(extra_left, extra_right, other_side)
        grid[int(extra_row), a : b + 1] = SHELF
        support_rows_by_col[extra_left if other_side == "left" else extra_right] = int(extra_row)

    rects: list[tuple[int, int, int, int]] = []

    for idx, color in enumerate(COLORS):
        left, right = windows[idx]
        left_land = support_rows_by_col.get(left, h) - 1 if left in support_rows_by_col else h - 1
        right_land = support_rows_by_col.get(right, h) - 1 if right in support_rows_by_col else h - 1
        if left_land == right_land:
            return None

        rect = (min(left_land, right_land), left, max(left_land, right_land), right)
        if any(_boxes_overlap(rect, prev) for prev in rects):
            return None
        rects.append(rect)

        left_start_max = left_land if left in support_rows_by_col else h - 2
        right_start_max = right_land if right in support_rows_by_col else h - 2
        if left_start_max <= 0 or right_start_max <= 0:
            return None

        left_start = random.randint(0, left_start_max - 1)
        right_start = random.randint(0, right_start_max - 1)
        grid[left_start, left] = color
        grid[right_start, right] = color

    if not _valid_input(grid, num_shelves):
        return None
    return grid


def _valid_input(grid: np.ndarray, expected_shelves: int) -> bool:
    shelf_mask = grid == SHELF
    shelf_rows = np.flatnonzero(shelf_mask.any(axis=1))
    if len(shelf_rows) != int(expected_shelves):
        return False

    for r in shelf_rows:
        cols = np.flatnonzero(shelf_mask[int(r)])
        if cols.size < 3 or cols.size > 6:
            return False
        if not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            return False

    seen_rects: list[tuple[int, int, int, int]] = []
    for color in COLORS:
        coords = np.argwhere(grid == color)
        if coords.shape != (2, 2):
            return False
        if int(coords[0, 1]) == int(coords[1, 1]):
            return False
        landed = [(_settled_row(shelf_mask, int(r), int(c)), int(c)) for r, c in coords]
        rows = [p[0] for p in landed]
        cols = [p[1] for p in landed]
        if len(set(rows)) != 2:
            return False
        rect = (min(rows), min(cols), max(rows), max(cols))
        if any(_boxes_overlap(rect, prev) for prev in seen_rects):
            return False
        seen_rects.append(rect)

    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-rectangle input")
