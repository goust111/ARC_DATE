"""ARC task 8: apply a propagated, signed row-displacement field."""

from __future__ import annotations

import random

import numpy as np


# Fixed ARC role colors stated by the puzzle.
BACKGROUND = 0
NEGATIVE = 2       # red bars move rows left
SEPARATOR = 5      # grey vertical separator
POSITIVE = 8       # teal bars move rows right
OBJECT_COLORS = (1, 3, 4, 6, 7, 9)


def _separator_column(grid: np.ndarray) -> int:
    """Return the unique full-height grey separator column."""
    columns = np.flatnonzero(np.all(grid == SEPARATOR, axis=0))
    assert columns.size == 1, "expected exactly one full grey separator column"
    sep = int(columns[0])
    assert 1 <= sep < grid.shape[1] - 1, "separator must have regions on both sides"
    return sep


def _row_displacements(grid: np.ndarray, sep: int, bg: int) -> np.ndarray:
    """Decode bars and propagate the most recent signed reading downward."""
    height = int(grid.shape[0])
    shifts = np.zeros(height, dtype=int)
    current = 0
    for r in range(height):
        row = grid[r, :sep]
        occupied = np.flatnonzero(row != bg)
        if occupied.size:
            length = int(occupied.size)
            assert np.array_equal(occupied, np.arange(length)), (
                "each field bar must start at the left edge and be contiguous"
            )
            color = int(row[0])
            assert color in (POSITIVE, NEGATIVE), "field bars must be teal or red"
            assert np.all(row[:length] == color), "a field bar must be monochromatic"
            current = length if color == POSITIVE else -length
        shifts[r] = current
    return shifts


def main(grid: np.ndarray) -> np.ndarray:
    """Shear the right-hand object with the signed row field encoded in the left column.

    Input structure:
    - A unique full-height grey vertical line separates a narrow left field panel
      from a wider right region on a dominant background.
    - Each nonempty panel row is one left-aligned teal or red horizontal bar;
      the right region contains one solid monochrome foreground object.

    Rule:
    - A bar's length is its horizontal displacement magnitude: teal means right
      and red means left.  Empty rows inherit the nearest bar above, while rows
      above the first bar have displacement zero.
    - Move every object pixel in each right-region row by that row's displacement,
      clearing its former cells.  Preserve the field panel, separator, and all
      other right-region cells.

    Key insight:
    - The colored bars are a propagated signed displacement field, not independent
      row statistics: a blank row keeps the previous reading and therefore still
      participates in the shear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 2 <= height <= 30 and 3 <= width <= 30, "grid must fit ARC bounds"

    sep = _separator_column(g)
    right = g[:, sep + 1 :]

    # The broad right region makes its unique dominant value the visible canvas
    # background; this avoids assigning any ordering meaning to color ids.
    values, counts = np.unique(right, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg not in (SEPARATOR, POSITIVE, NEGATIVE), "role colors must be distinct"
    shifts = _row_displacements(g, sep, bg)

    object_colors = set(map(int, np.unique(right))) - {bg}
    assert len(object_colors) == 1, "expected one monochrome right-region object"
    object_color = object_colors.pop()
    object_mask = right == object_color
    assert object_mask.any(), "the right-region object must be nonempty"

    out = g.copy()
    right_out = out[:, sep + 1 :]
    right_width = int(right.shape[1])
    for r in range(height):
        cols = np.flatnonzero(object_mask[r])
        if cols.size == 0:
            continue
        moved = cols + int(shifts[r])
        assert int(moved.min()) >= 0 and int(moved.max()) < right_width, (
            "shifted object row must remain entirely inside the right region"
        )
        right_out[r, cols] = bg
        right_out[r, moved] = object_color
    return out


def _sample_field(object_top: int, object_height: int, left_width: int) -> list[tuple[int, int]]:
    """Choose bars whose zero, direct, inherited, positive, and negative cases all matter."""
    # The first object row is above the first bar (zero); every bar then has at
    # least one blank object row below it, making propagation observable.
    first = object_top + random.choice((1, 2))
    latest_second = object_top + object_height - 2
    second = random.randint(first + 2, latest_second)

    signs = [1, -1]
    random.shuffle(signs)
    bars = [
        (first, signs[0] * random.randint(1, left_width)),
        (second, signs[1] * random.randint(1, left_width)),
    ]

    # Often add a third reading of either sign.  It must also affect a following
    # blank object row, so it cannot be placed on the last object row.
    candidates = list(range(first + 2, second)) + list(
        range(second + 2, object_top + object_height - 1)
    )
    if candidates and random.random() < 0.5:
        row = random.choice(candidates)
        bars.append((row, random.choice((-1, 1)) * random.randint(1, left_width)))
    bars.sort()
    return bars


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    """Check puzzle semantics and reject shortcut-friendly generated instances."""
    try:
        sep = _separator_column(grid)
        shifts = _row_displacements(grid, sep, BACKGROUND)
        right_mask = grid[:, sep + 1 :] != BACKGROUND
        object_rows = np.flatnonzero(np.any(right_mask, axis=1))
        bar_rows = np.flatnonzero(np.any(grid[:, :sep] != BACKGROUND, axis=1))
        if object_rows.size < 6 or bar_rows.size < 2:
            return False
        if set(np.sign(shifts[object_rows])) != {-1, 0, 1}:
            return False

        # At least one nonzero inherited row crosses the object, defeating the
        # shortcut "blank means no movement".
        inherited = [
            int(r) for r in object_rows
            if shifts[r] != 0 and np.all(grid[r, :sep] == BACKGROUND)
        ]
        if not inherited:
            return False

        # Every visible bar must directly control an object row.  Thus none is
        # decorative, and both signed branches necessarily alter the result.
        if not all(right_mask[int(r)].any() for r in bar_rows):
            return False
        if not any(shifts[r] > 0 for r in bar_rows) or not any(shifts[r] < 0 for r in bar_rows):
            return False

        actual = main(grid)
        if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
            return False

        # Counterfactual oracle: treating blank object rows as zero must fail.
        wrong = grid.copy()
        wrong_right = wrong[:, sep + 1 :]
        source_right = grid[:, sep + 1 :]
        object_color = next(iter(set(map(int, np.unique(source_right))) - {BACKGROUND}))
        for r in bar_rows:
            cols = np.flatnonzero(source_right[int(r)] == object_color)
            wrong_right[int(r), cols] = BACKGROUND
            wrong_right[int(r), cols + shifts[int(r)]] = object_color
        if np.array_equal(wrong, expected):
            return False
        return True
    except (AssertionError, IndexError, StopIteration):
        return False


def generate_input() -> np.ndarray:
    """Generate diverse, nondegenerate displacement-field instances."""
    for _ in range(300):
        height = random.randint(10, 22)
        left_width = random.randint(3, 6)
        right_width = random.randint(max(12, 2 * left_width + 3), min(20, 29 - left_width))
        width = left_width + 1 + right_width

        object_height = random.randint(6, min(11, height - 1))
        object_top = random.randint(0, height - object_height)
        object_width = random.randint(
            3, min(8, right_width // 2, right_width - 2 * left_width)
        )
        bars = _sample_field(object_top, object_height, left_width)

        min_shift = min([0] + [shift for _, shift in bars])
        max_shift = max([0] + [shift for _, shift in bars])
        col_lo = -min_shift
        col_hi = right_width - object_width - max_shift
        if col_lo > col_hi:
            continue
        object_left = random.randint(col_lo, col_hi)
        object_color = random.choice(OBJECT_COLORS)

        grid = np.full((height, width), BACKGROUND, dtype=int)
        grid[:, left_width] = SEPARATOR
        grid[
            object_top : object_top + object_height,
            left_width + 1 + object_left : left_width + 1 + object_left + object_width,
        ] = object_color
        for r, shift in bars:
            color = POSITIVE if shift > 0 else NEGATIVE
            grid[r, : abs(shift)] = color

        # Construct an independent expected result from the sampled semantic
        # parameters, rather than calling the decoder used by main.
        expected = grid.copy()
        current = 0
        field_by_row = {r: shift for r, shift in bars}
        right_start = left_width + 1
        for r in range(height):
            if r in field_by_row:
                current = field_by_row[r]
            if object_top <= r < object_top + object_height:
                old0 = right_start + object_left
                expected[r, old0 : old0 + object_width] = BACKGROUND
                new0 = old0 + current
                expected[r, new0 : new0 + object_width] = object_color

        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid displacement-field input")
