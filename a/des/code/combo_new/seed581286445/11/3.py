from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
REPLACEMENT_COLORS = (5, 6, 7, 8, 9)

INNER_HEIGHT_RANGE = (6, 11)
INNER_WIDTH_RANGE = (6, 11)
EXTRA_SEGMENT_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 400


def _apply_transform(
    inner: np.ndarray,
    flip_cols: np.ndarray,
    flip_rows: np.ndarray,
    row_replacements: np.ndarray,
    col_replacements: np.ndarray,
) -> np.ndarray:
    out = np.array(inner, dtype=int, copy=True)

    for c in np.flatnonzero(flip_cols):
        out[:, c] = out[::-1, c]

    for r in np.flatnonzero(flip_rows):
        out[r, :] = out[r, ::-1]

    for r, color in enumerate(row_replacements):
        color = int(color)
        if color != BLACK:
            mask = out[r, :] == GREEN
            out[r, mask] = color

    for c, color in enumerate(col_replacements):
        color = int(color)
        if color != BLACK:
            mask = out[:, c] == YELLOW
            out[mask, c] = color

    return out


def _invert_flips(inner_after_flips: np.ndarray, flip_cols: np.ndarray, flip_rows: np.ndarray) -> np.ndarray:
    raw = np.array(inner_after_flips, dtype=int, copy=True)

    for r in np.flatnonzero(flip_rows):
        raw[r, :] = raw[r, ::-1]

    for c in np.flatnonzero(flip_cols):
        raw[:, c] = raw[::-1, c]

    return raw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode border instruction strips to flip selected rows and columns, then recolor selected greens and yellows.

    Input structure:
    - The background is black.
    - The outermost ring is an instruction frame: the top and bottom edges are one-pixel rows, and the left and right
      edges are one-pixel columns.
    - Top-edge non-corner cells use red for "flip this interior column vertically" and black for "leave it unchanged".
    - Left-edge non-corner cells use blue for "flip this interior row horizontally" and black for "leave it unchanged".
    - Right-edge non-corner cells recolor that interior row's green pixels to the shown non-black color.
    - Bottom-edge non-corner cells recolor that interior column's yellow pixels to the shown non-black color.
    - The interior region contains only black, green, and yellow pixels.

    Rule:
    - Remove the outer frame conceptually and work on the interior rectangle.
    - First, vertically reverse every interior column whose top instruction cell is red.
    - Second, horizontally reverse every interior row whose left instruction cell is blue.
    - Third, for each row with a non-black right instruction, replace all green pixels in that row with the instruction color.
    - Fourth, for each column with a non-black bottom instruction, replace all yellow pixels in that column with the
      instruction color.
    - Output only the transformed interior rectangle, without the instruction frame.

    Key insight:
    - Each edge of the frame controls a different operation type, and the flips must happen before recoloring because
      the flips change which rows contain green pixels and which columns contain yellow pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    assert int(g[0, 0]) == BLACK
    assert int(g[0, -1]) == BLACK
    assert int(g[-1, 0]) == BLACK
    assert int(g[-1, -1]) == BLACK

    top = g[0, 1:-1]
    bottom = g[-1, 1:-1]
    left = g[1:-1, 0]
    right = g[1:-1, -1]
    inner = g[1:-1, 1:-1]

    assert inner.shape == (h - 2, w - 2)
    assert np.all((top == BLACK) | (top == RED))
    assert np.all((left == BLACK) | (left == BLUE))
    assert np.all((inner == BLACK) | (inner == GREEN) | (inner == YELLOW))

    return _apply_transform(inner, top == RED, left == BLUE, right, bottom)


def _sample_flip_mask(length: int) -> np.ndarray:
    count = random.randint(1, length - 2)
    mask = np.zeros(length, dtype=bool)
    mask[random.sample(range(length), count)] = True
    return mask


def _sample_replacements(length: int, guaranteed_idx: int, forbidden_idx: int) -> np.ndarray:
    colors = np.full(length, BLACK, dtype=int)
    colors[guaranteed_idx] = random.choice(REPLACEMENT_COLORS)

    extras = [idx for idx in range(length) if idx not in (guaranteed_idx, forbidden_idx)]
    extra_count = random.randint(0, len(extras))
    for idx in random.sample(extras, extra_count):
        colors[idx] = random.choice(REPLACEMENT_COLORS)
    return colors


def _noncenter_indices(length: int) -> list[int]:
    return [idx for idx in range(length) if idx != length - 1 - idx]


def _pick_empty_in_row(grid: np.ndarray, row: int) -> int | None:
    choices = [c for c in range(grid.shape[1]) if int(grid[row, c]) == BLACK]
    if not choices:
        return None
    return random.choice(choices)


def _pick_empty_in_col(grid: np.ndarray, col: int) -> int | None:
    choices = [r for r in range(grid.shape[0]) if int(grid[r, col]) == BLACK]
    if not choices:
        return None
    return random.choice(choices)


def _ensure_row_target(precolor: np.ndarray, row: int, color: int) -> None:
    if np.any(precolor[row, :] == color):
        return
    col = _pick_empty_in_row(precolor, row)
    if col is None:
        return
    precolor[row, col] = color


def _ensure_col_target(precolor: np.ndarray, col: int, color: int) -> None:
    if np.any(precolor[:, col] == color):
        return
    row = _pick_empty_in_col(precolor, col)
    if row is None:
        return
    precolor[row, col] = color


def _ensure_flip_row_visible(precolor: np.ndarray, row: int) -> None:
    width = precolor.shape[1]
    if any(int(precolor[row, c]) != BLACK and c != width - 1 - c for c in range(width)):
        return
    choices = _noncenter_indices(width)
    col = random.choice(choices if choices else list(range(width)))
    precolor[row, col] = random.choice((GREEN, YELLOW))


def _ensure_flip_col_visible(precolor: np.ndarray, col: int) -> None:
    height = precolor.shape[0]
    if any(int(precolor[r, col]) != BLACK and r != height - 1 - r for r in range(height)):
        return
    choices = _noncenter_indices(height)
    row = random.choice(choices if choices else list(range(height)))
    precolor[row, col] = random.choice((GREEN, YELLOW))


def _add_extra_segments(precolor: np.ndarray) -> None:
    height, width = precolor.shape
    for _ in range(random.randint(*EXTRA_SEGMENT_RANGE)):
        color = random.choice((GREEN, YELLOW))
        horizontal = random.choice((False, True))
        if horizontal:
            length = random.randint(1, min(3, width))
            row = random.randrange(height)
            col = random.randrange(width - length + 1)
            coords = [(row, col + offset) for offset in range(length)]
        else:
            length = random.randint(1, min(3, height))
            row = random.randrange(height - length + 1)
            col = random.randrange(width)
            coords = [(row + offset, col) for offset in range(length)]

        if any(int(precolor[r, c]) not in (BLACK, color) for r, c in coords):
            continue
        if all(int(precolor[r, c]) == color for r, c in coords):
            continue
        for r, c in coords:
            if int(precolor[r, c]) == BLACK:
                precolor[r, c] = color


def _render_input(
    inner: np.ndarray,
    flip_cols: np.ndarray,
    flip_rows: np.ndarray,
    row_replacements: np.ndarray,
    col_replacements: np.ndarray,
) -> np.ndarray:
    height, width = inner.shape
    grid = np.full((height + 2, width + 2), BLACK, dtype=int)
    grid[1:-1, 1:-1] = inner
    grid[0, 1:-1] = np.where(flip_cols, RED, BLACK)
    grid[1:-1, 0] = np.where(flip_rows, BLUE, BLACK)
    grid[1:-1, -1] = row_replacements
    grid[-1, 1:-1] = col_replacements
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*INNER_HEIGHT_RANGE)
    width = random.randint(*INNER_WIDTH_RANGE)

    flip_rows = _sample_flip_mask(height)
    flip_cols = _sample_flip_mask(width)

    unflipped_rows = [r for r in range(height) if not flip_rows[r]]
    unflipped_cols = [c for c in range(width) if not flip_cols[c]]
    green_row_choices = [r for r in unflipped_rows if r != height - 1 - r]
    yellow_col_choices = [c for c in unflipped_cols if c != width - 1 - c]
    if not green_row_choices or not yellow_col_choices:
        return None

    green_row = random.choice(green_row_choices)
    green_col = random.choice(np.flatnonzero(flip_cols).tolist())
    mirror_green_row = height - 1 - green_row

    yellow_row = random.choice(np.flatnonzero(flip_rows).tolist())
    yellow_col = random.choice(yellow_col_choices)
    mirror_yellow_col = width - 1 - yellow_col

    row_replacements = _sample_replacements(height, green_row, mirror_green_row)
    col_replacements = _sample_replacements(width, yellow_col, mirror_yellow_col)

    precolor = np.full((height, width), BLACK, dtype=int)
    precolor[green_row, green_col] = GREEN
    precolor[yellow_row, yellow_col] = YELLOW

    for row in np.flatnonzero(row_replacements):
        _ensure_row_target(precolor, int(row), GREEN)

    for col in np.flatnonzero(col_replacements):
        _ensure_col_target(precolor, int(col), YELLOW)

    safe_green_col = random.randrange(width)
    if int(precolor[mirror_green_row, safe_green_col]) == BLACK:
        precolor[mirror_green_row, safe_green_col] = GREEN

    safe_yellow_row = random.randrange(height)
    if int(precolor[safe_yellow_row, mirror_yellow_col]) == BLACK:
        precolor[safe_yellow_row, mirror_yellow_col] = YELLOW

    _add_extra_segments(precolor)

    for row in np.flatnonzero(flip_rows):
        _ensure_flip_row_visible(precolor, int(row))

    for col in np.flatnonzero(flip_cols):
        _ensure_flip_col_visible(precolor, int(col))

    if not np.any(precolor == GREEN) or not np.any(precolor == YELLOW):
        return None

    inner = _invert_flips(precolor, flip_cols, flip_rows)
    expected = _apply_transform(inner, flip_cols, flip_rows, row_replacements, col_replacements)
    if not np.array_equal(expected, _apply_transform(precolor, np.zeros(width, dtype=bool), np.zeros(height, dtype=bool), row_replacements, col_replacements)):
        return None

    grid = _render_input(inner, flip_cols, flip_rows, row_replacements, col_replacements)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-frame puzzle")
