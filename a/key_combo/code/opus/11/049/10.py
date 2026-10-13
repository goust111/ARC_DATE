import random

import numpy as np


BG = 0
FG = 1
AXIS = 5
ODD_SIDES = (9, 11, 13, 15, 17, 19)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int, int, int]:
    bg = _most_common_color(grid)
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        full_rows = np.where(np.all(grid == color, axis=1))[0]
        full_cols = np.where(np.all(grid == color, axis=0))[0]
        if len(full_rows) > 0 and len(full_cols) > 0:
            candidates.append((color, full_rows, full_cols))

    assert len(candidates) == 1, "expected one full-row/full-column cross"
    axis_color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "cross must use one row and one column"
    axis_r = int(rows[0])
    axis_c = int(cols[0])
    assert axis_r * 2 + 1 == grid.shape[0] and axis_c * 2 + 1 == grid.shape[1], "cross must be centered"
    return int(bg), int(axis_color), axis_r, axis_c


def _wave_shift(distance_from_axis: int) -> int:
    return 1 if distance_from_axis % 2 == 1 else 0


def _render_wave_rectangle(
    shape: tuple[int, int], axis_r: int, axis_c: int, top: int, left: int, height: int, width: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    bottom = axis_r - 1
    right = left + width - 1
    for r in range(top, bottom + 1):
        shift = _wave_shift(bottom - r)
        mask[r, left + shift : right + shift + 1] = True
    return mask


def _recover_base_rectangle(source_mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = source_mask.shape
    base = np.zeros_like(source_mask, dtype=bool)
    bottom = h - 1

    for r in range(h):
        cols = np.flatnonzero(source_mask[r])
        if cols.size == 0:
            continue
        shift = _wave_shift(bottom - r)
        if shift:
            assert int(cols.min()) >= shift, "shifted rows must be recoverable"
            cols = cols - shift
        base[r, cols] = True

    occupied_rows = np.where(base.any(axis=1))[0]
    assert occupied_rows.size > 0, "source quadrant must contain a wave object"
    top = int(occupied_rows[0])
    assert np.array_equal(occupied_rows, np.arange(top, h)), "base rectangle must be bottom-anchored"

    reference_cols = np.flatnonzero(base[top])
    assert reference_cols.size > 0
    left = int(reference_cols[0])
    right = int(reference_cols[-1])
    expected_cols = np.arange(left, right + 1)

    for r in range(top, h):
        cols = np.flatnonzero(base[r])
        assert np.array_equal(cols, expected_cols), "deshifted rows must form one rectangle"

    rerendered = _render_wave_rectangle(source_mask.shape, h, w, top, left, h - top, right - left + 1)
    assert np.array_equal(source_mask, rerendered), "source quadrant must match the anchored wave rule"
    return top, left, h - top, right - left + 1


def _render_output_from_rect(
    shape: tuple[int, int], axis_r: int, axis_c: int, top: int, left: int, height: int, width: int
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[axis_r, :] = AXIS
    out[:, axis_c] = AXIS

    bottom = axis_r - 1
    for r in range(top, top + height):
        shift = _wave_shift(bottom - r)
        for c in range(left, left + width):
            src_c = c + shift
            pts = (
                (r, src_c),
                (r, 2 * axis_c - src_c),
                (2 * axis_r - r, src_c),
                (2 * axis_r - r, 2 * axis_c - src_c),
            )
            for rr, cc in pts:
                out[rr, cc] = FG
    return out


def _render_rectangle_only_output(
    shape: tuple[int, int], axis_r: int, axis_c: int, top: int, left: int, height: int, width: int
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[axis_r, :] = AXIS
    out[:, axis_c] = AXIS
    for r in range(top, top + height):
        for c in range(left, left + width):
            pts = (
                (r, c),
                (r, 2 * axis_c - c),
                (2 * axis_r - r, c),
                (2 * axis_r - r, 2 * axis_c - c),
            )
            for rr, cc in pts:
                out[rr, cc] = FG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the centered cross by mirroring one quadrant's anchored horizontal wave into all four quadrants.

    Input structure
    - The grid uses a black background plus one full grey row and one full grey column that form a centered cross.
    - All blue foreground pixels lie in the upper-left quadrant, strictly off the cross.
    - Those blue pixels are a filled rectangle whose odd rows, counted upward from the row adjacent to the cross, have been shifted one cell to the right.

    Rule
    - Identify the grey cross as the horizontal and vertical mirror axes.
    - Recover the bottom-anchored base rectangle by undoing the one-cell right shift on odd-distance rows in the upper-left quadrant.
    - Reapply that same anchored wave while reflecting the rectangle across the row axis, the column axis, and both axes, so the right-side copies become the natural mirrored left-shifted version.

    Key insight
    - The wave phase is tied to distance from the center cross, so direct coordinate mirroring of the waved shape is correct only because it is the mirrored form of the same recovered rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, axis_color, axis_r, axis_c = _find_cross_axes(g)
    fg_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, axis_color)]
    assert len(fg_colors) == 1, "expected exactly one foreground color besides the cross"
    fg = fg_colors[0]

    other_quadrants = [
        g[:axis_r, axis_c + 1 :],
        g[axis_r + 1 :, :axis_c],
        g[axis_r + 1 :, axis_c + 1 :],
    ]
    assert all(np.all(block == bg) for block in other_quadrants), "only the upper-left quadrant may be populated"

    source_mask = g[:axis_r, :axis_c] == fg
    top, left, height, width = _recover_base_rectangle(source_mask)

    out = np.full(g.shape, bg, dtype=int)
    out[axis_r, :] = axis_color
    out[:, axis_c] = axis_color

    bottom = axis_r - 1
    for r in range(top, top + height):
        shift = _wave_shift(bottom - r)
        for c in range(left, left + width):
            src_c = c + shift
            pts = (
                (r, src_c),
                (r, 2 * axis_c - src_c),
                (2 * axis_r - r, src_c),
                (2 * axis_r - r, 2 * axis_c - src_c),
            )
            for rr, cc in pts:
                out[rr, cc] = fg

    return out


def _make_input(height: int, width: int, left: int, rows: int, cols: int) -> np.ndarray:
    axis_r = rows // 2
    axis_c = cols // 2
    top = axis_r - height
    source_mask = _render_wave_rectangle((axis_r, axis_c), axis_r, axis_c, top, left, height, width)

    grid = np.full((rows, cols), BG, dtype=int)
    grid[axis_r, :] = AXIS
    grid[:, axis_c] = AXIS
    grid[:axis_r, :axis_c][source_mask] = FG
    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        bg, axis_color, axis_r, axis_c = _find_cross_axes(grid)
        if bg != BG or axis_color != AXIS:
            return False

        if set(map(int, np.unique(grid))) != {BG, FG, AXIS}:
            return False

        source_mask = grid[:axis_r, :axis_c] == FG
        top, left, height, width = _recover_base_rectangle(source_mask)
        if height < 3 or width < 2:
            return False

        expected = _render_output_from_rect(grid.shape, axis_r, axis_c, top, left, height, width)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False

        if np.array_equal(grid, solved):
            return False

        rectangle_only = _render_rectangle_only_output(grid.shape, axis_r, axis_c, top, left, height, width)
        if np.array_equal(expected, rectangle_only):
            return False

        unshifted_ul = np.zeros((axis_r, axis_c), dtype=bool)
        unshifted_ul[top : top + height, left : left + width] = True
        if np.array_equal(unshifted_ul, source_mask):
            return False

        if np.count_nonzero(source_mask) * 4 != np.count_nonzero(solved == FG):
            return False

        return True
    except AssertionError:
        return False


def _attempt_generate() -> np.ndarray | None:
    rows = random.choice(ODD_SIDES)
    cols = random.choice(ODD_SIDES)
    axis_r = rows // 2
    axis_c = cols // 2

    max_height = axis_r
    max_width = axis_c - 1
    if max_height < 3 or max_width < 2:
        return None

    height = random.randint(3, max_height)
    width = random.randint(2, max_width)
    left = random.randint(0, axis_c - width - 1)

    grid = _make_input(height, width, left, rows, cols)
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-wave symmetry input")
