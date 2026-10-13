import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_CHOICES = (5, 6, 7)
MIN_FOREGROUND = 6
MAX_FOREGROUND = 18
MIN_VISIBLE_FOREGROUND = 4
MIN_DISTINCT_VISIBLE_COLORS = 2
GENERATION_RETRIES = 400


def _expand_with_sizes(grid: np.ndarray, row_heights: np.ndarray, col_widths: np.ndarray) -> np.ndarray:
    row_heights = np.asarray(row_heights, dtype=int)
    col_widths = np.asarray(col_widths, dtype=int)
    out = np.full((int(row_heights.sum()), int(col_widths.sum())), BG, dtype=int)
    row_starts = np.cumsum(np.r_[0, row_heights[:-1]])
    col_starts = np.cumsum(np.r_[0, col_widths[:-1]])

    for r in range(grid.shape[0]):
        h = int(row_heights[r])
        if h == 0:
            continue
        r0 = int(row_starts[r])
        r1 = r0 + h
        for c in range(grid.shape[1]):
            w = int(col_widths[c])
            if w == 0:
                continue
            color = int(grid[r, c])
            if color == BG:
                continue
            c0 = int(col_starts[c])
            c1 = c0 + w
            out[r0:r1, c0:c1] = color
    return out


def _oracle(grid: np.ndarray) -> np.ndarray:
    row_counts = np.count_nonzero(grid != BG, axis=1)
    col_counts = np.count_nonzero(grid != BG, axis=0)
    return _expand_with_sizes(grid, col_counts, row_counts)


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each cell into a rectangle whose height and width come from the opposite indexed axis.

    Input structure:
    - The input is a square grid on a black background.
    - Colored foreground pixels are sparse, and some entire rows or columns can be empty.
    - Line indices matter: row i is paired with column i of the same original grid.

    Rule:
    - Output row i gets height equal to the number of foreground pixels in input column i.
    - Output column j gets width equal to the number of foreground pixels in input row j.
    - Any row or column whose assigned size is zero occupies no output space.
    - Each input cell expands to a solid block of that cell's color on a black background using its assigned row height and column width.

    Key insight:
    - The scaling factors are transposed across the diagonal: rows read their thickness from same-index columns, and columns read theirs from same-index rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square grid"
    return _oracle(g)


def _self_axis_output(grid: np.ndarray) -> np.ndarray:
    row_counts = np.count_nonzero(grid != BG, axis=1)
    col_counts = np.count_nonzero(grid != BG, axis=0)
    return _expand_with_sizes(grid, row_counts, col_counts)


def _cropped_output_if_square(grid: np.ndarray) -> np.ndarray | None:
    row_counts = np.count_nonzero(grid != BG, axis=1)
    col_counts = np.count_nonzero(grid != BG, axis=0)
    rows = np.flatnonzero(row_counts)
    cols = np.flatnonzero(col_counts)
    if rows.size == 0 or rows.size != cols.size:
        return None
    cropped = grid[np.ix_(rows, cols)]
    return _oracle(cropped)


def _make_mask(size: int) -> np.ndarray:
    row_count = random.randint(2, size - 1)
    col_count = random.randint(2, size - 1)
    rows = sorted(random.sample(range(size), row_count))
    cols = sorted(random.sample(range(size), col_count))
    density = random.uniform(0.35, 0.72)

    submask = np.random.rand(row_count, col_count) < density
    if not submask.any():
        submask[random.randrange(row_count), random.randrange(col_count)] = True

    for rr in range(row_count):
        if not submask[rr].any():
            submask[rr, random.randrange(col_count)] = True
    for cc in range(col_count):
        if not submask[:, cc].any():
            submask[random.randrange(row_count), cc] = True

    if submask.all():
        rr = random.randrange(row_count)
        cc = random.randrange(col_count)
        if submask[rr].sum() > 1 and submask[:, cc].sum() > 1:
            submask[rr, cc] = False

    holes = random.randint(0, 2)
    for _ in range(holes):
        candidates = [
            (rr, cc)
            for rr in range(row_count)
            for cc in range(col_count)
            if submask[rr, cc] and submask[rr].sum() > 1 and submask[:, cc].sum() > 1
        ]
        if not candidates:
            break
        rr, cc = random.choice(candidates)
        submask[rr, cc] = False

    mask = np.zeros((size, size), dtype=bool)
    mask[np.ix_(rows, cols)] = submask
    return mask


def _colorize(mask: np.ndarray) -> np.ndarray:
    row_counts = np.count_nonzero(mask, axis=1)
    col_counts = np.count_nonzero(mask, axis=0)
    active_rows = [int(r) for r in np.flatnonzero(row_counts)]
    active_cols = [int(c) for c in np.flatnonzero(col_counts)]
    row_rank = {r: i for i, r in enumerate(active_rows)}
    col_rank = {c: i for i, c in enumerate(active_cols)}

    palette_size = random.randint(2, min(5, int(mask.sum())))
    palette = random.sample(COLOR_POOL, palette_size)
    grid = np.zeros(mask.shape, dtype=int)

    for r, c in np.argwhere(mask):
        visible = col_counts[r] > 0 and row_counts[c] > 0
        code = 3 * row_rank[int(r)] + 5 * col_rank[int(c)] + int(visible) + int(r) + int(c)
        grid[int(r), int(c)] = palette[code % palette_size]
    return grid


def _is_interesting(mask: np.ndarray, grid: np.ndarray, out: np.ndarray) -> bool:
    row_counts = np.count_nonzero(mask, axis=1)
    col_counts = np.count_nonzero(mask, axis=0)
    foreground = int(mask.sum())
    if not (MIN_FOREGROUND <= foreground <= MAX_FOREGROUND):
        return False
    if not np.any(row_counts == 0) or not np.any(col_counts == 0):
        return False
    if (row_counts > 0).sum() < 2 or (col_counts > 0).sum() < 2:
        return False
    if len(set(map(int, row_counts[row_counts > 0]))) < 2:
        return False
    if len(set(map(int, col_counts[col_counts > 0]))) < 2:
        return False
    if np.array_equal(row_counts, col_counts):
        return False
    if not np.any((row_counts == 0) != (col_counts == 0)):
        return False

    visible_mask = mask & (col_counts > 0)[:, None] & (row_counts > 0)[None, :]
    visible_fg = int(visible_mask.sum())
    invisible_fg = foreground - visible_fg
    if visible_fg < MIN_VISIBLE_FOREGROUND or invisible_fg < 1:
        return False
    if invisible_fg * 2 > foreground:
        return False

    active_rows = np.flatnonzero(row_counts)
    active_cols = np.flatnonzero(col_counts)
    if np.all(mask[np.ix_(active_rows, active_cols)]):
        return False

    if out.shape != (foreground, foreground) or max(out.shape) > 30:
        return False
    out_colors = set(map(int, out.ravel())) - {BG}
    if len(out_colors) < MIN_DISTINCT_VISIBLE_COLORS:
        return False
    if np.array_equal(out, _self_axis_output(grid)):
        return False

    cropped = _cropped_output_if_square(grid)
    if cropped is not None and np.array_equal(out, cropped):
        return False

    if not any(
        (row_counts[i] == 0 and col_counts[i] > 0) or (col_counts[i] == 0 and row_counts[i] > 0)
        for i in range(mask.shape[0])
    ):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZE_CHOICES)
    mask = _make_mask(size)
    grid = _colorize(mask)
    expected = _oracle(grid)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _is_interesting(mask, grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
