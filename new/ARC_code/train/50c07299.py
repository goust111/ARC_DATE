import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move the diagonal foreground segment to the next longer run along the same up-right diagonal.

    Input structure
    The grid has one background color and one foreground color. The foreground pixels form a
    contiguous one-pixel-wide segment on a single anti-diagonal.

    Rule
    Erase the current segment. If its columns are c0..c1 and its length is L, draw the same
    color on columns c1+1..c1+L+1 along the same anti-diagonal.

    Key insight
    The segment advances past its current upper-right end by one whole segment position, and
    the next segment is one pixel longer than the current one.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    coords = np.argwhere(g == fg)
    assert coords.size > 0
    diag_sums = {int(r + c) for r, c in coords}
    assert len(diag_sums) == 1
    diag_sum = next(iter(diag_sums))

    cols = sorted(int(c) for _r, c in coords)
    assert cols == list(range(cols[0], cols[-1] + 1))
    length = len(cols)
    out_cols = range(cols[-1] + 1, cols[-1] + length + 2)

    out = np.full_like(g, bg)
    for c in out_cols:
        r = diag_sum - c
        assert in_bounds(g.shape, r, c)
        out[r, c] = fg
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette bounds.
GRID_SIZE_RANGE = (10, 20)
COLOR_COUNT = 10

# Segment sizes; output length is one pixel longer than input length.
OUTPUT_LEN_MIN, OUTPUT_LEN_MAX_CAP = 2, 8

# Rejection-sampling retry budget.
GENERATION_RETRIES = 300


def _fits_segment(n, diag_sum, c0, length):
    c1 = c0 + length - 1
    return 0 <= c0 and c1 < n and 0 <= diag_sum - c1 and diag_sum - c0 < n


def _valid_pairs(n, in_len, out_len):
    pairs = []
    for c_in in range(n):
        c_out = c_in + in_len
        if c_out + out_len > n:
            continue
        for diag_sum in range(2 * n - 1):
            if _fits_segment(n, diag_sum, c_in, in_len) and _fits_segment(n, diag_sum, c_out, out_len):
                pairs.append((diag_sum, c_in))
    return pairs


def _random_colors():
    bg = random.randrange(COLOR_COUNT)
    fg = random.randrange(COLOR_COUNT - 1)
    if fg >= bg:
        fg += 1
    return bg, fg


def _draw_segment(n, bg, fg, diag_sum, c0, length):
    grid = np.full((n, n), bg, dtype=int)
    cols = np.arange(c0, c0 + length)
    grid[diag_sum - cols, cols] = fg
    return grid


def _gen_attempt():
    n = random.randint(*GRID_SIZE_RANGE)
    bg, fg = _random_colors()

    max_out_len = min(OUTPUT_LEN_MAX_CAP, n // 2)
    out_len = random.randint(OUTPUT_LEN_MIN, max_out_len)
    in_len = out_len - 1

    pairs = _valid_pairs(n, in_len, out_len)
    if not pairs:
        return None

    diag_sum, c_in = random.choice(pairs)
    c_out = c_in + in_len

    out = _draw_segment(n, bg, fg, diag_sum, c_out, out_len)
    inp = _draw_segment(n, bg, fg, diag_sum, c_in, in_len)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid 50c07299 input")
