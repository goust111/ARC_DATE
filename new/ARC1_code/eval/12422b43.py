import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Continue the grey-bracketed top motif downward through the empty suffix.

    Input structure
    The grid has a black background. A contiguous run of grey pixels begins at
    the top of the leftmost column; those grey pixels mark the rows of a colored
    motif. Below the marked rows there may be other foreground rows, followed by
    an all-background suffix.

    Rule
    Copy the marked rows as a motif after changing the grey marker cells to
    background. Starting just below the lowest existing foreground row, write
    those motif rows cyclically into the empty suffix, preserving the original
    grid everywhere else.

    Key insight
    The grey segment is a bracket for the motif height; the fill below the
    existing drawing restarts from the first marked row rather than continuing
    from any unmarked foreground row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) > 0, "expected a nonempty grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    marker_rows = np.flatnonzero(g[:, 0] == Color.GREY)
    assert marker_rows.size > 0, "expected grey marker rows"
    assert np.array_equal(marker_rows, np.arange(marker_rows.size)), "grey marker must be a top-left run"
    assert np.count_nonzero(g == Color.GREY) == marker_rows.size, "grey pixels are only the row markers"

    motif = g[: marker_rows.size].copy()
    motif[:, 0] = bg
    assert np.all(np.any(motif != bg, axis=1)), "each marked row must contain motif pixels"

    foreground_rows = np.flatnonzero(np.any(g != bg, axis=1))
    assert foreground_rows.size > 0, "expected foreground"
    start = int(foreground_rows[-1]) + 1
    assert np.all(g[start:] == bg), "suffix below the drawing must be empty"

    out = g.copy()
    for r in range(start, g.shape[0]):
        motif_row = motif[(r - start) % motif.shape[0]]
        paint = motif_row != bg
        out[r, paint] = motif_row[paint]
    return out

# --------------- generator (o5x) ---------------

# Palette and fixed colors.
_GEN_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]

# Canvas and vertical zone bounds.
_GEN_WIDTH_RANGE = (5, 7)
_GEN_TOTAL_H_RANGE = (5, 13)
_GEN_MARKER_H_RANGE = (1, 4)
_GEN_MIDDLE_H_RANGE = (1, 3)
_GEN_SUFFIX_H_RANGE = (2, 8)
_GEN_RIGHT_PAD_RANGE = (1, 2)
_GEN_CONTENT_COL_START = 1

# Motif shape and color bounds.
_GEN_MOTIF_W_RANGE = (2, 3)
_GEN_MOTIF_ROW_PIXELS_RANGE = (1, 3)
_GEN_MOTIF_COLOR_RANGE = (2, 3)

# Middle-zone content bounds.
_GEN_MIDDLE_PIXEL_MAX = 5
_GEN_MIDDLE_COLOR_RANGE = (1, 3)
_GEN_MIDDLE_OVERLAP_PROB = 0.75

# Rejection sampling limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_pick_dimensions() -> tuple[int, int, int, int]:
    width = random.randint(*_GEN_WIDTH_RANGE)
    marker_h = random.randint(*_GEN_MARKER_H_RANGE)
    middle_h = random.randint(*_GEN_MIDDLE_H_RANGE)
    min_suffix = max(
        _GEN_SUFFIX_H_RANGE[0],
        marker_h,
        _GEN_TOTAL_H_RANGE[0] - marker_h - middle_h,
    )
    max_suffix = min(_GEN_SUFFIX_H_RANGE[1], _GEN_TOTAL_H_RANGE[1] - marker_h - middle_h)
    suffix_h = random.randint(min_suffix, max_suffix)
    height = marker_h + middle_h + suffix_h
    return height, width, marker_h, middle_h


def _gen_usable_cols(width: int, right_pad: int) -> list[int]:
    return list(range(_GEN_CONTENT_COL_START, width - right_pad))


def _gen_assign_colors(
    positions: list[tuple[int, int]], color_count: int
) -> dict[tuple[int, int], int]:
    colors = random.sample(_GEN_COLORS, color_count)
    shuffled = positions[:]
    random.shuffle(shuffled)
    values = colors[:] + [random.choice(colors) for _ in range(len(shuffled) - color_count)]
    random.shuffle(values)
    return dict(zip(shuffled, values))


def _gen_paint_positions(
    grid: np.ndarray, positions: list[tuple[int, int]], color_count: int
) -> None:
    for (r, c), color in _gen_assign_colors(positions, color_count).items():
        grid[r, c] = color


def _gen_contiguous_run(
    cols: list[int],
    count: int,
    *,
    include_left: bool = False,
    include_right: bool = False,
    overlap: list[int] | None = None,
) -> set[int]:
    starts = list(range(len(cols) - count + 1))
    if include_left:
        starts = [0]
    if include_right:
        starts = [len(cols) - count]
    if overlap is not None:
        overlap_set = set(overlap)
        starts = [
            i for i in starts if any(c in overlap_set for c in cols[i : i + count])
        ]
    start = random.choice(starts)
    return set(cols[start : start + count])


def _gen_build_motif(
    grid: np.ndarray, marker_h: int, width: int, right_pad: int
) -> list[int]:
    usable_cols = _gen_usable_cols(width, right_pad)
    motif_w = random.randint(_GEN_MOTIF_W_RANGE[0], min(_GEN_MOTIF_W_RANGE[1], len(usable_cols)))
    start_c = random.choice(usable_cols[: len(usable_cols) - motif_w + 1])
    motif_cols = list(range(start_c, start_c + motif_w))

    max_per_row = min(_GEN_MOTIF_ROW_PIXELS_RANGE[1], motif_w)
    row_counts = []
    for _ in range(marker_h):
        low = _GEN_MOTIF_ROW_PIXELS_RANGE[0]
        if marker_h == 1:
            low = motif_w
        row_counts.append(random.randint(low, max_per_row))

    used = [
        _gen_contiguous_run(motif_cols, count)
        for count in row_counts
    ]
    left_row = random.randrange(marker_h)
    used[left_row] = _gen_contiguous_run(
        motif_cols, row_counts[left_row], include_left=True
    )
    right_candidates = [
        r for r in range(marker_h) if r != left_row or row_counts[r] == motif_w
    ]
    right_row = random.choice(right_candidates)
    used[right_row] = _gen_contiguous_run(
        motif_cols, row_counts[right_row], include_right=True
    )

    positions = [(r, c) for r in range(marker_h) for c in sorted(used[r])]
    color_count = random.randint(_GEN_MOTIF_COLOR_RANGE[0], min(_GEN_MOTIF_COLOR_RANGE[1], len(positions)))
    _gen_paint_positions(grid, positions, color_count)

    return motif_cols


def _gen_build_middle(
    grid: np.ndarray,
    marker_h: int,
    middle_h: int,
    width: int,
    right_pad: int,
    motif_cols: list[int],
) -> None:
    usable_cols = _gen_usable_cols(width, right_pad)
    max_pixels = min(_GEN_MIDDLE_PIXEL_MAX, middle_h * len(usable_cols))
    pixel_count = random.randint(middle_h, max_pixels)

    row_counts = [1] * middle_h
    for _ in range(pixel_count - middle_h):
        choices = [r for r, count in enumerate(row_counts) if count < len(usable_cols)]
        row_counts[random.choice(choices)] += 1

    anchor_row = random.randrange(middle_h)
    used = []
    for r in range(middle_h):
        overlap = motif_cols if r == anchor_row or random.random() < _GEN_MIDDLE_OVERLAP_PROB else None
        used.append(_gen_contiguous_run(usable_cols, row_counts[r], overlap=overlap))

    positions = [
        (marker_h + r, c)
        for r in range(middle_h)
        for c in sorted(used[r])
    ]
    color_count = random.randint(_GEN_MIDDLE_COLOR_RANGE[0], min(_GEN_MIDDLE_COLOR_RANGE[1], len(positions)))
    _gen_paint_positions(grid, positions, color_count)


def _gen_expected_output(grid: np.ndarray, marker_h: int, middle_h: int) -> np.ndarray:
    out = grid.copy()
    motif = grid[:marker_h].copy()
    motif[:, 0] = Color.BLACK
    start = marker_h + middle_h
    for r in range(start, grid.shape[0]):
        motif_row = motif[(r - start) % marker_h]
        paint = motif_row != Color.BLACK
        out[r, paint] = motif_row[paint]
    return out


def _gen_attempt() -> np.ndarray | None:
    height, width, marker_h, middle_h = _gen_pick_dimensions()
    right_pad = random.randint(*_GEN_RIGHT_PAD_RANGE)
    if width - right_pad - _GEN_CONTENT_COL_START < _GEN_MOTIF_W_RANGE[0]:
        right_pad = _GEN_RIGHT_PAD_RANGE[0]

    grid = np.zeros((height, width), dtype=int)
    grid[:marker_h, 0] = Color.GREY
    motif_cols = _gen_build_motif(grid, marker_h, width, right_pad)
    _gen_build_middle(grid, marker_h, middle_h, width, right_pad, motif_cols)

    expected = _gen_expected_output(grid, marker_h, middle_h)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 12422b43 input")
