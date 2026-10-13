import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
SEGMENT_COUNT_RANGE = (2, 7)
SEGMENT_WIDTH_RANGE = (1, 8)
TOTAL_WIDTH_RANGE = (4, 26)
MAIN_HEIGHT_MARGIN_RANGE = (1, 10)
PATTERN_MODES = ("contiguous", "sparse", "multi_run")


def _bottom_segments(row: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    start = 0
    width = int(row.shape[0])
    while start < width:
        color = int(row[start])
        assert color != BG, "marker row must be fully covered by colored segments"
        end = start + 1
        while end < width and int(row[end]) == color:
            end += 1
        segments.append((color, start, end, end - start))
        start = end
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each bottom marker segment into one output column whose new height equals that segment's width.

    Input structure:
    - The background is black.
    - The bottom row is a marker row made of contiguous non-black color segments with no gaps.
    - Above the marker row, each column uses only black plus the marker color beneath it.
    - All columns inside the same marker segment repeat the same single-column pattern.

    Rule:
    - Split the bottom marker row into contiguous color segments.
    - For each segment, pick any one column above it as the source pattern and measure its foreground height as the number of non-black cells in that column.
    - Replace the whole segment by one output column in the same color; its new foreground height is the segment width rather than the source height.
    - Draw each output column as a bottom-aligned vertical stack directly above a one-cell-wide copy of the marker row.

    Key insight:
    - The marker row does two jobs at once: contiguous runs define which columns belong together, and each run's width is the target height for that group's compressed output column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 2 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    canvas = g[:-1]
    segments = _bottom_segments(g[-1])
    out_h = max(width for _color, _start, _end, width in segments) + 1
    out = np.full((out_h, len(segments)), BG, dtype=int)

    for idx, (color, start, end, seg_width) in enumerate(segments):
        ref_col = canvas[:, start]
        assert np.all((ref_col == BG) | (ref_col == color)), "segment columns must use only bg or segment color"
        source_height = int(np.count_nonzero(ref_col == color))
        assert source_height > 0, "each segment must encode a non-empty source pattern"

        for col in range(start + 1, end):
            cur = canvas[:, col]
            assert np.all((cur == BG) | (cur == color)), "segment columns must match their marker color"
            assert np.array_equal(cur, ref_col), "all columns inside one segment must repeat the same pattern"

        out[-1, idx] = color
        start_row = out_h - 1 - seg_width
        out[start_row : out_h - 1, idx] = color

    return out


def _sample_widths() -> list[int]:
    for _ in range(200):
        n_segments = random.randint(*SEGMENT_COUNT_RANGE)
        widths = [random.randint(*SEGMENT_WIDTH_RANGE) for _ in range(n_segments)]
        total_width = sum(widths)
        if not (TOTAL_WIDTH_RANGE[0] <= total_width <= TOTAL_WIDTH_RANGE[1]):
            continue
        if max(widths) == 1:
            continue
        return widths
    raise RuntimeError("failed to sample segment widths")


def _sample_segment_colors(n_segments: int) -> list[int]:
    palette_size = random.randint(2, min(6, n_segments, 9))
    palette = random.sample(range(1, 10), palette_size)
    colors = [random.choice(palette)]
    for _ in range(1, n_segments):
        options = [color for color in palette if color != colors[-1]]
        colors.append(random.choice(options))
    return colors


def _sample_source_counts(widths: list[int], main_height: int) -> list[int]:
    n_segments = len(widths)
    modes: list[str | None] = [None] * n_segments

    smaller = [idx for idx, width in enumerate(widths) if width > 1]
    larger = [idx for idx, width in enumerate(widths) if width < main_height]
    if smaller:
        modes[random.choice(smaller)] = "smaller"
    remaining_larger = [idx for idx in larger if modes[idx] is None]
    if remaining_larger:
        modes[random.choice(remaining_larger)] = "larger"

    counts: list[int] = []
    for idx, width in enumerate(widths):
        mode = modes[idx]
        if mode is None:
            choices = ["equal"]
            if width > 1:
                choices.append("smaller")
            if width < main_height:
                choices.append("larger")
            mode = random.choice(choices)

        if mode == "smaller":
            counts.append(random.randint(1, width - 1))
        elif mode == "larger":
            counts.append(random.randint(width + 1, main_height))
        else:
            counts.append(width)

    return counts


def _count_runs(mask: np.ndarray) -> int:
    runs = 0
    inside = False
    for value in mask:
        if value and not inside:
            runs += 1
            inside = True
        elif not value:
            inside = False
    return runs


def _multi_run_mask(height: int, count: int) -> np.ndarray | None:
    target_runs = min(random.randint(2, 3), count)
    for _ in range(100):
        rows = sorted(random.sample(range(height), count))
        mask = np.zeros(height, dtype=bool)
        mask[rows] = True
        if _count_runs(mask) == target_runs:
            return mask
    return None


def _sample_mask(height: int, count: int) -> np.ndarray:
    mode = random.choice(PATTERN_MODES)
    if mode == "contiguous":
        start = random.randint(0, height - count)
        mask = np.zeros(height, dtype=bool)
        mask[start : start + count] = True
        return mask
    if mode == "multi_run" and count >= 2:
        mask = _multi_run_mask(height, count)
        if mask is not None:
            return mask

    mask = np.zeros(height, dtype=bool)
    mask[random.sample(range(height), count)] = True
    return mask


def _render_input(widths: list[int], colors: list[int], masks: list[np.ndarray]) -> np.ndarray:
    main_height = int(masks[0].shape[0])
    grid = np.full((main_height + 1, sum(widths)), BG, dtype=int)
    col = 0
    for width, color, mask in zip(widths, colors, masks):
        rows = np.flatnonzero(mask)
        grid[-1, col : col + width] = color
        grid[np.ix_(rows, np.arange(col, col + width))] = color
        col += width
    return grid


def _expected_output(widths: list[int], colors: list[int]) -> np.ndarray:
    out_h = max(widths) + 1
    out = np.full((out_h, len(widths)), BG, dtype=int)
    for idx, (width, color) in enumerate(zip(widths, colors)):
        out[-1, idx] = color
        out[out_h - 1 - width : out_h - 1, idx] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    widths = _sample_widths()
    max_width = max(widths)
    max_main_height = min(29, max_width + MAIN_HEIGHT_MARGIN_RANGE[1])
    min_main_height = min(29, max_width + MAIN_HEIGHT_MARGIN_RANGE[0])
    main_height = random.randint(min_main_height, max_main_height)
    counts = _sample_source_counts(widths, main_height)
    masks = [_sample_mask(main_height, count) for count in counts]
    colors = _sample_segment_colors(len(widths))

    grid = _render_input(widths, colors, masks)
    expected = _expected_output(widths, colors)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
