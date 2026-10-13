import random

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = (1, 2, 3, 4, 6)

GRID_HEIGHT_RANGE = (13, 22)
GRID_WIDTH_RANGE = (14, 24)
BAR_COUNT_CHOICES = (4, 5)
BAR_LENGTH_RANGE = (2, 8)
MAX_GENERATION_ATTEMPTS = 500


def _extract_horizontal_bars(grid: np.ndarray) -> list[dict]:
    """Extract maximal horizontal foreground runs using the ARC (r, c) convention."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    height, width = g.shape
    covered = np.zeros_like(g, dtype=bool)
    bars: list[dict] = []

    for r in range(height):
        c = 0
        while c < width:
            color = int(g[r, c])
            if color == BACKGROUND:
                c += 1
                continue

            c0 = c
            while c + 1 < width and int(g[r, c + 1]) == color:
                c += 1
            c1 = c

            if c0 > 0 and int(g[r, c0 - 1]) != BACKGROUND:
                raise ValueError("foreground bars must be horizontally isolated")
            if c1 + 1 < width and int(g[r, c1 + 1]) != BACKGROUND:
                raise ValueError("foreground bars must be horizontally isolated")

            for cc in range(c0, c1 + 1):
                if r > 0 and int(g[r - 1, cc]) != BACKGROUND:
                    raise ValueError("foreground bars must not touch vertically")
                if r + 1 < height and int(g[r + 1, cc]) != BACKGROUND:
                    raise ValueError("foreground bars must not touch vertically")
                covered[r, cc] = True

            bars.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "color": int(color),
                    "length": int(c1 - c0 + 1),
                }
            )
            c += 1

    if np.any((g != BACKGROUND) & ~covered):
        raise ValueError("all foreground pixels must belong to horizontal bars")

    return bars


def _render_concentric_frames(colors_in_order: list[int]) -> np.ndarray:
    n = len(colors_in_order)
    size = 2 * n - 1
    out = np.full((size, size), BACKGROUND, dtype=int)

    for layer, color in enumerate(colors_in_order):
        lo = layer
        hi = size - 1 - layer
        out[lo, lo : hi + 1] = int(color)
        out[hi, lo : hi + 1] = int(color)
        out[lo : hi + 1, lo] = int(color)
        out[lo : hi + 1, hi] = int(color)

    return out


def _sorted_bars(bars: list[dict]) -> list[dict]:
    return sorted(bars, key=lambda bar: (-int(bar["length"]), int(bar["row"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure scattered horizontal bar lengths and redraw their colors as concentric square frames.

    Input structure:
    - A black background contains several single-color horizontal bar segments.
    - Each foreground object occupies exactly one row and one contiguous column interval.
    - Bar colors are distinct foreground colors, and bars do not touch each other.

    Rule:
    - Identify every horizontal bar segment and record its color and pixel length.
    - Sort the bars by length from largest to smallest; if lengths tie, the smaller row index comes first.
    - Create a (2N-1) x (2N-1) output grid, where N is the number of bars, and draw concentric square-frame rings from outermost to innermost using the sorted color order.

    Key insight:
    - The input geometry itself is discarded: only each bar's measured length determines the nesting order of the output frames.
    """
    bars = _extract_horizontal_bars(np.asarray(grid, dtype=int))
    ordered = _sorted_bars(bars)
    return _render_concentric_frames([bar["color"] for bar in ordered])


def _sample_rows(height: int, count: int) -> list[int]:
    candidates = list(range(1, height - 1, 2))
    if len(candidates) < count:
        raise ValueError("grid too small for the requested number of isolated rows")
    return sorted(random.sample(candidates, count))


def _bar_signature_orders(bars: list[dict]) -> tuple[list[int], list[int], list[int], list[int]]:
    true_order = [bar["color"] for bar in _sorted_bars(bars)]
    row_order = [bar["color"] for bar in sorted(bars, key=lambda bar: int(bar["row"]))]
    col_order = [bar["color"] for bar in sorted(bars, key=lambda bar: (int(bar["c0"]), int(bar["row"])))]
    color_order = [bar["color"] for bar in sorted(bars, key=lambda bar: int(bar["color"]))]
    return true_order, row_order, col_order, color_order


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        raise ValueError("generated input must be a 2D ARC grid no larger than 30x30")

    bars = _extract_horizontal_bars(g)
    count = len(bars)
    if count not in BAR_COUNT_CHOICES:
        raise ValueError("generator must produce 4 or 5 bars")
    if len({int(bar["row"]) for bar in bars}) != count:
        raise ValueError("bars must occupy distinct rows")

    colors = [int(bar["color"]) for bar in bars]
    lengths = [int(bar["length"]) for bar in bars]
    if len(set(colors)) != count:
        raise ValueError("bar colors must be distinct")
    if any(color not in FOREGROUND_COLORS for color in colors):
        raise ValueError("bar colors must come from the allowed visible palette")
    if any(length < BAR_LENGTH_RANGE[0] or length > BAR_LENGTH_RANGE[1] for length in lengths):
        raise ValueError("bar lengths must stay within the configured bounds")
    if len(set(lengths)) != count:
        raise ValueError("bar lengths must be unique")

    output = main(g)
    if output.shape != (2 * count - 1, 2 * count - 1):
        raise ValueError("output size must be determined only by the number of bars")

    true_order, row_order, col_order, color_order = _bar_signature_orders(bars)
    if true_order == row_order:
        raise ValueError("sample degenerates to row-order sorting")
    if true_order == col_order:
        raise ValueError("sample degenerates to left-position sorting")
    if true_order == color_order:
        raise ValueError("sample degenerates to color-id sorting")

    if np.count_nonzero(g) != sum(lengths):
        raise ValueError("foreground pixels must equal the total bar lengths")


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(BAR_COUNT_CHOICES)

    if 2 * count - 1 > min(height, width):
        return None

    max_length = min(BAR_LENGTH_RANGE[1], width - 2)
    if max_length < BAR_LENGTH_RANGE[0] or max_length - BAR_LENGTH_RANGE[0] + 1 < count:
        return None

    rows = _sample_rows(height, count)
    lengths = random.sample(range(BAR_LENGTH_RANGE[0], max_length + 1), count)
    colors = random.sample(list(FOREGROUND_COLORS), count)
    random.shuffle(lengths)
    random.shuffle(colors)

    grid = np.zeros((height, width), dtype=int)
    bars: list[dict] = []
    for row, length, color in zip(rows, lengths, colors):
        start = random.randint(1, width - length - 1)
        grid[row, start : start + length] = int(color)
        bars.append(
            {
                "row": int(row),
                "c0": int(start),
                "c1": int(start + length - 1),
                "color": int(color),
                "length": int(length),
            }
        )

    try:
        _validate_generated(grid)
    except ValueError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bar-sorting input")
