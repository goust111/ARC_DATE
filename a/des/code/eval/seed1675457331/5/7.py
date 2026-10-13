import random
from collections import Counter

import numpy as np


GEN_RETRIES = 300
INNER_WIDTH_RANGE = (4, 13)
PERIOD_RANGE = range(1, 6)
INPUT_HEIGHT_RANGE = (8, 24)
OUTPUT_HEIGHT_RANGE = (8, 30)
MIN_COMPLETE_SEGMENTS = 4


def _run_lengths(seq) -> list[tuple[int, int]]:
    seq = [int(x) for x in seq]
    runs: list[tuple[int, int]] = []
    cur = seq[0]
    length = 1
    for value in seq[1:]:
        if value == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = value
            length = 1
    runs.append((cur, length))
    return runs


def _measure_period(seq) -> int:
    runs = _run_lengths(seq)
    lengths = [length for _, length in runs]
    assert lengths, "frequency bar must contain at least one run"
    if len(lengths) > 1 and len(set(lengths[:-1])) == 1 and lengths[-1] != lengths[0]:
        lengths = lengths[:-1]
    return int(Counter(lengths).most_common(1)[0][0])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1] if int(colors[0]) == int(color) else colors[0])


def _paint_column(out: np.ndarray, top: int, col: int, height: int, period: int, start_color: int, colors: tuple[int, int]) -> None:
    alt_color = _other_color(colors, start_color)
    for offset in range(height):
        color = start_color if (offset // period) % 2 == 0 else alt_color
        out[top + offset, col] = int(color)


def _paint_interior(
    out: np.ndarray,
    top: int,
    height: int,
    left: int,
    right: int,
    period: int,
    start_color: int,
    colors: tuple[int, int],
) -> None:
    alt_color = _other_color(colors, start_color)
    for offset in range(height):
        color = start_color if (offset // period) % 2 == 0 else alt_color
        out[top + offset, left:right] = int(color)


def _render_grid(
    band_height: int,
    inner_width: int,
    left_period: int,
    stripe_period: int,
    right_period: int,
    colors: tuple[int, int],
    stripe_start: int,
    left_start: int,
    right_start: int,
    top_pad: int = 0,
    bottom_pad: int = 0,
) -> np.ndarray:
    total_height = int(top_pad + band_height + bottom_pad)
    width = int(inner_width + 2)
    out = np.zeros((total_height, width), dtype=int)
    band_top = int(top_pad)
    _paint_column(out, band_top, 0, band_height, left_period, left_start, colors)
    _paint_interior(out, band_top, band_height, 1, width - 1, stripe_period, stripe_start, colors)
    _paint_column(out, band_top, width - 1, band_height, right_period, right_start, colors)
    return out


def _build_valid_configs() -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    expand: list[tuple[int, int, int]] = []
    compress: list[tuple[int, int, int]] = []
    for in_period in PERIOD_RANGE:
        for out_period in PERIOD_RANGE:
            if in_period == out_period:
                continue
            for band_height in range(INPUT_HEIGHT_RANGE[0], INPUT_HEIGHT_RANGE[1] + 1):
                if band_height % in_period != 0 or band_height % out_period != 0:
                    continue
                if band_height // in_period < MIN_COMPLETE_SEGMENTS:
                    continue
                if band_height // out_period < MIN_COMPLETE_SEGMENTS:
                    continue
                out_height = band_height * out_period // in_period
                if not (OUTPUT_HEIGHT_RANGE[0] <= out_height <= OUTPUT_HEIGHT_RANGE[1]):
                    continue
                item = (in_period, out_period, band_height)
                if out_period > in_period:
                    expand.append(item)
                else:
                    compress.append(item)
    return expand, compress


EXPAND_CONFIGS, COMPRESS_CONFIGS = _build_valid_configs()


def _band_bounds_and_colors(grid: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    inner = grid[:, 1:-1]
    nonzero_colors = sorted({int(v) for v in inner.flat if int(v) != 0})
    if len(nonzero_colors) == 2:
        stripe_colors = tuple(nonzero_colors)
    else:
        all_colors = sorted({int(v) for v in inner.flat})
        assert len(all_colors) == 2, "expected exactly two stripe colors"
        stripe_colors = tuple(all_colors)

    mask = np.isin(inner, list(stripe_colors))
    band_rows = np.flatnonzero(mask.any(axis=1))
    assert band_rows.size > 0, "stripe band must exist"
    top = int(band_rows[0])
    bottom = int(band_rows[-1])
    for row in range(top, bottom + 1):
        row_vals = {int(v) for v in inner[row]}
        assert len(row_vals) == 1, "each stripe row must use exactly one color"
        assert row_vals <= set(stripe_colors), "band rows must be monochrome stripes"
        assert row_vals, "band row cannot be empty"
    return top, bottom, stripe_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch or compress horizontal stripes using the frequency bars on the left and right edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3, "expected a 2D grid with two edge bars"

    top, bottom, stripe_colors = _band_bounds_and_colors(g)
    band_height = int(bottom - top + 1)

    stripe_start = int(g[top, 1])
    left_start = int(g[top, 0])
    right_start = int(g[top, -1])

    left_period = _measure_period(g[top : bottom + 1, 0])
    right_period = _measure_period(g[top : bottom + 1, -1])
    input_period = _measure_period(g[top : bottom + 1, 1])

    assert input_period == left_period, "left bar must encode the input stripe width"
    assert band_height % input_period == 0, "input stripe band must be an integer number of runs"

    out_band_height = band_height * right_period // input_period
    assert out_band_height > 0 and band_height * right_period % input_period == 0

    top_pad = int(top)
    bottom_pad = int(g.shape[0] - 1 - bottom)
    out = _render_grid(
        band_height=out_band_height,
        inner_width=int(g.shape[1] - 2),
        left_period=left_period,
        stripe_period=right_period,
        right_period=right_period,
        colors=stripe_colors,
        stripe_start=stripe_start,
        left_start=left_start,
        right_start=right_start,
        top_pad=top_pad,
        bottom_pad=bottom_pad,
    )
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray:
    branch_pool = EXPAND_CONFIGS if rng.random() < 0.5 else COMPRESS_CONFIGS
    in_period, out_period, band_height = rng.choice(branch_pool)
    inner_width = rng.randint(*INNER_WIDTH_RANGE)
    colors = tuple(rng.sample(range(1, 10), 2))
    stripe_start = rng.choice(colors)
    left_start = rng.choice(colors)
    right_start = rng.choice(colors)

    inp = _render_grid(
        band_height=band_height,
        inner_width=inner_width,
        left_period=in_period,
        stripe_period=in_period,
        right_period=out_period,
        colors=colors,
        stripe_start=stripe_start,
        left_start=left_start,
        right_start=right_start,
    )

    expected = _render_grid(
        band_height=band_height * out_period // in_period,
        inner_width=inner_width,
        left_period=in_period,
        stripe_period=out_period,
        right_period=out_period,
        colors=colors,
        stripe_start=stripe_start,
        left_start=left_start,
        right_start=right_start,
    )
    assert np.array_equal(main(inp), expected)
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
