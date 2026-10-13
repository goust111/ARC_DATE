from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _row_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    rows = np.flatnonzero(mask.any(axis=1))
    assert len(rows) > 0, "expected at least one encoded row"

    blocks: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for r in map(int, rows[1:]):
        if r != prev + 1:
            blocks.append((start, prev))
            start = r
        prev = r
    blocks.append((start, prev))
    return blocks


def _trim_cols(grid: np.ndarray, bg: int) -> np.ndarray:
    cols = np.flatnonzero((grid != bg).any(axis=0))
    assert len(cols) > 0, "encoded row must contain colored pixels"
    return grid[:, int(cols[0]) : int(cols[-1]) + 1]


def _decode_band(band: np.ndarray, bg: int) -> list[int]:
    patch = _trim_cols(band, bg)
    h, w = patch.shape
    assert h >= 1 and w >= 2, "encoded row is too small"
    assert w % 3 == 2, "encoded row width must be 3*k-1"

    unit_count = (w + 1) // 3
    decoded: list[int] = []

    for i in range(unit_count):
        c0 = 3 * i
        c1 = c0 + 1

        indicator_col = patch[:, c0]
        assert int(indicator_col[-1]) != bg, "indicator pixel must sit on the bottom row"
        assert np.all(indicator_col[:-1] == bg), "indicator column must contain one pixel"
        color = int(indicator_col[-1])

        bar_col = patch[:, c1]
        nz = np.flatnonzero(bar_col != bg)
        assert len(nz) > 0, "length bar must be present"
        assert int(nz[-1]) == h - 1, "length bar must touch the bottom row"
        expected_nz = np.arange(h - len(nz), h)
        assert np.array_equal(nz, expected_nz), "length bar must be vertically contiguous"
        assert np.all(bar_col[nz] == color), "indicator and length bar must share a color"

        decoded.extend([color] * len(nz))

        if i < unit_count - 1:
            assert np.all(patch[:, c0 + 2] == bg), "units must be separated by one background column"

    expected = np.full_like(patch, bg)
    cursor = 0
    for i in range(unit_count):
        color = int(patch[-1, cursor])
        bar_len = int(np.count_nonzero(patch[:, cursor + 1] != bg))
        expected[-1, cursor] = color
        expected[h - bar_len :, cursor + 1] = color
        cursor += 3
    assert np.array_equal(patch, expected), "encoded row contains unexpected extra pixels"

    return decoded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each encoded row by reading color/height units and concatenating the runs.

    Each input row-band is made of 2-column units separated by one background
    column. The left column is a single bottom-row color indicator. The right
    column is a same-color vertical bar whose height encodes the run length.
    Decoding every band from top to bottom yields the output rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    blocks = _row_blocks(g != bg)

    rows = [_decode_band(g[r0 : r1 + 1, :], bg) for r0, r1 in blocks]
    widths = {len(row) for row in rows}
    assert len(widths) == 1, "all decoded rows must have the same width"

    return np.array(rows, dtype=int)


MAX_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))
MIN_OUTPUT_HEIGHT = 2
MAX_OUTPUT_HEIGHT = 5
MIN_OUTPUT_WIDTH = 6
MAX_OUTPUT_WIDTH = 18
MIN_UNITS = 2
MAX_UNITS = 6
MAX_RUN = 6
MIN_V_GAP = 1
MAX_V_GAP = 3
MIN_H_MARGIN = 1
MAX_H_MARGIN = 6
MIN_V_MARGIN = 1
MAX_V_MARGIN = 4


def _bounded_composition(total: int, parts: int, max_each: int) -> list[int] | None:
    if parts <= 0 or total < parts or total > parts * max_each:
        return None

    values: list[int] = []
    remaining = total
    for i in range(parts):
        parts_left = parts - i - 1
        lo = max(1, remaining - parts_left * max_each)
        hi = min(max_each, remaining - parts_left)
        if lo > hi:
            return None
        values.append(random.randint(lo, hi))
        remaining -= values[-1]

    random.shuffle(values)
    return values


def _sample_colors(count: int, palette: list[int]) -> list[int]:
    colors: list[int] = []
    prev = None
    for _ in range(count):
        choices = [c for c in palette if c != prev]
        color = random.choice(choices)
        colors.append(color)
        prev = color
    return colors


def _encode_row(lengths: list[int], colors: list[int], bg: int) -> tuple[np.ndarray, np.ndarray]:
    height = max(lengths)
    width = 3 * len(lengths) - 1
    band = np.full((height, width), bg, dtype=int)

    decoded: list[int] = []
    c = 0
    for i, (run_len, color) in enumerate(zip(lengths, colors)):
        band[-1, c] = color
        band[height - run_len :, c + 1] = color
        decoded.extend([color] * run_len)
        c += 2
        if i < len(lengths) - 1:
            c += 1

    return band, np.array(decoded, dtype=int)


def _sample_row_spec(output_width: int) -> list[int] | None:
    min_units = max(MIN_UNITS, (output_width + MAX_RUN - 1) // MAX_RUN)
    max_units = min(MAX_UNITS, output_width)
    if min_units > max_units:
        return None

    unit_count = random.randint(min_units, max_units)
    lengths = _bounded_composition(output_width, unit_count, MAX_RUN)
    if lengths is None:
        return None

    return lengths


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(ALL_COLORS)
    palette = [c for c in ALL_COLORS if c != bg]

    out_h = random.randint(MIN_OUTPUT_HEIGHT, MAX_OUTPUT_HEIGHT)
    out_w = random.randint(MIN_OUTPUT_WIDTH, MAX_OUTPUT_WIDTH)

    row_bands: list[np.ndarray] = []
    output_rows: list[np.ndarray] = []

    for _ in range(out_h):
        lengths = _sample_row_spec(out_w)
        if lengths is None:
            return None
        colors = _sample_colors(len(lengths), palette)
        band, decoded = _encode_row(lengths, colors, bg)
        row_bands.append(band)
        output_rows.append(decoded)

    band_widths = [band.shape[1] for band in row_bands]
    band_heights = [band.shape[0] for band in row_bands]

    left_margins = [random.randint(MIN_H_MARGIN, MAX_H_MARGIN) for _ in row_bands]
    right_margins = [random.randint(MIN_H_MARGIN, MAX_H_MARGIN) for _ in row_bands]
    grid_w = max(left + width + right for left, width, right in zip(left_margins, band_widths, right_margins))

    top_margin = random.randint(MIN_V_MARGIN, MAX_V_MARGIN)
    bottom_margin = random.randint(MIN_V_MARGIN, MAX_V_MARGIN)
    gaps = [random.randint(MIN_V_GAP, MAX_V_GAP) for _ in range(len(row_bands) - 1)]
    grid_h = top_margin + sum(band_heights) + sum(gaps) + bottom_margin

    inp = np.full((grid_h, grid_w), bg, dtype=int)

    r = top_margin
    for idx, (band, left) in enumerate(zip(row_bands, left_margins)):
        h, w = band.shape
        inp[r : r + h, left : left + w] = band
        r += h
        if idx < len(gaps):
            r += gaps[idx]

    out = np.vstack(output_rows)

    bg_count = int(np.count_nonzero(inp == bg))
    non_bg_count = int(inp.size - bg_count)
    if bg_count <= non_bg_count:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
