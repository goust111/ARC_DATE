from __future__ import annotations

import random

import numpy as np


YELLOW = 4
PALETTE = tuple(range(10))
INPUT_COLOR_CHOICES = tuple(color for color in PALETTE if color != YELLOW)
GRID_MIN = 8
GRID_MAX = 18
MAX_BANDS = 4
MAX_THICKNESS = 3
MAX_ATTEMPTS = 400
VISIBILITY_MODES = ("rows", "cols", "checker_rows", "checker_cols")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the two visible foreground colors as two independent full-span masks and
    reveal their overlap with yellow.

    In the generated family, one mask occupies whole columns and the other occupies
    whole rows. Cells where the row-mask and column-mask overlap may show either
    source color in the input, so the hidden second layer is recovered from the
    global row/column structure rather than the local pixel color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    assert YELLOW not in set(map(int, g.ravel()))

    full_rows = np.flatnonzero(np.all(g != bg, axis=1))
    full_cols = np.flatnonzero(np.all(g != bg, axis=0))
    h, w = g.shape
    assert 0 < len(full_rows) < h
    assert 0 < len(full_cols) < w

    row_only_cols = np.flatnonzero(np.any(g == bg, axis=0))
    col_only_rows = np.flatnonzero(np.any(g == bg, axis=1))
    assert len(row_only_cols) > 0
    assert len(col_only_rows) > 0

    row_only_values = set(map(int, np.unique(g[np.ix_(full_rows, row_only_cols)]))) - {bg}
    col_only_values = set(map(int, np.unique(g[np.ix_(col_only_rows, full_cols)]))) - {bg}
    assert len(row_only_values) == 1
    assert len(col_only_values) == 1
    row_color = row_only_values.pop()
    col_color = col_only_values.pop()
    assert row_color != col_color

    support = g != bg
    expected_support = np.zeros_like(support, dtype=bool)
    expected_support[full_rows, :] = True
    expected_support[:, full_cols] = True
    assert np.array_equal(support, expected_support)

    overlap_values = set(map(int, np.unique(g[np.ix_(full_rows, full_cols)]))) - {bg}
    assert overlap_values <= {row_color, col_color}

    out = np.full_like(g, bg)
    out[:, full_cols] = col_color
    out[full_rows, :] = row_color
    out[np.ix_(full_rows, full_cols)] = YELLOW
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_bands(length: int) -> list[tuple[int, int]] | None:
    max_count = min(MAX_BANDS, (length + 1) // 2)
    if max_count <= 0:
        return None

    for _ in range(80):
        count = random.randint(1, max_count)
        max_thickness = min(MAX_THICKNESS, length - count + 1)
        if max_thickness <= 0:
            continue
        thicknesses = [random.randint(1, max_thickness) for _ in range(count)]
        min_used = sum(thicknesses) + (count - 1)
        if min_used >= length:
            continue

        extras = _random_composition(length - min_used, count + 1)
        gaps = extras[:]
        for idx in range(1, count):
            gaps[idx] += 1

        bands: list[tuple[int, int]] = []
        pos = gaps[0]
        for idx, thickness in enumerate(thicknesses):
            bands.append((pos, thickness))
            pos += thickness + gaps[idx + 1]
        return bands
    return None


def _band_indices(bands: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for start, thickness in bands:
        indices.extend(range(start, start + thickness))
    return indices


def _paint_intersections(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    row_color: int,
    col_color: int,
    mode: str,
) -> None:
    for row_idx, (r0, rh) in enumerate(row_bands):
        for col_idx, (c0, cw) in enumerate(col_bands):
            if mode == "rows":
                visible = row_color
            elif mode == "cols":
                visible = col_color
            elif mode == "checker_rows":
                visible = row_color if (row_idx + col_idx) % 2 == 0 else col_color
            else:
                visible = col_color if (row_idx + col_idx) % 2 == 0 else row_color
            grid[r0 : r0 + rh, c0 : c0 + cw] = visible


def _build_pair() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    row_bands = _sample_bands(height)
    col_bands = _sample_bands(width)
    if row_bands is None or col_bands is None:
        return None

    row_indices = _band_indices(row_bands)
    col_indices = _band_indices(col_bands)
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None
    if len(row_indices) >= height or len(col_indices) >= width:
        return None

    union_area = len(row_indices) * width + len(col_indices) * height - len(row_indices) * len(col_indices)
    if union_area * 2 >= height * width:
        return None

    bg = random.choice(INPUT_COLOR_CHOICES)
    remaining = [color for color in INPUT_COLOR_CHOICES if color != bg]
    row_color, col_color = random.sample(remaining, 2)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    inp[:, col_indices] = col_color
    inp[row_indices, :] = row_color
    out[:, col_indices] = col_color
    out[row_indices, :] = row_color
    out[np.ix_(row_indices, col_indices)] = YELLOW

    mode = random.choice(VISIBILITY_MODES)
    _paint_intersections(inp, row_bands, col_bands, row_color, col_color, mode)

    if len(set(map(int, np.unique(inp))) - {bg}) != 2:
        return None
    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _build_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input grid")
