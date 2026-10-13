from __future__ import annotations

import random

import numpy as np


SOLID = "solid"
DASHED = "dashed"
STYLE_CHOICES = (SOLID, DASHED)
GRID_SHAPE_OPTIONS = ((2, 2), (2, 3), (3, 2))
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _classify_segment(segment: np.ndarray, bg: int, divider: int) -> str:
    arr = np.asarray(segment, dtype=int).ravel()
    assert arr.size >= 2, "boundary segment must have length at least two"
    assert np.all((arr == bg) | (arr == divider)), "boundary segment may contain only bg/divider colors"
    if np.all(arr == divider):
        return SOLID
    if np.any(arr == divider) and np.any(arr == bg) and np.all(arr[:-1] != arr[1:]):
        return DASHED
    raise AssertionError("boundary segment is neither solid nor dashed")


def _separator_rows_and_cols(grid: np.ndarray, bg: int, divider: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows = [
        r
        for r in range(h)
        if np.all((grid[r] == bg) | (grid[r] == divider)) and np.count_nonzero(grid[r] == divider) >= (w + 1) // 2
    ]
    cols = [
        c
        for c in range(w)
        if np.all((grid[:, c] == bg) | (grid[:, c] == divider)) and np.count_nonzero(grid[:, c] == divider) >= (h + 1) // 2
    ]
    assert len(rows) >= 3 and len(cols) >= 3, "expected framed stripe partition in both axes"
    assert rows[0] == 0 and rows[-1] == h - 1, "top/bottom frame rows missing"
    assert cols[0] == 0 and cols[-1] == w - 1, "left/right frame cols missing"
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "separator rows must leave non-empty regions"
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:])), "separator cols must leave non-empty regions"
    return rows, cols


def _detect_divider_color(grid: np.ndarray, bg: int) -> int:
    best_score = -1
    best_color = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        try:
            rows, cols = _separator_rows_and_cols(grid, bg, color)
        except AssertionError:
            continue
        score = len(rows) + len(cols)
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "failed to identify divider color"
    return int(best_color)


def _paint_diamond(
    out: np.ndarray,
    color: int,
    seed_r: int,
    seed_c: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> None:
    radius = min(seed_r - top, bottom - seed_r, seed_c - left, right - seed_c)
    for r in range(seed_r - radius, seed_r + radius + 1):
        span = radius - abs(r - seed_r)
        out[r, seed_c - span : seed_c + span + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a line or a diamond according to whether its region's long boundary dividers are solid or dashed.

    Input structure:
    - The background is the dominant color.
    - Grey one-cell divider lines form an outer frame plus full horizontal and vertical separators, partitioning the grid into rectangular bands.
    - Divider lines are either solid or dashed; dashed dividers alternate grey and background pixels.
    - Every rectangular region contains exactly one non-background, non-divider seed pixel.

    Rule:
    - Preserve the divider lines exactly.
    - For each region, compare its interior height and width to determine its long side direction.
    - If the pair of long-side dividers is solid, extend the seed into a full straight line parallel to those long sides and spanning the region interior.
    - If the pair of long-side dividers is dashed, fill the largest axis-aligned-inscribed diamond centered on the seed, stopping when the next growth step would cross the region boundary.

    Key insight:
    - The seed color does not choose the output shape; the decisive clue is the visual texture of the region's longer boundary pair, with solid boundaries meaning a full stripe and dashed boundaries meaning a diamond.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    divider = _detect_divider_color(g, bg)
    row_dividers, col_dividers = _separator_rows_and_cols(g, bg, divider)

    out = g.copy()
    for r0, r1 in zip(row_dividers, row_dividers[1:]):
        for c0, c1 in zip(col_dividers, col_dividers[1:]):
            top = r0 + 1
            bottom = r1 - 1
            left = c0 + 1
            right = c1 - 1
            assert top <= bottom and left <= right, "region interior must be non-empty"
            region = g[top : bottom + 1, left : right + 1]
            assert not np.any(region == divider), "divider color may not appear inside a region"

            seed_coords = np.argwhere((region != bg) & (region != divider))
            assert seed_coords.shape[0] == 1, "each region must contain exactly one seed"
            seed_r = top + int(seed_coords[0, 0])
            seed_c = left + int(seed_coords[0, 1])
            seed_color = int(g[seed_r, seed_c])

            height = bottom - top + 1
            width = right - left + 1
            assert height != width, "regions must have a unique long-side direction"

            if height > width:
                left_style = _classify_segment(g[top : bottom + 1, c0], bg, divider)
                right_style = _classify_segment(g[top : bottom + 1, c1], bg, divider)
                assert left_style == right_style, "paired long boundaries must share one style"
                if left_style == SOLID:
                    out[top : bottom + 1, seed_c] = seed_color
                else:
                    _paint_diamond(out, seed_color, seed_r, seed_c, top, bottom, left, right)
            else:
                top_style = _classify_segment(g[r0, left : right + 1], bg, divider)
                bottom_style = _classify_segment(g[r1, left : right + 1], bg, divider)
                assert top_style == bottom_style, "paired long boundaries must share one style"
                if top_style == SOLID:
                    out[seed_r, left : right + 1] = seed_color
                else:
                    _paint_diamond(out, seed_color, seed_r, seed_c, top, bottom, left, right)

    return out


def _draw_horizontal_divider(grid: np.ndarray, row: int, style: str, phase: int, color: int) -> None:
    if style == SOLID:
        grid[row, :] = color
        return
    for c in range(grid.shape[1]):
        if (c + phase) % 2 == 0:
            grid[row, c] = color


def _draw_vertical_divider(grid: np.ndarray, col: int, style: str, phase: int, color: int) -> None:
    if style == SOLID:
        grid[:, col] = color
        return
    for r in range(grid.shape[0]):
        if (r + phase) % 2 == 0:
            grid[r, col] = color


def _axis_sizes(count: int, small: int, large: int) -> list[int]:
    sizes = [small, large]
    if count == 3:
        sizes.append(random.choice((small, large)))
    random.shuffle(sizes)
    return sizes


def _separator_positions(lengths: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in lengths:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _apply_region_growth(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    seed_r: int,
    seed_c: int,
    color: int,
    style: str,
    long_is_vertical: bool,
) -> None:
    if style == SOLID:
        if long_is_vertical:
            grid[top : bottom + 1, seed_c] = color
        else:
            grid[seed_r, left : right + 1] = color
        return
    _paint_diamond(grid, color, seed_r, seed_c, top, bottom, left, right)


def _symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    if code < 4:
        return np.rot90(grid, code).copy()
    return np.fliplr(np.rot90(grid, code - 4)).copy()


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    band_rows, band_cols = random.choice(GRID_SHAPE_OPTIONS)

    w_small = random.randint(3, 4)
    h_small = random.randint(w_small + 1, 6)
    h_large = random.randint(h_small + 1, 8)
    w_large = random.randint(h_large + 1, 10)

    row_sizes = _axis_sizes(band_rows, h_small, h_large)
    col_sizes = _axis_sizes(band_cols, w_small, w_large)
    h = sum(row_sizes) + band_rows + 1
    w = sum(col_sizes) + band_cols + 1
    if h > 30 or w > 30:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg, divider = palette[:2]
    seed_colors = palette[2 : 2 + band_rows * band_cols]
    if len(seed_colors) != band_rows * band_cols:
        return None

    horizontal_style = random.choice(STYLE_CHOICES)
    vertical_style = random.choice(STYLE_CHOICES)
    row_phases = [random.randint(0, 1) for _ in range(band_rows + 1)]
    col_phases = [random.randint(0, 1) for _ in range(band_cols + 1)]

    inp = np.full((h, w), bg, dtype=int)
    row_dividers = _separator_positions(row_sizes)
    col_dividers = _separator_positions(col_sizes)
    for idx, row in enumerate(row_dividers):
        _draw_horizontal_divider(inp, row, horizontal_style, row_phases[idx], divider)
    for idx, col in enumerate(col_dividers):
        _draw_vertical_divider(inp, col, vertical_style, col_phases[idx], divider)

    out = inp.copy()
    color_iter = iter(seed_colors)
    solid_regions = 0
    dashed_regions = 0

    for r0, r1 in zip(row_dividers, row_dividers[1:]):
        for c0, c1 in zip(col_dividers, col_dividers[1:]):
            top = r0 + 1
            bottom = r1 - 1
            left = c0 + 1
            right = c1 - 1
            height = bottom - top + 1
            width = right - left + 1
            assert height != width

            long_is_vertical = height > width
            style = vertical_style if long_is_vertical else horizontal_style
            if style == SOLID:
                solid_regions += 1
            else:
                dashed_regions += 1

            candidates = []
            for seed_r in range(top, bottom + 1):
                for seed_c in range(left, right + 1):
                    if style == DASHED:
                        margin = min(seed_r - top, bottom - seed_r, seed_c - left, right - seed_c)
                        if margin < 1:
                            continue
                    candidates.append((seed_r, seed_c))
            if not candidates:
                return None

            seed_r, seed_c = random.choice(candidates)
            color = next(color_iter)
            inp[seed_r, seed_c] = color
            out[seed_r, seed_c] = color
            _apply_region_growth(out, top, bottom, left, right, seed_r, seed_c, color, style, long_is_vertical)

    if solid_regions == 0 or dashed_regions == 0:
        return None

    code = random.randrange(8)
    return _symmetry(inp, code), _symmetry(out, code)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _build_case()
        if candidate is None:
            continue
        inp, expected = candidate
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid stripe-divider growth puzzle")
