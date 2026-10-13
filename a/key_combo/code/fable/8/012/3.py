from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

LEFT_WIDTH_CHOICES = (5, 7)
BAND_COUNT_RANGE = (3, 6)
BAND_MIN_HEIGHT = 2
GRID_HEIGHT_EXTRA_RANGE = (0, 5)
HORIZONTAL_STEP_RANGE = (2, 5)
VERTICAL_STEP_RANGE = (2, 4)
RIGHT_PAD_MIN = 1
RIGHT_PAD_MAX_EXTRA = 5
RETRY_LIMIT = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _compress_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _majority_color(row: np.ndarray) -> int:
    counts = np.bincount(np.asarray(row, dtype=int), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1, "each left-band row must have a unique majority color"
    return int(winners[0])


def _separator_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg or not np.all(grid[:, c] == color):
            continue
        if c == 0 or c == w - 1:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if np.any(left == bg):
            continue
        if not np.any(right == bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _paint_segment(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        if hi - lo > 1:
            out[r0, lo + 1 : hi] = color
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        if hi - lo > 1:
            out[lo + 1 : hi, c0] = color
        return
    raise AssertionError("consecutive stations must share a row or a column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left jagged color bands into an itinerary and use it to connect the matching stations on the right.

    Input structure:
    - A full-height separator column splits the grid into a narrow left band zone and a wider right field.
    - The left zone is fully covered by colored horizontal bands whose boundaries are slightly jagged across columns.
    - The right field is mostly black and contains one single-pixel station for each left-band color.

    Rule:
    - Read the left zone row by row, taking the majority color of each row and compressing consecutive duplicate rows into one top-to-bottom color sequence.
    - In that order, connect each consecutive station pair by filling the cells strictly between them with the earlier station's color.
    - Keep the stations, the left bands, and the separator column unchanged.

    Key insight:
    - The right-side geometry does not determine the visiting order; the left jagged panel must first be reduced to an ordered color list that acts as the travel plan.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    sep_col = _separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size > 0 and right.size > 0

    sequence = _compress_runs([_majority_color(row) for row in left])
    assert len(sequence) >= 2
    assert len(sequence) == len(set(sequence)), "each band color should appear in one contiguous block"

    right_colors = set(map(int, right[right != bg]))
    assert right_colors == set(sequence), "right field stations must match the compressed color sequence"
    assert int(np.count_nonzero(right != bg)) == len(sequence), "each station must be a single pixel"

    stations: dict[int, tuple[int, int]] = {}
    for color in sequence:
        coords = np.argwhere(right == color)
        assert len(coords) == 1, f"expected one station for color {color}"
        r, c = map(int, coords[0])
        stations[color] = (r, c + sep_col + 1)

    out = g.copy()
    for color0, color1 in zip(sequence, sequence[1:]):
        _paint_segment(out, stations[color0], stations[color1], color0)
    return out


def _random_partition(total: int, count: int, min_size: int) -> list[int]:
    assert total >= count * min_size
    parts = []
    remaining = total
    for i in range(count - 1):
        lo = min_size
        hi = remaining - min_size * (count - i - 1)
        part = random.randint(lo, hi)
        parts.append(part)
        remaining -= part
    parts.append(remaining)
    return parts


def _station_path_local_coords(count: int, start_horizontal: bool) -> list[tuple[int, int]]:
    coords = [(0, 0)]
    r = c = 0
    for i in range(count - 1):
        horizontal = start_horizontal if i % 2 == 0 else not start_horizontal
        if horizontal:
            c += random.randint(*HORIZONTAL_STEP_RANGE)
        else:
            r += random.randint(*VERTICAL_STEP_RANGE)
        coords.append((r, c))
    return coords


def _build_left_bands(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    band_heights = _random_partition(height, len(colors), BAND_MIN_HEIGHT)
    base_boundaries = np.cumsum(band_heights[:-1]).tolist()

    core_count = width // 2 + 1
    core_start = (width - core_count) // 2
    core_stop = core_start + core_count
    jagged_used = False
    left = np.zeros((height, width), dtype=int)

    for c in range(width):
        if core_start <= c < core_stop:
            boundaries = list(base_boundaries)
        else:
            boundaries = []
            prev = 0
            for i, base in enumerate(base_boundaries):
                lo = max(prev + 1, base - 1)
                hi = min(height - (len(base_boundaries) - i), base + 1)
                if lo > hi:
                    return None
                boundary = random.randint(lo, hi)
                boundaries.append(boundary)
                prev = boundary
            if boundaries != base_boundaries:
                jagged_used = True

        starts = [0] + boundaries
        stops = boundaries + [height]
        for color, r0, r1 in zip(colors, starts, stops):
            left[r0:r1, c] = color

    if not jagged_used:
        return None
    return left


def _render_input(
    height: int,
    left_width: int,
    right_width: int,
    left: np.ndarray,
    colors: list[int],
    local_stations: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    grid = np.full((height, left_width + 1 + right_width), BACKGROUND, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = SEPARATOR

    used_h = max(r for r, _ in local_stations) + 1
    used_w = max(c for _, c in local_stations) + 1
    top_pad = random.randint(RIGHT_PAD_MIN, height - used_h - RIGHT_PAD_MIN)
    left_pad = random.randint(RIGHT_PAD_MIN, right_width - used_w - RIGHT_PAD_MIN)

    stations: dict[int, tuple[int, int]] = {}
    for color, (r, c) in zip(colors, local_stations):
        rr = top_pad + r
        cc = left_width + 1 + left_pad + c
        grid[rr, cc] = color
        stations[color] = (rr, cc)
    return grid, stations


def _expected_output(inp: np.ndarray, colors: list[int], stations: dict[int, tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for color0, color1 in zip(colors, colors[1:]):
        _paint_segment(out, stations[color0], stations[color1], color0)
    return out


def _generate_attempt() -> np.ndarray | None:
    band_count = random.randint(*BAND_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, band_count)
    left_width = random.choice(LEFT_WIDTH_CHOICES)

    local_stations = _station_path_local_coords(
        band_count,
        start_horizontal=bool(random.getrandbits(1)),
    )
    path_h = max(r for r, _ in local_stations) + 1
    path_w = max(c for _, c in local_stations) + 1

    min_height = max(path_h + 2, band_count * BAND_MIN_HEIGHT)
    max_height = min(24, min_height + GRID_HEIGHT_EXTRA_RANGE[1])
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    left = _build_left_bands(height, left_width, colors)
    if left is None:
        return None

    min_right_width = path_w + 2
    max_right_width = min(22, min_right_width + RIGHT_PAD_MAX_EXTRA)
    if min_right_width > max_right_width:
        return None
    right_width = random.randint(min_right_width, max_right_width)

    total_width = left_width + 1 + right_width
    if total_width > 30:
        return None

    inp, stations = _render_input(height, left_width, right_width, left, colors, local_stations)
    expected = _expected_output(inp, colors, stations)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
