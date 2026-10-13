import math
import random
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
PATTERN_PALETTE = (1, 4, 5, 6, 7, 8, 9)

HEIGHT_RANGE = (6, 16)
WIDTH_RANGE = (8, 20)
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the last-column control strip to swap the two foreground colors of selected rows, then remove that strip.

    Input structure:
    - The background color is black.
    - The first W-1 columns form the main pattern area; each row contains exactly
      two non-black colors and at least one black cell.
    - The last column is a control strip whose cells are green or red.

    Rule:
    - Remove the final control column in the output.
    - A green control cell means the corresponding row of the main pattern area is
      kept unchanged.
    - A red control cell means the corresponding row swaps its two non-black
      colors while black background cells stay fixed.
    - The swap is performed independently per row because different rows can use
      different foreground color pairs.

    Key insight:
    - The last column is metadata rather than picture content, and a red signal
      means "exchange the two foreground colors in this row."
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a control column"

    pattern = g[:, :-1].copy()
    control = g[:, -1]
    assert np.all(np.isin(control, (RED, GREEN))), "control column must be red/green only"

    for r in range(pattern.shape[0]):
        row = pattern[r]
        colors = [int(color) for color in np.unique(row) if int(color) != BLACK]
        assert len(colors) == 2, "each row must contain exactly two foreground colors"
        assert np.any(row == BLACK), "each row must contain some black background"

        if int(control[r]) == RED:
            first, second = colors
            first_mask = row == first
            second_mask = row == second
            pattern[r, first_mask] = second
            pattern[r, second_mask] = first

    return pattern


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _wave_offset(row: int, period: int, amplitude: int, phase: int) -> int:
    if amplitude <= 0:
        return 0
    angle = 2.0 * math.pi * (row + phase) / period
    return int(round(amplitude * math.sin(angle)))


def _single_band_labels(height: int, width: int) -> np.ndarray | None:
    max_band_width = min(8, width - 2)
    if max_band_width < 3:
        return None

    band_width = random.randint(3, max_band_width)
    split = random.randint(1, band_width - 1)
    base = random.randint(0, width - band_width)
    drift = random.choice((-1, 0, 1))
    amplitude = min(3, width - band_width)
    period = random.randint(3, max(4, height))
    phase = random.randint(0, period - 1)

    labels = np.zeros((height, width), dtype=int)
    for row in range(height):
        start = base + drift * (row - height // 2) + _wave_offset(row, period, amplitude, phase)
        start = _clamp(start, 0, width - band_width)
        labels[row, start : start + split] = 1
        labels[row, start + split : start + band_width] = 2
    return labels


def _double_windows_labels(height: int, width: int) -> np.ndarray | None:
    max_window = min(4, width // 3)
    if max_window < 1:
        return None

    left_width = random.randint(1, max_window)
    right_width = random.randint(1, max_window)
    min_gap = random.randint(1, max(1, width // 5))
    if left_width + right_width + min_gap >= width:
        return None

    left_base = random.randint(0, width - (left_width + right_width + min_gap))
    right_base = random.randint(left_base + left_width + min_gap, width - right_width)
    spare_gap = right_base - (left_base + left_width) - min_gap
    amplitude = min(2, left_base, width - right_base - right_width, spare_gap // 2)
    period = random.randint(3, max(4, height))
    phase = random.randint(0, period - 1)

    labels = np.zeros((height, width), dtype=int)
    for row in range(height):
        offset = _wave_offset(row, period, amplitude, phase)
        left_start = left_base + offset
        right_start = right_base - offset
        labels[row, left_start : left_start + left_width] = 1
        labels[row, right_start : right_start + right_width] = 2
    return labels


def _tiled_labels(height: int, width: int) -> np.ndarray | None:
    a_len = random.randint(1, 3)
    b_len = random.randint(1, 3)
    max_gap = min(3, width - a_len - b_len)
    if max_gap < 1:
        return None

    gap_len = random.randint(1, max_gap)
    motif = np.array([1] * a_len + [2] * b_len + [0] * gap_len, dtype=int)
    step = random.choice((-2, -1, 1, 2))
    base_phase = random.randint(0, len(motif) - 1)

    labels = np.zeros((height, width), dtype=int)
    positions = np.arange(width, dtype=int)
    for row in range(height):
        phase = (base_phase + row * step) % len(motif)
        labels[row] = motif[(positions + phase) % len(motif)]
    return labels


def _make_labels(height: int, width: int) -> np.ndarray | None:
    family = random.choice((_single_band_labels, _double_windows_labels, _tiled_labels))
    labels = family(height, width)
    if labels is None:
        return None

    if random.random() < 0.5:
        labels = labels[:, ::-1].copy()
    if random.random() < 0.5:
        labels = labels[::-1, :].copy()

    for row in labels:
        values = set(map(int, row))
        if values != {0, 1, 2}:
            return None
    return labels


def _row_pairs(height: int) -> list[tuple[int, int]]:
    scene_color_count = random.randint(3, 5)
    scene_colors = random.sample(PATTERN_PALETTE, scene_color_count)
    pair_pool = list(combinations(scene_colors, 2))
    random.shuffle(pair_pool)

    offset = random.randrange(len(pair_pool))
    step = random.randrange(1, len(pair_pool) + 1)
    pairs: list[tuple[int, int]] = []
    for row in range(height):
        if random.random() < 0.65:
            idx = (offset + row * step) % len(pair_pool)
        else:
            idx = random.randrange(len(pair_pool))
        pairs.append(tuple(map(int, pair_pool[idx])))

    if len(pair_pool) > 1 and len({pair for pair in pairs}) == 1:
        pairs[-1] = tuple(map(int, pair_pool[1]))
    return pairs


def _control_column(height: int) -> np.ndarray:
    red_count = random.choice((height // 2, (height + 1) // 2))
    values = [RED] * red_count + [GREEN] * (height - red_count)
    random.shuffle(values)
    return np.array(values, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    labels = _make_labels(height, width)
    if labels is None:
        return None

    pairs = _row_pairs(height)
    pattern = np.full((height, width), BLACK, dtype=int)
    for row, (first, second) in enumerate(pairs):
        pattern[row, labels[row] == 1] = first
        pattern[row, labels[row] == 2] = second

    control = _control_column(height)
    grid = np.concatenate([pattern, control[:, None]], axis=1)

    if grid.shape[1] > 30:
        return None
    if not np.any(control == RED) or not np.any(control == GREEN):
        return None

    transformed = main(grid)
    if transformed.shape != (height, width):
        return None
    if np.array_equal(transformed, pattern):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
