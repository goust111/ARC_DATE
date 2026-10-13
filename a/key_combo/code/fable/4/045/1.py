from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
BASE_FACTORS = (2, 3, 4)
PROTOTYPE_SIDE_MIN = 4
PROTOTYPE_SIDE_MAX = 7
MAX_GRID_SIDE = 30
MAX_ATTEMPTS = 300


def _group_equal_slices(grid: np.ndarray, axis: int) -> tuple[list[int], list[int]]:
    length = int(grid.shape[axis])
    starts: list[int] = []
    widths: list[int] = []
    i = 0
    while i < length:
        j = i + 1
        current = grid[i, :] if axis == 0 else grid[:, i]
        while j < length:
            candidate = grid[j, :] if axis == 0 else grid[:, j]
            if not np.array_equal(candidate, current):
                break
            j += 1
        starts.append(i)
        widths.append(j - i)
        i = j
    return starts, widths


def _compress_grid(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    row_starts, row_widths = _group_equal_slices(grid, axis=0)
    row_compressed = grid[row_starts, :]
    col_starts, col_widths = _group_equal_slices(row_compressed, axis=1)
    compressed = row_compressed[:, col_starts]
    return compressed, row_widths, col_widths


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a doubly run-expanded pattern and recolor only doubly anomalous band intersections red.

    Input structure:
    - The input grid is formed by expanding a smaller prototype: each prototype row is copied into a horizontal band, and each prototype column is copied into a vertical band.
    - Adjacent prototype rows differ, adjacent prototype columns differ, and the hidden prototype itself contains no red.
    - Most row-band heights and column-band widths share one common baseline expansion factor, while a few bands use different widths.

    Rule:
    - Merge each maximal run of identical adjacent rows and identical adjacent columns to recover the compressed prototype.
    - Collect all recovered row-band heights and column-band widths, and use the most frequent width as the baseline factor.
    - Recolor a cell red if and only if its recovered row comes from an anomalous row band and its recovered column comes from an anomalous column band; keep every other compressed cell unchanged.

    Key insight:
    - The visible duplication only hides the prototype; the true signal is carried by the band widths, and only row-column anomaly intersections survive into the output.
    """
    g = np.asarray(grid, dtype=int)
    compressed, row_widths, col_widths = _compress_grid(g)

    baseline = Counter(row_widths + col_widths).most_common(1)[0][0]
    bad_rows = [idx for idx, width in enumerate(row_widths) if width != baseline]
    bad_cols = [idx for idx, width in enumerate(col_widths) if width != baseline]

    out = compressed.copy()
    if bad_rows and bad_cols:
        out[np.ix_(bad_rows, bad_cols)] = RED
    return out


def _sample_nonrepeating_codes(length: int, modulus: int) -> list[int]:
    codes = [random.randrange(modulus)]
    while len(codes) < length:
        choices = [value for value in range(modulus) if value != codes[-1]]
        codes.append(random.choice(choices))
    return codes


def _make_prototype(height: int, width: int) -> np.ndarray:
    color_count = random.randint(3, min(5, len(INPUT_COLORS)))
    palette = random.sample(INPUT_COLORS, color_count)
    row_codes = _sample_nonrepeating_codes(height, color_count)
    col_codes = _sample_nonrepeating_codes(width, color_count)

    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r, c] = palette[(row_codes[r] + col_codes[c]) % color_count]
    return grid


def _adjacent_rows_and_cols_differ(grid: np.ndarray) -> bool:
    for r in range(grid.shape[0] - 1):
        if np.array_equal(grid[r, :], grid[r + 1, :]):
            return False
    for c in range(grid.shape[1] - 1):
        if np.array_equal(grid[:, c], grid[:, c + 1]):
            return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    base = random.choice(BASE_FACTORS)
    proto_h = random.randint(PROTOTYPE_SIDE_MIN, PROTOTYPE_SIDE_MAX)
    proto_w = random.randint(PROTOTYPE_SIDE_MIN, PROTOTYPE_SIDE_MAX)

    row_anom_count = random.randint(1, min(2, proto_h - 1))
    col_anom_count = random.randint(1, min(2, proto_w - 1))
    anomaly_pool = [width for width in range(1, 6) if width != base]
    total_anomalies = row_anom_count + col_anom_count
    if total_anomalies > len(anomaly_pool):
        return None

    anomaly_values = random.sample(anomaly_pool, total_anomalies)
    row_indices = sorted(random.sample(range(proto_h), row_anom_count))
    col_indices = sorted(random.sample(range(proto_w), col_anom_count))

    row_widths = [base] * proto_h
    col_widths = [base] * proto_w
    for idx, width in zip(row_indices, anomaly_values[:row_anom_count]):
        row_widths[idx] = width
    for idx, width in zip(col_indices, anomaly_values[row_anom_count:]):
        col_widths[idx] = width

    if sum(row_widths) > MAX_GRID_SIDE or sum(col_widths) > MAX_GRID_SIDE:
        return None

    width_counts = Counter(row_widths + col_widths)
    baseline_count = width_counts[base]
    strongest_nonbaseline = max(
        (count for width, count in width_counts.items() if width != base),
        default=0,
    )
    if baseline_count <= strongest_nonbaseline:
        return None

    prototype = _make_prototype(proto_h, proto_w)
    if np.any(prototype == RED) or not _adjacent_rows_and_cols_differ(prototype):
        return None

    expanded = np.repeat(prototype, row_widths, axis=0)
    expanded = np.repeat(expanded, col_widths, axis=1)

    expected = prototype.copy()
    expected[np.ix_(row_indices, col_indices)] = RED

    if not np.array_equal(main(expanded), expected):
        return None
    return expanded, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
