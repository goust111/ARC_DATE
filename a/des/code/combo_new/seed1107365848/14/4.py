from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GENERATION_RETRIES = 500
PARTITION_RETRIES = 100
BAND_COUNT_RANGE = (1, 4)
REGION_SPAN_RANGE = (2, 7)
OTHER_AXIS_RANGE = (6, 22)
PAIR_COUNT_CAP = 6


def _band_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color in (BG, GREY):
            continue
        if bool(np.all(grid[r] == color)):
            rows.append(r)
    return rows


def _band_cols(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color in (BG, GREY):
            continue
        if bool(np.all(grid[:, c] == color)):
            cols.append(c)
    return cols


def _pair_candidates(
    points: list[tuple[int, int]],
    band_rows: list[int],
    band_cols: list[int],
) -> dict[tuple[int, int], tuple[int, int]]:
    nearest: dict[tuple[int, int], tuple[int, int]] = {}
    for point in points:
        pr, pc = point
        candidates: list[tuple[int, tuple[int, int]]] = []
        for other in points:
            if other == point:
                continue
            qr, qc = other
            if pr == qr:
                crossed = [c for c in band_cols if min(pc, qc) < c < max(pc, qc)]
                if len(crossed) == 1:
                    candidates.append((abs(pc - qc), other))
            elif pc == qc:
                crossed = [r for r in band_rows if min(pr, qr) < r < max(pr, qr)]
                if len(crossed) == 1:
                    candidates.append((abs(pr - qr), other))
        assert candidates, "every grey pixel must have an aligned partner across exactly one band"
        candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        best_dist = candidates[0][0]
        best_matches = [other for dist, other in candidates if dist == best_dist]
        assert len(best_matches) == 1, "grey pairing must be unambiguous"
        nearest[point] = best_matches[0]
    return nearest


def _find_pairs(
    points: list[tuple[int, int]],
    band_rows: list[int],
    band_cols: list[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    nearest = _pair_candidates(points, band_rows, band_cols)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for point, other in nearest.items():
        if nearest.get(other) == point and point < other:
            pairs.append((point, other))
    assert len(pairs) * 2 == len(points), "grey pixels must decompose into mutual nearest pairs"
    return pairs


def _paint_segment(
    out: np.ndarray,
    grid: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
    band_rows: list[int],
    band_cols: list[int],
) -> None:
    (r1, c1), (r2, c2) = pair
    if r1 == r2:
        crossed = [c for c in band_cols if min(c1, c2) < c < max(c1, c2)]
        assert len(crossed) == 1
        color = int(grid[r1, crossed[0]])
        out[r1, min(c1, c2) : max(c1, c2) + 1] = color
    else:
        crossed = [r for r in band_rows if min(r1, r2) < r < max(r1, r2)]
        assert len(crossed) == 1
        color = int(grid[crossed[0], c1])
        out[min(r1, r2) : max(r1, r2) + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every paired grey endpoint across its separator band using the band's own color.

    Input structure:
    - The background is Black.
    - Several one-cell-thick monochrome separator bands span the full grid as whole rows or whole columns, and the bands do not intersect.
    - Grey singleton pixels sit inside the black regions; each grey pixel has exactly one partner in the same row or the same column, and their segment crosses exactly one separator band.

    Rule:
    - Detect separator bands as uniform non-black, non-grey full rows or full columns.
    - Pair grey pixels by mutual nearest aligned partner whose connecting segment crosses exactly one separator band.
    - For each pair, draw the inclusive straight segment between the two grey endpoints.
    - Recolor that whole segment, including both endpoints, with the color of the crossed separator band; leave every other cell unchanged.

    Key insight:
    - The grey pixels only mark which two region interiors should be bridged; the separator they cross determines the bridge color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    band_rows = _band_rows(g)
    band_cols = _band_cols(g)
    grey_points = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    if not grey_points:
        return out

    for pair in _find_pairs(grey_points, band_rows, band_cols):
        _paint_segment(out, g, pair, band_rows, band_cols)
    return out


def _sample_partition(band_count: int) -> list[int] | None:
    for _ in range(PARTITION_RETRIES):
        spans = [random.randint(*REGION_SPAN_RANGE) for _ in range(band_count + 1)]
        if sum(spans) + band_count <= 30:
            return spans
    return None


def _gen_vertical_case() -> tuple[np.ndarray, np.ndarray] | None:
    band_count = random.randint(*BAND_COUNT_RANGE)
    region_widths = _sample_partition(band_count)
    if region_widths is None:
        return None

    height = random.randint(*OTHER_AXIS_RANGE)
    pair_count = random.randint(2, min(PAIR_COUNT_CAP, height))
    band_colors = random.sample(PALETTE, band_count)

    width = sum(region_widths) + band_count
    grid = np.zeros((height, width), dtype=int)

    regions: list[tuple[int, int]] = []
    col = 0
    for idx, region_width in enumerate(region_widths):
        regions.append((col, col + region_width - 1))
        col += region_width
        if idx < band_count:
            grid[:, col] = band_colors[idx]
            col += 1

    rows = random.sample(range(height), pair_count)
    records: list[tuple[int, int, int, int]] = []
    for row in rows:
        band_idx = random.randrange(band_count)
        left_col = random.randint(*regions[band_idx])
        right_col = random.randint(*regions[band_idx + 1])
        grid[row, left_col] = GREY
        grid[row, right_col] = GREY
        records.append((row, left_col, right_col, band_colors[band_idx]))

    expected = grid.copy()
    for row, left_col, right_col, color in records:
        expected[row, min(left_col, right_col) : max(left_col, right_col) + 1] = color
    return grid, expected


def _gen_horizontal_case() -> tuple[np.ndarray, np.ndarray] | None:
    band_count = random.randint(*BAND_COUNT_RANGE)
    region_heights = _sample_partition(band_count)
    if region_heights is None:
        return None

    width = random.randint(*OTHER_AXIS_RANGE)
    pair_count = random.randint(2, min(PAIR_COUNT_CAP, width))
    band_colors = random.sample(PALETTE, band_count)

    height = sum(region_heights) + band_count
    grid = np.zeros((height, width), dtype=int)

    regions: list[tuple[int, int]] = []
    row = 0
    for idx, region_height in enumerate(region_heights):
        regions.append((row, row + region_height - 1))
        row += region_height
        if idx < band_count:
            grid[row, :] = band_colors[idx]
            row += 1

    cols = random.sample(range(width), pair_count)
    records: list[tuple[int, int, int, int]] = []
    for col in cols:
        band_idx = random.randrange(band_count)
        top_row = random.randint(*regions[band_idx])
        bottom_row = random.randint(*regions[band_idx + 1])
        grid[top_row, col] = GREY
        grid[bottom_row, col] = GREY
        records.append((col, top_row, bottom_row, band_colors[band_idx]))

    expected = grid.copy()
    for col, top_row, bottom_row, color in records:
        expected[min(top_row, bottom_row) : max(top_row, bottom_row) + 1, col] = color
    return grid, expected


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    if random.random() < 0.5:
        return _gen_vertical_case()
    return _gen_horizontal_case()


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _gen_attempt()
        if case is None:
            continue
        grid, expected = case
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input for the separator-bridge task")
