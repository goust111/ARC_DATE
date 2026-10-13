from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _separator_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
        cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
        if rows or cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for stop in sorted(separators):
        if start < stop:
            spans.append((start, stop))
        start = stop + 1
    if start < length:
        spans.append((start, length))
    return spans


def _region_seeds(
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    separator_color: int,
) -> list[tuple[int, int, int]]:
    r0, r1 = row_span
    c0, c1 = col_span
    seeds: list[tuple[int, int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            color = int(grid[r, c])
            if color not in (BACKGROUND, separator_color):
                seeds.append((r, c, color))
    assert 2 <= len(seeds) <= 4, "each region must contain 2-4 seeds"
    assert len({color for _, _, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds


def _growth_axis(seeds: list[tuple[int, int, int]]) -> str:
    rows = [r for r, _, _ in seeds]
    cols = [c for _, c, _ in seeds]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    assert row_span != col_span, "seed layout must have a clear dominant axis"
    return "vertical" if row_span < col_span else "horizontal"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each region's seed pixels into full stripes along the axis orthogonal
    to the seeds' dominant arrangement direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    separator_color, separator_rows, separator_cols = _separator_structure(g)

    row_regions = _intervals(g.shape[0], separator_rows)
    col_regions = _intervals(g.shape[1], separator_cols)
    assert row_regions and col_regions, "expected at least one rectangular region"

    out = np.where(g == separator_color, separator_color, BACKGROUND)

    for row_span in row_regions:
        for col_span in col_regions:
            seeds = _region_seeds(g, row_span, col_span, separator_color)
            axis = _growth_axis(seeds)
            for r, c, color in sorted(seeds):
                if axis == "vertical":
                    for rr in range(row_span[0], row_span[1]):
                        if out[rr, c] == BACKGROUND:
                            out[rr, c] = color
                else:
                    for cc in range(col_span[0], col_span[1]):
                        if out[r, cc] == BACKGROUND:
                            out[r, cc] = color
    return out


ROW_REGION_RANGE = (1, 3)
COL_REGION_RANGE = (1, 4)
REGION_HEIGHT_RANGE = (4, 7)
REGION_WIDTH_RANGE = (4, 6)
SEED_COUNT_RANGE = (2, 4)
OVERLAP_PROBABILITY = 0.35
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 200


def _sample_axis_positions(
    low: int,
    high: int,
    count: int,
    allow_overlap: bool,
    rng: random.Random,
) -> list[int] | None:
    width = high - low + 1
    needed_unique = count - 1 if allow_overlap else count
    if width < needed_unique:
        return None

    if count == 2:
        positions = [low, high]
        rng.shuffle(positions)
        return positions

    positions = [low, high]
    if allow_overlap:
        positions.append(rng.choice((low, high)))
        pool = [value for value in range(low + 1, high) if value not in positions]
        extra = count - 3
    else:
        pool = list(range(low + 1, high))
        extra = count - 2

    if extra > len(pool):
        return None

    if extra:
        positions.extend(rng.sample(pool, extra))
    rng.shuffle(positions)
    return positions


def _sample_secondary_positions(
    primary_positions: list[int],
    low: int,
    high: int,
    rng: random.Random,
) -> list[int] | None:
    occupied: set[tuple[int, int]] = set()
    secondaries: list[int] = []
    choices = list(range(low, high + 1))
    for primary in primary_positions:
        candidates = choices[:]
        rng.shuffle(candidates)
        for secondary in candidates:
            if (primary, secondary) not in occupied:
                occupied.add((primary, secondary))
                secondaries.append(secondary)
                break
        else:
            return None
    return secondaries


def _sample_seed_positions(
    height: int,
    width: int,
    count: int,
    arrangement: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        if arrangement == "horizontal":
            minor_limit = min(2, height - 1)
            minor_span = rng.randint(0, minor_limit)
            allow_overlap = count >= 3 and minor_span >= 1 and rng.random() < OVERLAP_PROBABILITY
            major_min = max(minor_span + 1, (count - 1 if allow_overlap else count) - 1)
            if major_min > width - 1:
                continue
            major_span = rng.randint(major_min, width - 1)
            row_low = rng.randint(0, height - 1 - minor_span)
            col_low = rng.randint(0, width - 1 - major_span)
            cols = _sample_axis_positions(col_low, col_low + major_span, count, allow_overlap, rng)
            if cols is None:
                continue
            rows = _sample_secondary_positions(cols, row_low, row_low + minor_span, rng)
            if rows is None:
                continue
            points = list(zip(rows, cols))
        else:
            minor_limit = min(2, width - 1)
            minor_span = rng.randint(0, minor_limit)
            allow_overlap = count >= 3 and minor_span >= 1 and rng.random() < OVERLAP_PROBABILITY
            major_min = max(minor_span + 1, (count - 1 if allow_overlap else count) - 1)
            if major_min > height - 1:
                continue
            major_span = rng.randint(major_min, height - 1)
            row_low = rng.randint(0, height - 1 - major_span)
            col_low = rng.randint(0, width - 1 - minor_span)
            rows = _sample_axis_positions(row_low, row_low + major_span, count, allow_overlap, rng)
            if rows is None:
                continue
            cols = _sample_secondary_positions(rows, col_low, col_low + minor_span, rng)
            if cols is None:
                continue
            points = list(zip(rows, cols))

        row_span = max(r for r, _ in points) - min(r for r, _ in points)
        col_span = max(c for _, c in points) - min(c for _, c in points)
        if arrangement == "horizontal" and row_span < col_span:
            return points
        if arrangement == "vertical" and col_span < row_span:
            return points
    return None


def _region_spans(lengths: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, length in enumerate(lengths):
        spans.append((cursor, cursor + length))
        cursor += length
        if index + 1 < len(lengths):
            cursor += 1
    return spans


def _balanced_arrangements(total: int, rng: random.Random) -> list[str]:
    if total == 1:
        return [rng.choice(("horizontal", "vertical"))]
    horizontal_count = total // 2
    vertical_count = total - horizontal_count
    arrangements = ["horizontal"] * horizontal_count + ["vertical"] * vertical_count
    rng.shuffle(arrangements)
    return arrangements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    for _ in range(PLACEMENT_RETRIES):
        row_regions = rng.randint(*ROW_REGION_RANGE)
        col_regions = rng.randint(*COL_REGION_RANGE)
        if 2 <= row_regions * col_regions <= 8:
            break
    else:
        return None

    row_heights = [rng.randint(*REGION_HEIGHT_RANGE) for _ in range(row_regions)]
    col_widths = [rng.randint(*REGION_WIDTH_RANGE) for _ in range(col_regions)]

    h = sum(row_heights) + row_regions - 1
    w = sum(col_widths) + col_regions - 1
    grid = np.zeros((h, w), dtype=int)

    row_spans = _region_spans(row_heights)
    col_spans = _region_spans(col_widths)
    for r0, r1 in row_spans[:-1]:
        grid[r1, :] = SEPARATOR
    for c0, c1 in col_spans[:-1]:
        grid[:, c1] = SEPARATOR

    arrangements = _balanced_arrangements(len(row_spans) * len(col_spans), rng)
    index = 0
    for row_span in row_spans:
        for col_span in col_spans:
            height = row_span[1] - row_span[0]
            width = col_span[1] - col_span[0]
            count = rng.randint(*SEED_COUNT_RANGE)
            points = _sample_seed_positions(height, width, count, arrangements[index], rng)
            index += 1
            if points is None:
                return None
            colors = rng.sample(SEED_COLORS, count)
            for (local_r, local_c), color in zip(points, colors):
                grid[row_span[0] + local_r, col_span[0] + local_c] = color

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
