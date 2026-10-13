from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
OUTPUT_PALETTE = (BLUE, RED, GREEN, YELLOW)

GEN_HEIGHT_RANGE = (12, 16)
GEN_WIDTH_RANGE = (16, 22)
GEN_PLATFORM_COUNTS = (2, 3)
GEN_PLATFORM_GAP = 2
GEN_ROW_GAP = 2
GEN_TOTAL_PIXELS = (8, 12)
GEN_STRAY_PIXELS = (1, 2)
GEN_MAX_STACK = 3
GEN_RETRIES = 500


def _find_platforms(grid: np.ndarray) -> list[dict[str, int]]:
    platforms: list[dict[str, int]] = []
    gray_mask = grid == GRAY
    for r in range(grid.shape[0]):
        cols = np.where(gray_mask[r])[0].tolist()
        if not cols:
            continue
        start = cols[0]
        prev = cols[0]
        for c in cols[1:] + [None]:
            if c is None or c != prev + 1:
                platforms.append({"row": int(r), "c0": int(start), "c1": int(prev)})
                if c is not None:
                    start = c
            if c is not None:
                prev = c
    return platforms


def _intervals_disjoint(platforms: list[dict[str, int]]) -> bool:
    ordered = sorted(platforms, key=lambda p: p["c0"])
    for left, right in zip(ordered, ordered[1:]):
        if left["c1"] >= right["c0"]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored pixels onto gray platforms, then recolor each platform stack by local occupied-column rank.

    Input structure:
    - The grid has black background, 2-3 horizontal gray platform segments, and scattered single colored pixels.
    - Platform segments occupy different column ranges and act as collectors for pixels falling in those columns.
    - Colored pixels may start above a platform range or in columns with no platform beneath them.

    Rule:
    - Remove the original colored pixels and let every non-black, non-gray pixel fall straight downward.
    - In each column, a pixel stops immediately above the nearest gray platform cell or above an already settled pixel in that same column.
    - Pixels in columns with no reachable gray platform fall out of the grid and disappear.
    - After settling, process each gray platform independently: sort its occupied columns from left to right, recolor ranks 1-4 to blue, red, green, yellow, and erase any ranks beyond 4.

    Key insight:
    - Gravity decides which platform a pixel belongs to, and the recoloring rank is local to that platform's occupied columns rather than a single global column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    platforms = _find_platforms(g)
    assert platforms, "expected at least one gray platform"
    assert _intervals_disjoint(platforms), "platform column ranges must not overlap"

    out = np.full((h, w), BLACK, dtype=int)
    platform_cell_to_id: dict[tuple[int, int], int] = {}
    blockers_by_col: list[dict[int, int]] = [dict() for _ in range(w)]
    for pid, platform in enumerate(platforms):
        row, c0, c1 = platform["row"], platform["c0"], platform["c1"]
        out[row, c0 : c1 + 1] = GRAY
        for c in range(c0, c1 + 1):
            platform_cell_to_id[(row, c)] = pid
            blockers_by_col[c][row] = pid

    settled: list[tuple[int, int, int]] = []
    movable_mask = (g != BLACK) & (g != GRAY)
    for c in range(w):
        source_rows = sorted((int(r) for r in np.where(movable_mask[:, c])[0]), reverse=True)
        if not source_rows:
            continue
        blockers = dict(blockers_by_col[c])
        for src_r in source_rows:
            below = [row for row in blockers if row > src_r]
            if not below:
                continue
            blocker_row = min(below)
            dest_r = blocker_row - 1
            if dest_r < 0 or dest_r in blockers:
                continue
            pid = blockers[blocker_row]
            blockers[dest_r] = pid
            settled.append((dest_r, c, pid))

    pixels_by_platform: list[list[tuple[int, int]]] = [[] for _ in platforms]
    for r, c, pid in settled:
        pixels_by_platform[pid].append((r, c))

    for pid, pixels in enumerate(pixels_by_platform):
        used_cols = sorted({c for _, c in pixels})
        color_by_col = {c: OUTPUT_PALETTE[i] for i, c in enumerate(used_cols[: len(OUTPUT_PALETTE)])}
        for r, c in pixels:
            color = color_by_col.get(c)
            if color is not None:
                out[r, c] = color

    return out


def _sample_rows(count: int, upper_exclusive: int) -> list[int] | None:
    if count > upper_exclusive:
        return None
    rows = list(range(upper_exclusive))
    return sorted(random.sample(rows, count))


def _sample_platform_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(max(5, height // 2), height - 1))
    if len(candidates) < count:
        return None
    for _ in range(100):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= GEN_ROW_GAP for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _interval_has_gap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a1 + GEN_PLATFORM_GAP <= b0 or b1 + GEN_PLATFORM_GAP <= a0


def _sample_platform_intervals(width: int, occupied_counts: list[int]) -> list[tuple[int, int]] | None:
    order = list(range(len(occupied_counts)))
    random.shuffle(order)
    intervals: list[tuple[int, int] | None] = [None] * len(occupied_counts)
    for idx in order:
        occupied = occupied_counts[idx]
        max_width = min(occupied + 2, 7)
        interval_width = random.randint(occupied, max_width)
        starts = list(range(width - interval_width + 1))
        random.shuffle(starts)
        placed = False
        for start in starts:
            end = start + interval_width - 1
            if all(
                other is None or _interval_has_gap(start, end, other[0], other[1])
                for other in intervals
            ):
                intervals[idx] = (start, end)
                placed = True
                break
        if not placed:
            return None
    return [(int(a), int(b)) for a, b in intervals if a is not None and b is not None]


def _sample_occupied_counts(platform_count: int) -> list[int] | None:
    overflow = random.random() < 0.45
    overflow_idx = random.randrange(platform_count) if overflow else None
    counts: list[int] = []
    for i in range(platform_count):
        if i == overflow_idx:
            counts.append(5)
        elif platform_count == 3:
            counts.append(random.choice((2, 3)))
        else:
            counts.append(random.choice((2, 3, 4)))
    if sum(counts) + GEN_STRAY_PIXELS[0] > GEN_TOTAL_PIXELS[1]:
        return None
    return counts


def _choose_platform_columns(interval: tuple[int, int], count: int) -> list[int]:
    cols = list(range(interval[0], interval[1] + 1))
    return sorted(random.sample(cols, count))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    platform_count = random.choice(GEN_PLATFORM_COUNTS)

    occupied_counts = _sample_occupied_counts(platform_count)
    if occupied_counts is None:
        return None

    platform_rows = _sample_platform_rows(height, platform_count)
    if platform_rows is None:
        return None

    intervals = _sample_platform_intervals(width, occupied_counts)
    if intervals is None or len(intervals) != platform_count:
        return None

    stray_count = random.randint(*GEN_STRAY_PIXELS)
    base_pixels = sum(occupied_counts) + stray_count
    if base_pixels > GEN_TOTAL_PIXELS[1]:
        return None
    target_total = random.randint(max(GEN_TOTAL_PIXELS[0], base_pixels), GEN_TOTAL_PIXELS[1])

    platform_columns = [_choose_platform_columns(intervals[i], occupied_counts[i]) for i in range(platform_count)]
    heights_by_platform: list[dict[int, int]] = [{c: 1 for c in cols} for cols in platform_columns]
    extra_capacity = sum((GEN_MAX_STACK - 1) * len(cols) for cols in platform_columns)
    extra_needed = target_total - base_pixels
    if extra_needed > extra_capacity:
        return None

    eligible_slots = [(pid, c) for pid, cols in enumerate(platform_columns) for c in cols]
    while extra_needed > 0:
        options = [(pid, c) for pid, c in eligible_slots if heights_by_platform[pid][c] < GEN_MAX_STACK]
        if not options:
            return None
        pid, col = random.choice(options)
        heights_by_platform[pid][col] += 1
        extra_needed -= 1

    grid = np.full((height, width), BLACK, dtype=int)
    platform_specs: list[dict[str, int]] = []
    covered_cols: set[int] = set()
    for pid in range(platform_count):
        row = platform_rows[pid]
        c0, c1 = intervals[pid]
        grid[row, c0 : c1 + 1] = GRAY
        platform_specs.append({"row": row, "c0": c0, "c1": c1})
        covered_cols.update(range(c0, c1 + 1))

    for pid, col_heights in enumerate(heights_by_platform):
        row = platform_specs[pid]["row"]
        for col, count in col_heights.items():
            src_rows = _sample_rows(count, row)
            if src_rows is None:
                return None
            for src_r in src_rows:
                grid[src_r, col] = random.choice(INPUT_COLORS)

    stray_cols = [c for c in range(width) if c not in covered_cols]
    if len(stray_cols) < stray_count:
        return None
    for col in random.sample(stray_cols, stray_count):
        src_r = random.randint(0, height - 2)
        grid[src_r, col] = random.choice(INPUT_COLORS)

    movable = int(np.count_nonzero((grid != BLACK) & (grid != GRAY)))
    if not (GEN_TOTAL_PIXELS[0] <= movable <= GEN_TOTAL_PIXELS[1]):
        return None

    for pid, cols in enumerate(platform_columns):
        if len(cols) < 2:
            return None
        row = platform_specs[pid]["row"]
        pixel_count = int(np.count_nonzero((grid[:row] != BLACK) & (grid[:row] != GRAY) & np.isin(np.arange(width), cols)))
        if pixel_count < 2:
            return None

    if not any(c not in covered_cols for c in np.where((grid != BLACK) & (grid != GRAY))[1]):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    output_pixels = np.count_nonzero((out != BLACK) & (out != GRAY))
    if output_pixels == 0:
        return None

    if any(len(cols) > 4 for cols in platform_columns):
        if output_pixels >= movable:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
