from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PLATFORM = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_HEIGHT = 10
MAX_HEIGHT = 24
MIN_WIDTH = 10
MAX_WIDTH = 24
SOURCE_GAP = 4
MAX_GENERATION_ATTEMPTS = 400


def _scan_platforms(grid: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Return horizontal platform objects as (left, right, marker_color) per row."""
    height, width = grid.shape
    rows: list[list[tuple[int, int, int]]] = [[] for _ in range(height)]
    for r in range(height):
        c = 0
        while c < width:
            if int(grid[r, c]) == BG:
                c += 1
                continue
            left = c
            while c + 1 < width and int(grid[r, c + 1]) != BG:
                c += 1
            right = c
            gseg = grid[r, left : right + 1]

            if right - left + 1 >= 3:
                if int(gseg[0]) not in {BG, PLATFORM} and np.all(gseg[1:] == PLATFORM):
                    rows[r].append((int(left), int(right), int(gseg[0])))
                elif int(gseg[-1]) not in {BG, PLATFORM} and np.all(gseg[:-1] == PLATFORM):
                    rows[r].append((int(left), int(right), int(gseg[-1])))
                else:
                    assert r == 0, "non-source foreground runs must be tagged horizontal platforms"
            c += 1
    return rows


def _find_platform(
    platforms_by_row: list[list[tuple[int, int, int]]], row: int, col: int
) -> tuple[int, int, int] | None:
    for left, right, marker_color in platforms_by_row[row]:
        if left <= col <= right:
            return int(left), int(right), int(marker_color)
    return None


def _paint_bg_cell(out: np.ndarray, inp: np.ndarray, row: int, col: int, color: int) -> None:
    assert int(inp[row, col]) == BG, "generated path may only paint background cells"
    prev = int(out[row, col])
    assert prev in {BG, color}, "two streams may not conflict on a painted cell"
    out[row, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split top-row sources down a set of horizontal platforms and recolor branches by platform markers.

    Input structure:
    - Black background with colored source pixels on the top row.
    - Gray horizontal platform segments.
    - Each platform has exactly one colored marker pixel occupying its left or right endpoint.

    Rule:
    - Each source falls straight down and paints its own color on background cells.
    - A falling ray stops one row above the first platform it meets.
    - From that stop row, two new branch pixels appear just outside the platform ends.
    - Both child rays continue downward in the platform marker's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    platforms_by_row = _scan_platforms(g)
    sources = [
        (0, int(c), int(g[0, c]))
        for c in range(width)
        if int(g[0, c]) not in {BG, PLATFORM}
    ]
    assert sources, "expected at least one top-row source"

    out = g.copy()
    queued_colors: dict[tuple[int, int], int] = {}
    queue = deque(sources)

    while queue:
        start_row, col, color = queue.popleft()
        key = (int(start_row), int(col))
        prev = queued_colors.get(key)
        assert prev in {None, color}, "a stream start may not receive two different colors"
        if prev is not None:
            continue
        queued_colors[key] = int(color)

        if int(g[start_row, col]) == BG:
            _paint_bg_cell(out, g, int(start_row), int(col), int(color))
        else:
            assert int(g[start_row, col]) == int(color), "non-branch starts must already carry the stream color"

        row = int(start_row)
        while True:
            next_row = row + 1
            if next_row >= height:
                break

            below = int(g[next_row, col])
            if below == PLATFORM:
                platform = _find_platform(platforms_by_row, int(next_row), int(col))
                assert platform is not None, "every gray hit must belong to a scanned platform"
                left, right, new_color = platform
                for new_col in (left - 1, right + 1):
                    assert 0 <= new_col < width
                    _paint_bg_cell(out, g, int(row), int(new_col), int(new_color))
                    child_key = (int(row), int(new_col))
                    existing = queued_colors.get(child_key)
                    assert existing in {None, new_color}, "branch starts may not recolor an existing branch"
                    if existing is None:
                        queue.append((int(row), int(new_col), int(new_color)))
                break

            assert below == BG, "vertical fall path may only cross background until a platform"
            row = next_row
            _paint_bg_cell(out, g, int(row), int(col), int(color))

    return out


def _sample_spaced_positions(count: int, lo: int, hi: int, min_diff: int) -> list[int] | None:
    span = hi - lo + 1
    padding = min_diff - 1
    needed = count + max(0, count - 1) * padding
    if span < needed:
        return None
    compressed = span - max(0, count - 1) * padding
    picks = sorted(random.sample(range(compressed), count))
    return [int(lo + picks[i] + i * padding) for i in range(count)]


def _source_intervals(source_cols: list[int], width: int) -> list[tuple[int, int]]:
    if len(source_cols) == 1:
        return [(0, width - 1)]

    mids = [(source_cols[i] + source_cols[i + 1]) // 2 for i in range(len(source_cols) - 1)]
    intervals: list[tuple[int, int]] = []
    left = 0
    for i in range(len(source_cols)):
        right = width - 1 if i == len(source_cols) - 1 else mids[i]
        intervals.append((int(left), int(right)))
        left = right + 1
    return intervals


def _build_stream_tree(
    start_row: int,
    col: int,
    depth: int,
    left_bound: int,
    right_bound: int,
    height: int,
    max_depth: int,
    split_prob: float,
    platforms: list[tuple[int, int, int, int, str]],
    platform_depths: list[int],
) -> None:
    min_shelf_row = start_row + 3
    max_shelf_row = min(height - 1, start_row + 7)
    interval_width = right_bound - left_bound + 1
    if (
        depth >= max_depth
        or min_shelf_row > max_shelf_row
        or interval_width < 5
        or col <= left_bound
        or col >= right_bound
        or random.random() > split_prob
    ):
        return

    for _ in range(20):
        child_left = random.randint(left_bound, col - 1)
        child_right = random.randint(col + 1, right_bound)
        if child_right - child_left < 4:
            continue

        left = child_left + 1
        right = child_right - 1
        shelf_row = random.randint(min_shelf_row, max_shelf_row)
        marker_color = random.choice(PALETTE)
        marker_side = random.choice(("left", "right"))
        if marker_side == "left" and col == left:
            continue
        if marker_side == "right" and col == right:
            continue

        platforms.append((int(shelf_row), int(left), int(right), int(marker_color), marker_side))
        platform_depths.append(int(depth))

        child_row = shelf_row - 1
        _build_stream_tree(
            int(child_row),
            int(child_left),
            int(depth + 1),
            int(left_bound),
            int(child_left),
            int(height),
            int(max_depth),
            float(split_prob),
            platforms,
            platform_depths,
        )
        _build_stream_tree(
            int(child_row),
            int(child_right),
            int(depth + 1),
            int(child_right),
            int(right_bound),
            int(height),
            int(max_depth),
            float(split_prob),
            platforms,
            platform_depths,
        )
        return


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)

    max_sources = 1 if width < 16 else 2 if width < 24 else 3
    source_count = random.randint(1, max_sources)
    source_gap = 6 if source_count >= 2 else SOURCE_GAP
    source_cols = _sample_spaced_positions(source_count, 1, width - 2, source_gap)
    if source_cols is None:
        return None

    source_colors = random.sample(PALETTE, source_count)
    intervals = _source_intervals(source_cols, width)
    if any(not (left <= col <= right) for col, (left, right) in zip(source_cols, intervals)):
        return None

    max_depth = random.randint(2, 4)
    split_prob = random.uniform(0.72, 0.92)
    platforms: list[tuple[int, int, int, int, str]] = []
    platform_depths: list[int] = []

    for (col, _color), (left_bound, right_bound) in zip(zip(source_cols, source_colors), intervals):
        if right_bound - left_bound + 1 < 5:
            continue
        _build_stream_tree(
            0,
            int(col),
            0,
            int(left_bound),
            int(right_bound),
            int(height),
            int(max_depth),
            float(split_prob),
            platforms,
            platform_depths,
        )

    if len(platforms) < 2:
        return None
    if len(platforms) >= 3 and max(platform_depths, default=-1) < 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    for col, color in zip(source_cols, source_colors):
        grid[0, int(col)] = int(color)

    for row, left, right, marker_color, marker_side in platforms:
        if left <= 0 or right >= width - 1:
            return None
        grid[int(row), int(left) : int(right) + 1] = PLATFORM
        marker_col = int(left) if marker_side == "left" else int(right)
        grid[int(row), marker_col] = int(marker_color)

    try:
        out = main(grid.copy())
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < max(8, height // 2):
        return None

    structural = grid != BG
    if not np.array_equal(out[structural], grid[structural]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid waterfall recoloring puzzle")
