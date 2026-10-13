import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
PAIR_COLOR_POOL = (1, 5, 6, 7, 8, 9)

GRID_MIN = 12
GRID_MAX = 24
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 5
SEGMENT_LEN_MIN = 4
SEGMENT_LEN_MAX = 10
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 500
EXTRA_MARKER_MAX = 3


def _sorted_endpoints(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    a = (int(coords[0, 0]), int(coords[0, 1]))
    b = (int(coords[1, 0]), int(coords[1, 1]))
    if a[0] == b[0]:
        return (a, b) if a[1] <= b[1] else (b, a)
    assert a[1] == b[1], "paired pixels must share a row or a column"
    return (a, b) if a[0] <= b[0] else (b, a)


def _line_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    if start[0] == end[0]:
        r = start[0]
        c0, c1 = sorted((start[1], end[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    assert start[1] == end[1], "line must be horizontal or vertical"
    c = start[1]
    r0, r1 = sorted((start[0], end[0]))
    return [(r, c) for r in range(r0, r1 + 1)]


def _line_halo(
    height: int, width: int, start: tuple[int, int], end: tuple[int, int]
) -> set[tuple[int, int]]:
    if start[0] == end[0]:
        r = start[0]
        c0, c1 = sorted((start[1], end[1]))
        return {
            (rr, cc)
            for rr in range(max(0, r - 1), min(height, r + 2))
            for cc in range(max(0, c0 - 1), min(width, c1 + 2))
        }
    c = start[1]
    r0, r1 = sorted((start[0], end[0]))
    return {
        (rr, cc)
        for rr in range(max(0, r0 - 1), min(height, r1 + 2))
        for cc in range(max(0, c - 1), min(width, c + 2))
    }


def _neighbors8(height: int, width: int, cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (rr, cc)
        for rr in range(max(0, r - 1), min(height, r + 2))
        for cc in range(max(0, c - 1), min(width, c + 2))
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pixel pair with a context-sensitive line and replace the endpoints with the line color.

    Input structure:
    - The background is black.
    - Several foreground colors appear exactly twice, and each such same-color pair lies on one row or one column.
    - A separate yellow marker color may appear on some cells along those potential row/column segments and on unrelated background cells.

    Rule:
    - For every color that appears exactly twice in one row or one column, consider the full inclusive straight segment between the two pixels.
    - If at least one yellow marker lies on that segment, repaint the whole segment red; otherwise repaint the whole segment green.
    - Yellow markers that are not on any pair segment stay yellow, while the original pair endpoints are overwritten by the new line color.

    Key insight:
    - The pair endpoints identify where to draw, but the segment color is determined by path context: the presence or absence of the third marker color on that path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert colors, "expected at least one foreground color"

    pair_colors: list[int] = []
    for color in colors:
        coords = np.argwhere(g == color)
        if coords.shape[0] != 2:
            continue
        same_row = int(coords[0, 0]) == int(coords[1, 0])
        same_col = int(coords[0, 1]) == int(coords[1, 1])
        if same_row or same_col:
            pair_colors.append(color)

    marker_colors = [color for color in colors if color not in pair_colors]
    assert pair_colors, "expected at least one pair color"
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker_color = marker_colors[0]

    out = g.copy()
    for color in pair_colors:
        coords = np.argwhere(g == color)
        start, end = _sorted_endpoints(coords)
        cells = _line_cells(start, end)
        assert len(cells) >= 2, "paired pixels must be distinct"

        allowed = {BLACK, color, marker_color}
        for r, c in cells:
            value = int(g[r, c])
            assert value in allowed, "pair segment must contain only background or marker cells"

        if start[0] == end[0]:
            r = start[0]
            c0, c1 = start[1], end[1]
            segment = g[r, c0 : c1 + 1]
            has_marker = bool(np.any(segment[1:-1] == marker_color))
            out[r, c0 : c1 + 1] = RED if has_marker else GREEN
        else:
            c = start[1]
            r0, r1 = start[0], end[0]
            segment = g[r0 : r1 + 1, c]
            has_marker = bool(np.any(segment[1:-1] == marker_color))
            out[r0 : r1 + 1, c] = RED if has_marker else GREEN

    return out


def _sample_segment(height: int, width: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    orientation = rng.choice(("h", "v"))
    if orientation == "h":
        max_len = min(SEGMENT_LEN_MAX, width)
        if max_len < SEGMENT_LEN_MIN:
            return None
        length = rng.randint(SEGMENT_LEN_MIN, max_len)
        r = rng.randrange(height)
        c0 = rng.randrange(width - length + 1)
        return (r, c0), (r, c0 + length - 1)

    max_len = min(SEGMENT_LEN_MAX, height)
    if max_len < SEGMENT_LEN_MIN:
        return None
    length = rng.randint(SEGMENT_LEN_MIN, max_len)
    c = rng.randrange(width)
    r0 = rng.randrange(height - length + 1)
    return (r0, c), (r0 + length - 1, c)


def _place_extra_markers(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    marker_forbidden: set[tuple[int, int]],
    existing_marker_count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    height, width = map(int, grid.shape)
    options = [count for count in range(EXTRA_MARKER_MAX + 1) if existing_marker_count + count != 2]
    rng.shuffle(options)
    if any(count > 0 for count in options) and rng.random() < 0.7:
        options.sort(key=lambda count: count == 0)

    for target_count in options:
        chosen: list[tuple[int, int]] = []
        local_forbidden = set(marker_forbidden)
        candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if int(grid[r, c]) == BLACK and (r, c) not in blocked and (r, c) not in local_forbidden
        ]
        rng.shuffle(candidates)

        for cell in candidates:
            if cell in local_forbidden:
                continue
            chosen.append(cell)
            local_forbidden.update(_neighbors8(height, width, cell))
            if len(chosen) == target_count:
                return chosen

        if target_count == 0:
            return []

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    capacity_limit = max(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, height * width // 45 + 1))
    pair_count = rng.randint(PAIR_COUNT_MIN, capacity_limit)
    pair_colors = rng.sample(PAIR_COLOR_POOL, pair_count)

    red_count = rng.randint(1, pair_count - 1)
    red_indices = set(rng.sample(range(pair_count), red_count))

    segments: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()
    for index, color in enumerate(pair_colors):
        placed = False
        needs_marker = index in red_indices
        for _ in range(PLACEMENT_RETRIES):
            endpoints = _sample_segment(height, width, rng)
            if endpoints is None:
                return None
            start, end = endpoints
            cells = _line_cells(start, end)
            if any(cell in blocked for cell in cells):
                continue
            segments.append(
                {
                    "color": color,
                    "start": start,
                    "end": end,
                    "cells": cells,
                    "needs_marker": needs_marker,
                }
            )
            blocked.update(_line_halo(height, width, start, end))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    marker_forbidden: set[tuple[int, int]] = set()
    marker_count = 0
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        color = int(segment["color"])
        cells = segment["cells"]
        grid[start] = color
        grid[end] = color

        if bool(segment["needs_marker"]):
            marker_cell = rng.choice(cells[1:-1])
            grid[marker_cell] = YELLOW
            marker_forbidden.update(_neighbors8(height, width, marker_cell))
            marker_count += 1

    extra_markers = _place_extra_markers(grid, blocked, marker_forbidden, marker_count, rng)
    if extra_markers is None:
        return None
    for cell in extra_markers:
        grid[cell] = YELLOW

    if int(np.count_nonzero(grid == YELLOW)) == 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == RED) or not np.any(out == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
