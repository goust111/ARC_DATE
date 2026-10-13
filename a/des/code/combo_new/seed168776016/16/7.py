from __future__ import annotations

import random

import numpy as np


BG = 0
START_COLOR = 2
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
SIDES = ("top", "right", "bottom", "left")
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]

MIN_SEGMENTS = 2
MAX_SEGMENTS = 6
MIN_LENGTH = 1
MAX_LENGTH = 7
MIN_TOTAL_PATH = 5
LEGEND_STEPS = (1, 2)
MAX_ATTEMPTS = 300


def _boundary_coords(shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(w)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    if side == "bottom":
        return [(h - 1, c) for c in range(w)]
    if side == "left":
        return [(r, 0) for r in range(h)]
    raise AssertionError(side)


def _legend_info(grid: np.ndarray) -> tuple[str, list[tuple[int, int]], list[int]]:
    h, w = grid.shape
    boundary_fg = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and int(grid[r, c]) not in (BG, START_COLOR)
    }

    candidates: list[tuple[str, list[tuple[int, int]], list[int]]] = []
    for side in SIDES:
        coords = _boundary_coords(grid.shape, side)
        coord_set = set(coords)
        if not boundary_fg or not boundary_fg <= coord_set:
            continue

        ordered = [coord for coord in coords if coord in boundary_fg]
        colors = [int(grid[r, c]) for r, c in ordered]
        if len(colors) < 2:
            continue
        if len(colors) != len(set(colors)):
            continue
        if any(int(np.count_nonzero(grid == color)) < 2 for color in colors):
            continue
        candidates.append((side, ordered, colors))

    assert len(candidates) == 1, "expected one boundary legend"
    return candidates[0]


def _segment_length(grid: np.ndarray, color: int) -> int:
    coords = np.argwhere(grid == int(color))
    assert coords.size, f"missing segment color {color}"

    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        sorted_cols = np.sort(cols)
        assert int(sorted_cols[-1] - sorted_cols[0] + 1) == int(len(sorted_cols))
    elif np.all(cols == cols[0]):
        sorted_rows = np.sort(rows)
        assert int(sorted_rows[-1] - sorted_rows[0] + 1) == int(len(sorted_rows))
    else:
        raise AssertionError(f"segment color {color} is not a straight line")

    return int(len(coords))


def _draw_path(shape: tuple[int, int], start: tuple[int, int], order: list[int], lengths: dict[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    sr, sc = start
    out[sr, sc] = START_COLOR

    r, c = sr, sc
    for idx, color in enumerate(order):
        dr, dc = DIRS[idx % len(DIRS)]
        for _ in range(lengths[color]):
            r += dr
            c += dc
            assert 0 <= r < shape[0] and 0 <= c < shape[1], "reconstructed path must fit in-grid"
            out[r, c] = int(color)

    out[sr, sc] = START_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the scattered colored line segments into one continuous clockwise-turning polyline from the red start.

    Input structure:
    - The background is black.
    - Exactly one red single pixel marks the starting point.
    - Several other colors each form one isolated straight horizontal or vertical line segment in the interior.
    - One grid edge carries a one-pixel color legend whose boundary order gives the segment order.

    Rule:
    - Read the legend colors along the occupied boundary from one end to the other.
    - Ignore the legend pixels when measuring the corresponding segment lengths in the interior.
    - Clear the original segments and the legend, keep the red start, then redraw the segments in legend order.
    - The first segment extends right from the start, and each following segment turns 90 degrees clockwise relative to the previous one.

    Key insight:
    - The boundary colors encode only the concatenation order; each segment keeps its original color and pixel count while direction comes from a fixed right-down-left-up cycle.
    """
    g = np.asarray(grid, dtype=int)
    start_pixels = np.argwhere(g == START_COLOR)
    assert len(start_pixels) == 1, "expected one red start pixel"
    start = tuple(map(int, start_pixels[0]))

    _side, legend_cells, order = _legend_info(g)
    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = BG

    colors_present = set(map(int, np.unique(work))) - {BG, START_COLOR}
    assert colors_present == set(order), "legend colors must match the interior segments"

    lengths = {color: _segment_length(work, color) for color in order}
    return _draw_path(g.shape, start, order, lengths)


def _path_points(lengths: list[int]) -> list[tuple[int, int]]:
    r = c = 0
    points = [(0, 0)]
    for idx, length in enumerate(lengths):
        dr, dc = DIRS[idx % len(DIRS)]
        for _ in range(length):
            r += dr
            c += dc
            points.append((r, c))
    return points


def _is_simple_path(lengths: list[int]) -> bool:
    points = _path_points(lengths)
    if len(set(points)) != len(points):
        return False

    for i, (r0, c0) in enumerate(points):
        for j in range(i + 2, len(points)):
            r1, c1 = points[j]
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _path_extents(lengths: list[int]) -> tuple[int, int, int, int]:
    points = _path_points(lengths)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _legend_cells(shape: tuple[int, int], side: str, count: int, step: int) -> list[tuple[int, int]] | None:
    h, w = shape
    side_len = w if side in ("top", "bottom") else h
    span = 1 + (count - 1) * step
    if side_len < span + 2:
        return None

    offset = random.randint(1, side_len - span - 1)
    cells = []
    for i in range(count):
        pos = offset + i * step
        if side == "top":
            cells.append((0, pos))
        elif side == "right":
            cells.append((pos, w - 1))
        elif side == "bottom":
            cells.append((h - 1, pos))
        else:
            cells.append((pos, 0))
    return cells


def _can_place_segment(occupied: np.ndarray, r0: int, c0: int, length: int, orient: str) -> bool:
    h, w = occupied.shape
    if orient == "h":
        r1, c1 = r0, c0 + length - 1
    else:
        r1, c1 = r0 + length - 1, c0

    if r1 >= h or c1 >= w:
        return False

    rr0 = max(0, r0 - 1)
    rr1 = min(h, r1 + 2)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c1 + 2)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _segment_cells(r0: int, c0: int, length: int, orient: str) -> list[tuple[int, int]]:
    if orient == "h":
        return [(r0, c0 + i) for i in range(length)]
    return [(r0 + i, c0) for i in range(length)]


def _place_segments(grid: np.ndarray, start: tuple[int, int], items: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    occupied = grid != BG
    occupied[start] = True

    shuffled = items[:]
    random.shuffle(shuffled)
    for color, length in shuffled:
        candidates: list[tuple[str, int, int]] = []

        if w >= length + 2:
            for r in range(1, h - 1):
                for c in range(1, w - length):
                    if _can_place_segment(occupied, r, c, length, "h"):
                        candidates.append(("h", r, c))

        if h >= length + 2:
            for r in range(1, h - length):
                for c in range(1, w - 1):
                    if _can_place_segment(occupied, r, c, length, "v"):
                        candidates.append(("v", r, c))

        if not candidates:
            return False

        orient, r0, c0 = random.choice(candidates)
        for r, c in _segment_cells(r0, c0, length, orient):
            grid[r, c] = int(color)
            occupied[r, c] = True

    return True


def _expected_output(shape: tuple[int, int], start: tuple[int, int], order: list[int], lengths: list[int]) -> np.ndarray:
    return _draw_path(shape, start, order, dict(zip(order, lengths, strict=True)))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    colors = random.sample(COLOR_POOL, count)

    lengths = [random.randint(MIN_LENGTH, MAX_LENGTH) for _ in range(count)]
    if sum(lengths) < MIN_TOTAL_PATH or not _is_simple_path(lengths):
        return None

    side = random.choice(SIDES)
    step = random.choice(LEGEND_STEPS)
    min_r, max_r, min_c, max_c = _path_extents(lengths)
    legend_span = 1 + (count - 1) * step
    longest = max(lengths)

    min_h = max(max_r - min_r + 3, longest + 2, 8)
    min_w = max(max_c - min_c + 3, longest + 2, 8)
    if side in ("top", "bottom"):
        min_w = max(min_w, legend_span + 2)
    else:
        min_h = max(min_h, legend_span + 2)
    if min_h > 30 or min_w > 30:
        return None

    max_h = min(30, min_h + 8)
    max_w = min(30, min_w + 8)
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    legend_cells = _legend_cells((h, w), side, count, step)
    if legend_cells is None:
        return None

    start_row_min = 1 - min_r
    start_row_max = h - 2 - max_r
    start_col_min = 1 - min_c
    start_col_max = w - 2 - max_c
    if start_row_min > start_row_max or start_col_min > start_col_max:
        return None
    start = (
        random.randint(start_row_min, start_row_max),
        random.randint(start_col_min, start_col_max),
    )

    grid = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(legend_cells, colors, strict=True):
        grid[r, c] = int(color)
    grid[start] = START_COLOR

    if not _place_segments(grid, start, list(zip(colors, lengths, strict=True))):
        return None

    expected = _expected_output((h, w), start, colors, lengths)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
