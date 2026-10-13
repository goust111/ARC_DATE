from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


BACKGROUND = Color.BLACK
AXIS_COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
RAY_COLOR_POOL = [Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

HEIGHT_RANGE = (11, 19)
WIDTH_OPTIONS = [13, 15, 17, 19]
SEGMENT_COUNT_WEIGHTS = ((1, 2), (2, 4), (3, 3))
SEGMENT_LENGTH_MIN = 1
SEGMENT_LENGTH_MAX = 4
MAX_TOTAL_AXIS_PIXELS = 6
MIN_TOTAL_AXIS_PIXELS = 2
SAFE_MARKER_DISTANCE = 5
TIE_PROBABILITY = 0.35
DECOY_PROBABILITY = 0.45
TARGET_NON_AXIS_RANGE = (5, 10)
GEN_ATTEMPTS = 400


def _axis_runs(grid: np.ndarray, axis_col: int) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    h = int(grid.shape[0])
    r = 0
    while r < h:
        color = int(grid[r, axis_col])
        if color == BACKGROUND:
            r += 1
            continue
        r1 = r
        while r1 + 1 < h and int(grid[r1 + 1, axis_col]) == color:
            r1 += 1
        length = r1 - r + 1
        for rr in range(r, r1 + 1):
            runs.append({"row": rr, "length": length, "axis_color": color})
        r = r1 + 1
    return runs


def _nearest_non_axis_color(grid: np.ndarray, row: int, axis_col: int) -> int:
    candidates = [
        c
        for c in range(int(grid.shape[1]))
        if c != axis_col and int(grid[row, c]) != BACKGROUND
    ]
    assert candidates, "each axis foreground row must have a non-axis foreground pixel"
    best_col = min(candidates, key=lambda c: (abs(c - axis_col), 0 if c < axis_col else 1, c))
    return int(grid[row, best_col])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the center-axis pixels and emit diagonal rays whose lengths come from local axis runs.

    Input structure:
    - The background is black, and the grid width is odd so a unique center column exists.
    - The center column contains one or more vertical foreground runs; each run is monochrome.
    - Every foreground row on the center axis also contains at least one foreground pixel away from the axis.

    Rule:
    - Keep only the center-axis foreground pixels from the input and erase every non-axis foreground pixel to black.
    - For each center-axis foreground pixel, measure the length L of its contiguous same-color vertical run.
    - Find the nearest non-axis foreground pixel on that same row; if left and right are equally close, choose the left one.
    - Use that row pixel's color to paint two diagonal rays from the axis source: one to the upper-right and one to the lower-left, each extending L cells and clipped by the grid boundary.

    Key insight:
    - The axis pixels act as ray sources, but each source needs two different local cues: run length from the axis itself and ray color from the nearest off-axis pixel in the same row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    axis_col = int(g.shape[1] // 2)

    out = np.full_like(g, BACKGROUND)
    out[:, axis_col] = g[:, axis_col]

    best_len = np.full(g.shape, -1, dtype=int)
    best_row = np.full(g.shape, g.shape[0] + 1, dtype=int)

    for source in _axis_runs(g, axis_col):
        row = int(source["row"])
        length = int(source["length"])
        ray_color = _nearest_non_axis_color(g, row, axis_col)

        for dr, dc in ((-1, 1), (1, -1)):
            for step in range(1, length + 1):
                rr = row + dr * step
                cc = axis_col + dc * step
                if not in_bounds(g.shape, rr, cc):
                    continue
                if length > best_len[rr, cc] or (
                    length == best_len[rr, cc] and row < best_row[rr, cc]
                ):
                    best_len[rr, cc] = length
                    best_row[rr, cc] = row
                    out[rr, cc] = ray_color

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    dividers = sorted(random.sample(range(total + parts - 1), parts - 1))
    result: list[int] = []
    prev = -1
    for divider in dividers + [total + parts - 1]:
        result.append(divider - prev - 1)
        prev = divider
    return result


def _sample_segment_lengths() -> list[int]:
    choices = [count for count, _weight in SEGMENT_COUNT_WEIGHTS]
    weights = [weight for _count, weight in SEGMENT_COUNT_WEIGHTS]
    for _ in range(200):
        segment_count = random.choices(choices, weights=weights, k=1)[0]
        lengths = [random.randint(SEGMENT_LENGTH_MIN, SEGMENT_LENGTH_MAX) for _ in range(segment_count)]
        total = sum(lengths)
        if MIN_TOTAL_AXIS_PIXELS <= total <= MAX_TOTAL_AXIS_PIXELS:
            return lengths
    raise RuntimeError("failed to sample valid axis run lengths")


def _place_segment_starts(height: int, lengths: list[int]) -> list[int] | None:
    required = 2 + sum(lengths) + (len(lengths) - 1)
    if required > height:
        return None
    extras = _random_composition(height - required, len(lengths) + 1)
    starts: list[int] = []
    row = 1 + extras[0]
    for i, length in enumerate(lengths):
        starts.append(row)
        row += length
        if i < len(lengths) - 1:
            row += 1 + extras[i + 1]
    return starts


def _safe_distances(axis_col: int, width: int, side: str) -> list[int]:
    max_distance = axis_col if side == "left" else width - 1 - axis_col
    return [distance for distance in range(SAFE_MARKER_DISTANCE, max_distance + 1)]


def _paint_marker(grid: np.ndarray, row: int, axis_col: int, side: str, distance: int, color: int) -> None:
    col = axis_col - distance if side == "left" else axis_col + distance
    assert int(grid[row, col]) == BACKGROUND
    grid[row, col] = int(color)


def _add_row_markers(grid: np.ndarray, axis_rows: list[dict[str, int]], axis_col: int) -> None:
    width = int(grid.shape[1])
    for source in axis_rows:
        row = int(source["row"])
        left_distances = _safe_distances(axis_col, width, "left")
        right_distances = _safe_distances(axis_col, width, "right")
        use_tie = bool(left_distances and right_distances and random.random() < TIE_PROBABILITY)

        if use_tie:
            distance = random.choice(sorted(set(left_distances) & set(right_distances)))
            left_color, right_color = random.sample(RAY_COLOR_POOL, 2)
            _paint_marker(grid, row, axis_col, "left", distance, left_color)
            _paint_marker(grid, row, axis_col, "right", distance, right_color)
            continue

        available_sides = [side for side, distances in (("left", left_distances), ("right", right_distances)) if distances]
        side = random.choice(available_sides)
        distances = left_distances if side == "left" else right_distances
        distance = random.choice(distances)
        color = random.choice(RAY_COLOR_POOL)
        _paint_marker(grid, row, axis_col, side, distance, color)

        other_side = "right" if side == "left" else "left"
        other_distances = right_distances if other_side == "right" else left_distances
        farther = [d for d in other_distances if d > distance]
        if farther and random.random() < DECOY_PROBABILITY:
            decoy_color = random.choice(RAY_COLOR_POOL)
            _paint_marker(grid, row, axis_col, other_side, random.choice(farther), decoy_color)


def _add_noise_pixels(grid: np.ndarray, protected_rows: set[int], axis_col: int, target_non_axis: int) -> None:
    h, w = map(int, grid.shape)
    current_non_axis = int(np.count_nonzero(grid) - np.count_nonzero(grid[:, axis_col]))
    candidates = [
        (r, c)
        for r in range(h)
        if r not in protected_rows
        for c in range(w)
        if c != axis_col and abs(c - axis_col) >= SAFE_MARKER_DISTANCE and int(grid[r, c]) == BACKGROUND
    ]
    random.shuffle(candidates)

    for r, c in candidates:
        if current_non_axis >= target_non_axis:
            break
        grid[r, c] = random.choice(RAY_COLOR_POOL)
        current_non_axis += 1


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.choice(WIDTH_OPTIONS)
    axis_col = width // 2

    lengths = _sample_segment_lengths()
    starts = _place_segment_starts(height, lengths)
    if starts is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    axis_colors = random.sample(AXIS_COLOR_POOL, k=len(lengths))
    axis_rows: list[dict[str, int]] = []

    for start, length, color in zip(starts, lengths, axis_colors):
        for row in range(start, start + length):
            grid[row, axis_col] = color
            axis_rows.append({"row": row, "length": length, "axis_color": color})

    _add_row_markers(grid, axis_rows, axis_col)

    target_non_axis = random.randint(
        max(TARGET_NON_AXIS_RANGE[0], int(np.count_nonzero(grid) - np.count_nonzero(grid[:, axis_col]))),
        max(TARGET_NON_AXIS_RANGE[1], int(np.count_nonzero(grid) - np.count_nonzero(grid[:, axis_col]))),
    )
    _add_noise_pixels(grid, {int(source["row"]) for source in axis_rows}, axis_col, target_non_axis)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out[:, :axis_col])) + int(np.count_nonzero(out[:, axis_col + 1 :])) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
