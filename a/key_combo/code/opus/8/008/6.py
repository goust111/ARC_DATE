from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

DIR_BY_COLOR = {
    BLUE: (0, 1),
    RED: (0, -1),
    GREEN: (1, 0),
    MAGENTA: (-1, 0),
}

HORIZONTAL_COLORS = {BLUE, RED}
VERTICAL_COLORS = {GREEN, MAGENTA}
INPUT_COLOR_OPTIONS = (
    (BLUE, RED, GREEN, MAGENTA),
    (BLUE, RED, GREEN),
    (BLUE, RED, MAGENTA),
    (BLUE, GREEN, MAGENTA),
    (RED, GREEN, MAGENTA),
)
INPUT_COLOR_WEIGHTS = (6, 2, 2, 2, 2)
MAX_GENERATION_ATTEMPTS = 300


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BACKGROUND)]
    markers.sort()
    return markers


def _ray_path(marker: tuple[int, int, int], shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    r, c, color = marker
    dr, dc = DIR_BY_COLOR[color]
    height, width = map(int, shape)
    step = 0
    path: list[tuple[int, int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        path.append((r, c, step))
        r += dr
        c += dc
        step += 1
    return path


def _stripe_color(marker_color: int, step: int) -> int:
    return marker_color if step % 2 == 0 else GREY


def _base_output_and_intersections(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    markers = _extract_markers(grid)

    horiz_first: dict[tuple[int, int], int] = {}
    vert_first: dict[tuple[int, int], int] = {}

    for marker in markers:
        color = marker[2]
        if color not in DIR_BY_COLOR:
            raise AssertionError("unexpected marker color")
        target = horiz_first if color in HORIZONTAL_COLORS else vert_first
        for r, c, step in _ray_path(marker, grid.shape):
            target.setdefault((r, c), _stripe_color(color, step))

    intersections = set(horiz_first) & set(vert_first)
    out = np.full((height, width), BACKGROUND, dtype=int)
    for cell in set(horiz_first) | set(vert_first):
        if cell in intersections:
            out[cell] = YELLOW
        elif cell in horiz_first:
            out[cell] = horiz_first[cell]
        else:
            out[cell] = vert_first[cell]
    return out, intersections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend color-coded rays with odd-even striping, mark horizontal/vertical crossings yellow, then emit down-left yellow diagonals.

    Input structure:
    - The grid is mostly black background with 3 or 4 isolated single-pixel markers.
    - Marker colors encode ray directions: blue points right, red points left, green points down, and magenta points up.
    - Grey and yellow do not appear in the input; every non-black pixel is one of the direction markers.

    Rule:
    - From each marker, extend a ray in its color-specific direction until the grid boundary.
    - Along each ray, keep the marker cell in its own color and alternate thereafter between grey on odd steps and the marker color on even steps.
    - Any cell occupied by at least one horizontal ray and at least one vertical ray becomes yellow.
    - From every yellow crossing cell, extend a solid yellow diagonal toward down-left, overwriting anything already on that path.

    Key insight:
    - The hidden part is the color-to-direction mapping; once that is known, crossings are determined by row/column alignment, and those crossings seed the final yellow diagonals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"

    out, intersections = _base_output_and_intersections(g)
    height, width = map(int, g.shape)
    for r0, c0 in intersections:
        r, c = int(r0), int(c0)
        while 0 <= r < height and 0 <= c < width:
            out[r, c] = YELLOW
            r += 1
            c -= 1
    return out


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(100):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_layout(height: int, width: int, colors: tuple[int, ...]) -> tuple[list[int], list[int]] | None:
    n_h = sum(color in HORIZONTAL_COLORS for color in colors)
    n_v = sum(color in VERTICAL_COLORS for color in colors)
    for _ in range(100):
        rows = _sample_spaced_values(4, height - 5, n_h, 3)
        cols = _sample_spaced_values(4, width - 5, n_v, 3)
        if rows is None or cols is None:
            continue
        if len(rows) == 2 and len(cols) == 2 and (rows[1] - rows[0] == cols[1] - cols[0]):
            continue
        return rows, cols
    return None


def _pick_in_range(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_marker_grid(height: int, width: int, colors: tuple[int, ...]) -> np.ndarray | None:
    layout = _sample_layout(height, width, colors)
    if layout is None:
        return None
    rows, cols = layout
    horiz_colors = [color for color in colors if color in HORIZONTAL_COLORS]
    vert_colors = [color for color in colors if color in VERTICAL_COLORS]
    random.shuffle(horiz_colors)
    random.shuffle(vert_colors)

    row_by_color = {color: rows[idx] for idx, color in enumerate(horiz_colors)}
    col_by_color = {color: cols[idx] for idx, color in enumerate(vert_colors)}

    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    positions: dict[int, tuple[int, int]] = {}
    if BLUE in colors:
        c = _pick_in_range(1, min_col - 2)
        if c is None:
            return None
        positions[BLUE] = (row_by_color[BLUE], c)
    if RED in colors:
        c = _pick_in_range(max_col + 2, width - 2)
        if c is None:
            return None
        positions[RED] = (row_by_color[RED], c)
    if GREEN in colors:
        r = _pick_in_range(1, min_row - 2)
        if r is None:
            return None
        positions[GREEN] = (r, col_by_color[GREEN])
    if MAGENTA in colors:
        r = _pick_in_range(max_row + 2, height - 2)
        if r is None:
            return None
        positions[MAGENTA] = (r, col_by_color[MAGENTA])

    if len(set(positions.values())) != len(positions):
        return None

    grid = np.zeros((height, width), dtype=int)
    for color, (r, c) in positions.items():
        grid[int(r), int(c)] = int(color)
    return grid


def _count_crossings(grid: np.ndarray) -> int:
    _, intersections = _base_output_and_intersections(grid)
    return len(intersections)


def _is_valid_input(grid: np.ndarray, colors: tuple[int, ...]) -> bool:
    if grid is None:
        return False
    if np.count_nonzero(grid) != len(colors):
        return False
    if any(color in (YELLOW, GREY) for color in np.unique(grid)):
        return False
    if _count_crossings(grid) < 2:
        return False
    out = main(grid)
    yellow_ratio = float(np.mean(out == YELLOW))
    if yellow_ratio > 0.35:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(13, 20)
        width = random.randint(13, 20)
        colors = random.choices(INPUT_COLOR_OPTIONS, weights=INPUT_COLOR_WEIGHTS, k=1)[0]
        grid = _build_marker_grid(height, width, colors)
        if _is_valid_input(grid, colors):
            return grid
    raise RuntimeError("failed to generate a valid input")
