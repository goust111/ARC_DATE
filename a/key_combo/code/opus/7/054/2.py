from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MARKER_COLORS = (BLUE, RED)
GEN_RETRIES = 300


def _extract_bar_and_markers(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    cols = np.flatnonzero(np.any(grid != BLACK, axis=0))
    assert len(cols) == 1, "expected exactly one occupied column"
    bar_col = int(cols[0])

    column = grid[:, bar_col]
    occupied_rows = np.flatnonzero(column != BLACK)
    assert len(occupied_rows) > 0, "expected a non-empty vertical bar"
    assert int(occupied_rows[0]) == 0, "expected a top-anchored bar"
    assert np.array_equal(occupied_rows, np.arange(int(occupied_rows[-1]) + 1)), "bar must be contiguous"

    bar_height = int(occupied_rows[-1]) + 1
    column_colors = set(map(int, np.unique(column[:bar_height])))
    assert column_colors <= {GREY, *MARKER_COLORS}, "unexpected bar colors"

    marker_rows = [int(r) for r in occupied_rows if int(column[r]) in MARKER_COLORS]
    assert 2 <= len(marker_rows) <= 4, "expected 2-4 markers"
    assert all(b - a >= 2 for a, b in zip(marker_rows, marker_rows[1:])), "markers must be non-adjacent"

    markers = [(row, int(column[row])) for row in marker_rows]
    return bar_col, bar_height, markers


def _build_output(grid: np.ndarray, bar_col: int, bar_height: int, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape

    top_color = markers[0][1]

    for col in range(bar_col - 1, -1, -1):
        distance = bar_col - col
        fill_height = min(height, bar_height + 2 * distance)
        out[:fill_height, col] = top_color

    for col in range(bar_col + 1, width):
        distance = col - bar_col
        fill_height = bar_height - 2 * distance
        if fill_height <= 0:
            continue
        out[: min(height, fill_height), col] = top_color

    for row, color in markers:
        out[row, :bar_col] = color
        out[row, bar_col + 1 :] = color

    out[:, bar_col] = grid[:, bar_col]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit horizontal marker rays from a central bar and use their reach to drive asymmetric top-down side fills.

    Input structure:
    - A black background contains exactly one top-anchored one-cell-wide vertical bar in a single column.
    - The bar occupies rows 0..H-1 and is grey except for 2-4 red or blue marker cells embedded in that column.
    - Markers lie on distinct non-adjacent rows, and there is background space on both sides of the bar.

    Rule:
    - Read the occupied column as the source bar, with its contiguous top segment defining bar height H.
    - Each marker emits a horizontal ray of its own color to both grid edges on that row, leaving the bar column unchanged.
    - For every column d cells left of the bar, fill from the top for H+2d cells; for every column d cells right of the bar, fill from the top for H-2d cells when positive. When several rays trigger the same side column, use the color of the topmost triggering marker; then redraw all horizontal rays so each marker row keeps its own color.

    Key insight:
    - The marker rows supply color-coded horizontal selectors, while the bar-to-edge distance of each side column determines an asymmetric vertical fill profile anchored at the top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREY}, "unexpected input colors"

    bar_col, bar_height, markers = _extract_bar_and_markers(g)
    return _build_output(g, bar_col, bar_height, markers)


def _sample_nonadjacent_rows(bar_height: int, count: int) -> list[int] | None:
    choices = list(range(bar_height))
    picked: list[int] = []
    for _ in range(count):
        valid = [row for row in choices if all(abs(row - prev) >= 2 for prev in picked)]
        if not valid:
            return None
        picked.append(random.choice(valid))
    picked.sort()
    return picked


def _sample_marker_colors(count: int) -> list[int]:
    top_color = random.choice(MARKER_COLORS)
    colors = [top_color]
    if count == 1:
        return colors

    if random.random() < 0.65:
        alternate = RED if top_color == BLUE else BLUE
        other_colors = [alternate]
        while len(other_colors) < count - 1:
            other_colors.append(random.choice(MARKER_COLORS))
        random.shuffle(other_colors)
        colors.extend(other_colors[: count - 1])
    else:
        colors.extend(random.choice(MARKER_COLORS) for _ in range(count - 1))
    return colors


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(11, 18)
    width = random.randint(11, 17)
    bar_col = random.randint(3, width - 4)
    bar_height = random.randint(4, min(7, height - 2))

    max_markers = min(4, (bar_height + 1) // 2)
    if max_markers < 2:
        return None
    marker_count = random.randint(2, max_markers)
    marker_rows = _sample_nonadjacent_rows(bar_height, marker_count)
    if marker_rows is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:bar_height, bar_col] = GREY
    for row, color in zip(marker_rows, _sample_marker_colors(marker_count)):
        grid[row, bar_col] = color

    expected = _build_output(grid, bar_col, bar_height, [(row, int(grid[row, bar_col])) for row in marker_rows])
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
