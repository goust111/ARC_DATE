from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import itertools
import random


def _marker_column(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = []
    for c in range(w):
        if all(np.unique(np.delete(g[r], c)).size == 1 for r in range(h)):
            candidates.append(c)
    assert len(candidates) == 1, "expected one marker column"
    return int(candidates[0])


def _source_bars(g: np.ndarray, bg: int, marker_c: int) -> tuple[list[tuple[int, int]], int, int]:
    h = g.shape[0]
    row_colors = []
    for r in range(h):
        outside = np.delete(g[r], marker_c)
        vals = np.unique(outside)
        assert vals.size == 1, "each row must be monochrome outside the marker column"
        row_colors.append(int(vals[0]))

    bg_rows = [r for r, color in enumerate(row_colors) if color == bg]
    source_rows = [r for r, color in enumerate(row_colors) if color != bg]
    assert bg_rows and source_rows, "expected background rows and source bar rows"

    bg_markers = {int(g[r, marker_c]) for r in bg_rows}
    source_markers = {int(g[r, marker_c]) for r in source_rows}
    assert len(bg_markers) == 1, "background rows must use one marker color"
    assert len(source_markers) == 1, "source bars must use one marker color"

    bg_marker = bg_markers.pop()
    source_marker = source_markers.pop()
    assert bg_marker != source_marker, "the two marker roles must be distinct"

    marker_colors = {bg_marker, source_marker}
    bars = [(r, row_colors[r]) for r in source_rows]
    assert all(color not in marker_colors for _r, color in bars), "bar colors must be distinct from marker colors"
    return bars, source_marker, bg_marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every row from the nearest marked horizontal bar, turning bar rows and color-boundary ties into separator rows.

    Input structure
    - A dominant background fills the grid except for full-width horizontal bars and one vertical marker column.
    - Outside the marker column, every row is monochrome: either background or a source bar color.
    - Background rows share one marker color in the marker column; source bar rows share the other.

    Rule
    - Each non-source row takes the color of the nearest source bar, with that row's marker-column pixel set to the source-marker color.
    - If a row is exactly midway between nearest bars of different colors, make the whole row the source-marker color.
    - Each original source bar row becomes a source-marker row with the marker-column pixel changed to the background-row marker color.

    Key insight
    - The marked bars are horizontal seeds for a one-dimensional nearest-seed partition over rows; equal-distance rows are separators only when the competing seed colors differ.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker_c = _marker_column(g)
    bars, source_marker, bg_marker = _source_bars(g, bg, marker_c)
    source_rows = {r for r, _color in bars}
    assert len(source_rows) == len(bars), "source bar rows must be unique"

    out = np.empty_like(g)
    for r in range(g.shape[0]):
        if r in source_rows:
            out[r, :] = source_marker
            out[r, marker_c] = bg_marker
            continue

        nearest_distance = min(abs(r - bar_r) for bar_r, _color in bars)
        nearest_colors = [color for bar_r, color in bars if abs(r - bar_r) == nearest_distance]
        unique_nearest_colors = set(nearest_colors)
        if len(unique_nearest_colors) == 1:
            out[r, :] = nearest_colors[0]
            out[r, marker_c] = source_marker
        else:
            assert len(nearest_colors) == 2, "row ties must be between two nearest bars"
            out[r, :] = source_marker

    return out

# --------------- generator (o5x) ---------------
# Canvas and fixed task colors.
CANVAS_H, CANVAS_W = 16, 16
BACKGROUND_COLOR, BACKGROUND_MARKER_COLOR, SOURCE_MARKER_COLOR = 7, 8, 1
BAR_PALETTE = [0, 2, 3, 4, 5, 6, 9]

# Source-bar count, row-spacing, and retry bounds.
BAR_COUNT_MIN, BAR_COUNT_MAX = 2, 4
BAR_ROW_MIN, BAR_ROW_MAX = 1, CANVAS_H - 2
BAR_GAP_MIN, BAR_GAP_MAX = 1, 8
BAR_COLOR_COUNT_MIN, BAR_COLOR_COUNT_MAX = 2, 3
GENERATION_RETRIES = 300

VALID_BAR_ROWS_BY_COUNT = {
    num_bars: tuple(
        rows
        for rows in itertools.combinations(range(BAR_ROW_MIN, BAR_ROW_MAX + 1), num_bars)
        if all(BAR_GAP_MIN <= rows[i + 1] - rows[i] - 1 <= BAR_GAP_MAX for i in range(num_bars - 1))
    )
    for num_bars in range(BAR_COUNT_MIN, BAR_COUNT_MAX + 1)
}


def _contiguous_run_lengths(total: int, num_runs: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), num_runs - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _bar_color_sequence(num_bars: int) -> list[int]:
    num_colors = (
        BAR_COLOR_COUNT_MIN if num_bars == BAR_COUNT_MIN
        else random.randint(BAR_COLOR_COUNT_MIN, min(BAR_COLOR_COUNT_MAX, num_bars))
    )
    colors = random.sample(BAR_PALETTE, num_colors)
    sizes = _contiguous_run_lengths(num_bars, num_colors)

    random.shuffle(colors)
    seq: list[int] = []
    for color, size in zip(colors, sizes):
        seq.extend([color] * size)
    return seq


def _render_input(bar_rows: tuple[int, ...], bar_colors: list[int], marker_c: int) -> np.ndarray:
    grid = np.full((CANVAS_H, CANVAS_W), BACKGROUND_COLOR, dtype=int)
    grid[:, marker_c] = BACKGROUND_MARKER_COLOR
    for r, color in zip(bar_rows, bar_colors):
        grid[r, :] = color
        grid[r, marker_c] = SOURCE_MARKER_COLOR
    return grid


def _expected_output(bar_rows: tuple[int, ...], bar_colors: list[int], marker_c: int) -> np.ndarray:
    out = np.empty((CANVAS_H, CANVAS_W), dtype=int)
    bar_by_row = dict(zip(bar_rows, bar_colors))

    for r in range(CANVAS_H):
        if r in bar_by_row:
            out[r, :] = SOURCE_MARKER_COLOR
            out[r, marker_c] = BACKGROUND_MARKER_COLOR
            continue

        nearest_distance = min(abs(r - bar_r) for bar_r in bar_rows)
        nearest_colors = {
            color
            for bar_r, color in zip(bar_rows, bar_colors)
            if abs(r - bar_r) == nearest_distance
        }
        if len(nearest_colors) == 1:
            out[r, :] = next(iter(nearest_colors))
            out[r, marker_c] = SOURCE_MARKER_COLOR
        else:
            out[r, :] = SOURCE_MARKER_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    marker_c = random.randrange(CANVAS_W)
    num_bars = random.randint(BAR_COUNT_MIN, BAR_COUNT_MAX)
    bar_rows = random.choice(VALID_BAR_ROWS_BY_COUNT[num_bars])
    bar_colors = _bar_color_sequence(num_bars)

    grid = _render_input(bar_rows, bar_colors, marker_c)
    expected = _expected_output(bar_rows, bar_colors, marker_c)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
