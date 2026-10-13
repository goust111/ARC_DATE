import random
from collections import Counter

import numpy as np


COLOR_VALUES = tuple(range(10))
GRID_SIZE_RANGE = (9, 24)
STRIPE_COUNT_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 300
NON_ADJACENT_SAMPLE_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(np.asarray(grid, dtype=int)) if int(v) not in blocked]


def _majority_color(values) -> int:
    counts = Counter(int(v) for v in values)
    assert counts, "expected at least one color"
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _sample_non_adjacent(candidates: list[int], count: int) -> list[int] | None:
    if count == 0:
        return []
    if count > len(candidates):
        return None

    for _ in range(NON_ADJACENT_SAMPLE_ATTEMPTS):
        picked = sorted(random.sample(candidates, count))
        if all(b - a > 1 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _render_input(
    shape: tuple[int, int],
    horizontal_rows: list[int],
    vertical_cols: list[int],
    bg: int,
    h_color: int,
    v_color: int,
    marker_color: int,
    marked_h_rows: set[int],
    marked_v_cols: set[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[:, vertical_cols] = v_color
    out[horizontal_rows, :] = h_color
    for row in marked_h_rows:
        out[row, 0] = marker_color
    for col in marked_v_cols:
        out[0, col] = marker_color
    return out


def _render_woven(
    shape: tuple[int, int],
    horizontal_rows: list[int],
    vertical_cols: list[int],
    bg: int,
    h_color: int,
    v_color: int,
    marked_h_rows: set[int],
    marked_v_cols: set[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[:, vertical_cols] = v_color
    out[horizontal_rows, :] = h_color

    for row_index, row in enumerate(horizontal_rows):
        h_over_0 = row not in marked_h_rows
        for col_index, col in enumerate(vertical_cols):
            v_over_0 = col not in marked_v_cols
            h_over = h_over_0 ^ (col_index % 2 == 1)
            v_over = v_over_0 ^ (row_index % 2 == 1)
            out[row, col] = v_color if v_over and not h_over else h_color

    return out


def _infer_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    horizontal_rows = [r for r in range(h) if np.all(g[r] != bg)]
    vertical_cols = [c for c in range(w) if np.all(g[:, c] != bg)]
    assert horizontal_rows, "expected at least one horizontal strip"
    assert vertical_cols, "expected at least one vertical strip"

    cross_values = [int(v) for v in g[np.ix_(horizontal_rows, vertical_cols)].reshape(-1) if int(v) != bg]
    h_color = _majority_color(cross_values)

    body_rows = [r for r in range(h) if r not in set(horizontal_rows)]
    assert body_rows, "expected at least one non-horizontal row"
    vertical_values = [
        int(v)
        for v in g[np.ix_(body_rows, vertical_cols)].reshape(-1)
        if int(v) != bg and int(v) != h_color
    ]
    assert vertical_values, "expected visible vertical-strip cells"
    v_color = _majority_color(vertical_values)
    assert h_color != v_color, "horizontal and vertical colors must differ"

    fg_colors = colors_present(g, exclude={bg})
    marker_candidates = [color for color in fg_colors if color not in {h_color, v_color}]
    assert len(marker_candidates) <= 1, "expected at most one marker color"
    marker_color = marker_candidates[0] if marker_candidates else None

    allowed_h = {h_color} if marker_color is None else {h_color, marker_color}
    for row in horizontal_rows:
        assert set(map(int, g[row])) <= allowed_h, "horizontal rows should contain only strip color and marker"

    allowed_v = {v_color} if marker_color is None else {v_color, marker_color}
    for col in vertical_cols:
        column_values = set(map(int, g[body_rows, col]))
        assert column_values <= allowed_v, "vertical columns should contain only strip color and marker off crossings"

    marked_h_rows = set()
    marked_v_cols = set()
    if marker_color is not None:
        marker_positions = {tuple(map(int, pos)) for pos in np.argwhere(g == marker_color)}
        expected_marker_positions = {(row, 0) for row in horizontal_rows} | {(0, col) for col in vertical_cols}
        assert marker_positions <= expected_marker_positions, "markers must live on strip start pixels"

        marked_h_rows = {row for row in horizontal_rows if int(g[row, 0]) == marker_color}
        marked_v_cols = {col for col in vertical_cols if int(g[0, col]) == marker_color}

    return bg, horizontal_rows, vertical_cols, h_color, v_color, marked_h_rows, marked_v_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw edge-marked horizontal and vertical strips as an alternating woven crossing pattern.

    Input structure:
    - The background is the dominant color.
    - Several one-cell horizontal strips span the full width, all sharing one color.
    - Several one-cell vertical strips span the full height, all sharing a second color.
    - Some horizontal strip starts on the left edge and some vertical strip starts on the top edge are replaced by one marker color.
    - Every crossing in the input is shown as the horizontal-strip color.

    Rule:
    - Detect the full-width strip rows and full-height strip columns.
    - A marked horizontal start means that strip begins under; an unmarked one begins over.
    - A marked vertical start means that strip begins under; an unmarked one begins over.
    - Each strip flips its over/under state after every crossing along its travel direction.
    - Restore the marker cells to their strip colors and redraw each crossing; show the vertical color only when the vertical strip is over and the horizontal strip is under, otherwise show the horizontal color.

    Key insight:
    - The border markers encode each strip's initial layer, and the woven pattern comes from alternating that layer at successive crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, horizontal_rows, vertical_cols, h_color, v_color, marked_h_rows, marked_v_cols = _infer_scene(g)
    return _render_woven(
        g.shape,
        horizontal_rows,
        vertical_cols,
        bg,
        h_color,
        v_color,
        marked_h_rows,
        marked_v_cols,
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    max_h = min(STRIPE_COUNT_RANGE[1], (height - 1 + 1) // 2)
    max_v = min(STRIPE_COUNT_RANGE[1], (width - 1 + 1) // 2)
    if max_h < STRIPE_COUNT_RANGE[0] or max_v < STRIPE_COUNT_RANGE[0]:
        return None

    h_count = random.randint(STRIPE_COUNT_RANGE[0], max_h)
    v_count = random.randint(STRIPE_COUNT_RANGE[0], max_v)

    horizontal_rows = _sample_non_adjacent(list(range(1, height)), h_count)
    vertical_cols = _sample_non_adjacent(list(range(1, width)), v_count)
    if horizontal_rows is None or vertical_cols is None:
        return None

    marked_h_rows = set(random.sample(horizontal_rows, random.randint(1, h_count - 1)))
    marked_v_cols = set(random.sample(vertical_cols, random.randint(1, v_count - 1)))

    bg, h_color, v_color, marker_color = random.sample(COLOR_VALUES, 4)
    grid = _render_input(
        (height, width),
        horizontal_rows,
        vertical_cols,
        bg,
        h_color,
        v_color,
        marker_color,
        marked_h_rows,
        marked_v_cols,
    )

    color_counts = Counter(map(int, grid.reshape(-1)))
    if color_counts[bg] <= max(count for color, count in color_counts.items() if color != bg):
        return None

    expected = _render_woven(
        (height, width),
        horizontal_rows,
        vertical_cols,
        bg,
        h_color,
        v_color,
        marked_h_rows,
        marked_v_cols,
    )
    cross = expected[np.ix_(horizontal_rows, vertical_cols)]
    if not (np.any(cross == h_color) and np.any(cross == v_color)):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-strip input")
