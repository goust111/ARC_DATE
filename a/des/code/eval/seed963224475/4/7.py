import random

import numpy as np


BG = 0
AZURE = 8
MAGENTA = 6
MARKER_COLORS = (AZURE, MAGENTA)
STATE_HORIZONTAL = 0
STATE_VERTICAL = 1

STRIP_COLORS = (1, 2, 3, 4, 5, 7, 9)
COUNT_OPTIONS = [
    (n_h, n_v)
    for n_h in range(2, 5)
    for n_v in range(2, 5)
    if n_h + n_v <= len(STRIP_COLORS)
]
GAP_CHOICES = (2, 3)
EXTENSION_CHOICES = (1, 2, 3)
PADDING_CHOICES = (0, 1, 2)
MAX_GENERATION_ATTEMPTS = 300


def _flip_state(state: int) -> int:
    return STATE_VERTICAL if state == STATE_HORIZONTAL else STATE_HORIZONTAL


def _marker_to_state(color: int) -> int:
    if color == AZURE:
        return STATE_HORIZONTAL
    if color == MAGENTA:
        return STATE_VERTICAL
    raise AssertionError(f"unexpected marker color {color}")


def _state_to_marker(state: int) -> int:
    return AZURE if state == STATE_HORIZONTAL else MAGENTA


def _extract_lattice(grid: np.ndarray) -> tuple[list[int], list[int]]:
    marker_mask = np.isin(grid, MARKER_COLORS)
    marker_coords = np.argwhere(marker_mask)
    assert len(marker_coords) > 0, "expected at least one weaving marker"

    rows = sorted({int(r) - 1 for r, _c in marker_coords})
    cols = sorted({int(c) - 1 for _r, c in marker_coords})
    assert rows and cols, "expected at least one horizontal and one vertical strip"
    assert min(rows) >= 0 and min(cols) >= 0, "markers must sit down-right of intersections"

    actual = {tuple(map(int, coord)) for coord in marker_coords}
    expected = {(r + 1, c + 1) for r in rows for c in cols}
    assert actual == expected, "markers must occupy the full translated crossing lattice"
    return rows, cols


def _single_strip_color(values: np.ndarray) -> int:
    colors = {
        int(value)
        for value in np.asarray(values).reshape(-1)
        if int(value) != BG and int(value) not in MARKER_COLORS
    }
    assert len(colors) == 1, f"expected one strip color, got {colors}"
    return next(iter(colors))


def _infer_horizontal_colors(grid: np.ndarray, rows: list[int], cols: list[int]) -> dict[int, int]:
    col_mask = np.ones(grid.shape[1], dtype=bool)
    col_mask[cols] = False
    colors = {int(r): _single_strip_color(grid[int(r), col_mask]) for r in rows}
    return colors


def _infer_vertical_colors(grid: np.ndarray, rows: list[int], cols: list[int]) -> dict[int, int]:
    row_mask = np.ones(grid.shape[0], dtype=bool)
    row_mask[rows] = False
    colors = {int(c): _single_strip_color(grid[row_mask, int(c)]) for c in cols}
    return colors


def _repair_row_states(raw_states: list[int]) -> list[int]:
    assert raw_states, "each strip row must have at least one crossing"
    repaired = [int(raw_states[0])]
    for state in raw_states[1:]:
        repaired.append(int(state) if int(state) != repaired[-1] else _flip_state(int(state)))
    for left, right in zip(repaired, repaired[1:]):
        assert left != right, "repaired weaving states must alternate"
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Correct a marked weaving pattern and render the visible crossing colors.

    Input encoding used by the generator:
    - black background
    - one-pixel horizontal and vertical strips, each with a unique non-marker color
    - every crossing has a marker at its down-right diagonal neighbor
    - the input crossing pixel is rendered from the raw marker state

    Rule:
    - azure marker => horizontal strip is on top at that crossing
    - magenta marker => vertical strip is on top at that crossing
    - along each horizontal strip, repaired top/bottom states must strictly alternate;
      when a marker repeats the previous repaired state, flip that marker state
    - remove all markers after rendering the repaired crossings
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rows, cols = _extract_lattice(g)
    horizontal_colors = _infer_horizontal_colors(g, rows, cols)
    vertical_colors = _infer_vertical_colors(g, rows, cols)

    strip_colors = list(horizontal_colors.values()) + list(vertical_colors.values())
    assert len(set(strip_colors)) == len(strip_colors), "all strips must use distinct colors"

    out = g.copy()
    out[np.isin(out, MARKER_COLORS)] = BG

    for r in rows:
        raw_states = [_marker_to_state(int(g[r + 1, c + 1])) for c in cols]
        repaired_states = _repair_row_states(raw_states)
        for c, state in zip(cols, repaired_states):
            h_color = int(horizontal_colors[r])
            v_color = int(vertical_colors[c])
            assert int(g[r, c]) in {h_color, v_color}, "crossing must show one of its strip colors"
            out[r, c] = h_color if state == STATE_HORIZONTAL else v_color

    return out


def _choose_axis_positions(count: int) -> list[int]:
    positions = [random.randint(1, 3)]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.choice(GAP_CHOICES))
    return positions


def _sample_raw_states(n_h: int, n_v: int) -> list[list[int]]:
    raw_rows: list[list[int]] = []
    final_rows: list[list[int]] = []
    invalid_positions: list[tuple[int, int]] = []

    for row_idx in range(n_h):
        start_state = random.choice((STATE_HORIZONTAL, STATE_VERTICAL))
        final_row = [
            start_state if col_idx % 2 == 0 else _flip_state(start_state)
            for col_idx in range(n_v)
        ]
        raw_row = [final_row[0]]
        for col_idx in range(1, n_v):
            invalid = random.random() < 0.5
            if invalid:
                raw_row.append(final_row[col_idx - 1])
                invalid_positions.append((row_idx, col_idx))
            else:
                raw_row.append(final_row[col_idx])
        raw_rows.append(raw_row)
        final_rows.append(final_row)

    if not invalid_positions and random.random() < 0.75:
        row_idx = random.randrange(n_h)
        col_idx = random.randrange(1, n_v)
        raw_rows[row_idx][col_idx] = final_rows[row_idx][col_idx - 1]

    return raw_rows


def _validate_generated_input(grid: np.ndarray) -> None:
    rows, cols = _extract_lattice(grid)
    horizontal_colors = _infer_horizontal_colors(grid, rows, cols)
    vertical_colors = _infer_vertical_colors(grid, rows, cols)
    strip_colors = list(horizontal_colors.values()) + list(vertical_colors.values())
    assert len(set(strip_colors)) == len(strip_colors)
    for r in rows:
        for c in cols:
            assert int(grid[r, c]) in {int(horizontal_colors[r]), int(vertical_colors[c])}
            assert int(grid[r + 1, c + 1]) in MARKER_COLORS


def _gen_attempt() -> np.ndarray:
    n_h, n_v = random.choice(COUNT_OPTIONS)
    colors = random.sample(STRIP_COLORS, n_h + n_v)
    horizontal_colors = colors[:n_h]
    vertical_colors = colors[n_h:]

    rows = _choose_axis_positions(n_h)
    cols = _choose_axis_positions(n_v)

    top_edge = rows[0] - random.choice(EXTENSION_CHOICES)
    left_edge = cols[0] - random.choice(EXTENSION_CHOICES)
    if top_edge < 0 or left_edge < 0:
        return None

    bottom_edge = rows[-1] + random.choice(EXTENSION_CHOICES)
    right_edge = cols[-1] + random.choice(EXTENSION_CHOICES)

    height = bottom_edge + 1 + random.choice(PADDING_CHOICES)
    width = right_edge + 1 + random.choice(PADDING_CHOICES)
    grid = np.full((height, width), BG, dtype=int)

    raw_rows = _sample_raw_states(n_h, n_v)

    for r, color in zip(rows, horizontal_colors):
        grid[r, left_edge : right_edge + 1] = int(color)
    for c, color in zip(cols, vertical_colors):
        grid[top_edge : bottom_edge + 1, c] = int(color)

    for row_idx, r in enumerate(rows):
        for col_idx, c in enumerate(cols):
            state = int(raw_rows[row_idx][col_idx])
            grid[r, c] = int(horizontal_colors[row_idx]) if state == STATE_HORIZONTAL else int(vertical_colors[col_idx])
            marker_r, marker_c = r + 1, c + 1
            assert int(grid[marker_r, marker_c]) == BG, "marker cells must stay separate from strips"
            grid[marker_r, marker_c] = _state_to_marker(state)

    _validate_generated_input(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
