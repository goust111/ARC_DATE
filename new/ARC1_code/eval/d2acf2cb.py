from __future__ import annotations

import numpy as np

from arc_utils import Color
from itertools import combinations


TOGGLE_COLOR = {
    Color.BLACK: Color.TEAL,
    Color.TEAL: Color.BLACK,
    Color.PINK: Color.ORANGE,
    Color.ORANGE: Color.PINK,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Toggle the colors on straight row or column segments bracketed by opposite-edge yellow markers.

    Input structure
    - Yellow marker pixels appear in aligned pairs on one pair of opposite grid edges.
    - A left-edge/right-edge marker pair selects that full row segment; a top-edge/bottom-edge marker pair selects that column segment.
    - The strict interior cells of selected segments use only black, teal, pink, and orange.

    Rule
    - Keep the yellow endpoint markers unchanged.
    - For every selected segment, recolor each strict interior cell by toggling black <-> teal and pink <-> orange.
    - Leave all cells outside the selected segments unchanged.

    Key insight
    - The marker pair orientation selects the path; the path cells are not filled with a constant color but have their two color pairs inverted in place.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2, "expected a grid with opposite edges"

    edge_rows = [r for r in range(h) if g[r, 0] == Color.YELLOW and g[r, w - 1] == Color.YELLOW]
    edge_cols = [c for c in range(w) if g[0, c] == Color.YELLOW and g[h - 1, c] == Color.YELLOW]
    assert bool(edge_rows) != bool(edge_cols), "expected yellow marker pairs on exactly one axis"

    expected_markers: set[tuple[int, int]] = set()
    if edge_rows:
        for r in edge_rows:
            expected_markers.add((r, 0))
            expected_markers.add((r, w - 1))
    else:
        for c in edge_cols:
            expected_markers.add((0, c))
            expected_markers.add((h - 1, c))

    actual_markers = {(int(r), int(c)) for r, c in np.argwhere(g == Color.YELLOW)}
    assert actual_markers == expected_markers, "expected all yellow markers to be paired on opposite edges"

    out = g.copy()
    if edge_rows:
        for r in edge_rows:
            for c in range(1, w - 1):
                color = int(g[r, c])
                assert color in TOGGLE_COLOR, "expected segment interior to use toggle colors"
                out[r, c] = TOGGLE_COLOR[color]
    else:
        for c in edge_cols:
            for r in range(1, h - 1):
                color = int(g[r, c])
                assert color in TOGGLE_COLOR, "expected segment interior to use toggle colors"
                out[r, c] = TOGGLE_COLOR[color]

    return out

# --------------- generator (o5x) ---------------

# Canvas dimensions.
GRID_HEIGHT_RANGE, GRID_WIDTH = (6, 10), 9
# Marker layout.
AXIS_ROW, AXIS_COL = "row", "col"
ROW_AXIS_PROB = 0.5
MARKER_COUNT_RANGE, MARKER_MIN_SPACING, INTERIOR_INSET = (1, 3), 2, 1
# Background and segment variation.
PINK_PROB_RANGE, PRE_TOGGLED_PROB = (0.35, 0.66), 0.18
SEGMENT_TOGGLE_LOOKUP = np.array([TOGGLE_COLOR.get(i, i) for i in range(Color.PAD + 1)], dtype=int)
# Validation retries.
MAX_ATTEMPTS = 300


def _gen_spaced_choices(limit: int, count: int, rng: np.random.Generator) -> list[int] | None:
    """Return one valid marker index combination, or None when impossible."""
    choices = range(INTERIOR_INSET, limit - INTERIOR_INSET)
    valid = [combo for combo in combinations(choices, count) if all(b - a >= MARKER_MIN_SPACING for a, b in zip(combo, combo[1:]))]
    return None if not valid else list(valid[int(rng.integers(len(valid)))])


def _choose_layout(rng: np.random.Generator) -> tuple[int, int, str, list[int]] | None:
    """Sample canvas size, marker axis, and spaced marker indices."""
    h = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = GRID_WIDTH
    axis = AXIS_ROW if rng.random() < ROW_AXIS_PROB else AXIS_COL
    marker_limit = h if axis == AXIS_ROW else w
    count_min, count_max = MARKER_COUNT_RANGE
    feasible_counts = [n for n in range(count_min, count_max + 1) if _gen_spaced_choices(marker_limit, n, rng) is not None]
    if not feasible_counts:
        return None
    count = int(rng.choice(feasible_counts))
    return h, w, axis, _gen_spaced_choices(marker_limit, count, rng)


def _segment_rc(shape: tuple[int, int], indices: list[int], axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return broadcast row/column arrays for selected segment interiors."""
    idx = np.asarray(indices)
    if axis == AXIS_ROW:
        return idx[:, None], np.arange(INTERIOR_INSET, shape[1] - INTERIOR_INSET)
    return np.arange(INTERIOR_INSET, shape[0] - INTERIOR_INSET)[:, None], idx


def _render_input(layout: tuple[int, int, str, list[int]], rng: np.random.Generator) -> np.ndarray:
    """Build an input grid for a sampled marker layout."""
    h, w, axis, indices = layout
    pink_prob = float(rng.uniform(*PINK_PROB_RANGE))
    grid = np.where(rng.random((h, w)) < pink_prob, Color.PINK, Color.BLACK).astype(int)
    if axis == AXIS_ROW:
        grid[np.ix_(indices, [0, w - 1])] = Color.YELLOW
    else:
        grid[np.ix_([0, h - 1], indices)] = Color.YELLOW
    if rng.random() < PRE_TOGGLED_PROB:
        rc = _segment_rc(grid.shape, indices, axis)
        grid[rc] = np.where(grid[rc] == Color.BLACK, Color.TEAL, Color.ORANGE)
    return grid


def _gen_toggle_segments(grid: np.ndarray, indices: list[int], axis: str) -> np.ndarray:
    """Return expected output by toggling selected segment interiors."""
    out = grid.copy()
    rc = _segment_rc(grid.shape, indices, axis)
    out[rc] = SEGMENT_TOGGLE_LOOKUP[grid[rc]]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    layout = _choose_layout(rng)
    if layout is None:
        return None
    _, _, axis, indices = layout
    grid = _render_input(layout, rng)
    expected = _gen_toggle_segments(grid, indices, axis)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d2acf2cb input")
