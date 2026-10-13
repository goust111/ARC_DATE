from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


RING_CYCLE = np.array(
    [
        Color.GREEN,
        Color.YELLOW,
        Color.BLACK,
        Color.GREY,
        Color.RED,
        Color.TEAL,
        Color.MAROON,
        Color.PINK,
        Color.BLUE,
    ],
    dtype=int,
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace an orange canvas with concentric Manhattan-distance color rings centered on the lone marker pixel.

    Input structure:
    - The grid is an orange background with exactly one non-orange marker pixel.
    - The marker color is one of the colors in the fixed non-orange ring cycle.

    Rule:
    - The marker location is distance 0 and keeps the marker color.
    - Each step of Manhattan distance away from the marker advances one position
      through the cycle green, yellow, black, grey, red, teal, maroon, pink, blue.
    - Fill every pixel of the original grid by this distance-indexed cycle.

    Key insight: The output is not copied from existing objects; it is the
    distance field around the single marker, colored by a cyclic palette that
    excludes the orange background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.ORANGE, "expected an orange background"

    marker_coords = np.argwhere(g != bg)
    assert marker_coords.shape == (1, 2), "expected exactly one marker pixel"
    marker_r, marker_c = map(int, marker_coords[0])
    marker_color = int(g[marker_r, marker_c])

    cycle_indices = np.flatnonzero(RING_CYCLE == marker_color)
    assert cycle_indices.size == 1, "marker color must be in the ring cycle"
    start = int(cycle_indices[0])

    rows, cols = np.indices(g.shape)
    distances = np.abs(rows - marker_r) + np.abs(cols - marker_c)
    return RING_CYCLE[(start + distances) % len(RING_CYCLE)]

# --------------- generator (att2) ---------------

# Canvas bounds use numpy's exclusive high endpoint.
CANVAS_SIZE_RANGE = (5, 16)
GEN_BACKGROUND_COLOR = 7
GEN_RING_CYCLE = np.array([3, 4, 0, 5, 2, 8, 9, 6, 1], dtype=int)
MAX_GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*CANVAS_SIZE_RANGE))
    w = int(np.random.randint(*CANVAS_SIZE_RANGE))
    marker_r = int(np.random.randint(h))
    marker_c = int(np.random.randint(w))
    marker_color = int(np.random.choice(GEN_RING_CYCLE))

    grid = np.full((h, w), GEN_BACKGROUND_COLOR, dtype=int)
    grid[marker_r, marker_c] = marker_color
    return grid


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    marker_r, marker_c = map(int, np.argwhere(grid != GEN_BACKGROUND_COLOR)[0])
    marker_color = int(grid[marker_r, marker_c])
    start = int(np.flatnonzero(GEN_RING_CYCLE == marker_color)[0])
    rows, cols = np.indices(grid.shape)
    distances = np.abs(rows - marker_r) + np.abs(cols - marker_c)
    return GEN_RING_CYCLE[(start + distances) % len(GEN_RING_CYCLE)]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        try:
            out = main(candidate)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and np.array_equal(out, _gen_expected_output(candidate)):
            return candidate
    raise RuntimeError("failed to generate valid input for aaef0977")
