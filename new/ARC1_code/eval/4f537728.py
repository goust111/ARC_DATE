from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the row and column bands crossing the unique colored marker in a blue lattice.

    Input structure
    - The background is the dominant empty color.
    - The foreground has exactly two colors: a large repeated lattice color and
      one distinct marker color.
    - The marker color forms a single filled 2x2 block embedded in the lattice.

    Rule
    - Find the marker's two rows and two columns.
    - Recolor every lattice-color pixel in those rows or columns to the marker color.
    - Leave background pixels and all other lattice pixels unchanged.

    Key insight
    - The marker does not move; it selects the horizontal and vertical lattice
      bands that pass through its 2x2 block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected one lattice color and one marker color"

    marker_colors = [color for color in fg_colors if np.count_nonzero(g == color) == 4]
    assert len(marker_colors) == 1, "expected a unique four-pixel marker color"
    marker = marker_colors[0]
    lattice_colors = [color for color in fg_colors if color != marker]
    assert len(lattice_colors) == 1, "expected a unique lattice color"
    lattice = lattice_colors[0]
    assert np.count_nonzero(g == lattice) > np.count_nonzero(g == marker)

    marker_box = bbox(g == marker)
    assert marker_box is not None, "expected marker pixels"
    r0, c0, r1, c1 = marker_box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must be a 2x2 block"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == marker), "marker block must be solid"

    row_band = np.zeros(g.shape, dtype=bool)
    row_band[r0 : r1 + 1, :] = True
    col_band = np.zeros(g.shape, dtype=bool)
    col_band[:, c0 : c1 + 1] = True
    cross_band = row_band | col_band
    assert np.any(cross_band & (g == lattice)), "selected bands must contain lattice pixels"

    out = g.copy()
    out[cross_band & (g == lattice)] = marker
    return out

# --------------- generator (o5x) ---------------

# Canvas/colors.
GRID_SHAPE = (20, 20)
BG_COLOR = 0
LATTICE_COLOR = 1
MARKER_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
# Lattice/marker geometry.
LATTICE_PERIOD = 3
PHASE_OPTIONS = [1, 2]
MARKER_SIZE = 2
# Retry guard.
MAX_GENERATION_ATTEMPTS = 300


def _gen_width2_band_starts(mask: np.ndarray) -> list[int]:
    starts = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if j - i == MARKER_SIZE:
            starts.append(i)
        i = j
    return starts


def _gen_expected_output(grid: np.ndarray, marker: int, r0: int, c0: int) -> np.ndarray:
    out = grid.copy()
    cross = np.zeros(grid.shape, dtype=bool)
    cross[r0 : r0 + MARKER_SIZE, :] = True
    cross[:, c0 : c0 + MARKER_SIZE] = True
    out[cross & (grid == LATTICE_COLOR)] = marker
    return out


def _gen_attempt() -> np.ndarray | None:
    row_phase = int(np.random.choice(PHASE_OPTIONS))
    col_phase = int(np.random.choice(PHASE_OPTIONS))

    rows = np.arange(GRID_SHAPE[0])
    cols = np.arange(GRID_SHAPE[1])
    row_mask = rows % LATTICE_PERIOD != row_phase
    col_mask = cols % LATTICE_PERIOD != col_phase

    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    grid[np.ix_(row_mask, col_mask)] = LATTICE_COLOR

    row_starts = _gen_width2_band_starts(row_mask)
    col_starts = _gen_width2_band_starts(col_mask)
    if not row_starts or not col_starts:
        return None

    r0 = row_starts[int(np.random.randint(len(row_starts)))]
    c0 = col_starts[int(np.random.randint(len(col_starts)))]
    marker = int(np.random.choice(MARKER_COLORS))
    grid[r0 : r0 + MARKER_SIZE, c0 : c0 + MARKER_SIZE] = marker

    expected = _gen_expected_output(grid, marker, r0, c0)
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
    raise RuntimeError("failed to generate a valid input for 4f537728")
