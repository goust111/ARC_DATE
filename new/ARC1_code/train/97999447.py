from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each colored marker pixel into a rightward alternating marker/grey stripe ending at the grid edge.

    Input structure
    - A dominant background with single-cell non-background marker pixels.
    - Each marker row contains exactly one marker pixel.
    - The marker pixels are not grey; grey is reserved for the alternating separator cells in the output.

    Rule
    - Preserve all cells to the left of each marker.
    - From each marker through the right edge of its row, write the marker color on even offsets from the marker and grey on odd offsets.
    - Rows with no marker remain unchanged.

    Key insight
    - The marker is the start of a horizontal ray; distance parity from the marker determines whether a cell receives the marker color or grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.GREY, "grey is the separator color, not the background"

    coords = np.argwhere(g != bg)
    assert coords.shape[0] > 0, "expected at least one marker"
    marker_rows = [int(r) for r in coords[:, 0]]
    assert len(marker_rows) == len(set(marker_rows)), "expected at most one marker per row"

    out = g.copy()
    width = int(g.shape[1])
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c0 = int(c_raw)
        marker_color = int(g[r, c0])
        assert marker_color != Color.GREY, "marker cannot use the separator color"

        for c in range(c0, width):
            out[r, c] = marker_color if (c - c0) % 2 == 0 else Color.GREY

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds; upper range endpoints are exclusive for numpy integers().
_GEN_HEIGHT_RANGE, _GEN_WIDTH_RANGE = (7, 11), (8, 13)
# Marker layout constraints.
_GEN_MARKER_COUNT_RANGE = (1, 5)
_GEN_ROW_MARGIN, _GEN_COL_MIN, _GEN_MIN_RIGHT_RAY = 1, 1, 4
# Palette and retry policy.
_GEN_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_SEPARATOR_COLOR, _GEN_MAX_ATTEMPTS = 5, 300
_GEN_RNG = np.random.default_rng()


def _paint_marker(grid: np.ndarray, expected: np.ndarray, r: int, c0: int, color: int) -> None:
    grid[r, c0] = color
    cols = np.arange(c0, grid.shape[1])
    expected[r, c0:] = np.where((cols - c0) % 2 == 0, color, _GEN_SEPARATOR_COLOR)


def _gen_attempt() -> np.ndarray | None:
    h = int(_GEN_RNG.integers(*_GEN_HEIGHT_RANGE))
    w = int(_GEN_RNG.integers(*_GEN_WIDTH_RANGE))
    grid = np.zeros((h, w), dtype=int)
    expected = grid.copy()
    k = int(_GEN_RNG.integers(*_GEN_MARKER_COUNT_RANGE))
    rows = _GEN_RNG.choice(np.arange(_GEN_ROW_MARGIN, h - _GEN_ROW_MARGIN), size=k, replace=False)
    colors = _GEN_RNG.choice(_GEN_COLORS, size=k, replace=False)
    for r_raw, color_raw in zip(rows, colors):
        r = int(r_raw)
        c0 = int(_GEN_RNG.integers(_GEN_COL_MIN, w - _GEN_MIN_RIGHT_RAY + 1))
        color = int(color_raw)
        _paint_marker(grid, expected, r, c0, color)

    try:
        return grid if np.array_equal(main(grid), expected) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 97999447 input")
