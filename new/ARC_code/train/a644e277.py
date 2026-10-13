from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _is_regular_spacing(positions: list[int]) -> bool:
    if len(positions) < 2:
        return False
    gaps = np.diff(np.asarray(positions, dtype=int))
    return bool(np.all(gaps == gaps[0]))


def _find_lattice(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    H, W = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in colors_present(g, exclude={bg}):
        rows = [r for r in range(H) if np.count_nonzero(g[r, :] == color) >= W - 2]
        cols = [c for c in range(W) if np.count_nonzero(g[:, c] == color) >= H - 2]
        if _is_regular_spacing(rows) and _is_regular_spacing(cols):
            candidates.append((int(color), rows, cols))

    assert len(candidates) == 1, "expected a unique regular lattice color"
    _line_color, rows, cols = candidates[0]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the lattice subregion bounded by the four background-colored corner holes.

    Input structure:
    - A regular one-cell-thick lattice has one line color, one background color,
      and sparse marker pixels.
    - Exactly four lattice-line intersections are background-colored holes instead
      of line-color pixels.

    Rule:
    - Find the unique non-background color that forms regular horizontal and
      vertical lattice lines.
    - Use the four background-colored lattice intersections as the corners of one
      rectangle.
    - Return the inclusive crop between those corner rows and columns.

    Key insight:
    - The crop is marked by the missing lattice intersections, not by the marker
      pixels inside the cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rows, cols = _find_lattice(g, bg)

    holes = [(r, c) for r in rows for c in cols if int(g[r, c]) == bg]
    assert len(holes) == 4, "expected exactly four background holes on lattice intersections"

    hole_rows = sorted({r for r, _c in holes})
    hole_cols = sorted({c for _r, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must form rectangle corners"

    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()

# --------------- generator (att2) ---------------
COLOR_COUNT = 10
# Lattice dimensions and line counts.
CELL_SIZES = [3, 4]
CELL3_LAYOUT = (4, 5, (17, 20))
CELL4_H_LINE_OPTIONS = [3, 4]
CELL4_V_LINES = 4
CELL4_HEIGHT_RANGES = {3: (17, 20), 4: (20, 23)}
WIDTH_RANGE = (20, 23)
# Hole rectangle and marker scatter limits.
MAX_HOLE_SPAN = 2
MARKER_DENSITY_RANGE = (0.10, 0.20)
MAX_MARKERS_PER_CELL_REGION = 3
GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg, lattice_color, marker_color = [int(c) for c in rng.choice(COLOR_COUNT, size=3, replace=False)]
    cell_size = int(rng.choice(CELL_SIZES))
    if cell_size == 3:
        num_h_lines, num_v_lines, height_range = CELL3_LAYOUT
        height = int(rng.integers(*height_range))
    else:
        num_h_lines = int(rng.choice(CELL4_H_LINE_OPTIONS))
        num_v_lines = CELL4_V_LINES
        height = int(rng.integers(*CELL4_HEIGHT_RANGES[num_h_lines]))
    width = int(rng.integers(*WIDTH_RANGE))
    spacing = cell_size + 1
    line_rows = [cell_size + i * spacing for i in range(num_h_lines)]
    line_cols = [cell_size + i * spacing for i in range(num_v_lines)]
    grid = np.full((height, width), bg, dtype=int)
    grid[line_rows, :] = lattice_color
    grid[:, line_cols] = lattice_color
    r_span = int(rng.integers(1, min(MAX_HOLE_SPAN, num_h_lines - 1) + 1))
    c_span = int(rng.integers(1, min(MAX_HOLE_SPAN, num_v_lines - 1) + 1))
    r_i0 = int(rng.integers(0, num_h_lines - r_span))
    c_i0 = int(rng.integers(0, num_v_lines - c_span))
    hole_rows = [line_rows[r_i0], line_rows[r_i0 + r_span]]
    hole_cols = [line_cols[c_i0], line_cols[c_i0 + c_span]]

    for r in hole_rows:
        for c in hole_cols:
            grid[r, c] = bg

    interior = np.ones(grid.shape, dtype=bool)
    interior[line_rows, :] = False
    interior[:, line_cols] = False
    interior_positions = np.flatnonzero(interior)
    num_cell_regions = (num_h_lines + 1) * (num_v_lines + 1)
    min_markers = max(int(np.ceil(MARKER_DENSITY_RANGE[0] * len(interior_positions))), num_cell_regions)
    max_markers = min(
        int(np.floor(MARKER_DENSITY_RANGE[1] * len(interior_positions))),
        MAX_MARKERS_PER_CELL_REGION * num_cell_regions,
    )
    marker_count = int(rng.integers(min_markers, max_markers + 1))
    marker_positions = rng.choice(interior_positions, size=marker_count, replace=False)
    grid.ravel()[marker_positions] = marker_color

    expected = grid[hole_rows[0] : hole_rows[1] + 1, hole_cols[0] : hole_cols[1] + 1].copy()
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid a644e277 input")
