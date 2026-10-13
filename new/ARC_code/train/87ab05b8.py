from __future__ import annotations

import numpy as np

from arc_utils import Color, coords_where, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase all distractor pixels and expand the single red control pixel into the grid quadrant that contains it.

    Input structure:
    - The grid has one dominant background color.
    - Exactly one red pixel is the control pixel.
    - Other non-background pixels are distractors.

    Rule:
    - Clear the canvas to the background color.
    - Split the grid into equal top/bottom and left/right halves.
    - Fill the quadrant containing the red control pixel with red.

    Key insight:
    - The red control pixel selects a whole half-grid quadrant by its row half and column half; all other foreground pixels are noise.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H % 2 == 0 and W % 2 == 0, "grid must split into equal quadrants"

    bg = get_bg(g)
    marker_color = Color.RED
    marker_coords = coords_where(g == marker_color)
    assert len(marker_coords) == 1, "expected exactly one red control pixel"

    half_h = H // 2
    half_w = W // 2
    marker_r, marker_c = map(int, marker_coords[0])
    r0 = 0 if marker_r < half_h else half_h
    c0 = 0 if marker_c < half_w else half_w

    out = np.full_like(g, bg)
    out[r0 : r0 + half_h, c0 : c0 + half_w] = marker_color
    return out

# --------------- generator (att2) ---------------

# Canvas and palette settings.
RED = 2
COLORS = tuple(range(10))
BG_COLORS = tuple(c for c in COLORS if c != RED)
GRID_SHAPE = (4, 4)
QUADRANT_SIZE = 2
QUADRANT_AXIS_COUNT = 2
QUADRANTS = tuple((r, c) for r in range(QUADRANT_AXIS_COUNT) for c in range(QUADRANT_AXIS_COUNT))

# Distractor count and quadrant-spread settings.
DISTRACTOR_COUNT_RANGE = (3, 8)
SPREAD_QUADRANT_RANGE = (2, 4)

# Retry limit for oracle validation.
MAX_GENERATION_ATTEMPTS = 300


def _quadrant_cells(qr, qc):
    r0, c0 = qr * QUADRANT_SIZE, qc * QUADRANT_SIZE
    return [(r, c) for r in range(r0, r0 + QUADRANT_SIZE) for c in range(c0, c0 + QUADRANT_SIZE)]


def _place_marker(grid):
    """Place the red control pixel; return ((qr, qc), (r, c))."""
    quadrant = (int(np.random.randint(QUADRANT_AXIS_COUNT)), int(np.random.randint(QUADRANT_AXIS_COUNT)))
    marker_cells = _quadrant_cells(*quadrant)
    marker_cell = marker_cells[int(np.random.randint(len(marker_cells)))]
    grid[marker_cell] = RED
    return quadrant, marker_cell


def _sample_distractor_cells(marker_cell, distractor_count):
    """Sample cells with at least one distractor in each chosen quadrant."""
    free_by_quadrant = {q: [cell for cell in _quadrant_cells(*q) if cell != marker_cell] for q in QUADRANTS}
    spread_min, spread_max = SPREAD_QUADRANT_RANGE
    spread_count = int(np.random.randint(spread_min, min(spread_max, distractor_count) + 1))
    chosen_ids = np.random.choice(len(QUADRANTS), size=spread_count, replace=False)
    chosen_quadrants = [QUADRANTS[int(i)] for i in chosen_ids]

    allocation = {q: 1 for q in chosen_quadrants}
    for _ in range(distractor_count - spread_count):
        expandable = [q for q in chosen_quadrants if allocation[q] < len(free_by_quadrant[q])]
        q = expandable[int(np.random.randint(len(expandable)))]
        allocation[q] += 1

    distractor_cells: list[tuple[int, int]] = []
    for q, count in allocation.items():
        cells = free_by_quadrant[q]
        picked = np.random.choice(len(cells), size=count, replace=False)
        distractor_cells.extend(cells[int(i)] for i in picked)
    return distractor_cells


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(BG_COLORS))
    grid = np.full(GRID_SHAPE, bg, dtype=int)

    quadrant, marker_cell = _place_marker(grid)
    distractor_count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    distractor_cells = _sample_distractor_cells(marker_cell, distractor_count)
    distractor_palette = [c for c in COLORS if c not in (bg, RED)]
    for r, c in distractor_cells:
        grid[r, c] = int(np.random.choice(distractor_palette))

    expected = np.full_like(grid, bg)
    qr, qc = quadrant
    r0, c0 = qr * QUADRANT_SIZE, qc * QUADRANT_SIZE
    expected[r0 : r0 + QUADRANT_SIZE, c0 : c0 + QUADRANT_SIZE] = RED
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
