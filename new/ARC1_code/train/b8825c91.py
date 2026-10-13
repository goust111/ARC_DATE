from __future__ import annotations

import numpy as np

from arc_utils import colors_present
import random


Coord = tuple[int, int]


def _reflection_orbits(shape: tuple[int, int]) -> list[tuple[Coord, ...]]:
    height, width = map(int, shape)
    seen: set[frozenset[Coord]] = set()
    orbits: list[tuple[Coord, ...]] = []

    for r in range(height):
        for c in range(width):
            orbit = frozenset(
                {
                    (r, c),
                    (height - 1 - r, c),
                    (r, width - 1 - c),
                    (height - 1 - r, width - 1 - c),
                }
            )
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(tuple(orbit))

    return orbits


def _missing_color(g: np.ndarray, orbits: list[tuple[Coord, ...]]) -> int:
    candidates: list[int] = []
    for color in colors_present(g):
        valid = True
        for orbit in orbits:
            known = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != color}
            if len(known) != 1:
                valid = False
                break
        if valid:
            candidates.append(int(color))

    assert len(candidates) == 1, "expected exactly one color to behave as the missing-cell marker"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the missing-marker cells so the whole grid has matching horizontal and vertical reflections.

    Input structure:
    - The grid is a complete reflection-symmetric pattern except for cells of one marker color.
    - Under reflection across the horizontal and vertical center lines, each orbit has exactly one non-marker color.
    - The marker color is structurally unique: treating any other color as missing breaks at least one orbit.

    Rule:
    - Group cells by their four-way reflection orbit: original, vertical mirror, horizontal mirror, and 180-degree mirror.
    - Infer the unique marker color whose removal leaves one known color in every orbit.
    - Fill every cell in each orbit with that known color.

    Key insight:
    - The yellow cells are not objects to move; they are holes in a two-axis mirror pattern, and their values are copied from the visible cells in the same reflection orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orbits = _reflection_orbits(g.shape)
    marker = _missing_color(g, orbits)

    out = g.copy()
    for orbit in orbits:
        known = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != marker}
        assert len(known) == 1, "each reflection orbit must reveal exactly one fill color"
        fill = known.pop()
        for r, c in orbit:
            out[r, c] = fill

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE, HALF_SIZE = 16, 8
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
QUADRANT_SHAPE = (HALF_SIZE, HALF_SIZE)
QUADRANT_CELLS = HALF_SIZE * HALF_SIZE
PATTERN_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
OUTPUT_COLOR_COUNTS = (7, 8)
MARKER_COLOR = 4

# Marker rectangle geometry.
RECT_MIN_SIDE, RECT_MAX_SIDE = 2, 4
RECT_MIN_AREA = 6
MARKER_MIN_AREA, MARKER_MAX_AREA = 12, 25
MARKER_BLOCK_COUNT = 2
REGION_COUNT = 4

# Retry policy.
GEN_MAX_ATTEMPTS = 300

Rect = tuple[int, int, int, int]
RectDims = tuple[int, int]

RECT_DIMS = [(h, w) for h in range(RECT_MIN_SIDE, RECT_MAX_SIDE + 1) for w in range(RECT_MIN_SIDE, RECT_MAX_SIDE + 1) if h * w >= RECT_MIN_AREA]
RECT_DIM_PAIRS = [(d1, d2) for d1 in RECT_DIMS for d2 in RECT_DIMS if MARKER_MIN_AREA <= d1[0] * d1[1] + d2[0] * d2[1] <= MARKER_MAX_AREA]


def _gen_reflect_output(top_left: np.ndarray) -> np.ndarray:
    out = np.zeros(GRID_SHAPE, dtype=int)
    out[:HALF_SIZE, :HALF_SIZE] = top_left
    out[:HALF_SIZE, HALF_SIZE:] = np.fliplr(top_left)
    out[HALF_SIZE:, :HALF_SIZE] = np.flipud(top_left)
    out[HALF_SIZE:, HALF_SIZE:] = np.flipud(np.fliplr(top_left))
    return out


def _gen_top_left_quadrant(colors: list[int]) -> np.ndarray:
    flat = np.array([random.choice(colors) for _ in range(QUADRANT_CELLS)], dtype=int)
    for idx, color in zip(random.sample(range(QUADRANT_CELLS), len(colors)), colors):
        flat[idx] = color
    return flat.reshape(QUADRANT_SHAPE)


def _gen_rect_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, h, w = rect
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _gen_orbit_cells(r: int, c: int) -> frozenset[tuple[int, int]]:
    last = GRID_SIZE - 1
    return frozenset({(r, c), (last - r, c), (r, last - c), (last - r, last - c)})


def _gen_has_full_marker_orbit(cells: set[tuple[int, int]]) -> bool:
    return any(_gen_orbit_cells(r, c) <= cells for r, c in cells)


def _gen_rect_options(h: int, w: int, region: int) -> list[Rect]:
    top = region < 2
    left = region % 2 == 0
    row_range = range(1, HALF_SIZE) if top else range(HALF_SIZE, GRID_SIZE - h)
    col_range = range(1, HALF_SIZE) if left else range(HALF_SIZE, GRID_SIZE - w)
    return [(r, c, h, w) for r in row_range for c in col_range]


def _gen_rectangles_have_gap(rect1: Rect, rect2: Rect) -> bool:
    r1, c1, h1, w1 = rect1
    r2, c2, h2, w2 = rect2
    return (
        r1 + h1 < r2
        or r2 + h2 < r1
        or c1 + w1 < c2
        or c2 + w2 < c1
    )


def _gen_valid_rect_pairs(dim1: RectDims, dim2: RectDims, regions: list[int]) -> list[tuple[Rect, Rect]]:
    h1, w1 = dim1
    h2, w2 = dim2
    pair_options = []
    for rect1 in _gen_rect_options(h1, w1, regions[0]):
        cells1 = _gen_rect_cells(rect1)
        for rect2 in _gen_rect_options(h2, w2, regions[1]):
            if not _gen_rectangles_have_gap(rect1, rect2):
                continue
            marker_cells = cells1 | _gen_rect_cells(rect2)
            if not _gen_has_full_marker_orbit(marker_cells):
                pair_options.append((rect1, rect2))
    return pair_options


def _gen_pick_marker_mask() -> np.ndarray | None:
    dim1, dim2 = random.choice(RECT_DIM_PAIRS)
    regions = random.sample(range(REGION_COUNT), MARKER_BLOCK_COUNT)
    pair_options = _gen_valid_rect_pairs(dim1, dim2, regions)

    if not pair_options:
        return None

    mask = np.zeros(GRID_SHAPE, dtype=bool)
    for r, c, h, w in random.choice(pair_options):
        mask[r:r + h, c:c + w] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice(OUTPUT_COLOR_COUNTS)
    colors = random.sample(PATTERN_COLORS, n_colors)
    output = _gen_reflect_output(_gen_top_left_quadrant(colors))
    marker_mask = _gen_pick_marker_mask()
    if marker_mask is None:
        return None

    inp = output.copy()
    inp[marker_mask] = MARKER_COLOR
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b8825c91 input")
