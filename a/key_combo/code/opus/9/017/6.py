import random
from collections import Counter

import numpy as np


class Color:
    """Minimal ARC palette constants copied from the shared toolbox."""

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7


BG_COLOR = Color.BLACK
AXIS_COLOR = Color.GREEN
MARKER_PALETTE = (Color.BLUE, Color.RED, Color.YELLOW, Color.ORANGE)

GRID_HEIGHT_RANGE = (13, 20)
GRID_WIDTH_RANGE = (13, 20)
MARKER_COUNT_RANGE = (3, 5)
GEN_RETRY_LIMIT = 400
PLACEMENT_RETRY_LIMIT = 120


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    """Sorted unique colors present in the grid, excluding optional colors."""
    colors = set(map(int, np.asarray(grid).ravel()))
    colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r, c) lies within a grid of shape (rows, cols)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _find_dynamic_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Locate the unique full-row/full-column cross that defines the mirror axes."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in colors_present(g, exclude={bg}):
        full_rows = np.flatnonzero(np.all(g == color, axis=1))
        full_cols = np.flatnonzero(np.all(g == color, axis=0))
        if len(full_rows) != 1 or len(full_cols) != 1:
            continue
        axis_r = int(full_rows[0])
        axis_c = int(full_cols[0])
        if axis_r in (0, h - 1) or axis_c in (0, w - 1):
            continue
        if int(np.count_nonzero(g == color)) != h + w - 1:
            continue
        candidates.append((int(color), axis_r, axis_c))
    assert len(candidates) == 1, "expected exactly one dynamic full-row/full-column cross"
    return candidates[0]


def _quadrant_name(r: int, c: int, axis_r: int, axis_c: int) -> str | None:
    if r == axis_r or c == axis_c:
        return None
    if r < axis_r and c < axis_c:
        return "tl"
    if r < axis_r and c > axis_c:
        return "tr"
    if r > axis_r and c < axis_c:
        return "bl"
    return "br"


def _reflections(r: int, c: int, axis_r: int, axis_c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c),
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse markers by mirroring them across a cross whose axes are defined by edge-triggered rays.

    Input structure:
    - The background is black.
    - One non-background color forms exactly one full row and one full column, intersecting as a green cross.
    - Sparse single-pixel markers of other colors lie away from that cross, all in exactly one quadrant.

    Rule:
    - Identify the cross row and cross column; these are the dynamic horizontal and vertical symmetry axes.
    - Find the unique quadrant containing the sparse colored markers.
    - For each marker at (r, c), copy its color to the positions reflected across the horizontal axis, the vertical axis, and both axes.
    - Keep the original markers, the green cross, and all background cells unchanged otherwise.

    Key insight:
    - The mirror axes are not the grid center; they must be inferred from the full green row and column produced by the border control rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    axis_color, axis_r, axis_c = _find_dynamic_cross(g, bg)

    marker_coords = np.argwhere((g != bg) & (g != axis_color))
    assert marker_coords.size > 0, "expected sparse markers outside the cross"

    quadrants: dict[str, list[tuple[int, int]]] = {"tl": [], "tr": [], "bl": [], "br": []}
    for r0, c0 in marker_coords:
        r = int(r0)
        c = int(c0)
        quad = _quadrant_name(r, c, axis_r, axis_c)
        assert quad is not None, "markers must not lie on the symmetry cross"
        quadrants[quad].append((r, c))

    occupied = [name for name, cells in quadrants.items() if cells]
    assert len(occupied) == 1, "expected markers in exactly one quadrant"

    out = g.copy()
    for r, c in quadrants[occupied[0]]:
        color = int(g[r, c])
        for rr, cc in _reflections(r, c, axis_r, axis_c):
            assert in_bounds(g.shape, rr, cc), "generated reflection fell outside the grid"
            assert rr != axis_r and cc != axis_c, "generated reflection fell onto an axis"
            existing = int(out[rr, cc])
            assert existing in (bg, color), "reflected markers must not conflict with other colors"
            out[rr, cc] = color

    return out


def _choose_axis(length: int) -> int:
    choices = []
    for axis in range(3, length - 3):
        near = axis
        far = length - 1 - axis
        if min(near, far) < 2:
            continue
        if near == far:
            continue
        choices.append(axis)
    assert choices, f"no valid off-center axis for length {length}"
    return int(random.choice(choices))


def _usable_source_rows(axis_r: int, height: int) -> list[int]:
    span = min(axis_r, height - 1 - axis_r)
    return list(range(axis_r - span, axis_r))


def _usable_source_cols(axis_c: int, width: int) -> list[int]:
    span = min(axis_c, width - 1 - axis_c)
    return list(range(axis_c - span, axis_c))


def _well_separated(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = candidate
    return all(max(abs(r - r0), abs(c - c0)) >= 2 for r0, c0 in chosen)


def _sample_sparse_positions(rows: list[int], cols: list[int], count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in rows for c in cols]
    for _ in range(PLACEMENT_RETRY_LIMIT):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if _well_separated(cell, chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        if len({r for r, _ in chosen}) < 2 or len({c for _, c in chosen}) < 2:
            continue
        return sorted(chosen)
    return None


def _sample_marker_colors(count: int) -> list[int]:
    distinct = random.randint(2, min(len(MARKER_PALETTE), count))
    base = random.sample(MARKER_PALETTE, distinct)
    colors = [base[i % distinct] for i in range(count)]
    while len(colors) < count:
        colors.append(random.choice(base))
    random.shuffle(colors)
    return colors[:count]


def _render_input(
    height: int,
    width: int,
    axis_r: int,
    axis_c: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[axis_r, :] = AXIS_COLOR
    grid[:, axis_c] = AXIS_COLOR
    for r, c, color in markers:
        grid[r, c] = int(color)
    return grid


def _expected_output_from_markers(
    grid: np.ndarray,
    axis_r: int,
    axis_c: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c, color in markers:
        for rr, cc in _reflections(r, c, axis_r, axis_c):
            out[rr, cc] = int(color)
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    axis_r: int,
    axis_c: int,
    markers: list[tuple[int, int, int]],
) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False
    if get_bg(grid) != BG_COLOR:
        return False

    try:
        axis_color, found_r, found_c = _find_dynamic_cross(grid, BG_COLOR)
    except AssertionError:
        return False
    if axis_color != AXIS_COLOR or found_r != axis_r or found_c != axis_c:
        return False

    marker_positions = {(int(r), int(c)): int(color) for r, c, color in markers}
    for (r, c), color in marker_positions.items():
        if grid[r, c] != color:
            return False
        if _quadrant_name(r, c, axis_r, axis_c) != "tl":
            return False

    seen_quadrants = set()
    reflected_positions = set()
    for r, c, color in markers:
        seen_quadrants.add(_quadrant_name(r, c, axis_r, axis_c))
        for rr, cc in _reflections(r, c, axis_r, axis_c):
            if not in_bounds(grid.shape, rr, cc):
                return False
            if rr == axis_r or cc == axis_c:
                return False
            reflected_positions.add((rr, cc))
            if (rr, cc) != (r, c) and grid[rr, cc] != BG_COLOR:
                return False
            if expected[rr, cc] != color:
                return False
    if seen_quadrants != {"tl"}:
        return False
    if len(reflected_positions) != 4 * len(markers):
        return False

    marker_mask = (grid != BG_COLOR) & (grid != AXIS_COLOR)
    if int(marker_mask.sum()) != len(markers):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randrange(*GRID_HEIGHT_RANGE)
    width = random.randrange(*GRID_WIDTH_RANGE)

    axis_r = _choose_axis(height)
    axis_c = _choose_axis(width)
    if abs((height - 1) - 2 * axis_r) < 2 and abs((width - 1) - 2 * axis_c) < 2:
        return None

    rows = _usable_source_rows(axis_r, height)
    cols = _usable_source_cols(axis_c, width)
    count = random.randint(*MARKER_COUNT_RANGE)
    if len(rows) * len(cols) < count:
        return None

    positions = _sample_sparse_positions(rows, cols, count)
    if positions is None:
        return None

    colors = _sample_marker_colors(count)
    markers = [(r, c, color) for (r, c), color in zip(positions, colors)]

    grid = _render_input(height, width, axis_r, axis_c, markers)
    expected = _expected_output_from_markers(grid, axis_r, axis_c, markers)
    if not _validate_case(grid, expected, axis_r, axis_c, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dynamic-cross reflection input")
