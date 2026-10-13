import numpy as np

from arc_utils import colors_present, get_bg
import random


def _line_bounds(g: np.ndarray, scaffold_color: int, axis: int) -> list[int]:
    if axis == 0:
        bounds = [r for r in range(g.shape[0]) if not np.any(g[r, :] == scaffold_color)]
    else:
        bounds = [c for c in range(g.shape[1]) if not np.any(g[:, c] == scaffold_color)]

    assert len(bounds) >= 2, "expected at least two scaffold boundary lines"
    gaps = [b - a for a, b in zip(bounds, bounds[1:])]
    assert len(set(gaps)) == 1, "expected regularly spaced scaffold boundary lines"
    assert gaps[0] % 2 == 0, "expected scaffold cells with a central hole"
    return bounds


def _covering_interval(x: int, bounds: list[int]) -> tuple[int, int]:
    if x in bounds:
        return x, x

    containing = [(a, b) for a, b in zip(bounds, bounds[1:]) if a < x < b]
    assert len(containing) == 1, "marker must lie on or inside the scaffold lattice"
    return containing[0]


def _centers_between(bounds: list[int], lo: int, hi: int) -> list[int]:
    centers = []
    for a, b in zip(bounds, bounds[1:]):
        if lo <= a and b <= hi:
            assert (a + b) % 2 == 0
            centers.append((a + b) // 2)
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker color into a scaffold-aligned rectangle border and fill the centers of the enclosed frames.

    Input structure
    - A black background holds a repeated scaffold of hollow 3x3 frames separated by empty boundary rows and columns.
    - Each non-scaffold foreground color appears as marker pixels on or inside one scaffold-aligned rectangular region.

    Rule
    - Infer the scaffold boundary rows and columns from the high-count frame color.
    - For each marker color, map every marker pixel to the boundary point or cell interval containing it.
    - Paint the smallest boundary-aligned rectangle covering those intervals, and color the center hole of each enclosed frame.

    Key insight
    - The markers do not define freehand paths; they select a rectangle in the scaffold's boundary lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) >= 2, "expected a scaffold color and at least one marker color"

    color_counts = {color: int(np.sum(g == color)) for color in foreground_colors}
    max_count = max(color_counts.values())
    scaffold_candidates = [color for color, count in color_counts.items() if count == max_count]
    assert len(scaffold_candidates) == 1, "expected a unique high-count scaffold color"
    scaffold_color = scaffold_candidates[0]
    marker_colors = [color for color in foreground_colors if color != scaffold_color]

    row_bounds = _line_bounds(g, scaffold_color, axis=0)
    col_bounds = _line_bounds(g, scaffold_color, axis=1)

    planned: dict[tuple[int, int], int] = {}
    for color in marker_colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2, "expected each marker color to define a rectangle"

        row_intervals = [_covering_interval(int(r), row_bounds) for r, _ in coords]
        col_intervals = [_covering_interval(int(c), col_bounds) for _, c in coords]
        r0 = min(a for a, _ in row_intervals)
        r1 = max(b for _, b in row_intervals)
        c0 = min(a for a, _ in col_intervals)
        c1 = max(b for _, b in col_intervals)
        assert r0 < r1 and c0 < c1, "marker extent must expand to a nonempty rectangle"

        target_cells: set[tuple[int, int]] = set()
        for c in range(c0, c1 + 1):
            target_cells.add((r0, c))
            target_cells.add((r1, c))
        for r in range(r0, r1 + 1):
            target_cells.add((r, c0))
            target_cells.add((r, c1))
        for r in _centers_between(row_bounds, r0, r1):
            for c in _centers_between(col_bounds, c0, c1):
                target_cells.add((r, c))

        for cell in target_cells:
            previous = planned.get(cell)
            assert previous is None or previous == color, "conflicting colored rectangles"
            planned[cell] = color

    out = g.copy()
    for (r, c), color in planned.items():
        assert out[r, c] in (bg, color), "rectangle paint must land on background or same-color markers"
        out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Scaffold geometry.
_GEN_BG = 0
_GEN_SCAFFOLD = 8
_GEN_BOUNDS = [0, 4, 8, 12, 16]
_GEN_GRID_SHAPE = (17, 17)
_GEN_FRAME_ORIGINS = [b + 1 for b in _GEN_BOUNDS[:-1]]
_GEN_FRAME_SIZE = 3
_GEN_CENTER_OFFSET = 2

# Marker and rectangle sampling knobs.
_GEN_MARKER_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
_GEN_MARKER_COUNT_RANGE = (1, 2)
_GEN_MARKER_PIXELS_RANGE = (2, 4)
_GEN_RECT_SPANS = (1, 2, 3)
_GEN_MAX_RECT_AREA = 6
_GEN_RETRY_LIMIT = 300


def _gen_base_scaffold() -> np.ndarray:
    grid = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    for r0 in _GEN_FRAME_ORIGINS:
        for c0 in _GEN_FRAME_ORIGINS:
            grid[r0 : r0 + _GEN_FRAME_SIZE, c0 : c0 + _GEN_FRAME_SIZE] = _GEN_SCAFFOLD
            grid[r0 + 1, c0 + 1] = _GEN_BG
    return grid


def _gen_rect_bounds(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    rb0, cb0, rb1, cb1 = rect
    return _GEN_BOUNDS[rb0], _GEN_BOUNDS[cb0], _GEN_BOUNDS[rb1], _GEN_BOUNDS[cb1]


def _gen_rect_cells(rect: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    rb0, cb0, rb1, cb1 = rect
    r0, c0, r1, c1 = _gen_rect_bounds(rect)
    cells: set[tuple[int, int]] = set()

    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for rb in range(rb0, rb1):
        for cb in range(cb0, cb1):
            cells.add((_GEN_BOUNDS[rb] + _GEN_CENTER_OFFSET, _GEN_BOUNDS[cb] + _GEN_CENTER_OFFSET))

    return frozenset(cells)


def _gen_rect_candidates() -> list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]]:
    candidates = []
    for height in _GEN_RECT_SPANS:
        for width in _GEN_RECT_SPANS:
            if height * width > _GEN_MAX_RECT_AREA:
                continue
            for rb0 in range(len(_GEN_BOUNDS) - height):
                for cb0 in range(len(_GEN_BOUNDS) - width):
                    rect = (rb0, cb0, rb0 + height, cb0 + width)
                    candidates.append((rect, _gen_rect_cells(rect)))
    return candidates


_GEN_RECTS = _gen_rect_candidates()
_GEN_RECT_PAIRS = [
    (_GEN_RECTS[i], _GEN_RECTS[j])
    for i in range(len(_GEN_RECTS))
    for j in range(i + 1, len(_GEN_RECTS))
    if _GEN_RECTS[i][1].isdisjoint(_GEN_RECTS[j][1])
]


def _gen_paint_rect(grid, cells, color) -> None:
    for r, c in cells:
        assert grid[r, c] in (_GEN_BG, color)
        grid[r, c] = color


def _gen_marker_positions(rect, cells, pixel_count) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _gen_rect_bounds(rect)
    corners = random.choice(([(r0, c0), (r1, c1)], [(r0, c1), (r1, c0)]))
    available = sorted(cells - set(corners))
    return corners + random.sample(available, pixel_count - 2)


def _gen_sample_layout():
    marker_count = random.randint(*_GEN_MARKER_COUNT_RANGE)
    colors = random.sample(_GEN_MARKER_COLORS, marker_count)
    if marker_count == 1:
        return [random.choice(_GEN_RECTS)], colors
    return list(random.choice(_GEN_RECT_PAIRS)), colors


def _gen_render_output(chosen, colors) -> np.ndarray:
    output = _gen_base_scaffold()
    for (_, cells), color in zip(chosen, colors):
        _gen_paint_rect(output, cells, color)
    return output


def _gen_render_input(chosen, colors) -> np.ndarray:
    input_grid = _gen_base_scaffold()
    for (rect, cells), color in zip(chosen, colors):
        pixel_count = random.randint(*_GEN_MARKER_PIXELS_RANGE)
        for r, c in _gen_marker_positions(rect, cells, pixel_count):
            input_grid[r, c] = color
    return input_grid


def _gen_matches_expected(input_grid: np.ndarray, output: np.ndarray) -> bool:
    try:
        actual = main(input_grid)
    except AssertionError:
        return False
    return np.array_equal(actual, output)


def _gen_attempt() -> np.ndarray | None:
    chosen, colors = _gen_sample_layout()
    output = _gen_render_output(chosen, colors)
    input_grid = _gen_render_input(chosen, colors)
    return input_grid if _gen_matches_expected(input_grid, output) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 97239e3d input")
