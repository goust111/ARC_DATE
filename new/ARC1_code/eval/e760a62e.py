from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import Color
import random


def _cell_ranges(length: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for wall in sorted(wall_indices):
        if start <= wall - 1:
            ranges.append((start, wall - 1))
        start = wall + 1
    if start <= length - 1:
        ranges.append((start, length - 1))
    return ranges


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    matches = [i for i, (lo, hi) in enumerate(bands) if lo <= pos <= hi]
    assert len(matches) == 1, "marker must lie in exactly one lattice cell band"
    return matches[0]


def _unique_dominant_color(colors: list[int], counts: Counter[int]) -> int:
    max_count = max(counts[color] for color in colors)
    candidates = [color for color in colors if counts[color] == max_count]
    assert len(candidates) == 1, "expected a unique dominant empty color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill whole lattice cells along straight red and green marker runs, coloring red/green intersections pink.

    Input structure
    - A teal lattice divides the canvas into rectangular background cells.
    - Red and green marker pixels sit inside some cells of that lattice.
    - Markers of the same color are paired by sharing a lattice row or a lattice column.

    Rule
    - For each color, any lattice row with at least two same-color markers is filled between the leftmost and rightmost marker cells.
    - Likewise, any lattice column with at least two same-color markers is filled between the topmost and bottommost marker cells.
    - Fill entire cells, preserve the teal lattice, and color cells reached by both red and green runs pink.

    Key insight
    - The marker pixels are not expanded locally; they are endpoints in a coarser grid whose rows and columns are the lattice cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    H, W = g.shape

    colors = [int(color) for color in np.unique(g)]
    lattice_candidates = []
    for color in colors:
        wall_rows = [r for r in range(H) if np.all(g[r, :] == color)]
        wall_cols = [c for c in range(W) if np.all(g[:, c] == color)]
        if wall_rows and wall_cols:
            lattice_candidates.append((color, wall_rows, wall_cols))
    assert len(lattice_candidates) == 1, "expected one color to form complete lattice rows and columns"
    wall_color, wall_rows, wall_cols = lattice_candidates[0]
    assert wall_color == Color.TEAL, "expected a teal lattice"

    counts = Counter(map(int, g.ravel()))
    empty_color = _unique_dominant_color([color for color in colors if color != wall_color], counts)
    assert empty_color == Color.BLACK, "expected black empty cells"

    marker_colors = {color for color in colors if color not in {wall_color, empty_color}}
    assert marker_colors == {Color.RED, Color.GREEN}, "expected red and green markers"

    row_bands = _cell_ranges(H, wall_rows)
    col_bands = _cell_ranges(W, wall_cols)
    assert row_bands and col_bands, "lattice must enclose at least one cell band"

    markers_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r_raw, c_raw in np.argwhere((g != wall_color) & (g != empty_color)):
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        markers_by_color[color].append((_band_index(r, row_bands), _band_index(c, col_bands)))

    filled: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, marker_cells in markers_by_color.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for row_idx, col_idx in marker_cells:
            by_row[row_idx].append(col_idx)
            by_col[col_idx].append(row_idx)

        used_marker_cells: set[tuple[int, int]] = set()
        for row_idx, col_indices in by_row.items():
            if len(col_indices) < 2:
                continue
            for col_idx in range(min(col_indices), max(col_indices) + 1):
                filled[(row_idx, col_idx)].add(color)
            used_marker_cells.update((row_idx, col_idx) for col_idx in col_indices)

        for col_idx, row_indices in by_col.items():
            if len(row_indices) < 2:
                continue
            for row_idx in range(min(row_indices), max(row_indices) + 1):
                filled[(row_idx, col_idx)].add(color)
            used_marker_cells.update((row_idx, col_idx) for row_idx in row_indices)

        assert set(marker_cells) <= used_marker_cells, "every marker must have a row or column mate"

    out = g.copy()
    for (row_idx, col_idx), cell_colors in filled.items():
        if len(cell_colors) == 1:
            fill_color = next(iter(cell_colors))
        else:
            assert cell_colors == {Color.RED, Color.GREEN}, "only red/green intersections are supported"
            fill_color = Color.PINK

        r0, r1 = row_bands[row_idx]
        c0, c1 = col_bands[col_idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out

# --------------- generator (o5x) ---------------
# Canvas and lattice bounds.
GRID_SIZE_BOUNDS = (22, 28)
CELL_SIZE_BOUNDS = (2, 5)
WALL_COUNT_BOUNDS = (4, 8)
TOTAL_CELLS_BOUNDS = (5, 8)
LATTICE_WALL_WIDTH = 1

# Marker topology and run spans.
MIN_RUN_SPAN = 3
MAX_RUN_SPAN = 7
MAX_RUN_SPAN_PERCENT = 88
PERCENT_DENOMINATOR = 100
RED_MARKER_COUNTS = (2, 3, 4)
DIRECTIONS = (-1, 1)
LINE_ORIENTATION_FLAGS = (True, False)

# Green line selection and overlap with red fills.
PINK_TARGET_PROBABILITY = 0.65
PINK_INTERSECTION_TARGETS = (1, 0)
MAX_GREEN_RED_INTERSECTIONS = 1

# Palette.
BACKGROUND_COLOR, LATTICE_COLOR, RED_COLOR, GREEN_COLOR, INTERSECTION_COLOR = (
    Color.BLACK, Color.TEAL, Color.RED, Color.GREEN, Color.PINK
)

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_lattice_choices_by_size() -> dict[int, list[tuple[int, int, int]]]:
    choices_by_size: dict[int, list[tuple[int, int, int]]] = {}
    for cell_size in range(CELL_SIZE_BOUNDS[0], CELL_SIZE_BOUNDS[1] + 1):
        for n_walls in range(WALL_COUNT_BOUNDS[0], WALL_COUNT_BOUNDS[1] + 1):
            for tail in range(cell_size):
                size = n_walls * (cell_size + LATTICE_WALL_WIDTH) + tail
                total_cells = n_walls + (1 if tail else 0)
                if GRID_SIZE_BOUNDS[0] <= size <= GRID_SIZE_BOUNDS[1] and TOTAL_CELLS_BOUNDS[0] <= total_cells <= TOTAL_CELLS_BOUNDS[1]:
                    choices_by_size.setdefault(size, []).append((cell_size, n_walls, tail))
    return choices_by_size


LATTICE_CHOICES_BY_SIZE = _gen_lattice_choices_by_size()


def _gen_lattice_params() -> tuple[int, int, int, int, int]:
    size = random.choice(sorted(LATTICE_CHOICES_BY_SIZE))
    cell_size, n_walls, tail = random.choice(LATTICE_CHOICES_BY_SIZE[size])
    total_cells = n_walls + (1 if tail else 0)
    return size, cell_size, n_walls, tail, total_cells


def _gen_base_lattice(size: int, cell_size: int, n_walls: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    grid = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    wall_indices = [cell_size + i * (cell_size + LATTICE_WALL_WIDTH) for i in range(n_walls)]
    grid[wall_indices, :] = LATTICE_COLOR
    grid[:, wall_indices] = LATTICE_COLOR
    return grid, _cell_ranges(size, wall_indices)


def _gen_max_span(marker_axis: int, total_cells: int) -> int:
    return min(MAX_RUN_SPAN, marker_axis, (MAX_RUN_SPAN_PERCENT * total_cells) // PERCENT_DENOMINATOR)


def _gen_oriented_endpoint(span: int, axis_len: int) -> tuple[int, int]:
    direction = random.choice(DIRECTIONS)
    if direction < 0:
        start = random.randint(span - 1, axis_len - 1)
        return start, start - span + 1
    start = random.randint(0, axis_len - span)
    return start, start + span - 1


def _gen_branch_endpoints(origin: int, max_span: int, axis_len: int) -> list[int]:
    endpoints = []
    for span in range(MIN_RUN_SPAN, max_span + 1):
        for direction in DIRECTIONS:
            endpoint = origin + direction * (span - 1)
            if 0 <= endpoint < axis_len:
                endpoints.append(endpoint)
    return sorted(set(endpoints))


def _gen_red_line_markers(axis_len: int, max_span: int) -> list[tuple[int, int]]:
    span = random.randint(MIN_RUN_SPAN, max_span)
    fixed = random.randrange(axis_len)
    start = random.randint(0, axis_len - span)
    end = start + span - 1
    if random.choice(LINE_ORIENTATION_FLAGS):
        return [(fixed, start), (fixed, end)]
    return [(start, fixed), (end, fixed)]


def _gen_red_corner_markers(axis_len: int, max_span: int) -> list[tuple[int, int]]:
    h_span = random.randint(MIN_RUN_SPAN, max_span)
    v_span = random.randint(MIN_RUN_SPAN, max_span)
    row0, row1 = _gen_oriented_endpoint(v_span, axis_len)
    col0, col1 = _gen_oriented_endpoint(h_span, axis_len)
    return [(row0, col0), (row0, col1), (row1, col0)]


def _gen_red_four_markers(axis_len: int, max_span: int) -> list[tuple[int, int]]:
    choices = []
    for h_span in range(MIN_RUN_SPAN, max_span + 1):
        for row_idx in range(axis_len):
            branch_rows = _gen_branch_endpoints(row_idx, max_span, axis_len)
            for start in range(0, axis_len - h_span + 1):
                end = start + h_span - 1
                for left_row in branch_rows:
                    for right_row in branch_rows:
                        if left_row != right_row:
                            choices.append(
                                [
                                    (row_idx, start),
                                    (row_idx, end),
                                    (left_row, start),
                                    (right_row, end),
                                ]
                            )
    return random.choice(choices)


def _gen_red_markers(axis_len: int, max_span: int) -> list[tuple[int, int]]:
    """Sample red endpoints as a line, corner, or two-branch shape."""
    red_count = random.choice(RED_MARKER_COUNTS)
    if red_count == 2:
        return _gen_red_line_markers(axis_len, max_span)
    if red_count == 4:
        return _gen_red_four_markers(axis_len, max_span)
    return _gen_red_corner_markers(axis_len, max_span)


def _gen_fill_cells(markers: list[tuple[int, int]]) -> set[tuple[int, int]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for row_idx, col_idx in markers:
        by_row[row_idx].append(col_idx)
        by_col[col_idx].append(row_idx)

    filled: set[tuple[int, int]] = set()
    for row_idx, cols in by_row.items():
        if len(cols) >= 2:
            for col_idx in range(min(cols), max(cols) + 1):
                filled.add((row_idx, col_idx))
    for col_idx, rows in by_col.items():
        if len(rows) >= 2:
            for row_idx in range(min(rows), max(rows) + 1):
                filled.add((row_idx, col_idx))
    return filled


def _gen_line_fill(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    if a[0] == b[0]:
        row_idx = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return {(row_idx, col_idx) for col_idx in range(c0, c1 + 1)}
    col_idx = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return {(row_idx, col_idx) for row_idx in range(r0, r1 + 1)}


GreenChoice = tuple[int, int, list[tuple[int, int]], set[tuple[int, int]]]


def _gen_green_line_choices(
    axis_len: int,
    max_span: int,
    red_markers: set[tuple[int, int]],
    red_fill: set[tuple[int, int]],
) -> list[GreenChoice]:
    """Enumerate green lines that avoid red markers and make at most one pink cell."""
    choices: list[GreenChoice] = []
    for span in range(MIN_RUN_SPAN, max_span + 1):
        for fixed in range(axis_len):
            for start in range(0, axis_len - span + 1):
                end = start + span - 1
                for a, b in (((fixed, start), (fixed, end)), ((start, fixed), (end, fixed))):
                    if a in red_markers or b in red_markers:
                        continue
                    green_fill = _gen_line_fill(a, b)
                    intersections = len(green_fill & red_fill)
                    if intersections <= MAX_GREEN_RED_INTERSECTIONS:
                        choices.append((intersections, span, [a, b], green_fill))
    return choices


def _gen_choose_green_line(choices: list[GreenChoice]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    target = PINK_INTERSECTION_TARGETS[0] if random.random() < PINK_TARGET_PROBABILITY else PINK_INTERSECTION_TARGETS[1]
    targeted = [choice for choice in choices if choice[0] == target]
    by_span: dict[int, list[GreenChoice]] = {}
    for choice in targeted or choices:
        by_span.setdefault(choice[1], []).append(choice)
    span = random.choice(sorted(by_span))
    _, _, markers, fill = random.choice(by_span[span])
    return markers, fill


def _gen_green_markers(
    axis_len: int,
    max_span: int,
    red_markers: set[tuple[int, int]],
    red_fill: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    choices = _gen_green_line_choices(axis_len, max_span, red_markers, red_fill)
    if not choices:
        return None
    return _gen_choose_green_line(choices)


def _gen_paint_cells(
    grid: np.ndarray,
    bands: list[tuple[int, int]],
    cells: set[tuple[int, int]],
    color: int,
) -> None:
    for row_idx, col_idx in cells:
        r0, r1 = bands[row_idx]
        c0, c1 = bands[col_idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _gen_marker_offset(cell_size: int) -> tuple[int, int]:
    if cell_size % 2:
        offset = cell_size // 2
        return offset, offset
    offsets = [cell_size // 2 - 1, cell_size // 2]
    return random.choice(offsets), random.choice(offsets)


def _gen_place_markers(
    grid: np.ndarray,
    bands: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    color: int,
    row_offset: int,
    col_offset: int,
) -> None:
    for row_idx, col_idx in markers:
        r0, _ = bands[row_idx]
        c0, _ = bands[col_idx]
        grid[r0 + row_offset, c0 + col_offset] = color


def _gen_attempt() -> np.ndarray | None:
    size, cell_size, n_walls, _tail, total_cells = _gen_lattice_params()
    lattice, bands = _gen_base_lattice(size, cell_size, n_walls)

    marker_axis = n_walls
    max_span = _gen_max_span(marker_axis, total_cells)
    if max_span < MIN_RUN_SPAN:
        return None

    red_markers = _gen_red_markers(marker_axis, max_span)
    red_fill = _gen_fill_cells(red_markers)

    green_choice = _gen_green_markers(marker_axis, max_span, set(red_markers), red_fill)
    if green_choice is None:
        return None
    green_markers, green_fill = green_choice

    out = lattice.copy()
    intersections = red_fill & green_fill
    _gen_paint_cells(out, bands, red_fill - intersections, RED_COLOR)
    _gen_paint_cells(out, bands, green_fill - intersections, GREEN_COLOR)
    _gen_paint_cells(out, bands, intersections, INTERSECTION_COLOR)

    inp = out.copy()
    _gen_paint_cells(inp, bands, red_fill | green_fill, BACKGROUND_COLOR)
    row_offset, col_offset = _gen_marker_offset(cell_size)
    _gen_place_markers(inp, bands, red_markers, RED_COLOR, row_offset, col_offset)
    _gen_place_markers(inp, bands, green_markers, GREEN_COLOR, row_offset, col_offset)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e760a62e input")
