from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
INTERSECTION_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (8, 18)
GRID_WIDTH_RANGE = (8, 18)
GROUP_COUNT_MAX = 3
MARKER_COUNT_MIN = 2
MARKER_COUNT_MAX = 4
GENERATION_ATTEMPTS = 400
SAMPLE_ATTEMPTS = 120


HorizontalGroup = tuple[int, int, list[int]]
VerticalGroup = tuple[int, int, list[int]]


def _parse_groups(grid: np.ndarray) -> tuple[list[HorizontalGroup], list[VerticalGroup]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    assert colors, "expected at least one foreground color"
    assert INTERSECTION_COLOR not in colors, "grey is reserved for output intersections"

    horizontals: list[HorizontalGroup] = []
    verticals: list[VerticalGroup] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 2, "each color must mark at least two cells"

        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})

        if len(rows) == 1 and len(cols) >= 2:
            row = rows[0]
            assert row not in used_rows, "each row may host at most one horizontal group"
            used_rows.add(row)
            horizontals.append((color, row, cols))
        elif len(cols) == 1 and len(rows) >= 2:
            col = cols[0]
            assert col not in used_cols, "each column may host at most one vertical group"
            used_cols.add(col)
            verticals.append((color, col, rows))
        else:
            raise AssertionError("each color must form markers on exactly one row or one column")

    assert horizontals, "expected at least one horizontal group"
    assert verticals, "expected at least one vertical group"
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color row markers and column markers, then highlight their crossings in grey.

    Input structure:
    - The background is black.
    - Each non-background color appears as scattered single-pixel markers.
    - All markers of one color lie either on one row (horizontal group) or on one column (vertical group).

    Rule:
    - Extend each horizontal group to the full segment between its leftmost and rightmost markers.
    - Extend each vertical group to the full segment between its topmost and bottommost markers.
    - Recolor every crossing of a horizontal segment and a vertical segment to grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g != BACKGROUND), "expected at least one marker"

    horizontals, verticals = _parse_groups(g)
    out = g.copy()

    horizontal_cells: set[tuple[int, int]] = set()
    for color, row, cols in horizontals:
        c0, c1 = cols[0], cols[-1]
        out[row, c0 : c1 + 1] = color
        for col in range(c0, c1 + 1):
            horizontal_cells.add((row, col))

    vertical_cells: set[tuple[int, int]] = set()
    for color, col, rows in verticals:
        r0, r1 = rows[0], rows[-1]
        out[r0 : r1 + 1, col] = color
        for row in range(r0, r1 + 1):
            vertical_cells.add((row, col))

    for row, col in horizontal_cells & vertical_cells:
        out[row, col] = INTERSECTION_COLOR

    return out


def _max_nonadjacent_count(candidates: list[int]) -> int:
    count = 0
    last = -10**9
    for value in sorted(candidates):
        if value - last >= 2:
            count += 1
            last = value
    return count


def _sample_nonadjacent_positions(
    candidates: list[int], count: int, required: tuple[int, ...] = ()
) -> list[int] | None:
    ordered = sorted(set(int(x) for x in candidates))
    required_set = sorted(set(int(x) for x in required))
    if len(required_set) != len(required):
        return None
    if len(required_set) > count or not set(required_set).issubset(ordered):
        return None
    if not all(b - a >= 2 for a, b in zip(required_set, required_set[1:])):
        return None
    if len(ordered) < count or _max_nonadjacent_count(ordered) < count:
        return None

    pool = [x for x in ordered if x not in required_set]
    for _ in range(SAMPLE_ATTEMPTS):
        extra_count = count - len(required_set)
        if len(pool) < extra_count:
            return None
        extra = random.sample(pool, extra_count) if extra_count else []
        chosen = sorted(required_set + extra)
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _count_intersections(
    horizontals: list[HorizontalGroup], verticals: list[VerticalGroup]
) -> int:
    count = 0
    for _h_color, row, cols in horizontals:
        c0, c1 = cols[0], cols[-1]
        for _v_color, col, rows in verticals:
            r0, r1 = rows[0], rows[-1]
            if c0 <= col <= c1 and r0 <= row <= r1:
                count += 1
    return count


def _render_grid(
    height: int,
    width: int,
    horizontals: list[HorizontalGroup],
    verticals: list[VerticalGroup],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for color, row, cols in horizontals:
        for col in cols:
            grid[row, col] = color
    for color, col, rows in verticals:
        for row in rows:
            grid[row, col] = color
    return grid


def _sample_crossing_scene(height: int, width: int, h_count: int, v_count: int) -> np.ndarray | None:
    horizontal_rows = sorted(random.sample(range(1, height - 1), h_count))
    available_rows = [row for row in range(height) if row not in horizontal_rows]
    vertical_cols = sorted(random.sample(range(1, width - 1), v_count))
    horizontal_candidates = [col for col in range(width) if col not in vertical_cols]

    max_h_markers = min(MARKER_COUNT_MAX, _max_nonadjacent_count(horizontal_candidates))
    max_v_markers = min(MARKER_COUNT_MAX, _max_nonadjacent_count(available_rows))
    if max_h_markers < MARKER_COUNT_MIN or max_v_markers < MARKER_COUNT_MIN:
        return None

    horizontal_counts = [random.randint(MARKER_COUNT_MIN, max_h_markers) for _ in range(h_count)]
    vertical_counts = [random.randint(MARKER_COUNT_MIN, max_v_markers) for _ in range(v_count)]
    colors = random.sample(COLOR_POOL, h_count + v_count)
    horizontal_colors = colors[:h_count]
    vertical_colors = colors[h_count:]

    cross_h = random.randrange(h_count)
    cross_v = random.randrange(v_count)
    cross_row = horizontal_rows[cross_h]
    cross_col = vertical_cols[cross_v]

    left_cols = [col for col in horizontal_candidates if col < cross_col]
    right_cols = [col for col in horizontal_candidates if col > cross_col]
    above_rows = [row for row in available_rows if row < cross_row]
    below_rows = [row for row in available_rows if row > cross_row]
    if not left_cols or not right_cols or not above_rows or not below_rows:
        return None

    horizontals: list[HorizontalGroup] = []
    for idx, (color, row, marker_count) in enumerate(
        zip(horizontal_colors, horizontal_rows, horizontal_counts)
    ):
        required = ()
        if idx == cross_h:
            required = (random.choice(left_cols), random.choice(right_cols))
        cols = _sample_nonadjacent_positions(horizontal_candidates, marker_count, required)
        if cols is None:
            return None
        horizontals.append((color, row, cols))

    verticals: list[VerticalGroup] = []
    for idx, (color, col, marker_count) in enumerate(
        zip(vertical_colors, vertical_cols, vertical_counts)
    ):
        required = ()
        if idx == cross_v:
            required = (random.choice(above_rows), random.choice(below_rows))
        rows = _sample_nonadjacent_positions(available_rows, marker_count, required)
        if rows is None:
            return None
        verticals.append((color, col, rows))

    if _count_intersections(horizontals, verticals) <= 0:
        return None
    return _render_grid(height, width, horizontals, verticals)


def _sample_separated_scene(height: int, width: int, h_count: int, v_count: int) -> np.ndarray | None:
    horizontal_rows = sorted(random.sample(range(1, height - 1), h_count))
    available_rows = [row for row in range(height) if row not in horizontal_rows]
    max_v_markers = min(MARKER_COUNT_MAX, _max_nonadjacent_count(available_rows))
    if max_v_markers < MARKER_COUNT_MIN:
        return None

    max_h_region = width - v_count - 1
    if max_h_region < 3:
        return None
    horizontal_region_size = random.randint(3, max_h_region)
    horizontal_on_left = bool(random.getrandbits(1))

    if horizontal_on_left:
        horizontal_candidates = list(range(horizontal_region_size))
        vertical_candidates = list(range(horizontal_region_size + 1, width))
    else:
        vertical_candidates = list(range(width - horizontal_region_size - 1))
        horizontal_candidates = list(range(width - horizontal_region_size, width))

    max_h_markers = min(MARKER_COUNT_MAX, _max_nonadjacent_count(horizontal_candidates))
    if len(vertical_candidates) < v_count or max_h_markers < MARKER_COUNT_MIN:
        return None

    horizontal_counts = [random.randint(MARKER_COUNT_MIN, max_h_markers) for _ in range(h_count)]
    vertical_counts = [random.randint(MARKER_COUNT_MIN, max_v_markers) for _ in range(v_count)]
    colors = random.sample(COLOR_POOL, h_count + v_count)
    horizontal_colors = colors[:h_count]
    vertical_colors = colors[h_count:]
    vertical_cols = sorted(random.sample(vertical_candidates, v_count))

    horizontals: list[HorizontalGroup] = []
    for color, row, marker_count in zip(horizontal_colors, horizontal_rows, horizontal_counts):
        cols = _sample_nonadjacent_positions(horizontal_candidates, marker_count)
        if cols is None:
            return None
        horizontals.append((color, row, cols))

    verticals: list[VerticalGroup] = []
    for color, col, marker_count in zip(vertical_colors, vertical_cols, vertical_counts):
        rows = _sample_nonadjacent_positions(available_rows, marker_count)
        if rows is None:
            return None
        verticals.append((color, col, rows))

    if _count_intersections(horizontals, verticals) != 0:
        return None
    return _render_grid(height, width, horizontals, verticals)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    h_count = random.randint(1, min(GROUP_COUNT_MAX, height - 2))
    v_count = random.randint(1, min(GROUP_COUNT_MAX, width - 2))

    if height - h_count < MARKER_COUNT_MIN:
        return None

    want_crossings = bool(random.getrandbits(1))
    if want_crossings:
        grid = _sample_crossing_scene(height, width, h_count, v_count)
    else:
        grid = _sample_separated_scene(height, width, h_count, v_count)
    if grid is None:
        return None

    horizontals, verticals = _parse_groups(grid)
    crossing_count = _count_intersections(horizontals, verticals)
    if want_crossings != (crossing_count > 0):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out == INTERSECTION_COLOR)) != crossing_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
