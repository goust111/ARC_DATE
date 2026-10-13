from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MARKER_COLORS = (1, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 500


def _edge_marker_color(values: np.ndarray, role: str) -> int:
    colors = [int(x) for x in np.unique(values) if int(x) != BLACK]
    assert len(colors) == 1, f"expected one {role} marker color"
    color = colors[0]
    assert color not in {RED, GREY}, f"{role} marker color must not be red or grey"
    return color


def _read_markers(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 5 and width >= 5, "expected room for rays and walls"
    assert int(g[0, 0]) == BLACK, "the corner between marker bands must be black"

    top_color = _edge_marker_color(g[0, 1:], "top-row")
    left_color = _edge_marker_color(g[1:, 0], "left-column")
    assert top_color != left_color, "row and column markers must use distinct colors"

    assert np.all((g[0, 1:] == BLACK) | (g[0, 1:] == top_color))
    assert np.all((g[1:, 0] == BLACK) | (g[1:, 0] == left_color))
    interior_colors = set(map(int, np.unique(g[1:, 1:])))
    assert interior_colors.issubset({BLACK, GREY}), "interior must contain only background and grey walls"

    marker_cols = [c for c in range(1, width) if int(g[0, c]) == top_color]
    marker_rows = [r for r in range(1, height) if int(g[r, 0]) == left_color]
    assert marker_cols, "expected at least one top marker"
    assert marker_rows, "expected at least one left marker"
    return top_color, left_color, marker_cols, marker_rows


def _first_wall_down(g: np.ndarray, col: int) -> int | None:
    for r in range(1, g.shape[0]):
        if int(g[r, col]) == GREY:
            return r
    return None


def _first_wall_right(g: np.ndarray, row: int) -> int | None:
    for c in range(1, g.shape[1]):
        if int(g[row, c]) == GREY:
            return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stain the first grey wall hit by each border ray and paint still-visible row-column crossings red.

    Input structure:
    - The grid background is black.
    - The top row contains one color of column marker pixels; the left column contains a different color of row marker pixels.
    - The interior contains only black cells and scattered grey wall cells.

    Rule:
    - Each top marker casts a ray downward in its column; its first grey wall, if any, is recolored to the top-marker color.
    - Each left marker casts a ray rightward in its row; its first grey wall, if any, is recolored to the left-marker color.
    - For every marked row and marked column crossing, paint the crossing red only when it lies before both corresponding first walls.
    - Border markers, unhit grey walls, and rays with no wall hit are otherwise left unchanged.

    Key insight:
    - The red cells are not all marked row-column crossings; they are exactly the crossings where both perpendicular rays are still alive before their own first wall.
    """
    g = np.asarray(grid, dtype=int)
    top_color, left_color, marker_cols, marker_rows = _read_markers(g)

    col_stops = {c: _first_wall_down(g, c) for c in marker_cols}
    row_stops = {r: _first_wall_right(g, r) for r in marker_rows}

    stains: dict[tuple[int, int], int] = {}
    for c, stop in col_stops.items():
        if stop is None:
            continue
        stains[(stop, c)] = top_color

    for r, stop in row_stops.items():
        if stop is None:
            continue
        cell = (r, stop)
        previous = stains.get(cell)
        assert previous is None or previous == left_color, "a first wall cannot be claimed by both marker colors"
        stains[cell] = left_color

    out = g.copy()
    for (r, c), color in stains.items():
        out[r, c] = color

    for r in marker_rows:
        row_stop = row_stops[r]
        for c in marker_cols:
            col_stop = col_stops[c]
            column_alive = col_stop is None or r < col_stop
            row_alive = row_stop is None or c < row_stop
            if column_alive and row_alive:
                assert int(g[r, c]) == BLACK, "visible crossings must be empty before painting"
                out[r, c] = RED

    return out


def _sample_positions(start: int, stop: int, count: int) -> list[int]:
    values = list(range(start, stop + 1))
    for gap in (2, 1):
        for _ in range(100):
            sample = sorted(random.sample(values, count))
            if all(b - a >= gap for a, b in zip(sample, sample[1:])):
                return sample
    return sorted(random.sample(values, count))


def _choose_cut(markers: list[int], limit: int, forbidden: set[int]) -> int | None:
    splitters = [
        x
        for x in range(1, limit + 1)
        if x not in forbidden and any(m < x for m in markers) and any(m > x for m in markers)
    ]
    if splitters:
        return random.choice(splitters)

    fallback = [x for x in range(1, limit + 1) if x not in forbidden]
    if fallback:
        return random.choice(fallback)
    return None


def _extra_wall_candidates(
    height: int,
    width: int,
    marker_cols: list[int],
    marker_rows: list[int],
    col_stops: dict[int, int | None],
    row_stops: dict[int, int | None],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    col_set = set(marker_cols)
    row_set = set(marker_rows)
    candidates = []
    for r in range(1, height):
        for c in range(1, width):
            if (r, c) in occupied:
                continue
            if c in col_set:
                stop = col_stops[c]
                if stop is None or r <= stop:
                    continue
            if r in row_set:
                stop = row_stops[r]
                if stop is None or c <= stop:
                    continue
            candidates.append((r, c))
    return candidates


def _visible_crossings(
    marker_cols: list[int],
    marker_rows: list[int],
    col_stops: dict[int, int | None],
    row_stops: dict[int, int | None],
) -> tuple[set[tuple[int, int]], bool, bool]:
    red_cells: set[tuple[int, int]] = set()
    has_column_blocked_only = False
    has_row_blocked_only = False
    for r in marker_rows:
        for c in marker_cols:
            column_alive = col_stops[c] is None or r < col_stops[c]
            row_alive = row_stops[r] is None or c < row_stops[r]
            if column_alive and row_alive:
                red_cells.add((r, c))
            elif not column_alive and row_alive:
                has_column_blocked_only = True
            elif column_alive and not row_alive:
                has_row_blocked_only = True
    return red_cells, has_column_blocked_only, has_row_blocked_only


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        top_color, left_color, marker_cols, marker_rows = _read_markers(inp)
    except AssertionError:
        return False

    col_stops = {c: _first_wall_down(inp, c) for c in marker_cols}
    row_stops = {r: _first_wall_right(inp, r) for r in marker_rows}
    if not any(v is None for v in col_stops.values()):
        return False
    if not any(v is None for v in row_stops.values()):
        return False
    if not any(v is not None for v in col_stops.values()):
        return False
    if not any(v is not None for v in row_stops.values()):
        return False

    for c, stop in col_stops.items():
        if stop is not None and int(out[stop, c]) != top_color:
            return False
    for r, stop in row_stops.items():
        if stop is not None and int(out[r, stop]) != left_color:
            return False

    red_cells, has_column_blocked_only, has_row_blocked_only = _visible_crossings(
        marker_cols, marker_rows, col_stops, row_stops
    )
    total_crossings = len(marker_cols) * len(marker_rows)
    actual_red = {tuple(map(int, cell)) for cell in np.argwhere(out == RED)}
    if actual_red != red_cells:
        return False
    if not red_cells or len(red_cells) >= total_crossings:
        return False
    if not has_column_blocked_only or not has_row_blocked_only:
        return False

    untouched_grey = np.argwhere((inp == GREY) & (out == GREY))
    if untouched_grey.size == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 18)
    width = random.randint(10, 18)
    marker_col_count = random.randint(3, min(5, width - 2))
    marker_row_count = random.randint(3, min(5, height - 2))

    marker_cols = _sample_positions(1, width - 1, marker_col_count)
    marker_rows = _sample_positions(1, height - 1, marker_row_count)
    top_color, left_color = random.sample(MARKER_COLORS, 2)

    grid = np.zeros((height, width), dtype=int)
    for c in marker_cols:
        grid[0, c] = top_color
    for r in marker_rows:
        grid[r, 0] = left_color

    blocked_cols = set(random.sample(marker_cols, random.randint(1, len(marker_cols) - 1)))
    blocked_rows = set(random.sample(marker_rows, random.randint(1, len(marker_rows) - 1)))

    col_stops: dict[int, int | None] = {c: None for c in marker_cols}
    row_stops: dict[int, int | None] = {r: None for r in marker_rows}
    occupied: set[tuple[int, int]] = set()

    forbidden_rows = set(marker_rows)
    forbidden_cols = set(marker_cols)
    for c in blocked_cols:
        stop = _choose_cut(marker_rows, height - 1, forbidden_rows)
        if stop is None:
            return None
        col_stops[c] = stop
        occupied.add((stop, c))

    for r in blocked_rows:
        stop = _choose_cut(marker_cols, width - 1, forbidden_cols)
        if stop is None:
            return None
        row_stops[r] = stop
        occupied.add((r, stop))

    for r, c in occupied:
        grid[r, c] = GREY

    extra_candidates = _extra_wall_candidates(
        height, width, marker_cols, marker_rows, col_stops, row_stops, occupied
    )
    if not extra_candidates:
        return None
    extra_count = random.randint(1, min(len(extra_candidates), max(2, (height * width) // 30), 8))
    for r, c in random.sample(extra_candidates, extra_count):
        grid[r, c] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _semantic_check(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid line-of-sight wall input")
