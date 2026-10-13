from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


ORANGE = 7
BLUE = 1
RED = 2
GREEN = 3
MARKER_COLORS = (BLUE, RED)


def _row_col_maps(coords: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))
    for values in rows.values():
        values.sort()
    for values in cols.values():
        values.sort()
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect orthogonally aligned same-color marker pairs, mark red/blue crossings in green,
    and erase original markers except for crowded-row/column exceptions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 6 and w >= 6, "grid is too small for a readable instance"

    palette = set(map(int, np.unique(g)))
    assert palette <= {ORANGE, BLUE, RED}, "unexpected input colors"
    assert any(color in palette for color in MARKER_COLORS), "expected at least one marker color"

    blue_mask = np.zeros((h, w), dtype=bool)
    red_mask = np.zeros((h, w), dtype=bool)
    preserve: dict[int, set[tuple[int, int]]] = {BLUE: set(), RED: set()}

    for color, mask in ((BLUE, blue_mask), (RED, red_mask)):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        rows, cols = _row_col_maps(coords)

        for r, row_cols in rows.items():
            if len(row_cols) == 2:
                c0, c1 = row_cols
                mask[r, c0 : c1 + 1] = True
            elif len(row_cols) > 2:
                preserve[color].update((r, c) for c in row_cols)

        for c, col_rows in cols.items():
            if len(col_rows) == 2:
                r0, r1 = col_rows
                mask[r0 : r1 + 1, c] = True
            elif len(col_rows) > 2:
                preserve[color].update((r, c) for r in col_rows)

    overlap = blue_mask & red_mask
    out = np.full((h, w), ORANGE, dtype=int)
    out[blue_mask] = BLUE
    out[red_mask] = RED
    out[overlap] = GREEN

    for color in MARKER_COLORS:
        for r, c in preserve[color]:
            if out[r, c] == ORANGE:
                out[r, c] = color

    return out


# --------------- generator ---------------

MIN_SIZE = 8
MAX_SIZE = 17
MAX_ATTEMPTS = 400
PAIR_COUNT_RANGE = (0, 2)
NOISE_COUNT_RANGE = (0, 3)
FORCE_CROSS_PROB = 0.55
FORCE_CROWD_PROB = 0.55


def _counts_for_color(coords: set[tuple[int, int]], axis: int, index: int) -> int:
    return sum(1 for cell in coords if cell[axis] == index)


def _can_use_cells(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w and (r, c) not in occupied for r, c in cells)


def _place_cells(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
    cells: list[tuple[int, int]],
) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
        color_coords[color].add((r, c))


def _try_add_horizontal_pair(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
) -> bool:
    h, w = grid.shape
    coords = color_coords[color]
    row_choices = [r for r in range(h) if _counts_for_color(coords, 0, r) == 0]
    random.shuffle(row_choices)
    for r in row_choices:
        candidate_cols = [c for c in range(w) if _counts_for_color(coords, 1, c) == 0 and (r, c) not in occupied]
        if len(candidate_cols) < 2:
            continue
        random.shuffle(candidate_cols)
        for i in range(len(candidate_cols)):
            for j in range(i + 1, len(candidate_cols)):
                c0, c1 = sorted((candidate_cols[i], candidate_cols[j]))
                if c1 - c0 < 2:
                    continue
                _place_cells(grid, occupied, color_coords, color, [(r, c0), (r, c1)])
                return True
    return False


def _try_add_vertical_pair(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
) -> bool:
    h, w = grid.shape
    coords = color_coords[color]
    col_choices = [c for c in range(w) if _counts_for_color(coords, 1, c) == 0]
    random.shuffle(col_choices)
    for c in col_choices:
        candidate_rows = [r for r in range(h) if _counts_for_color(coords, 0, r) == 0 and (r, c) not in occupied]
        if len(candidate_rows) < 2:
            continue
        random.shuffle(candidate_rows)
        for i in range(len(candidate_rows)):
            for j in range(i + 1, len(candidate_rows)):
                r0, r1 = sorted((candidate_rows[i], candidate_rows[j]))
                if r1 - r0 < 2:
                    continue
                _place_cells(grid, occupied, color_coords, color, [(r0, c), (r1, c)])
                return True
    return False


def _try_add_crowded_row(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
) -> bool:
    h, w = grid.shape
    coords = color_coords[color]
    row_choices = [r for r in range(h) if _counts_for_color(coords, 0, r) == 0]
    random.shuffle(row_choices)
    for r in row_choices:
        cols = [c for c in range(w) if _counts_for_color(coords, 1, c) == 0 and (r, c) not in occupied]
        if len(cols) < 3:
            continue
        random.shuffle(cols)
        picked = sorted(cols[:3])
        _place_cells(grid, occupied, color_coords, color, [(r, c) for c in picked])
        return True
    return False


def _try_add_crowded_col(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
) -> bool:
    h, w = grid.shape
    coords = color_coords[color]
    col_choices = [c for c in range(w) if _counts_for_color(coords, 1, c) == 0]
    random.shuffle(col_choices)
    for c in col_choices:
        rows = [r for r in range(h) if _counts_for_color(coords, 0, r) == 0 and (r, c) not in occupied]
        if len(rows) < 3:
            continue
        random.shuffle(rows)
        picked = sorted(rows[:3])
        _place_cells(grid, occupied, color_coords, color, [(r, c) for r in picked])
        return True
    return False


def _try_add_isolated_marker(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
    color: int,
) -> bool:
    h, w = grid.shape
    coords = color_coords[color]
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in occupied and _counts_for_color(coords, 0, r) == 0 and _counts_for_color(coords, 1, c) == 0
    ]
    if not candidates:
        return False
    cell = random.choice(candidates)
    _place_cells(grid, occupied, color_coords, color, [cell])
    return True


def _try_add_crossing(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_coords: dict[int, set[tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    horizontal_color, vertical_color = random.choice(((BLUE, RED), (RED, BLUE)))
    h_coords = color_coords[horizontal_color]
    v_coords = color_coords[vertical_color]

    row_choices = [r for r in range(h) if _counts_for_color(h_coords, 0, r) == 0]
    col_choices = [c for c in range(w) if _counts_for_color(v_coords, 1, c) == 0]
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for r in row_choices:
        for c in col_choices:
            if (r, c) in occupied:
                continue
            left_cols = [x for x in range(0, c) if _counts_for_color(h_coords, 1, x) == 0 and (r, x) not in occupied]
            right_cols = [x for x in range(c + 1, w) if _counts_for_color(h_coords, 1, x) == 0 and (r, x) not in occupied]
            upper_rows = [y for y in range(0, r) if _counts_for_color(v_coords, 0, y) == 0 and (y, c) not in occupied]
            lower_rows = [y for y in range(r + 1, h) if _counts_for_color(v_coords, 0, y) == 0 and (y, c) not in occupied]
            if not left_cols or not right_cols or not upper_rows or not lower_rows:
                continue

            c0 = random.choice(left_cols)
            c1 = random.choice(right_cols)
            r0 = random.choice(upper_rows)
            r1 = random.choice(lower_rows)
            cells = [(r, c0), (r, c1), (r0, c), (r1, c)]
            if not _can_use_cells(cells, occupied, h, w):
                continue
            _place_cells(grid, occupied, color_coords, horizontal_color, [(r, c0), (r, c1)])
            _place_cells(grid, occupied, color_coords, vertical_color, [(r0, c), (r1, c)])
            return True
    return False


def _branch_counts(grid: np.ndarray) -> tuple[int, int]:
    out = main(grid)
    crossing_count = int(np.count_nonzero(out == GREEN))
    crowded_survivors = 0
    for color in MARKER_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rows, cols = _row_col_maps(coords)
        crowded = {
            (r, c)
            for r, row_cols in rows.items()
            if len(row_cols) > 2
            for c in row_cols
        }
        crowded.update(
            (r, c)
            for c, col_rows in cols.items()
            if len(col_rows) > 2
            for r in col_rows
        )
        crowded_survivors += sum(out[r, c] == color for r, c in crowded)
    return crossing_count, crowded_survivors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    grid = np.full((h, w), ORANGE, dtype=int)
    occupied: set[tuple[int, int]] = set()
    color_coords: dict[int, set[tuple[int, int]]] = {BLUE: set(), RED: set()}

    want_cross = random.random() < FORCE_CROSS_PROB
    want_crowd = random.random() < FORCE_CROWD_PROB

    if want_cross and not _try_add_crossing(grid, occupied, color_coords):
        return None

    for color in MARKER_COLORS:
        pair_attempts = random.randint(*PAIR_COUNT_RANGE)
        for _ in range(pair_attempts):
            if random.random() < 0.5:
                _try_add_horizontal_pair(grid, occupied, color_coords, color)
            else:
                _try_add_vertical_pair(grid, occupied, color_coords, color)

    if want_crowd:
        color = random.choice(MARKER_COLORS)
        crowded_ok = _try_add_crowded_row(grid, occupied, color_coords, color)
        if not crowded_ok:
            crowded_ok = _try_add_crowded_col(grid, occupied, color_coords, color)
        if not crowded_ok:
            return None

    for color in MARKER_COLORS:
        for _ in range(random.randint(*NOISE_COUNT_RANGE)):
            _try_add_isolated_marker(grid, occupied, color_coords, color)

    total_markers = sum(len(coords) for coords in color_coords.values())
    if total_markers < 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, np.full_like(grid, ORANGE)):
        return None

    crossing_count, crowded_survivors = _branch_counts(grid)
    if want_cross and crossing_count == 0:
        return None
    if want_crowd and crowded_survivors == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
