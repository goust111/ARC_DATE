from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


class Color:
    BLACK = 0
    RED = 2


FILL_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 24
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    black = np.argwhere(grid == Color.BLACK)
    assert black.shape == (4, 2), "expected exactly one 2x2 black marker"
    r0, c0, r1, c1 = _bbox_coords(black)
    assert r1 == r0 + 1 and c1 == c0 + 1, "black cells must form a 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK), "marker must be solid"
    return r0, c0


def _relay_closure(grid: np.ndarray) -> tuple[set[int], set[int]]:
    r0, c0 = _find_marker(grid)
    red = grid == Color.RED

    active_rows = {r0, r0 + 1}
    active_cols = {c0, c0 + 1}
    row_queue = deque(active_rows)
    col_queue = deque(active_cols)
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()

    while row_queue or col_queue:
        while row_queue:
            row = int(row_queue.popleft())
            if row in seen_rows:
                continue
            seen_rows.add(row)
            for col in np.flatnonzero(red[row]):
                col = int(col)
                if col not in active_cols:
                    active_cols.add(col)
                    col_queue.append(col)

        while col_queue:
            col = int(col_queue.popleft())
            if col in seen_cols:
                continue
            seen_cols.add(col)
            for row in np.flatnonzero(red[:, col]):
                row = int(row)
                if row not in active_rows:
                    active_rows.add(row)
                    row_queue.append(row)

    return active_rows, active_cols


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    active_rows, active_cols = _relay_closure(g)

    erase = np.zeros(g.shape, dtype=bool)
    erase[list(active_rows), :] = True
    erase[:, list(active_cols)] = True

    out = g.copy()
    out[erase & (g != Color.RED)] = Color.BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the 2x2 black marker to launch a red-relayed row/column erasure closure.

    Input structure:
    - The grid is otherwise filled with non-black colors and contains exactly one 2x2 black marker.
    - Red pixels may lie on the marker's rows or columns or elsewhere in the canvas.
    - Red pixels are the only cells that can survive erasure and relay it perpendicularly.

    Rule:
    - Start by erasing the two marker rows and the two marker columns: every selected non-red cell becomes black.
    - Whenever an erased row hits a red pixel, that red pixel relays the erasure to its whole column.
    - Whenever an erased column hits a red pixel, that red pixel relays the erasure to its whole row.
    - Repeat until no new rows or columns are added; cells outside the final erased rows/columns stay unchanged.

    Key insight:
    - Red pixels are not mere holes in the first cross: they define a bipartite row-column relay graph whose closure determines the full erased pattern.
    """
    return _apply_rule(grid)


def _pick_lines(limit: int, banned: set[int], count: int) -> list[int] | None:
    candidates = [idx for idx in range(limit) if idx not in banned]
    if len(candidates) < count:
        return None

    for _ in range(120):
        chosen = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen

    return sorted(random.sample(candidates, count))


def _graph_closure(
    seed_rows: set[int], seed_cols: set[int], red_positions: set[tuple[int, int]]
) -> tuple[set[int], set[int]]:
    row_to_cols: dict[int, set[int]] = defaultdict(set)
    col_to_rows: dict[int, set[int]] = defaultdict(set)
    for row, col in red_positions:
        row_to_cols[int(row)].add(int(col))
        col_to_rows[int(col)].add(int(row))

    active_rows = set(map(int, seed_rows))
    active_cols = set(map(int, seed_cols))
    row_queue = deque(active_rows)
    col_queue = deque(active_cols)
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()

    while row_queue or col_queue:
        while row_queue:
            row = int(row_queue.popleft())
            if row in seen_rows:
                continue
            seen_rows.add(row)
            for col in row_to_cols[row]:
                if col not in active_cols:
                    active_cols.add(col)
                    col_queue.append(col)

        while col_queue:
            col = int(col_queue.popleft())
            if col in seen_cols:
                continue
            seen_cols.add(col)
            for row in col_to_rows[col]:
                if row not in active_rows:
                    active_rows.add(row)
                    row_queue.append(row)

    return active_rows, active_cols


def _first_wave_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    r0, c0 = _find_marker(g)
    erase = np.zeros(g.shape, dtype=bool)
    erase[r0 : r0 + 2, :] = True
    erase[:, c0 : c0 + 2] = True
    out = g.copy()
    out[erase & (g != Color.RED)] = Color.BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    marker_r = random.randint(1, height - 3)
    marker_c = random.randint(1, width - 3)

    seed_rows = {marker_r, marker_r + 1}
    seed_cols = {marker_c, marker_c + 1}

    extra_row_count = random.randint(3, 4)
    extra_col_count = random.randint(3, 4)
    extra_rows = _pick_lines(height, seed_rows, extra_row_count)
    extra_cols = _pick_lines(width, seed_cols, extra_col_count)
    if extra_rows is None or extra_cols is None:
        return None

    row_a, row_b, row_c = extra_rows[:3]
    col_a, col_b, col_c = extra_cols[:3]

    intended_rows = seed_rows | set(extra_rows)
    intended_cols = seed_cols | set(extra_cols)

    seed_row_list = sorted(seed_rows)
    seed_col_list = sorted(seed_cols)
    red_positions = {
        (seed_row_list[0], col_a),
        (seed_row_list[1], col_b),
        (row_a, seed_col_list[0]),
        (row_b, seed_col_list[1]),
        (row_a, col_c),
        (row_c, col_a),
    }

    if len(extra_cols) > 3:
        col_d = extra_cols[3]
        red_positions.add((random.choice([row_b, row_c]), col_d))
    if len(extra_rows) > 3:
        row_d = extra_rows[3]
        red_positions.add((row_d, random.choice([col_b, col_c])))

    inactive_rows = [row for row in range(height) if row not in intended_rows]
    inactive_cols = [col for col in range(width) if col not in intended_cols]
    if len(inactive_rows) < 2 or len(inactive_cols) < 2:
        return None

    distractor_pool = [(row, col) for row in inactive_rows for col in inactive_cols]
    random.shuffle(distractor_pool)
    distractor_count = min(len(distractor_pool), random.randint(2, 6))
    distractors = set(distractor_pool[:distractor_count])
    red_positions |= distractors

    closure_rows, closure_cols = _graph_closure(seed_rows, seed_cols, red_positions)
    if closure_rows != intended_rows or closure_cols != intended_cols:
        return None

    palette = random.sample(FILL_COLORS, random.randint(4, len(FILL_COLORS)))
    grid = np.random.choice(palette, size=(height, width)).astype(int)
    for row, col in red_positions:
        grid[row, col] = Color.RED
    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = Color.BLACK

    try:
        solved = main(grid)
    except Exception:
        return None

    if np.array_equal(solved, _first_wave_only(grid)):
        return None

    solved_rows, solved_cols = _relay_closure(grid)
    if solved_rows != intended_rows or solved_cols != intended_cols:
        return None

    for row, col in distractors:
        if row in solved_rows or col in solved_cols:
            return None
        if solved[row, col] != Color.RED:
            return None

    if solved[row_c, col_a] != Color.RED:
        return None
    if solved[row_a, col_c] != Color.RED:
        return None

    erased_mask = np.zeros(grid.shape, dtype=bool)
    erased_mask[list(solved_rows), :] = True
    erased_mask[:, list(solved_cols)] = True
    if not np.any(erased_mask & (grid != Color.RED) & (grid != Color.BLACK)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
