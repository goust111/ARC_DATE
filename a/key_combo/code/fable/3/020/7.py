from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
FLOOR_COLOR = 2
MOVING_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

# Generator bounds.
GRID_HEIGHT_RANGE = (12, 25)
GRID_WIDTH_RANGE = (12, 25)
RED_COUNT_RANGE = (2, 4)
MAX_ACTIVE_COLUMNS = 7
MAX_STACK_HEIGHT = 3
GEN_ATTEMPTS = 400


def _sorted_red_points(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid == FLOOR_COLOR)
    assert coords.size > 0, "expected at least one red floor marker"
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    assert len(set(rows.tolist())) == len(rows), "red markers must use distinct rows"
    assert len(set(cols.tolist())) == len(cols), "red markers must use distinct columns"
    order = np.argsort(cols, kind="stable")
    return rows[order], cols[order]


def _unique_owner(red_cols: np.ndarray, col: int) -> int:
    distances = np.abs(red_cols - int(col))
    best = int(distances.min())
    owners = np.flatnonzero(distances == best)
    assert owners.size == 1, "moving pixel column must have a unique nearest red marker"
    return int(owners[0])


def _column_assignments(red_rows: np.ndarray, red_cols: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    owners = np.full(width, -1, dtype=int)
    floors = np.full(width, -1, dtype=int)
    for col in range(width):
        distances = np.abs(red_cols - col)
        best = int(distances.min())
        hits = np.flatnonzero(distances == best)
        if hits.size == 1:
            owner = int(hits[0])
            owners[col] = owner
            floors[col] = int(red_rows[owner])
    return owners, floors


def _is_isolated(grid: np.ndarray, row: int, col: int) -> bool:
    h, w = grid.shape
    for rr in range(max(0, row - 1), min(h, row + 2)):
        for cc in range(max(0, col - 1), min(w, col + 2)):
            if (rr, cc) != (row, col) and int(grid[rr, cc]) != BACKGROUND:
                return False
    return True


def _sample_spaced_values(values: Iterable[int], count: int, min_gap: int, attempts: int = 120) -> list[int] | None:
    pool = list(values)
    if len(pool) < count:
        return None
    for _ in range(attempts):
        picks = sorted(random.sample(pool, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _safe_rows_for_column(col: int, floor_row: int, red_rows: np.ndarray, red_cols: np.ndarray) -> list[int]:
    safe = []
    for row in range(floor_row):
        blocked = False
        for rr, cc in zip(red_rows.tolist(), red_cols.tolist()):
            if abs(row - rr) <= 1 and abs(col - cc) <= 1:
                blocked = True
                break
        if not blocked:
            safe.append(row)
    return safe


def _max_spaced_count(rows: list[int]) -> int:
    count = 0
    last = None
    for row in sorted(rows, reverse=True):
        if last is None or last - row >= 2:
            count += 1
            last = row
    return count


def _sample_spaced_rows(rows: list[int], count: int, attempts: int = 100) -> list[int] | None:
    if len(rows) < count:
        return None
    for _ in range(attempts):
        picks = sorted(random.sample(rows, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    red_rows, red_cols = _sorted_red_points(grid)
    moving_coords = np.argwhere((grid != BACKGROUND) & (grid != FLOOR_COLOR))
    if moving_coords.size == 0:
        return False
    for row, col in np.argwhere(grid != BACKGROUND):
        if not _is_isolated(grid, int(row), int(col)):
            return False
    counts_by_col: dict[int, int] = {}
    for row, col in moving_coords:
        owner = _unique_owner(red_cols, int(col))
        floor = int(red_rows[owner])
        if int(row) >= floor:
            return False
        counts_by_col[int(col)] = counts_by_col.get(int(col), 0) + 1
        if counts_by_col[int(col)] > floor:
            return False
    out = main(grid)
    return not np.array_equal(grid, out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored singleton onto the invisible full-row floor owned by its column's nearest red marker.

    Input structure:
    - The grid has a black background, several isolated red single pixels, and several isolated non-red single pixels.
    - Red pixels occupy distinct rows and distinct columns; each red pixel represents an invisible full-width floor at its row.
    - Every non-red pixel lies above the floor row of the uniquely nearest red pixel in horizontal distance.

    Rule:
    - Keep every red pixel fixed in place.
    - For each non-red pixel, assign its column to the red pixel whose column is horizontally nearest; that red pixel's row is the floor row for that column.
    - Move each non-red pixel straight downward within its own column until it reaches the cell just above that floor, stacking multiple pixels from the same column upward while preserving their bottom-to-top order.

    Key insight:
    - The visible red point is only a marker for an invisible whole-row floor, and the relevant floor for a column is chosen by horizontal nearest ownership rather than by vertical proximity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_rows, red_cols = _sorted_red_points(g)
    out = np.full_like(g, BACKGROUND)
    out[g == FLOOR_COLOR] = FLOOR_COLOR

    falling_by_col: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    for row, col in np.argwhere((g != BACKGROUND) & (g != FLOOR_COLOR)):
        row = int(row)
        col = int(col)
        owner = _unique_owner(red_cols, col)
        floor_row = int(red_rows[owner])
        assert row < floor_row, "moving pixels must begin above their owned floor"
        bucket = falling_by_col.setdefault(col, (floor_row, []))
        assert bucket[0] == floor_row, "each column should map to a single floor"
        bucket[1].append((row, int(g[row, col])))

    for col, (floor_row, items) in falling_by_col.items():
        ordered = sorted(items, key=lambda item: item[0], reverse=True)
        assert len(ordered) <= floor_row, "stack must fit above the floor"
        for offset, (_src_row, color) in enumerate(ordered, start=1):
            out[floor_row - offset, col] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_reds = min(RED_COUNT_RANGE[1], max(RED_COUNT_RANGE[0], width // 4), height - 6)
    if max_reds < RED_COUNT_RANGE[0]:
        return None
    red_count = random.randint(RED_COUNT_RANGE[0], max_reds)

    red_cols = _sample_spaced_values(range(1, width - 1), red_count, min_gap=3)
    if red_cols is None:
        return None
    red_rows = random.sample(range(4, height - 2), red_count)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for row, col in zip(red_rows, red_cols):
        grid[row, col] = FLOOR_COLOR

    red_rows_arr = np.array(red_rows, dtype=int)
    red_cols_arr = np.array(red_cols, dtype=int)
    owners, floors = _column_assignments(red_rows_arr, red_cols_arr, width)

    grouped_candidates: dict[int, list[int]] = {}
    safe_rows_by_col: dict[int, list[int]] = {}
    capacity_by_col: dict[int, int] = {}
    for col in range(width):
        owner = int(owners[col])
        floor_row = int(floors[col])
        if owner < 0 or floor_row < 4:
            continue
        safe_rows = _safe_rows_for_column(col, floor_row, red_rows_arr, red_cols_arr)
        capacity = min(MAX_STACK_HEIGHT, _max_spaced_count(safe_rows))
        if capacity <= 0:
            continue
        grouped_candidates.setdefault(owner, []).append(col)
        safe_rows_by_col[col] = safe_rows
        capacity_by_col[col] = capacity

    if len(grouped_candidates) != red_count:
        return None

    chosen_cols: list[int] = []
    for owner in range(red_count):
        candidates = grouped_candidates[owner][:]
        random.shuffle(candidates)
        preferred = [col for col in candidates if col != red_cols[owner]]
        if preferred and random.random() < 0.7:
            candidates = preferred + [col for col in candidates if col == red_cols[owner]]
        picked = None
        for col in candidates:
            if all(abs(col - other) >= 2 for other in chosen_cols):
                picked = col
                break
        if picked is None:
            return None
        chosen_cols.append(picked)

    all_candidates = [col for cols in grouped_candidates.values() for col in cols]
    random.shuffle(all_candidates)
    target_active = random.randint(red_count, min(MAX_ACTIVE_COLUMNS, len(set(all_candidates))))
    for col in all_candidates:
        if col in chosen_cols:
            continue
        if len(chosen_cols) >= target_active:
            break
        if all(abs(col - other) >= 2 for other in chosen_cols):
            chosen_cols.append(col)

    stack_candidates = [col for col in chosen_cols if capacity_by_col[col] >= 2]
    forced_stack_col = random.choice(stack_candidates) if stack_candidates and random.random() < 0.75 else None

    pixel_specs: list[tuple[int, int, int]] = []
    for idx, col in enumerate(sorted(chosen_cols)):
        capacity = capacity_by_col[col]
        if col == forced_stack_col:
            stack_size = random.randint(2, capacity)
        elif capacity >= 2 and random.random() < 0.35:
            stack_size = random.randint(1, capacity)
        else:
            stack_size = 1
        rows = _sample_spaced_rows(safe_rows_by_col[col], stack_size)
        if rows is None:
            return None
        if idx == 0 and col != forced_stack_col:
            final_rows = [int(floors[col]) - offset for offset in range(1, stack_size + 1)]
            if rows == sorted(final_rows):
                movable_rows = [row for row in safe_rows_by_col[col] if row not in final_rows]
                alt_rows = _sample_spaced_rows(movable_rows, stack_size)
                if alt_rows is None:
                    return None
                rows = alt_rows
        for row in rows:
            pixel_specs.append((row, col, random.choice(MOVING_COLORS)))

    if len(pixel_specs) < red_count + 1:
        return None

    if len(pixel_specs) >= 2 and len({color for _, _, color in pixel_specs}) == 1:
        first = pixel_specs[0]
        replacement_pool = [color for color in MOVING_COLORS if color != first[2]]
        pixel_specs[0] = (first[0], first[1], random.choice(replacement_pool))

    for row, col, color in pixel_specs:
        if not _is_isolated(grid, row, col):
            return None
        grid[row, col] = color

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
