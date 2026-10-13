from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_RETRY_LIMIT = 500
HEIGHT_RANGE = (9, 15)
LEFT_HALF_RANGE = (5, 8)
RED_COUNT_RANGE = (3, 5)
BLUE_COUNT_RANGE = (1, 2)
YELLOW_COUNT_RANGE = (2, 3)
BLUE_BLOCK_PROB = 0.65
OBSTACLE_PRIORITY_PROB = 0.5


def _separator_col(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full grey separator column"
    sep = cols[0]
    assert sep == w // 2, "separator must be the central column"
    assert w % 2 == 1, "grid width must be odd"
    assert sep == w - sep - 1, "separator must split the grid into equal halves"
    assert int(np.count_nonzero(grid == GREY)) == h, "grey may only appear on the separator"
    return sep


def _apply_mirror(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep = _separator_col(g)
    for r in range(g.shape[0]):
        for c in range(sep):
            color = int(g[r, c])
            if color in (BG, GREY):
                continue
            mc = g.shape[1] - 1 - c
            if int(out[r, mc]) != YELLOW:
                out[r, mc] = color
    return out


def _row_is_clear_corridor(grid: np.ndarray, row: int, sep: int) -> bool:
    if int(grid[row, 0]) == BG or int(grid[row, -1]) == BG:
        return False
    if int(grid[row, 0]) != int(grid[row, -1]):
        return False
    return bool(np.all(grid[row, 1:sep] == BG) and np.all(grid[row, sep + 1 : -1] == BG))


def _fill_corridors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep = _separator_col(g)
    for r in range(g.shape[0]):
        if _row_is_clear_corridor(g, r, sep):
            out[r, 1:sep] = GREEN
            out[r, sep + 1 : -1] = GREEN
    return out


def _expected_clear_rows(inp: np.ndarray) -> set[int]:
    sep = _separator_col(inp)
    clear_rows: set[int] = set()
    for r in range(inp.shape[0]):
        if int(inp[r, 0]) != RED:
            continue
        if np.any(inp[r, 1:sep] == BLUE):
            continue
        if np.any(inp[r, sep + 1 : -1] == YELLOW):
            continue
        clear_rows.add(r)
    return clear_rows


def _validate_generated_input(inp: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    sep = _separator_col(g)

    left = g[:, :sep]
    right = g[:, sep + 1 :]

    allowed = {BG, BLUE, RED, YELLOW, GREY}
    assert set(map(int, np.unique(g))) <= allowed
    assert not np.any(g == GREEN), "green is reserved for the output"

    red_rows = {int(r) for r in np.flatnonzero(g[:, 0] == RED)}
    assert RED_COUNT_RANGE[0] <= len(red_rows) <= RED_COUNT_RANGE[1]
    assert np.all((g[:, 0] == BG) | (g[:, 0] == RED))

    blue_coords = np.argwhere(g == BLUE)
    yellow_coords = np.argwhere(g == YELLOW)
    assert BLUE_COUNT_RANGE[0] <= len(blue_coords) <= BLUE_COUNT_RANGE[1]
    assert YELLOW_COUNT_RANGE[0] <= len(yellow_coords) <= YELLOW_COUNT_RANGE[1]
    assert np.all(left[:, 1:] != GREY)
    assert np.all(right[:, :-1] != GREY)

    for r, c in blue_coords:
        assert 0 <= int(r) < g.shape[0]
        assert 1 <= int(c) < sep, "blue must be strictly inside the left half"

    for r, c in yellow_coords:
        assert 0 <= int(r) < g.shape[0]
        assert sep < int(c) < g.shape[1] - 1, "yellow must be strictly inside the right half"

    red_rows_with_yellow = {int(r) for r in red_rows if np.any(g[r, sep + 1 : -1] == YELLOW)}
    red_rows_without_yellow = red_rows - red_rows_with_yellow
    assert red_rows_with_yellow, "need at least one red row blocked by yellow"
    assert red_rows_without_yellow, "need at least one red row without yellow"

    expected_rows = _expected_clear_rows(g)
    assert expected_rows, "need at least one unobstructed corridor row"
    assert red_rows_with_yellow - expected_rows, "yellow-blocked red rows must remain blocked"

    mirrored = _apply_mirror(g)
    expected_out = _fill_corridors(mirrored)
    actual_out = main(g)
    assert np.array_equal(actual_out, expected_out)

    actual_rows = {int(r) for r in range(g.shape[0]) if np.any(actual_out[r] == GREEN)}
    assert actual_rows == expected_rows
    for r in actual_rows:
        assert int(actual_out[r, 0]) == RED and int(actual_out[r, -1]) == RED
        assert int(actual_out[r, sep]) == GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left-half markings to create right-edge endpoints, then fill only the unobstructed full-row corridors.

    Input structure:
    - A black background grid has odd width and one full central grey separator column.
    - The left boundary contains several red pixels on distinct rows, and the left interior contains one or two blue pixels.
    - The right half is otherwise empty except for a few internal yellow obstacle pixels.

    Rule:
    - Reflect every non-background, non-grey left-half pixel across the grey separator into the right half, preserving its color.
    - If a reflected target cell already holds a yellow obstacle, keep the yellow obstacle instead of overwriting it.
    - After reflection, scan each row: if the leftmost and rightmost cells are equal non-background colors and every other cell between them is background except the grey separator, fill those background cells with green.
    - Keep all original red, blue, yellow, grey, and reflected pixels unchanged; do not color the separator.

    Key insight:
    - Only mirrored left-edge reds can create same-color edge endpoints, while blue pixels and yellow obstacles act as blockers inside the span, so corridor filling depends on both the reflection step and the obstacle test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _separator_col(g)
    assert sep >= 2 and g.shape[1] - sep - 1 >= 2, "both halves must have interior cells"
    mirrored = _apply_mirror(g)
    return _fill_corridors(mirrored)


def _sample_dimensions() -> tuple[int, int]:
    h = random.randint(*HEIGHT_RANGE)
    left_w = random.randint(*LEFT_HALF_RANGE)
    return h, 2 * left_w + 1


def _choose_row_categories(
    h: int, red_count: int, blue_count: int, yellow_count: int
) -> tuple[set[int], set[int], set[int]]:
    red_rows = set(random.sample(range(h), red_count))
    remaining = list(red_rows)
    random.shuffle(remaining)

    clear_rows = {remaining.pop()}
    blue_block_rows: set[int] = set()
    if remaining and blue_count >= 1 and random.random() < BLUE_BLOCK_PROB:
        blue_block_rows.add(remaining.pop())

    max_yellow_red_rows = min(yellow_count, len(remaining))
    assert max_yellow_red_rows >= 1
    yellow_red_count = random.randint(1, max_yellow_red_rows)
    yellow_rows = set(remaining[:yellow_red_count])
    clear_rows.update(remaining[yellow_red_count:])
    return clear_rows, yellow_rows, blue_block_rows


def _place_reds(grid: np.ndarray, rows: set[int]) -> None:
    for r in rows:
        grid[r, 0] = RED


def _place_blues(
    grid: np.ndarray,
    sep: int,
    red_rows: set[int],
    clear_rows: set[int],
    blue_block_rows: set[int],
    blue_count: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []

    for r in blue_block_rows:
        c = random.randint(1, sep - 1)
        grid[r, c] = BLUE
        positions.append((r, c))

    needed = blue_count - len(positions)
    if needed <= 0:
        return positions

    candidates = [r for r in range(grid.shape[0]) if r not in red_rows]
    random.shuffle(candidates)
    for r in candidates[:needed]:
        c = random.randint(1, sep - 1)
        grid[r, c] = BLUE
        positions.append((r, c))

    assert len(positions) == blue_count
    assert clear_rows.isdisjoint({r for r, _ in positions}), "clear rows cannot contain blue blockers"
    return positions


def _random_yellow_col(sep: int, width: int) -> int:
    return random.randint(sep + 1, width - 2)


def _place_yellows(
    grid: np.ndarray,
    sep: int,
    yellow_rows: set[int],
    yellow_count: int,
    blue_positions: list[tuple[int, int]],
    protected_rows: set[int],
) -> None:
    used: set[tuple[int, int]] = set()

    for r in yellow_rows:
        c = _random_yellow_col(sep, grid.shape[1])
        grid[r, c] = YELLOW
        used.add((r, c))

    while len(used) < yellow_count:
        placed = False
        if blue_positions and random.random() < OBSTACLE_PRIORITY_PROB:
            candidates = blue_positions[:]
            random.shuffle(candidates)
            for r, c in candidates:
                mc = grid.shape[1] - 1 - c
                if (r, mc) in used or r in protected_rows:
                    continue
                grid[r, mc] = YELLOW
                used.add((r, mc))
                placed = True
                break
        if placed:
            continue

        candidate_rows = [r for r in range(grid.shape[0]) if r not in protected_rows]
        random.shuffle(candidate_rows)
        for r in candidate_rows:
            c = _random_yellow_col(sep, grid.shape[1])
            if (r, c) in used:
                continue
            grid[r, c] = YELLOW
            used.add((r, c))
            placed = True
            break
        assert placed, "failed to place the required yellow obstacles"


def _attempt_generate() -> np.ndarray | None:
    h, w = _sample_dimensions()
    sep = w // 2
    grid = np.full((h, w), BG, dtype=int)
    grid[:, sep] = GREY

    red_count = random.randint(*RED_COUNT_RANGE)
    blue_count = random.randint(*BLUE_COUNT_RANGE)
    yellow_count = random.randint(*YELLOW_COUNT_RANGE)

    clear_rows, yellow_rows, blue_block_rows = _choose_row_categories(
        h, red_count, blue_count, yellow_count
    )
    red_rows = clear_rows | yellow_rows | blue_block_rows
    assert len(red_rows) == red_count

    _place_reds(grid, red_rows)
    blue_positions = _place_blues(grid, sep, red_rows, clear_rows, blue_block_rows, blue_count)
    protected_rows = clear_rows | blue_block_rows
    _place_yellows(grid, sep, yellow_rows, yellow_count, blue_positions, protected_rows)

    try:
        _validate_generated_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-corridor input")
