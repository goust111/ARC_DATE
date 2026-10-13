import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_MIN = 10
GRID_MAX = 14
BLUE_COUNT_MIN = 5
BLUE_COUNT_MAX = 8
GEN_RETRIES = 400


def _settle_blue_columns(grid: np.ndarray) -> np.ndarray:
    """Apply stable downward gravity to blue pixels in non-border columns."""
    settled = np.asarray(grid, dtype=int).copy()
    height, width = map(int, settled.shape)
    for col in range(1, width - 1):
        column = settled[:, col].copy()
        blue_rows = np.flatnonzero(column == BLUE)
        if blue_rows.size == 0:
            continue
        column[blue_rows] = BLACK
        column[height - blue_rows.size :] = BLUE
        settled[:, col] = column
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the interior blue pixels by column gravity, then color every newly unobstructed red-to-red corridor green.

    Input structure:
    - The grid has a black background.
    - Some rows have red cells on both the left and right borders; other border rows may have only one red endpoint or none.
    - The interior uses sparse blue single pixels as blockers, with at most two blues per non-border column.

    Rule:
    - For each non-border column, collect its blue pixels from top to bottom, clear those cells to black, and place the same number of blue pixels packed against the bottom of that column.
    - After gravity, scan each row. If its leftmost and rightmost cells are red and every interior cell is black, recolor that full interior span green.
    - Keep all red border markers and all gravity-settled blue pixels unchanged.

    Key insight:
    - Corridor detection happens after gravity, so rows that were blocked in the input can become clear only because the blue blockers fall to the bottom first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected room for interior cells"

    settled = _settle_blue_columns(g)
    out = settled.copy()
    for row in range(height):
        if int(settled[row, 0]) == RED and int(settled[row, width - 1]) == RED:
            if np.all(settled[row, 1 : width - 1] == BLACK):
                out[row, 1 : width - 1] = GREEN
    return out


def _choose_column_counts(width: int, total_blue: int, rng: np.random.Generator) -> dict[int, int] | None:
    internal_cols = list(range(1, width - 1))
    options: list[tuple[int, int]] = []
    for two_count in range(1, min(len(internal_cols), total_blue // 2) + 1):
        one_count = total_blue - 2 * two_count
        if one_count < 0:
            continue
        col_count = two_count + one_count
        if 3 <= col_count <= len(internal_cols):
            options.append((two_count, one_count))
    if not options:
        return None

    two_count, one_count = options[int(rng.integers(len(options)))]
    chosen_cols = rng.choice(np.array(internal_cols), size=two_count + one_count, replace=False)
    counts = [2] * two_count + [1] * one_count
    rng.shuffle(counts)
    return {int(col): count for col, count in zip(chosen_cols, counts)}


def _assign_source_rows(
    height: int,
    column_counts: dict[int, int],
    opened_rows: list[int],
    rng: np.random.Generator,
) -> dict[int, list[int]] | None:
    assignments = {col: [] for col in column_counts}
    columns = list(column_counts)
    rng.shuffle(columns)

    for row in opened_rows:
        candidates = [
            col
            for col in columns
            if len(assignments[col]) < column_counts[col] and row not in assignments[col]
        ]
        if not candidates:
            return None
        remaining = [column_counts[col] - len(assignments[col]) for col in candidates]
        max_remaining = max(remaining)
        filtered = [col for col in candidates if column_counts[col] - len(assignments[col]) == max_remaining]
        chosen = int(filtered[int(rng.integers(len(filtered)))])
        assignments[chosen].append(row)

    movable_rows = list(range(0, max(height - 1, 1)))
    all_rows = list(range(height))
    for col, count in column_counts.items():
        while len(assignments[col]) < count:
            preferred = [row for row in movable_rows if row not in assignments[col]]
            fallback = [row for row in all_rows if row not in assignments[col]]
            pool = preferred if preferred else fallback
            if not pool:
                return None
            row = int(pool[int(rng.integers(len(pool)))])
            assignments[col].append(row)
        assignments[col].sort()
    return assignments


def _paired_rows_that_clear(input_grid: np.ndarray, settled: np.ndarray) -> tuple[list[int], list[int]]:
    clear_rows: list[int] = []
    blocked_rows: list[int] = []
    for row in range(input_grid.shape[0]):
        if int(input_grid[row, 0]) != RED or int(input_grid[row, -1]) != RED:
            continue
        if np.all(settled[row, 1:-1] == BLACK):
            clear_rows.append(row)
        else:
            blocked_rows.append(row)
    return clear_rows, blocked_rows


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    if height < 6 or width < 5:
        return None

    open_pool = list(range(1, height - 3))
    if len(open_pool) < 2:
        return None
    open_count = int(rng.integers(2, min(4, len(open_pool)) + 1))
    opened_rows = sorted(int(row) for row in rng.choice(np.array(open_pool), size=open_count, replace=False))

    total_blue = int(rng.integers(BLUE_COUNT_MIN, BLUE_COUNT_MAX + 1))
    column_counts = _choose_column_counts(width, total_blue, rng)
    if column_counts is None or max(column_counts.values()) < 2:
        return None

    source_rows = _assign_source_rows(height, column_counts, opened_rows, rng)
    if source_rows is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)

    paired_rows = set(opened_rows)
    paired_rows.add(height - 2)
    if rng.random() < 0.5:
        paired_rows.add(height - 1)
    spare_rows = [row for row in range(height) if row not in paired_rows]
    if spare_rows and rng.random() < 0.5:
        paired_rows.add(int(spare_rows[int(rng.integers(len(spare_rows)))]))

    for row in paired_rows:
        grid[row, 0] = RED
        grid[row, width - 1] = RED

    non_paired = [row for row in range(height) if row not in paired_rows]
    rng.shuffle(non_paired)
    extra_count = int(rng.integers(0, min(4, len(non_paired)) + 1))
    for idx, row in enumerate(non_paired[:extra_count]):
        if idx % 2 == 0:
            grid[row, 0] = RED
        else:
            grid[row, width - 1] = RED

    for col, rows in source_rows.items():
        for row in rows:
            if grid[row, col] != BLACK:
                return None
            grid[row, col] = BLUE

    if int(np.count_nonzero(grid == BLUE)) != total_blue:
        return None

    for col in range(1, width - 1):
        if int(np.count_nonzero(grid[:, col] == BLUE)) > 2:
            return None

    settled = _settle_blue_columns(grid)
    clear_rows, blocked_rows = _paired_rows_that_clear(grid, settled)
    opened_after = [row for row in opened_rows if row in clear_rows]
    opened_before = [row for row in opened_rows if np.any(grid[row, 1:-1] == BLUE)]
    if len(opened_after) != len(opened_rows):
        return None
    if len(opened_before) != len(opened_rows):
        return None
    if (height - 2) not in blocked_rows:
        return None
    if np.array_equal(grid, settled):
        return None

    out = main(grid)
    for row in opened_rows:
        if not np.all(out[row, 1:-1] == GREEN):
            return None
    if np.any(out[height - 2, 1:-1] == GREEN):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
