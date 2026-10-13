import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8
FOREGROUND_COLORS = (RED, TEAL)

GENERATION_RETRIES = 400
WINNER_AXES = ("row", "col")


def _find_corridors(grid: np.ndarray) -> list[tuple[str, int, int]]:
    h, w = map(int, grid.shape)
    corridors: list[tuple[str, int, int]] = []

    for r in range(h):
        if int(grid[r, 0]) != BLACK and int(grid[r, w - 1]) != BLACK and np.all(grid[r, 1 : w - 1] == BLACK):
            corridors.append(("row", r, w - 2))

    for c in range(w):
        if int(grid[0, c]) != BLACK and int(grid[h - 1, c]) != BLACK and np.all(grid[1 : h - 1, c] == BLACK):
            corridors.append(("col", c, h - 2))

    return corridors


def main(grid: np.ndarray) -> np.ndarray:
    """Repeatedly paint the unique longest border-to-border clear corridor green until none remain.

    Input structure:
    - The grid has a black background with red and teal foreground markers on the outer border.
    - Some rows have foreground cells at both horizontal edges, and some columns have foreground cells at both vertical edges.
    - Interior foreground cells act as blockers; a corridor is valid only when its whole interior span is still black.

    Rule:
    - Recompute all candidate row and column corridors on the current grid.
    - A row is a candidate when its left and right edge cells are foreground and every interior cell in that row is black; columns are defined analogously.
    - Select the unique candidate with the largest interior length, paint its interior cells green, then treat those new green cells as blockers for later iterations.
    - Repeat until no candidate corridor remains, preserving all other cells.

    Key insight:
    - The corridors cannot be painted independently from the initial grid: painting one corridor creates new foreground blockers that can destroy shorter crossing candidates, so the choice must be recomputed greedily after each step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "expected room for corridor interiors"

    out = g.copy()
    while True:
        corridors = _find_corridors(out)
        if not corridors:
            return out

        max_len = max(length for _, _, length in corridors)
        longest = [corridor for corridor in corridors if corridor[2] == max_len]
        assert len(longest) == 1, "expected a unique longest corridor"

        axis, index, _ = longest[0]
        if axis == "row":
            out[index, 1 : w - 1] = GREEN
        else:
            out[1 : h - 1, index] = GREEN


def _random_fg(rng: np.random.Generator) -> int:
    return int(rng.choice(FOREGROUND_COLORS))


def _paint_row_endpoints(grid: np.ndarray, row: int, rng: np.random.Generator) -> None:
    grid[row, 0] = _random_fg(rng)
    grid[row, -1] = _random_fg(rng)


def _paint_col_endpoints(grid: np.ndarray, col: int, rng: np.random.Generator) -> None:
    grid[0, col] = _random_fg(rng)
    grid[-1, col] = _random_fg(rng)


def _choose_dimensions(rng: np.random.Generator, winner_axis: str) -> tuple[int, int]:
    if winner_axis == "row":
        h = int(rng.integers(8, 14))
        w = int(rng.integers(max(10, h + 2), min(20, h + 7) + 1))
    else:
        w = int(rng.integers(8, 14))
        h = int(rng.integers(max(10, w + 2), min(20, w + 7) + 1))
    return h, w


def _sample_without_replacement(
    rng: np.random.Generator, values: list[int], count_min: int, count_max: int
) -> list[int]:
    if not values:
        return []
    count = int(rng.integers(count_min, min(count_max, len(values)) + 1))
    if count == 0:
        return []
    return [int(v) for v in rng.choice(values, size=count, replace=False)]


def _add_blocked_rows(
    grid: np.ndarray,
    rows: list[int],
    forbidden_cols: set[int],
    rng: np.random.Generator,
) -> bool:
    h, w = map(int, grid.shape)
    assert h >= 3 and w >= 3
    for row in rows:
        _paint_row_endpoints(grid, row, rng)
        blocker_cols = [c for c in range(1, w - 1) if c not in forbidden_cols and int(grid[row, c]) == BLACK]
        if not blocker_cols:
            return False
        col = int(rng.choice(blocker_cols))
        grid[row, col] = _random_fg(rng)
    return True


def _add_blocked_cols(
    grid: np.ndarray,
    cols: list[int],
    forbidden_rows: set[int],
    rng: np.random.Generator,
) -> bool:
    h, w = map(int, grid.shape)
    assert h >= 3 and w >= 3
    for col in cols:
        _paint_col_endpoints(grid, col, rng)
        blocker_rows = [r for r in range(1, h - 1) if r not in forbidden_rows and int(grid[r, col]) == BLACK]
        if not blocker_rows:
            return False
        row = int(rng.choice(blocker_rows))
        grid[row, col] = _random_fg(rng)
    return True


def _add_extra_blockers(
    grid: np.ndarray,
    clear_rows: set[int],
    clear_cols: set[int],
    rng: np.random.Generator,
) -> None:
    h, w = map(int, grid.shape)
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if r not in clear_rows and c not in clear_cols and int(grid[r, c]) == BLACK
    ]
    if not candidates:
        return

    count = int(rng.integers(2, min(8, len(candidates)) + 1))
    picks = rng.choice(len(candidates), size=count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = candidates[int(idx)]
        grid[r, c] = _random_fg(rng)


def _ensure_both_input_colors(grid: np.ndarray, rng: np.random.Generator) -> None:
    colors_present = {int(v) for v in np.unique(grid) if int(v) != BLACK}
    if colors_present == set(FOREGROUND_COLORS):
        return
    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        return
    r, c = map(int, coords[int(rng.integers(len(coords)))])
    grid[r, c] = TEAL if RED in colors_present else RED


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    winner_axis = str(rng.choice(WINNER_AXES))
    h, w = _choose_dimensions(rng, winner_axis)
    grid = np.zeros((h, w), dtype=int)

    if winner_axis == "row":
        winner_row = int(rng.integers(1, h - 1))
        shorter_col = int(rng.integers(1, w - 1))
        clear_rows = {winner_row}
        clear_cols = {shorter_col}
        expected = {
            ("row", winner_row, w - 2),
            ("col", shorter_col, h - 2),
        }
        _paint_row_endpoints(grid, winner_row, rng)
        _paint_col_endpoints(grid, shorter_col, rng)
    else:
        winner_col = int(rng.integers(1, w - 1))
        shorter_row = int(rng.integers(1, h - 1))
        clear_rows = {shorter_row}
        clear_cols = {winner_col}
        expected = {
            ("row", shorter_row, w - 2),
            ("col", winner_col, h - 2),
        }
        _paint_row_endpoints(grid, shorter_row, rng)
        _paint_col_endpoints(grid, winner_col, rng)

    available_rows = [r for r in range(1, h - 1) if r not in clear_rows]
    available_cols = [c for c in range(1, w - 1) if c not in clear_cols]

    blocked_rows = _sample_without_replacement(rng, available_rows, 1, 3)
    blocked_cols = _sample_without_replacement(rng, available_cols, 1, 3)

    if not _add_blocked_rows(grid, blocked_rows, clear_cols, rng):
        return None
    if not _add_blocked_cols(grid, blocked_cols, clear_rows, rng):
        return None

    _add_extra_blockers(grid, clear_rows, clear_cols, rng)
    _ensure_both_input_colors(grid, rng)

    if set(_find_corridors(grid)) != expected:
        return None

    initial_corridors = _find_corridors(grid)
    longest = max(length for _, _, length in initial_corridors)
    if sum(length == longest for _, _, length in initial_corridors) != 1:
        return None

    solved = main(grid)
    if _find_corridors(solved):
        return None

    new_green = (solved == GREEN) & (grid != GREEN)
    if int(new_green.sum()) != longest:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-greedy input")
