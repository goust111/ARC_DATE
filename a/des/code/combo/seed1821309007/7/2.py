import numpy as np


BG = 8
COLOR_POOL = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
MAX_FOREGROUND_PER_ROW = 5
HEIGHT_RANGE = (6, 16)
WIDTH_RANGE = (9, 19)
MAX_GENERATION_ATTEMPTS = 300


def _target_columns(width: int, count: int) -> np.ndarray:
    """Return centered, evenly spaced target columns for one row."""
    if count == 0:
        return np.empty(0, dtype=int)
    if count == 1:
        return np.array([width // 2], dtype=int)

    cols = np.rint(np.linspace(0, width - 1, count + 2)[1:-1]).astype(int)
    assert np.all(cols[1:] > cols[:-1]), "target columns must be distinct"
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """Repack each row's foreground pixels into a centered even distribution.

    Input structure
    - The background is azure.
    - Each row contains 0-5 foreground pixels.
    - Pixels in one row may have different colors.

    Rule
    - For each row, keep only the left-to-right foreground color order.
    - Clear the row to azure.
    - Place the same colors at evenly spaced columns centered within the row.
    - Rows with no foreground stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    height, width = map(int, g.shape)
    out = np.full((height, width), BG, dtype=int)

    for r in range(height):
        row = g[r]
        fg_cols = np.flatnonzero(row != BG)
        count = int(fg_cols.size)
        assert count <= MAX_FOREGROUND_PER_ROW, "too many foreground pixels in one row"
        if count == 0:
            continue

        colors = row[fg_cols]
        target_cols = _target_columns(width, count)
        out[r, target_cols] = colors

    return out


def _sample_row(width: int, rng: np.random.Generator) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    count = int(rng.integers(0, MAX_FOREGROUND_PER_ROW + 1))
    if count == 0:
        return row

    cols = np.sort(rng.choice(width, size=count, replace=False))
    colors = rng.choice(COLOR_POOL, size=count, replace=True).astype(int)
    row[cols] = colors
    return row


def _is_degenerate(grid: np.ndarray) -> bool:
    out = main(grid)
    if np.array_equal(grid, out):
        return True

    counts = [int(np.count_nonzero(row != BG)) for row in grid]
    if not any(count == 0 for count in counts):
        return True
    if not any(count >= 2 for count in counts):
        return True

    present = {count for count in counts if count > 0}
    return len(present) < 2


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))

    grid = np.full((height, width), BG, dtype=int)
    for r in range(height):
        grid[r] = _sample_row(width, rng)

    if _is_degenerate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
