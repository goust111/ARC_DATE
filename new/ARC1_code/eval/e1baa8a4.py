import numpy as np


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            keep.append(i)
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse a grid made by stretching a smaller color matrix into duplicate row and column runs.

    Input structure
    The input is a 2-D color grid with no empty background; each answer row has been expanded into
    one or more identical adjacent rows, and each answer column into one or more identical adjacent
    columns.

    Rule
    Keep the first row from each maximal run of identical neighboring rows, and keep the first
    column from each maximal run of identical neighboring columns.

    Key insight
    The visible colors already form the answer matrix; the only operation is removing redundant
    row and column repetitions from the expanded blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] > 0 and g.shape[1] > 0

    row_starts = _run_start_indices(g)
    col_starts = _run_start_indices(g.T)
    assert len(row_starts) < g.shape[0]
    assert len(col_starts) < g.shape[1]

    out = g[np.ix_(row_starts, col_starts)].copy()

    assert out.shape[0] == len(row_starts)
    assert out.shape[1] == len(col_starts)
    return out

# --------------- generator (o5x) ---------------

# Base/output and expanded input dimensions.
BASE_ROWS_MIN, BASE_ROWS_MAX = 2, 5
BASE_COLS_MIN, BASE_COLS_MAX = 2, 4
INPUT_HEIGHT_MIN, INPUT_HEIGHT_MAX = 12, 15
INPUT_WIDTH_MIN, INPUT_WIDTH_MAX = 14, 15
# Run lengths used to stretch each output row/column.
RUN_LENGTH_MIN, RUN_LENGTH_MAX = 2, 11
# Palette, distinct color bounds, and rejection budget.
COLOR_MIN, COLOR_MAX = 1, 9
MIN_DISTINCT_COLORS, MAX_DISTINCT_COLORS = 4, 9
GENERATION_ATTEMPTS = 300


def _gen_valid_partitions(total: int, parts: int) -> list[tuple[int, ...]]:
    out = []

    def rec(prefix: list[int], remaining: int) -> None:
        slots = parts - len(prefix)
        if slots == 0:
            if remaining == 0 and len(set(prefix)) > 1:
                out.append(tuple(prefix))
            return

        lo = max(RUN_LENGTH_MIN, remaining - RUN_LENGTH_MAX * (slots - 1))
        hi = min(RUN_LENGTH_MAX, remaining - RUN_LENGTH_MIN * (slots - 1))
        for value in range(lo, hi + 1):
            prefix.append(value)
            rec(prefix, remaining - value)
            prefix.pop()

    rec([], total)
    return out


def _gen_choose_partition(rng: np.random.Generator, total: int, parts: int) -> np.ndarray | None:
    choices = _gen_valid_partitions(total, parts)
    if not choices:
        return None
    return np.array(choices[int(rng.integers(len(choices)))], dtype=int)


def _gen_base_matrix(rng: np.random.Generator, rows: int, cols: int, num_colors: int) -> np.ndarray:
    palette = rng.choice(np.arange(COLOR_MIN, COLOR_MAX + 1), size=num_colors, replace=False)
    rng.shuffle(palette)
    base = np.empty((rows, cols), dtype=int)

    anchor_positions = [(0, 0)]
    anchor_positions.extend((0, c) for c in range(1, cols))
    anchor_positions.extend((r, 0) for r in range(1, rows))

    anchor_count = len(anchor_positions)
    anchor_values = list(palette[: min(num_colors, anchor_count)])
    while len(anchor_values) < anchor_count:
        r, c = anchor_positions[len(anchor_values)]
        forbidden = None
        if r == 0 and c > 0:
            forbidden = anchor_values[len(anchor_values) - 1]
        elif c == 0 and r > 0:
            forbidden = base[r - 1, 0]
        options = [int(color) for color in palette if int(color) != forbidden]
        anchor_values.append(options[int(rng.integers(len(options)))])

    for (r, c), color in zip(anchor_positions, anchor_values):
        base[r, c] = int(color)

    used = set(int(color) for color in anchor_values)
    missing = [int(color) for color in palette if int(color) not in used]
    interior_positions = [(r, c) for r in range(1, rows) for c in range(1, cols)]
    rng.shuffle(interior_positions)

    interior_values = missing[:]
    interior_values.extend(int(palette[int(rng.integers(num_colors))]) for _ in range(len(interior_positions) - len(missing)))
    rng.shuffle(interior_values)

    for (r, c), color in zip(interior_positions, interior_values):
        base[r, c] = color

    return base


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    valid_shapes = [
        (r, c)
        for r in range(BASE_ROWS_MIN, BASE_ROWS_MAX + 1)
        for c in range(BASE_COLS_MIN, BASE_COLS_MAX + 1)
        if max(MIN_DISTINCT_COLORS, (r * c) // 2) <= min(MAX_DISTINCT_COLORS, r * c)
    ]
    rows, cols = valid_shapes[int(rng.integers(len(valid_shapes)))]
    height = int(rng.integers(INPUT_HEIGHT_MIN, INPUT_HEIGHT_MAX + 1))
    width = int(rng.integers(INPUT_WIDTH_MIN, INPUT_WIDTH_MAX + 1))
    row_runs = _gen_choose_partition(rng, height, rows)
    col_runs = _gen_choose_partition(rng, width, cols)
    if row_runs is None or col_runs is None:
        return None

    cells = rows * cols
    min_colors = max(MIN_DISTINCT_COLORS, cells // 2)
    max_colors = min(MAX_DISTINCT_COLORS, cells)
    num_colors = int(rng.integers(min_colors, max_colors + 1))
    out = _gen_base_matrix(rng, rows, cols, num_colors)

    inp = np.repeat(np.repeat(out, row_runs, axis=0), col_runs, axis=1)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
