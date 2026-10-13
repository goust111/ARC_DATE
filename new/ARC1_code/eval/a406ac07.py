import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each interior cell whose row label and column label have the same color.

    Input structure
    - The background fills the interior rectangle excluding the bottom row and rightmost column.
    - The right edge gives a non-background color label for each interior row.
    - The bottom edge gives a non-background color label for each interior column.

    Rule
    - Keep the bottom row and rightmost column unchanged.
    - For every interior cell, compare its row's right-edge label with its column's bottom-edge label.
    - Recolor the cell to that label color exactly when the two labels match; otherwise keep it background.

    Key insight
    - The border colors are row and column headers, so the output is the equality table of those two header sequences.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    assert height >= 2 and width >= 2, "expected row and column labels around a nonempty interior"

    bg = get_bg(g)
    row_labels = g[:-1, -1]
    col_labels = g[-1, :-1]

    assert np.all(g[:-1, :-1] == bg), "expected background-filled interior"
    assert np.all(row_labels != bg), "expected right-edge row labels"
    assert np.all(col_labels != bg), "expected bottom-edge column labels"
    assert int(g[-1, -1]) != bg, "expected non-background bottom-right border label"

    out = g.copy()
    same_label = row_labels[:, None] == col_labels[None, :]
    out[:-1, :-1] = np.where(same_label, row_labels[:, None], bg)
    return out

# --------------- generator (o5x) ---------------
# Canvas bounds.
GRID_SIZE_MIN = 7
GRID_SIZE_MAX_EXCLUSIVE = 13

# Label color and run constraints.
BACKGROUND = 0
COLOR_POOL = np.arange(1, 10)
COLOR_COUNT_OPTIONS = (4, 5)
RUN_LENGTH_MIN = 1
RUN_LENGTH_MAX = 3

# Rejection loop.
GENERATION_RETRIES = 300


def _gen_partitions(total, parts):
    choices = []

    def rec(remaining, slots, prefix):
        if slots == 0:
            if remaining == 0:
                choices.append(tuple(prefix))
            return
        low = max(RUN_LENGTH_MIN, remaining - RUN_LENGTH_MAX * (slots - 1))
        high = min(RUN_LENGTH_MAX, remaining - RUN_LENGTH_MIN * (slots - 1))
        for value in range(low, high + 1):
            prefix.append(value)
            rec(remaining - value, slots - 1, prefix)
            prefix.pop()

    rec(total, parts, [])
    return choices


def _gen_labels(colors, lengths):
    return np.repeat(np.asarray(colors, dtype=int), np.asarray(lengths, dtype=int))


def _gen_expected_output(inp):
    out = inp.copy()
    row_labels = inp[:-1, -1]
    col_labels = inp[-1, :-1]
    same_label = row_labels[:, None] == col_labels[None, :]
    out[:-1, :-1] = np.where(same_label, row_labels[:, None], BACKGROUND)
    return out


def _gen_attempt():
    n = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE))
    total = n - 1
    feasible_k = [
        k for k in COLOR_COUNT_OPTIONS
        if RUN_LENGTH_MIN * k <= total <= RUN_LENGTH_MAX * k
    ]
    k = int(feasible_k[np.random.randint(len(feasible_k))])

    colors = np.random.choice(COLOR_POOL, size=k, replace=False)
    row_options = _gen_partitions(total, k)
    col_options = _gen_partitions(total, k)
    row_lengths = row_options[int(np.random.randint(len(row_options)))]
    col_lengths = col_options[int(np.random.randint(len(col_options)))]

    row_labels = _gen_labels(colors, row_lengths)
    col_labels = _gen_labels(colors, col_lengths)

    inp = np.full((n, n), BACKGROUND, dtype=int)
    inp[:-1, -1] = row_labels
    inp[-1, :-1] = col_labels
    inp[-1, -1] = int(colors[-1])
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, _gen_expected_output(inp)):
            return inp
    raise RuntimeError('failed to generate valid input')
