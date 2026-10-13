import numpy as np

from arc_utils import Color
import random
from itertools import combinations


def _palette_in_appearance_order(g: np.ndarray, bg: int) -> tuple[int, ...]:
    palette: list[int] = []
    for value in g.ravel():
        color = int(value)
        if color != bg and color not in palette:
            palette.append(color)
    return tuple(palette)


def _validate_partial_latin_grid(g: np.ndarray, palette: tuple[int, ...], bg: int) -> None:
    for row in g:
        seen = [int(value) for value in row if int(value) != bg]
        assert len(seen) == len(set(seen)), "a row may not repeat a non-blank color"
    for col in g.T:
        seen = [int(value) for value in col if int(value) != bg]
        assert len(seen) == len(set(seen)), "a column may not repeat a non-blank color"


def _complete_latin_grid(g: np.ndarray, palette: tuple[int, ...], bg: int) -> np.ndarray:
    out = g.copy()
    palette_set = set(palette)
    solutions: list[np.ndarray] = []

    def domain(r: int, c: int) -> list[int]:
        row_used = {int(value) for value in out[r, :] if int(value) != bg}
        col_used = {int(value) for value in out[:, c] if int(value) != bg}
        return [color for color in palette if color not in row_used and color not in col_used]

    def search() -> None:
        if len(solutions) > 1:
            return

        blanks = np.argwhere(out == bg)
        if len(blanks) == 0:
            assert all({int(value) for value in row} == palette_set for row in out), "every row must use the palette"
            assert all({int(value) for value in col} == palette_set for col in out.T), "every column must use the palette"
            solutions.append(out.copy())
            return

        options: list[tuple[int, int, int, list[int]]] = []
        for r_raw, c_raw in blanks:
            r, c = int(r_raw), int(c_raw)
            choices = domain(r, c)
            if not choices:
                return
            options.append((len(choices), r, c, choices))

        options.sort(key=lambda item: (item[0], item[1], item[2]))
        _, r, c, choices = options[0]
        for color in choices:
            out[r, c] = color
            search()
            out[r, c] = bg

    search()
    assert len(solutions) == 1, "the partial Latin grid must have exactly one completion"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black blanks so the grid becomes a Latin-square table over the foreground colors.

    Input structure:
    - The grid is square, and black pixels are blanks in an otherwise partially filled color table.
    - The non-black colors form the table palette, with one palette color required in each row and column.
    - Existing non-black cells already obey the no-duplicate row and column constraints.

    Rule:
    - Keep every non-black input cell fixed.
    - Replace each black cell with the only color assignment that makes every row and every column contain the full palette exactly once.
    - Assert that the row/column constraints determine a unique completion.

    Key insight:
    The missing pixels are constrained by Latin-square row and column uniqueness, not by local neighbor copying or color frequency.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BACKGROUND

    h, w = g.shape
    assert h == w, "Latin table must be square"
    assert np.any(g == bg), "input must contain blank cells"

    palette = _palette_in_appearance_order(g, bg)
    assert len(palette) == h, "palette size must equal the square side length"
    _validate_partial_latin_grid(g, palette, bg)

    return _complete_latin_grid(g, palette, bg)

# --------------- generator (o5x) ---------------

# Latin square structure.
GRID_SIZE = 4
GRID_INDICES = tuple(range(GRID_SIZE))
PALETTE = (1, 2, 3, 4)
LATIN_USE_XOR_CHOICES = (False, True)
BACKGROUND = 0
REQUIRED_INPUT_COLORS = frozenset((BACKGROUND, *PALETTE))

# Blank-count and row/column-margin patterns.
BLANK_COUNTS = (5, 6)
ROW_PATTERNS_BY_BLANK_COUNT = {
    5: ((2, 1, 1, 1),),
    6: ((2, 2, 1, 1), (3, 1, 1, 1)),
}
COL_PATTERNS_BY_BLANK_COUNT = {
    5: ((2, 1, 1, 1), (2, 2, 1, 0), (3, 1, 1, 0)),
    6: ((2, 2, 1, 1), (3, 1, 1, 1), (3, 2, 1, 0), (2, 2, 2, 0)),
}

# Rejection sampling budget.
GENERATION_ATTEMPTS = 300


def _gen_latin_square() -> np.ndarray:
    palette = random.sample(PALETTE, GRID_SIZE)
    row_perm = random.sample(GRID_INDICES, GRID_SIZE)
    col_perm = random.sample(GRID_INDICES, GRID_SIZE)
    if random.choice(LATIN_USE_XOR_CHOICES):
        values = np.bitwise_xor.outer(row_perm, col_perm)
    else:
        values = (np.array(row_perm)[:, None] + np.array(col_perm)[None, :]) % GRID_SIZE
    return np.asarray(palette, dtype=int)[values]


def _gen_shuffled_counts(patterns: tuple[tuple[int, ...], ...]) -> list[int]:
    counts = list(random.choice(patterns))
    random.shuffle(counts)
    return counts


def _margins_feasible(
    row_order: list[int],
    row_counts: list[int],
    col_rem: list[int],
    next_i: int,
) -> bool:
    """Reject states that cannot satisfy the still-unassigned row margins."""
    rem_rows = row_order[next_i:]
    if sum(col_rem) != sum(row_counts[r] for r in rem_rows):
        return False
    if col_rem and max(col_rem) > len(rem_rows):
        return False
    positive_cols = sum(count > 0 for count in col_rem)
    return all(row_counts[r] <= positive_cols for r in rem_rows)


def _gen_mask_from_margins(row_counts: list[int], col_counts: list[int]) -> np.ndarray | None:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    col_rem = col_counts[:]
    row_order = list(GRID_INDICES)
    random.shuffle(row_order)
    row_order.sort(key=lambda r: row_counts[r], reverse=True)

    def search(i: int) -> bool:
        if i == len(row_order):
            return all(count == 0 for count in col_rem)

        r = row_order[i]
        choices = [
            cols for cols in combinations(GRID_INDICES, row_counts[r])
            if all(col_rem[c] > 0 for c in cols)
        ]
        random.shuffle(choices)
        for cols in choices:
            for c in cols:
                col_rem[c] -= 1
            mask[r, list(cols)] = True
            if _margins_feasible(row_order, row_counts, col_rem, i + 1) and search(i + 1):
                return True
            mask[r, list(cols)] = False
            for c in cols:
                col_rem[c] += 1
        return False

    if search(0):
        return mask
    return None


def _gen_blank_mask(n_blanks: int) -> np.ndarray | None:
    row_counts = _gen_shuffled_counts(ROW_PATTERNS_BY_BLANK_COUNT[n_blanks])
    col_counts = _gen_shuffled_counts(COL_PATTERNS_BY_BLANK_COUNT[n_blanks])
    return _gen_mask_from_margins(row_counts, col_counts)


def _solver_recovers_output(inp: np.ndarray, out: np.ndarray) -> bool:
    """Return whether the task solver uniquely reconstructs the sampled output."""
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    out = _gen_latin_square()
    mask = _gen_blank_mask(random.choice(BLANK_COUNTS))
    if mask is None:
        return None

    inp = out.copy()
    inp[mask] = BACKGROUND
    if set(np.unique(inp)) != REQUIRED_INPUT_COLORS:
        return None
    if not _solver_recovers_output(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely completable Latin grid")
