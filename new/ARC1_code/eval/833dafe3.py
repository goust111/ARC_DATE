import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the whole grid into a 2x2 reflection pattern with the input in the lower-right quadrant.

    Input structure
    - The input is a non-empty rectangular 2D grid.
    - Every input pixel belongs to the pattern; there is no separate background object.

    Rule
    - Keep the input unchanged as the lower-right quadrant.
    - Put the left-right reflection in the lower-left quadrant.
    - Put the top-bottom reflection in the upper-right quadrant.
    - Put the reflection across both axes in the upper-left quadrant.

    Key insight
    - The output completes four-way mirror symmetry around the input's upper-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a non-empty grid"

    return np.block([[np.flipud(np.fliplr(g)), np.flipud(g)], [np.fliplr(g), g]])

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GEN_SIDE_MIN, GEN_SIDE_MAX = 3, 8
GEN_PALETTE, GEN_COLOR_COUNT = range(1, 10), 3
# Fill, asymmetry, and retry controls.
GEN_BLACK_FRAC_MIN, GEN_BLACK_FRAC_MAX = 0.33, 0.56
GEN_FORCED_COLOR_CELLS, GEN_EMPTY = 4, -1
GEN_MAX_ATTEMPTS = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    return np.block([[np.flipud(np.fliplr(grid)), np.flipud(grid)], [np.fliplr(grid), grid]])


def _gen_black_count(total: int) -> int:
    min_black = int(np.ceil(total * GEN_BLACK_FRAC_MIN))
    max_black = int(np.floor(total * GEN_BLACK_FRAC_MAX))
    return random.randint(min_black, min(total - GEN_FORCED_COLOR_CELLS, max_black))


def _gen_disjoint_mirror_pairs(n: int):
    lr_pairs = [((r, c), (r, n - 1 - c)) for r in range(n) for c in range(n // 2)]
    ud_pairs = [((r, c), (n - 1 - r, c)) for r in range(n // 2) for c in range(n)]

    random.shuffle(lr_pairs)
    random.shuffle(ud_pairs)
    for lr_pair in lr_pairs:
        lr_cells = set(lr_pair)
        for ud_pair in ud_pairs:
            if lr_cells.isdisjoint(ud_pair):
                return lr_pair, ud_pair
    return None, None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GEN_SIDE_MIN, GEN_SIDE_MAX)
    total = n * n
    colors = random.sample(GEN_PALETTE, GEN_COLOR_COUNT)
    black_count = _gen_black_count(total)
    colored_count = total - black_count

    lr_pair, ud_pair = _gen_disjoint_mirror_pairs(n)
    if lr_pair is None:
        return None

    grid = np.full((n, n), GEN_EMPTY, dtype=int)
    forced = {
        lr_pair[0]: colors[0],
        lr_pair[1]: colors[1],
        ud_pair[0]: colors[1],
        ud_pair[1]: colors[2],
    }
    for (r, c), value in forced.items():
        grid[r, c] = value

    remaining_cells = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == GEN_EMPTY]
    remaining_values = [0] * black_count
    remaining_values.extend(
        random.choice(colors) for _ in range(colored_count - GEN_FORCED_COLOR_CELLS)
    )
    random.shuffle(remaining_values)

    for (r, c), value in zip(remaining_cells, remaining_values):
        grid[r, c] = value

    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return None

    expected = _gen_expected_output(grid)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 833dafe3")
