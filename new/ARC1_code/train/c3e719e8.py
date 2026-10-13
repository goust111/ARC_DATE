import numpy as np
import random


def _unique_most_frequent_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique most frequent color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the whole input pattern into output block slots marked by its unique most frequent color.

    Input structure
    - The input is a 3x3 grid of non-black colors.
    - Exactly one color appears more often than every other color.

    Rule
    - Treat the input as both a source sprite and a 3x3 placement mask.
    - Create a 9x9 black output divided into 3x3 block slots.
    - For each input cell whose color is the unique most frequent color, copy
      the entire input grid into the corresponding output block slot.

    Key insight
    - The dominant color does not get expanded by itself; its positions select
      where full copies of the original 3x3 pattern are stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert (h, w) == (3, 3), "expected a 3x3 input grid"
    assert not np.any(g == 0), "input pattern should not contain black"

    marker_color = _unique_most_frequent_color(g)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == marker_color:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out

# --------------- generator (o5x) ---------------

# Grid, palette, and retry constraints.
GRID_SIZE = 3
GRID_CELLS = GRID_SIZE * GRID_SIZE
BLACK_COLOR = 0
NONBLACK_COLORS = range(1, 10)
COLOR_COUNT_OPTIONS = (3, 4)
DOMINANT_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 300


def _gen_feasible_nondominant_counts(total: int, parts: int, max_each: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if 1 <= total <= max_each else []
    counts = []
    for first in range(1, max_each + 1):
        for rest in _gen_feasible_nondominant_counts(total - first, parts - 1, max_each):
            counts.append((first,) + rest)
    return counts


def _gen_expected_output(grid: np.ndarray, dominant: int) -> np.ndarray:
    out = np.zeros((GRID_CELLS, GRID_CELLS), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if int(grid[r, c]) == dominant:
                out[r * GRID_SIZE : (r + 1) * GRID_SIZE, c * GRID_SIZE : (c + 1) * GRID_SIZE] = grid
    return out


def _gen_attempt() -> np.ndarray | None:
    dominant_count = random.randint(*DOMINANT_COUNT_RANGE)
    color_count_options = [k for k in COLOR_COUNT_OPTIONS if GRID_CELLS - dominant_count <= (k - 1) * (dominant_count - 1)]
    color_count = random.choice(color_count_options)
    colors = random.sample(NONBLACK_COLORS, color_count)
    dominant = random.choice(colors)
    other_colors = [c for c in colors if c != dominant]

    compositions = _gen_feasible_nondominant_counts(
        GRID_CELLS - dominant_count, len(other_colors), dominant_count - 1
    )
    if not compositions:
        return None
    other_counts = random.choice(compositions)

    cells = [dominant] * dominant_count
    for color, count in zip(other_colors, other_counts):
        cells.extend([color] * count)
    random.shuffle(cells)

    grid = np.array(cells, dtype=int).reshape(GRID_SIZE, GRID_SIZE)
    values, counts = np.unique(grid, return_counts=True)
    if np.any(grid == BLACK_COLOR) or values.size != color_count or counts.max() != dominant_count:
        return None
    if np.count_nonzero(counts == dominant_count) != 1:
        return None
    if not np.array_equal(main(grid), _gen_expected_output(grid, dominant)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c3e719e8 input")
