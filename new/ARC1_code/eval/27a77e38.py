import numpy as np


def _separator_row(g: np.ndarray) -> int:
    height = int(g.shape[0])
    candidates: list[int] = []

    for row in range(1, height - 1):
        if not np.all(g[row, :] == g[row, 0]):
            continue

        lower = g[row + 1 :, :]
        bg = int(lower[0, 0])
        if not np.all(lower == bg):
            continue

        sep = int(g[row, 0])
        if sep == bg:
            continue
        if row != height - row - 1:
            continue

        upper = g[:row, :]
        if np.any(upper == bg) or np.any(upper == sep):
            continue

        candidates.append(row)

    assert len(candidates) == 1, "expected one centered separator above a uniform lower region"
    return int(candidates[0])


def _unique_most_frequent_color(region: np.ndarray) -> int:
    colors, counts = np.unique(region, return_counts=True)
    assert colors.size > 0, "source region must contain at least one color"

    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "source region must have a unique most frequent color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Write the majority color from the upper pattern into the bottom-center cell below the divider.

    Input structure
    - The grid is split horizontally by one full-width separator row.
    - The separator is centered: the colored source region above it and the empty lower region below it have equal height.
    - The lower region is one uniform background color, and the upper region uses neither that background nor the separator color.
    Rule
    - Count colors in the upper source region.
    - Find the unique most frequent color.
    - Copy the input grid and paint that color at the bottom row's center column.
    Key insight
    - The lower half is a result area; its bottom-center cell records the visible majority color of the pattern above the separator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1, "expected an odd width with a center column"

    sep_row = _separator_row(g)
    source = g[:sep_row, :]
    color = _unique_most_frequent_color(source)

    out = g.copy()
    out[height - 1, width // 2] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas/layout bounds.
K_MIN, K_MAX_EXCLUSIVE = 1, 5

# Fixed task colors and source palette.
BG_COLOR, SEP_COLOR = 0, 5
SOURCE_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

# Source color-count bounds.
SOURCE_COLOR_MIN, SOURCE_COLOR_MAX = 2, 8

# Oracle validation retries.
MAX_GENERATION_ATTEMPTS = 300


def _force_unique_majority(counts):
    """Pick a max count and transfer from tied maxima until it is unique."""
    tied_max = np.flatnonzero(counts == counts.max())
    majority_index = int(np.random.choice(tied_max))
    while np.count_nonzero(counts == counts.max()) > 1:
        tied_others = [
            int(i) for i in np.flatnonzero(counts == counts.max()) if i != majority_index
        ]
        counts[tied_others[0]] -= 1
        counts[majority_index] += 1
    return counts


def _gen_source_region(k, width):
    total = int(k * width)
    max_colors = min(SOURCE_COLOR_MAX, total - 1)
    n_colors = int(np.random.randint(SOURCE_COLOR_MIN, max_colors + 1))
    colors = np.random.choice(SOURCE_PALETTE, size=n_colors, replace=False)
    counts = np.ones(n_colors, dtype=int)
    counts += np.random.multinomial(total - n_colors, np.ones(n_colors) / n_colors)
    counts = _force_unique_majority(counts)

    cells = np.repeat(colors.astype(int), counts.astype(int))
    np.random.shuffle(cells)
    return cells.reshape(k, width)


def _gen_attempt():
    k = int(np.random.randint(K_MIN, K_MAX_EXCLUSIVE))
    side = 2 * k + 1

    grid = np.full((side, side), BG_COLOR, dtype=int)
    source = _gen_source_region(k, side)
    grid[:k, :] = source
    grid[k, :] = SEP_COLOR

    colors, counts = np.unique(source, return_counts=True)
    expected = grid.copy()
    expected[side - 1, k] = int(colors[np.argmax(counts)])

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 27a77e38 input")
