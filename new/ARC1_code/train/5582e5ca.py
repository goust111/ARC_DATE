import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the whole grid with the unique most frequent input color.

    Input structure
    - The input is a nonempty 2D grid.
    - Exactly one color appears more often than every other color.

    Rule
    - Find the unique most frequent color in the input.
    - Replace every pixel with that color, preserving the input shape.

    Key insight
    - The scattered colors are reduced to a single consensus color determined only by frequency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a nonempty grid"

    colors, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    dominant = colors[counts == max_count]
    assert dominant.size == 1, "expected a unique most frequent color"

    return np.full(g.shape, int(dominant[0]), dtype=int)

# --------------- generator (o5x) ---------------
# Sampling controls.
GRID_SHAPE = (3, 3)
COLOR_POOL = np.arange(10)
PALETTE_SIZE_CHOICES = [5, 6]
DOMINANT_COUNT = 3
REPEATED_OTHER_COUNTS = {5: 2, 6: 1}
MAX_ATTEMPTS = 300

def _sample_palette():
    num_colors = int(np.random.choice(PALETTE_SIZE_CHOICES))
    colors = np.random.choice(COLOR_POOL, size=num_colors, replace=False)
    return num_colors, int(colors[0]), [int(c) for c in colors[1:]]

def _sample_repeated_others(num_colors, other_colors):
    repeat_count = REPEATED_OTHER_COUNTS[num_colors]
    if repeat_count == 1:
        return {int(np.random.choice(other_colors))}
    return set(np.random.choice(other_colors, size=repeat_count, replace=False).tolist())

def _gen_attempt() -> np.ndarray:
    num_colors, dominant, other_colors = _sample_palette()
    repeated_others = _sample_repeated_others(num_colors, other_colors)
    cells = [dominant] * DOMINANT_COUNT
    for color in other_colors:
        cells.extend([color] * (2 if color in repeated_others else 1))
    shuffled = np.array(cells, dtype=int)
    np.random.shuffle(shuffled)
    return shuffled.reshape(GRID_SHAPE)

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        colors, counts = np.unique(inp, return_counts=True)
        dominant = int(colors[np.argmax(counts)])
        out = np.full(inp.shape, dominant, dtype=int)
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input")
