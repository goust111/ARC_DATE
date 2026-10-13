import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Select the color whose count grows the most below the gray divider and output it as a 2x2 square.

    Input structure
    A mostly background grid contains one full-width gray horizontal divider and two non-background,
    non-gray colored marker masks above and below that divider.

    Rule
    For each marker color, count its pixels above the divider and below the divider. Choose the
    color with the largest value of below_count - above_count, and return a 2x2 grid filled with
    that color.

    Key insight
    The gray bar splits the scene into before/after regions; the answer is the color with the
    greater net increase from the upper region to the lower region.
    """
    g = np.asarray(grid, dtype=int)

    bg = get_bg(g)
    divider_rows = np.where(np.all(g == Color.GREY, axis=1))[0]
    assert len(divider_rows) == 1
    divider = int(divider_rows[0])

    marker_colors = colors_present(g, exclude={bg, Color.GREY})
    assert len(marker_colors) == 2

    scores = []
    for color in marker_colors:
        above = int(np.count_nonzero(g[:divider] == color))
        below = int(np.count_nonzero(g[divider + 1 :] == color))
        scores.append((below - above, color))

    best_score = max(score for score, _ in scores)
    winners = [color for score, color in scores if score == best_score]
    assert len(winners) == 1

    return np.full((2, 2), winners[0], dtype=int)

# --------------- generator (o5x) ---------------

# Canvas and divider geometry.
REGION_HEIGHT_CHOICES, GRID_WIDTH = (4, 5), 9
BACKGROUND_COLOR, DIVIDER_COLOR = 0, 5
# Marker colors and column placement.
MARKER_COUNT, MIN_COLUMN_SEPARATION = 2, 2
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
# Bar growth, output size, and retry limits.
MIN_ABOVE_HEIGHT, MIN_GROWTH = 1, 1
OUTPUT_SIZE, MAX_GENERATION_ATTEMPTS = 2, 300


def _gen_attempt() -> np.ndarray | None:
    region_h = int(np.random.choice(REGION_HEIGHT_CHOICES))
    height = 2 * region_h + 1
    grid = np.full((height, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    grid[region_h, :] = DIVIDER_COLOR

    colors = [int(c) for c in np.random.choice(MARKER_PALETTE, size=MARKER_COUNT, replace=False)]
    column_pairs = [(a, b) for a in range(GRID_WIDTH) for b in range(GRID_WIDTH) if abs(a - b) >= MIN_COLUMN_SEPARATION]
    columns = column_pairs[int(np.random.randint(len(column_pairs)))]

    above_heights = [int(np.random.randint(MIN_ABOVE_HEIGHT, region_h)) for _ in range(MARKER_COUNT)]
    below_heights = [int(np.random.randint(above_h + MIN_GROWTH, region_h + 1)) for above_h in above_heights]
    diffs = [below_h - above_h for above_h, below_h in zip(above_heights, below_heights)]
    if diffs[0] == diffs[1]:
        return None

    for color, col, above_h, below_h in zip(colors, columns, above_heights, below_heights):
        grid[:above_h, col] = color
        grid[region_h + 1 : region_h + 1 + below_h, col] = color

    winner = colors[int(np.argmax(diffs))]
    expected = np.full((OUTPUT_SIZE, OUTPUT_SIZE), winner, dtype=int)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _gen_i in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8597cfd7 input")
