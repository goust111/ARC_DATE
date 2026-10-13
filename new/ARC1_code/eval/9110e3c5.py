from collections import Counter

import numpy as np

from arc_utils import Color
import random


_MARKER_TO_STENCIL = {
    Color.BLUE: ((0, 0, 1), (1, 1, 0), (0, 1, 0)),
    Color.RED: ((0, 0, 0), (1, 1, 1), (0, 0, 0)),
    Color.GREEN: ((0, 1, 1), (0, 1, 0), (0, 1, 0)),
}


def _dominant_foreground_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    fg_counts = {color: n for color, n in counts.items() if color != Color.BLACK}
    assert fg_counts, "expected at least one foreground color"

    ranked = sorted(fg_counts.items(), key=lambda item: item[1], reverse=True)
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], (
        "expected a unique dominant foreground color"
    )
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the unique dominant non-black color into a fixed 3x3 teal stencil.

    Input structure
    - The input is a 7x7 black-background grid with noisy foreground pixels.
    - Exactly one non-black color is the unique most frequent foreground color.
    - The dominant foreground color is blue, red, or green; other foreground colors are distractors.

    Rule
    - Use the task codebook: blue -> the lower-left hooked stencil, red -> the middle row, green -> the upper-right hooked stencil.
    - Return a 3x3 black grid with the selected stencil painted teal.

    Key insight
    - The clutter positions vary; the stable signal is the dominant foreground color acting as a compact class symbol.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (7, 7), "expected the task's 7x7 input grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= set(Color.ALL_COLORS), "expected ARC palette colors"
    assert Color.BLACK in colors, "expected black background pixels"

    marker_color = _dominant_foreground_color(g)
    assert marker_color in _MARKER_TO_STENCIL, "encountered an unknown dominant marker color"

    mask = np.array(_MARKER_TO_STENCIL[marker_color], dtype=bool)
    out = np.zeros((3, 3), dtype=int)
    out[mask] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------
# Canvas geometry.
GEN_GRID_SHAPE = (7, 7)
GEN_N_CELLS = GEN_GRID_SHAPE[0] * GEN_GRID_SHAPE[1]
# Foreground population and dominance.
GEN_MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
GEN_DOMINANT_PIXELS_RANGE = (12, 21)
GEN_TOTAL_FG_PIXELS_RANGE = (18, 29)
GEN_DOMINANT_FRACTION_RANGE = (0.61, 0.78)
# Distractor allocation and retries.
GEN_DISTRACTOR_COLOR_COUNT_RANGE = (4, 7)
GEN_DISTRACTOR_TOTAL_PIXELS_RANGE = (5, 10)
GEN_DISTRACTOR_PIXELS_PER_COLOR_RANGE = (1, 4)
GEN_MAX_ATTEMPTS = 300

def _gen_weighted_choice(items, weights):
    """Sample one item from weights aligned with items."""
    total = sum(weights)
    pick = random.uniform(0, total)
    for item, weight in zip(items, weights):
        pick -= weight
        if pick <= 0:
            return item
    return items[-1]

def _gen_distractor_counts(total_pixels, n_colors):
    """Sample an ordered per-color split uniformly from feasible splits."""
    choices = []
    per_min, per_max = GEN_DISTRACTOR_PIXELS_PER_COLOR_RANGE
    def build(prefix, remaining, slots_left):
        if slots_left == 0:
            if remaining == 0:
                choices.append(tuple(prefix))
            return
        lo = max(per_min, remaining - per_max * (slots_left - 1))
        hi = min(per_max, remaining - per_min * (slots_left - 1))
        for count in range(lo, hi + 1):
            build(prefix + [count], remaining - count, slots_left - 1)

    build([], total_pixels, n_colors)
    assert choices, "expected a valid distractor count split"
    return list(random.choice(choices))

def _gen_pixel_values():
    """Sample the marker color and shuffled foreground values for one grid."""
    marker_color = random.choice(GEN_MARKER_COLORS)
    dominant_pixels = random.randint(*GEN_DOMINANT_PIXELS_RANGE)
    n_distractor_colors = random.randint(*GEN_DISTRACTOR_COLOR_COUNT_RANGE)
    total_min, total_max = GEN_DISTRACTOR_TOTAL_PIXELS_RANGE
    fg_min, fg_max = GEN_TOTAL_FG_PIXELS_RANGE
    frac_min, frac_max = GEN_DOMINANT_FRACTION_RANGE
    per_min, per_max = GEN_DISTRACTOR_PIXELS_PER_COLOR_RANGE
    distractor_totals = [
        total
        for total in range(max(total_min, per_min * n_distractor_colors), total_max + 1)
        if total <= per_max * n_distractor_colors
        and fg_min <= dominant_pixels + total <= fg_max
        and frac_min <= dominant_pixels / (dominant_pixels + total) <= frac_max
    ]
    assert distractor_totals, "expected feasible distractor total"
    weights = [total - n_distractor_colors + 1 for total in distractor_totals]
    distractor_pixels = _gen_weighted_choice(distractor_totals, weights)
    distractor_palette = [color for color in Color.NOT_BLACK if color != marker_color]
    distractor_colors = random.sample(distractor_palette, n_distractor_colors)
    distractor_counts = _gen_distractor_counts(distractor_pixels, n_distractor_colors)
    values = [marker_color] * dominant_pixels
    for color, count in zip(distractor_colors, distractor_counts):
        values.extend([color] * count)
    random.shuffle(values)
    return marker_color, values

def _scatter_values(values):
    """Scatter values into distinct random cells of the fixed canvas."""
    grid = np.zeros(GEN_GRID_SHAPE, dtype=int)
    flat_indices = random.sample(range(GEN_N_CELLS), len(values))
    grid.reshape(-1)[flat_indices] = values
    return grid

def _gen_attempt():
    """Build one candidate grid and verify it against the solver oracle."""
    marker_color, values = _gen_pixel_values()
    grid = _scatter_values(values)
    mask = np.array(_MARKER_TO_STENCIL[marker_color], dtype=bool)
    expected = np.zeros(mask.shape, dtype=int)
    expected[mask] = Color.TEAL
    if not np.array_equal(main(grid), expected):
        return None
    return grid

def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
