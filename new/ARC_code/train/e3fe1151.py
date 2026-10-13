from collections import Counter

import numpy as np

from arc_utils import get_bg
import random


def _solid_separator_cross(g: np.ndarray, bg: int) -> tuple[int, int]:
    height, width = g.shape
    solid_rows = [r for r in range(height) if np.all(g[r, :] == bg)]
    solid_cols = [c for c in range(width) if np.all(g[:, c] == bg)]
    assert len(solid_rows) == 1, "expected exactly one solid background separator row"
    assert len(solid_cols) == 1, "expected exactly one solid background separator column"
    return solid_rows[0], solid_cols[0]


def _quadrants(cross_row: int, cross_col: int, height: int, width: int) -> list[tuple[slice, slice]]:
    top_h = cross_row
    bottom_h = height - cross_row - 1
    left_w = cross_col
    right_w = width - cross_col - 1
    assert top_h > 0 and left_w > 0, "separator must leave nonempty corner regions"
    assert top_h == bottom_h, "separator row must split equal-height regions"
    assert left_w == right_w, "separator column must split equal-width regions"

    return [
        (slice(0, cross_row), slice(0, cross_col)),
        (slice(0, cross_row), slice(cross_col + 1, width)),
        (slice(cross_row + 1, height), slice(0, cross_col)),
        (slice(cross_row + 1, height), slice(cross_col + 1, width)),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the one blank in each corner region so all four regions have the same color multiset.

    Input structure:
    - The dominant background color forms one solid separator row and one solid
      separator column.
    - The separator cross splits the grid into four equal rectangular corner regions.
    - Each corner region contains exactly one background-colored blank pixel.

    Rule:
    - Count the non-background colors in each corner region.
    - The completed region multiset is defined by the maximum observed count of
      each color across the four regions.
    - Replace each region's blank pixel with the single color needed to match that
      completed multiset.

    Key insight:
    - The separator color inside a corner region is a missing value marker; comparing
      the four regions reveals the shared multiset without using numeric color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    color_counts = Counter(map(int, g.reshape(-1)))
    assert all(
        color == bg or count < color_counts[bg] for color, count in color_counts.items()
    ), "background color must be uniquely dominant"

    cross_row, cross_col = _solid_separator_cross(g, bg)
    regions = _quadrants(cross_row, cross_col, height, width)
    region_shape = g[regions[0]].shape
    region_area = int(np.prod(region_shape))

    counts_by_region: list[Counter[int]] = []
    blank_positions: list[tuple[int, int]] = []
    for row_slice, col_slice in regions:
        region = g[row_slice, col_slice]
        assert region.shape == region_shape, "all corner regions must have the same shape"
        blanks = np.argwhere(region == bg)
        assert len(blanks) == 1, "each corner region must have exactly one blank"

        counts_by_region.append(Counter(map(int, region[region != bg].reshape(-1))))
        local_r, local_c = map(int, blanks[0])
        blank_positions.append((row_slice.start + local_r, col_slice.start + local_c))

    target_counts: Counter[int] = Counter()
    for counts in counts_by_region:
        for color, count in counts.items():
            target_counts[color] = max(target_counts[color], count)
    assert sum(target_counts.values()) == region_area, "completed region multiset must be unique"

    out = g.copy()
    for counts, (row, col) in zip(counts_by_region, blank_positions):
        missing = target_counts - counts
        assert len(missing) == 1, "each blank must have exactly one matching fill color"
        fill_color, fill_count = next(iter(missing.items()))
        assert fill_count == 1, "each corner region must need one pixel of the fill color"
        out[row, col] = fill_color

    return out

# --------------- generator (o5x) ---------------
# Canvas/palette bounds.
GRID_HALF_MIN = 2
GRID_HALF_MAX = 6
COLOR_COUNT = 10
MIN_TARGET_COLORS = 3
MAX_NON_BG_COLORS = COLOR_COUNT - 1
# Layout probabilities and constraints.
PREFERRED_GRID_HALF = (2, 2)
PREFERRED_GRID_PROB = 0.65
QUADRANT_COUNT = 4
SEPARATOR_WIDTH = 1
BG_DOMINANCE_MARGIN = 1
# Retry limit.
MAX_GENERATION_ATTEMPTS = 300

def _gen_bounded_composition(total: int, parts: int, max_each: int) -> list[int] | None:
    if parts <= 0 or total < parts or total > parts * max_each:
        return None

    counts = []
    remaining = total
    for i in range(parts):
        slots_left = parts - i - 1
        lo = max(1, remaining - slots_left * max_each)
        hi = min(max_each, remaining - slots_left)
        counts.append(random.randint(lo, hi))
        remaining -= counts[-1]

    random.shuffle(counts)
    return counts


def _gen_grid_options() -> list[tuple[int, int, list[int], int]]:
    options = []
    for k in range(GRID_HALF_MIN, GRID_HALF_MAX + 1):
        for m in range(GRID_HALF_MIN, GRID_HALF_MAX + 1):
            area = k * m
            bg_count = 2 * k + 2 * m + SEPARATOR_WIDTH + QUADRANT_COUNT
            max_target_count = (bg_count - BG_DOMINANCE_MARGIN) // QUADRANT_COUNT
            max_colors = min(MAX_NON_BG_COLORS, area)
            valid_counts = [
                c
                for c in range(MIN_TARGET_COLORS, max_colors + 1)
                if area <= c * max_target_count
            ]
            if valid_counts:
                options.append((k, m, valid_counts, max_target_count))
    return options


def _gen_pick_grid_option() -> tuple[int, int, list[int], int]:
    options = _gen_grid_options()
    if random.random() < PREFERRED_GRID_PROB:
        preferred_k, preferred_m = PREFERRED_GRID_HALF
        for option in options:
            if option[0] == preferred_k and option[1] == preferred_m:
                return option
    return random.choice(options)


def _gen_quadrant_offsets(k: int, m: int) -> list[tuple[int, int]]:
    sep = SEPARATOR_WIDTH
    return [(0, 0), (0, m + sep), (k + sep, 0), (k + sep, m + sep)]


def _gen_missing_colors(non_bg_colors: list[int]) -> list[int]:
    missing_colors = [random.choice(non_bg_colors) for _ in range(QUADRANT_COUNT)]
    if len(set(missing_colors)) == 1:
        alternatives = [c for c in non_bg_colors if c != missing_colors[0]]
        missing_colors[-1] = random.choice(alternatives)
    return missing_colors


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(COLOR_COUNT)
    k, m, valid_color_counts, max_target_count = _gen_pick_grid_option()
    area = k * m

    color_count = random.choice(valid_color_counts)
    non_bg_colors = random.sample([c for c in range(COLOR_COUNT) if c != bg], color_count)
    target_counts = _gen_bounded_composition(area, color_count, max_target_count)
    if target_counts is None:
        return None

    target_values = []
    for color, count in zip(non_bg_colors, target_counts):
        target_values.extend([color] * count)

    missing_colors = _gen_missing_colors(non_bg_colors)
    out = np.full((2 * k + SEPARATOR_WIDTH, 2 * m + SEPARATOR_WIDTH), bg, dtype=int)
    inp = out.copy()

    for (r0, c0), missing_color in zip(_gen_quadrant_offsets(k, m), missing_colors):
        values = target_values[:]
        random.shuffle(values)
        quadrant_out = np.array(values, dtype=int).reshape(k, m)

        candidates = np.argwhere(quadrant_out == missing_color)
        blank_r, blank_c = map(int, candidates[random.randrange(len(candidates))])
        quadrant_in = quadrant_out.copy()
        quadrant_in[blank_r, blank_c] = bg

        out[r0 : r0 + k, c0 : c0 + m] = quadrant_out
        inp[r0 : r0 + k, c0 : c0 + m] = quadrant_in

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e3fe1151 input")
