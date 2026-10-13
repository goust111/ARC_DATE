import math

import numpy as np

from arc_utils import Color, find_components
import random
from scipy.ndimage import label


STATIC_COLORS = {Color.YELLOW, Color.ORANGE}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the separated outer regions by rotating their colors one step clockwise around the yellow-orange divider.

    Input structure:
    - The grid contains a fixed yellow/orange divider structure.
    - All other colors form separated 4-connected monochrome regions around that divider.

    Rule:
    - Leave the yellow and orange pixels unchanged.
    - Sort the non-divider regions by their centroid angle around the divider centroid in screen-coordinate clockwise order.
    - Each region keeps its cells and receives the color of the previous region in that cyclic order.

    Key insight:
    - The objects do not move; only the color labels cycle through the spatial order of the regions around the divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    present = set(map(int, np.unique(g)))
    assert STATIC_COLORS <= present, "expected yellow and orange divider colors"
    moving_colors = present - STATIC_COLORS
    assert len(moving_colors) >= 2, "expected at least two regions to cycle"

    components = find_components(
        g,
        bg=-1,
        colors=sorted(moving_colors),
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == len(moving_colors), "each moving color must form exactly one 4-connected region"

    divider_coords = np.argwhere(np.isin(g, sorted(STATIC_COLORS)))
    assert divider_coords.size, "expected divider pixels"
    center_r, center_c = divider_coords.mean(axis=0)

    ordered = []
    for component in components:
        coords = component["coords"]
        centroid_r, centroid_c = coords.mean(axis=0)
        angle = math.atan2(float(centroid_r - center_r), float(centroid_c - center_c))
        ordered.append((angle, component))

    ordered.sort(key=lambda item: item[0])
    angles = [angle for angle, _component in ordered]
    for left, right in zip(angles, angles[1:]):
        assert not math.isclose(left, right), "region angles must define a unique cyclic order"

    regions = [component for _angle, component in ordered]
    out = g.copy()
    for i, region in enumerate(regions):
        source = regions[(i - 1) % len(regions)]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(source["color"])

    return out

# --------------- generator (att1) ---------------
# Canvas and palette.
GRID_SHAPE = (8, 8)
GRID_ROWS, GRID_COLS = GRID_SHAPE
MIN_COL, MAX_COL = 0, GRID_COLS - 1
EMPTY_COLOR = -1
YELLOW_COLOR, ORANGE_COLOR = Color.YELLOW, Color.ORANGE
DIVIDER_COLORS = (YELLOW_COLOR, ORANGE_COLOR)
AVAILABLE_MOVING_COLORS = [0, 1, 2, 3, 5, 6, 8, 9]

# Target sector count sampling.
SECTOR_COUNTS, SECTOR_COUNT_WEIGHTS = [2, 3, 4], [3, 7, 2]

# Convex yellow blob row span sampling.
N4_TOP_ROW_CHOICES, N4_BOTTOM_ROW_CHOICES = [0, 1, 1, 1], [6, 6, 6, 7]
DEFAULT_TOP_ROW_RANGE, DEFAULT_BOTTOM_ROW_RANGE = (0, 2), (5, 7)
MIN_BLOB_ROWS = 3

# Convex yellow blob width sampling.
N4_END_LEFT_RANGE = (2, 4)
N4_END_RIGHT_CAP, N4_END_RIGHT_EXTRA_MAX = 5, 2
N3_LEFT_PEAK_RIGHT_RANGE, N3_LEFT_END_LEFT_RANGE = (5, 7), (1, 4)
N3_RIGHT_PEAK_LEFT_RANGE = (0, 2)
N3_RIGHT_END_LEFT_CAP, N3_RIGHT_END_RIGHT_CAP = 4, 6
N2_END_LEFT_RANGE, N2_END_RIGHT_RANGE = (0, 3), (4, 7)

# Sector validation.
MIN_YELLOW_PIXELS = 8
MIN_SECTOR_SIZE, MAX_SECTOR_SIZE = 3, 19
MIN_DIVIDER_FRACTION, MAX_DIVIDER_FRACTION = 0.35, 0.70
MIN_ANGLE_GAP_DEG = 60

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _sample_blob_widths(target_n: int) -> tuple[int, int, int, int, int, int]:
    if target_n == 4:
        peak_left = MIN_COL
        peak_right = MAX_COL
        top_left = random.randint(*N4_END_LEFT_RANGE)
        top_right = random.randint(top_left + 1, min(top_left + N4_END_RIGHT_EXTRA_MAX, N4_END_RIGHT_CAP))
        bot_left = random.randint(*N4_END_LEFT_RANGE)
        bot_right = random.randint(bot_left + 1, min(bot_left + N4_END_RIGHT_EXTRA_MAX, N4_END_RIGHT_CAP))
    elif target_n == 3:
        if random.random() < 0.5:
            peak_left = MIN_COL
            peak_right = random.randint(*N3_LEFT_PEAK_RIGHT_RANGE)
            top_left = random.randint(*N3_LEFT_END_LEFT_RANGE)
            top_right = random.randint(top_left + 1, peak_right)
            bot_left = random.randint(*N3_LEFT_END_LEFT_RANGE)
            bot_right = random.randint(bot_left + 1, peak_right)
        else:
            peak_left = random.randint(*N3_RIGHT_PEAK_LEFT_RANGE)
            peak_right = MAX_COL
            top_left = random.randint(peak_left, N3_RIGHT_END_LEFT_CAP)
            top_right = random.randint(top_left + 1, N3_RIGHT_END_RIGHT_CAP)
            bot_left = random.randint(peak_left, N3_RIGHT_END_LEFT_CAP)
            bot_right = random.randint(bot_left + 1, N3_RIGHT_END_RIGHT_CAP)
    else:
        peak_left = MIN_COL
        peak_right = MAX_COL
        top_left = random.randint(*N2_END_LEFT_RANGE)
        top_right = random.randint(*N2_END_RIGHT_RANGE)
        bot_left = random.randint(*N2_END_LEFT_RANGE)
        bot_right = random.randint(*N2_END_RIGHT_RANGE)

    return top_left, top_right, peak_left, peak_right, bot_left, bot_right


def _random_convex_blob(target_n: int) -> list | None:
    """Return one row interval per grid row; validation later checks exact sectors."""
    if target_n == 4:
        r_top = random.choice(N4_TOP_ROW_CHOICES)
        r_bot = random.choice(N4_BOTTOM_ROW_CHOICES)
    else:
        r_top = random.randint(*DEFAULT_TOP_ROW_RANGE)
        r_bot = random.randint(*DEFAULT_BOTTOM_ROW_RANGE)

    n_rows = r_bot - r_top + 1
    if n_rows < MIN_BLOB_ROWS:
        return None

    peak_idx = random.randint(1, n_rows - 2)
    top_left, top_right, peak_left, peak_right, bot_left, bot_right = _sample_blob_widths(target_n)

    lefts = [0] * n_rows
    rights = [0] * n_rows
    for i in range(n_rows):
        if i <= peak_idx:
            t = i / peak_idx
        else:
            t = (i - peak_idx) / (n_rows - 1 - peak_idx)

        if i <= peak_idx:
            lefts[i] = int(round(top_left + t * (peak_left - top_left)))
            rights[i] = int(round(top_right + t * (peak_right - top_right)))
        else:
            lefts[i] = int(round(peak_left + t * (bot_left - peak_left)))
            rights[i] = int(round(peak_right + t * (bot_right - peak_right)))

        lefts[i] = max(MIN_COL, min(lefts[i], MAX_COL))
        rights[i] = max(MIN_COL, min(rights[i], MAX_COL))
        if lefts[i] > rights[i]:
            return None

    row_ranges = [None] * GRID_ROWS
    for i in range(n_rows):
        row_ranges[r_top + i] = (lefts[i], rights[i])
    return row_ranges


def _build_divider_grid(row_ranges: list) -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, EMPTY_COLOR, dtype=int)
    for r, rng in enumerate(row_ranges):
        if rng is not None:
            grid[r, rng[0]:rng[1] + 1] = YELLOW_COLOR
    if int(np.sum(grid == YELLOW_COLOR)) < MIN_YELLOW_PIXELS:
        return None

    yellow = grid == YELLOW_COLOR
    border = np.zeros(GRID_SHAPE, dtype=bool)
    border[1:, :] |= yellow[:-1, :]
    border[:-1, :] |= yellow[1:, :]
    border[:, 1:] |= yellow[:, :-1]
    border[:, :-1] |= yellow[:, 1:]
    grid[border & ~yellow] = ORANGE_COLOR
    return grid


def _label_valid_sectors(grid: np.ndarray, target_n: int) -> tuple[np.ndarray, int] | None:
    divider_fraction = float(np.mean(np.isin(grid, DIVIDER_COLORS)))
    if not (MIN_DIVIDER_FRACTION <= divider_fraction <= MAX_DIVIDER_FRACTION):
        return None

    remaining = ~np.isin(grid, DIVIDER_COLORS)
    labeled, N = label(remaining)
    if N != target_n:
        return None

    for s in range(1, N + 1):
        sz = int(np.sum(labeled == s))
        if sz < MIN_SECTOR_SIZE or sz > MAX_SECTOR_SIZE:
            return None
    return labeled, N


def _build_output(grid: np.ndarray, labeled: np.ndarray, sector_count: int) -> np.ndarray:
    colors = random.sample(AVAILABLE_MOVING_COLORS, sector_count)
    output = grid.copy()
    for s in range(1, sector_count + 1):
        output[labeled == s] = colors[s - 1]
    return output


def _ordered_sectors(output: np.ndarray, labeled: np.ndarray, sector_count: int) -> list[int] | None:
    divider_coords = np.argwhere(np.isin(output, DIVIDER_COLORS))
    center_r, center_c = divider_coords.mean(axis=0)

    angle_sector = []
    for s in range(1, sector_count + 1):
        coords = np.argwhere(labeled == s)
        cr, cc = coords.mean(axis=0)
        angle = math.atan2(float(cr - center_r), float(cc - center_c))
        angle_sector.append((angle, s))
    angle_sector.sort()

    angles = [a for a, _sector in angle_sector]
    if any(math.isclose(left, right) for left, right in zip(angles, angles[1:])):
        return None

    min_gap = math.radians(MIN_ANGLE_GAP_DEG)
    if min((angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))) < min_gap:
        return None

    return [s for _, s in angle_sector]


def _build_input(output: np.ndarray, labeled: np.ndarray, ordered_sectors: list[int]) -> np.ndarray:
    inp = output.copy()
    for i, s in enumerate(ordered_sectors):
        next_s = ordered_sectors[(i + 1) % len(ordered_sectors)]
        inp[labeled == s] = output[labeled == next_s][0]
    return inp


def _gen_attempt() -> np.ndarray | None:
    target_n = random.choices(SECTOR_COUNTS, weights=SECTOR_COUNT_WEIGHTS)[0]

    row_ranges = _random_convex_blob(target_n)
    if row_ranges is None:
        return None

    grid = _build_divider_grid(row_ranges)
    if grid is None:
        return None

    sectors = _label_valid_sectors(grid, target_n)
    if sectors is None:
        return None
    labeled, sector_count = sectors

    output = _build_output(grid, labeled, sector_count)
    ordered_sectors = _ordered_sectors(output, labeled, sector_count)
    if ordered_sectors is None:
        return None

    inp = _build_input(output, labeled, ordered_sectors)
    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
