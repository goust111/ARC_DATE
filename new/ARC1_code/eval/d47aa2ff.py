import numpy as np

from arc_utils import Color, get_bg
import itertools
import random


def _separator_column(g: np.ndarray, bg: int) -> int:
    height, width = map(int, g.shape)
    assert height > 0 and width >= 3 and width % 2 == 1, "expected two panels plus a divider"

    candidates: list[int] = []
    for col in range(width):
        color = int(g[0, col])
        if color == bg:
            continue
        if not np.all(g[:, col] == color):
            continue
        if col == 0 or col == width - 1 or col != width - col - 1:
            continue
        without_col = np.concatenate((g[:, :col], g[:, col + 1 :]), axis=1)
        if np.any(without_col == color):
            continue
        candidates.append(col)

    assert len(candidates) == 1, "expected one solid divider column splitting equal panels"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two separated panels, keeping shared pixels and marking pixels that appear in only one panel.

    Input structure
    - The input has two equal-width panels separated by one solid non-background divider column.
    - Background is the dominant color, and the divider color appears only in the divider.

    Rule
    - Remove the divider and compare the left and right panels cell by cell.
    - If both panels have the same value at a cell, copy that value.
    - If only the left panel has foreground at a cell, write red.
    - If only the right panel has foreground at a cell, write blue.

    Key insight
    - The output is a per-position comparison map of two aligned panels, not a crop of either panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep_col = _separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "divider must split two equal-size panels"

    overlap = (left != bg) & (right != bg)
    assert np.all(left[overlap] == right[overlap]), "overlapping foreground must agree in color"

    out = np.full(left.shape, bg, dtype=int)
    same = left == right
    out[same] = left[same]
    out[(left != bg) & (right == bg)] = Color.RED
    out[(left == bg) & (right != bg)] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds and color choices.
PANEL_WIDTH_RANGE = (8, 12)
PANEL_HEIGHT_RANGE = (8, 12)
BACKGROUND_COLOR = 0
SEPARATOR_COLORS = (3, 4, 5, 6, 7, 8, 9)
FOREGROUND_COLORS = (3, 4, 5, 6, 7, 8, 9)

# Base panel structure.
COLOR_COUNT_CHOICES = (2, 3)
PIXELS_PER_COLOR_CHOICES = (2, 3, 4)
TOTAL_PIXEL_RANGE = (7, 10)
SHIFT_COUNT_RANGE = (1, 2)

# Shift placement and retry limits.
SHIFT_SOURCE_MARKERS = (Color.RED, Color.BLUE)
SHIFT_SEARCH_RADII = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 300

COUNT_OPTIONS_BY_COLOR_COUNT = {
    n: [list(c) for c in itertools.product(PIXELS_PER_COLOR_CHOICES, repeat=n)
        if TOTAL_PIXEL_RANGE[0] <= sum(c) <= TOTAL_PIXEL_RANGE[1]]
    for n in COLOR_COUNT_CHOICES
}


def _gen_neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _gen_isolated_cells(shape, occupied):
    height, width = shape
    cells = []
    for r in range(height):
        for c in range(width):
            if (r, c) in occupied:
                continue
            if any((nr, nc) in occupied for nr, nc in _gen_neighbors4(r, c)):
                continue
            cells.append((r, c))
    return cells


def _gen_panel_isolated(panel):
    fg = panel != BACKGROUND_COLOR
    return not (np.any(fg[1:, :] & fg[:-1, :]) or np.any(fg[:, 1:] & fg[:, :-1]))


def _gen_choose_palette():
    """Choose divider color, foreground colors, and per-color pixel counts."""
    sep_color = random.choice(SEPARATOR_COLORS)
    n_colors = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample([c for c in FOREGROUND_COLORS if c != sep_color], n_colors)
    counts = random.choice(COUNT_OPTIONS_BY_COLOR_COUNT[n_colors])
    return sep_color, colors, counts


def _gen_place_base_pixels(shape, colors, counts):
    """Place isolated base pixels for each foreground color."""
    by_color = {}
    occupied = set()
    for color, count in zip(colors, counts):
        coords = []
        for _ in range(count):
            choices = _gen_isolated_cells(shape, occupied)
            if not choices:
                return None
            coord = random.choice(choices)
            coords.append(coord)
            occupied.add(coord)
        by_color[color] = coords
    return by_color, occupied


def _gen_shift_destination(shape, source, blocked, target_occupied):
    height, width = shape
    sr, sc = source
    candidates = []
    for radius in SHIFT_SEARCH_RADII:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if (dr == 0 and dc == 0) or max(abs(dr), abs(dc)) != radius:
                    continue
                r, c = sr + dr, sc + dc
                if not (0 <= r < height and 0 <= c < width):
                    continue
                if (r, c) in blocked or (r, c) in target_occupied:
                    continue
                if any((nr, nc) in target_occupied for nr, nc in _gen_neighbors4(r, c)):
                    continue
                candidates.append((r, c))
        if candidates:
            return random.choice(candidates)
    return None


def _gen_shift_plans(shape, colors, by_color, occupied):
    """Choose shifted source pixels and assign valid destination pixels."""
    shift_count = random.randint(SHIFT_COUNT_RANGE[0], min(SHIFT_COUNT_RANGE[1], len(colors) - 1))
    shift_colors = random.sample(colors, shift_count)
    plans = []
    left_occupied = set(occupied)
    right_occupied = set(occupied)
    for color in shift_colors:
        source = random.choice(by_color[color])
        source_marker = random.choice(SHIFT_SOURCE_MARKERS)
        if source_marker == Color.RED:
            right_occupied.remove(source)
        else:
            left_occupied.remove(source)
        plans.append((color, source, source_marker))
    blocked = set(occupied)
    completed = []
    for color, source, source_marker in plans:
        target = right_occupied if source_marker == Color.RED else left_occupied
        dest = _gen_shift_destination(shape, source, blocked, target)
        if dest is None:
            return None
        target.add(dest)
        blocked.add(dest)
        completed.append((color, source, dest, source_marker))
    return completed


def _gen_render_grid(shape, by_color, shift_plans, sep_color):
    """Render panels, validate invariants, and return the assembled input grid."""
    left = np.zeros(shape, dtype=int)
    right = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    for color, coords in by_color.items():
        for r, c in coords:
            left[r, c] = color
            right[r, c] = color
            expected[r, c] = color

    for color, source, dest, source_marker in shift_plans:
        sr, sc = source
        dr, dc = dest
        if source_marker == Color.RED:
            right[sr, sc] = BACKGROUND_COLOR
            right[dr, dc] = color
            expected[sr, sc] = Color.RED
            expected[dr, dc] = Color.BLUE
        else:
            left[sr, sc] = BACKGROUND_COLOR
            left[dr, dc] = color
            expected[sr, sc] = Color.BLUE
            expected[dr, dc] = Color.RED

    if not _gen_panel_isolated(left) or not _gen_panel_isolated(right):
        return None
    if np.count_nonzero(left) != np.count_nonzero(right):
        return None
    if np.any(left == sep_color) or np.any(right == sep_color):
        return None

    sep = np.full((shape[0], 1), sep_color, dtype=int)
    grid = np.concatenate([left, sep, right], axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*PANEL_WIDTH_RANGE)
    height = random.randint(*PANEL_HEIGHT_RANGE)
    shape = (height, width)
    sep_color, colors, counts = _gen_choose_palette()
    placed = _gen_place_base_pixels(shape, colors, counts)
    if placed is None:
        return None
    by_color, occupied = placed
    shift_plans = _gen_shift_plans(shape, colors, by_color, occupied)
    if shift_plans is None:
        return None
    return _gen_render_grid(shape, by_color, shift_plans, sep_color)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d47aa2ff input")
