import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each blue marker into a centered 3x3 blue patch, filling only background cells.

    Input structure:
    - The grid is a 2D noisy canvas with a dominant background color.
    - Blue pixels are interior markers.
    - Other non-background colors are obstacles/noise that must be preserved.

    Rule:
    - For every blue marker, inspect its centered 3x3 neighborhood.
    - Recolor background pixels in that neighborhood to blue.
    - Leave all existing non-background pixels unchanged.

    Key insight:
    - The output is not a global pattern; each blue pixel locally stamps a clipped-by-obstacles 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected room for centered 3x3 neighborhoods"

    bg = get_bg(g)
    stamp_color = Color.BLUE
    assert bg != stamp_color, "blue markers must be foreground"

    centers = np.argwhere(g == stamp_color)
    assert centers.size > 0, "expected at least one blue marker"
    assert np.all(
        (centers[:, 0] > 0) & (centers[:, 0] < height - 1)
    ), "blue markers must be away from top/bottom edge"
    assert np.all(
        (centers[:, 1] > 0) & (centers[:, 1] < width - 1)
    ), "blue markers must be away from left/right edge"

    out = g.copy()
    for r, c in centers:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if g[rr, cc] == bg:
                    out[rr, cc] = stamp_color

    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and marker placement.
GRID_HEIGHT, GRID_WIDTH = 15, 15
BACKGROUND_COLOR, MARKER_COLOR = Color.BLACK, Color.BLUE
OBSTACLE_PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
MARKER_COUNT_MIN, MARKER_COUNT_MAX = 4, 8
MARKER_BORDER, MARKER_MIN_SEPARATION, NEIGHBORHOOD_RADIUS = 1, 3, 1

# Obstacle density, per-color balance, and retry limit.
OBSTACLE_COLOR_COUNT_MIN, OBSTACLE_COLOR_COUNT_MAX = 4, 7
OBSTACLE_TOTAL_MIN, OBSTACLE_TOTAL_MAX = 70, 101
OBSTACLE_PER_COLOR_MIN, OBSTACLE_PER_COLOR_MAX = 4, 32
GEN_MAX_ATTEMPTS = 300


def _gen_split_counts(total: int, parts: int, low: int, high: int) -> list[int]:
    """Split a total into shuffled bounded per-color counts."""
    counts = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        lo = max(low, remaining - slots * high)
        hi = min(high, remaining - slots * low)
        val = random.randint(lo, hi)
        counts.append(val)
        remaining -= val
    random.shuffle(counts)
    return counts


def _gen_marker_positions(count: int) -> list[tuple[int, int]] | None:
    """Pick separated interior marker centers, or return None if placement fails."""
    cells = [(r, c) for r in range(MARKER_BORDER, GRID_HEIGHT - MARKER_BORDER) for c in range(MARKER_BORDER, GRID_WIDTH - MARKER_BORDER)]
    random.shuffle(cells)
    centers = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) >= MARKER_MIN_SEPARATION for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == count:
                return centers
    return None


def _gen_obstacle_plan() -> tuple[list[int], list[int], int]:
    """Choose colors, total obstacle count, and bounded per-color counts."""
    color_count = random.randint(OBSTACLE_COLOR_COUNT_MIN, OBSTACLE_COLOR_COUNT_MAX)
    obstacle_colors = random.sample(OBSTACLE_PALETTE, color_count)
    obstacle_total = random.randint(OBSTACLE_TOTAL_MIN, OBSTACLE_TOTAL_MAX)
    color_counts = _gen_split_counts(obstacle_total, color_count, OBSTACLE_PER_COLOR_MIN, OBSTACLE_PER_COLOR_MAX)
    return obstacle_colors, color_counts, obstacle_total


def _seed_marker_contacts(markers: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return forced obstacle cells and add them to used."""
    obstacle_cells: list[tuple[int, int]] = []
    touched_count = (len(markers) + 1) // 2
    for r, c in random.sample(markers, touched_count):
        nbrs = [(rr, cc) for rr in range(r - NEIGHBORHOOD_RADIUS, r + NEIGHBORHOOD_RADIUS + 1) for cc in range(c - NEIGHBORHOOD_RADIUS, c + NEIGHBORHOOD_RADIUS + 1) if (rr, cc) not in used]
        random.shuffle(nbrs)
        cell = nbrs[0]
        obstacle_cells.append(cell)
        used.add(cell)
    return obstacle_cells


def _paint_obstacles(grid: np.ndarray, obstacle_cells: list[tuple[int, int]], obstacle_colors: list[int], color_counts: list[int]) -> None:
    obstacle_values = []
    for color, count in zip(obstacle_colors, color_counts):
        obstacle_values.extend([color] * count)
    random.shuffle(obstacle_values)

    for (r, c), color in zip(obstacle_cells, obstacle_values):
        grid[r, c] = color


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r, c in np.argwhere(grid == MARKER_COLOR):
        for rr in range(int(r) - NEIGHBORHOOD_RADIUS, int(r) + NEIGHBORHOOD_RADIUS + 1):
            for cc in range(int(c) - NEIGHBORHOOD_RADIUS, int(c) + NEIGHBORHOOD_RADIUS + 1):
                if grid[rr, cc] == BACKGROUND_COLOR:
                    out[rr, cc] = MARKER_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

    marker_count = random.randint(MARKER_COUNT_MIN, MARKER_COUNT_MAX)
    markers = _gen_marker_positions(marker_count)
    if markers is None:
        return None

    obstacle_colors, color_counts, obstacle_total = _gen_obstacle_plan()

    used = set(markers)
    obstacle_cells = _seed_marker_contacts(markers, used)
    free_cells = [(r, c) for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if (r, c) not in used]
    random.shuffle(free_cells)
    need = obstacle_total - len(obstacle_cells)
    if need < 0 or need > len(free_cells):
        return None
    obstacle_cells.extend(free_cells[:need])
    random.shuffle(obstacle_cells)

    _paint_obstacles(grid, obstacle_cells, obstacle_colors, color_counts)
    for r, c in markers:
        grid[r, c] = MARKER_COLOR
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid f0df5ff0 input")
