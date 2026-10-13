from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg


ARM_OFFSETS = (
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
)
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _assert_two_cell_bars(g: np.ndarray, bg: int, fg: int) -> None:
    components = find_components(g, bg=bg, connectivity=4, colors=[fg])
    assert components, "expected foreground bars"
    for component in components:
        coords = np.asarray(component["coords"], dtype=int)
        assert coords.shape == (2, 2), "each foreground object must be a two-pixel bar"
        r0, c0 = map(int, coords[0])
        r1, c1 = map(int, coords[1])
        is_bar = (r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1)
        assert is_bar, "foreground objects must be horizontal or vertical adjacent pairs"


def _is_cross_center(g: np.ndarray, r: int, c: int, bg: int, fg: int) -> bool:
    if int(g[r, c]) != bg:
        return False
    return all(int(g[r + dr, c + dc]) == fg for dr, dc in ARM_OFFSETS) and all(
        int(g[r + dr, c + dc]) == bg for dr, dc in GAP_OFFSETS
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the missing center of each plus-shaped arrangement of four two-pixel bars with yellow.

    Input structure:
    - The background is the dominant color.
    - The single foreground color forms isolated 4-connected objects, each a horizontal or vertical two-pixel bar.
    - Some bars form an implied plus sign: two horizontal bars and two vertical bars around an empty center, with a one-pixel gap between the center and each bar.

    Rule:
    - Keep the input unchanged except at implied plus-sign centers.
    - For every background pixel with foreground bars at offsets 2-3 cells left, right, above, and below, recolor that center yellow.
    - The one-pixel gaps immediately next to the center in the four cardinal directions must remain background.

    Key insight:
    - The yellow pixels are not extending a line; they identify the shared missing center where four separated two-pixel bars align as a plus sign.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 7 and width >= 7, "grid must be large enough for the plus motif"

    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 1, "expected exactly one foreground color"
    fg = int(foreground[0])
    assert bg != Color.YELLOW and fg != Color.YELLOW, "yellow is reserved for output markers"
    _assert_two_cell_bars(g, bg, fg)

    centers = [
        (r, c)
        for r in range(3, height - 3)
        for c in range(3, width - 3)
        if _is_cross_center(g, r, c, bg, fg)
    ]
    assert centers, "expected at least one missing plus center"

    out = g.copy()
    for r, c in centers:
        out[r, c] = Color.YELLOW
    return out

# --------------- generator (att1) ---------------

# Canvas and color bounds.
GRID_SIZE_MIN = 8
GRID_SIZE_MAX_EXCLUSIVE = 21
BACKGROUND_COLOR = 8
FOREGROUND_COLOR = 1

# Plus motif placement.
PLUS_COUNT_MIN = 1
PLUS_COUNT_MAX_EXCLUSIVE = 5
CENTER_MARGIN = 3
CENTER_TRIES = 100

# Distractor bar placement.
DISTRACTOR_RATIO_MIN = 0.2
DISTRACTOR_RATIO_MAX = 0.6
DISTRACTOR_RETRY_MULTIPLIER = 20
NEW_PLUS_SCAN_RADIUS = 3

# Overall generation retries.
GENERATION_ATTEMPTS = 300
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _offset_cells(r: int, c: int, offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in offsets]


def _all_cells_bg(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    height, width = grid.shape
    for r, c in cells:
        if r < 0 or r >= height or c < 0 or c >= width:
            return False
        if grid[r, c] != bg:
            return False
    return True


def _cells_are_isolated(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    """Check that cells have no 4-neighbor foreground outside their own component."""
    height, width = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != bg:
                return False
    return True


def _place_plus_centers(grid: np.ndarray, rng: np.random.Generator, count: int, bg: int, fg: int) -> list[tuple[int, int]]:
    """Place up to count complete plus motifs and return their hidden centers."""
    height, width = grid.shape
    centers = []
    for _ in range(count):
        found = False
        for _try in range(CENTER_TRIES):
            r = int(rng.integers(CENTER_MARGIN, height - CENTER_MARGIN))
            c = int(rng.integers(CENTER_MARGIN, width - CENTER_MARGIN))
            arm_cells = _offset_cells(r, c, ARM_OFFSETS)
            gap_cells = _offset_cells(r, c, GAP_OFFSETS)
            all_cells = arm_cells + gap_cells + [(r, c)]

            if not _all_cells_bg(grid, all_cells, bg):
                continue
            if not _cells_are_isolated(grid, arm_cells, bg):
                continue

            for ar, ac in arm_cells:
                grid[ar, ac] = fg
            centers.append((r, c))
            found = True
            break

        if not found:
            break
    return centers


def _target_distractor_count(rng: np.random.Generator, plus_count: int) -> int:
    arm_bars = plus_count * 4
    ratio = rng.uniform(DISTRACTOR_RATIO_MIN, DISTRACTOR_RATIO_MAX)
    return max(1, int(round(arm_bars * ratio / (1 - ratio))))


def _candidate_centers_near_bar(bar_cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    check_cells = set()
    for br, bc in bar_cells:
        for dr in range(-NEW_PLUS_SCAN_RADIUS, NEW_PLUS_SCAN_RADIUS + 1):
            for dc in range(-NEW_PLUS_SCAN_RADIUS, NEW_PLUS_SCAN_RADIUS + 1):
                cr, cc = br + dr, bc + dc
                if CENTER_MARGIN <= cr < height - CENTER_MARGIN and CENTER_MARGIN <= cc < width - CENTER_MARGIN:
                    check_cells.add((cr, cc))
    return check_cells


def _would_create_extra_plus(
    grid: np.ndarray, bar_cells: list[tuple[int, int]], centers: list[tuple[int, int]], bg: int, fg: int
) -> bool:
    height, width = grid.shape
    for cr, cc in _candidate_centers_near_bar(bar_cells, height, width):
        if (cr, cc) in centers:
            continue
        if grid[cr, cc] != bg:
            continue
        arms_ok = all(0 <= cr + dr < height and 0 <= cc + dc < width and grid[cr + dr, cc + dc] == fg
                      for dr, dc in ARM_OFFSETS)
        gaps_ok = all(0 <= cr + dr < height and 0 <= cc + dc < width and grid[cr + dr, cc + dc] == bg
                      for dr, dc in GAP_OFFSETS)
        if arms_ok and gaps_ok:
            return True
    return False


def _place_distractors(
    grid: np.ndarray, rng: np.random.Generator, target_count: int, centers: list[tuple[int, int]], bg: int, fg: int
) -> int:
    height, width = grid.shape
    placed = 0
    for _ in range(target_count * DISTRACTOR_RETRY_MULTIPLIER):
        if placed >= target_count:
            break

        orientation = rng.integers(0, 2)
        r0 = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width))
        if orientation == 0:
            r1, c1 = r0, c0 + 1
        else:
            r1, c1 = r0 + 1, c0

        if r1 >= height or c1 >= width:
            continue
        if grid[r0, c0] != bg or grid[r1, c1] != bg:
            continue

        bar_cells = [(r0, c0), (r1, c1)]
        if not _cells_are_isolated(grid, bar_cells, bg):
            continue

        for r, c in bar_cells:
            grid[r, c] = fg

        if _would_create_extra_plus(grid, bar_cells, centers, bg, fg):
            for r, c in bar_cells:
                grid[r, c] = bg
            continue

        placed += 1
    return placed


def _passes_solver_oracle(grid: np.ndarray, centers: list[tuple[int, int]]) -> bool:
    try:
        out = main(grid.copy())
    except (AssertionError, Exception):
        return False

    expected = grid.copy()
    for r, c in centers:
        expected[r, c] = 4
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    height = rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE)
    width = rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE)
    bg = BACKGROUND_COLOR
    fg = FOREGROUND_COLOR

    grid = np.full((height, width), bg, dtype=int)

    plus_count = rng.integers(PLUS_COUNT_MIN, PLUS_COUNT_MAX_EXCLUSIVE)
    if height < 2 * CENTER_MARGIN + 1 or width < 2 * CENTER_MARGIN + 1:
        return None

    centers = _place_plus_centers(grid, rng, plus_count, bg, fg)
    if len(centers) == 0:
        return None

    distractor_count = _target_distractor_count(rng, len(centers))
    placed_distractors = _place_distractors(grid, rng, distractor_count, centers, bg, fg)
    if placed_distractors < 1:
        return None

    if not _passes_solver_oracle(grid, centers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
