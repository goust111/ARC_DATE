from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each top-row control pixel downward as a vertical toggle ray in its own color.

    Input structure:
    - The background is the dominant color.
    - Non-background pixels on the top row are control pixels, each identifying
      one column and one ray color.
    - Foreground regions below may intersect these control columns.

    Rule:
    - For every top-row control pixel, process the cells below it in the same
      column down to the bottom edge.
    - A cell that already has the control color is erased to the background.
    - Every other cell on that downward ray is painted with the control color.
    - The top-row control pixels themselves remain unchanged.

    Key insight:
    The top row is a set of vertical ray launchers; each ray toggles only cells
    matching its own color and overwrites all other colors along that column.
    """
    g = np.asarray(grid, dtype=int)
    H, _ = g.shape
    assert H >= 2, "expected at least one row below the control row"

    bg = get_bg(g)
    marker_cols = np.flatnonzero(g[0] != bg)
    assert marker_cols.size > 0, "expected non-background control pixels in the top row"

    has_same_color_hit = False
    has_different_color_hit = False
    for c in marker_cols:
        color = int(g[0, c])
        ray = g[1:, c]
        has_same_color_hit |= bool(np.any(ray == color))
        has_different_color_hit |= bool(np.any((ray != bg) & (ray != color)))
    assert has_same_color_hit, "expected at least one control ray to hit its own color"
    assert has_different_color_hit, "expected at least one control ray to hit a different foreground color"

    out = g.copy()
    for c in marker_cols:
        color = int(g[0, c])
        out[1:, c] = np.where(g[1:, c] == color, bg, color)

    return out

# --------------- generator (att1) ---------------

# Palette and canvas bounds.
ALL_COLORS = list(range(10))
GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 30

# Top-row control pixels.
CONTROL_COUNT_MIN = 2
CONTROL_COUNT_MAX = 6
CONTROL_MIN_GAP = 3

# Solid rectangles painted below the control row.
RECT_COUNT_MIN = 1
RECT_COUNT_MAX = 6
RECT_HEIGHT_MIN = 5
RECT_HEIGHT_MAX = 14
RECT_WIDTH_MIN = 6
RECT_WIDTH_MAX = 19
FIRST_DRAW_ROW = 1

# Retry limits.
BRANCH_COVERAGE_RETRIES = 20
GENERATION_RETRIES = 300


def _ray_branch_flags(grid, control_cols, control_colors, bg):
    has_same_color_hit = False
    has_different_color_hit = False
    for col, color in zip(control_cols, control_colors):
        ray = grid[1:, col]
        has_same_color_hit |= bool(np.any(ray == color))
        has_different_color_hit |= bool(np.any((ray != bg) & (ray != color)))
    return has_same_color_hit, has_different_color_hit


def _paint_rect(grid, r0, c0, rh, rw, color):
    grid[r0:r0 + rh, c0:c0 + rw] = color


def _paint_rect_covering_col(grid, col, color):
    """Paint one random valid rectangle whose horizontal span includes col."""
    H, W = grid.shape
    rh = random.randint(RECT_HEIGHT_MIN, min(RECT_HEIGHT_MAX, H - 1))
    rw = random.randint(RECT_WIDTH_MIN, min(RECT_WIDTH_MAX, W))
    c0_min = max(0, col - rw + 1)
    c0_max = min(W - rw, col)
    if c0_max < c0_min:
        return False
    r0 = random.randint(FIRST_DRAW_ROW, H - rh)
    c0 = random.randint(c0_min, c0_max)
    _paint_rect(grid, r0, c0, rh, rw, color)
    return True


def _ensure_branch_coverage(grid, control_cols, control_colors, bg, non_bg_colors):
    for _ in range(BRANCH_COVERAGE_RETRIES):
        has_same, has_diff = _ray_branch_flags(grid, control_cols, control_colors, bg)
        if has_same and has_diff:
            return True

        if not has_same:
            idx = random.randrange(len(control_cols))
            if not _paint_rect_covering_col(grid, control_cols[idx], control_colors[idx]):
                return False

        if not has_diff:
            idx = random.randrange(len(control_cols))
            diff_choices = [c for c in non_bg_colors if c != control_colors[idx]]
            if not diff_choices:
                return False
            if not _paint_rect_covering_col(grid, control_cols[idx], random.choice(diff_choices)):
                return False

    return _ray_branch_flags(grid, control_cols, control_colors, bg) == (True, True)


def _select_control_columns(size, n_controls):
    available_cols = list(range(size))
    control_cols = []
    for _ in range(n_controls):
        valid = [c for c in available_cols
                 if all(abs(c - chosen) >= CONTROL_MIN_GAP for chosen in control_cols)]
        if not valid:
            return None
        col = random.choice(valid)
        control_cols.append(col)
    control_cols.sort()
    return control_cols


def _sample_rect(size, non_bg_colors):
    rh = random.randint(RECT_HEIGHT_MIN, RECT_HEIGHT_MAX)
    rw = random.randint(RECT_WIDTH_MIN, RECT_WIDTH_MAX)
    max_r = size - rh
    max_c = size - rw
    if max_r < FIRST_DRAW_ROW or max_c < 0:
        return None
    r0 = random.randint(FIRST_DRAW_ROW, max_r)
    c0 = random.randint(0, max_c)
    color = random.choice(non_bg_colors)
    return r0, c0, rh, rw, color


def _paint_initial_rectangles(grid, non_bg_colors):
    n_rects = random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    rects = []
    for _ in range(n_rects):
        rect = _sample_rect(grid.shape[0], non_bg_colors)
        if rect is None:
            continue
        rects.append(rect)

    for (r0, c0, rh, rw, color) in rects:
        _paint_rect(grid, r0, c0, rh, rw, color)


def _ensure_control_rays_hit(grid, control_cols, bg, non_bg_colors):
    H, W = grid.shape
    for col in control_cols:
        col_data = grid[1:, col]
        if not np.any(col_data != bg):
            rh = random.randint(RECT_HEIGHT_MIN, min(RECT_HEIGHT_MAX, H - 1))
            rw = random.randint(RECT_WIDTH_MIN, min(RECT_WIDTH_MAX, W))
            c0_min = max(0, col - rw + 1)
            c0_max = min(W - rw, col)
            if c0_max < c0_min:
                return False
            c0 = random.randint(c0_min, c0_max)
            r0 = random.randint(FIRST_DRAW_ROW, max(FIRST_DRAW_ROW, H - rh))
            color = random.choice(non_bg_colors)
            _paint_rect(grid, r0, c0, rh, rw, color)
    return True


def _set_control_row(grid, control_cols, control_colors, bg):
    grid[0, :] = bg
    for col, color in zip(control_cols, control_colors):
        grid[0, col] = color


def _oracle_accepts(grid, bg, control_cols, control_colors):
    if _ray_branch_flags(grid, control_cols, control_colors, bg) != (True, True):
        return False
    if get_bg(grid) != bg:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    """Generate an input grid for the vertical toggle ray task."""
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg = random.choice(ALL_COLORS)
    grid = np.full((size, size), bg, dtype=int)

    n_controls = random.randint(CONTROL_COUNT_MIN, CONTROL_COUNT_MAX)
    non_bg_colors = [c for c in ALL_COLORS if c != bg]
    control_cols = _select_control_columns(size, n_controls)
    if control_cols is None:
        return None
    control_colors = [random.choice(non_bg_colors) for _ in range(n_controls)]

    _paint_initial_rectangles(grid, non_bg_colors)
    if not _ensure_control_rays_hit(grid, control_cols, bg, non_bg_colors):
        return None

    if not _ensure_branch_coverage(grid, control_cols, control_colors, bg, non_bg_colors):
        return None

    _set_control_row(grid, control_cols, control_colors, bg)
    if not _oracle_accepts(grid, bg, control_cols, control_colors):
        return None

    return grid


def generate_input() -> np.ndarray:
    """Retry loop for generating valid inputs."""
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate a valid input after {GENERATION_RETRIES} attempts")
