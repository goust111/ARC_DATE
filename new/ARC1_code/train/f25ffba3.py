from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty upper half with a vertical mirror of the lower-half pattern.

    Input structure
    The grid has even height. The upper half is entirely background, and the
    lower half contains the foreground pattern to be completed.

    Rule
    Copy the lower half into the upper half after reversing its row order.
    Leave the original lower half unchanged.

    Key insight
    The blank upper region is the missing reflected counterpart of the visible
    lower region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _ = g.shape
    assert height % 2 == 0, "expected an even-height grid"

    bg = get_bg(g)
    mid = height // 2
    assert np.all(g[:mid] == bg), "expected the upper half to be background"
    assert np.any(g[mid:] != bg), "expected a nonempty lower-half pattern"

    out = g.copy()
    out[:mid] = g[mid:][::-1]
    return out

# --------------- generator (o5x) ---------------
# Canvas/object size bounds.
HEIGHT_CHOICES, WIDTH_MIN, WIDTH_MAX, BACKGROUND = [6, 8, 10, 12, 14], 3, 5, 0
# Color assignment bounds.
FOREGROUND_COLORS, COLOR_COUNT_MIN, COLOR_COUNT_MAX_EXCLUSIVE = np.arange(1, 10), 3, 5
# Secondary-color mutation bounds.
SECONDARY_COL_MAX, SECONDARY_DEPTH_MAX = 2, 2
# Retry limit.
MAX_ATTEMPTS = 300

def _sample_layout():
    """Return dimensions and shuffled distinct lower-half bar heights."""
    height = int(np.random.choice(HEIGHT_CHOICES))
    half = height // 2
    width = int(np.random.randint(WIDTH_MIN, min(WIDTH_MAX, half) + 1))
    heights = list(np.random.choice(np.arange(1, half), size=width - 1, replace=False))
    heights.append(half)
    return height, half, width, [int(v) for v in np.random.permutation(heights)]

def _assign_primary_colors(width):
    """Return primary colors by column plus unused colors needing coverage."""
    color_count = int(np.random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX_EXCLUSIVE))
    colors = [int(v) for v in np.random.choice(FOREGROUND_COLORS, size=color_count, replace=False)]
    cols = [int(v) for v in np.random.permutation(width)]
    primary = [None] * width
    for col, color in zip(cols, colors):
        primary[col] = color
    if width >= color_count:
        for col in cols[color_count:]:
            primary[col] = int(np.random.choice(colors))
        return primary, colors, []
    return primary, colors, colors[width:]

def _paint_lower_half(half, width, heights, primary, colors, unused_colors):
    """Render the lower half, or return None when constraints cannot fit."""
    lower = np.zeros((half, width), dtype=int)
    for col, bar_height in enumerate(heights):
        lower[half - bar_height :, col] = primary[col]
    eligible = [col for col, bar_height in enumerate(heights) if 2 <= bar_height < half]
    max_two_color = min(SECONDARY_COL_MAX, (width - 1) // 2, len(eligible))
    need_two_color = 1 if unused_colors else 0
    if max_two_color < need_two_color:
        return None
    two_color_count = int(np.random.randint(need_two_color, max_two_color + 1))
    if not two_color_count:
        return lower
    two_color_cols = [int(v) for v in np.random.choice(eligible, size=two_color_count, replace=False)]
    shared_second = unused_colors[0] if unused_colors else int(np.random.choice(colors))
    for col in two_color_cols:
        choices = [c for c in colors if c != primary[col]]
        second = shared_second if shared_second != primary[col] else int(np.random.choice(choices))
        depth = int(np.random.randint(1, min(SECONDARY_DEPTH_MAX, heights[col] - 1) + 1))
        lower[half - depth :, col] = second
    return lower

def _gen_attempt() -> np.ndarray | None:
    height, half, width, heights = _sample_layout()
    primary, colors, unused_colors = _assign_primary_colors(width)
    lower = _paint_lower_half(half, width, heights, primary, colors, unused_colors)
    if lower is None:
        return None
    out = np.zeros((height, width), dtype=int)
    out[half:], out[:half] = lower, lower[::-1]
    inp = out.copy()
    inp[:half] = BACKGROUND
    return inp if np.array_equal(main(inp), out) else None

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
