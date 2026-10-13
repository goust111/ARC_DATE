from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and np.array_equal(
        indices, np.arange(int(indices[0]), int(indices[-1]) + 1)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the visible color of the crossing area between two perpendicular full-span bars.

    Input structure:
    - The dominant background surrounds two non-background colors.
    - One horizontal bar occupies a contiguous band of rows with no background cells.
    - One vertical bar occupies a contiguous band of columns with no background cells.
    - Their rectangular crossing is uniformly one of the two foreground colors.

    Rule:
    - Preserve the grid except for the crossing rectangle.
    - Recolor every crossing pixel to the other foreground color, making the hidden bar visible there.

    Key insight:
    - The crossing is exactly the intersection of all-foreground rows and all-foreground columns; its current color identifies the bar on top, so the output flips it to the other bar color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    foreground = g != bg
    full_rows = np.flatnonzero(foreground.all(axis=1))
    full_cols = np.flatnonzero(foreground.all(axis=0))
    assert _is_contiguous(full_rows), "expected one contiguous horizontal bar band"
    assert _is_contiguous(full_cols), "expected one contiguous vertical bar band"

    crossing = g[np.ix_(full_rows, full_cols)]
    crossing_colors = np.unique(crossing)
    assert crossing_colors.size == 1, "crossing must have a single visible color"
    visible_color = int(crossing_colors[0])
    assert visible_color in fg_colors, "crossing color must be foreground"

    replacement = [color for color in fg_colors if color != visible_color]
    assert len(replacement) == 1, "expected one hidden bar color"

    out = g.copy()
    out[np.ix_(full_rows, full_cols)] = replacement[0]
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry limits.
GEN_SIZE_RANGE = range(6, 14)
GEN_COLORS = np.arange(1, 10)
GEN_MAX_ATTEMPTS = 300

# Bar thicknesses, interior placement margin, and layer choices.
GEN_H_THICKNESSES = range(1, 4)
GEN_V_THICKNESSES = range(1, 3)
GEN_EDGE_MARGIN = 1
GEN_TOP_FLAGS = (False, True)


def _gen_bg_dominates(H: int, W: int, t_h: int, t_v: int, top_is_h: bool) -> bool:
    crossing = t_h * t_v
    bg_count = H * W - (t_h * W + t_v * H - crossing)
    h_count = t_h * W - (0 if top_is_h else crossing)
    v_count = t_v * H - (crossing if top_is_h else 0)
    return bg_count > max(h_count, v_count)


GEN_VALID_PARAMS = [
    (H, W, t_h, t_v, top_is_h)
    for H in GEN_SIZE_RANGE
    for W in GEN_SIZE_RANGE
    for t_h in GEN_H_THICKNESSES
    for t_v in GEN_V_THICKNESSES
    for top_is_h in GEN_TOP_FLAGS
    if _gen_bg_dominates(H, W, t_h, t_v, top_is_h)
]


def _gen_attempt() -> np.ndarray | None:
    H, W, t_h, t_v, top_is_h = GEN_VALID_PARAMS[
        int(np.random.randint(len(GEN_VALID_PARAMS)))
    ]
    r0 = int(np.random.randint(GEN_EDGE_MARGIN, H - t_h))
    c0 = int(np.random.randint(GEN_EDGE_MARGIN, W - t_v))
    c_h, c_v = map(int, np.random.choice(GEN_COLORS, size=2, replace=False))

    grid = np.zeros((H, W), dtype=int)
    grid[r0 : r0 + t_h, :] = c_h
    grid[:, c0 : c0 + t_v] = c_v
    grid[r0 : r0 + t_h, c0 : c0 + t_v] = c_h if top_is_h else c_v

    expected = grid.copy()
    expected[r0 : r0 + t_h, c0 : c0 + t_v] = c_v if top_is_h else c_h
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ba97ae07 input")
