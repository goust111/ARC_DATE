import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine the two stacked masks by marking the cells that are empty in both panels.

    Input structure:
    - The input is a 2D grid with black background.
    - A single solid non-background row uses a color that appears only in that row.
    - That separator row splits the grid into two nonempty same-size panels, each with one foreground color.

    Rule:
    - Remove the separator row and align the top and bottom panels by position.
    - Output a black grid with the panel shape.
    - Paint a teal pixel at each position where both aligned panel pixels are black.

    Key insight:
    - The output is the intersection of the two panels' background masks, not a combination of their foreground shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    separator_rows = []
    for row in range(height):
        row_color = int(g[row, 0])
        if row_color == bg:
            continue
        if np.all(g[row, :] == row_color) and np.count_nonzero(g == row_color) == width:
            separator_rows.append(row)
    assert len(separator_rows) == 1, "expected exactly one unique solid separator row"

    separator_row = separator_rows[0]
    top = g[:separator_row, :]
    bottom = g[separator_row + 1 :, :]
    assert top.shape == bottom.shape, "expected equal-size panels above and below the separator"
    assert top.size > 0, "expected nonempty panels"

    separator_color = int(g[separator_row, 0])
    top_colors = set(map(int, np.unique(top))) - {bg}
    bottom_colors = set(map(int, np.unique(bottom))) - {bg}
    assert separator_color not in (top_colors | bottom_colors), "separator color must not appear inside panels"
    assert len(top_colors) == 1, "expected one foreground color in the top panel"
    assert len(bottom_colors) == 1, "expected one foreground color in the bottom panel"

    out = np.full(top.shape, Color.BLACK, dtype=int)
    out[(top == bg) & (bottom == bg)] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Panel dimensions sampled uniformly within inclusive bounds.
PANEL_ROW_RANGE = (4, 8)
PANEL_COL_RANGE = (3, 6)
# Output mask density and hard minimum for teal cells.
TEAL_DENSITY_RANGE = (0.25, 0.38)
MIN_TEAL_CELLS = 3
TEAL_COLOR = 8
# Foreground density bounds for each reconstructed panel.
PANEL_FG_DENSITY_RANGE = (0.38, 0.58)
# Color pool excludes black and teal; retry budget covers rare rejects.
COLOR_CHOICES = np.array([1, 2, 3, 4, 5, 6, 7, 9])
GEN_MAX_ATTEMPTS = 300


def _gen_count_bounds(total: int, low: float, high: float) -> tuple[int, int]:
    return int(np.ceil(total * low)), int(np.floor(total * high))


def _gen_panel_size() -> tuple[int, int]:
    rows = int(np.random.randint(PANEL_ROW_RANGE[0], PANEL_ROW_RANGE[1] + 1))
    cols = int(np.random.randint(PANEL_COL_RANGE[0], PANEL_COL_RANGE[1] + 1))
    return rows, cols


def _gen_teal_count(rows: int, cols: int) -> int | None:
    total = rows * cols
    min_teal, max_teal = _gen_count_bounds(total, *TEAL_DENSITY_RANGE)
    min_teal = max(min_teal, cols, MIN_TEAL_CELLS)
    if min_teal > max_teal:
        return None
    return int(np.random.randint(min_teal, max_teal + 1))


def _gen_teal_mask(rows: int, cols: int, count: int) -> np.ndarray:
    """Return exactly count teal cells with at least one teal in every column."""
    mask = np.zeros((rows, cols), dtype=bool)
    base_rows = np.random.randint(0, rows, size=cols)
    mask[base_rows, np.arange(cols)] = True
    extra = count - cols
    if extra > 0:
        remaining = np.flatnonzero(~mask.ravel())
        chosen = np.random.choice(remaining, size=extra, replace=False)
        mask.ravel()[chosen] = True
    return mask


def _gen_panel_masks(teal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return panel foreground masks whose shared background is teal_mask."""
    rows, cols = map(int, teal_mask.shape)
    total = rows * cols
    black_coords = np.argwhere(~teal_mask)
    black_count = int(len(black_coords))
    min_fg, max_fg = _gen_count_bounds(total, *PANEL_FG_DENSITY_RANGE)
    max_fg = min(max_fg, black_count)
    if min_fg > max_fg:
        return None
    valid_pairs = [
        (top_count, bottom_count)
        for top_count in range(min_fg, max_fg + 1)
        for bottom_count in range(min_fg, max_fg + 1)
        if top_count + bottom_count >= black_count
    ]
    if not valid_pairs:
        return None
    top_count, bottom_count = valid_pairs[np.random.randint(0, len(valid_pairs))]
    overlap_count = top_count + bottom_count - black_count
    top_only_count = top_count - overlap_count
    perm = np.random.permutation(black_count)
    both_idx = perm[:overlap_count]
    top_only_idx = perm[overlap_count : overlap_count + top_only_count]
    bottom_only_idx = perm[overlap_count + top_only_count :]
    top_mask = np.zeros((rows, cols), dtype=bool)
    bottom_mask = np.zeros((rows, cols), dtype=bool)
    top_points = black_coords[np.concatenate([both_idx, top_only_idx])]
    bottom_points = black_coords[np.concatenate([both_idx, bottom_only_idx])]
    top_mask[top_points[:, 0], top_points[:, 1]] = True
    bottom_mask[bottom_points[:, 0], bottom_points[:, 1]] = True
    return top_mask, bottom_mask


def _render_grid(
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, int]:
    separator_color, top_color, bottom_color = map(int, colors)
    rows, cols = map(int, top_mask.shape)
    top = np.zeros((rows, cols), dtype=int)
    bottom = np.zeros((rows, cols), dtype=int)
    top[top_mask] = top_color
    bottom[bottom_mask] = bottom_color
    separator = np.full((1, cols), separator_color, dtype=int)
    grid = np.vstack([top, separator, bottom])
    return grid, separator_color


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _gen_panel_size()
    teal_count = _gen_teal_count(rows, cols)
    if teal_count is None:
        return None
    teal_mask = _gen_teal_mask(rows, cols, teal_count)
    panel_masks = _gen_panel_masks(teal_mask)
    if panel_masks is None:
        return None
    top_mask, bottom_mask = panel_masks
    colors = np.random.choice(COLOR_CHOICES, size=3, replace=False)
    grid, separator_color = _render_grid(top_mask, bottom_mask, colors)
    expected = np.zeros((rows, cols), dtype=int)
    expected[teal_mask] = TEAL_COLOR
    if np.count_nonzero(grid == separator_color) != cols:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 0c9aba6e input")
