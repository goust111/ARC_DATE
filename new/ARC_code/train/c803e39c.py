import numpy as np


def _unique_dominant_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    max_count = counts.max()
    candidates = vals[counts == max_count]
    assert len(candidates) == 1
    return int(candidates[0])


def _extract_panel_mask(panel: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    content = panel != bg
    coords = np.argwhere(content)
    assert len(coords) > 0

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop = panel[r0 : r1 + 1, c0 : c1 + 1]
    crop_content = content[r0 : r1 + 1, c0 : c1 + 1]

    colors = np.unique(crop[crop_content])
    assert len(colors) == 1
    color = int(colors[0])
    mask = crop == color
    assert np.array_equal(mask, crop_content)
    return mask, color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the second mask into a block grid by stamping the first mask with the two payload colors.

    Input structure
    - A background grid has four panels separated by three full-height separator columns.
    - The first two panels contain same-size binary masks; the last two contain solid same-size color squares.
    Rule
    - Use the second mask as an outer layout. For each foreground cell in that outer mask, place a tile equal to the first mask, coloring its foreground with the first solid-square color and its background with the second solid-square color; outer background cells become all second solid-square color.
    Key insight
    - The output is the Kronecker product of the two binary masks, with the two right-hand panel colors providing the palette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _unique_dominant_color(g)

    sep_cols = [
        c
        for c in range(w)
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg
    ]
    assert len(sep_cols) == 3
    sep_colors = {int(g[0, c]) for c in sep_cols}
    assert len(sep_colors) == 1

    cuts = [-1, *sep_cols, w]
    spans = [(cuts[i] + 1, cuts[i + 1]) for i in range(4)]
    assert all(c0 < c1 for c0, c1 in spans)

    masks: list[np.ndarray] = []
    colors: list[int] = []
    for c0, c1 in spans:
        mask, color = _extract_panel_mask(g[:, c0:c1], bg)
        masks.append(mask)
        colors.append(color)

    n = masks[0].shape[0]
    assert n > 0
    assert h == n + 2
    assert all(mask.shape == (n, n) for mask in masks)
    assert masks[2].all()
    assert masks[3].all()
    assert colors[2] != colors[3]

    painted = np.kron(masks[1], masks[0])
    out = np.full((n * n, n * n), colors[3], dtype=int)
    out[painted] = colors[2]
    assert out.shape == (n * n, n * n)
    return out

# --------------- generator (att1) ---------------

# Canvas and panel geometry.
MIN_MASK_SIDE = 3
MAX_MASK_SIDE = 5
BORDER = 1
PANEL_COUNT = 4
SEPARATOR_COLOR = 5
BACKGROUND_COLOR = 0
CLIP_LAST_COL_PROB = 0.2

# Binary mask sampling.
MASK_FILL_MIN = 0.4
MASK_FILL_MAX = 0.8
MIN_ACTUAL_FILL = 0.3
MAX_ACTUAL_FILL = 1.0

# Palette and retry limits.
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 300


def _sample_mask(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an n x n random mask whose actual fill passes the solver-facing bounds."""
    while True:
        fill = rng.uniform(MASK_FILL_MIN, MASK_FILL_MAX)
        mask = rng.random((n, n)) < fill
        actual_fill = mask.sum() / (n * n)
        if MIN_ACTUAL_FILL < actual_fill < MAX_ACTUAL_FILL:
            return mask


def _grid_width(n: int, rng: np.random.Generator) -> int:
    panel_width = n + 2 * BORDER
    width = PANEL_COUNT * panel_width + (PANEL_COUNT - 1)
    if rng.random() < CLIP_LAST_COL_PROB:
        width -= 1
    return width


def _separator_columns(n: int) -> tuple[int, int, int]:
    panel_width = n + 2 * BORDER
    return panel_width, 2 * panel_width + 1, 3 * panel_width + 2


def _paint_mask_panel(grid: np.ndarray, mask: np.ndarray, color: int, c_start: int) -> None:
    panel = grid[BORDER : BORDER + mask.shape[0], c_start : c_start + mask.shape[1]]
    panel[mask] = color


def _paint_solid_panel(grid: np.ndarray, n: int, color: int, c_start: int) -> None:
    grid[BORDER : BORDER + n, c_start : c_start + n] = color


def _build_input_grid(
    n: int,
    mask0: np.ndarray,
    mask1: np.ndarray,
    color0: int,
    color1: int,
    color_fg: int,
    color_bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the four-panel input with optional right-border clipping."""
    h = n + 2 * BORDER
    w = _grid_width(n, rng)
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)

    sep1, sep2, sep3 = _separator_columns(n)
    grid[:, sep1] = SEPARATOR_COLOR
    grid[:, sep2] = SEPARATOR_COLOR
    grid[:, sep3] = SEPARATOR_COLOR

    _paint_mask_panel(grid, mask0, color0, BORDER)
    _paint_mask_panel(grid, mask1, color1, sep1 + 1 + BORDER)
    _paint_solid_panel(grid, n, color_fg, sep2 + 1 + BORDER)
    _paint_solid_panel(grid, n, color_bg, sep3 + 1 + BORDER)
    return grid


def _expected_output(n: int, mask0: np.ndarray, mask1: np.ndarray, color_fg: int, color_bg: int) -> np.ndarray:
    painted = np.kron(mask1, mask0)
    expected = np.full((n * n, n * n), color_bg, dtype=int)
    expected[painted] = color_fg
    return expected


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n = rng.integers(MIN_MASK_SIDE, MAX_MASK_SIDE + 1)
    mask0 = _sample_mask(rng, n)
    mask1 = _sample_mask(rng, n)

    chosen = rng.choice(AVAILABLE_COLORS, size=4, replace=False)
    color0, color1, color_fg, color_bg = int(chosen[0]), int(chosen[1]), int(chosen[2]), int(chosen[3])
    grid = _build_input_grid(n, mask0, mask1, color0, color1, color_fg, color_bg, rng)

    # Verify with solver
    try:
        out = main(grid)
        expected = _expected_output(n, mask0, mask1, color_fg, color_bg)
        if np.array_equal(out, expected):
            return grid
    except Exception:
        pass

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
