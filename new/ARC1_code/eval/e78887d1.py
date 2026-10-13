import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Emit the next 3-row band in a cyclic table of colored 3x3 sprite masks.

    Input structure
    The input is a 3x3 sprite lattice: sprite rows are separated by full
    background rows, and sprite columns are separated by full background
    columns.  Each column keeps one color across all sprite rows.

    Rule
    Ignoring color, each sprite row is the previous sprite row shifted one slot
    left.  The output is the next shifted sprite row, recolored by the fixed
    colors of the output columns.

    Key insight
    The colors mark the columns, while the uncolored 3x3 masks form the cyclic
    sequence; after m input sprite rows, column j receives mask (j + m) modulo
    the number of columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    h, w = g.shape
    assert (h - 1) % 4 == 0
    assert (w + 1) % 4 == 0

    n_rows = (h - 1) // 4
    n_cols = (w + 1) // 4
    assert n_rows >= 1
    assert n_cols >= 2
    assert np.all(g[0::4, :] == bg)
    assert np.all(g[:, 3::4] == bg)

    masks: list[list[np.ndarray]] = []
    col_colors: list[int] = []
    for br in range(n_rows):
        row_masks = []
        r0 = 1 + 4 * br
        for bc in range(n_cols):
            c0 = 4 * bc
            block = g[r0 : r0 + 3, c0 : c0 + 3]
            colors = set(map(int, block[block != bg]))
            assert len(colors) == 1
            color = colors.pop()
            if br == 0:
                col_colors.append(color)
            else:
                assert color == col_colors[bc]
            row_masks.append(block != bg)
        masks.append(row_masks)

    for br in range(n_rows):
        for bc in range(n_cols):
            expected = masks[0][(bc + br) % n_cols]
            assert np.array_equal(masks[br][bc], expected)

    out = np.full((3, w), bg, dtype=int)
    for bc, color in enumerate(col_colors):
        c0 = 4 * bc
        mask = masks[0][(bc + n_rows) % n_cols]
        tile = out[:, c0 : c0 + 3]
        tile[mask] = color

    return out

# --------------- generator (o5x) ---------------

# Lattice dimensions and separator cadence.
GEN_N_COLS_MIN, GEN_N_COLS_MAX_EXCLUSIVE = 3, 5
GEN_N_ROWS_MIN, GEN_N_ROWS_MAX_EXCLUSIVE = 1, 4
GEN_SPRITE_SIZE, GEN_STRIDE, GEN_BG = 3, 4, 0

# Sprite mask sampling.
GEN_MASK_DENSITY_MIN, GEN_MASK_DENSITY_MAX_EXCLUSIVE = 4, 8
GEN_MASK_RETRIES = 100

# Palette and generation retries.
GEN_COLORS = np.array([1, 2, 3, 5])
GEN_ATTEMPT_RETRIES = 300


def _gen_random_mask():
    density = int(np.random.randint(GEN_MASK_DENSITY_MIN, GEN_MASK_DENSITY_MAX_EXCLUSIVE))
    mask = np.zeros((GEN_SPRITE_SIZE, GEN_SPRITE_SIZE), dtype=bool)
    mask.flat[np.random.choice(mask.size, size=density, replace=False)] = True
    return mask


def _gen_distinct_masks(n_masks):
    """Return n distinct 3x3 masks, or None if retries are exhausted."""
    masks = []
    seen = set()
    for _ in range(GEN_MASK_RETRIES):
        mask = _gen_random_mask()
        key = tuple(map(bool, mask.ravel()))
        if key in seen:
            continue
        masks.append(mask)
        seen.add(key)
        if len(masks) == n_masks:
            return masks
    return None


def _gen_cyclic_band(width, n_cols, base_masks, colors, band_index):
    band = np.full((GEN_SPRITE_SIZE, width), GEN_BG, dtype=int)
    for col in range(n_cols):
        c0 = GEN_STRIDE * col
        mask = base_masks[(col + band_index) % n_cols]
        tile = band[:, c0 : c0 + GEN_SPRITE_SIZE]
        tile[mask] = int(colors[col])
    return band


def _gen_input_grid(height, width, n_rows, n_cols, base_masks, colors):
    grid = np.full((height, width), GEN_BG, dtype=int)
    for band_r in range(n_rows):
        r0 = 1 + GEN_STRIDE * band_r
        grid[r0 : r0 + GEN_SPRITE_SIZE, :] = _gen_cyclic_band(
            width, n_cols, base_masks, colors, band_r
        )
    return grid


def _gen_attempt():
    n_cols = int(np.random.randint(GEN_N_COLS_MIN, GEN_N_COLS_MAX_EXCLUSIVE))
    n_rows = int(np.random.randint(GEN_N_ROWS_MIN, GEN_N_ROWS_MAX_EXCLUSIVE))
    colors = np.random.permutation(GEN_COLORS)[:n_cols]
    base_masks = _gen_distinct_masks(n_cols)
    if base_masks is None:
        return None

    height = GEN_STRIDE * n_rows + 1
    width = GEN_STRIDE * n_cols - 1
    inp = _gen_input_grid(height, width, n_rows, n_cols, base_masks, colors)
    expected = _gen_cyclic_band(width, n_cols, base_masks, colors, n_rows)
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input():
    for _ in range(GEN_ATTEMPT_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
