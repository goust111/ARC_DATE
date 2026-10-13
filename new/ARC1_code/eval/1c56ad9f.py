import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg, paste
import random


def _wave_shift(distance_from_bottom: int) -> int:
    if distance_from_bottom % 2 == 0:
        return 0
    return -1 if (distance_from_bottom // 2) % 2 == 0 else 1


def main(grid: np.ndarray) -> np.ndarray:
    """Bend the single foreground lattice into a one-pixel horizontal wave anchored at its bottom row.

    Input structure
    - The grid has a dominant background and exactly one foreground color.
    - All foreground pixels form one 4-connected lattice-like object.
    - Each row inside the foreground object's bounding box contains foreground pixels.

    Rule
    - Keep the bottom row of the object's bounding box fixed.
    - Moving upward from that bottom row, leave every other row fixed.
    - Shift the intervening rows by one pixel, alternating left, right, left, right.

    Key insight
    - The phase of the wave is determined from the bottom of the object, not from the top.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    mask = g != bg
    assert len(components_mask(mask, connectivity=4)) == 1
    box = bbox(mask)
    assert box is not None
    r0, _, r1, _ = box

    out = g.copy()
    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0
        shift = _wave_shift(r1 - r)
        if shift == 0:
            continue

        new_cols = cols + shift
        assert int(new_cols.min()) >= 0
        assert int(new_cols.max()) < g.shape[1]
        out[r, cols] = bg
        assert bool(np.all(out[r, new_cols] == bg))
        out[r, new_cols] = fg

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and placement margins.
GEN_GRID_ROW_BOUNDS, GEN_GRID_COL_BOUNDS = (13, 15), (14, 15)
GEN_BG, GEN_NO_SHIFT, GEN_FG_BOUNDS = 0, 0, (1, 9)
GEN_VERTICAL_MARGIN, GEN_HORIZONTAL_MARGIN = 1, 2

# Lattice count, gap, and object-size bounds.
GEN_H_BAR_COUNTS, GEN_V_GROUP_COUNTS = range(2, 5), range(2, 4)
GEN_CELL_HEIGHTS, GEN_CELL_WIDTHS = range(2, 12), range(1, 5)
GEN_OBJECT_HEIGHT_MIN, GEN_OBJECT_HEIGHT_MAX = 7, 13
GEN_OBJECT_WIDTH_MIN, GEN_OBJECT_WIDTH_MAX = 5, 9
GEN_GROUP_WIDTH_PATTERNS = {2: ((1, 1),), 3: ((1, 1, 1), (1, 2, 1))}

# Validation and retry limits.
GEN_REQUIRED_COMPONENTS, GEN_MAX_ATTEMPTS = 1, 300


def _gen_shift_for_distance(distance_from_bottom):
    if distance_from_bottom % 2 == 0:
        return GEN_NO_SHIFT
    return -1 if (distance_from_bottom // 2) % 2 == 0 else 1


def _gen_horizontal_options(rows):
    by_count = {}
    for n_hbars in GEN_H_BAR_COUNTS:
        opts = []
        for cell_height in GEN_CELL_HEIGHTS:
            height = n_hbars + (n_hbars - 1) * cell_height
            if GEN_OBJECT_HEIGHT_MIN <= height <= GEN_OBJECT_HEIGHT_MAX and height <= rows - 2 * GEN_VERTICAL_MARGIN:
                opts.append((cell_height, height))
        if opts:
            by_count[n_hbars] = opts
    return by_count


def _gen_vertical_options(cols):
    by_count = {}
    for n_vgroups in GEN_V_GROUP_COUNTS:
        opts = []
        for cell_width in GEN_CELL_WIDTHS:
            for group_widths in GEN_GROUP_WIDTH_PATTERNS[n_vgroups]:
                width = sum(group_widths) + (n_vgroups - 1) * cell_width
                if GEN_OBJECT_WIDTH_MIN <= width <= GEN_OBJECT_WIDTH_MAX and width <= cols - 2 * GEN_HORIZONTAL_MARGIN:
                    opts.append((cell_width, width, group_widths))
        if opts:
            by_count[n_vgroups] = opts
    return by_count


def _gen_lattice_params(rows, cols):
    """Choose one valid lattice geometry for the selected canvas size."""
    horizontal_options = _gen_horizontal_options(rows)
    vertical_options = _gen_vertical_options(cols)
    if not horizontal_options or not vertical_options: return None

    n_hbars = random.choice(list(horizontal_options))
    cell_height, height = random.choice(horizontal_options[n_hbars])

    n_vgroups = random.choice(list(vertical_options))
    cell_width, width, group_widths = random.choice(vertical_options[n_vgroups])

    return height, width, cell_height, cell_width, group_widths


def _gen_lattice(height, width, cell_height, group_widths, cell_width, fg):
    patch = np.zeros((height, width), dtype=int)

    patch[:: cell_height + 1, :] = fg

    c = 0
    for group_width in group_widths:
        patch[:, c : c + group_width] = fg
        c += group_width + cell_width

    return patch


def _gen_apply_wave(grid):
    mask = grid != GEN_BG
    box = bbox(mask)
    if box is None: return None

    r0, _, r1, _ = box
    fg = int(grid[mask][0])
    out = grid.copy()

    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0: return None
        shift = _gen_shift_for_distance(r1 - r)
        if shift == 0:
            continue
        new_cols = cols + shift
        if int(new_cols.min()) < 0 or int(new_cols.max()) >= grid.shape[1]: return None
        out[r, cols] = GEN_BG
        if not np.all(out[r, new_cols] == GEN_BG): return None
        out[r, new_cols] = fg

    return out


def _gen_place_lattice(rows, cols, patch):
    grid = np.zeros((rows, cols), dtype=int)
    height, width = patch.shape

    r0 = random.randint(GEN_VERTICAL_MARGIN, rows - height - GEN_VERTICAL_MARGIN)
    c0 = random.randint(GEN_HORIZONTAL_MARGIN, cols - width - GEN_HORIZONTAL_MARGIN)
    paste(grid, patch, r0, c0, transparent=GEN_BG)
    return grid


def _gen_matches_oracle(grid):
    """Return True when the constructed input satisfies the solver's wave rule."""
    if len(components_mask(grid != GEN_BG, connectivity=4)) != GEN_REQUIRED_COMPONENTS:
        return False

    expected = _gen_apply_wave(grid)
    if expected is None: return False

    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt():
    rows = random.randint(*GEN_GRID_ROW_BOUNDS)
    cols = random.randint(*GEN_GRID_COL_BOUNDS)
    fg = random.randint(*GEN_FG_BOUNDS)
    params = _gen_lattice_params(rows, cols)
    if params is None: return None

    height, width, cell_height, cell_width, group_widths = params
    patch = _gen_lattice(height, width, cell_height, group_widths, cell_width, fg)
    grid = _gen_place_lattice(rows, cols, patch)

    if not _gen_matches_oracle(grid):
        return None

    return grid


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 1c56ad9f input")
