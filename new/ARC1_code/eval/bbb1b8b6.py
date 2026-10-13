import numpy as np

from arc_utils import colors_present, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the right side-panel mask onto the left panel only when the two masks do not overlap.

    Input structure:
    - The input is split by a solid non-background separator column into two equal-size panels.
    - The side panels share one background color.
    - Each side panel contains exactly one foreground color, forming a positional mask.

    Rule:
    - Remove the separator column and use the left panel as the output canvas.
    - If any foreground cell in the right panel coincides with a foreground cell in the left panel, ignore the right panel.
    - Otherwise, copy the right panel foreground cells into their corresponding positions on the left panel.

    Key insight:
    - The right mask is accepted only when it fits entirely into the empty cells of the left mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1 and width >= 3, "expected two equal panels plus a separator"

    bg = get_bg(g)
    sep_col = width // 2
    sep_color = int(g[0, sep_col])
    assert sep_color != bg, "separator must be non-background"
    assert np.all(g[:, sep_col] == sep_color), "expected a solid separator column"

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape == (height, sep_col), "expected equal-size side panels"

    left_colors = colors_present(left, exclude={bg})
    right_colors = colors_present(right, exclude={bg})
    assert len(left_colors) == 1 and len(right_colors) == 1, "each panel must have one foreground color"
    assert sep_color not in left_colors and sep_color not in right_colors, "separator color must not appear in panels"

    left_mask = left != bg
    right_mask = right != bg
    out = left.copy()
    if not np.any(left_mask & right_mask):
        out[right_mask] = right[right_mask]
    return out

# --------------- generator (o5x) ---------------
# Panel geometry and colors.
PANEL_H, PANEL_W, LEFT_COLOR, SEPARATOR_COLOR = 4, 4, 1, 5
PANEL_CELLS = PANEL_H * PANEL_W
RIGHT_COLORS = (2, 3, 6, 7)

# Mask/component constraints and case mix.
COMPONENT_MIN, COMPONENT_MAX = 1, 2
LEFT_FG_MIN, LEFT_FG_MAX = 7, 12
RIGHT_FG_MIN, RIGHT_FG_MAX = 4, 9
COMPLEMENT_CASE_PROB = 0.55

# Overlap-case background filter and retry policy.
OVERLAP_BG_TOTAL, OVERLAP_BG_FLOOR = 2 * PANEL_CELLS, 4
GEN_RETRY_LIMIT = 300

def _gen_component_count_ok(mask):
    count = len(components_mask(mask, connectivity=4))
    return COMPONENT_MIN <= count <= COMPONENT_MAX

def _gen_mask_catalog():
    """Return valid masks grouped by foreground count, plus complement-safe masks."""
    cache = getattr(_gen_mask_catalog, "_cache", None)
    if cache is not None:
        return cache

    by_count = {k: [] for k in range(PANEL_CELLS + 1)}
    complement_ok = {k: [] for k in by_count}
    for bits in range(1, 1 << PANEL_CELLS):
        flat = [bool((bits >> i) & 1) for i in range(PANEL_CELLS)]
        mask = np.array(flat, dtype=bool).reshape(PANEL_H, PANEL_W)
        count = int(mask.sum())
        if _gen_component_count_ok(mask):
            by_count[count].append(mask)

    for count, masks in by_count.items():
        complement_ok[count] = [m for m in masks if _gen_component_count_ok(~m)]

    cache = by_count, complement_ok
    _gen_mask_catalog._cache = cache
    return cache

def _gen_choice(seq):
    return seq[int(np.random.randint(0, len(seq)))]

def _gen_panel(mask, color):
    panel = np.zeros((PANEL_H, PANEL_W), dtype=int)
    panel[mask] = int(color)
    return panel

def _gen_case_panels(left_mask, right_mask, right_color, applies_right):
    """Build both side panels and the oracle-expected output panel."""
    left_panel = _gen_panel(left_mask, LEFT_COLOR)
    right_panel = _gen_panel(right_mask, right_color)
    expected = left_panel.copy()
    if applies_right:
        expected[right_mask] = right_color
    return left_panel, right_panel, expected

def _gen_complement_case(complement_ok, right_color):
    """Sample a complement case where the right mask fills every left background cell."""
    left_count = int(np.random.randint(LEFT_FG_MIN, LEFT_FG_MAX + 1))
    pool = complement_ok[left_count]
    if not pool:
        return None
    left_mask = _gen_choice(pool).copy()
    return _gen_case_panels(left_mask, ~left_mask, right_color, applies_right=True)

def _gen_overlap_case(by_count, right_color):
    """Sample an overlap case where the solver should ignore the right mask."""
    left_count = int(np.random.randint(LEFT_FG_MIN, LEFT_FG_MAX + 1))
    left_pool = by_count[left_count]
    if not left_pool:
        return None
    left_mask = _gen_choice(left_pool).copy()

    right_options = []
    for right_count in range(RIGHT_FG_MIN, RIGHT_FG_MAX + 1):
        bg_count = OVERLAP_BG_TOTAL - left_count - right_count
        if bg_count <= max(left_count, right_count, OVERLAP_BG_FLOOR):
            continue
        candidates = [m for m in by_count[right_count] if np.any(left_mask & m)]
        if candidates:
            right_options.append(candidates)
    if not right_options:
        return None
    right_mask = _gen_choice(_gen_choice(right_options)).copy()
    return _gen_case_panels(left_mask, right_mask, right_color, applies_right=False)

def _gen_attempt():
    by_count, complement_ok = _gen_mask_catalog()
    right_color = int(_gen_choice(RIGHT_COLORS))
    if np.random.random() < COMPLEMENT_CASE_PROB:
        case = _gen_complement_case(complement_ok, right_color)
    else:
        case = _gen_overlap_case(by_count, right_color)
    if case is None:
        return None

    left_panel, right_panel, expected = case
    inp = np.zeros((PANEL_H, PANEL_W * 2 + 1), dtype=int)
    inp[:, :PANEL_W] = left_panel
    inp[:, PANEL_W] = SEPARATOR_COLOR
    inp[:, PANEL_W + 1 :] = right_panel
    return inp if np.array_equal(main(inp), expected) else None

def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bbb1b8b6 input")
