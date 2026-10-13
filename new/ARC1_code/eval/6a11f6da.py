import numpy as np

from arc_utils import colors_present, find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite three stacked same-size panels into one panel, with the bottom panel painted last.

    Input structure
    - The input is three equal-height panels stacked vertically on a common background.
    - Each panel contains exactly one foreground color.
    - Corresponding cells across the three panels occupy the same output position.

    Rule
    - Start with the middle panel as the base.
    - Copy every non-background pixel from the top panel onto the same position.
    - Copy every non-background pixel from the bottom panel onto the same position, overwriting both earlier layers.

    Key insight
    - Cell conflicts are resolved by panel position/layer role: middle first, then top, then bottom, not by numeric color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H % 3 == 0
    panel_h = H // 3
    assert panel_h > 0 and W > 0

    bg = get_bg(g)
    top = g[:panel_h, :]
    middle = g[panel_h : 2 * panel_h, :]
    bottom = g[2 * panel_h :, :]

    panel_colors = [colors_present(panel, exclude={bg}) for panel in (top, middle, bottom)]
    assert all(len(colors) == 1 for colors in panel_colors)
    assert len({colors[0] for colors in panel_colors}) == 3

    out = middle.copy()
    top_mask = top != bg
    bottom_mask = bottom != bg
    out[top_mask] = top[top_mask]
    out[bottom_mask] = bottom[bottom_mask]
    return out

# --------------- generator (o5x) ---------------
# Size, palette, and fill bounds.
PANEL_H_RANGE, WIDTH_RANGE = (3, 10), (3, 10)
FOREGROUND_COLORS = range(1, 10)
OUTPUT_FILL_PCT, PANEL_FILL_PCT = (85, 96), (36, 72)
# Structural and retry limits.
MAX_PANEL_COMPONENTS, GENERATION_RETRIES = 8, 300

def _pct_bounds(n, pct_range):
    return (pct_range[0] * n + 99) // 100, (pct_range[1] * n) // 100

def _gen_count_options(n, min_panel, max_panel):
    out_lo, out_hi = _pct_bounds(n, OUTPUT_FILL_PCT)
    by_filled = {}
    for filled in range(out_lo, out_hi + 1):
        triples = []
        for bot_count in range(min_panel, max_panel + 1):
            rem = filled - bot_count
            for top_count in range(1, min(max_panel - 1, rem - 1) + 1):
                mid_count = rem - top_count
                if 1 <= mid_count <= max_panel - 1:
                    triples.append((top_count, mid_count, bot_count))
        if triples:
            by_filled[filled] = triples
    return by_filled

def _sample_visible_plan(n, min_panel, max_panel):
    count_options = _gen_count_options(n, min_panel, max_panel)
    if not count_options:
        return None
    filled = random.choice(list(count_options))
    top_count, mid_count, bot_count = random.choice(count_options[filled])
    colors = random.sample(FOREGROUND_COLORS, 3)
    flat = list(range(n))
    random.shuffle(flat)
    bot_idx = flat[:bot_count]
    top_idx = flat[bot_count : bot_count + top_count]
    mid_idx = flat[bot_count + top_count : bot_count + top_count + mid_count]
    return colors, (top_idx, mid_idx, bot_idx)

def _render_visible_layers(panel_h, width, colors, indices):
    top_idx, mid_idx, bot_idx = indices
    top, middle, bottom = [np.zeros((panel_h, width), dtype=int) for _ in range(3)]
    for panel, idx, color in (
        (top, top_idx, colors[0]),
        (middle, mid_idx, colors[1]),
        (bottom, bot_idx, colors[2]),
    ):
        panel.flat[idx] = color
    out = top + middle + bottom
    return out, top, middle, bottom

def _add_hidden_overlap(top, middle, colors, indices, min_panel, max_panel):
    c_top, c_mid, _ = colors
    top_idx, mid_idx, bot_idx = indices
    top_count, mid_count, bot_count = map(len, (top_idx, mid_idx, bot_idx))
    top_low = max(min_panel, top_count + 1)
    top_high = min(max_panel, top_count + bot_count)
    mid_low = max(min_panel, mid_count + 1)
    mid_high = min(max_panel, mid_count + top_count + bot_count)
    if top_low > top_high or mid_low > mid_high:
        return None

    top_target = random.randint(top_low, top_high)
    top_hidden = random.sample(bot_idx, top_target - top_count)
    top.flat[top_hidden] = c_top
    shared_hidden = random.choice(top_hidden)
    mid_target = random.randint(mid_low, mid_high)
    mid_pool = [i for i in top_idx + bot_idx if i != shared_hidden]
    mid_hidden = [shared_hidden] + random.sample(mid_pool, mid_target - mid_count - 1)
    middle.flat[mid_hidden] = c_mid
    return True

def _panel_component_count(panel):
    return len(find_components((panel != 0).astype(int), bg=0, connectivity=4, monochromatic=True))

def _panels_are_valid(panels, min_panel, max_panel):
    top, middle, bottom = panels
    counts = [int(np.count_nonzero(panel)) for panel in panels]
    overlaps = ((top, middle), (top, bottom), (middle, bottom))
    return (
        all(min_panel <= count <= max_panel for count in counts)
        and all(_panel_component_count(panel) <= MAX_PANEL_COMPONENTS for panel in panels)
        and all(np.any((a != 0) & (b != 0)) for a, b in overlaps)
    )

def _checked_input(panels, out):
    inp = np.vstack(panels)
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None

def _gen_attempt():
    panel_h, width = random.randint(*PANEL_H_RANGE), random.randint(*WIDTH_RANGE)
    n = panel_h * width
    min_panel, max_panel = _pct_bounds(n, PANEL_FILL_PCT)
    plan = _sample_visible_plan(n, min_panel, max_panel)
    if plan is None:
        return None
    colors, indices = plan
    out, top, middle, bottom = _render_visible_layers(panel_h, width, colors, indices)
    if not _add_hidden_overlap(top, middle, colors, indices, min_panel, max_panel):
        return None

    panels = (top, middle, bottom)
    if not _panels_are_valid(panels, min_panel, max_panel):
        return None
    return _checked_input(panels, out)

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
