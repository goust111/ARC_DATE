import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the solid horizontal divider and overlay the equal-size panels above and below it.

    Input structure:
    - The grid has a dominant background color.
    - A single non-background solid row spans the full width and separates two equal-height panels.
    - The separator color does not appear inside either panel, and panel foreground pixels do not collide at the same local coordinate.

    Rule:
    - Delete the separator row.
    - Align the bottom panel over the top panel by local row and column.
    - Preserve every non-background pixel from both panels on the merged output canvas.

    Key insight:
    - The full-width row is not output content; it is a divider telling us to superimpose the two same-size drawings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 1, "expected two panels separated by a row"

    bg = get_bg(g)
    divider_rows = [
        row
        for row in range(height)
        if np.all(g[row, :] == int(g[row, 0])) and int(g[row, 0]) != bg
    ]
    assert len(divider_rows) == 1, "expected exactly one full-width non-background divider row"

    divider = divider_rows[0]
    top = g[:divider, :]
    bottom = g[divider + 1 :, :]
    assert top.shape == bottom.shape, "divider must split equal-size panels"
    assert top.size > 0, "panels must be nonempty"

    separator = int(g[divider, 0])
    assert not np.any(top == separator), "separator color must not appear in the top panel"
    assert not np.any(bottom == separator), "separator color must not appear in the bottom panel"

    top_fg = top != bg
    bottom_fg = bottom != bg
    assert not np.any(top_fg & bottom_fg), "panel foreground pixels must not overlap"

    out = np.full(top.shape, bg, dtype=int)
    out[top_fg] = top[top_fg]
    out[bottom_fg] = bottom[bottom_fg]
    return out

# --------------- generator (o5x) ---------------

# Canvas/object size bounds.
BG_COLOR, PANEL_H_RANGE, WIDTH_RANGE = 0, (3, 8), (7, 16)
# Foreground color/count bounds.
COLOR_POOL, TOTAL_FG_RANGE, MIN_PANEL_FG = np.arange(1, 10), (6, 12), 3
MAX_GENERATION_ATTEMPTS = 300  # Generation retry limit.


def _sample_foreground(panel_h: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint flat local positions for the top and bottom panels."""
    capacity = panel_h * width
    total_fg = int(np.random.randint(TOTAL_FG_RANGE[0], min(TOTAL_FG_RANGE[1], capacity) + 1))
    top_count = int(np.random.randint(MIN_PANEL_FG, total_fg - MIN_PANEL_FG + 1))
    chosen = np.random.choice(capacity, size=total_fg, replace=False)
    return chosen[:top_count], chosen[top_count:]


def _build_example(panel_h: int, width: int, colors, top_flat, bottom_flat):
    """Render input and expected merged output from local foreground positions."""
    color_top, color_bot, color_div = colors
    top_rr, top_cc = np.divmod(top_flat, width)
    bottom_rr, bottom_cc = np.divmod(bottom_flat, width)

    out = np.full((panel_h, width), BG_COLOR, dtype=int)
    out[top_rr, top_cc] = color_top
    out[bottom_rr, bottom_cc] = color_bot

    inp = np.full((2 * panel_h + 1, width), BG_COLOR, dtype=int)
    inp[top_rr, top_cc] = color_top
    inp[panel_h, :] = color_div
    inp[panel_h + 1 + bottom_rr, bottom_cc] = color_bot
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    panel_h = int(np.random.randint(*PANEL_H_RANGE))
    width = int(np.random.randint(*WIDTH_RANGE))
    colors = tuple(map(int, np.random.choice(COLOR_POOL, size=3, replace=False)))
    top_flat, bottom_flat = _sample_foreground(panel_h, width)
    inp, out = _build_example(panel_h, width, colors, top_flat, bottom_flat)

    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise AssertionError("failed to generate a valid e98196ab input")
