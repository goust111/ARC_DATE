import numpy as np

from arc_utils import bbox_h, crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Move each solid colored rectangle upward by exactly its own height.

    Input structure
    A black-background grid contains one or more separated monochromatic
    foreground objects. Each object is a solid axis-aligned rectangle, with an
    equally sized empty region immediately above it.

    Rule
    Clear every rectangle from its original position, then paste the same
    rectangle directly above that position, offset upward by the rectangle's
    height.

    Key insight
    The vertical displacement is not shared globally: each rectangle jumps by
    its own bbox height, so its new bottom edge is just above its old top edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps, "expected at least one foreground rectangle"

    moves = []
    target_mask = np.zeros(g.shape, dtype=bool)
    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = bbox_h(comp["bbox"])
        w = c1 - c0 + 1
        assert int(comp["area"]) == h * w, "foreground object must fill its bbox"

        patch = crop_bbox(g, comp["bbox"])
        assert np.all(patch == color), "rectangle must be monochromatic and solid"

        dst_r0 = r0 - h
        assert dst_r0 >= 0, "rectangle must fit above its original position"
        assert np.all(g[dst_r0:r0, c0 : c1 + 1] == bg), "destination must be empty"

        target = target_mask[dst_r0 : dst_r0 + h, c0 : c1 + 1]
        assert not np.any(target), "shifted rectangles must not overlap"
        target[:, :] = True
        moves.append((comp["coords"], patch, dst_r0, c0))

    out = g.copy()
    for coords, _patch, _dst_r0, _c0 in moves:
        out[coords[:, 0], coords[:, 1]] = bg
    for _coords, patch, dst_r0, c0 in moves:
        paste(out, patch, dst_r0, c0)
    return out

# --------------- generator (o5x) ---------------

from itertools import product

# Canvas and palette.
GRID_ROWS, GRID_COLS, BACKGROUND = 15, 15, 0
RECT_COLORS = np.array([1, 2, 4], dtype=int)

# Rectangle sizes, spacing, and retry budget.
RECT_COUNT, GEN_ATTEMPTS = 3, 300
RECT_WIDTH_RANGE = range(1, 6)
RECT_HEIGHT_RANGE = range(1, 8)
RECT_GAP_RANGE = range(1, 4)

LAYOUTS = [
    (widths, gaps, span)
    for widths in product(RECT_WIDTH_RANGE, repeat=RECT_COUNT)
    for gaps in product(RECT_GAP_RANGE, repeat=RECT_COUNT - 1)
    if (span := sum(widths) + sum(gaps)) <= GRID_COLS
]


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_ROWS, GRID_COLS), BACKGROUND, dtype=int)
    expected = np.full_like(grid, BACKGROUND)

    widths, gaps, span = LAYOUTS[np.random.randint(len(LAYOUTS))]
    left = int(np.random.randint(0, GRID_COLS - span + 1))
    starts = [left]
    for width, gap in zip(widths[:-1], gaps):
        starts.append(starts[-1] + width + gap)
    heights = np.random.randint(RECT_HEIGHT_RANGE.start, RECT_HEIGHT_RANGE.stop, size=RECT_COUNT)
    colors = np.random.permutation(RECT_COLORS)

    for h, w, c0, color in zip(heights, widths, starts, colors):
        r0 = GRID_ROWS - int(h)
        patch = np.full((int(h), int(w)), int(color), dtype=int)
        paste(grid, patch, r0, int(c0))
        paste(expected, patch, r0 - int(h), int(c0))

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5521c0d9 input")
