import numpy as np

from arc_utils import find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Stack all colored horizontal segments by increasing length, right-aligned above the solid bottom stripe.

    Input structure
    The grid has a dominant background, one full-width non-background stripe on
    the bottom row, and one or more 4-connected monochrome foreground objects
    above it. Each non-stripe object is a one-cell-thick horizontal segment.

    Rule
    Clear the segment objects from their original positions. Sort the segments
    by length from shortest to longest, then place them in consecutive rows
    immediately above the bottom stripe, with each segment right-aligned to the
    grid's right edge. Keep the bottom stripe unchanged.

    Key insight
    The scattered segments are not moved by their original row or color; their
    lengths define a stack order against the fixed bottom stripe.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    stripe_rows = [
        r for r in range(h)
        if np.unique(g[r]).size == 1 and int(g[r, 0]) != bg
    ]
    assert len(stripe_rows) == 1, "expected one full-width non-background stripe"
    stripe_row = int(stripe_rows[0])
    assert stripe_row == h - 1, "stripe must be on the bottom row"
    stripe_color = int(g[stripe_row, 0])

    bars = []
    found_stripe = False
    for component in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = component["bbox"]
        color = int(component["color"])
        area = int(component["area"])

        is_stripe = (
            r0 == stripe_row
            and r1 == stripe_row
            and c0 == 0
            and c1 == w - 1
            and area == w
        )
        if is_stripe:
            assert color == stripe_color
            found_stripe = True
            continue

        assert r0 < stripe_row, "moving segments must be above the stripe"
        assert r0 == r1, "moving object must be one cell high"
        assert area == c1 - c0 + 1, "moving segment must be contiguous"
        bars.append((area, color))

    assert found_stripe, "bottom stripe component must be present"
    assert bars, "expected at least one moving segment"
    assert len(bars) <= stripe_row, "segments must fit above the stripe"
    lengths = [length for length, _color in bars]
    assert len(set(lengths)) == len(lengths), "segment lengths must give a unique order"

    out = np.full_like(g, bg)
    out[stripe_row, :] = stripe_color
    for i, (length, color) in enumerate(sorted(bars)):
        r = stripe_row - len(bars) + i
        out[r, w - length : w] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas/object bounds.
MIN_SEGMENTS, MAX_SEGMENTS = 2, 7
WIDTH_EXTRA_COLUMNS, BOTTOM_STRIPE_ROWS = 1, 1
HEIGHT_MAX_SEGMENT_MULTIPLIER = 2
# Colors.
BACKGROUND_COLOR, STRIPE_COLOR = 0, 8
SEGMENT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
# Scatter-row spacing and retry budget.
MIN_ROW_GAP, MAX_ROW_GAP = 1, 2
STRIPE_ADJACENT_EMPTY_ROWS, MAX_GENERATION_ATTEMPTS = 1, 300

def _empty_grid_with_stripe(h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    grid[h - BOTTOM_STRIPE_ROWS, :] = STRIPE_COLOR
    return grid

def _gen_segment_rows(n_segs: int, h: int) -> list[int]:
    """Sample unique rows with 1-2 gaps, reserving the stripe-adjacent row when possible."""
    stripe_row = h - BOTTOM_STRIPE_ROWS
    max_last = stripe_row - STRIPE_ADJACENT_EMPTY_ROWS - 1 if stripe_row > n_segs else stripe_row - 1
    if max_last < n_segs - 1:
        max_last = stripe_row - 1
    max_extra_gaps = min(n_segs - 1, max_last - (n_segs - 1) * MIN_ROW_GAP)
    n_two_gaps = random.randint(0, max_extra_gaps)
    gaps = [MAX_ROW_GAP] * n_two_gaps + [MIN_ROW_GAP] * (n_segs - 1 - n_two_gaps)
    random.shuffle(gaps)
    span = sum(gaps)
    start = random.randint(0, max_last - span)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows

def _gen_attempt() -> np.ndarray | None:
    n_segs = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    w = n_segs + WIDTH_EXTRA_COLUMNS
    h = random.randint(n_segs + BOTTOM_STRIPE_ROWS, HEIGHT_MAX_SEGMENT_MULTIPLIER * n_segs + BOTTOM_STRIPE_ROWS)

    colors = random.sample(SEGMENT_COLORS, n_segs)
    out = _empty_grid_with_stripe(h, w)

    bars = []
    for length, color in zip(range(1, n_segs + 1), colors):
        r = h - BOTTOM_STRIPE_ROWS - n_segs + length - 1
        out[r, w - length : w] = color
        bars.append((length, color))

    inp = _empty_grid_with_stripe(h, w)

    rows = _gen_segment_rows(n_segs, h)
    random.shuffle(bars)
    for row, (length, color) in zip(rows, bars):
        c0 = random.randint(0, w - length)
        inp[row, c0 : c0 + length] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid beb8660c input")
