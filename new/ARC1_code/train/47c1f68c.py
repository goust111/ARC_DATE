import numpy as np

from arc_utils import colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Remove the central cross and mirror the upper-left mask into all four quadrants using the cross color.

    Input structure
    A square grid has a background, one full horizontal and vertical line of the
    same color crossing at the center, and a non-cross foreground mask entirely
    in the upper-left quadrant.

    Rule
    Delete the cross row and column, recolor the upper-left mask to the cross
    color, and place reflected copies in the other three quadrants.

    Key insight
    The cross is a divider and palette source; after it is removed, the lone
    upper-left mask supplies the shape for a four-way mirror pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    assert h == w and h % 2 == 1

    crosses = []
    for color in colors_present(g, exclude={bg}):
        full_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        crosses.extend((color, r, c) for r in full_rows for c in full_cols)

    assert len(crosses) == 1
    cross_color, cross_r, cross_c = crosses[0]
    assert h == 2 * cross_r + 1 and w == 2 * cross_c + 1

    source_mask = (g != bg) & (g != cross_color)
    source_region = np.zeros_like(source_mask, dtype=bool)
    source_region[:cross_r, :cross_c] = True
    assert np.any(source_mask)
    assert np.array_equal(source_mask, source_mask & source_region)

    quadrant_mask = source_mask[:cross_r, :cross_c]
    out = np.full((h - 1, w - 1), bg, dtype=int)
    out[:cross_r, :cross_c][quadrant_mask] = cross_color
    out[:cross_r, cross_c:][np.fliplr(quadrant_mask)] = cross_color
    out[cross_r:, :cross_c][np.flipud(quadrant_mask)] = cross_color
    out[cross_r:, cross_c:][np.flipud(np.fliplr(quadrant_mask))] = cross_color
    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
BACKGROUND_COLOR = 0
CANVAS_SIZES = [7, 9, 11, 13]
NON_BACKGROUND_COLORS = np.arange(1, 10)
# Mask density and retry bounds.
MASK_PIXELS_BY_Q = {3: [4], 4: [4, 5, 6, 7], 5: [5, 6, 7], 6: [7]}
RETRY_LIMIT = 300

def _choose_params() -> tuple[int, int, int, int, int]:
    """Sample grid size, palette, and mask pixel count."""
    n = int(np.random.choice(CANVAS_SIZES))
    q = (n - 1) // 2
    cross_color, mask_color = [
        int(x)
        for x in np.random.choice(NON_BACKGROUND_COLORS, size=2, replace=False)
    ]
    k = int(np.random.choice(MASK_PIXELS_BY_Q[q]))
    return n, q, cross_color, mask_color, k

def _gen_quadrant_mask(q: int, k: int) -> np.ndarray:
    """Return a q x q asymmetric mask with k sampled pixels."""
    mask = np.zeros((q, q), dtype=bool)
    corners = [(0, 0), (0, q - 1), (q - 1, 0), (q - 1, q - 1)]
    anchor = corners[int(np.random.randint(len(corners)))]
    mask[anchor] = True

    ar, ac = anchor
    blocked = {(ar, q - 1 - ac), (q - 1 - ar, ac)}
    candidates = [
        (r, c)
        for r in range(q)
        for c in range(q)
        if (r, c) != anchor and (r, c) not in blocked
    ]
    chosen = np.random.choice(len(candidates), size=k - 1, replace=False)
    for idx in np.atleast_1d(chosen):
        r, c = candidates[int(idx)]
        mask[r, c] = True
    return mask

def _render_pair(n, q, mask, cross_color, mask_color):
    """Build matching input and oracle output for one quadrant mask."""
    expected = np.full((n - 1, n - 1), BACKGROUND_COLOR, dtype=int)
    expected[:q, :q][mask] = cross_color
    expected[:q, q:][np.fliplr(mask)] = cross_color
    expected[q:, :q][np.flipud(mask)] = cross_color
    expected[q:, q:][np.flipud(np.fliplr(mask))] = cross_color
    inp = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    inp[q, :] = cross_color
    inp[:, q] = cross_color
    inp[:q, :q][mask] = mask_color
    return inp, expected

def _gen_attempt() -> np.ndarray | None:
    n, q, cross_color, mask_color, k = _choose_params()
    quadrant_mask = _gen_quadrant_mask(q, k)
    inp, expected = _render_pair(n, q, quadrant_mask, cross_color, mask_color)
    return inp if np.array_equal(main(inp), expected) else None

def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {RETRY_LIMIT} attempts")
