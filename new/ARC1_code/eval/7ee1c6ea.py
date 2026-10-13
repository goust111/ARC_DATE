import numpy as np

from arc_utils import Color, bbox


def _find_rectangular_frame(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(g)):
        mask = g == color
        box = bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        frame = np.zeros_like(mask, dtype=bool)
        frame[r0, c0 : c1 + 1] = True
        frame[r1, c0 : c1 + 1] = True
        frame[r0 : r1 + 1, c0] = True
        frame[r0 : r1 + 1, c1] = True
        if np.array_equal(mask, frame):
            candidates.append((color, box))

    assert len(candidates) == 1
    frame_color, box = candidates[0]
    assert frame_color != Color.BLACK
    return frame_color, box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two colored fills inside the unique rectangular frame while preserving black gaps.

    Input structure:
    - The grid contains one one-cell-thick rectangular frame.
    - The frame interior contains black pixels and exactly two non-black, non-frame colors.
    - Pixels outside the frame may use the same colors but are not part of the editable region.

    Rule:
    - Find the unique rectangular frame and take the region strictly inside it.
    - Within that interior, exchange the two non-black, non-frame colors.
    - Preserve black interior pixels, the frame, and all pixels outside the frame.

    Key insight:
    - The frame, not the outer scattered pixels, selects the only region where the color swap applies.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1) = _find_rectangular_frame(g)

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    interior_colors = set(map(int, np.unique(interior)))
    assert Color.BLACK in interior_colors
    assert frame_color not in interior_colors
    swap_colors = [color for color in interior_colors if color != Color.BLACK]
    assert len(swap_colors) == 2

    first, second = swap_colors
    out = g.copy()
    out_region = out[r0 + 1 : r1, c0 + 1 : c1]
    out_region[interior == first] = second
    out_region[interior == second] = first
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, frame geometry, fill densities, and retry budget.
GRID_SIZE, FRAME_COLOR = 10, 5
SWAP_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
FRAME_SIZE_RANGE = (6, 9)
FRAME_MARGIN = 1
OUTSIDE_BLACK_FRAC_RANGE = (0.25, 0.40)
INTERIOR_BLACK_FRAC_RANGE = (0.15, 0.50)
MAX_GENERATION_ATTEMPTS = 300


def _gen_fill_values(n_cells, colors, black_frac_range):
    black_min_frac, black_max_frac = black_frac_range
    lo = int(np.ceil(black_min_frac * n_cells))
    hi = int(np.floor(black_max_frac * n_cells))
    black_count = int(np.random.randint(lo, hi + 1))
    colored_count = n_cells - black_count
    first_count = int(np.random.randint(1, colored_count))
    vals = [Color.BLACK] * black_count + [colors[0]] * first_count
    vals += [colors[1]] * (colored_count - first_count)
    vals = np.array(vals, dtype=int)
    np.random.shuffle(vals)
    return vals


def _sample_frame_box():
    """Return inclusive frame bounds plus size using the original RNG order."""
    h = int(np.random.randint(*FRAME_SIZE_RANGE))
    w = int(np.random.randint(*FRAME_SIZE_RANGE))
    r0 = int(np.random.randint(FRAME_MARGIN, GRID_SIZE - h))
    c0 = int(np.random.randint(FRAME_MARGIN, GRID_SIZE - w))
    return r0, c0, r0 + h - 1, c0 + w - 1, h, w


def _gen_attempt():
    r0, c0, r1, c1, h, w = _sample_frame_box()
    colors = list(np.random.choice(SWAP_COLOR_POOL, size=2, replace=False))
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    outside = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    g[outside] = _gen_fill_values(int(outside.sum()), colors, OUTSIDE_BLACK_FRAC_RANGE)

    g[r0, c0 : c1 + 1] = FRAME_COLOR
    g[r1, c0 : c1 + 1] = FRAME_COLOR
    g[r0 : r1 + 1, c0] = FRAME_COLOR
    g[r0 : r1 + 1, c1] = FRAME_COLOR

    interior_n = (h - 2) * (w - 2)
    g[r0 + 1 : r1, c0 + 1 : c1] = _gen_fill_values(
        interior_n, colors, INTERIOR_BLACK_FRAC_RANGE
    ).reshape(h - 2, w - 2)

    try:
        main(g)
    except Exception:
        return None
    return g


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input")
