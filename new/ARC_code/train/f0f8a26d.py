import numpy as np

from arc_utils import bbox_coords, colors_present, find_components, get_bg
import random


def _rotated_bar_coords(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(coords.shape[0])

    assert height == 1 or width == 1
    length = max(height, width)
    assert area == length
    assert length % 2 == 1

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    half = length // 2

    if length == 1:
        return [(cr, cc)]
    if height == 1:
        return [(r, cc) for r in range(cr - half, cr + half + 1)]
    return [(cr, c) for c in range(cc - half, cc + half + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate every foreground bar 90 degrees around its own center on the same canvas.

    Input structure
    The background is the dominant color, and all foreground pixels share one color.
    Each 4-connected foreground object is a solid one-cell-thick horizontal or vertical
    bar with odd length, with single-pixel bars allowed.

    Rule
    Clear the canvas to the background, then redraw every bar with its midpoint, length,
    and color preserved while swapping horizontal and vertical orientation.

    Key insight
    The bars are independent objects: their centers and lengths identify the output
    placement, so no matching, ordering, or global symmetry is needed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1

    out = np.full_like(g, bg)
    h, w = g.shape
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=fg_colors,
    )
    assert components

    for comp in components:
        color = int(comp["color"])
        for r, c in _rotated_bar_coords(comp["coords"]):
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] == bg
            out[r, c] = color

    return out

# --------------- generator (c) ---------------

# Palette and canvas options.
BG_COLOR = 7
FG_COLORS = tuple(c for c in range(10) if c != BG_COLOR)
GRID_SIDES = (9, 11, 13)

# Bar counts and weighted odd-length sampling.
MIN_BARS, EXTRA_LENGTH_CANDIDATES = 5, 5
MAX_BARS_BY_SIDE = {9: 6, 11: 8, 13: 10}
LENGTH_WEIGHT_BY_SIZE = {1: 8, 3: 56, 5: 26, 7: 6, 9: 3}
DEFAULT_LENGTH_WEIGHT = 1
ORIENTATIONS = ("H", "V")

# Retry and placement search limits.
OUTER_ATTEMPTS, INNER_ATTEMPTS = 10, 50
CENTER_TRIAL_LIMIT = 80

def _sample_lengths(N, K):
    possible_lengths = list(range(1, N + 1, 2))
    weights = [LENGTH_WEIGHT_BY_SIZE.get(l, DEFAULT_LENGTH_WEIGHT) for l in possible_lengths]
    lengths = random.choices(possible_lengths, weights=weights, k=K + EXTRA_LENGTH_CANDIDATES)
    lengths.sort(reverse=True)
    return lengths

def _bar_pixels(cr, cc, length, orient):
    half = length // 2
    if orient == "H":
        return {(cr, cc + d) for d in range(-half, half + 1)}
    return {(cr + d, cc) for d in range(-half, half + 1)}

def _place_one_bar(N, length, occupied):
    """Return input pixels for one placed bar, mutating occupied on success."""
    half = length // 2
    lo, hi = half, N - 1 - half
    if lo > hi:
        return None
    orient = random.choice(ORIENTATIONS)
    out_orient = "V" if orient == "H" else "H"
    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates[:CENTER_TRIAL_LIMIT]:
        inp_pixels = _bar_pixels(cr, cc, length, orient)
        out_pixels = _bar_pixels(cr, cc, length, out_orient)
        footprint = inp_pixels | out_pixels
        if footprint & occupied:
            continue
        occupied |= footprint
        return inp_pixels
    return None

def _place_bars(N, K):
    occupied = set()
    bars_input = []
    for length in _sample_lengths(N, K):
        if len(bars_input) >= K:
            break
        inp_pixels = _place_one_bar(N, length, occupied)
        if inp_pixels is not None:
            bars_input.append(inp_pixels)
    return bars_input if len(bars_input) >= K else None

def _gen_attempt(N: int, K: int) -> np.ndarray | None:
    fg = random.choice(FG_COLORS)
    bars_input = _place_bars(N, K)
    if bars_input is None:
        return None

    grid = np.full((N, N), BG_COLOR, dtype=int)
    for bar_pixels in bars_input:
        for r, c in bar_pixels:
            grid[r, c] = fg

    try:
        main(grid)
    except Exception:
        return None
    return grid

def generate_input() -> np.ndarray:
    for _outer in range(OUTER_ATTEMPTS):
        N = random.choice(GRID_SIDES)
        K = random.randint(MIN_BARS, MAX_BARS_BY_SIDE[N])
        for _ in range(INNER_ATTEMPTS):
            result = _gen_attempt(N, K)
            if result is not None:
                return result
    raise RuntimeError("Failed to generate valid input after many attempts")
