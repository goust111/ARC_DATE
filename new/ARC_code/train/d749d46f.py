import numpy as np

from arc_utils import colors_present, find_components, get_bg
import random


def _assert_output_footprints_disjoint(dims: list[tuple[int, int]]) -> None:
    occupied: set[tuple[int, int]] = set()

    c = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        cells = {(r, cc) for r in range(short) for cc in range(c, c + long)}
        assert not (occupied & cells), "normalized output rectangles must not overlap"
        occupied |= cells
        c += long + 1

    c = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        cells = {
            (r, cc)
            for r in range(10 - long, 10)
            for cc in range(c, c + short)
        }
        assert not (occupied & cells), "normalized output rectangles must not overlap"
        occupied |= cells
        c += short + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

    Input structure
    The grid has one background color and one foreground color; the foreground forms
    separate solid rectangles, ordered from left to right.

    Rule
    Build a 10-row background canvas whose width is the packed width of the wide
    versions of the rectangles. For each source rectangle, paint a top-aligned
    rectangle of shape min(h,w) by max(h,w), with one background column between
    consecutive rectangles. Then, starting again from the left edge, paint a
    bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
    again separated by one background column.

    Key insight
    The input rectangles are not copied in their original orientations; each object
    contributes one horizontal-facing and one vertical-facing normalized rectangle,
    preserving left-to-right object order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    components = find_components(g, bg=bg, connectivity=4, colors=[fg])
    assert components
    components = sorted(components, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))

    dims: list[tuple[int, int]] = []
    for obj in components:
        r0, c0, r1, c1 = obj["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert obj["area"] == h * w
        dims.append((h, w))

    out_h = 10
    out_w = sum(max(h, w) for h, w in dims) + len(dims) - 1
    _assert_output_footprints_disjoint(dims)
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        assert long <= out_h
        out[:short, c : c + long] = fg
        c += long + 1

    c = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        assert long <= out_h
        out[out_h - long : out_h, c : c + short] = fg
        c += short + 1

    return out

# --------------- generator (att1) ---------------

# Color palette.
COLOR_VALUES = tuple(range(10))

# Rectangle counts and dimensions.
RECT_COUNT_MIN = 4
RECT_COUNT_MAX = 6
SHORT_SIDE_MIN = 1
SHORT_SIDE_MAX = 3
SIDE_SUM_MAX = 10
LONG_SIDE_MAX = SIDE_SUM_MAX - SHORT_SIDE_MIN

# Canvas and packing constraints.
ARC_GRID_MAX = 30
OUTPUT_HEIGHT = 10
GAP_WIDTH = 1

# Retry policy.
MAX_GENERATION_ATTEMPTS = 300


def _gen_d749d46f_output_footprints_disjoint(dims):
    occupied = set()

    col = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        cells = {(r, c) for r in range(short) for c in range(col, col + long)}
        if occupied & cells:
            return False
        occupied |= cells
        col += long + GAP_WIDTH

    col = 0
    for h, w in dims:
        short, long = min(h, w), max(h, w)
        cells = {
            (r, c)
            for r in range(OUTPUT_HEIGHT - long, OUTPUT_HEIGHT)
            for c in range(col, col + short)
        }
        if occupied & cells:
            return False
        occupied |= cells
        col += short + GAP_WIDTH

    return True


def _sample_dimensions(n_rects):
    """Sample source rectangle dimensions under the output-width budget."""
    dims = []
    remaining_budget = ARC_GRID_MAX - GAP_WIDTH * (n_rects - 1)

    for i in range(n_rects):
        rects_left = n_rects - i
        min_needed = rects_left - 1
        max_for_this = min(remaining_budget - min_needed, LONG_SIDE_MAX)
        if max_for_this < SHORT_SIDE_MIN:
            return None

        short = random.randint(SHORT_SIDE_MIN, SHORT_SIDE_MAX)
        max_long = min(SIDE_SUM_MAX - short, max_for_this)
        if max_long < short:
            short = SHORT_SIDE_MIN
            max_long = min(LONG_SIDE_MAX, max_for_this)
            if max_long < SHORT_SIDE_MIN:
                return None

        long = random.randint(short, max_long)
        remaining_budget -= long

        if random.random() < 0.5:
            h, w = long, short
        else:
            h, w = short, long
        dims.append((h, w))

    return dims


def _input_shape(dims):
    return max(h for h, _ in dims), sum(w for _, w in dims) + GAP_WIDTH * (len(dims) - 1)


def _background_is_dominant(dims, inp_h, inp_w):
    fg_pixels = sum(h * w for h, w in dims)
    total_pixels = inp_h * inp_w
    bg_pixels = total_pixels - fg_pixels
    return bg_pixels > fg_pixels


def _build_input_grid(dims, bg, fg, inp_h, inp_w):
    grid = np.full((inp_h, inp_w), bg, dtype=int)
    col = 0
    for h, w in dims:
        grid[:h, col:col + w] = fg
        col += w + GAP_WIDTH
    return grid


def _passes_solver_oracle(grid):
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape[0] == OUTPUT_HEIGHT


def _gen_attempt():
    """Generate one valid input grid, or None when sampled dimensions fail constraints."""
    bg, fg = random.sample(COLOR_VALUES, 2)
    n_rects = random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    dims = _sample_dimensions(n_rects)
    if dims is None:
        return None

    inp_h, inp_w = _input_shape(dims)
    if inp_w > ARC_GRID_MAX:
        return None
    if not _gen_d749d46f_output_footprints_disjoint(dims):
        return None
    if inp_h > ARC_GRID_MAX or inp_h < 1:
        return None
    if not _background_is_dominant(dims, inp_h, inp_w):
        return None

    grid = _build_input_grid(dims, bg, fg, inp_h, inp_w)
    if not _passes_solver_oracle(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
