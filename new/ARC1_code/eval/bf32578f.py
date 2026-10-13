import numpy as np

from arc_utils import colors_present, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the interior implied by a left half-outline and its horizontal mirror.

    Input structure
    A 2D grid contains one background color and one foreground color. The
    foreground is the left half of a horizontally symmetric closed outline, with
    its global right edge immediately left of the vertical symmetry axis.

    Rule
    Clear the outline. For each row containing foreground, take the foreground
    pixel closest to the axis as the left boundary, mirror it across the
    half-column axis just to the right of the global right edge, and fill only
    the cells strictly between those two boundary pixels.

    Key insight
    Row segments near the top and bottom are outline, not fill; the row's
    rightmost foreground pixel is the inner boundary that determines that row's
    hidden symmetric interior span.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    color = fg_colors[0]

    coords = np.argwhere(g == color)
    assert coords.size > 0
    axis_left = int(coords[:, 1].max())

    out = np.full_like(g, bg)
    for r in np.unique(coords[:, 0]):
        row_cols = np.sort(coords[coords[:, 0] == r, 1])
        assert np.array_equal(row_cols, np.arange(row_cols[0], row_cols[-1] + 1))

        inner_left = int(row_cols[-1])
        right_boundary = 2 * axis_left + 1 - inner_left
        assert right_boundary < g.shape[1]

        if inner_left + 1 < right_boundary:
            out[int(r), inner_left + 1 : right_boundary] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas/palette.
CANVAS_SIZES = (6, 8, 10)
BACKGROUND_COLOR = 0
FOREGROUND_COLOR_MIN, FOREGROUND_COLOR_MAX_EXCLUSIVE = 1, 10
# Shape/profile.
FILL_ROW_COUNTS = (3, 4)
CAP_ROW_COUNT = 2
MAX_TOP_PADDING = 3
EDGE_WIDTH_MIN, EDGE_WIDTH_MAX = 1, 2
MAX_WIDTH_GROWTH = 2
# Outline/oracle.
OUTLINE_PIXEL_MIN, OUTLINE_PIXEL_MAX = 5, 9
INTERVAL_TOUCH_GAP = 1
OUTLINE_CONNECTIVITY = 8
# Retry policy.
GENERATION_RETRIES = 300

def _gen_pick(items):
    return items[int(np.random.randint(0, len(items)))]

def _gen_intervals_touch(left_a, right_a, left_b, right_b):
    return max(left_a, left_b) <= min(right_a, right_b) + INTERVAL_TOUCH_GAP

def _gen_min_outers(inners):
    outers = list(inners)
    for i in range(len(inners) - 1):
        if _gen_intervals_touch(outers[i], inners[i], outers[i + 1], inners[i + 1]):
            continue
        if inners[i] > inners[i + 1]:
            outers[i] = min(outers[i], inners[i + 1] + 1)
        elif inners[i + 1] > inners[i]:
            outers[i + 1] = min(outers[i + 1], inners[i] + 1)

    for i in range(len(inners) - 2, -1, -1):
        if _gen_intervals_touch(outers[i], inners[i], outers[i + 1], inners[i + 1]):
            continue
        if inners[i] > inners[i + 1]:
            outers[i] = min(outers[i], inners[i + 1] + 1)
        elif inners[i + 1] > inners[i]:
            outers[i + 1] = min(outers[i + 1], inners[i] + 1)

    return outers

def _gen_outline_variants(inners):
    base_outers = _gen_min_outers(inners)
    n_rows = len(inners)
    groups = [(i, n_rows - 1 - i) for i in range((n_rows + 1) // 2)]
    variants = []

    def rec(group_i, current):
        if group_i == len(groups):
            outers = [0] * n_rows
            for group, val in zip(groups, current):
                a, b = group
                outers[a] = val
                outers[b] = val

            pixel_count = sum(inner - outer + 1 for outer, inner in zip(outers, inners))
            if not (OUTLINE_PIXEL_MIN <= pixel_count <= OUTLINE_PIXEL_MAX):
                return
            for i in range(n_rows - 1):
                if not _gen_intervals_touch(outers[i], inners[i], outers[i + 1], inners[i + 1]):
                    return
            variants.append((outers, pixel_count))
            return

        a, b = groups[group_i]
        hi = min(base_outers[a], base_outers[b])
        for val in range(hi, -1, -1):
            current.append(val)
            rec(group_i + 1, current)
            current.pop()

    rec(0, [])
    if not variants:
        return []

    by_count = {}
    for outers, pixel_count in variants:
        by_count.setdefault(pixel_count, []).append(outers)
    pixel_count = _gen_pick(sorted(by_count))
    return by_count[pixel_count]

def _gen_width_profiles(axis_left, fill_rows):
    profiles = []
    for edge_width in range(EDGE_WIDTH_MIN, min(EDGE_WIDTH_MAX, axis_left) + 1):
        max_peak = min(axis_left, edge_width + MAX_WIDTH_GROWTH)
        for peak_width in range(edge_width, max_peak + 1):
            if fill_rows == 3:
                widths = [edge_width, peak_width, edge_width]
            else:
                widths = [edge_width, peak_width, peak_width, edge_width]
            inners = [axis_left] + [axis_left - w for w in widths] + [axis_left]
            if _gen_outline_variants(inners):
                profiles.append(widths)
    return profiles

def _gen_sample_design():
    n = int(_gen_pick(CANVAS_SIZES))
    axis_left = n // 2 - 1
    fill_rows = int(_gen_pick(FILL_ROW_COUNTS))
    profiles = _gen_width_profiles(axis_left, fill_rows)
    if not profiles:
        return None

    widths = list(_gen_pick(profiles))
    color = int(np.random.randint(FOREGROUND_COLOR_MIN, FOREGROUND_COLOR_MAX_EXCLUSIVE))
    shape_rows = fill_rows + CAP_ROW_COUNT
    max_top = min(MAX_TOP_PADDING, n - shape_rows)
    top = int(np.random.randint(0, max_top + 1))
    return n, axis_left, fill_rows, widths, color, top

def _gen_build_output_and_inners(n, axis_left, top, fill_rows, widths, color):
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for j, half_width in enumerate(widths):
        row = top + 1 + j
        inner_left = axis_left - half_width
        out[row, inner_left + 1 : 2 * axis_left + 1 - inner_left] = color

    inners = [axis_left]
    for j in range(fill_rows):
        row = top + 1 + j
        filled = np.flatnonzero(out[row] == color)
        if filled.size == 0:
            return None, None
        inners.append(int(filled[0] - 1))
    inners.append(axis_left)
    return out, inners

def _gen_passes_oracle(inp, out, outers, inners):
    comps = find_components(inp, bg=BACKGROUND_COLOR, connectivity=OUTLINE_CONNECTIVITY, monochromatic=True, return_format="dict")
    if len(comps) != 1:
        return False
    expected_area = sum(inner - outer + 1 for outer, inner in zip(outers, inners))
    if comps[0]["area"] != expected_area:
        return False
    return np.array_equal(main(inp), out)

def _gen_attempt():
    design = _gen_sample_design()
    if design is None:
        return None

    n, axis_left, fill_rows, widths, color, top = design
    out, inners = _gen_build_output_and_inners(n, axis_left, top, fill_rows, widths, color)
    if inners is None:
        return None

    outer_choices = _gen_outline_variants(inners)
    if not outer_choices:
        return None
    outers = list(_gen_pick(outer_choices))

    inp = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for j, (outer_left, inner_left) in enumerate(zip(outers, inners)):
        row = top + j
        inp[row, outer_left : inner_left + 1] = color
    inp[:, axis_left + 1 :] = BACKGROUND_COLOR

    if not _gen_passes_oracle(inp, out, outers, inners):
        return None
    return inp

def generate_input():
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate bf32578f input")
