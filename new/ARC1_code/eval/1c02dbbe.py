from collections import Counter

import numpy as np

from arc_utils import colors_present
import random


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a nonempty region"
    max_count = max(counts.values())
    modes = [color for color, count in counts.items() if count == max_count]
    assert len(modes) == 1, "expected a unique dominant color"
    return int(modes[0])


def _boundary_background(g: np.ndarray) -> int:
    border = np.zeros(g.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return _unique_mode(g[border])


def _largest_foreground_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    counts.pop(int(bg), None)
    assert counts, "expected foreground colors"
    max_count = max(counts.values())
    colors = [color for color, count in counts.items() if count == max_count]
    assert len(colors) == 1, "expected a unique large grey mask color"
    return int(colors[0])


def _target_rectangle(g: np.ndarray, color: int, mask_color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == int(color))
    assert pts.shape == (3, 2), "each marker color must appear exactly three times"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)

    candidates: list[tuple[int, int, int, int]] = []
    for row_trim in ("top", "bottom"):
        rr0 = int(r0) + (row_trim == "top")
        rr1 = int(r1) - (row_trim == "bottom")
        for col_trim in ("left", "right"):
            cc0 = int(c0) + (col_trim == "left")
            cc1 = int(c1) - (col_trim == "right")
            if rr0 > rr1 or cc0 > cc1:
                continue

            sub = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            sub_colors = set(map(int, sub.reshape(-1)))
            if (
                int(np.count_nonzero(sub == int(color))) == 1
                and sub_colors <= {int(mask_color), int(color)}
                and int(mask_color) in sub_colors
            ):
                candidates.append((rr0, cc0, rr1, cc1))

    assert len(candidates) == 1, "expected a unique grey rectangle controlled by the markers"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace grey rectangles indicated by three same-color corner markers with that marker color.

    Input structure:
    - A black empty background surrounds one large grey mask.
    - Each non-grey foreground color appears exactly three times as markers.
    - The three markers of one color form an axis-aligned bounding box around
      a grey rectangle: two markers are just outside adjacent sides, and one
      marker lies on the rectangle corner.

    Rule:
    - For each marker color, inspect the four rectangles made by trimming one
      vertical and one horizontal side from that color's marker bounding box.
    - The target is the unique trimmed rectangle containing only grey plus that
      marker color and exactly one of the three markers.
    - Clear all marker pixels to background, then recolor each target rectangle
      with its marker color.

    Key insight:
    - The grey mask disambiguates which two sides of each marker bounding box are
      outside the intended rectangle; the valid inside rectangle contains no
      background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _boundary_background(g)
    mask_color = _largest_foreground_color(g, bg)
    marker_colors = [color for color in colors_present(g, exclude={bg, mask_color})]
    assert marker_colors, "expected at least one marker color"

    rects = [(color, _target_rectangle(g, color, mask_color)) for color in marker_colors]

    out = g.copy()
    for color in marker_colors:
        out[g == int(color)] = bg

    painted = np.zeros(g.shape, dtype=bool)
    for color, (r0, c0, r1, c1) in rects:
        region = painted[r0 : r1 + 1, c0 : c1 + 1]
        assert not region.any(), "target rectangles must not overlap"
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        region[:, :] = True

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and slots.
_GEN_GRID_SIZE, _GEN_BG, _GEN_MASK_COLOR = 15, 0, 5
_GEN_GRID_SHAPE = (_GEN_GRID_SIZE, _GEN_GRID_SIZE)
_GEN_MARKER_COLORS, _GEN_CORNERS = [1, 2, 3, 4, 6, 7, 8, 9], ["TL", "TR", "BL", "BR"]
_GEN_WIDTH_PAIRS = (("TL", "TR"), ("BL", "BR"))

# Sampling bounds.
_GEN_MARKER_COUNT_RANGE = (1, 4)
_GEN_RECT_H_RANGE = (4, 7)
_GEN_RECT_W_RANGE = (3, 8)
_GEN_MASK_SIZE_RANGE = (10, 13)
_GEN_GAP_RANGE, _GEN_RETRY_LIMIT = (1, 3), 300

def _gen_pick_sum_pair(lo, hi, max_sum):
    total = random.randint(2 * lo, max_sum)
    a_min = max(lo, total - hi)
    a_max = min(hi, total - lo)
    a = random.randint(a_min, a_max)
    if random.random() < 0.5:
        return a, total - a
    return total - a, a

def _gen_gap_pair(size):
    lo, hi = _GEN_GAP_RANGE
    choices = []
    for first in range(lo, hi + 1):
        second = _GEN_GRID_SIZE - int(size) - first
        if lo <= second <= hi:
            choices.append((first, second))
    if not choices:
        return None
    return random.choice(choices)

def _gen_rect_for_corner(corner, mr0, mc0, mr1, mc1, h, w):
    top = corner[0] == "T"
    left = corner[1] == "L"
    r0 = mr0 if top else mr1 - h + 1
    r1 = mr0 + h - 1 if top else mr1
    c0 = mc0 if left else mc1 - w + 1
    c1 = mc0 + w - 1 if left else mc1
    return r0, c0, r1, c1

def _gen_marker_points(corner, mr0, mc0, mr1, mc1, rect):
    r0, c0, r1, c1 = rect
    top = corner[0] == "T"
    left = corner[1] == "L"
    inside = (mr0 if top else mr1, mc0 if left else mc1)
    outside_row = (mr0 - 1 if top else mr1 + 1, c1 if left else c0)
    outside_col = (r1 if top else r0, mc0 - 1 if left else mc1 + 1)
    return [inside, outside_row, outside_col]

def _gen_sample_dimensions(corners):
    """Return side groups plus sampled rectangle dimensions for each corner."""
    top = [corner for corner in corners if corner[0] == "T"]
    bottom = [corner for corner in corners if corner[0] == "B"]
    h_min, h_max = _GEN_RECT_H_RANGE
    heights = {}
    if top and bottom:
        top_cap, bottom_cap = _gen_pick_sum_pair(h_min, h_max, _GEN_MASK_SIZE_RANGE[1])
        for corner in top:
            heights[corner] = random.randint(h_min, top_cap)
        for corner in bottom:
            heights[corner] = random.randint(h_min, bottom_cap)
    else:
        for corner in corners:
            heights[corner] = random.randint(h_min, h_max)

    w_min, w_max = _GEN_RECT_W_RANGE
    widths = {}
    for left, right in _GEN_WIDTH_PAIRS:
        if left in corners and right in corners:
            widths[left], widths[right] = _gen_pick_sum_pair(
                w_min, w_max, _GEN_MASK_SIZE_RANGE[1]
            )
    for corner in corners:
        if corner not in widths:
            widths[corner] = random.randint(w_min, w_max)
    return top, bottom, heights, widths

def _gen_sample_mask_bounds(corners, top, bottom, heights, widths):
    """Sample inclusive mask bounds after enforcing non-overlap minimums."""
    mask_min, mask_max = _GEN_MASK_SIZE_RANGE
    min_h = max(mask_min, max(heights.values()))
    if top and bottom:
        min_h = max(
            min_h,
            max(heights[corner] for corner in top)
            + max(heights[corner] for corner in bottom),
        )

    min_w = max(mask_min, max(widths.values()))
    for left, right in _GEN_WIDTH_PAIRS:
        if left in corners and right in corners:
            min_w = max(min_w, widths[left] + widths[right])
    if min_h > mask_max or min_w > mask_max:
        return None

    mask_h = random.randint(min_h, mask_max)
    mask_w = random.randint(min_w, mask_max)
    row_gaps = _gen_gap_pair(mask_h)
    col_gaps = _gen_gap_pair(mask_w)
    if row_gaps is None or col_gaps is None:
        return None
    mr0, mc0 = row_gaps[0], col_gaps[0]
    return mr0, mc0, mr0 + mask_h - 1, mc0 + mask_w - 1

def _gen_render_pair(corners, colors, heights, widths, mask_bounds):
    """Render expected output and marker input; reject overlap or bad markers."""
    mr0, mc0, mr1, mc1 = mask_bounds
    out = np.full(_GEN_GRID_SHAPE, _GEN_BG, dtype=int)
    out[mr0 : mr1 + 1, mc0 : mc1 + 1] = _GEN_MASK_COLOR
    rects = []
    painted = np.zeros(_GEN_GRID_SHAPE, dtype=bool)
    for corner, color in zip(corners, colors):
        rect = _gen_rect_for_corner(
            corner, mr0, mc0, mr1, mc1, heights[corner], widths[corner]
        )
        r0, c0, r1, c1 = rect
        sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
        if painted[sl].any():
            return None
        painted[sl] = True
        out[sl] = color
        rects.append((corner, color, rect))

    inp = np.full(_GEN_GRID_SHAPE, _GEN_BG, dtype=int)
    inp[mr0 : mr1 + 1, mc0 : mc1 + 1] = _GEN_MASK_COLOR
    marker_seen = set()
    for corner, color, rect in rects:
        pts = _gen_marker_points(corner, mr0, mc0, mr1, mc1, rect)
        if len(set(pts)) != 3:
            return None
        for r, c in pts:
            marker = (r, c)
            in_bounds = 0 <= r < _GEN_GRID_SIZE and 0 <= c < _GEN_GRID_SIZE
            if not in_bounds or marker in marker_seen:
                return None
            marker_seen.add(marker)
            inp[marker] = color
    return inp, out

def _gen_attempt():
    count = random.randint(*_GEN_MARKER_COUNT_RANGE)
    corners = random.sample(_GEN_CORNERS, count)
    colors = random.sample(_GEN_MARKER_COLORS, len(corners))
    top, bottom, heights, widths = _gen_sample_dimensions(corners)
    mask_bounds = _gen_sample_mask_bounds(corners, top, bottom, heights, widths)
    if mask_bounds is None:
        return None

    rendered = _gen_render_pair(corners, colors, heights, widths, mask_bounds)
    if rendered is None:
        return None
    inp, out = rendered
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp

def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1c02dbbe input")
