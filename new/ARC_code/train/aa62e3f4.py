from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4
import random


def _unique_color_with_count(counts: Counter[int], colors: list[int], target_count: int) -> int:
    matches = [color for color in colors if counts[color] == target_count]
    assert len(matches) == 1, "expected a unique structural color by area"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the outlined object with the outside contour of that outline in the marker color.

    Input structure:
    - A dominant background surrounds one 8-connected foreground outline.
    - The outline is the unique largest foreground color.
    - One or more smaller interior colors are inside the outline; the marker color
      is the unique smallest non-outline color.

    Rule:
    - Erase all original foreground pixels to background.
    - Paint every background pixel that is 4-adjacent to the outline with the
      marker color.
    - Keep the grid shape unchanged.

    Key insight: The colored interior marker supplies the paint color, while the
    largest outline supplies only the contour whose outside edge is traced.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    counts = Counter(map(int, g.flatten()))
    foreground_colors = [color for color in counts if color != bg]
    assert len(foreground_colors) >= 2, "expected an outline color and a marker color"

    frame_count = max(counts[color] for color in foreground_colors)
    frame_color = _unique_color_with_count(counts, foreground_colors, frame_count)
    assert len(components_mask(g == frame_color, connectivity=8)) == 1, "outline must be 8-connected"

    marker_candidates = [color for color in foreground_colors if color != frame_color]
    marker_count = min(counts[color] for color in marker_candidates)
    marker_color = _unique_color_with_count(counts, marker_candidates, marker_count)

    out = np.full_like(g, bg)
    contour: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == frame_color):
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == bg:
                contour.add((int(nr), int(nc)))
    assert contour, "expected outline to touch background contour cells"

    for r, c in contour:
        out[r, c] = marker_color
    return out

# --------------- generator (att2) ---------------

GRID_MIN, GRID_MAX = 5, 16
N_COLORS = 10

SHAPE_MIN_HEIGHT, SHAPE_MAX_HEIGHT = 5, 7
SHAPE_MIN_WIDTH, SHAPE_MAX_WIDTH = 3, 5
START_WIDTH_MIN, START_WIDTH_MAX = 1, 2
MAX_LOCAL_WIDTH = 5

MARKER_MIN_CELLS, MARKER_MAX_CELLS = 1, 3

GEN_ATTEMPTS = 300

EDGE_TOUCH_TOP_BOTTOM = "TB"
CORNER_EDGE_PAIRS = ["TL", "TR", "BL", "BR"]
ZERO_DELTA_SHIFTS = [-1, 0, 0, 0, 1]


def _gen_grow_parts(total, n):
    n_twos = total - n
    parts = [2] * n_twos + [1] * (n - n_twos)
    random.shuffle(parts)
    return parts

def _gen_drop_parts(total, n):
    if n == 0:
        return []
    n_twos = random.randint(max(0, total - n), min(total // 2, n))
    n_ones = total - 2 * n_twos
    parts = [2] * n_twos + [1] * n_ones + [0] * (n - n_twos - n_ones)
    random.shuffle(parts)
    return parts

def _gen_widths(height, max_width):
    start = random.randint(START_WIDTH_MIN, START_WIDTH_MAX)
    diff = max_width - start
    grow_steps = random.randint((diff + 1) // 2, min(diff, height - 1))
    widths = [start]
    for inc in _gen_grow_parts(diff, grow_steps):
        widths.append(widths[-1] + inc)

    remaining = height - len(widths)
    min_end = max(1, max_width - 2 * remaining)
    end_width = random.randint(min_end, max_width)
    for dec in _gen_drop_parts(max_width - end_width, remaining):
        widths.append(widths[-1] - dec)
    return widths

def _gen_intervals(widths):
    left, right = 0, widths[0] - 1
    intervals = [(left, right)]
    for prev_width, width in zip(widths, widths[1:]):
        delta = width - prev_width
        if delta == 2:
            left -= 1
            right += 1
        elif delta == 1:
            if random.choice([False, True]):
                left -= 1
            else:
                right += 1
        elif delta == 0:
            shift = random.choice(ZERO_DELTA_SHIFTS) if width >= 2 else 0
            left += shift
            right += shift
        elif delta == -1:
            if random.choice([False, True]):
                left += 1
            else:
                right -= 1
        elif delta == -2:
            left += 1
            right -= 1
        intervals.append((left, right))

    min_left = min(l for l, _ in intervals)
    intervals = [(l - min_left, r - min_left) for l, r in intervals]
    if max(r for _, r in intervals) + 1 > MAX_LOCAL_WIDTH:
        return None
    return intervals

def _gen_local_mask(intervals):
    height = len(intervals)
    width = max(r for _, r in intervals) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, (c0, c1) in enumerate(intervals):
        mask[r, c0 : c1 + 1] = True
    return mask

def _gen_place_shape(grid_shape, local_mask, edge_pair):
    rows, cols = grid_shape
    height, width = local_mask.shape
    if edge_pair == EDGE_TOUCH_TOP_BOTTOM:
        if height != rows or width > cols - 2:
            return None
        return 0, random.randint(1, cols - width - 1)
    if height >= rows or width >= cols:
        return None
    if edge_pair == "TL":
        return 0, 0
    if edge_pair == "TR":
        return 0, cols - width
    if edge_pair == "BL":
        return rows - height, 0
    return rows - height, cols - width

def _gen_full_mask(rows, cols, local_mask, r0, c0):
    mask = np.zeros((rows, cols), dtype=bool)
    height, width = local_mask.shape
    mask[r0 : r0 + height, c0 : c0 + width] = local_mask
    return mask

def _gen_layers(mask):
    remaining = mask.copy()
    layers = []
    rows, cols = mask.shape
    while np.any(remaining):
        layer = np.zeros_like(mask)
        for r, c in np.argwhere(remaining):
            for nr, nc in neighbors4(int(r), int(c)):
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not remaining[nr, nc]:
                    layer[r, c] = True
                    break
        layers.append(layer)
        remaining[layer] = False
    return layers

def _gen_contour(outline, shape):
    rows, cols = shape.shape
    contour = np.zeros_like(shape)
    for r, c in np.argwhere(outline):
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < rows and 0 <= nc < cols and not shape[nr, nc]:
                contour[nr, nc] = True
    return contour

def _choose_canvas_and_shape_profile():
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(N_COLORS)

    edge_options = []
    if rows <= SHAPE_MAX_HEIGHT:
        edge_options.append(EDGE_TOUCH_TOP_BOTTOM)
    if rows >= SHAPE_MIN_HEIGHT + 1:
        edge_options.extend(CORNER_EDGE_PAIRS)
    if not edge_options:
        return None

    edge_pair = random.choice(edge_options)
    height = rows if edge_pair == EDGE_TOUCH_TOP_BOTTOM else random.randint(
        SHAPE_MIN_HEIGHT, min(SHAPE_MAX_HEIGHT, rows - 1)
    )
    max_width_limit = cols - 2 if edge_pair == EDGE_TOUCH_TOP_BOTTOM else cols - 1
    if max_width_limit < SHAPE_MIN_WIDTH:
        return None
    max_width = random.randint(SHAPE_MIN_WIDTH, min(SHAPE_MAX_WIDTH, max_width_limit))

    intervals = _gen_intervals(_gen_widths(height, max_width))
    if intervals is None:
        return None
    return rows, cols, bg, edge_pair, intervals

def _build_shape(rows, cols, edge_pair, intervals):
    local_mask = _gen_local_mask(intervals)
    placement = _gen_place_shape((rows, cols), local_mask, edge_pair)
    if placement is None:
        return None

    shape = _gen_full_mask(rows, cols, local_mask, *placement)
    shape_area = int(np.sum(shape))
    if rows * cols - shape_area <= shape_area:
        return None
    return shape

def _split_valid_layers(shape):
    layers = _gen_layers(shape)
    if len(layers) < 2:
        return None

    outline = layers[0]
    marker = layers[-1]
    fill = shape & ~outline & ~marker

    outline_count = int(np.sum(outline))
    marker_count = int(np.sum(marker))
    fill_count = int(np.sum(fill))
    if marker_count < MARKER_MIN_CELLS or marker_count > MARKER_MAX_CELLS:
        return None
    if len(components_mask(outline, connectivity=8)) != 1:
        return None
    if fill_count and not (marker_count < fill_count < outline_count):
        return None
    return outline, fill, marker, fill_count

def _render_input(rows, cols, bg, masks, colors):
    outline, fill, marker = masks
    outline_color, fill_color, marker_color = colors
    inp = np.full((rows, cols), bg, dtype=int)
    inp[outline] = outline_color
    if fill_color is not None:
        inp[fill] = fill_color
    inp[marker] = marker_color
    return inp

def _gen_attempt():
    profile = _choose_canvas_and_shape_profile()
    if profile is None:
        return None
    rows, cols, bg, edge_pair, intervals = profile

    shape = _build_shape(rows, cols, edge_pair, intervals)
    if shape is None:
        return None

    layer_parts = _split_valid_layers(shape)
    if layer_parts is None:
        return None
    outline, fill, marker, fill_count = layer_parts

    contour = _gen_contour(outline, shape)
    if not np.any(contour):
        return None

    colors = [c for c in range(N_COLORS) if c != bg]
    random.shuffle(colors)
    outline_color, marker_color = colors[:2]
    fill_color = colors[2] if fill_count else None
    inp = _render_input(rows, cols, bg, (outline, fill, marker), (outline_color, fill_color, marker_color))

    expected = np.full((rows, cols), bg, dtype=int)
    expected[contour] = marker_color
    if not np.array_equal(main(inp), expected):
        return None
    return inp

def generate_input():
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid aa62e3f4 input")
