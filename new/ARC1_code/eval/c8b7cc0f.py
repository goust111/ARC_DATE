import numpy as np

from arc_utils import bbox, colors_present, get_bg


def _rect_border_mask(shape, box):
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the marker pixels inside the rectangular frame and render that count as a 3x3 row-major tally.
    Input structure
    A background grid contains one one-cell-thick rectangular frame color and one marker color scattered both inside and outside the frame.
    Rule
    Return a 3x3 background grid with the first N cells in row-major order set to the marker color, where N is the number of marker pixels strictly inside the frame.
    Key insight
    The rectangular frame defines the only region to summarize; same-colored markers outside it are distractors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_candidates = []
    for color in colors_present(g, exclude={bg}):
        color_mask = g == color
        box = bbox(color_mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rect_border_mask(g.shape, box)
        if int(color_mask.sum()) == int(border.sum()) and np.all(color_mask[border]):
            frame_candidates.append((color, box))

    assert len(frame_candidates) == 1
    frame_color, (r0, c0, r1, c1) = frame_candidates[0]

    marker_colors = colors_present(g, exclude={bg, frame_color})
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    count = int(np.count_nonzero(interior == marker_color))
    assert 0 <= count <= 9

    out = np.full((3, 3), bg, dtype=int)
    out.flat[:count] = marker_color
    return out

# --------------- generator (o5x) ---------------
# Canvas and colors.
BG_COLOR, FRAME_COLOR = 0, 1
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (7, 10), (7, 10)
MARKER_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

# Frame geometry and edge-touch constraints.
FRAME_HEIGHT_MIN, FRAME_WIDTH_MIN, FRAME_WIDTH_MAX = 4, 5, 7
INTERIOR_AREA_MIN, INTERIOR_AREA_MAX = 8, 20
INTERIOR_AREA_BAG = [8, 9, 10, 12, 15, 16, 18, 20]
TOUCH_COUNT_MIN, TOUCH_COUNT_MAX = 1, 2

# Marker counts, output shape, and retry limit.
INSIDE_COUNT_RANGE, OUTSIDE_COUNT_RANGE = (1, 10), (3, 7)
OUTPUT_SHAPE, GEN_RETRY_LIMIT = (3, 3), 300

_gen_area_bag = []

def _gen_interior_area(rect_h, rect_w):
    return (rect_h - 2) * (rect_w - 2)

def _gen_choose_interior_area(available_areas):
    global _gen_area_bag
    available = set(available_areas)
    if not _gen_area_bag:
        _gen_area_bag = [int(area) for area in np.random.permutation(INTERIOR_AREA_BAG)]
    for idx, area in enumerate(_gen_area_bag):
        if area in available:
            return _gen_area_bag.pop(idx)
    available_areas = sorted(available)
    return available_areas[int(np.random.randint(len(available_areas)))]

def _gen_rect_positions(height, width, rect_h, rect_w):
    positions = []
    for r0 in range(height - rect_h + 1):
        for c0 in range(width - rect_w + 1):
            touches = (
                int(r0 == 0)
                + int(r0 + rect_h == height)
                + int(c0 == 0)
                + int(c0 + rect_w == width)
            )
            if TOUCH_COUNT_MIN <= touches <= TOUCH_COUNT_MAX:
                positions.append((r0, c0))
    return positions

def _gen_frame_options(height, width, inside_count, outside_count):
    options = []
    for rect_h in range(FRAME_HEIGHT_MIN, height):
        for rect_w in range(FRAME_WIDTH_MIN, min(FRAME_WIDTH_MAX, width - 1) + 1):
            interior_area = _gen_interior_area(rect_h, rect_w)
            if not (INTERIOR_AREA_MIN <= interior_area <= INTERIOR_AREA_MAX and inside_count <= interior_area):
                continue

            border_area = 2 * rect_h + 2 * rect_w - 4
            bg_area = height * width - border_area - inside_count - outside_count
            if bg_area <= border_area or bg_area <= inside_count + outside_count:
                continue

            for r0, c0 in _gen_rect_positions(height, width, rect_h, rect_w):
                options.append((r0, c0, rect_h, rect_w))
    return options

def _gen_select_frame(height, width, inside_count, outside_count):
    options = _gen_frame_options(height, width, inside_count, outside_count)
    if not options:
        return None

    interior_areas = sorted({_gen_interior_area(opt[2], opt[3]) for opt in options})
    target_area = _gen_choose_interior_area(interior_areas)
    area_options = [opt for opt in options if _gen_interior_area(opt[2], opt[3]) == target_area]
    return area_options[int(np.random.randint(len(area_options)))]

def _gen_expected_output(grid):
    box = bbox(grid == FRAME_COLOR)
    marker_colors = [int(color) for color in np.unique(grid) if color not in (BG_COLOR, FRAME_COLOR)]
    if len(marker_colors) != 1 or box is None:
        return None

    r0, c0, r1, c1 = box
    marker_color = marker_colors[0]
    count = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == marker_color))
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    out.flat[:count] = marker_color
    return out

def _gen_render_grid(height, width, frame, inside_count, outside_count, marker_color):
    r0, c0, rect_h, rect_w = frame
    r1, c1 = r0 + rect_h - 1, c0 + rect_w - 1

    grid = np.zeros((height, width), dtype=int)
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR

    interior_coords = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == BG_COLOR)
    chosen_inside = np.random.choice(len(interior_coords), size=inside_count, replace=False)
    inside_coords = interior_coords[chosen_inside]
    grid[r0 + 1 + inside_coords[:, 0], c0 + 1 + inside_coords[:, 1]] = marker_color

    outside_mask = np.ones((height, width), dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_coords = np.argwhere(outside_mask)
    if len(outside_coords) < outside_count:
        return None
    chosen_outside = np.random.choice(len(outside_coords), size=outside_count, replace=False)
    outside_coords = outside_coords[chosen_outside]
    grid[outside_coords[:, 0], outside_coords[:, 1]] = marker_color

    return grid

def _gen_attempt():
    height = int(np.random.randint(*GRID_HEIGHT_RANGE))
    width = int(np.random.randint(*GRID_WIDTH_RANGE))
    inside_count = int(np.random.randint(*INSIDE_COUNT_RANGE))
    outside_count = int(np.random.randint(*OUTSIDE_COUNT_RANGE))
    marker_color = int(np.random.choice(MARKER_COLORS))

    frame = _gen_select_frame(height, width, inside_count, outside_count)
    if frame is None:
        return None
    return _gen_render_grid(height, width, frame, inside_count, outside_count, marker_color)

def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        if expected is None:
            continue
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            pass
    raise RuntimeError("failed to generate valid c8b7cc0f input")
