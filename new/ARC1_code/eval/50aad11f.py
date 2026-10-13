import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste
import random


def _bbox_distance(point, box):
    r, c = point
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _component_mask(component):
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each repeated source object by its nearest singleton marker and pack the cropped sprites along the markers' dominant axis.

    Input structure
    - The background is the dominant color.
    - Exactly one non-background color forms multiple equal-size 4-connected source objects.
    - Every other non-background color is a single marker pixel near one source object.

    Rule
    - Match each marker to the nearest source-object bbox.
    - Crop every source object to its bbox and recolor its mask with the matched marker color.
    - If marker rows span more than marker columns, stack the recolored crops top-to-bottom by marker row; otherwise concatenate them left-to-right by marker column.

    Key insight
    - The marker pixels provide both the replacement colors and the output packing direction/order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})

    source_colors = [color for color in colors if np.count_nonzero(g == color) > 1]
    assert len(source_colors) == 1, source_colors
    source_color = source_colors[0]

    markers = []
    for color in colors:
        if color == source_color:
            continue
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), (color, coords.shape)
        markers.append(((int(coords[0, 0]), int(coords[0, 1])), int(color)))

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[source_color],
        return_format="dict",
    )
    assert len(components) == len(markers)
    assert len(components) >= 2

    masks = [_component_mask(component) for component in components]
    shapes = {mask.shape for mask in masks}
    assert len(shapes) == 1, shapes

    pairs = []
    used_component_idxs = set()
    for marker_point, marker_color in markers:
        distances = [_bbox_distance(marker_point, component["bbox"]) for component in components]
        min_distance = min(distances)
        nearest_idxs = [idx for idx, dist in enumerate(distances) if dist == min_distance]
        assert len(nearest_idxs) == 1, (marker_point, distances)
        component_idx = nearest_idxs[0]
        assert component_idx not in used_component_idxs, (marker_point, component_idx)
        used_component_idxs.add(component_idx)
        pairs.append((marker_point, marker_color, masks[component_idx]))
    assert len(used_component_idxs) == len(components)

    marker_rows = [point[0] for point, _color, _mask in pairs]
    marker_cols = [point[1] for point, _color, _mask in pairs]
    row_span = max(marker_rows) - min(marker_rows)
    col_span = max(marker_cols) - min(marker_cols)
    assert row_span != col_span, (row_span, col_span)

    axis = 0 if row_span > col_span else 1
    order_coords = marker_rows if axis == 0 else marker_cols
    assert len(set(order_coords)) == len(order_coords), order_coords

    blocks = []
    for _point, color, mask in sorted(pairs, key=lambda pair: pair[0][axis]):
        block = np.full(mask.shape, bg, dtype=int)
        block[mask] = color
        blocks.append(block)

    return np.concatenate(blocks, axis=axis)

# --------------- generator (o5x) ---------------

# Colors and canvas bounds.
BACKGROUND_COLOR = 0
SOURCE_COLOR = 6
MARKER_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
GRID_SIZE_RANGE = (9, 18)
MIN_AXIS_DIM_BY_COUNT = {2: 9, 3: 12}
EDGE_MARGIN = 1

# Source sprite sampling.
SPRITE_SIZE = 4
SPRITE_SHAPE = (SPRITE_SIZE, SPRITE_SIZE)
OBJECT_COUNT_RANGE = (2, 3)
FILL_COUNT_RANGE = (7, 10)
PATH_MASK_PROB = 0.55
HORIZONTAL_SPINE_PROB = 0.5
PATH_CORNERS = [(0, 0), (0, 3), (3, 0), (3, 3)]
SPINE_ROWS_OR_COLS = [1, 2]

# Placement constraints and retries.
MIN_BBOX_GAP = 3
OTHER_BBOX_MARKER_MIN_DISTANCE = 4
SOURCE_TOP_ROW_OFFSETS_FROM_MARKER = (-5, 2)
DISTINCT_MASK_RETRIES = 80
GENERATION_RETRIES = 300


def _gen_point_bbox_distance(point, box):
    r, c = point
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _gen_bbox_distance(box_a, box_b):
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    dr = max(0, ar0 - br1, br0 - ar1)
    dc = max(0, ac0 - bc1, bc0 - ac1)
    return dr + dc


def _gen_sprite(mask, color):
    sprite = np.zeros(SPRITE_SHAPE, dtype=int)
    sprite[mask] = color
    return sprite


def _gen_add_frontier_cells(cells, fill_count):
    while len(cells) < fill_count:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < SPRITE_SIZE and 0 <= nc < SPRITE_SIZE and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))


def _gen_path_cells():
    cells = set()
    start = random.choice(PATH_CORNERS)
    end = (SPRITE_SIZE - 1 - start[0], SPRITE_SIZE - 1 - start[1])
    row_step = 1 if end[0] > start[0] else -1
    col_step = 1 if end[1] > start[1] else -1
    moves = [(row_step, 0)] * (SPRITE_SIZE - 1) + [(0, col_step)] * (SPRITE_SIZE - 1)
    random.shuffle(moves)

    r, c = start
    cells.add((r, c))
    for dr, dc in moves:
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _gen_spine_cells():
    cells = set()
    if random.random() < HORIZONTAL_SPINE_PROB:
        spine_r = random.choice(SPINE_ROWS_OR_COLS)
        cells.update((spine_r, c) for c in range(SPRITE_SIZE))
        top_c = random.randrange(SPRITE_SIZE)
        bot_c = random.randrange(SPRITE_SIZE)
        cells.update((r, top_c) for r in range(0, spine_r))
        cells.update((r, bot_c) for r in range(spine_r + 1, SPRITE_SIZE))
    else:
        spine_c = random.choice(SPINE_ROWS_OR_COLS)
        cells.update((r, spine_c) for r in range(SPRITE_SIZE))
        left_r = random.randrange(SPRITE_SIZE)
        right_r = random.randrange(SPRITE_SIZE)
        cells.update((left_r, c) for c in range(0, spine_c))
        cells.update((right_r, c) for c in range(spine_c + 1, SPRITE_SIZE))
    return cells


def _gen_connected_mask(fill_count):
    cells = _gen_path_cells() if random.random() < PATH_MASK_PROB else _gen_spine_cells()
    _gen_add_frontier_cells(cells, fill_count)
    mask = np.zeros(SPRITE_SHAPE, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_distinct_masks(count):
    masks = []
    seen = set()
    for _ in range(count):
        for _ in range(DISTINCT_MASK_RETRIES):
            mask = _gen_connected_mask(random.randint(*FILL_COUNT_RANGE))
            key = tuple(mask.reshape(-1).tolist())
            if key not in seen:
                seen.add(key)
                masks.append(mask)
                break
        else:
            return None
    return masks


def _gen_output_from_masks(masks, colors, axis):
    blocks = [_gen_sprite(mask, color) for mask, color in zip(masks, colors)]
    return np.concatenate(blocks, axis=axis)


def _gen_marker_candidates(marker_row, marker_col, height, width):
    candidates = []
    for row_offset in SOURCE_TOP_ROW_OFFSETS_FROM_MARKER:
        r0 = marker_row + row_offset
        if not (EDGE_MARGIN <= r0 and r0 + SPRITE_SIZE <= height - EDGE_MARGIN):
            continue
        for col_offset in range(SPRITE_SIZE):
            c0 = marker_col - col_offset
            if EDGE_MARGIN <= c0 and c0 + SPRITE_SIZE <= width - EDGE_MARGIN:
                box = (r0, c0, r0 + SPRITE_SIZE - 1, c0 + SPRITE_SIZE - 1)
                candidates.append(((marker_row, marker_col), box))
    random.shuffle(candidates)
    return candidates


def _gen_fits_with_chosen(candidate, chosen):
    point, box = candidate
    for prev_point, prev_box in chosen:
        if _gen_bbox_distance(box, prev_box) < MIN_BBOX_GAP:
            return False
        if _gen_point_bbox_distance(point, prev_box) < OTHER_BBOX_MARKER_MIN_DISTANCE:
            return False
        if _gen_point_bbox_distance(prev_point, box) < OTHER_BBOX_MARKER_MIN_DISTANCE:
            return False
    return True


def _gen_column_sets(marker_cols, count):
    col_sets = []
    for a in range(len(marker_cols)):
        for b in range(a + 1, len(marker_cols)):
            if count == 2:
                col_sets.append((marker_cols[a], marker_cols[b]))
            else:
                for c in range(b + 1, len(marker_cols)):
                    col_sets.append((marker_cols[a], marker_cols[b], marker_cols[c]))
    return col_sets


def _gen_search_layout(candidate_lists, chosen, idx=0):
    if idx == len(candidate_lists):
        return True
    for candidate in candidate_lists[idx]:
        if _gen_fits_with_chosen(candidate, chosen):
            chosen.append(candidate)
            if _gen_search_layout(candidate_lists, chosen, idx + 1):
                return True
            chosen.pop()
    return False


def _gen_choose_marker_layout(count, height, width):
    marker_rows = list(range(EDGE_MARGIN, height - EDGE_MARGIN))
    marker_cols = list(range(EDGE_MARGIN, width - EDGE_MARGIN))
    random.shuffle(marker_rows)

    col_sets = _gen_column_sets(marker_cols, count)
    random.shuffle(col_sets)

    for marker_row in marker_rows:
        for cols in col_sets:
            candidate_lists = [
                _gen_marker_candidates(marker_row, marker_col, height, width)
                for marker_col in cols
            ]
            if any(not candidates for candidates in candidate_lists):
                continue

            chosen = []
            if _gen_search_layout(candidate_lists, chosen):
                return chosen
    return None


def _gen_place_horizontal(masks, colors, height, width):
    layout = _gen_choose_marker_layout(len(masks), height, width)
    if layout is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for mask, (_point, box) in zip(masks, layout):
        r0, c0, _r1, _c1 = box
        paste(grid, _gen_sprite(mask, SOURCE_COLOR), r0, c0, transparent=BACKGROUND_COLOR)

    for color, (point, _box) in zip(colors, layout):
        r, c = point
        if grid[r, c] != BACKGROUND_COLOR:
            return None
        grid[r, c] = color
    return grid


def _gen_canvas_size(count, axis):
    axis_dim_min = MIN_AXIS_DIM_BY_COUNT[count]
    if axis == 1:
        height = random.randint(*GRID_SIZE_RANGE)
        width = random.randint(axis_dim_min, GRID_SIZE_RANGE[1])
    else:
        height = random.randint(axis_dim_min, GRID_SIZE_RANGE[1])
        width = random.randint(*GRID_SIZE_RANGE)
    return height, width


def _gen_attempt():
    count, axis = random.randint(*OBJECT_COUNT_RANGE), random.randint(0, 1)
    masks = _gen_distinct_masks(count)
    if masks is None:
        return None

    colors = random.sample(MARKER_COLORS, count)
    expected = _gen_output_from_masks(masks, colors, axis)

    height, width = _gen_canvas_size(count, axis)
    if axis == 1:
        grid = _gen_place_horizontal(masks, colors, height, width)
    else:
        canon = _gen_place_horizontal([mask.T for mask in masks], colors, width, height)
        grid = None if canon is None else canon.T.copy()

    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
