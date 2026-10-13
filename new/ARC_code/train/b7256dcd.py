import numpy as np

from arc_utils import get_bg, label_mask, neighbors4


def _neutral_color(g: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(g, return_counts=True)
    foreground = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != bg]
    assert foreground, "expected foreground pixels"

    top_count = max(count for _color, count in foreground)
    candidates = [color for color, count in foreground if count == top_count]
    assert len(candidates) == 1, "neutral mask color must be the unique most frequent foreground color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each colored marker to its adjacent neutral mask component, then erase the markers.

    Input structure
    - The background is the most common color.
    - The neutral mask color is the unique most frequent foreground color.
    - Other foreground colors are marker pixels that may touch neutral 4-connected mask components.
    Rule
    - Remove every marker pixel by recoloring it to the background.
    - For each 4-connected neutral mask component, find marker colors touching it by edge adjacency.
    - If a component touches one marker color, recolor the whole neutral component to that marker color.
    - If a component touches no marker color, leave it neutral.
    Key insight
    - Marker pixels label neighboring neutral regions but are not part of the final colored objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    neutral = _neutral_color(g, bg)
    marker_mask = (g != bg) & (g != neutral)
    assert np.any(marker_mask), "expected at least one marker pixel"

    out = g.copy()
    out[marker_mask] = bg

    labels, n_labels = label_mask(g == neutral, connectivity=4)
    assert n_labels > 0, "expected neutral mask components"
    H, W = g.shape
    recolored_any = False

    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        touching_colors = set()
        for r, c in coords:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and marker_mask[rr, cc]:
                    touching_colors.add(int(g[rr, cc]))

        if not touching_colors:
            continue
        assert len(touching_colors) == 1, "each neutral component must touch at most one marker color"
        out[labels == label_id] = touching_colors.pop()
        recolored_any = True

    assert recolored_any, "expected at least one labeled neutral component"
    return out

# --------------- generator (att2) ---------------
# Canvas, object, and foreground/background constraints.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (5, 8), (4, 8)
COMPONENT_COUNT_RANGE, COMPONENT_SIZE_RANGE = (2, 4), (2, 6)
UNASSIGNED_COMPONENT_PROB, DISTRACTOR_MARKER_PROB = 0.35, 0.35
MIN_ASSIGNED_MARKERS = 2
MIN_TOTAL_MARKERS, MAX_TOTAL_MARKERS = 2, 4
MIN_BG_FRACTION, MAX_BG_FRACTION = 0.52, 0.82
MIN_NEUTRAL_PIXELS, COLOR_VALUES = 2, range(10)
GENERATION_ATTEMPTS, POLYOMINO_CONNECTIVITY = 300, 4


def _gen_choice(items):
    return items[int(np.random.randint(len(items)))]


def _gen_color_plan(marker_count):
    colors = list(COLOR_VALUES)
    np.random.shuffle(colors)
    return colors[0], colors[1], colors[2 : 2 + marker_count]


def _gen_assignment(n_components):
    unassigned_count = 1 if np.random.random() < UNASSIGNED_COMPONENT_PROB else 0
    assigned_count = n_components - unassigned_count
    distractor_count = 1 if np.random.random() < DISTRACTOR_MARKER_PROB else 0
    if assigned_count < MIN_ASSIGNED_MARKERS:
        distractor_count = 1
    marker_count = assigned_count + distractor_count
    ids = list(range(n_components))
    np.random.shuffle(ids)
    unassigned = set(ids[:unassigned_count])
    assigned = [idx for idx in range(n_components) if idx not in unassigned]
    return assigned, unassigned, distractor_count, marker_count


def _gen_size_plan(area, n_components, marker_count):
    plans = []
    min_size, max_size = COMPONENT_SIZE_RANGE
    for a in range(min_size, max_size):
        for b in range(min_size, max_size):
            if n_components == 2:
                sizes = (a, b)
                total = a + b
                bg_count = area - total - marker_count
                bg_fraction = bg_count / area
                if bg_count > total and MIN_BG_FRACTION <= bg_fraction <= MAX_BG_FRACTION:
                    plans.append(sizes)
            else:
                for c in range(min_size, max_size):
                    sizes = (a, b, c)
                    total = a + b + c
                    bg_count = area - total - marker_count
                    bg_fraction = bg_count / area
                    if bg_count > total and MIN_BG_FRACTION <= bg_fraction <= MAX_BG_FRACTION:
                        plans.append(sizes)
    if not plans:
        return None
    sizes = list(_gen_choice(plans))
    np.random.shuffle(sizes)
    return sizes


def _gen_allowed_mask(height, width, blocked):
    allowed = np.ones((height, width), dtype=bool)
    for br, bc in blocked:
        for rr, cc in neighbors4(int(br), int(bc)):
            if 0 <= rr < height and 0 <= cc < width:
                allowed[rr, cc] = False
        allowed[int(br), int(bc)] = False
    return allowed


def _gen_polyomino(height, width, size, blocked):
    allowed = _gen_allowed_mask(height, width, blocked)
    labels, n_labels = label_mask(allowed, connectivity=POLYOMINO_CONNECTIVITY)
    regions = [np.argwhere(labels == i) for i in range(1, n_labels + 1)]
    regions = [region for region in regions if len(region) >= size]
    if not regions:
        return None

    region = _gen_choice(regions)
    region_cells = {(int(r), int(c)) for r, c in region}
    start_r, start_c = region[int(np.random.randint(len(region)))]
    component = {(int(start_r), int(start_c))}
    frontier = {
        (rr, cc)
        for rr, cc in neighbors4(int(start_r), int(start_c))
        if (rr, cc) in region_cells
    }

    while len(component) < size:
        frontier = {cell for cell in frontier if cell not in component}
        if not frontier:
            return None
        cell = _gen_choice(list(frontier))
        component.add(cell)
        for rr, cc in neighbors4(cell[0], cell[1]):
            if (rr, cc) in region_cells and (rr, cc) not in component:
                frontier.add((rr, cc))
    return sorted(component)


def _gen_touching_components(cell, component_sets):
    touched = []
    r, c = cell
    for idx, cells in enumerate(component_sets):
        if any((rr, cc) in cells for rr, cc in neighbors4(r, c)):
            touched.append(idx)
    return touched


def _gen_marker_candidates(height, width, occupied, component_sets, target_idx):
    candidates = []
    for r in range(height):
        for c in range(width):
            if (r, c) in occupied:
                continue
            if _gen_touching_components((r, c), component_sets) == [target_idx]:
                candidates.append((r, c))
    return candidates


def _gen_distractor_candidates(height, width, occupied, component_sets):
    candidates = []
    for r in range(height):
        for c in range(width):
            if (r, c) in occupied:
                continue
            if _gen_touching_components((r, c), component_sets):
                continue
            if any((rr, cc) in occupied for rr, cc in neighbors4(r, c)):
                continue
            candidates.append((r, c))
    return candidates


def _gen_render_grids(height, width, bg, neutral, components, assigned, marker_colors):
    out = np.full((height, width), bg, dtype=int)
    input_grid = out.copy()
    assigned_colors = {idx: color for idx, color in zip(assigned, marker_colors[: len(assigned)])}

    for idx, component in enumerate(components):
        color = assigned_colors.get(idx, neutral)
        for r, c in component:
            out[r, c] = color
            input_grid[r, c] = neutral
    return input_grid, out, assigned_colors


def _gen_place_markers(input_grid, height, width, blocked, component_sets, assigned, assigned_colors,
                       distractor_count, marker_colors):
    occupied = set(blocked)
    marker_positions = []
    for idx in assigned:
        candidates = _gen_marker_candidates(height, width, occupied, component_sets, idx)
        if not candidates:
            return None
        r, c = _gen_choice(candidates)
        color = assigned_colors[idx]
        input_grid[r, c] = color
        occupied.add((r, c))
        marker_positions.append((r, c))

    if distractor_count:
        candidates = _gen_distractor_candidates(height, width, occupied, component_sets)
        if not candidates:
            return None
        r, c = _gen_choice(candidates)
        input_grid[r, c] = marker_colors[-1]
        occupied.add((r, c))
        marker_positions.append((r, c))
    return marker_positions


def _gen_is_valid(input_grid, out, bg, neutral, marker_positions):
    neutral_count = int(np.sum(input_grid == neutral))
    bg_count = int(np.sum(input_grid == bg))
    if neutral_count < MIN_NEUTRAL_PIXELS or bg_count <= neutral_count:
        return False
    if not (MIN_TOTAL_MARKERS <= len(marker_positions) <= MAX_TOTAL_MARKERS):
        return False

    try:
        actual = main(input_grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, out))


def _gen_attempt():
    height = int(np.random.randint(*GRID_HEIGHT_RANGE))
    width = int(np.random.randint(*GRID_WIDTH_RANGE))
    n_components = int(np.random.randint(*COMPONENT_COUNT_RANGE))

    assigned, _unassigned, distractor_count, marker_count = _gen_assignment(n_components)
    sizes = _gen_size_plan(height * width, n_components, marker_count)
    if sizes is None:
        return None

    bg, neutral, marker_colors = _gen_color_plan(marker_count)
    components = []
    blocked = set()
    for size in sizes:
        component = _gen_polyomino(height, width, size, blocked)
        if component is None:
            return None
        components.append(component)
        blocked.update(component)

    input_grid, out, assigned_colors = _gen_render_grids(
        height, width, bg, neutral, components, assigned, marker_colors
    )
    component_sets = [set(component) for component in components]
    marker_positions = _gen_place_markers(
        input_grid,
        height,
        width,
        blocked,
        component_sets,
        assigned,
        assigned_colors,
        distractor_count,
        marker_colors,
    )
    if marker_positions is None:
        return None
    if not _gen_is_valid(input_grid, out, bg, neutral, marker_positions):
        return None
    return input_grid


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
