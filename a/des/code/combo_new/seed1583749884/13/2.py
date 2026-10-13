import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

BG = 8
PALETTE = tuple(color for color in range(10) if color != BG)
GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 27)
GRID_W_RANGE = (15, 27)
OBJECT_COUNT_RANGE = (4, 7)
PLACEMENT_MARGIN = 1
PLACEMENT_ATTEMPTS = 250
TIE_GAP_RANGE = (1, 4)


def _get_bg(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalize_coords(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, height, width


def _orient_coords(coords, rng):
    out = list(coords)
    if rng.random() < 0.5:
        out = [(c, r) for r, c in out]
    out, height, width = _normalize_coords(out)
    if rng.random() < 0.5:
        out = [(height - 1 - r, c) for r, c in out]
    if rng.random() < 0.5:
        out = [(r, width - 1 - c) for r, c in out]
    out, _, _ = _normalize_coords(out)
    return out


def _random_walk_shape(rng):
    canvas_h = int(rng.integers(4, 7))
    canvas_w = int(rng.integers(4, 7))
    target_area = int(rng.integers(3, min(9, canvas_h * canvas_w) + 1))
    start = (canvas_h // 2, canvas_w // 2)
    coords = {start}

    while len(coords) < target_area:
        frontier = []
        for r, c in coords:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        if not frontier:
            break
        frontier = list(dict.fromkeys(frontier))
        weights = []
        for cand in frontier:
            degree = 0
            for dr, dc in DIRS4:
                if (cand[0] + dr, cand[1] + dc) in coords:
                    degree += 1
            weights.append(1 + degree)
        choice = frontier[int(rng.choice(len(frontier), p=np.array(weights) / np.sum(weights)))]
        coords.add(choice)

    return _normalize_coords(coords)[0]


def _random_shape(rng):
    mode = str(rng.choice(["rect", "line", "ell", "walk"]))

    if mode == "rect":
        h = int(rng.integers(1, 5))
        w = int(rng.integers(1, 5))
        if h * w == 1:
            if rng.random() < 0.5:
                h = 2
            else:
                w = 2
        coords = [(r, c) for r in range(h) for c in range(w)]
    elif mode == "line":
        length = int(rng.integers(2, 7))
        coords = [(0, c) for c in range(length)]
    elif mode == "ell":
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 6))
        corner_row = 0 if rng.random() < 0.5 else h - 1
        coords = {(r, 0) for r in range(h)}
        coords.update((corner_row, c) for c in range(w))
        coords = list(coords)
    else:
        coords = _random_walk_shape(rng)

    return _orient_coords(coords, rng)


def _sample_shape_with_area(rng, min_area, max_area):
    for _ in range(80):
        shape = _random_shape(rng)
        if min_area <= len(shape) <= max_area:
            return shape
    return None


def _shape_dims(coords):
    _, h, w = _normalize_coords(coords)
    return h, w


def _can_place(shape_coords, top, left, occupied, grid_shape):
    for r, c in shape_coords:
        rr, cc = top + r, left + c
        if not _in_bounds(grid_shape, rr, cc):
            return False
        if occupied[rr, cc]:
            return False
        for dr, dc in DIRS8:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(grid_shape, nr, nc) and occupied[nr, nc]:
                return False
    return True


def _stamp(grid, occupied, shape_coords, top, left, color):
    for r, c in shape_coords:
        rr, cc = top + r, left + c
        grid[rr, cc] = color
        occupied[rr, cc] = True


def _place_random_object(grid, occupied, shape_coords, color, rng):
    h, w = _shape_dims(shape_coords)
    max_top = grid.shape[0] - PLACEMENT_MARGIN - h
    max_left = grid.shape[1] - PLACEMENT_MARGIN - w
    min_top = PLACEMENT_MARGIN
    min_left = PLACEMENT_MARGIN
    if max_top < min_top or max_left < min_left:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(rng.integers(min_top, max_top + 1))
        left = int(rng.integers(min_left, max_left + 1))
        if _can_place(shape_coords, top, left, occupied, grid.shape):
            _stamp(grid, occupied, shape_coords, top, left, color)
            return True
    return False


def _place_tie_triplet(grid, occupied, triplet, colors, rng):
    a_shape, b_shape, c_shape = triplet
    a_color, b_color, c_color = colors
    a_h, a_w = _shape_dims(a_shape)
    b_h, b_w = _shape_dims(b_shape)
    c_h, c_w = _shape_dims(c_shape)
    gap = int(rng.integers(TIE_GAP_RANGE[0], TIE_GAP_RANGE[1]))
    orientation = str(rng.choice(["horizontal", "vertical"]))

    if orientation == "horizontal":
        b_off = (a_h - b_h) // 2
        c_off = (a_h - c_h) // 2
        top_min = max(PLACEMENT_MARGIN, PLACEMENT_MARGIN - b_off, PLACEMENT_MARGIN - c_off)
        top_max = min(
            grid.shape[0] - PLACEMENT_MARGIN - a_h,
            grid.shape[0] - PLACEMENT_MARGIN - b_h - b_off,
            grid.shape[0] - PLACEMENT_MARGIN - c_h - c_off,
        )
        left_min = PLACEMENT_MARGIN + b_w + gap
        left_max = grid.shape[1] - PLACEMENT_MARGIN - a_w - gap - c_w
        if top_min > top_max or left_min > left_max:
            return False
        a_top = int(rng.integers(top_min, top_max + 1))
        a_left = int(rng.integers(left_min, left_max + 1))
        b_top, b_left = a_top + b_off, a_left - b_w - gap
        c_top, c_left = a_top + c_off, a_left + a_w + gap
    else:
        b_off = (a_w - b_w) // 2
        c_off = (a_w - c_w) // 2
        left_min = max(PLACEMENT_MARGIN, PLACEMENT_MARGIN - b_off, PLACEMENT_MARGIN - c_off)
        left_max = min(
            grid.shape[1] - PLACEMENT_MARGIN - a_w,
            grid.shape[1] - PLACEMENT_MARGIN - b_w - b_off,
            grid.shape[1] - PLACEMENT_MARGIN - c_w - c_off,
        )
        top_min = PLACEMENT_MARGIN + b_h + gap
        top_max = grid.shape[0] - PLACEMENT_MARGIN - a_h - gap - c_h
        if top_min > top_max or left_min > left_max:
            return False
        a_top = int(rng.integers(top_min, top_max + 1))
        a_left = int(rng.integers(left_min, left_max + 1))
        b_top, b_left = a_top - b_h - gap, a_left + b_off
        c_top, c_left = a_top + a_h + gap, a_left + c_off

    placements = [
        (a_shape, a_top, a_left, a_color),
        (b_shape, b_top, b_left, b_color),
        (c_shape, c_top, c_left, c_color),
    ]
    for shape_coords, top, left, _color in placements:
        if not _can_place(shape_coords, top, left, occupied, grid.shape):
            return False
    for shape_coords, top, left, color in placements:
        _stamp(grid, occupied, shape_coords, top, left, color)
    return True


def _find_components(grid, bg):
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coord_set = set(coords)
            boundary = []
            for cr, cc in coords:
                if any((cr + dr, cc + dc) not in coord_set for dr, dc in DIRS4):
                    boundary.append((cr, cc))
            anchor = (min(r0 for r0, _ in coords), min(c0 for _, c0 in coords))
            comps.append(
                {
                    "color": color,
                    "coords": sorted(coords),
                    "coord_set": coord_set,
                    "boundary": boundary,
                    "area": len(coords),
                    "anchor": anchor,
                }
            )

    comps.sort(key=lambda comp: comp["anchor"])
    return comps


def _boundary_gap(comp_a, comp_b):
    best = None
    for r1, c1 in comp_a["boundary"]:
        for r2, c2 in comp_b["boundary"]:
            gap = abs(r1 - r2) + abs(c1 - c2) - 1
            if best is None or gap < best:
                best = gap
    return max(int(best), 0)


def _analyze_neighbors(components):
    n = len(components)
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gap = _boundary_gap(components[i], components[j])
            distances[i][j] = gap
            distances[j][i] = gap

    analyses = []
    for i in range(n):
        candidate_ids = [j for j in range(n) if j != i]
        min_gap = min(distances[i][j] for j in candidate_ids)
        nearest = [j for j in candidate_ids if distances[i][j] == min_gap]
        chosen = min(
            nearest,
            key=lambda j: (
                -components[j]["area"],
                components[j]["anchor"][0],
                components[j]["anchor"][1],
                j,
            ),
        )
        analyses.append(
            {
                "nearest_index": chosen,
                "nearest_candidates": nearest,
                "min_gap": min_gap,
            }
        )
    return analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object with the color of its spatially nearest non-touching neighbor.

    Input structure:
    - The background is Azure and occupies the empty space between objects.
    - The foreground consists of several 4-connected monochrome objects in distinct positions.
    - Objects do not overlap or touch, but their separations and areas can differ.

    Rule:
    - Identify every foreground object as a separate 4-connected monochrome component.
    - For each object A, find the other object B with the smallest gap between their boundary pixels, measured as the minimum Manhattan interval through background cells.
    - If several objects are tied as nearest to A, choose the one with the largest area.
    - Recolor all pixels of A to B's original color, applying all replacements simultaneously.

    Key insight:
    - The transformation depends on spatial proximity between whole objects, not on matching shape, size, or the numeric identities of the colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) >= 2, "expected at least two foreground objects"

    analyses = _analyze_neighbors(components)
    out = g.copy()
    for comp, info in zip(components, analyses):
        target_color = int(components[info["nearest_index"]]["color"])
        for r, c in comp["coords"]:
            out[r, c] = target_color
    return out


def _valid_generated_grid(grid, require_tie):
    components = _find_components(grid, BG)
    if not (OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]):
        return False

    analyses = _analyze_neighbors(components)
    has_area_tiebreak = False
    for info in analyses:
        nearest = info["nearest_candidates"]
        if len(nearest) <= 1:
            continue
        candidate_areas = [components[j]["area"] for j in nearest]
        best_area = max(candidate_areas)
        if candidate_areas.count(best_area) != 1:
            return False
        if candidate_areas.count(best_area) == 1:
            has_area_tiebreak = True

    if require_tie and not has_area_tiebreak:
        return False
    if not require_tie and any(len(info["nearest_candidates"]) > 1 for info in analyses):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if len(np.unique(out[out != BG])) < 2:
        return False
    return True


def _gen_attempt(rng):
    height = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    width = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    require_tie = bool(rng.integers(0, 2))

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    colors = [int(c) for c in rng.choice(PALETTE, size=count, replace=False)]

    next_color = 0
    if require_tie:
        a_shape = _sample_shape_with_area(rng, 3, 6)
        b_shape = _sample_shape_with_area(rng, 6, 10)
        c_shape = _sample_shape_with_area(rng, 2, 5)
        if a_shape is None or b_shape is None or c_shape is None:
            return None

        ok = _place_tie_triplet(
            grid,
            occupied,
            (a_shape, b_shape, c_shape),
            colors[:3],
            rng,
        )
        if not ok:
            return None
        next_color = 3

    while next_color < count:
        shape = _random_shape(rng)
        if not _place_random_object(grid, occupied, shape, colors[next_color], rng):
            return None
        next_color += 1

    if not _valid_generated_grid(grid, require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid proximity-recolor ARC sample")
