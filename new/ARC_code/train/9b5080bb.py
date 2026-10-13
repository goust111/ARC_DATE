import numpy as np

from arc_utils import components_mask, neighbors4
import random


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _neighbor_colors(g: np.ndarray, coords: np.ndarray, color: int) -> set[int]:
    coord_set = {tuple(map(int, p)) for p in coords}
    colors = set()
    for r, c in coord_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and (nr, nc) not in coord_set:
                if int(g[nr, nc]) != color:
                    colors.add(int(g[nr, nc]))
    return colors


def _object_components(g: np.ndarray) -> tuple[int, list[np.ndarray], set[int]]:
    candidates = []
    for color in set(map(int, g.ravel())):
        inner_components = [
            coords
            for coords in components_mask(g == color, connectivity=4)
            if len(coords) >= 5 and not _touches_boundary(coords, g.shape)
        ]
        if len(inner_components) < 2:
            continue

        panel_colors = set()
        for coords in inner_components:
            neighbors = _neighbor_colors(g, coords, color)
            if len(neighbors) != 1:
                break
            panel_colors.update(neighbors)
        else:
            if len(panel_colors) == 2:
                candidates.append((color, inner_components, panel_colors))

    assert len(candidates) == 1
    return candidates[0]


def _valid_core_rect(
    object_cells: set[tuple[int, int]],
    missing: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    rect: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = rect

    for r, c in missing:
        distances = [r - r0, r1 - r, c - c0, c1 - c]
        if distances.count(min(distances)) != 1:
            return False

    for r, c in outside:
        side_count = int(r < r0) + int(r > r1) + int(c < c0) + int(c > c1)
        if side_count != 1:
            return False
        if r < r0:
            inner = (r + 1, c)
        elif r > r1:
            inner = (r - 1, c)
        elif c < c0:
            inner = (r, c + 1)
        else:
            inner = (r, c - 1)
        if inner not in object_cells:
            return False

    return True


def _core_rect(coords: np.ndarray) -> tuple[int, int, int, int]:
    object_cells = {tuple(map(int, p)) for p in coords}
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    candidates = []
    for rr0 in (r0, r0 + 1):
        for rr1 in (r1, r1 - 1):
            for cc0 in (c0, c0 + 1):
                for cc1 in (c1, c1 - 1):
                    if rr0 > rr1 or cc0 > cc1:
                        continue
                    rect_cells = {
                        (r, c)
                        for r in range(rr0, rr1 + 1)
                        for c in range(cc0, cc1 + 1)
                    }
                    missing = rect_cells - object_cells
                    outside = object_cells - rect_cells
                    rect = (rr0, cc0, rr1, cc1)
                    if not _valid_core_rect(object_cells, missing, outside, rect):
                        continue
                    score = (len(missing) + len(outside), -len(rect_cells))
                    candidates.append((score, rect))

    assert candidates
    best_score = min(score for score, _ in candidates)
    best_rects = [rect for score, rect in candidates if score == best_score]
    assert len(best_rects) == 1
    return best_rects[0]


def _nearest_rect_side(r: int, c: int, rect: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = rect
    distances = {
        "top": r - r0,
        "bottom": r1 - r,
        "left": c - c0,
        "right": c1 - c,
    }
    nearest = [side for side, dist in distances.items() if dist == min(distances.values())]
    assert len(nearest) == 1
    return nearest[0]


def _defect_bar_cells(coords: np.ndarray, rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    object_cells = {tuple(map(int, p)) for p in coords}
    r0, c0, r1, c1 = rect
    rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    bars = set()

    for r, c in rect_cells - object_cells:
        side = _nearest_rect_side(r, c, rect)
        if side == "top":
            bars.update({(r - 1, c), (r, c)})
        elif side == "bottom":
            bars.update({(r, c), (r + 1, c)})
        elif side == "left":
            bars.update({(r, c - 1), (r, c)})
        else:
            bars.update({(r, c), (r, c + 1)})

    for r, c in object_cells - rect_cells:
        if r < r0:
            bars.update({(r, c), (r + 1, c)})
        elif r > r1:
            bars.update({(r - 1, c), (r, c)})
        elif c < c0:
            bars.update({(r, c), (r, c + 1)})
        else:
            assert c > c1
            bars.update({(r, c - 1), (r, c)})

    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the small defect bars on each shared-color object with the opposite panel color.

    Input structure
    - The grid contains two large panel colors and one shared object color.
    - Each shared-color object is a mostly solid rectangle inside exactly one panel.
    Rule
    - Find the near-solid rectangular core of each object.
    - Missing core pixels and one-cell protrusions define two-pixel bars perpendicular to the nearest core edge.
    - Recolor those bars with the other panel color; leave all other pixels unchanged.
    Key insight
    - The colors to add are not tied to numeric ids: each object's surrounding panel identifies the opposite panel color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    object_color, components, panel_colors = _object_components(g)
    for coords in components:
        neighbors = _neighbor_colors(g, coords, object_color)
        assert len(neighbors) == 1
        panel_color = next(iter(neighbors))
        target_colors = panel_colors - {panel_color}
        assert len(target_colors) == 1
        target_color = next(iter(target_colors))

        rect = _core_rect(coords)
        for r, c in _defect_bar_cells(coords, rect):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert int(g[r, c]) in {object_color, panel_color}
            out[r, c] = target_color

    return out

# --------------- generator (att1) ---------------

# Palette and canvas bounds.
COLOR_IDS = list(range(10))
GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 27

# Panel layout bounds and options.
MIN_PANEL_DIM = 9
MIN_VISIBLE_BORDERS = 2
BORDER_CHOICES = (0, 1)
OFFSET_MIN = 2
OFFSET_MAX = 6
HORIZONTAL_LAYOUT = 0
VERTICAL_LAYOUT = 1
HORIZONTAL_OFFSETS = ("left", "right", "none")
VERTICAL_OFFSETS = ("top", "bottom", "none")

# Object geometry and placement constraints.
OBJECTS_PER_PANEL = (1, 2)
CORE_H_MIN = 3
CORE_H_MAX = 7
CORE_W_MIN = 3
CORE_W_MAX = 9
DEFECT_COUNT_MIN = 1
DEFECT_COUNT_MAX = 3
DEFECT_SIDES = ("top", "bottom", "left", "right")
DEFECT_TYPES = ("missing", "protrusion")
PANEL_MARGIN = 1
OBJECT_GAP = 2
MAX_PLACE_TRIES = 30

# Generation retry budget.
MAX_GENERATE_ATTEMPTS = 300


def _choose_palette() -> tuple[int, int, int, int]:
    colors = COLOR_IDS[:]
    random.shuffle(colors)
    return tuple(colors[:4])


def _choose_borders() -> tuple[int, int, int, int]:
    borders = [random.choice(BORDER_CHOICES) for _ in range(4)]
    while sum(borders) < MIN_VISIBLE_BORDERS:
        borders[random.randint(0, 3)] = 1
    return tuple(borders)


def _build_horizontal_panels(
    grid: np.ndarray,
    panel_a_color: int,
    panel_b_color: int,
) -> list[tuple[int, int, int, int, int]] | None:
    """Create two panels sharing a horizontal border; returns panel records."""
    H, W = grid.shape
    split_r = random.randint(MIN_PANEL_DIM, H - MIN_PANEL_DIM - 1)
    border_top, border_bot, border_left, border_right = _choose_borders()

    pa_height = split_r - border_top
    pb_height = H - border_bot - split_r
    if pa_height < MIN_PANEL_DIM or pb_height < MIN_PANEL_DIM:
        return None

    offset_side = random.choice(HORIZONTAL_OFFSETS)
    if offset_side == "none":
        pa_c0, pa_c1 = border_left, W - 1 - border_right
        pb_c0, pb_c1 = border_left, W - 1 - border_right
    elif offset_side == "left":
        offset_cols = random.randint(OFFSET_MIN, min(OFFSET_MAX, W - 2 * MIN_PANEL_DIM))
        if random.random() < 0.5:
            pa_c0 = border_left
            pa_c1 = W - 1 - border_right
            pb_c0 = border_left + offset_cols
            pb_c1 = W - 1 - border_right
        else:
            pa_c0 = border_left + offset_cols
            pa_c1 = W - 1 - border_right
            pb_c0 = border_left
            pb_c1 = W - 1 - border_right
    else:
        offset_cols = random.randint(OFFSET_MIN, min(OFFSET_MAX, W - 2 * MIN_PANEL_DIM))
        if random.random() < 0.5:
            pa_c0 = border_left
            pa_c1 = W - 1 - border_right - offset_cols
            pb_c0 = border_left
            pb_c1 = W - 1 - border_right
        else:
            pa_c0 = border_left
            pa_c1 = W - 1 - border_right
            pb_c0 = border_left
            pb_c1 = W - 1 - border_right - offset_cols

    if pa_c1 - pa_c0 + 1 < MIN_PANEL_DIM or pb_c1 - pb_c0 + 1 < MIN_PANEL_DIM:
        return None
    if min(pa_c1, pb_c1) < max(pa_c0, pb_c0):
        return None

    pa_r0, pa_r1 = border_top, split_r - 1
    pb_r0, pb_r1 = split_r, H - 1 - border_bot
    grid[pa_r0:pa_r1 + 1, pa_c0:pa_c1 + 1] = panel_a_color
    grid[pb_r0:pb_r1 + 1, pb_c0:pb_c1 + 1] = panel_b_color
    return [
        (panel_a_color, pa_r0, pa_r1, pa_c0, pa_c1),
        (panel_b_color, pb_r0, pb_r1, pb_c0, pb_c1),
    ]


def _build_vertical_panels(
    grid: np.ndarray,
    panel_a_color: int,
    panel_b_color: int,
) -> list[tuple[int, int, int, int, int]] | None:
    """Create two panels sharing a vertical border; returns panel records."""
    H, W = grid.shape
    split_c = random.randint(MIN_PANEL_DIM, W - MIN_PANEL_DIM - 1)
    border_top, border_bot, border_left, border_right = _choose_borders()

    pa_width = split_c - border_left
    pb_width = W - border_right - split_c
    if pa_width < MIN_PANEL_DIM or pb_width < MIN_PANEL_DIM:
        return None

    offset_side = random.choice(VERTICAL_OFFSETS)
    if offset_side == "none":
        pa_r0, pa_r1 = border_top, H - 1 - border_bot
        pb_r0, pb_r1 = border_top, H - 1 - border_bot
    elif offset_side == "top":
        offset_rows = random.randint(OFFSET_MIN, min(OFFSET_MAX, H - 2 * MIN_PANEL_DIM))
        if random.random() < 0.5:
            pa_r0, pa_r1 = border_top, H - 1 - border_bot
            pb_r0, pb_r1 = border_top + offset_rows, H - 1 - border_bot
        else:
            pa_r0, pa_r1 = border_top + offset_rows, H - 1 - border_bot
            pb_r0, pb_r1 = border_top, H - 1 - border_bot
    else:
        offset_rows = random.randint(OFFSET_MIN, min(OFFSET_MAX, H - 2 * MIN_PANEL_DIM))
        if random.random() < 0.5:
            pa_r0, pa_r1 = border_top, H - 1 - border_bot - offset_rows
            pb_r0, pb_r1 = border_top, H - 1 - border_bot
        else:
            pa_r0, pa_r1 = border_top, H - 1 - border_bot
            pb_r0, pb_r1 = border_top, H - 1 - border_bot - offset_rows

    pa_c0, pa_c1 = border_left, split_c - 1
    pb_c0, pb_c1 = split_c, W - 1 - border_right
    if pa_r1 - pa_r0 + 1 < MIN_PANEL_DIM or pb_r1 - pb_r0 + 1 < MIN_PANEL_DIM:
        return None
    if min(pa_r1, pb_r1) < max(pa_r0, pb_r0):
        return None

    grid[pa_r0:pa_r1 + 1, pa_c0:pa_c1 + 1] = panel_a_color
    grid[pb_r0:pb_r1 + 1, pb_c0:pb_c1 + 1] = panel_b_color
    return [
        (panel_a_color, pa_r0, pa_r1, pa_c0, pa_c1),
        (panel_b_color, pb_r0, pb_r1, pb_c0, pb_c1),
    ]


def _make_panel_grid(
    panel_a_color: int,
    panel_b_color: int,
    bg_color: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]] | None:
    """Sample the canvas and panel layout."""
    H = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    grid = np.full((H, W), bg_color, dtype=int)

    orientation = random.randint(HORIZONTAL_LAYOUT, VERTICAL_LAYOUT)
    if orientation == HORIZONTAL_LAYOUT:
        panels = _build_horizontal_panels(grid, panel_a_color, panel_b_color)
    else:
        panels = _build_vertical_panels(grid, panel_a_color, panel_b_color)
    if panels is None:
        return None
    return grid, panels


def _object_overlaps(full_bbox: tuple[int, int, int, int], placed_bboxes: list[tuple[int, int, int, int]]) -> bool:
    full_r0, full_r1, full_c0, full_c1 = full_bbox
    for br0, br1, bc0, bc1 in placed_bboxes:
        if (
            full_r0 - OBJECT_GAP <= br1
            and full_r1 + OBJECT_GAP >= br0
            and full_c0 - OBJECT_GAP <= bc1
            and full_c1 + OBJECT_GAP >= bc0
        ):
            return True
    return False


def _place_core(
    panel_bbox: tuple[int, int, int, int],
    core_h: int,
    core_w: int,
    sides: list[str],
    placed_bboxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Find a non-overlapping core bbox and its full bbox including bars."""
    pr0, pr1, pc0, pc1 = panel_bbox
    extra_top = 1 if "top" in sides else 0
    extra_bot = 1 if "bottom" in sides else 0
    extra_left = 1 if "left" in sides else 0
    extra_right = 1 if "right" in sides else 0

    place_r0 = pr0 + PANEL_MARGIN + extra_top
    place_r1 = pr1 - PANEL_MARGIN - extra_bot - core_h + 1
    place_c0 = pc0 + PANEL_MARGIN + extra_left
    place_c1 = pc1 - PANEL_MARGIN - extra_right - core_w + 1
    if place_r1 < place_r0 or place_c1 < place_c0:
        return None

    for _ in range(MAX_PLACE_TRIES):
        cr0 = random.randint(place_r0, place_r1)
        cc0 = random.randint(place_c0, place_c1)
        cr1 = cr0 + core_h - 1
        cc1 = cc0 + core_w - 1
        full_bbox = (cr0 - extra_top, cr1 + extra_bot, cc0 - extra_left, cc1 + extra_right)
        if not _object_overlaps(full_bbox, placed_bboxes):
            return (cr0, cr1, cc0, cc1), full_bbox
    return None


def _apply_defect(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    core_bbox: tuple[int, int, int, int],
    side: str,
    defect_type: str,
    panel_color: int,
    obj_color: int,
    other_panel_color: int,
) -> bool:
    """Apply one sampled input defect and its two-cell output bar."""
    cr0, cr1, cc0, cc1 = core_bbox
    if side == "top":
        if defect_type == "missing":
            valid_cols = list(range(cc0 + 1, cc1))
            if not valid_cols:
                return False
            dc = random.choice(valid_cols)
            dr = cr0
            input_grid[dr, dc] = panel_color
            output_grid[dr - 1, dc] = other_panel_color
            output_grid[dr, dc] = other_panel_color
        else:
            dc = random.randint(cc0, cc1)
            dr = cr0 - 1
            input_grid[dr, dc] = obj_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr + 1, dc] = other_panel_color
    elif side == "bottom":
        if defect_type == "missing":
            valid_cols = list(range(cc0 + 1, cc1))
            if not valid_cols:
                return False
            dc = random.choice(valid_cols)
            dr = cr1
            input_grid[dr, dc] = panel_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr + 1, dc] = other_panel_color
        else:
            dc = random.randint(cc0, cc1)
            dr = cr1 + 1
            input_grid[dr, dc] = obj_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr - 1, dc] = other_panel_color
    elif side == "left":
        if defect_type == "missing":
            valid_rows = list(range(cr0 + 1, cr1))
            if not valid_rows:
                return False
            dr = random.choice(valid_rows)
            dc = cc0
            input_grid[dr, dc] = panel_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr, dc - 1] = other_panel_color
        else:
            dr = random.randint(cr0, cr1)
            dc = cc0 - 1
            input_grid[dr, dc] = obj_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr, dc + 1] = other_panel_color
    else:
        if defect_type == "missing":
            valid_rows = list(range(cr0 + 1, cr1))
            if not valid_rows:
                return False
            dr = random.choice(valid_rows)
            dc = cc1
            input_grid[dr, dc] = panel_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr, dc + 1] = other_panel_color
        else:
            dr = random.randint(cr0, cr1)
            dc = cc1 + 1
            input_grid[dr, dc] = obj_color
            output_grid[dr, dc] = other_panel_color
            output_grid[dr, dc - 1] = other_panel_color
    return True


def _place_panel_objects(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    panel: tuple[int, int, int, int, int],
    n_objs: int,
    obj_color: int,
    other_panel_color: int,
) -> bool:
    """Place all objects for one panel into input and output grids."""
    panel_color, pr0, pr1, pc0, pc1 = panel
    avail_r0 = pr0 + PANEL_MARGIN
    avail_r1 = pr1 - PANEL_MARGIN
    avail_c0 = pc0 + PANEL_MARGIN
    avail_c1 = pc1 - PANEL_MARGIN
    placed_bboxes = []

    for _ in range(n_objs):
        core_h = random.randint(CORE_H_MIN, min(CORE_H_MAX, avail_r1 - avail_r0 - 2))
        core_w = random.randint(CORE_W_MIN, min(CORE_W_MAX, avail_c1 - avail_c0 - 2))
        if core_h < CORE_H_MIN or core_w < CORE_W_MIN:
            return False

        n_defects = random.randint(DEFECT_COUNT_MIN, DEFECT_COUNT_MAX)
        sides = random.sample(DEFECT_SIDES, n_defects)
        defect_types = [random.choice(DEFECT_TYPES) for _ in range(n_defects)]

        placed = _place_core((pr0, pr1, pc0, pc1), core_h, core_w, sides, placed_bboxes)
        if placed is None:
            return False
        core_bbox, full_bbox = placed
        placed_bboxes.append(full_bbox)

        cr0, cr1, cc0, cc1 = core_bbox
        output_grid[cr0:cr1 + 1, cc0:cc1 + 1] = obj_color
        input_grid[cr0:cr1 + 1, cc0:cc1 + 1] = obj_color

        for side, defect_type in zip(sides, defect_types):
            if not _apply_defect(
                input_grid,
                output_grid,
                core_bbox,
                side,
                defect_type,
                panel_color,
                obj_color,
                other_panel_color,
            ):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    obj_color, panel_a_color, panel_b_color, bg_color = _choose_palette()
    layout = _make_panel_grid(panel_a_color, panel_b_color, bg_color)
    if layout is None:
        return None
    grid, panels = layout

    n_obj_a = random.choice(OBJECTS_PER_PANEL)
    n_obj_b = random.choice(OBJECTS_PER_PANEL)
    if not 2 <= n_obj_a + n_obj_b <= 4:
        return None

    output_grid = grid.copy()
    input_grid = grid.copy()
    object_counts = (n_obj_a, n_obj_b)
    other_colors = (panel_b_color, panel_a_color)

    for panel, n_objs, other_panel_color in zip(panels, object_counts, other_colors):
        if not _place_panel_objects(
            input_grid,
            output_grid,
            panel,
            n_objs,
            obj_color,
            other_panel_color,
        ):
            return None

    # Verify: run solver on input and check it produces the output
    try:
        result = main(input_grid)
        if np.array_equal(result, output_grid):
            return input_grid
    except (AssertionError, IndexError, ValueError):
        pass

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
