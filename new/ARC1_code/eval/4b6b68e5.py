from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, find_components, get_bg, neighbors4, paste
import random


def _enclosed_region_coords(wall_coords: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = map(int, box)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0])
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0]) for coords in components_mask(enclosed, connectivity=4)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed outline with the majority marker color inside it, while removing stray marker pixels.

    Input structure
    - The background is the dominant color.
    - Foreground consists of 4-connected monochrome outline objects and isolated one-pixel marker/control pixels.
    - Some outline objects enclose interior regions; open outlines have no enclosed region.

    Rule
    - Keep every multi-pixel outline object.
    - Remove all isolated marker pixels from the canvas.
    - For each enclosed interior region, fill all of its cells with the unique most frequent marker color found inside it.

    Key insight
    - The target to fill is the set of cells unreachable from the outline bbox boundary without crossing that outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected foreground components"

    out = np.full_like(g, bg)
    marker_mask = np.zeros(g.shape, dtype=bool)
    outlines: list[dict] = []

    for comp in components:
        coords = comp["coords"]
        if comp["area"] == 1:
            marker_mask[coords[0, 0], coords[0, 1]] = True
        else:
            outlines.append(comp)
            out[coords[:, 0], coords[:, 1]] = int(comp["color"])

    assert outlines, "expected at least one outline object"

    filled_regions = 0
    for outline in outlines:
        for region_coords in _enclosed_region_coords(outline["coords"], outline["bbox"]):
            rr, cc = region_coords[:, 0], region_coords[:, 1]
            inside_markers = marker_mask[rr, cc]
            marker_values = g[rr[inside_markers], cc[inside_markers]]
            assert marker_values.size > 0, "expected enclosed region to contain marker pixels"

            counts = Counter(map(int, marker_values))
            ranked = counts.most_common()
            assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority marker color"

            out[rr, cc] = int(ranked[0][0])
            filled_regions += 1

    assert filled_regions > 0, "expected at least one enclosed region to fill"
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, outline-count, and retry bounds.
GEN_GRID_ROWS, GEN_GRID_COLS = (16, 26), (14, 24)
GEN_COLORS = range(1, 10)
CLOSED_OUTLINE_COUNT, OPEN_OUTLINE_COUNT = (1, 3), (0, 1)
OUTLINE_BBOX_GAP = 3
CLOSED_PATCH_ATTEMPTS, OPEN_PATCH_ATTEMPTS, GENERATE_ATTEMPTS = 80, 60, 300

# Closed-loop and open-path geometry.
CLOSED_SIZE_BY_COUNT = {1: ((6, 13), (6, 11)), 2: ((6, 11), (6, 10)), 3: ((6, 9), (6, 8))}
CLOSED_AREA, CLOSED_BASE_AREA_MAX, CLOSED_TWO_NOTCH_AREA_MAX = (10, 46), 44, 42
CLOSED_NOTCH_COUNT, NOTCH_SPAN = (1, 2), (2, 4)
NOTCH_SIDES = ["top", "bottom", "left", "right"]
OPEN_SIZE, OPEN_AREA = ((5, 9), (7, 11)), (10, 46)
OPEN_JOG_CHOICES = [1, 2]
OPEN_BBOX_ROWS, OPEN_BBOX_COLS = (4, 13), (5, 11)

# Marker counts and adjacency spacing.
MINORITY_MARKER_TOTAL, SINGLE_MINORITY_PROB = (1, 2), 0.5
MAJORITY_MARKERS, MARKERS_PER_REGION_MAX = (2, 6), 7
TOTAL_MARKERS, STRAY_MARKERS_MAX = (7, 32), 14
FOUR_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

def _gen_add_segment(cells, r0, c0, r1, c1):
    if r0 == r1:
        a, b = sorted((int(c0), int(c1)))
        for c in range(a, b + 1):
            cells.add((int(r0), c))
    else:
        a, b = sorted((int(r0), int(r1)))
        for r in range(a, b + 1):
            cells.add((r, int(c0)))


def _gen_apply_notch(cells, h, w, side, depth_limit):
    depth_limit = max(1, int(depth_limit))
    if side in ("top", "bottom"):
        span = random.randint(NOTCH_SPAN[0], min(NOTCH_SPAN[1], w - 3))
        a = random.randint(1, w - span - 2)
        b = a + span
        depth = random.randint(1, min(depth_limit, max(1, (h - 3) // 2)))
        if side == "top":
            for c in range(a + 1, b):
                cells.discard((0, c))
            _gen_add_segment(cells, 0, a, depth, a)
            _gen_add_segment(cells, 0, b, depth, b)
            _gen_add_segment(cells, depth, a, depth, b)
        else:
            r = h - 1
            inner = r - depth
            for c in range(a + 1, b):
                cells.discard((r, c))
            _gen_add_segment(cells, r, a, inner, a)
            _gen_add_segment(cells, r, b, inner, b)
            _gen_add_segment(cells, inner, a, inner, b)
    else:
        span = random.randint(NOTCH_SPAN[0], min(NOTCH_SPAN[1], h - 3))
        a = random.randint(1, h - span - 2)
        b = a + span
        depth = random.randint(1, min(depth_limit, max(1, (w - 3) // 2)))
        if side == "left":
            for r in range(a + 1, b):
                cells.discard((r, 0))
            _gen_add_segment(cells, a, 0, a, depth)
            _gen_add_segment(cells, b, 0, b, depth)
            _gen_add_segment(cells, a, depth, b, depth)
        else:
            c = w - 1
            inner = c - depth
            for r in range(a + 1, b):
                cells.discard((r, c))
            _gen_add_segment(cells, a, c, a, inner)
            _gen_add_segment(cells, b, c, b, inner)
            _gen_add_segment(cells, a, inner, b, inner)


def _gen_rect_perimeter_cells(h, w):
    cells = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _gen_closed_patch(color, closed_count):
    for _ in range(CLOSED_PATCH_ATTEMPTS):
        h_bounds, w_bounds = CLOSED_SIZE_BY_COUNT[min(closed_count, 3)]
        h = random.randint(*h_bounds)
        w = random.randint(*w_bounds)

        base_area = 2 * h + 2 * w - 4
        if base_area > CLOSED_BASE_AREA_MAX:
            continue
        max_notches = CLOSED_NOTCH_COUNT[1] if base_area <= CLOSED_TWO_NOTCH_AREA_MAX else CLOSED_NOTCH_COUNT[0]
        notch_count = random.randint(CLOSED_NOTCH_COUNT[0], max_notches)
        depth_limit = max(1, (CLOSED_AREA[1] - base_area) // (2 * notch_count))

        cells = _gen_rect_perimeter_cells(h, w)
        for side in random.sample(NOTCH_SIDES, notch_count):
            _gen_apply_notch(cells, h, w, side, depth_limit)

        wall = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            wall[r, c] = True
        coords = np.argwhere(wall)
        if not (CLOSED_AREA[0] <= len(coords) <= CLOSED_AREA[1]):
            continue
        if len(components_mask(wall, connectivity=4)) != 1:
            continue
        regions = _enclosed_region_coords(coords, (0, 0, h - 1, w - 1))
        if len(regions) != 1 or len(regions[0]) < 3:
            continue

        patch = np.zeros((h, w), dtype=int)
        patch[wall] = int(color)
        return patch
    return None


def _gen_open_patch(color):
    for _ in range(OPEN_PATCH_ATTEMPTS):
        h = random.randint(*OPEN_SIZE[0])
        w = random.randint(*OPEN_SIZE[1])
        cells = set()
        jogs = random.choice(OPEN_JOG_CHOICES)
        if jogs == 1:
            r0, r1 = random.sample(range(h), 2)
            c0 = random.randint(2, w - 3)
            points = [(r0, 0), (r0, c0), (r1, c0), (r1, w - 1)]
        else:
            r0, r1, r2 = random.sample(range(h), 3)
            c0 = random.randint(2, w - 5)
            c1 = random.randint(c0 + 2, w - 3)
            points = [(r0, 0), (r0, c0), (r1, c0), (r1, c1), (r2, c1), (r2, w - 1)]
        for a, b in zip(points, points[1:]):
            _gen_add_segment(cells, a[0], a[1], b[0], b[1])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if random.random() < 0.5:
            mask = mask.T
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        coords = np.argwhere(mask)
        if not (OPEN_AREA[0] <= len(coords) <= OPEN_AREA[1]):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        h2, w2 = mask.shape
        if not (OPEN_BBOX_ROWS[0] <= h2 <= OPEN_BBOX_ROWS[1] and OPEN_BBOX_COLS[0] <= w2 <= OPEN_BBOX_COLS[1]):
            continue
        if _enclosed_region_coords(coords, (0, 0, h2 - 1, w2 - 1)):
            continue

        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        return patch
    return None


def _gen_bbox_distance(a, b):
    dr = 0
    if a[2] < b[0]:
        dr = b[0] - a[2]
    elif b[2] < a[0]:
        dr = a[0] - b[2]
    dc = 0
    if a[3] < b[1]:
        dc = b[1] - a[3]
    elif b[3] < a[1]:
        dc = a[1] - b[3]
    return dr + dc


def _gen_place_patch(canvas, patch, placed_bboxes):
    H, W = canvas.shape
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            bb = (r, c, r + h - 1, c + w - 1)
            if any(_gen_bbox_distance(bb, old) < OUTLINE_BBOX_GAP for old in placed_bboxes):
                continue
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_marker_plan(marker_palette):
    minority_total = random.randint(*MINORITY_MARKER_TOTAL)
    if minority_total == 1 or random.random() < SINGLE_MINORITY_PROB:
        minority_counts = [minority_total]
    else:
        minority_counts = [1, 1]
    majority_count = random.randint(
        max(MAJORITY_MARKERS[0], max(minority_counts) + 1),
        min(MAJORITY_MARKERS[1], MARKERS_PER_REGION_MAX - minority_total),
    )
    colors = random.sample(marker_palette, 1 + len(minority_counts))
    return [(colors[0], majority_count)] + list(zip(colors[1:], minority_counts))


def _gen_place_markers(inp, coords, plan, reserved):
    available = {tuple(map(int, p)) for p in coords}
    for r, c in list(reserved):
        available.discard((r, c))
        for dr, dc in FOUR_NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            available.discard((nr, nc))

    draws = []
    for color, count in plan:
        draws.extend([int(color)] * int(count))
    random.shuffle(draws)

    placed = []
    for color in draws:
        if not available:
            return None
        r, c = random.choice(tuple(available))
        inp[r, c] = int(color)
        placed.append((r, c))
        reserved.add((r, c))
        for dr, dc in FOUR_NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            available.discard((nr, nc))
            reserved.add((nr, nc))
        available.discard((r, c))
    return placed


def _gen_place_strays(inp, out, marker_palette, count, reserved):
    candidates = {tuple(map(int, p)) for p in np.argwhere((inp == 0) & (out == 0))}
    for r, c in list(reserved):
        candidates.discard((r, c))
    for _ in range(int(count)):
        if not candidates:
            return False
        r, c = random.choice(tuple(candidates))
        inp[r, c] = random.choice(marker_palette)
        reserved.add((r, c))
        for dr, dc in FOUR_NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            candidates.discard((nr, nc))
            reserved.add((nr, nc))
        candidates.discard((r, c))
    return True


def _gen_place_and_paste_patch(canvas, patch, placed_bboxes):
    """Place a patch under bbox-gap constraints, paste it, and record its bbox."""
    loc = _gen_place_patch(canvas, patch, placed_bboxes)
    if loc is None:
        return None
    r, c, bb = loc
    paste(canvas, patch, r, c, transparent=0)
    placed_bboxes.append(bb)
    return r, c, bb


def _gen_place_closed_outlines(out, outline_colors, closed_count, placed_bboxes):
    """Place closed outlines and return enclosed regions paired with outline colors."""
    closed_records = []
    for i in range(closed_count):
        patch = _gen_closed_patch(outline_colors[i], closed_count)
        if patch is None:
            return None
        placement = _gen_place_and_paste_patch(out, patch, placed_bboxes)
        if placement is None:
            return None
        r, c, bb = placement
        local_coords = np.argwhere(patch != 0)
        coords = local_coords + np.array([r, c])
        regions = _enclosed_region_coords(coords, bb)
        if len(regions) != 1 or len(regions[0]) < 3:
            return None
        closed_records.append((regions[0], outline_colors[i]))
    return closed_records


def _gen_place_open_outline(out, color, placed_bboxes):
    """Place one open outline; returns False if no valid placement exists."""
    patch = _gen_open_patch(color)
    if patch is None:
        return False
    return _gen_place_and_paste_patch(out, patch, placed_bboxes) is not None


def _gen_fill_regions_and_place_markers(inp, out, closed_records, marker_palette):
    """Fill output regions and place the corresponding input marker pixels."""
    inside_marker_total = 0
    reserved_marker_cells = set()
    for region, _outline_color in closed_records:
        plan = _gen_marker_plan(marker_palette)
        fill_color = plan[0][0]
        out[region[:, 0], region[:, 1]] = int(fill_color)
        placed = _gen_place_markers(inp, region, plan, reserved_marker_cells)
        if placed is None:
            return None
        inside_marker_total += len(placed)
    return inside_marker_total, reserved_marker_cells


def _gen_place_required_strays(inp, out, marker_palette, inside_marker_total, reserved_marker_cells):
    """Place enough stray markers to keep the total marker count in range."""
    stray_min = max(0, TOTAL_MARKERS[0] - inside_marker_total)
    stray_max = min(STRAY_MARKERS_MAX, TOTAL_MARKERS[1] - inside_marker_total)
    if stray_min > stray_max:
        return False
    stray_count = random.randint(stray_min, stray_max)
    return _gen_place_strays(inp, out, marker_palette, stray_count, reserved_marker_cells)


def _gen_solver_accepts(inp, out):
    """Return whether the solver maps this generated input to the planned output."""
    if get_bg(inp) != 0:
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt():
    H = random.randint(*GEN_GRID_ROWS)
    W = random.randint(*GEN_GRID_COLS)
    closed_count = random.randint(*CLOSED_OUTLINE_COUNT)
    open_count = random.randint(*OPEN_OUTLINE_COUNT)
    outline_colors = random.sample(GEN_COLORS, closed_count + open_count)
    marker_palette = [c for c in GEN_COLORS if c not in outline_colors]

    out = np.zeros((H, W), dtype=int)
    placed_bboxes = []

    closed_records = _gen_place_closed_outlines(out, outline_colors, closed_count, placed_bboxes)
    if closed_records is None:
        return None

    if open_count and not _gen_place_open_outline(out, outline_colors[-1], placed_bboxes):
        return None

    inp = out.copy()
    marker_result = _gen_fill_regions_and_place_markers(inp, out, closed_records, marker_palette)
    if marker_result is None:
        return None
    inside_marker_total, reserved_marker_cells = marker_result

    if not _gen_place_required_strays(inp, out, marker_palette, inside_marker_total, reserved_marker_cells):
        return None

    if not _gen_solver_accepts(inp, out):
        return None
    return inp


def generate_input():
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 4b6b68e5 input")
