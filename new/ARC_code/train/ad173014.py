import math

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous
import random


RED = 2
BLUE = 1


def _extract_frames(g: np.ndarray, bg: int) -> tuple[list[dict], list[np.ndarray]]:
    red_components = components_mask(g == RED, connectivity=4)
    assert len(red_components) >= 3

    frames = []
    for coords in red_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 3

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(border.sum()) == int(coords.shape[0])
        assert np.all(patch[border] == RED)
        assert np.all(patch[~border] != RED)

        interior = patch[1:-1, 1:-1]
        interior_colors = set(map(int, np.unique(interior))) - {int(bg)}
        assert len(interior_colors) == 1
        color = interior_colors.pop()
        mask = interior == color
        assert mask.any()

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2, (c0 + c1) / 2),
                "color": color,
                "mask": mask,
            }
        )

    return frames, red_components


def _blue_cycle_edges(g: np.ndarray, red_components: list[np.ndarray]) -> set[tuple[int, int]]:
    red_id = {}
    for i, coords in enumerate(red_components):
        for r, c in coords:
            red_id[(int(r), int(c))] = i

    edges = set()
    for coords in components_mask(g == BLUE, connectivity=4):
        touched = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                frame_id = red_id.get((nr, nc))
                if frame_id is not None:
                    touched.add(frame_id)
        assert len(touched) == 2
        a, b = touched
        edges.add(tuple(sorted((a, b))))

    n = len(red_components)
    assert len(edges) == n
    for i in range(n):
        assert sum(i in edge for edge in edges) == 2
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = [
        math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c)
        for frame in frames
    ]
    assert len({round(angle, 12) for angle in angles}) == len(angles)

    order = sorted(range(len(frames)), key=lambda i: angles[i])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each framed interior mask with the color from the next red frame clockwise around the blue connector cycle.

    Input structure
    - The grid contains hollow red square frames connected pairwise by blue line components.
    - The blue components connect the frames into one cycle, with every frame touching exactly two blue components.
    - Each frame interior contains background plus exactly one non-background color mask.

    Rule
    - Sort the red frames clockwise by their centers around the cycle.
    - Keep each frame's interior mask shape in place, but recolor it with the interior color from the next clockwise frame.
    - Leave the red frames, blue connectors, background, and interior background holes unchanged.

    Key insight
    - The colored masks do not move; the blue cycle gives the order in which their colors rotate.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames, red_components = _extract_frames(g, bg)
    edges = _blue_cycle_edges(g, red_components)
    order = _clockwise_order(frames, edges)

    out = g.copy()
    for k, frame_id in enumerate(order):
        next_id = order[(k + 1) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[frame_id]["mask"]] = frames[next_id]["color"]
    return out

# --------------- generator (att1) ---------------

# Palette constants.
BG = 0
CONNECTOR_BLUE = 1
FRAME_RED = 2
INTERIOR_COLORS = [3, 4, 5, 6, 7, 8, 9]

# Canvas and frame bounds.
FRAME_COUNT_RANGE = (3, 5)
FRAME_SIZE_CHOICES = [5, 7]
SMALL_FRAME_GRID_H_RANGE = (20, 27)
SMALL_FRAME_GRID_W_RANGE = (21, 27)
LARGE_FRAME_GRID_H_RANGE = (22, 27)
LARGE_FRAME_GRID_W_RANGE = (23, 27)
FRAME_GAP = 1

# Interior mask density.
MIN_INTERIOR_FILL = 0.30
MAX_INTERIOR_FILL = 0.56
MIN_INTERIOR_CELLS = 3

# Angular placement.
ANGLE_PERTURB_RANGE = (-0.3, 0.3)
MIN_RADIUS_PAD = 1
MAX_RADIUS_PAD = 1
ANGLE_EPS = 1e-6

# Connector constraints and retry budgets.
MAX_TWO_TURN_TRIES = 20
CONNECTOR_LENGTH_RANGE = (2, 24)
GENERATION_ATTEMPTS = 300

NEIGHBORS_8 = [
    (dr, dc)
    for dr in range(-1, 2)
    for dc in range(-1, 2)
    if not (dr == 0 and dc == 0)
]


def _gen_mask(interior_size, color):
    """Generate an 8-connected mask for frame interior with fill 30-56%."""
    n = interior_size
    total = n * n
    min_fill = max(MIN_INTERIOR_CELLS, int(math.ceil(MIN_INTERIOR_FILL * total)))
    max_fill = int(math.floor(MAX_INTERIOR_FILL * total))
    target = random.randint(min_fill, max_fill)

    mask = np.zeros((n, n), dtype=int)
    start = (n // 2, n // 2)
    mask[start] = color
    filled = [start]
    frontier = set()
    _add_mask_frontier(frontier, mask, start)

    while len(filled) < target and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        mask[cell] = color
        filled.append(cell)
        _add_mask_frontier(frontier, mask, cell)

    if len(filled) < min_fill:
        return None
    return mask


def _add_mask_frontier(frontier, mask, cell):
    n = mask.shape[0]
    for dr, dc in NEIGHBORS_8:
        nr, nc = cell[0] + dr, cell[1] + dc
        if 0 <= nr < n and 0 <= nc < n and mask[nr, nc] == 0:
            frontier.add((nr, nc))


def _axis_path(start, end):
    """Return an inclusive straight segment between row- or column-aligned cells."""
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec > sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    step = 1 if er > sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _one_turn_path(start, corner, end):
    return _axis_path(start, corner) + _axis_path(corner, end)[1:]


def _two_turn_vhv_path(start, mid_r, end):
    _, sc = start
    _, ec = end
    corner1 = (mid_r, sc)
    corner2 = (mid_r, ec)
    return (
        _axis_path(start, corner1)
        + _axis_path(corner1, corner2)[1:]
        + _axis_path(corner2, end)[1:]
    )


def _two_turn_hvh_path(start, mid_c, end):
    sr, _ = start
    er, _ = end
    corner1 = (sr, mid_c)
    corner2 = (er, mid_c)
    return (
        _axis_path(start, corner1)
        + _axis_path(corner1, corner2)[1:]
        + _axis_path(corner2, end)[1:]
    )


def _path_is_free(path, occupied):
    return all(p not in occupied for p in path)


def _find_path(start, end, occupied):
    """Find an L-shaped path (0-2 turns) from start to end avoiding occupied cells."""
    sr, sc = start
    er, ec = end

    if sr == er or sc == ec:
        path = _axis_path(start, end)
        if _path_is_free(path, occupied):
            return path

    corners = [(sr, ec), (er, sc)]
    random.shuffle(corners)
    for corner in corners:
        path = _one_turn_path(start, corner, end)
        if _path_is_free(path, occupied):
            return path

    for _ in range(MAX_TWO_TURN_TRIES):
        if sr != er and sc != ec:
            mid_r = random.randint(min(sr, er), max(sr, er))
            path = _two_turn_vhv_path(start, mid_r, end)
            if _path_is_free(path, occupied):
                return path

            mid_c = random.randint(min(sc, ec), max(sc, ec))
            path = _two_turn_hvh_path(start, mid_c, end)
            if _path_is_free(path, occupied):
                return path

    return None


def _pick_attachment(frame_bbox, side, occupied):
    """Pick an attachment point on the given side of a frame border.
    Returns (border_cell, connector_start) where connector_start is one step outside."""
    r0, c0, r1, c1 = frame_bbox
    if side == 'top':
        cols = list(range(c0 + 1, c1))
        random.shuffle(cols)
        for c in cols:
            border = (r0, c)
            outside = (r0 - 1, c)
            if outside not in occupied and 0 <= outside[0]:
                return border, outside
    elif side == 'bottom':
        cols = list(range(c0 + 1, c1))
        random.shuffle(cols)
        for c in cols:
            border = (r1, c)
            outside = (r1 + 1, c)
            if outside not in occupied:
                return border, outside
    elif side == 'left':
        rows = list(range(r0 + 1, r1))
        random.shuffle(rows)
        for r in rows:
            border = (r, c0)
            outside = (r, c0 - 1)
            if outside not in occupied and 0 <= outside[1]:
                return border, outside
    elif side == 'right':
        rows = list(range(r0 + 1, r1))
        random.shuffle(rows)
        for r in rows:
            border = (r, c1)
            outside = (r, c1 + 1)
            if outside not in occupied:
                return border, outside
    return None, None


def _best_side(frame_bbox, target_r, target_c):
    """Determine which side of a frame is closest to a target point."""
    r0, c0, r1, c1 = frame_bbox
    cr = (r0 + r1) / 2
    cc = (c0 + c1) / 2
    dr = target_r - cr
    dc = target_c - cc

    if abs(dr) > abs(dc):
        return 'bottom' if dr > 0 else 'top'
    else:
        return 'right' if dc > 0 else 'left'


def _choose_colors(n_frames):
    avail_colors = INTERIOR_COLORS[:]
    random.shuffle(avail_colors)
    return avail_colors[:n_frames]


def _choose_grid_size(frame_size):
    if frame_size == 7:
        h_range = LARGE_FRAME_GRID_H_RANGE
        w_range = LARGE_FRAME_GRID_W_RANGE
    else:
        h_range = SMALL_FRAME_GRID_H_RANGE
        w_range = SMALL_FRAME_GRID_W_RANGE
    return random.randint(*h_range), random.randint(*w_range)


def _frame_bboxes_overlap(a, b):
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    return not (
        r1a + FRAME_GAP < r0b
        or r1b + FRAME_GAP < r0a
        or c1a + FRAME_GAP < c0b
        or c1b + FRAME_GAP < c0a
    )


def _place_frame_bboxes(n_frames, frame_size, grid_h, grid_w):
    """Place non-overlapping frame boxes in clockwise angular order."""
    center_r = grid_h / 2
    center_c = grid_w / 2

    base_angles = [2 * math.pi * i / n_frames for i in range(n_frames)]
    angle_perturb = [random.uniform(*ANGLE_PERTURB_RANGE) for _ in range(n_frames)]
    angles = [base_angles[i] + angle_perturb[i] for i in range(n_frames)]
    angles.sort()

    min_radius = frame_size + MIN_RADIUS_PAD
    max_radius = min(grid_h, grid_w) / 2 - frame_size / 2 - MAX_RADIUS_PAD
    if max_radius < min_radius:
        return None

    frame_bboxes = []
    for angle in angles:
        radius = random.uniform(min_radius, max_radius)
        fr = center_r + radius * math.sin(angle) - frame_size / 2
        fc = center_c + radius * math.cos(angle) - frame_size / 2
        fr = int(round(fr))
        fc = int(round(fc))
        fr = max(0, min(grid_h - frame_size, fr))
        fc = max(0, min(grid_w - frame_size, fc))
        frame_bboxes.append((fr, fc, fr + frame_size - 1, fc + frame_size - 1))

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if _frame_bboxes_overlap(frame_bboxes[i], frame_bboxes[j]):
                return None

    for r0, c0, r1, c1 in frame_bboxes:
        if r0 < 0 or c0 < 0 or r1 >= grid_h or c1 >= grid_w:
            return None
    return frame_bboxes


def _gen_clockwise_order(frame_bboxes):
    """Return frame indexes sorted by angle around their shared centroid."""
    n_frames = len(frame_bboxes)
    centers = [((r0 + r1) / 2, (c0 + c1) / 2) for r0, c0, r1, c1 in frame_bboxes]
    mean_r = sum(c[0] for c in centers) / n_frames
    mean_c = sum(c[1] for c in centers) / n_frames
    cw_angles = [math.atan2(c[0] - mean_r, c[1] - mean_c) for c in centers]

    sorted_angles = sorted(enumerate(cw_angles), key=lambda x: x[1])
    angle_vals = [a for _, a in sorted_angles]
    for i in range(len(angle_vals) - 1):
        if abs(angle_vals[i + 1] - angle_vals[i]) < ANGLE_EPS:
            return None, None
    return [idx for idx, _ in sorted_angles], centers


def _draw_frame(grid, frame_bbox, occupied):
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0:c1 + 1] = FRAME_RED
    grid[r1, c0:c1 + 1] = FRAME_RED
    grid[r0:r1 + 1, c0] = FRAME_RED
    grid[r0:r1 + 1, c1] = FRAME_RED
    for c in range(c0, c1 + 1):
        occupied.add((r0, c))
        occupied.add((r1, c))
    for r in range(r0, r1 + 1):
        occupied.add((r, c0))
        occupied.add((r, c1))


def _place_mask(grid, frame_bbox, mask, occupied):
    r0, c0, _, _ = frame_bbox
    rows, cols = np.nonzero(mask)
    grid[r0 + 1 + rows, c0 + 1 + cols] = mask[rows, cols]
    occupied.update((r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in zip(rows, cols))


def _render_frames(grid, frame_bboxes, colors, interior_size):
    """Draw frame borders and colored interior masks; return the occupied cells."""
    masks = []
    occupied = set()
    for frame_bbox, color in zip(frame_bboxes, colors):
        _draw_frame(grid, frame_bbox, occupied)
        mask = _gen_mask(interior_size, color)
        if mask is None:
            return None
        masks.append(mask)
        _place_mask(grid, frame_bbox, mask, occupied)
    return occupied


def _path_in_bounds(path, grid_h, grid_w):
    return all(0 <= r < grid_h and 0 <= c < grid_w for r, c in path)


def _draw_path(grid, path, occupied):
    for r, c in path:
        grid[r, c] = CONNECTOR_BLUE
        occupied.add((r, c))


def _connect_cycle(grid, frame_bboxes, centers, cw_order, occupied):
    """Connect adjacent clockwise frames with bounded blue paths."""
    grid_h, grid_w = grid.shape
    n_frames = len(frame_bboxes)
    for k in range(n_frames):
        fi = cw_order[k]
        fj = cw_order[(k + 1) % n_frames]

        ci_r, ci_c = centers[fi]
        cj_r, cj_c = centers[fj]

        side_i = _best_side(frame_bboxes[fi], cj_r, cj_c)
        side_j = _best_side(frame_bboxes[fj], ci_r, ci_c)

        border_i, start_i = _pick_attachment(frame_bboxes[fi], side_i, occupied)
        if border_i is None:
            return False

        border_j, start_j = _pick_attachment(frame_bboxes[fj], side_j, occupied)
        if border_j is None:
            return False

        path = _find_path(start_i, start_j, occupied)
        if path is None:
            return False

        min_len, max_len = CONNECTOR_LENGTH_RANGE
        if len(path) < min_len or len(path) > max_len:
            return False

        if not _path_in_bounds(path, grid_h, grid_w):
            return False
        _draw_path(grid, path, occupied)
    return True


def _gen_attempt():
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    frame_size = random.choice(FRAME_SIZE_CHOICES)
    interior_size = frame_size - 2
    colors = _choose_colors(n_frames)
    grid_h, grid_w = _choose_grid_size(frame_size)

    frame_bboxes = _place_frame_bboxes(n_frames, frame_size, grid_h, grid_w)
    if frame_bboxes is None:
        return None

    cw_order, centers = _gen_clockwise_order(frame_bboxes)
    if cw_order is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = _render_frames(grid, frame_bboxes, colors, interior_size)
    if occupied is None:
        return None

    if not _connect_cycle(grid, frame_bboxes, centers, cw_order, occupied):
        return None

    try:
        main(grid)
    except (AssertionError, Exception):
        return None

    return grid


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
