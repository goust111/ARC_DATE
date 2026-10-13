from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import get_bg, in_bounds
import random


def _ring_windows(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rings: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h - 2):
        for c in range(w - 2):
            window = g[r : r + 3, c : c + 3]
            if np.any(window == bg):
                continue
            border = np.array(
                (
                    window[0, 0],
                    window[0, 1],
                    window[0, 2],
                    window[1, 0],
                    window[1, 2],
                    window[2, 0],
                    window[2, 1],
                    window[2, 2],
                )
            )
            if np.all(border == border[0]) and int(window[1, 1]) != int(border[0]):
                rings.append((r, c))
    return rings


def _checker_windows(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    checkers: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h - 2):
        for c in range(w - 2):
            window = g[r : r + 3, c : c + 3]
            if np.any(window == bg):
                continue
            colors = set(map(int, window.flat))
            if len(colors) != 2:
                continue

            even_color = int(window[0, 0])
            odd_color = int(window[0, 1])
            is_checker = all(
                int(window[rr, cc]) == (even_color if (rr + cc) % 2 == 0 else odd_color)
                for rr in range(3)
                for cc in range(3)
            )
            if is_checker:
                checkers.append((r, c))
    return checkers


def _mark_objects(
    shape: tuple[int, int], objects: list[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in objects:
        assert not np.any(mask[r : r + 3, c : c + 3]), "3x3 objects must not overlap"
        mask[r : r + 3, c : c + 3] = True
    return mask


def _adjacent_sites(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c + 1), (r + 3, c + 1), (r + 1, c - 1), (r + 1, c + 3))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the ring object reached by following the guide path from the checkerboard object.

    Input structure
    - The input has one dominant background color, one 3x3 checkerboard object,
      several 3x3 ring objects, and sparse guide pixels on a two-cell lattice.

    Rule
    - Find the checkerboard object and its unique adjacent guide pixel.
    - Follow guide pixels of that color through two-cell horizontal/vertical steps.
    - Output the unique 3x3 ring object adjacent to that connected guide path.

    Key insight
    - The checkerboard object is the source, and the color of its attached guide
      pixel selects the path whose far endpoint points at the answer ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    rings = _ring_windows(g, bg)
    assert rings, "expected at least one 3x3 ring object"
    checkers = _checker_windows(g, bg)
    assert len(checkers) == 1, "expected exactly one checkerboard source object"

    object_mask = _mark_objects(g.shape, rings + checkers)
    src_r, src_c = checkers[0]

    connectors: list[tuple[int, int, int]] = []
    for rr, cc in _adjacent_sites(src_r, src_c):
        if not in_bounds(g.shape, rr, cc):
            continue
        if object_mask[rr, cc] or int(g[rr, cc]) == bg:
            continue
        connectors.append((rr, cc, int(g[rr, cc])))
    assert len(connectors) == 1, "checkerboard source must have one adjacent guide pixel"

    start_r, start_c, path_color = connectors[0]
    guide_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere((g == path_color) & ~object_mask)
    }
    assert (start_r, start_c) in guide_nodes

    seen = {(start_r, start_c)}
    queue = deque([(start_r, start_c)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            neighbor = (r + dr, c + dc)
            if neighbor in guide_nodes and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    targets = [
        (r, c)
        for r, c in rings
        if any(site in seen for site in _adjacent_sites(r, c))
    ]
    assert len(targets) == 1, "guide path must point at exactly one ring object"

    dst_r, dst_c = targets[0]
    return g[dst_r : dst_r + 3, dst_c : dst_c + 3].copy()

# --------------- generator (att2) ---------------

# Canvas and lattice geometry.
GRID_SIZE = 22
LATTICE_MIN, LATTICE_MAX = 0, 10
OBJECT_GAP_LATTICE = 2

# Main guide path shape.
MAIN_PATH_LEN_MIN, MAIN_PATH_LEN_MAX = 14, 18
MAIN_TURNS_MIN, MAIN_TURNS_MAX = 1, 5

# Object and distractor counts.
RING_COUNT_MIN, RING_COUNT_MAX = 4, 5
DISTRACTOR_LEN_MIN, DISTRACTOR_LEN_MAX = 2, 9

# Palette constraints.
COLOR_COUNT = 10
NON_BG_COLORS_MIN, NON_BG_COLORS_MAX = 6, 8

# Retry budget.
GENERATION_ATTEMPTS = 300


def _gen_split_positive(total, count):
    if count == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), count - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _gen_main_path():
    node_count = random.randint(MAIN_PATH_LEN_MIN, MAIN_PATH_LEN_MAX)
    total_steps = node_count - 1
    turns = random.randint(MAIN_TURNS_MIN, MAIN_TURNS_MAX)
    seg_count = turns + 1
    if seg_count > total_steps:
        return None

    first_axis = random.randrange(2)
    axes = [(first_axis + i) % 2 for i in range(seg_count)]
    row_count = axes.count(0)
    col_count = axes.count(1)

    row_min = max(row_count, total_steps - LATTICE_MAX)
    row_max = min(LATTICE_MAX, total_steps - col_count)
    if row_min > row_max:
        return None

    row_total = random.randint(row_min, row_max)
    col_total = total_steps - row_total
    row_parts = _gen_split_positive(row_total, row_count)
    col_parts = _gen_split_positive(col_total, col_count)

    row_dir = random.choice((-1, 1))
    col_dir = random.choice((-1, 1))
    r0 = random.randint(LATTICE_MIN, LATTICE_MAX - row_total) if row_dir > 0 else random.randint(row_total, LATTICE_MAX)
    c0 = random.randint(LATTICE_MIN, LATTICE_MAX - col_total) if col_dir > 0 else random.randint(col_total, LATTICE_MAX)

    path = [(r0, c0)]
    r, c = r0, c0
    ri = 0
    ci = 0
    for axis in axes:
        if axis == 0:
            steps = row_parts[ri]
            ri += 1
            dr, dc = row_dir, 0
        else:
            steps = col_parts[ci]
            ci += 1
            dr, dc = 0, col_dir
        for _ in range(steps):
            r += dr
            c += dc
            path.append((r, c))

    if len(path) != node_count or len(set(path)) != node_count:
        return None
    return path


def _gen_dirs():
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _gen_neighbors(node):
    r, c = node
    return [(r + dr, c + dc) for dr, dc in _gen_dirs()]


def _gen_in_lattice(node):
    r, c = node
    return LATTICE_MIN <= r <= LATTICE_MAX and LATTICE_MIN <= c <= LATTICE_MAX


def _gen_center_bounds(parity):
    return range(1, LATTICE_MAX + 1) if parity == 0 else range(LATTICE_MIN, LATTICE_MAX)


def _gen_is_valid_center(node, parity):
    valid = _gen_center_bounds(parity)
    return node[0] in valid and node[1] in valid


def _gen_separated(a, b):
    return abs(a[0] - b[0]) >= OBJECT_GAP_LATTICE or abs(a[1] - b[1]) >= OBJECT_GAP_LATTICE


def _gen_adjacent_path_count(center, path_set):
    return sum(nb in path_set for nb in _gen_neighbors(center))


def _gen_endpoint_centers(endpoint, path_set, parity):
    centers = []
    for center in _gen_neighbors(endpoint):
        if not _gen_is_valid_center(center, parity):
            continue
        if center in path_set:
            continue
        if _gen_adjacent_path_count(center, path_set) == 1:
            centers.append(center)
    random.shuffle(centers)
    return centers


def _gen_touching_centers(node, centers):
    nr, nc = node
    return [
        center
        for center in centers
        if abs(center[0] - nr) + abs(center[1] - nc) == 1
    ]


def _gen_node_clear_for_distractor(node, occupied, centers, current_center, is_start):
    if not _gen_in_lattice(node):
        return False
    if node in occupied or node in centers:
        return False

    touching = _gen_touching_centers(node, centers)
    if is_start:
        return touching == [current_center]
    return not touching


def _gen_make_distractor_paths(center, length, occupied, centers):
    options = []
    centers_list = list(centers)
    for dr, dc in _gen_dirs():
        start = (center[0] + dr, center[1] + dc)
        if not _gen_node_clear_for_distractor(start, occupied, centers_list, center, True):
            continue

        primary = (dr, dc)
        if dr:
            turn_dirs = [(0, -1), (0, 1)]
        else:
            turn_dirs = [(-1, 0), (1, 0)]

        dir_sequences = [[primary] * (length - 1)]
        for turn_dir in turn_dirs:
            for turn_after in range(length - 1):
                dir_sequences.append([primary] * turn_after + [turn_dir] * (length - 1 - turn_after))

        for dirs in dir_sequences:
            path = [start]
            used = {start}
            r, c = start
            ok = True
            for step_dr, step_dc in dirs:
                r += step_dr
                c += step_dc
                node = (r, c)
                if node in used or not _gen_node_clear_for_distractor(
                    node, occupied | used, centers_list, center, False
                ):
                    ok = False
                    break
                path.append(node)
                used.add(node)
            if ok and len(path) == length:
                options.append(path)
    return options


def _gen_pick_distractor(center, occupied, centers):
    feasible = {}
    for length in range(DISTRACTOR_LEN_MIN, DISTRACTOR_LEN_MAX + 1):
        paths = _gen_make_distractor_paths(center, length, occupied, centers)
        if paths:
            feasible[length] = paths
    if not feasible:
        return None
    length = random.choice(list(feasible))
    return random.choice(feasible[length])


def _gen_color_plan(ring_count, bg):
    available = [color for color in range(COLOR_COUNT) if color != bg]
    distinct_count = random.randint(NON_BG_COLORS_MIN, NON_BG_COLORS_MAX)
    palette = random.sample(available, distinct_count)
    random.shuffle(palette)

    border_colors = palette[:ring_count]
    path_colors = palette[ring_count:]
    fill = [color for color in palette if color not in path_colors]
    random.shuffle(fill)
    path_colors = path_colors + fill[: ring_count - len(path_colors)]
    random.shuffle(path_colors)

    checker_colors = random.sample(palette, 2)
    center_colors = []
    for border in border_colors:
        center_choices = [color for color in palette if color != border]
        center_colors.append(random.choice(center_choices))
    return border_colors, center_colors, checker_colors, path_colors


def _gen_actual(node, parity):
    return parity + 2 * node[0], parity + 2 * node[1]


def _gen_paint_checker(grid, center, parity, colors):
    rr, cc = _gen_actual(center, parity)
    a, b = colors
    window = grid[rr - 1 : rr + 2, cc - 1 : cc + 2]
    window[::2, ::2] = a
    window[1, 1] = a
    window[0::2, 1] = b
    window[1, 0::2] = b


def _gen_paint_ring(grid, center, parity, border, middle):
    rr, cc = _gen_actual(center, parity)
    grid[rr - 1 : rr + 2, cc - 1 : cc + 2] = border
    grid[rr, cc] = middle


def _gen_paint_path(grid, path, parity, color):
    for node in path:
        rr, cc = _gen_actual(node, parity)
        grid[rr, cc] = color


def _gen_pick_endpoint_objects(main_path, path_set, parity):
    checker_options = _gen_endpoint_centers(main_path[0], path_set, parity)
    target_options = _gen_endpoint_centers(main_path[-1], path_set, parity)
    endpoint_pairs = [
        (checker_center, target_center)
        for checker_center in checker_options
        for target_center in target_options
        if _gen_separated(checker_center, target_center)
    ]
    if not endpoint_pairs:
        return None
    return random.choice(endpoint_pairs)


def _gen_candidate_extra_centers(parity, path_set, object_centers):
    candidate_centers = []
    for r in _gen_center_bounds(parity):
        for c in _gen_center_bounds(parity):
            center = (r, c)
            if center in path_set:
                continue
            if _gen_adjacent_path_count(center, path_set):
                continue
            if all(_gen_separated(center, other) for other in object_centers):
                candidate_centers.append(center)
    random.shuffle(candidate_centers)
    return candidate_centers


def _gen_pick_extra_rings(ring_count, parity, path_set, object_centers):
    extra_rings = []
    for center in _gen_candidate_extra_centers(parity, path_set, object_centers):
        if len(extra_rings) == ring_count - 1:
            break
        if all(_gen_separated(center, other) for other in object_centers + extra_rings):
            extra_rings.append(center)
    if len(extra_rings) != ring_count - 1:
        return None
    return extra_rings


def _gen_pick_distractors(main_path, checker_center, ring_centers):
    all_centers = set([checker_center] + ring_centers)
    occupied_guides = set(main_path)
    distractor_paths = []
    for center in ring_centers[1:]:
        path = _gen_pick_distractor(center, occupied_guides, all_centers)
        if path is None:
            return None
        distractor_paths.append(path)
        occupied_guides.update(path)
    return distractor_paths


def _gen_render_grid(parity, bg, checker_center, ring_centers, main_path, distractor_paths):
    border_colors, center_colors, checker_colors, path_colors = _gen_color_plan(len(ring_centers), bg)

    grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    for center, border, middle in zip(ring_centers, border_colors, center_colors):
        _gen_paint_ring(grid, center, parity, border, middle)
    _gen_paint_checker(grid, checker_center, parity, checker_colors)

    _gen_paint_path(grid, main_path, parity, path_colors[0])
    for path, color in zip(distractor_paths, path_colors[1:]):
        _gen_paint_path(grid, path, parity, color)

    target_center = ring_centers[0]
    target_r, target_c = _gen_actual(target_center, parity)
    expected = grid[target_r - 1 : target_r + 2, target_c - 1 : target_c + 2].copy()
    return grid, expected


def _gen_attempt():
    parity = random.randrange(2)
    bg = random.randrange(COLOR_COUNT)
    ring_count = random.randint(RING_COUNT_MIN, RING_COUNT_MAX)

    main_path = _gen_main_path()
    if main_path is None:
        return None
    path_set = set(main_path)

    endpoints = _gen_pick_endpoint_objects(main_path, path_set, parity)
    if endpoints is None:
        return None
    checker_center, target_center = endpoints

    extra_rings = _gen_pick_extra_rings(ring_count, parity, path_set, [checker_center, target_center])
    if extra_rings is None:
        return None

    ring_centers = [target_center] + extra_rings
    distractor_paths = _gen_pick_distractors(main_path, checker_center, ring_centers)
    if distractor_paths is None:
        return None

    grid, expected = _gen_render_grid(
        parity,
        bg,
        checker_center,
        ring_centers,
        main_path,
        distractor_paths,
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9ba4a9aa input")
