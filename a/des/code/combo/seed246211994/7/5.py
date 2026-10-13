from __future__ import annotations

from collections import deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _adjacent_sites(top: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = top
    return ((r - 1, c + 1), (r + 3, c + 1), (r + 1, c - 1), (r + 1, c + 3))


def _find_checkers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    checkers: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            colors = set(map(int, window.ravel()))
            if bg in colors or len(colors) != 2:
                continue
            even = int(window[0, 0])
            odd = int(window[0, 1])
            if even == odd:
                continue
            if all(
                int(window[rr, cc]) == (even if (rr + cc) % 2 == 0 else odd)
                for rr in range(3)
                for cc in range(3)
            ):
                checkers.append((r, c))
    return checkers


def _find_rings(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rings: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            if int(window[1, 1]) != bg:
                continue
            border = np.array(
                [
                    window[0, 0],
                    window[0, 1],
                    window[0, 2],
                    window[1, 0],
                    window[1, 2],
                    window[2, 0],
                    window[2, 1],
                    window[2, 2],
                ],
                dtype=int,
            )
            if int(border[0]) == bg:
                continue
            if np.all(border == border[0]):
                rings.append((r, c))
    return rings


def _mark_objects(
    shape: tuple[int, int], objects: list[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in objects:
        assert not np.any(mask[r : r + 3, c : c + 3]), "3x3 objects must not overlap"
        mask[r : r + 3, c : c + 3] = True
    return mask


def _trace_component(
    nodes: set[tuple[int, int]], start: tuple[int, int]
) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in nodes and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _path_endpoints(nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    for r, c in nodes:
        degree = sum((r + dr, c + dc) in nodes for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)))
        if degree <= 1:
            endpoints.append((r, c))
    return endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the guide path from the unique checkerboard source and return the 3x3
    object at the far endpoint, typically a ring with a background center.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    checkers = _find_checkers(g, bg)
    rings = _find_rings(g, bg)
    assert len(checkers) == 1, "expected exactly one checkerboard source object"
    assert rings, "expected at least one ring object"

    all_objects = checkers + rings
    object_mask = _mark_objects(g.shape, all_objects)
    src = checkers[0]

    connectors: list[tuple[int, int, int]] = []
    for rr, cc in _adjacent_sites(src):
        if not _in_bounds(g.shape, rr, cc):
            continue
        value = int(g[rr, cc])
        if value == bg or object_mask[rr, cc]:
            continue
        connectors.append((rr, cc, value))
    assert len(connectors) == 1, "checkerboard source must have one adjacent guide pixel"

    start_r, start_c, path_color = connectors[0]
    guide_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere((g == path_color) & ~object_mask)
    }
    assert (start_r, start_c) in guide_nodes, "guide path must include the source-adjacent pixel"

    component = _trace_component(guide_nodes, (start_r, start_c))
    endpoints = _path_endpoints(component)
    assert endpoints, "guide path must have at least one endpoint"

    start_adjacent = [node for node in endpoints if node in _adjacent_sites(src)]
    assert start_adjacent, "one path endpoint must touch the source checkerboard"
    start_node = start_adjacent[0]
    far_node = next((node for node in endpoints if node != start_node), start_node)

    ring_targets = [top for top in rings if far_node in _adjacent_sites(top)]
    if len(ring_targets) == 1:
        r, c = ring_targets[0]
        return g[r : r + 3, c : c + 3].copy()

    checker_targets = [top for top in checkers if far_node in _adjacent_sites(top)]
    if len(checker_targets) == 1:
        r, c = checker_targets[0]
        return g[r : r + 3, c : c + 3].copy()

    fallback = [top for top in rings if any(site in component for site in _adjacent_sites(top))]
    assert len(fallback) == 1, "guide path must identify exactly one ring target"
    r, c = fallback[0]
    return g[r : r + 3, c : c + 3].copy()


# --------------- generator ---------------

GEN_ATTEMPTS = 400

GRID_LATTICE_MIN = 8
GRID_LATTICE_MAX = 12

RING_COUNT_MIN = 3
RING_COUNT_MAX = 4

MAIN_STEPS_MIN = 7
MAIN_STEPS_MAX = 16
MAIN_TURNS_MIN = 1
MAIN_TURNS_MAX = 5

DISTRACTOR_LEN_MIN = 2
DISTRACTOR_LEN_MAX = 8

COLOR_COUNT = 10


def _gen_split_positive(total: int, count: int) -> list[int]:
    if count == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), count - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _gen_main_path(lh: int, lw: int) -> list[tuple[int, int]] | None:
    max_steps = min(MAIN_STEPS_MAX, lh + lw)
    if max_steps < MAIN_STEPS_MIN:
        return None

    total_steps = random.randint(MAIN_STEPS_MIN, max_steps)
    max_turns = min(MAIN_TURNS_MAX, total_steps - 1)
    if max_turns < MAIN_TURNS_MIN:
        return None

    turns = random.randint(MAIN_TURNS_MIN, max_turns)
    seg_count = turns + 1
    if seg_count > total_steps:
        return None

    first_axis = random.randrange(2)
    axes = [(first_axis + i) % 2 for i in range(seg_count)]
    row_count = axes.count(0)
    col_count = axes.count(1)

    row_min = max(row_count, total_steps - lw)
    row_max = min(lh, total_steps - col_count)
    if row_min > row_max:
        return None

    row_total = random.randint(row_min, row_max)
    col_total = total_steps - row_total
    row_parts = _gen_split_positive(row_total, row_count)
    col_parts = _gen_split_positive(col_total, col_count)

    row_dir = random.choice((-1, 1))
    col_dir = random.choice((-1, 1))

    if row_dir > 0:
        r0 = random.randint(0, lh - row_total)
    else:
        r0 = random.randint(row_total, lh)
    if col_dir > 0:
        c0 = random.randint(0, lw - col_total)
    else:
        c0 = random.randint(col_total, lw)

    path = [(r0, c0)]
    r, c = r0, c0
    row_i = 0
    col_i = 0
    for axis in axes:
        if axis == 0:
            steps = row_parts[row_i]
            row_i += 1
            dr, dc = row_dir, 0
        else:
            steps = col_parts[col_i]
            col_i += 1
            dr, dc = 0, col_dir
        for _ in range(steps):
            r += dr
            c += dc
            path.append((r, c))

    if len(path) != total_steps + 1 or len(set(path)) != len(path):
        return None
    return path


def _gen_dirs() -> tuple[tuple[int, int], ...]:
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    return [(r + dr, c + dc) for dr, dc in _gen_dirs()]


def _gen_center_rows(parity: int, lh: int) -> range:
    return range(1, lh + 1) if parity == 0 else range(0, lh)


def _gen_center_cols(parity: int, lw: int) -> range:
    return range(1, lw + 1) if parity == 0 else range(0, lw)


def _gen_is_valid_center(node: tuple[int, int], parity: int, lh: int, lw: int) -> bool:
    r, c = node
    return r in _gen_center_rows(parity, lh) and c in _gen_center_cols(parity, lw)


def _gen_in_lattice(node: tuple[int, int], lh: int, lw: int) -> bool:
    r, c = node
    return 0 <= r <= lh and 0 <= c <= lw


def _gen_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) >= 2 or abs(a[1] - b[1]) >= 2


def _gen_adjacent_path_count(center: tuple[int, int], path_set: set[tuple[int, int]]) -> int:
    return sum(neighbor in path_set for neighbor in _gen_neighbors(center))


def _gen_endpoint_centers(
    endpoint: tuple[int, int], path_set: set[tuple[int, int]], parity: int, lh: int, lw: int
) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for center in _gen_neighbors(endpoint):
        if not _gen_is_valid_center(center, parity, lh, lw):
            continue
        if center in path_set:
            continue
        if _gen_adjacent_path_count(center, path_set) == 1:
            centers.append(center)
    random.shuffle(centers)
    return centers


def _gen_touching_centers(
    node: tuple[int, int], centers: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    r, c = node
    return [center for center in centers if abs(center[0] - r) + abs(center[1] - c) == 1]


def _gen_node_clear_for_distractor(
    node: tuple[int, int],
    occupied: set[tuple[int, int]],
    centers: list[tuple[int, int]],
    current_center: tuple[int, int],
    is_start: bool,
    lh: int,
    lw: int,
) -> bool:
    if not _gen_in_lattice(node, lh, lw):
        return False
    if node in occupied or node in centers:
        return False

    touching = _gen_touching_centers(node, centers)
    if is_start:
        return touching == [current_center]
    return not touching


def _gen_make_distractor_paths(
    center: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    centers: set[tuple[int, int]],
    lh: int,
    lw: int,
) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    center_list = list(centers)
    for dr, dc in _gen_dirs():
        start = (center[0] + dr, center[1] + dc)
        if not _gen_node_clear_for_distractor(start, occupied, center_list, center, True, lh, lw):
            continue

        primary = (dr, dc)
        turn_dirs = ((0, -1), (0, 1)) if dr else ((-1, 0), (1, 0))
        dir_sequences = [[primary] * (length - 1)]
        for turn_dir in turn_dirs:
            for turn_after in range(length - 1):
                seq = [primary] * turn_after + [turn_dir] * (length - 1 - turn_after)
                dir_sequences.append(seq)

        for dirs in dir_sequences:
            path = [start]
            used = {start}
            r, c = start
            ok = True
            for step_dr, step_dc in dirs:
                r += step_dr
                c += step_dc
                node = (r, c)
                if node in used:
                    ok = False
                    break
                if not _gen_node_clear_for_distractor(
                    node, occupied | used, center_list, center, False, lh, lw
                ):
                    ok = False
                    break
                path.append(node)
                used.add(node)
            if ok and len(path) == length:
                options.append(path)
    return options


def _gen_pick_distractor(
    center: tuple[int, int],
    occupied: set[tuple[int, int]],
    centers: set[tuple[int, int]],
    lh: int,
    lw: int,
) -> list[tuple[int, int]] | None:
    feasible: dict[int, list[list[tuple[int, int]]]] = {}
    for length in range(DISTRACTOR_LEN_MIN, DISTRACTOR_LEN_MAX + 1):
        paths = _gen_make_distractor_paths(center, length, occupied, centers, lh, lw)
        if paths:
            feasible[length] = paths
    if not feasible:
        return None
    length = random.choice(list(feasible))
    return random.choice(feasible[length])


def _gen_pick_endpoint_objects(
    main_path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    parity: int,
    lh: int,
    lw: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    checker_options = _gen_endpoint_centers(main_path[0], path_set, parity, lh, lw)
    target_options = _gen_endpoint_centers(main_path[-1], path_set, parity, lh, lw)
    options = [
        (checker_center, target_center)
        for checker_center in checker_options
        for target_center in target_options
        if _gen_separated(checker_center, target_center)
    ]
    if not options:
        return None
    return random.choice(options)


def _gen_candidate_extra_centers(
    parity: int,
    path_set: set[tuple[int, int]],
    occupied_centers: list[tuple[int, int]],
    lh: int,
    lw: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in _gen_center_rows(parity, lh):
        for c in _gen_center_cols(parity, lw):
            center = (r, c)
            if center in path_set:
                continue
            if _gen_adjacent_path_count(center, path_set):
                continue
            if all(_gen_separated(center, other) for other in occupied_centers):
                candidates.append(center)
    random.shuffle(candidates)
    return candidates


def _gen_pick_extra_rings(
    ring_count: int,
    parity: int,
    path_set: set[tuple[int, int]],
    occupied_centers: list[tuple[int, int]],
    lh: int,
    lw: int,
) -> list[tuple[int, int]] | None:
    extra_rings: list[tuple[int, int]] = []
    for center in _gen_candidate_extra_centers(parity, path_set, occupied_centers, lh, lw):
        if len(extra_rings) == ring_count - 1:
            break
        if all(_gen_separated(center, other) for other in occupied_centers + extra_rings):
            extra_rings.append(center)
    if len(extra_rings) != ring_count - 1:
        return None
    return extra_rings


def _gen_pick_distractors(
    main_path: list[tuple[int, int]],
    checker_center: tuple[int, int],
    ring_centers: list[tuple[int, int]],
    lh: int,
    lw: int,
) -> list[list[tuple[int, int]]] | None:
    all_centers = set([checker_center] + ring_centers)
    occupied_guides = set(main_path)
    distractor_paths: list[list[tuple[int, int]]] = []
    for center in ring_centers[1:]:
        path = _gen_pick_distractor(center, occupied_guides, all_centers, lh, lw)
        if path is None:
            return None
        distractor_paths.append(path)
        occupied_guides.update(path)
    return distractor_paths


def _gen_color_plan(ring_count: int, bg: int) -> tuple[list[int], tuple[int, int], list[int]]:
    available = [color for color in range(COLOR_COUNT) if color != bg]
    distinct_count = random.randint(max(5, ring_count + 2), min(8, len(available)))
    palette = random.sample(available, distinct_count)
    random.shuffle(palette)

    ring_colors = palette[:ring_count]
    leftovers = palette[ring_count:]
    refill = [color for color in ring_colors if color not in leftovers]
    random.shuffle(refill)
    path_colors = leftovers + refill[: ring_count - len(leftovers)]
    random.shuffle(path_colors)

    checker_colors = tuple(random.sample(palette, 2))
    return ring_colors, checker_colors, path_colors


def _gen_actual(node: tuple[int, int], parity: int) -> tuple[int, int]:
    return parity + 2 * node[0], parity + 2 * node[1]


def _gen_paint_checker(
    grid: np.ndarray, center: tuple[int, int], parity: int, colors: tuple[int, int]
) -> None:
    rr, cc = _gen_actual(center, parity)
    a, b = colors
    window = grid[rr - 1 : rr + 2, cc - 1 : cc + 2]
    window[::2, ::2] = a
    window[1, 1] = a
    window[0::2, 1] = b
    window[1, 0::2] = b


def _gen_paint_ring(grid: np.ndarray, center: tuple[int, int], parity: int, color: int, bg: int) -> None:
    rr, cc = _gen_actual(center, parity)
    grid[rr - 1 : rr + 2, cc - 1 : cc + 2] = color
    grid[rr, cc] = bg


def _gen_paint_path(
    grid: np.ndarray, path: list[tuple[int, int]], parity: int, color: int
) -> None:
    for node in path:
        rr, cc = _gen_actual(node, parity)
        grid[rr, cc] = color


def _gen_render_grid(
    parity: int,
    bg: int,
    lh: int,
    lw: int,
    checker_center: tuple[int, int],
    ring_centers: list[tuple[int, int]],
    main_path: list[tuple[int, int]],
    distractor_paths: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    ring_colors, checker_colors, path_colors = _gen_color_plan(len(ring_centers), bg)

    grid = np.full((2 * lh + 2, 2 * lw + 2), bg, dtype=int)
    for center, color in zip(ring_centers, ring_colors):
        _gen_paint_ring(grid, center, parity, color, bg)
    _gen_paint_checker(grid, checker_center, parity, checker_colors)

    _gen_paint_path(grid, main_path, parity, path_colors[0])
    for path, color in zip(distractor_paths, path_colors[1:]):
        _gen_paint_path(grid, path, parity, color)

    target_center = ring_centers[0]
    rr, cc = _gen_actual(target_center, parity)
    expected = grid[rr - 1 : rr + 2, cc - 1 : cc + 2].copy()
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    lh = random.randint(GRID_LATTICE_MIN, GRID_LATTICE_MAX)
    lw = random.randint(GRID_LATTICE_MIN, GRID_LATTICE_MAX)
    parity = random.randrange(2)
    bg = random.randrange(COLOR_COUNT)
    ring_count = random.randint(RING_COUNT_MIN, RING_COUNT_MAX)

    main_path = _gen_main_path(lh, lw)
    if main_path is None:
        return None
    path_set = set(main_path)

    endpoints = _gen_pick_endpoint_objects(main_path, path_set, parity, lh, lw)
    if endpoints is None:
        return None
    checker_center, target_center = endpoints

    extra_rings = _gen_pick_extra_rings(
        ring_count, parity, path_set, [checker_center, target_center], lh, lw
    )
    if extra_rings is None:
        return None

    ring_centers = [target_center] + extra_rings
    distractor_paths = _gen_pick_distractors(main_path, checker_center, ring_centers, lh, lw)
    if distractor_paths is None:
        return None

    grid, expected = _gen_render_grid(
        parity,
        bg,
        lh,
        lw,
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
