from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, neighbors4
import random

Coord = tuple[int, int]
Candidate = tuple[int, Coord, tuple[Coord, ...]]

DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _cells(component: dict) -> set[Coord]:
    return {(int(r), int(c)) for r, c in component["coords"]}


def _neighbors_in(cell: Coord, cells: set[Coord]) -> list[Coord]:
    return [nbr for nbr in neighbors4(*cell) if nbr in cells]


def _endpoints(cells: set[Coord]) -> tuple[Coord, Coord]:
    degrees = {cell: len(_neighbors_in(cell, cells)) for cell in cells}
    assert all(degree in (1, 2) for degree in degrees.values()), "path objects must be unbranched"

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "each path object must have exactly two endpoints"
    return endpoints[0], endpoints[1]


def _ordered_path(cells: set[Coord], start: Coord) -> list[Coord]:
    assert start in cells, "path start must be part of the object"
    path = [start]
    prev: Coord | None = None
    cur = start

    while True:
        next_cells = [nbr for nbr in _neighbors_in(cur, cells) if nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path objects must not branch"
        prev, cur = cur, next_cells[0]
        path.append(cur)

    assert len(path) == len(cells), "ordered path must cover the whole object"
    return path


def _visible_endpoint_candidates(
    g: np.ndarray,
    bg: int,
    marker: Coord,
    coord_to_object: dict[Coord, int],
    endpoints: list[tuple[Coord, Coord]],
) -> dict[int, Candidate]:
    H, W = g.shape
    mr, mc = marker
    candidates: dict[int, Candidate] = {}

    for dr, dc in DIRECTIONS:
        gap: list[Coord] = []
        r = mr + dr
        c = mc + dc
        while 0 <= r < H and 0 <= c < W:
            if int(g[r, c]) != bg:
                hit = (r, c)
                object_index = coord_to_object.get(hit)
                if object_index is not None and hit in endpoints[object_index]:
                    assert object_index not in candidates, "marker sees the same object endpoint more than once"
                    candidates[object_index] = (object_index, hit, tuple(gap))
                break

            gap.append((r, c))
            r += dr
            c += dc

    assert candidates, "each marker must see at least one path endpoint"
    return candidates


def _unique_matching(candidate_maps: list[dict[int, Candidate]], object_count: int) -> list[Candidate]:
    solutions: list[tuple[Candidate, ...]] = []

    def backtrack(marker_index: int, used_objects: set[int], current: list[Candidate]) -> None:
        if marker_index == len(candidate_maps):
            if len(used_objects) == object_count:
                solutions.append(tuple(current))
            return

        for object_index, candidate in candidate_maps[marker_index].items():
            if object_index in used_objects:
                continue
            used_objects.add(object_index)
            current.append(candidate)
            backtrack(marker_index + 1, used_objects, current)
            current.pop()
            used_objects.remove(object_index)

    backtrack(0, set(), [])
    assert len(solutions) == 1, "marker-to-path assignment must be unique"
    return list(solutions[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each path object to its matched marker and recolor an equal-length tail from the path's opposite endpoint.

    Input structure
    - The background is the dominant color.
    - Non-background pixels form the same number of isolated marker pixels and 4-connected monochrome path objects.
    - Each path object is a one-pixel-wide, unbranched orthogonal path with exactly two endpoints.
    - Every marker has a clear row or column sightline to at least one path endpoint, and the global marker-to-path assignment is unique.

    Rule
    - Match markers to path objects by the unique one-to-one assignment where each marker sees its object's endpoint along a straight background gap.
    - Fill that gap with the matched path object's color, leaving the marker pixel itself unchanged.
    - From the opposite endpoint of the same path, recolor as many existing path cells with the marker color as there were gap cells.

    Key insight
    - The marker does not simply color the nearby endpoint; its distance from the path measures how much of the far end of that path is converted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    path_objects = [component for component in components if int(component["area"]) > 1]
    markers = [component for component in components if int(component["area"]) == 1]
    assert path_objects, "expected at least one path object"
    assert len(path_objects) == len(markers), "expected one marker per path object"

    object_cells = [_cells(component) for component in path_objects]
    object_endpoints = [_endpoints(cells) for cells in object_cells]

    coord_to_object: dict[Coord, int] = {}
    for object_index, cells in enumerate(object_cells):
        for cell in cells:
            assert cell not in coord_to_object, "path objects must not overlap"
            coord_to_object[cell] = object_index

    marker_coords = [next(iter(_cells(component))) for component in markers]
    candidate_maps = [
        _visible_endpoint_candidates(g, bg, marker, coord_to_object, object_endpoints)
        for marker in marker_coords
    ]
    assignment = _unique_matching(candidate_maps, len(path_objects))

    planned: dict[Coord, int] = {}

    def plan(cell: Coord, color: int) -> None:
        previous = planned.get(cell)
        assert previous is None or previous == color, "conflicting path rewrites"
        planned[cell] = int(color)

    for marker_component, (object_index, hit_endpoint, gap) in zip(markers, assignment):
        path_color = int(path_objects[object_index]["color"])
        marker_color = int(marker_component["color"])

        for cell in gap:
            assert int(g[cell]) == bg, "connector gap must contain only background cells"
            plan(cell, path_color)

        endpoint_a, endpoint_b = object_endpoints[object_index]
        assert hit_endpoint in (endpoint_a, endpoint_b), "matched sightline must hit a path endpoint"
        opposite_endpoint = endpoint_b if hit_endpoint == endpoint_a else endpoint_a
        path = _ordered_path(object_cells[object_index], opposite_endpoint)
        assert path[-1] == hit_endpoint, "ordered path must terminate at the visible endpoint"
        assert len(gap) <= len(path), "connector length cannot exceed the path length"

        for cell in path[: len(gap)]:
            plan(cell, marker_color)

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out

# --------------- generator (o5x) ---------------

# Grid and palette.
_GEN_BG = 0
_GEN_GRID_SIZE_RANGE = (20, 30)
_GEN_COLOR_IDS = range(1, 10)
_GEN_COLOR_COUNT = 4
# Pair and path structure.
_GEN_PAIR_COUNT = 2
_GEN_PATH_LENGTH_RANGE = (7, 16)
_GEN_TURN_RANGE = (1, 3)
_GEN_GAP_LENGTH_RANGE = (3, 6)
# Placement constraints.
_GEN_DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_PATH_EDGE_PADDING = 1
_GEN_MIN_OBJECT_DISTANCE = 4
# Generation retry policy.
_GEN_MAX_ATTEMPTS = 300


def _gen_segment_lengths(total_steps: int, segment_count: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total_steps), segment_count - 1))
    points = [0, *cuts, total_steps]
    return [points[i + 1] - points[i] for i in range(segment_count)]


def _gen_relative_path(length: int, turns: int) -> list[tuple[int, int]]:
    segment_lengths = _gen_segment_lengths(length - 1, turns + 1)
    start_axis = random.randrange(2)
    axis_signs = [random.choice((-1, 1)), random.choice((-1, 1))]
    reverse_candidates = []
    if turns >= 2 and segment_lengths[1] > 1:
        reverse_candidates.append(2)
    if turns >= 3 and segment_lengths[2] > 1:
        reverse_candidates.append(3)
    reverse_segment = random.choice([None, *reverse_candidates])
    coords = [(0, 0)]
    r = c = 0
    for i, steps in enumerate(segment_lengths):
        axis = (start_axis + i) % 2
        sign = -axis_signs[axis] if i == reverse_segment else axis_signs[axis]
        dr, dc = (sign, 0) if axis == 0 else (0, sign)
        for _ in range(steps):
            r += dr
            c += dc
            coords.append((r, c))
    return coords


def _gen_is_chain(coords: list[tuple[int, int]]) -> bool:
    cells = set(coords)
    if len(cells) != len(coords):
        return False
    degrees = [sum(nbr in cells for nbr in neighbors4(*cell)) for cell in cells]
    return degrees.count(1) == 2 and all(degree in (1, 2) for degree in degrees)


def _gen_min_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _gen_clear_gap(
    path_cells: set[tuple[int, int]],
    visible: tuple[int, int],
    direction: tuple[int, int],
    gap_len: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    dr, dc = direction
    gap = [(visible[0] + dr * k, visible[1] + dc * k) for k in range(1, gap_len + 1)]
    marker = (visible[0] + dr * (gap_len + 1), visible[1] + dc * (gap_len + 1))
    if marker in path_cells or any(cell in path_cells for cell in gap):
        return None

    for i, cell in enumerate(gap):
        allowed = {visible} if i == 0 else set()
        if any(nbr in path_cells and nbr not in allowed for nbr in neighbors4(*cell)):
            return None
    if _gen_min_distance({marker}, path_cells) < _GEN_MIN_OBJECT_DISTANCE:
        return None
    return gap, marker


def _gen_gap_options(
    rel_path: list[Coord],
    visible: Coord,
    inward: Coord,
    gap_len: int,
) -> list[tuple[list[Coord], Coord]]:
    """Return legal endpoint-to-marker gaps in direction iteration order."""
    path_cells = set(rel_path)
    options = []
    for direction in _GEN_DIRECTIONS:
        if direction == inward:
            continue
        gap_and_marker = _gen_clear_gap(path_cells, visible, direction, gap_len)
        if gap_and_marker is not None:
            options.append(gap_and_marker)
    return options


def _gen_translation_bounds(
    n: int,
    rel_path: list[Coord],
    footprint: list[Coord],
) -> tuple[int, int, int, int] | None:
    """Return inclusive translation bounds that keep the footprint in-grid."""
    path_rows = [r for r, _ in rel_path]
    path_cols = [c for _, c in rel_path]
    all_rows = [r for r, _ in footprint]
    all_cols = [c for _, c in footprint]
    edge = _GEN_PATH_EDGE_PADDING

    min_tr = max(-min(all_rows), edge - min(path_rows))
    max_tr = min(n - 1 - max(all_rows), n - 1 - edge - max(path_rows))
    min_tc = max(-min(all_cols), edge - min(path_cols))
    max_tc = min(n - 1 - max(all_cols), n - 1 - edge - max(path_cols))
    if min_tr > max_tr or min_tc > max_tc:
        return None
    return min_tr, max_tr, min_tc, max_tc


def _gen_try_place_pair(
    n: int,
    existing_objects: list[set[tuple[int, int]]],
):
    length = random.randint(*_GEN_PATH_LENGTH_RANGE)
    turns = random.randint(*_GEN_TURN_RANGE)
    rel_path = _gen_relative_path(length, turns)
    if not _gen_is_chain(rel_path):
        return None

    visible_at_start = random.choice((True, False))
    visible = rel_path[0] if visible_at_start else rel_path[-1]
    inward_nbr = rel_path[1] if visible_at_start else rel_path[-2]
    inward = (inward_nbr[0] - visible[0], inward_nbr[1] - visible[1])
    gap_len = random.randint(_GEN_GAP_LENGTH_RANGE[0], min(_GEN_GAP_LENGTH_RANGE[1], length))

    gap_options = _gen_gap_options(rel_path, visible, inward, gap_len)
    if not gap_options:
        return None
    rel_gap, rel_marker = random.choice(gap_options)

    footprint = [*rel_path, *rel_gap, rel_marker]
    bounds = _gen_translation_bounds(n, rel_path, footprint)
    if bounds is None:
        return None
    min_tr, max_tr, min_tc, max_tc = bounds

    tr = random.randint(min_tr, max_tr)
    tc = random.randint(min_tc, max_tc)
    path = [(r + tr, c + tc) for r, c in rel_path]
    gap = [(r + tr, c + tc) for r, c in rel_gap]
    marker = (rel_marker[0] + tr, rel_marker[1] + tc)
    new_objects = [set(path), {marker}]
    if any(
        _gen_min_distance(new_obj, old_obj) < _GEN_MIN_OBJECT_DISTANCE
        for new_obj in new_objects
        for old_obj in existing_objects
    ):
        return None
    if any(cell in obj for cell in gap for obj in existing_objects):
        return None
    return path, gap, marker, gap_len


def _gen_tail(path: list[Coord], gap: list[Coord], gap_len: int) -> list[Coord]:
    head_visible = abs(path[0][0] - gap[0][0]) + abs(path[0][1] - gap[0][1]) == 1
    tail_order = list(reversed(path)) if head_visible else list(path)
    return tail_order[:gap_len]


def _gen_paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    if cells:
        rows, cols = zip(*cells)
        grid[rows, cols] = color


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_GEN_GRID_SIZE_RANGE)
    colors = random.sample(_GEN_COLOR_IDS, _GEN_COLOR_COUNT)
    path_colors = colors[:_GEN_PAIR_COUNT]
    marker_colors = colors[_GEN_PAIR_COUNT:]

    out = np.full((n, n), _GEN_BG, dtype=int)
    pair_records = []
    existing_objects: list[set[tuple[int, int]]] = []
    used_footprint: set[tuple[int, int]] = set()

    for path_color, marker_color in zip(path_colors, marker_colors):
        placed = _gen_try_place_pair(n, existing_objects)
        if placed is None:
            return None
        path, gap, marker, gap_len = placed
        footprint = set(path) | set(gap) | {marker}
        if used_footprint & footprint:
            return None
        used_footprint |= footprint

        tail = _gen_tail(path, gap, gap_len)
        _gen_paint_cells(out, path, path_color)
        _gen_paint_cells(out, gap, path_color)
        _gen_paint_cells(out, tail, marker_color)
        out[marker] = marker_color

        pair_records.append((path, gap, tail, path_color))
        existing_objects.extend([set(path), {marker}])

    inp = out.copy()
    for _path, gap, tail, path_color in pair_records:
        _gen_paint_cells(inp, gap, _GEN_BG)
        _gen_paint_cells(inp, tail, path_color)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 696d4842 input")
