import numpy as np


BG = 0
PATH_COLOR = 1
START_COLOR = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 600
MIN_SIZE = 14
MAX_SIZE = 30
WALK_MARGIN = 2

_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_boundary_cell(coord: tuple[int, int], h: int, w: int) -> bool:
    r, c = coord
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _rotate_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _rotate_ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _segment_color(step_count: int) -> int:
    if step_count <= 1:
        return GREEN
    if step_count == 2:
        return YELLOW
    if step_count == 3:
        return ORANGE
    if step_count == 4:
        return MAROON
    return GREY


def _path_endpoints(path_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    coords = np.argwhere(path_mask)
    assert len(coords) >= 2, "expected an open path with at least two blue pixels"

    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        degree = sum(1 for rr, cc in _neighbors4(r, c, h, w) if path_mask[rr, cc])
        if degree == 1:
            endpoints.append((r, c))
        else:
            assert degree == 2, "blue pixels must form a single non-branching path"
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    return endpoints


def _find_start_endpoint(
    red_pos: tuple[int, int],
    endpoints: list[tuple[int, int]],
    path_mask: np.ndarray,
) -> tuple[int, int]:
    rr, cc = red_pos
    adjacent = [
        endpoint
        for endpoint in endpoints
        if abs(endpoint[0] - rr) + abs(endpoint[1] - cc) == 1
    ]
    if len(adjacent) == 1:
        return adjacent[0]

    h, w = path_mask.shape
    if rr == 0:
        dr, dc = 1, 0
    elif rr == h - 1:
        dr, dc = -1, 0
    elif cc == 0:
        dr, dc = 0, 1
    else:
        dr, dc = 0, -1

    cr = rr + dr
    cc2 = cc + dc
    while 0 <= cr < h and 0 <= cc2 < w:
        if path_mask[cr, cc2]:
            candidate = (cr, cc2)
            assert candidate in endpoints, "red marker must indicate one endpoint"
            return candidate
        cr += dr
        cc2 += dc

    raise AssertionError("failed to locate the red-marked start endpoint")


def _trace_path(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    ordered = [start]
    visited = {start}
    prev = None
    current = start

    while True:
        next_cells = []
        for rr, cc in _neighbors4(current[0], current[1], h, w):
            if not path_mask[rr, cc]:
                continue
            if prev is not None and (rr, cc) == prev:
                continue
            next_cells.append((rr, cc))
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unambiguous"
        nxt = next_cells[0]
        assert nxt not in visited, "path must not self-intersect"
        ordered.append(nxt)
        visited.add(nxt)
        prev, current = current, nxt

    assert len(ordered) == int(path_mask.sum()), "blue path must be a single connected component"
    return ordered


def _markers_from_path(path: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    assert len(path) >= 2
    markers: dict[tuple[int, int], int] = {}

    current_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    segment_steps = 1

    for i in range(1, len(path) - 1):
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if next_dir == current_dir:
            segment_steps += 1
            continue

        assert next_dir != (-current_dir[0], -current_dir[1]), "path cannot reverse direction"
        corner = path[i]
        if next_dir == _rotate_cw(current_dir):
            marker = (
                corner[0] + current_dir[0] + next_dir[0],
                corner[1] + current_dir[1] + next_dir[1],
            )
            markers[marker] = _segment_color(segment_steps)
        else:
            assert next_dir == _rotate_ccw(current_dir), "turns must be 90-degree left or right"

        current_dir = next_dir
        segment_steps = 1

    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the blue path from the red-marked endpoint and place colored pixels only at right turns.

    Input structure:
    - The background is black.
    - A single blue one-pixel-wide orthogonal path is 4-connected, open, and non-branching.
    - Exactly one red boundary pixel marks which endpoint is the start of the walk.

    Rule:
    - Start from the endpoint indicated by the red pixel and trace the path to the other endpoint.
    - At every corner, determine whether the walk turns left or right; only right turns create a marker.
    - Place each right-turn marker on the outer diagonal of that corner.
    - The marker color depends on the number of straight steps taken before that turn:
      1 -> green, 2 -> yellow, 3 -> orange, 4 -> maroon, 5 or more -> grey.
    - Keep the original blue path and the red start marker unchanged.

    Key insight:
    The puzzle has two independent parts: the red edge marker fixes the traversal direction so left and right are meaningful, and the pre-turn straight-run length alone determines the marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    foreground = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(foreground) == 2, "expected exactly one path color and one start marker color"

    red_candidates = []
    for color in foreground:
        pts = np.argwhere(g == color)
        if len(pts) != 1:
            continue
        coord = (int(pts[0, 0]), int(pts[0, 1]))
        if _is_boundary_cell(coord, h, w):
            red_candidates.append((color, coord))
    assert len(red_candidates) == 1, "expected one boundary singleton start marker"

    start_color, red_pos = red_candidates[0]
    path_colors = [color for color in foreground if color != start_color]
    assert len(path_colors) == 1
    path_color = path_colors[0]

    path_mask = g == path_color
    endpoints = _path_endpoints(path_mask)
    start = _find_start_endpoint(red_pos, endpoints, path_mask)
    ordered_path = _trace_path(path_mask, start)
    markers = _markers_from_path(ordered_path)

    out = g.copy()
    for (r, c), color in markers.items():
        assert 0 <= r < h and 0 <= c < w, "marker must stay inside the grid"
        assert out[r, c] == bg, "marker position must be empty in the input"
        out[r, c] = color
    return out


def _in_walk_area(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return WALK_MARGIN <= r < h - WALK_MARGIN and WALK_MARGIN <= c < w - WALK_MARGIN


def _length_options(category: int, rng: np.random.Generator) -> list[int]:
    if category == 1:
        options = [1]
    elif category == 2:
        options = [2]
    elif category == 3:
        options = [3]
    elif category == 4:
        options = [4]
    elif category == 5:
        options = [5, 6, 7]
    else:
        options = [1, 2, 3, 4, 5, 6, 7]
    rng.shuffle(options)
    return options


def _extend_segment(
    current: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    reserved_markers: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    prev = current
    temp_occupied = set(occupied)
    cells: list[tuple[int, int]] = []

    for _ in range(length):
        nxt = (prev[0] + direction[0], prev[1] + direction[1])
        if not _in_walk_area(nxt, h, w):
            return None
        if nxt in temp_occupied or nxt in reserved_markers:
            return None
        for rr, cc in _neighbors4(nxt[0], nxt[1], h, w):
            if (rr, cc) in temp_occupied and (rr, cc) != prev:
                return None
        temp_occupied.add(nxt)
        cells.append(nxt)
        prev = nxt
    return cells


def _can_start_next_segment(
    current: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_markers: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    nxt = (current[0] + direction[0], current[1] + direction[1])
    if not _in_walk_area(nxt, h, w):
        return False
    if nxt in occupied or nxt in reserved_markers:
        return False
    for rr, cc in _neighbors4(nxt[0], nxt[1], h, w):
        if (rr, cc) in occupied and (rr, cc) != current:
            return False
    return True


def _search_path(
    current: tuple[int, int],
    direction: tuple[int, int],
    seg_idx: int,
    length_categories: list[int],
    turn_sequence: list[int],
    occupied: set[tuple[int, int]],
    reserved_markers: set[tuple[int, int]],
    path: list[tuple[int, int]],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    for length in _length_options(length_categories[seg_idx], rng):
        segment = _extend_segment(current, direction, length, occupied, reserved_markers, h, w)
        if segment is None:
            continue

        new_path = path + segment
        new_current = new_path[-1]
        new_occupied = set(occupied)
        new_occupied.update(segment)
        new_reserved = set(reserved_markers)

        if seg_idx == len(length_categories) - 1:
            return new_path

        turn = turn_sequence[seg_idx]
        next_direction = _rotate_cw(direction) if turn == 1 else _rotate_ccw(direction)
        if turn == 1:
            marker = (
                new_current[0] + direction[0] + next_direction[0],
                new_current[1] + direction[1] + next_direction[1],
            )
            if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                continue
            if marker in new_occupied or marker in new_reserved:
                continue
            new_reserved.add(marker)

        if not _can_start_next_segment(new_current, next_direction, new_occupied, new_reserved, h, w):
            continue

        result = _search_path(
            new_current,
            next_direction,
            seg_idx + 1,
            length_categories,
            turn_sequence,
            new_occupied,
            new_reserved,
            new_path,
            h,
            w,
            rng,
        )
        if result is not None:
            return result

    return None


def _sample_turn_sequence(turn_count: int, rng: np.random.Generator) -> list[int]:
    right_count = int(rng.integers(1, turn_count))
    turns = [1] * right_count + [-1] * (turn_count - right_count)
    rng.shuffle(turns)
    return turns


def _sample_start(h: int, w: int, rng: np.random.Generator):
    side = int(rng.integers(0, 4))
    if side == 0:
        c = int(rng.integers(WALK_MARGIN, w - WALK_MARGIN))
        return (0, c), (1, c), (1, 0)
    if side == 1:
        c = int(rng.integers(WALK_MARGIN, w - WALK_MARGIN))
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == 2:
        r = int(rng.integers(WALK_MARGIN, h - WALK_MARGIN))
        return (r, 0), (r, 1), (0, 1)
    r = int(rng.integers(WALK_MARGIN, h - WALK_MARGIN))
    return (r, w - 1), (r, w - 2), (0, -1)


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    w = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    turn_count = int(rng.integers(2, 5))

    red_pos, start, direction = _sample_start(h, w, rng)
    turn_sequence = _sample_turn_sequence(turn_count, rng)
    length_categories = [int(rng.integers(1, 6)) for _ in range(turn_count)]
    length_categories.append(0)

    path = _search_path(
        current=start,
        direction=direction,
        seg_idx=0,
        length_categories=length_categories,
        turn_sequence=turn_sequence,
        occupied={start},
        reserved_markers=set(),
        path=[start],
        h=h,
        w=w,
        rng=rng,
    )
    if path is None:
        return None

    markers = _markers_from_path(path)
    if len(markers) != turn_sequence.count(1):
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[red_pos] = START_COLOR
    for r, c in path:
        if grid[r, c] != BG:
            return None
        grid[r, c] = PATH_COLOR

    for r, c in markers:
        if grid[r, c] != BG:
            return None

    expected = grid.copy()
    for (r, c), color in markers.items():
        expected[r, c] = color
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
