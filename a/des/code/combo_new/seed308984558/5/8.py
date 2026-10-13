from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_RETRIES = 400

Coord = tuple[int, int]
Segment = tuple[Coord, Coord]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _segment_cells(start: Coord, stop: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1, "segments must be axis-aligned"
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _ray_first_marker(start: Coord, direction: Coord, markers: set[Coord], shape: tuple[int, int]) -> tuple[Coord, int] | None:
    r, c = start
    dr, dc = direction
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not _in_bounds(shape, r, c):
            return None
        if (r, c) in markers:
            return (r, c), dist


def _start_candidates(start: Coord, markers: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    seen: list[tuple[int, Coord]] = []
    for direction in DIRS:
        hit = _ray_first_marker(start, direction, markers, shape)
        if hit is not None:
            seen.append((hit[1], hit[0]))
    if not seen:
        return []
    best = min(dist for dist, _ in seen)
    return [coord for dist, coord in seen if dist == best]


def _trace_from_start(
    start: Coord,
    first_stop: Coord,
    outgoing: dict[Coord, Coord],
    marker_count: int,
    closed: bool,
) -> list[Segment] | None:
    visited = {start, first_stop}
    segments: list[Segment] = [(start, first_stop)]
    current = first_stop

    for _ in range(marker_count + 1):
        if closed and current == start:
            return segments if len(visited) == marker_count else None

        if current not in outgoing:
            if not closed and current != start and len(visited) == marker_count:
                return segments
            return None

        nxt = outgoing[current]
        if closed and nxt == start:
            if len(visited) != marker_count:
                return None
            segments.append((current, start))
            current = start
            continue

        if nxt in visited:
            return None
        visited.add(nxt)
        segments.append((current, nxt))
        current = nxt

    return None


def _solve_instance(grid: np.ndarray) -> tuple[int, int, list[Segment]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, GREY)]
    assert len(colors) == 1, "expected exactly one marker color"
    marker_color = colors[0]

    markers = {tuple(map(int, coord)) for coord in np.argwhere(g == marker_color)}
    assert markers, "expected at least one marker"

    hinted_dirs: dict[Coord, Coord] = {}
    unhinted: list[Coord] = []
    for marker in sorted(markers):
        r, c = marker
        adj = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY:
                adj.append((dr, dc))
        assert len(adj) <= 1, "each marker must see at most one grey pointer"
        if adj:
            hinted_dirs[marker] = adj[0]
        else:
            unhinted.append(marker)

    assert len(unhinted) in (1, 2), "expected one closed-loop start or two open-path endpoints"
    closed = len(unhinted) == 1

    outgoing: dict[Coord, Coord] = {}
    for marker, direction in hinted_dirs.items():
        hit = _ray_first_marker(marker, direction, markers, g.shape)
        assert hit is not None, "every grey pointer must lead to another marker"
        outgoing[marker] = hit[0]

    successes: list[list[Segment]] = []
    for start in sorted(unhinted):
        for first_stop in _start_candidates(start, markers - {start}, g.shape):
            traced = _trace_from_start(start, first_stop, outgoing, len(markers), closed)
            if traced is not None:
                successes.append(traced)

    assert successes, "failed to trace a valid marker order"
    successes.sort()
    return bg, marker_color, successes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect scattered same-colored marker pixels into the directed orthogonal path implied by nearby grey pointers.

    Input structure:
    - The background is black.
    - One foreground color marks isolated single-pixel waypoints.
    - Some waypoints have exactly one grey 4-neighbor; its side relative to the marker gives the outgoing direction from that waypoint.

    Rule:
    - Markers without a grey neighbor are the open-path endpoints, or the unique loop start if the path closes back to it.
    - Start from the valid endpoint/start, move in the straight direction of its nearest aligned marker, then keep following the next direction dictated by each reached grey-marked waypoint.
    - Draw every horizontal or vertical segment between consecutive waypoints in the marker color, and erase the grey pointer blocks back to background.

    Key insight:
    - The marker order is not chosen by global distance; each grey neighbor locally forces the next travel direction, so the solution is obtained by tracing that directed waypoint chain.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker_color, segments = _solve_instance(g)

    out = np.full_like(g, bg)
    for start, stop in segments:
        for r, c in _segment_cells(start, stop):
            out[r, c] = marker_color
    return out


def _extend_simple_path(
    occupied: set[Coord],
    start: Coord,
    direction: Coord,
    length: int,
    shape: tuple[int, int],
) -> list[Coord] | None:
    dr, dc = direction
    r, c = start
    prev = start
    added: list[Coord] = []
    for step in range(1, length + 1):
        cell = (r + dr * step, c + dc * step)
        if not _in_bounds(shape, *cell):
            return None
        if cell in occupied:
            return None
        for nbr in _neighbors4(*cell):
            if nbr == prev:
                continue
            if nbr in occupied:
                return None
        added.append(cell)
        prev = cell
    return added


def _perpendicular_dirs(direction: Coord | None) -> list[Coord]:
    if direction is None:
        dirs = list(DIRS)
    else:
        dr, dc = direction
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _has_future_turn(occupied: set[Coord], endpoint: Coord, direction: Coord, shape: tuple[int, int], max_len: int) -> bool:
    for nxt_dir in _perpendicular_dirs(direction):
        for length in range(3, max_len + 1):
            if _extend_simple_path(occupied, endpoint, nxt_dir, length, shape) is not None:
                return True
    return False


def _build_open_vertices(shape: tuple[int, int], n_markers: int) -> list[Coord] | None:
    h, w = shape
    max_len = min(7, max(3, min(h, w) - 3))
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(starts)

    for start in starts:
        occupied = {start}
        vertices = [start]
        prev_dir: Coord | None = None
        ok = True
        for seg_idx in range(n_markers - 1):
            dirs = _perpendicular_dirs(prev_dir)
            min_len = 2 if seg_idx == 0 else 3
            lengths = list(range(min_len, max_len + 1))
            random.shuffle(lengths)

            placed = False
            for direction in dirs:
                for length in lengths:
                    added = _extend_simple_path(occupied, vertices[-1], direction, length, shape)
                    if added is None:
                        continue
                    new_occupied = occupied | set(added)
                    endpoint = added[-1]
                    if seg_idx < n_markers - 2 and not _has_future_turn(new_occupied, endpoint, direction, shape, max_len):
                        continue
                    occupied = new_occupied
                    vertices.append(endpoint)
                    prev_dir = direction
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if ok:
            return vertices
    return None


def _all_path_cells(vertices: list[Coord], closed: bool) -> set[Coord]:
    cells = {vertices[0]}
    stop = len(vertices) if closed else len(vertices) - 1
    for i in range(stop):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        for cell in _segment_cells(a, b)[1:]:
            cells.add(cell)
    return cells


def _transform_points(points: list[Coord]) -> list[Coord]:
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (c, -r),
        lambda r, c: (-r, -c),
        lambda r, c: (-c, r),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (c, r),
        lambda r, c: (-c, -r),
    ]
    transform = random.choice(transforms)
    pts = [transform(r, c) for r, c in points]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _place_shape(points: list[Coord]) -> tuple[tuple[int, int], list[Coord]]:
    pts = _transform_points(points)
    height = max(r for r, _ in pts) + 1
    width = max(c for _, c in pts) + 1
    h = min(30, height + random.randint(2, 8))
    w = min(30, width + random.randint(2, 8))
    if h < height + 2 or w < width + 2:
        raise AssertionError("shape must leave visible background margin")
    top = random.randint(1, h - height - 1)
    left = random.randint(1, w - width - 1)
    placed = [(r + top, c + left) for r, c in pts]
    return (h, w), placed


def _build_closed_vertices() -> tuple[tuple[int, int], list[Coord]]:
    if random.random() < 0.5:
        short = random.randint(3, 6)
        long = random.randint(short + 1, min(short + 6, 10))
        points = [(0, 0), (0, short), (long, short), (long, 0)]
        return _place_shape(points)

    first = random.randint(2, 5)
    final_up = random.randint(max(first + 1, 3), first + 4)
    low_row = random.randint(final_up + 3, final_up + 8)
    right_extra = random.randint(3, 5)
    points = [
        (0, 0),
        (0, first),
        (low_row, first),
        (low_row, first + right_extra),
        (final_up, first + right_extra),
        (final_up, 0),
    ]
    return _place_shape(points)


def _build_grid(vertices: list[Coord], closed: bool, marker_color: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in vertices:
        assert int(grid[r, c]) == BG
        grid[r, c] = marker_color

    if closed:
        indices = range(1, len(vertices))
    else:
        indices = range(1, len(vertices) - 1)

    for i in indices:
        src = vertices[i]
        dst = vertices[(i + 1) % len(vertices)]
        path = _segment_cells(src, dst)
        assert len(path) >= 4, "grey-sourced segments must leave a buffer before the next marker"
        gr, gc = path[1]
        assert int(grid[gr, gc]) == BG
        grid[gr, gc] = GREY

    return grid


def _structure_is_clear(grid: np.ndarray, marker_color: int) -> bool:
    markers = {tuple(map(int, coord)) for coord in np.argwhere(grid == marker_color)}
    greys = {tuple(map(int, coord)) for coord in np.argwhere(grid == GREY)}

    if np.count_nonzero(grid == BG) * 2 <= grid.size:
        return False

    for marker in markers:
        grey_neighbors = [nbr for nbr in _neighbors4(*marker) if nbr in greys]
        if len(grey_neighbors) > 1:
            return False
        for nbr in _neighbors4(*marker):
            if nbr in markers:
                return False

    unhinted = sum(1 for marker in markers if all(nbr not in greys for nbr in _neighbors4(*marker)))
    if unhinted not in (1, 2):
        return False

    for grey in greys:
        adjacent_markers = sum(1 for nbr in _neighbors4(*grey) if nbr in markers)
        if adjacent_markers != 1:
            return False

    return True


def _segments_from_vertices(vertices: list[Coord], closed: bool) -> list[Segment]:
    stop = len(vertices) if closed else len(vertices) - 1
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(stop)]


def _gen_attempt() -> np.ndarray | None:
    marker_color = random.choice(MARKER_COLORS)
    closed = random.random() < 0.45

    if closed:
        shape, vertices = _build_closed_vertices()
    else:
        shape = (random.randint(10, 24), random.randint(10, 24))
        n_markers = random.randint(3, 8)
        vertices = _build_open_vertices(shape, n_markers)
        if vertices is None:
            return None

    path_cells = _all_path_cells(vertices, closed)
    if len(path_cells) >= shape[0] * shape[1] // 2:
        return None

    grid = _build_grid(vertices, closed, marker_color, shape)
    if not _structure_is_clear(grid, marker_color):
        return None

    start_candidates = _start_candidates(vertices[0], set(vertices[1:]), shape)
    if start_candidates != [vertices[1]]:
        return None

    try:
        _, solved_color, solved_segments = _solve_instance(grid)
    except AssertionError:
        return None
    if solved_color != marker_color:
        return None
    if solved_segments != _segments_from_vertices(vertices, closed):
        return None

    out = main(grid)
    if np.any(out == GREY):
        return None
    if out.shape != grid.shape:
        return None
    if np.count_nonzero(out == marker_color) <= len(vertices):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
