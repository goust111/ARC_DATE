from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

GREY = 5
ORANGE = 7
CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator parameters.
MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
SEGMENT_STEPS = (2, 5)   # inclusive endpoints are steps + 1, so each segment has grey interior
CONNECTOR_STEPS = (2, 4)  # gives at least one orange cell in the output
GRID_ROWS = (14, 22)
GRID_COLS = (14, 22)
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 120
START_TRIALS = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in CARDINAL_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _segment_markers(comp: list[Coord], marker_set: set[Coord]) -> tuple[Coord, Coord]:
    adjacent_markers: set[Coord] = set()
    all_cells = set(comp)
    for r, c in comp:
        for dr, dc in CARDINAL_DIRS:
            pos = (r + dr, c + dc)
            if pos in marker_set:
                adjacent_markers.add(pos)

    assert len(adjacent_markers) == 2, "each grey segment must touch exactly two markers"
    line_cells = list(all_cells | adjacent_markers)
    rows = {r for r, _ in line_cells}
    cols = {c for _, c in line_cells}
    assert len(rows) == 1 or len(cols) == 1, "segment plus endpoints must form a straight line"
    if len(rows) == 1:
        row = next(iter(rows))
        ordered = sorted(c for _r, c in line_cells)
        assert ordered == list(range(ordered[0], ordered[-1] + 1)), "horizontal segment must be contiguous"
        markers = sorted(adjacent_markers, key=lambda p: p[1])
        assert markers[0][0] == row and markers[1][0] == row
        return markers[0], markers[1]

    col = next(iter(cols))
    ordered = sorted(r for r, _c in line_cells)
    assert ordered == list(range(ordered[0], ordered[-1] + 1)), "vertical segment must be contiguous"
    markers = sorted(adjacent_markers, key=lambda p: p[0])
    assert markers[0][1] == col and markers[1][1] == col
    return markers[0], markers[1]


def _shortest_connector(grid: np.ndarray, start: Coord, goal: Coord, bg: int) -> list[Coord]:
    if start[0] == goal[0]:
        row = start[0]
        c0, c1 = sorted((start[1], goal[1]))
        if all(int(grid[row, c]) == bg for c in range(c0 + 1, c1)):
            return [(row, c) for c in range(c0, c1 + 1)]
    if start[1] == goal[1]:
        col = start[1]
        r0, r1 = sorted((start[0], goal[0]))
        if all(int(grid[r, col]) == bg for r in range(r0 + 1, r1)):
            return [(r, col) for r in range(r0, r1 + 1)]

    h, w = grid.shape
    q: deque[Coord] = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in CARDINAL_DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if nxt != goal and int(grid[nr, nc]) != bg:
                continue
            prev[nxt] = (r, c)
            q.append(nxt)

    assert goal in prev, "matched markers must be linked by a background corridor"
    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join disconnected grey path segments into one chain using same-colored endpoint markers.

    Input structure:
    - The dominant color is the background.
    - Several disconnected straight grey path segments are present.
    - Each segment has two colored endpoint markers.
    - Every internal chain connection is encoded by a color that appears on exactly
      two markers belonging to two different segments.

    Rule:
    - Connect every same-colored marker pair with a shortest orthogonal connector.
    - Recolor the matched markers to grey so they merge into the path.
    - Keep the two unmatched outer endpoint markers in their original colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    marker_mask = (g != bg) & (g != GREY) & (g != ORANGE)
    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(marker_mask)]
    marker_set = set(marker_coords)
    assert len(marker_coords) >= 4, "expected multiple endpoint markers"

    grey_components = _components(g == GREY)
    assert len(grey_components) >= 2, "expected multiple grey segments"

    marker_to_segment: dict[Coord, int] = {}
    segments: list[tuple[Coord, Coord]] = []
    for idx, comp in enumerate(grey_components):
        endpoints = _segment_markers(comp, marker_set)
        segments.append(endpoints)
        for endpoint in endpoints:
            assert endpoint not in marker_to_segment, "marker may belong to only one segment"
            marker_to_segment[endpoint] = idx

    assert set(marker_to_segment) == marker_set, "every marker must terminate exactly one grey segment"

    color_to_markers: dict[int, list[Coord]] = {}
    for marker in marker_coords:
        color_to_markers.setdefault(int(g[marker]), []).append(marker)

    unmatched_colors = [color for color, coords in color_to_markers.items() if len(coords) == 1]
    matched_colors = [color for color, coords in color_to_markers.items() if len(coords) == 2]
    assert len(unmatched_colors) == 2, "the chain must have exactly two unmatched outer endpoints"
    assert len(matched_colors) == len(grey_components) - 1, "matched colors should connect the segments into a chain"

    adjacency = {idx: set() for idx in range(len(grey_components))}
    for color in matched_colors:
        a, b = color_to_markers[color]
        seg_a = marker_to_segment[a]
        seg_b = marker_to_segment[b]
        assert seg_a != seg_b, "matching markers must belong to different segments"
        adjacency[seg_a].add(seg_b)
        adjacency[seg_b].add(seg_a)

    degrees = sorted(len(neighbors) for neighbors in adjacency.values())
    assert degrees.count(1) == 2, "segment graph must be a chain"
    assert all(degree in (1, 2) for degree in degrees), "segment graph may not branch"
    seen = set()
    stack = [next(iter(adjacency))]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node] - seen)
    assert len(seen) == len(grey_components), "segment graph must be connected"

    out = g.copy()
    for color in matched_colors:
        start, goal = color_to_markers[color]
        path = _shortest_connector(g, start, goal, bg)
        out[start] = GREY
        out[goal] = GREY
        for cell in path[1:-1]:
            assert int(g[cell]) == bg, "connector interior must pass through background only"
            out[cell] = ORANGE
    return out


def _turn_dirs(direction: Coord) -> tuple[Coord, Coord]:
    dr, dc = direction
    return (-dc, dr), (dc, -dr)


def _direction_order(previous: Coord | None) -> list[Coord]:
    if previous is None:
        dirs = list(CARDINAL_DIRS)
        random.shuffle(dirs)
        return dirs
    left, right = _turn_dirs(previous)
    dirs = [previous, left, right]
    random.shuffle(dirs)
    return dirs


def _extend_run(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    shape: tuple[int, int],
) -> tuple[list[Coord], set[Coord]] | None:
    h, w = shape
    dr, dc = direction
    cur_path = list(path)
    cur_used = set(used)
    r, c = cur_path[-1]
    prev = cur_path[-2] if len(cur_path) >= 2 else None

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        nxt = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if nxt in cur_used:
            return None
        for adr, adc in CARDINAL_DIRS:
            adj = (nr + adr, nc + adc)
            if adj in cur_used and adj != (r, c):
                return None
        cur_path.append(nxt)
        cur_used.add(nxt)
        prev = (r, c)
        r, c = nxt

    return cur_path, cur_used


def _build_path(run_steps: list[int], shape: tuple[int, int]) -> list[Coord] | None:
    h, w = shape
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)

    def search(
        run_idx: int,
        previous_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if run_idx == len(run_steps):
            return path
        for direction in _direction_order(previous_dir):
            extended = _extend_run(path, used, direction, run_steps[run_idx], shape)
            if extended is None:
                continue
            found = search(run_idx + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts[:START_TRIALS]:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _boundary_indices(run_steps: list[int]) -> list[int]:
    idx = 0
    boundaries = [0]
    for steps in run_steps:
        idx += steps
        boundaries.append(idx)
    return boundaries


def _direct_expected(
    inp: np.ndarray,
    path: list[Coord],
    run_steps: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    boundaries = _boundary_indices(run_steps)

    for conn_idx in range(len(run_steps) // 2):
        left = boundaries[2 * conn_idx + 1]
        right = boundaries[2 * conn_idx + 2]
        out[path[left]] = GREY
        out[path[right]] = GREY
        for cell in path[left + 1 : right]:
            out[cell] = ORANGE

    out[path[0]] = marker_colors[0]
    out[path[-1]] = marker_colors[-1]
    return out


def _candidate_colors(bg: int, n_segments: int) -> list[int]:
    available = [color for color in range(10) if color not in (bg, GREY, ORANGE)]
    needed = n_segments + 1
    assert len(available) >= needed
    return random.sample(available, needed)


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    run_steps: list[int] = []
    for idx in range(2 * n_segments - 1):
        bounds = SEGMENT_STEPS if idx % 2 == 0 else CONNECTOR_STEPS
        run_steps.append(random.randint(*bounds))

    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    path = _build_path(run_steps, (rows, cols))
    if path is None:
        return None

    bg = random.choice([color for color in range(10) if color not in (GREY, ORANGE)])
    marker_colors = _candidate_colors(bg, n_segments)
    boundaries = _boundary_indices(run_steps)

    grid = np.full((rows, cols), bg, dtype=int)

    # Paint grey segment interiors.
    for seg_idx in range(n_segments):
        left = boundaries[2 * seg_idx]
        right = boundaries[2 * seg_idx + 1]
        for cell in path[left + 1 : right]:
            grid[cell] = GREY

    # Paint outer markers.
    grid[path[0]] = marker_colors[0]
    grid[path[-1]] = marker_colors[-1]

    # Paint matched marker pairs.
    for conn_idx in range(n_segments - 1):
        color = marker_colors[conn_idx + 1]
        left = boundaries[2 * conn_idx + 1]
        right = boundaries[2 * conn_idx + 2]
        grid[path[left]] = color
        grid[path[right]] = color

    counts = Counter(map(int, grid.flat))
    if any(counts[bg] <= count for color, count in counts.items() if color != bg):
        return None

    expected = _direct_expected(grid, path, run_steps, marker_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-segment input")
