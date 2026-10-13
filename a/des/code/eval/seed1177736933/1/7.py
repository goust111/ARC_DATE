from __future__ import annotations

from collections import Counter, deque

import numpy as np


PIPE = 5
VALVE = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GRID_MIN = 16
GRID_MAX = 30
MAX_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _source_info(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], int]]:
    mask = (grid != bg) & (grid != PIPE) & (grid != VALVE)
    return [
        ((int(r), int(c)), int(grid[r, c]))
        for r, c in np.argwhere(mask)
    ]


def _bfs_distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    dist = -np.ones((h, w), dtype=int)
    sr, sc = start
    if not passable[sr, sc]:
        return dist
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        base = int(dist[r, c])
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = base + 1
                q.append((rr, cc))
    return dist


def _solve_grid(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    source_mask = (g != bg) & (g != PIPE) & (g != VALVE)
    sources = _source_info(g, bg)
    out = g.copy()
    if not sources:
        return out, {"tie_cells": [], "multi_reachable": [], "distances": []}

    passable = (g == PIPE) | source_mask
    distances = [(pos, color, _bfs_distances(passable, pos)) for pos, color in sources]

    tie_cells: list[tuple[int, int]] = []
    multi_reachable: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == PIPE):
        rr = int(r)
        cc = int(c)
        best_dist = None
        winners: list[int] = []
        reachers = 0
        for _pos, color, dist in distances:
            d = int(dist[rr, cc])
            if d < 0:
                continue
            reachers += 1
            if best_dist is None or d < best_dist:
                best_dist = d
                winners = [int(color)]
            elif d == best_dist:
                winners.append(int(color))
        if reachers > 1:
            multi_reachable.append((rr, cc))
        if best_dist is None:
            continue
        if len(set(winners)) == 1:
            out[rr, cc] = winners[0]
        else:
            tie_cells.append((rr, cc))

    return out, {"tie_cells": tie_cells, "multi_reachable": multi_reachable, "distances": distances}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread fluid from colored pipe endpoints through a grey pipe network.

    Grey pixels are passable pipe cells. Maroon pixels are valves and block flow.
    Colored source pixels are starting points. Each grey pipe cell is colored by the
    uniquely closest reachable source in 4-neighbor pipe distance. Equal-distance
    contests stay grey.
    """
    solved, _info = _solve_grid(np.asarray(grid, dtype=int))
    return solved


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(int(ar), int(c)) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(int(r), int(ac)) for r in range(ar, br + step, step)]
    raise ValueError("segment endpoints must be orthogonally aligned")


def _pick_internal_point(
    a: tuple[int, int],
    b: tuple[int, int],
    rng: np.random.Generator,
    pad: int = 2,
) -> tuple[int, int]:
    cells = _segment(a, b)
    candidates = cells[pad : len(cells) - pad]
    if not candidates:
        raise ValueError("segment too short for padded internal point")
    return tuple(map(int, candidates[int(rng.integers(0, len(candidates)))]))


def _transform_point(point: tuple[int, int], rotation: int, mirror: bool) -> tuple[int, int]:
    r, c = point
    if mirror:
        c = -c
    for _ in range(rotation % 4):
        r, c = c, -r
    return int(r), int(c)


def _transform_points(
    points: list[tuple[int, int]] | set[tuple[int, int]],
    rotation: int,
    mirror: bool,
) -> list[tuple[int, int]]:
    return [_transform_point(point, rotation, mirror) for point in points]


def _choose_colors(rng: np.random.Generator, source_count: int) -> tuple[int, list[int]]:
    palette = [color for color in range(10) if color not in (PIPE, VALVE)]
    bg = int(rng.choice(np.array(palette, dtype=int)))
    source_palette = [color for color in palette if color != bg]
    perm = list(map(int, rng.permutation(np.array(source_palette, dtype=int))))
    return bg, perm[:source_count]


def _build_line_family(
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], bool, bool]:
    bent = bool(rng.integers(0, 2))
    if bent:
        first = int(rng.integers(4, 9))
        second = int(rng.integers(4, 9))
        points = [(0, 0), (0, first), (second, first)]
        valve_segment = (points[0], points[1]) if bool(rng.integers(0, 2)) else (points[1], points[2])
    else:
        length = int(rng.integers(7, 11))
        points = [(0, 0), (0, length)]
        valve_segment = (points[0], points[1])

    cells: set[tuple[int, int]] = set()
    for a, b in zip(points, points[1:]):
        cells.update(_segment(a, b))

    sources = [points[0], points[-1]]
    valves = [_pick_internal_point(valve_segment[0], valve_segment[1], rng)]
    return cells, sources, valves, False, False


def _build_cross_family(
    rng: np.random.Generator,
    want_tie: bool,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], bool, bool]:
    directions = ["N", "S", "W", "E"]
    perm = list(rng.permutation(np.array(directions, dtype=object)))
    source_count = int(rng.integers(3, 5))
    active = perm[:source_count]
    open_pair = list(rng.choice(np.array(active, dtype=object), size=2, replace=False))
    blocked = [direction for direction in active if direction not in open_pair]

    lengths: dict[str, int] = {}
    if want_tie:
        open_length = int(rng.integers(4, 9))
        lengths[str(open_pair[0])] = open_length
        lengths[str(open_pair[1])] = open_length
    else:
        first = int(rng.integers(4, 9))
        candidates = [value for value in range(4, 9) if value % 2 != first % 2]
        second = int(rng.choice(np.array(candidates, dtype=int)))
        lengths[str(open_pair[0])] = first
        lengths[str(open_pair[1])] = second

    for direction in blocked:
        lengths[str(direction)] = int(rng.integers(4, 9))

    def endpoint_for(direction: str) -> tuple[int, int]:
        length = lengths[direction]
        if direction == "N":
            return (-length, 0)
        if direction == "S":
            return (length, 0)
        if direction == "W":
            return (0, -length)
        return (0, length)

    hub = (0, 0)
    cells: set[tuple[int, int]] = set()
    sources: list[tuple[int, int]] = []
    valves: list[tuple[int, int]] = []
    for direction in active:
        endpoint = endpoint_for(str(direction))
        cells.update(_segment(hub, endpoint))
        sources.append(endpoint)
        if direction in blocked:
            valves.append(_pick_internal_point(endpoint, hub, rng))

    return cells, sources, valves, want_tie, True


def _build_zigzag_family(
    rng: np.random.Generator,
    want_tie: bool,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], bool, bool]:
    trunk1 = trunk2 = trunk3 = 0
    for _ in range(100):
        trunk1 = int(rng.integers(4, 9))
        trunk2 = int(rng.integers(4, 9))
        trunk3 = int(rng.integers(4, 9))
        if ((trunk1 + trunk2 + trunk3) % 2 == 0) == want_tie:
            break

    a = (0, 0)
    j1 = (0, trunk1)
    j2 = (trunk2, trunk1)
    b = (trunk2, trunk1 + trunk3)
    branch1 = (-int(rng.integers(4, 8)), trunk1)
    branch2 = (trunk2 + int(rng.integers(4, 8)), trunk1)

    branch_count = int(rng.integers(1, 3))
    branch_specs = [("j1", j1, branch1), ("j2", j2, branch2)]
    chosen = branch_specs if branch_count == 2 else [branch_specs[int(rng.integers(0, 2))]]

    cells: set[tuple[int, int]] = set()
    for p, q in ((a, j1), (j1, j2), (j2, b)):
        cells.update(_segment(p, q))

    sources = [a, b]
    valves: list[tuple[int, int]] = []
    for _name, junction, endpoint in chosen:
        cells.update(_segment(junction, endpoint))
        sources.append(endpoint)
        valves.append(_pick_internal_point(endpoint, junction, rng))

    return cells, sources, valves, want_tie, True


def _place_shape(
    cells: set[tuple[int, int]],
    sources: list[tuple[int, int]],
    valves: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    rotation = int(rng.integers(0, 4))
    mirror = bool(rng.integers(0, 2))

    transformed_cells = _transform_points(sorted(cells), rotation, mirror)
    transformed_sources = _transform_points(sources, rotation, mirror)
    transformed_valves = _transform_points(valves, rotation, mirror)

    all_points = transformed_cells
    min_r = min(r for r, _c in all_points)
    max_r = max(r for r, _c in all_points)
    min_c = min(c for _r, c in all_points)
    max_c = max(c for _r, c in all_points)
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1

    h_low = max(GRID_MIN, shape_h + 3)
    h_high = min(GRID_MAX, shape_h + 10)
    w_low = max(GRID_MIN, shape_w + 3)
    w_high = min(GRID_MAX, shape_w + 10)
    if h_low > h_high or w_low > w_high:
        return None

    height = int(rng.integers(h_low, h_high + 1))
    width = int(rng.integers(w_low, w_high + 1))
    row_offset = int(rng.integers(1, height - shape_h))
    col_offset = int(rng.integers(1, width - shape_w))

    def shift(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [
            (int(r - min_r + row_offset), int(c - min_c + col_offset))
            for r, c in points
        ]

    return set(shift(transformed_cells)), shift(transformed_sources), shift(transformed_valves)


def _network_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _validate_grid(grid: np.ndarray, want_tie: bool, require_multi: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    source_mask = (g != bg) & (g != PIPE) & (g != VALVE)
    sources = _source_info(g, bg)
    if not (2 <= len(sources) <= 4):
        return False

    network_mask = (g == PIPE) | (g == VALVE) | source_mask
    if not _network_connected(network_mask):
        return False

    points = {tuple(map(int, point)) for point in np.argwhere(network_mask)}
    degrees = {
        point: sum((point[0] + dr, point[1] + dc) in points for dr, dc in DIR4)
        for point in points
    }
    endpoints = {point for point, degree in degrees.items() if degree == 1}
    source_positions = {pos for pos, _color in sources}
    if endpoints != source_positions:
        return False

    if int((g == VALVE).sum()) < 1:
        return False
    if int((g == PIPE).sum()) < 6:
        return False

    out, info = _solve_grid(g)
    if np.array_equal(out, g):
        return False
    if not np.array_equal(out[source_mask], g[source_mask]):
        return False
    if not np.array_equal(out[g == VALVE], g[g == VALVE]):
        return False
    if np.any(out[~network_mask] != g[~network_mask]):
        return False

    source_colors = {color for _pos, color in sources}
    claimed_colors = {
        int(out[r, c])
        for r, c in np.argwhere(g == PIPE)
        if int(out[r, c]) in source_colors
    }
    if claimed_colors != source_colors:
        return False

    if require_multi and len(info["multi_reachable"]) == 0:
        return False
    if want_tie and len(info["tie_cells"]) == 0:
        return False
    if not want_tie and len(info["tie_cells"]) != 0:
        return False

    return True


def _make_grid(
    placed: tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]],
    rng: np.random.Generator,
) -> np.ndarray:
    cells, sources, valves = placed
    source_count = len(sources)
    bg, source_colors = _choose_colors(rng, source_count)

    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    grid = np.full((max_r + 1, max_c + 1), bg, dtype=int)
    for r, c in cells:
        grid[r, c] = PIPE
    for r, c in valves:
        grid[r, c] = VALVE
    for (r, c), color in zip(sources, source_colors):
        grid[r, c] = int(color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    family = int(rng.integers(0, 3))
    if family == 0:
        cells, sources, valves, want_tie, require_multi = _build_line_family(rng)
    elif family == 1:
        want_tie = bool(rng.integers(0, 2))
        cells, sources, valves, want_tie, require_multi = _build_cross_family(rng, want_tie)
    else:
        want_tie = bool(rng.integers(0, 2))
        cells, sources, valves, want_tie, require_multi = _build_zigzag_family(rng, want_tie)

    placed = _place_shape(cells, sources, valves, rng)
    if placed is None:
        return None

    grid = _make_grid(placed, rng)
    if not _validate_grid(grid, want_tie, require_multi):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
