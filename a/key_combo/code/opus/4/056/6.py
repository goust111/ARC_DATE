from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 600
MIN_RAY_STEPS = 3
MIN_BRIDGE_SPAN = 4
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
BRANCHES = (
    "horizontal_lr",
    "horizontal_tb",
    "vertical_tb",
    "vertical_lr",
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _square_cells_from_corner(corner: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = corner
    dr, dc = direction
    rows = sorted([cr, cr - dr])
    cols = sorted([cc, cc - dc])
    return {(r, c) for r in range(rows[0], rows[1] + 1) for c in range(cols[0], cols[1] + 1)}


def _find_squares(grid: np.ndarray) -> list[dict]:
    comps = _components_of_color(grid, RED)
    squares: list[dict] = []
    for comp in comps:
        if len(comp) != 4:
            raise AssertionError("expected each red component to be a 2x2 square")
        r0, c0, r1, c1 = _bbox(comp)
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if (r1 - r0, c1 - c0) != (1, 1) or set(comp) != expected:
            raise AssertionError("expected solid 2x2 red squares")
        squares.append({"cells": expected, "bbox": (r0, c0, r1, c1)})
    if len(squares) != 2:
        raise AssertionError("expected exactly two red squares")
    return squares


def _marker_match(marker: tuple[int, int], squares: list[dict]) -> dict:
    matches = []
    mr, mc = marker
    for idx, square in enumerate(squares):
        r0, c0, r1, c1 = square["bbox"]
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        for corner in corners:
            cr, cc = corner
            dr = mr - cr
            dc = mc - cc
            if (dr, dc) not in DIAG_DIRS:
                continue
            if marker in square["cells"]:
                continue
            if _square_cells_from_corner(corner, (dr, dc)) != square["cells"]:
                continue
            matches.append(
                {
                    "square_index": idx,
                    "corner": corner,
                    "direction": (dr, dc),
                }
            )
    if len(matches) != 1:
        raise AssertionError("expected each yellow marker to match exactly one square corner")
    return matches[0]


def _trace_ray(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while _in_bounds(shape, r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def _bridge_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        lo, hi = sorted([a[1], b[1]])
        return [(r, c) for c in range(lo, hi + 1)]
    if a[1] == b[1]:
        c = a[1]
        lo, hi = sorted([a[0], b[0]])
        return [(r, c) for r in range(lo, hi + 1)]
    raise AssertionError("endpoints must share a row or a column")


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    squares = _find_squares(g)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    if len(markers) != 2:
        raise AssertionError("expected exactly two yellow markers")

    rays = []
    used_square_indices = set()
    for marker in markers:
        match = _marker_match(marker, squares)
        square_index = int(match["square_index"])
        if square_index in used_square_indices:
            raise AssertionError("expected one marker per square")
        used_square_indices.add(square_index)
        path = _trace_ray(g.shape, marker, match["direction"])
        if len(path) - 1 < MIN_RAY_STEPS:
            raise AssertionError("each ray must extend at least three further cells")
        rays.append(
            {
                "marker": marker,
                "direction": match["direction"],
                "corner": match["corner"],
                "square_index": square_index,
                "path": path,
                "endpoint": path[-1],
            }
        )

    if used_square_indices != {0, 1}:
        raise AssertionError("expected both squares to own one marker")

    square_cells = set().union(*(square["cells"] for square in squares))
    for ray in rays:
        marker = ray["marker"]
        if marker in square_cells:
            raise AssertionError("marker cannot lie inside a square")
        for cell in ray["path"][1:]:
            if cell in square_cells:
                raise AssertionError("ray cannot pass through any red square")

    endpoints = [ray["endpoint"] for ray in rays]
    a, b = endpoints
    if a == b:
        raise AssertionError("ray endpoints must be distinct")
    if a[0] != b[0] and a[1] != b[1]:
        raise AssertionError("ray endpoints must share exactly one axis")
    bridge = _bridge_cells(a, b)
    if len(bridge) < 2:
        raise AssertionError("bridge must be non-trivial")
    if any(cell in square_cells for cell in bridge):
        raise AssertionError("bridge cannot cover a red square")

    return {
        "grid": g,
        "squares": squares,
        "markers": markers,
        "rays": rays,
        "bridge": bridge,
        "endpoints": endpoints,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend two diagonal rays from corner-adjacent markers and bridge their boundary endpoints with a green axis-aligned line.

    Input structure:
    - The grid has black background, exactly two red 2x2 solid squares, and exactly two yellow pixels.
    - Each yellow pixel sits one diagonal step outside a unique corner of one red square.
    - The diagonal direction from that square corner through the yellow marker can continue to the grid boundary.

    Rule:
    - Keep both red squares and both original yellow markers.
    - For each marker, extend a yellow diagonal ray in the direction away from its square until the next step would leave the grid; the last in-bounds ray cell is that ray's endpoint.
    - Draw one green horizontal or vertical segment between the two endpoints, including the endpoints.
    - If the green bridge covers any yellow ray cells, the green bridge overwrites them, but red square cells are never overwritten.

    Key insight:
    - The yellow markers do not just show a direction; each marker-square pair determines a full diagonal ray, and the bridge location is only knowable after finding the two boundary endpoints of those rays.
    """
    info = _analyze(grid)
    out = np.array(grid, dtype=int, copy=True)

    for ray in info["rays"]:
        for r, c in ray["path"]:
            if int(out[r, c]) == RED:
                raise AssertionError("ray cannot overwrite red")
            out[r, c] = YELLOW

    for r, c in info["bridge"]:
        if int(out[r, c]) == RED:
            raise AssertionError("bridge cannot overwrite red")
        out[r, c] = GREEN

    return out


def _manhattan_gap(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> int:
    return min(abs(ra - rb) + abs(ca - cb) for ra, ca in cells_a for rb, cb in cells_b)


def _square_from_endpoint(
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    er, ec = endpoint
    dr, dc = direction
    marker = (er - steps * dr, ec - steps * dc)
    corner = (marker[0] - dr, marker[1] - dc)
    return marker, _square_cells_from_corner(corner, direction)


def _valid_steps(
    shape: tuple[int, int],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
) -> list[int]:
    valid = []
    for steps in range(MIN_RAY_STEPS, 31):
        marker, square = _square_from_endpoint(endpoint, direction, steps)
        if not _in_bounds(shape, marker[0], marker[1]):
            continue
        if all(_in_bounds(shape, r, c) for r, c in square):
            valid.append(steps)
    return valid


def _sample_pair_from_range(length: int) -> tuple[int, int] | None:
    coords = list(range(2, length - 2))
    random.shuffle(coords)
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            if abs(a - b) >= MIN_BRIDGE_SPAN:
                return (a, b) if a < b else (b, a)
    return None


def _choose_geometry(shape: tuple[int, int]) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    branch = random.choice(BRANCHES)

    if branch == "horizontal_lr":
        row = random.randint(2, h - 3)
        endpoints = [(row, 0), (row, w - 1)]
        directions = [(random.choice([-1, 1]), -1), (random.choice([-1, 1]), 1)]
        return branch, endpoints, directions

    if branch == "horizontal_tb":
        pair = _sample_pair_from_range(w)
        if pair is None:
            raise RuntimeError("failed to sample horizontal boundary bridge")
        row = random.choice([0, h - 1])
        dr = -1 if row == 0 else 1
        endpoints = [(row, pair[0]), (row, pair[1])]
        directions = [(dr, random.choice([-1, 1])), (dr, random.choice([-1, 1]))]
        return branch, endpoints, directions

    if branch == "vertical_tb":
        col = random.randint(2, w - 3)
        endpoints = [(0, col), (h - 1, col)]
        directions = [(-1, random.choice([-1, 1])), (1, random.choice([-1, 1]))]
        return branch, endpoints, directions

    pair = _sample_pair_from_range(h)
    if pair is None:
        raise RuntimeError("failed to sample vertical boundary bridge")
    col = random.choice([0, w - 1])
    dc = -1 if col == 0 else 1
    endpoints = [(pair[0], col), (pair[1], col)]
    directions = [(random.choice([-1, 1]), dc), (random.choice([-1, 1]), dc)]
    return branch, endpoints, directions


def _input_grid_from_spec(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    squares: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for square in squares:
        for r, c in square:
            grid[r, c] = RED
    for r, c in markers:
        if int(grid[r, c]) != BG:
            raise AssertionError("marker placement overlaps another object")
        grid[r, c] = YELLOW
    return grid


def _validate_example(inp: np.ndarray) -> None:
    info = _analyze(inp)
    out = main(inp)
    expected = np.array(inp, dtype=int, copy=True)

    for ray in info["rays"]:
        for r, c in ray["path"]:
            expected[r, c] = YELLOW
    for r, c in info["bridge"]:
        expected[r, c] = GREEN

    if not np.array_equal(out, expected):
        raise AssertionError("solver output disagrees with analyzed geometry")

    rays_sets = [set(ray["path"]) for ray in info["rays"]]
    if rays_sets[0] & rays_sets[1]:
        raise AssertionError("generated rays should stay disjoint for visual clarity")

    bridge_set = set(info["bridge"])
    overlap_counts = [len(bridge_set & ray_cells) for ray_cells in rays_sets]
    if overlap_counts != [1, 1]:
        raise AssertionError("bridge should only cover each ray at its endpoint")

    for marker in info["markers"]:
        if int(inp[marker]) != YELLOW:
            raise AssertionError("markers must remain explicit in the input")

    square_gap = _manhattan_gap(info["squares"][0]["cells"], info["squares"][1]["cells"])
    if square_gap < 4:
        raise AssertionError("squares must stay well separated")

    nonzero_in = {tuple(map(int, rc)) for rc in np.argwhere(inp != BG)}
    nonzero_out = {tuple(map(int, rc)) for rc in np.argwhere(out != BG)}
    if not nonzero_in < nonzero_out:
        raise AssertionError("output must add new visible structure")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    try:
        _branch, endpoints, directions = _choose_geometry(shape)
    except RuntimeError:
        return None

    step_options = [_valid_steps(shape, endpoint, direction) for endpoint, direction in zip(endpoints, directions)]
    if any(not opts for opts in step_options):
        return None

    trials = 0
    while trials < 80:
        trials += 1
        steps = [random.choice(opts) for opts in step_options]
        markers: list[tuple[int, int]] = []
        squares: list[set[tuple[int, int]]] = []
        for endpoint, direction, step_count in zip(endpoints, directions, steps):
            marker, square = _square_from_endpoint(endpoint, direction, step_count)
            markers.append(marker)
            squares.append(square)

        if len(set(markers)) != 2:
            continue

        occupied_square_cells = set().union(*squares)
        if len(occupied_square_cells) != 8:
            continue
        if any(marker in occupied_square_cells for marker in markers):
            continue

        ray_paths = [_trace_ray(shape, marker, direction) for marker, direction in zip(markers, directions)]
        if any(len(path) - 1 < MIN_RAY_STEPS for path in ray_paths):
            continue
        if any(set(path[1:]) & occupied_square_cells for path in ray_paths):
            continue
        if set(ray_paths[0]) & set(ray_paths[1]):
            continue
        if any(markers[j] in set(ray_paths[i][1:]) for i in range(2) for j in range(2) if i != j):
            continue

        bridge = _bridge_cells(ray_paths[0][-1], ray_paths[1][-1])
        if any(cell in occupied_square_cells for cell in bridge):
            continue

        square_gap = _manhattan_gap(squares[0], squares[1])
        if square_gap < 4:
            continue

        inp = _input_grid_from_spec(shape, markers, squares)
        try:
            _validate_example(inp)
        except AssertionError:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray bridge puzzle")
