from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400
PATH_ATTEMPTS = 300
SIDES = ("top", "bottom", "left", "right")
INWARD_STEP = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _boundary_distance(cell: tuple[int, int], height: int, width: int) -> int:
    r, c = cell
    return min(r, c, height - 1 - r, width - 1 - c)


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == BACKGROUND:
                continue
            color = int(grid[r0, c0])
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _trace_path(coords: list[tuple[int, int]], height: int, width: int) -> tuple[list[tuple[int, int]], int]:
    coord_set = set(coords)
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        r, c = cell
        neighbor_map[cell] = [nbr for nbr in _neighbors4(r, c) if nbr in coord_set]

    endpoints = [cell for cell, nbrs in neighbor_map.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    edge_endpoints = [cell for cell in endpoints if _boundary_distance(cell, height, width) == 1]
    assert len(edge_endpoints) == 1, "each path must have one unique boundary-adjacent start"

    start = edge_endpoints[0]
    path = [start]
    prev: tuple[int, int] | None = None
    current = start
    prev_step: tuple[int, int] | None = None
    turns = 0

    while True:
        next_cells = [nbr for nbr in neighbor_map[current] if nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "paths must not branch"
        nxt = next_cells[0]
        step = (nxt[0] - current[0], nxt[1] - current[1])
        if prev_step is not None and step != prev_step:
            turns += 1
        path.append(nxt)
        prev = current
        current = nxt
        prev_step = step

    assert _boundary_distance(path[-1], height, width) != 1, "terminal endpoint must not be boundary-adjacent"
    return path, turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored path from its boundary-adjacent start and grow a square at the terminal cell.

    Input structure:
    - The background is black.
    - Each foreground component is a single-color, one-cell-wide 4-connected path with no branches.
    - Each path has two endpoints, and exactly one endpoint lies one cell away from a grid boundary.
    - Different paths are separated from one another and may use different colors.

    Rule:
    - For every path, choose the boundary-adjacent endpoint as the start and walk along the chain to the other endpoint.
    - Count each 90-degree direction change encountered while tracing the path.
    - If the turn count is T, draw a solid square of the same color and side length T+1 with the terminal endpoint as the square's top-left cell.
    - Clip that square to the grid if it would extend past the bottom or right edge, and keep all original path pixels unchanged.

    Key insight:
    - The only hidden cue is traversal direction: the edge-near endpoint identifies the start, and only the number of turns, not the total path length, controls the square size.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out = g.copy()

    for comp in _components(g):
        color = int(comp["color"])
        coords = list(comp["coords"])
        path, turns = _trace_path(coords, height, width)
        side = turns + 1
        end_r, end_c = path[-1]
        out[end_r : min(height, end_r + side), end_c : min(width, end_c + side)] = color

    return out


def _perpendicular_options(step: tuple[int, int]) -> list[tuple[int, int]]:
    if step[0] != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _sample_start(rng: np.random.Generator, height: int, width: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 1, int(rng.integers(2, width - 2))
    if side == "bottom":
        return height - 2, int(rng.integers(2, width - 2))
    if side == "left":
        return int(rng.integers(2, height - 2)), 1
    return int(rng.integers(2, height - 2)), width - 2


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_step: tuple[int, int] | None = None
    for idx in range(1, len(path)):
        step = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        if prev_step is not None and step != prev_step:
            turns += 1
        prev_step = step
    return turns


def _can_append(
    path: list[tuple[int, int]],
    candidate: tuple[int, int],
    height: int,
    width: int,
    blocked_paths: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> bool:
    r, c = candidate
    if not (0 <= r < height and 0 <= c < width):
        return False
    if candidate in blocked_paths or candidate in blocked_output:
        return False

    path_set = set(path)
    if candidate in path_set:
        return False

    current = path[-1]
    previous = path[-2] if len(path) >= 2 else None
    same_path_neighbors = 0
    for nr, nc in _neighbors8(r, c):
        nbr = (nr, nc)
        if nbr not in path_set:
            continue
        if nbr == current:
            same_path_neighbors += 1
            continue
        if previous is not None and nbr == previous and abs(nr - r) == 1 and abs(nc - c) == 1:
            continue
        return False

    if same_path_neighbors != 1:
        return False
    return True


def _square_cells(endpoint: tuple[int, int], side: int, height: int, width: int) -> set[tuple[int, int]]:
    end_r, end_c = endpoint
    cells = set()
    for r in range(end_r, min(height, end_r + side)):
        for c in range(end_c, min(width, end_c + side)):
            cells.add((r, c))
    return cells


def _validate_path_shape(path: list[tuple[int, int]], start: tuple[int, int], height: int, width: int) -> bool:
    if len(path) < 2:
        return False
    if path[0] != start or len(set(path)) != len(path):
        return False

    coord_set = set(path)
    degrees = {}
    for cell in path:
        r, c = cell
        degree = sum((nbr in coord_set) for nbr in _neighbors4(r, c))
        degrees[cell] = degree
        if degree not in (1, 2):
            return False

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        return False
    if start not in endpoints or path[-1] not in endpoints:
        return False
    if _boundary_distance(start, height, width) != 1:
        return False
    if _boundary_distance(path[-1], height, width) == 1:
        return False
    return True


def _path_buffer(path: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    buf: set[tuple[int, int]] = set()
    for r, c in path:
        buf.add((r, c))
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < height and 0 <= nc < width:
                buf.add((nr, nc))
    return buf


def _build_path(
    rng: np.random.Generator,
    height: int,
    width: int,
    turns: int,
    clip_mode: bool,
    blocked_paths: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    side_name = str(rng.choice(SIDES))
    start = _sample_start(rng, height, width, side_name)
    if start in blocked_paths or start in blocked_output:
        return None

    directions = [INWARD_STEP[side_name]]
    current_dir = directions[0]
    for turn_idx in range(turns):
        options = _perpendicular_options(current_dir)
        rng.shuffle(options)
        if clip_mode and turn_idx == turns - 1:
            options.sort(key=lambda step: (step[0] > 0) + (step[1] > 0), reverse=True)
        current_dir = options[0]
        directions.append(current_dir)

    path = [start]
    max_segment = max(4, min(8, min(height, width) // 2))
    for seg_idx, direction in enumerate(directions):
        min_len = 2 if seg_idx == 0 else 1
        candidates: list[tuple[int, int]] = []
        probe = list(path)
        while len(candidates) < max_segment:
            last_r, last_c = probe[-1]
            nxt = (last_r + direction[0], last_c + direction[1])
            if not _can_append(probe, nxt, height, width, blocked_paths, blocked_output):
                break
            probe.append(nxt)
            candidates.append(nxt)
        if len(candidates) < min_len:
            return None

        if clip_mode and seg_idx == len(directions) - 1:
            low = max(min_len, len(candidates) - min(3, len(candidates) - min_len))
            seg_len = int(rng.integers(low, len(candidates) + 1))
        else:
            seg_len = int(rng.integers(min_len, len(candidates) + 1))
        path.extend(candidates[:seg_len])

    if _count_turns(path) != turns:
        return None
    if not _validate_path_shape(path, start, height, width):
        return None

    square = _square_cells(path[-1], turns + 1, height, width)
    clipped = len(square) < (turns + 1) * (turns + 1)
    if clipped != clip_mode:
        return None
    if any(cell in occupied_input for cell in square if cell not in set(path)):
        return None
    if any(cell in occupied_output for cell in square):
        return None
    return path


def _sample_specs(rng: np.random.Generator, path_count: int) -> list[dict[str, int | bool]]:
    specs: list[dict[str, int | bool]] = [{"turns": int(rng.integers(1, 5)), "clip": False}]
    want_clip = bool(rng.integers(0, 2))
    want_straight = path_count >= 2 and bool(rng.integers(0, 2))

    if want_straight:
        specs.append({"turns": 0, "clip": False})

    while len(specs) < path_count:
        turns = int(rng.integers(0, 5))
        if turns == 0 and not any(int(spec["turns"]) > 0 for spec in specs):
            turns = int(rng.integers(1, 5))
        specs.append({"turns": turns, "clip": False})

    if want_clip:
        turned_indices = [idx for idx, spec in enumerate(specs) if int(spec["turns"]) > 0]
        clip_idx = int(rng.choice(np.array(turned_indices, dtype=int)))
        specs[clip_idx]["clip"] = True

    order = rng.permutation(len(specs))
    return [specs[int(idx)] for idx in order]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    path_count = int(rng.integers(1, 5))
    min_dim = 10 + 2 * path_count
    max_dim = min(30, min_dim + 8)
    height = int(rng.integers(min_dim, max_dim + 1))
    width = int(rng.integers(min_dim, max_dim + 1))

    colors = list(map(int, rng.choice(np.array(PATH_COLORS, dtype=int), size=path_count, replace=False)))
    specs = _sample_specs(rng, path_count)

    occupied_input: set[tuple[int, int]] = set()
    blocked_paths: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()
    accepted: list[tuple[int, list[tuple[int, int]]]] = []

    for color, spec in zip(colors, specs):
        turns = int(spec["turns"])
        clip_mode = bool(spec["clip"])
        path: list[tuple[int, int]] | None = None
        for _ in range(PATH_ATTEMPTS):
            candidate = _build_path(
                rng,
                height,
                width,
                turns,
                clip_mode,
                blocked_paths,
                occupied_input,
                blocked_output,
                occupied_output,
            )
            if candidate is not None:
                path = candidate
                break
        if path is None:
            return None

        square = _square_cells(path[-1], turns + 1, height, width)
        occupied_input.update(path)
        blocked_paths.update(_path_buffer(path, height, width))
        occupied_output.update(square)
        blocked_output.update(square)
        accepted.append((color, path))

    grid = np.zeros((height, width), dtype=int)
    for color, path in accepted:
        for r, c in path:
            grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-to-square ARC input")
