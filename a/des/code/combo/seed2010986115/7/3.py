from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG_COLOR = 0
BLUE_COLOR = 1
PATH_COLOR = 5
SOURCE_COLORS = (2, 3, 4, 6, 7, 8, 9)

DIR_OFFSETS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
DIR_ORDER = ("N", "E", "S", "W")
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
BRANCH_DIRECTIONS = {"H": ("N", "S"), "V": ("E", "W")}

# The preferred exit is encoded by a blue cue placed in a nearby corner cell.
# main() also accepts an orthogonally adjacent blue cue when present.
PREFERENCE_MARKER_OFFSETS = {
    "N": (-1, -1),
    "E": (-1, 1),
    "S": (1, 1),
    "W": (1, -1),
}

GRID_H_RANGE = (17, 29)
GRID_W_RANGE = (17, 29)
BRANCH_COUNT_RANGE = (2, 6)
BRANCH_LENGTH_RANGE = (3, 8)
PIVOT_CLEARANCE = 2
GEN_ATTEMPTS = 500


def _neighbors4(cell: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
    r, c = cell
    return [(name, (r + dr, c + dc)) for name, (dr, dc) in DIR_OFFSETS.items()]


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _segment_cells(segment: dict) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = segment["a"], segment["b"]
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    step = 1 if r2 >= r1 else -1
    return [(r, c1) for r in range(r1, r2 + step, step)]


def _path_degrees(path: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {cell: sum(neighbor in path for _dir, neighbor in _neighbors4(cell)) for cell in path}


def _is_straight_degree2(cell: tuple[int, int], path: set[tuple[int, int]]) -> bool:
    r, c = cell
    vertical = (r - 1, c) in path and (r + 1, c) in path
    horizontal = (r, c - 1) in path and (r, c + 1) in path
    return vertical or horizontal


def _detect_preferences(
    grid: np.ndarray,
    degrees: dict[tuple[int, int], int],
) -> dict[tuple[int, int], str] | None:
    prefs: dict[tuple[int, int], str] = {}
    for cell, degree in degrees.items():
        if degree < 3:
            continue
        r, c = cell
        found: list[str] = []
        for direction in DIR_ORDER:
            dr, dc = DIR_OFFSETS[direction]
            cue = (r + dr, c + dc)
            if _in_bounds(cue, grid.shape) and int(grid[cue]) == BLUE_COLOR:
                found.append(direction)
        for direction in DIR_ORDER:
            dr, dc = PREFERENCE_MARKER_OFFSETS[direction]
            cue = (r + dr, c + dc)
            if _in_bounds(cue, grid.shape) and int(grid[cue]) == BLUE_COLOR:
                found.append(direction)
        if len(found) != 1:
            return None
        prefs[cell] = found[0]
    return prefs


def _collect_sources(
    grid: np.ndarray,
    path: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]] | None:
    markers = np.argwhere((grid != BG_COLOR) & (grid != BLUE_COLOR) & (grid != PATH_COLOR))
    sources: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    seen_colors: set[int] = set()
    for r, c in markers:
        marker = (int(r), int(c))
        color = int(grid[marker])
        if color in seen_colors:
            return None
        seen_colors.add(color)
        adjacent_path = [neighbor for _dir, neighbor in _neighbors4(marker) if neighbor in path]
        if len(adjacent_path) != 1:
            return None
        start = adjacent_path[0]
        if degrees.get(start) != 1:
            return None
        sources.append((color, marker, start))
    sources.sort(key=lambda item: item[1])
    return sources


def _trace_signals(
    grid: np.ndarray,
) -> tuple[dict[tuple[int, int], tuple[int, int, int]], dict[str, int], bool] | None:
    path = {(int(r), int(c)) for r, c in np.argwhere(grid == PATH_COLOR)}
    if not path:
        return None

    degrees = _path_degrees(path)
    if any(degree == 2 and not _is_straight_degree2(cell, path) for cell, degree in degrees.items()):
        return None

    prefs = _detect_preferences(grid, degrees)
    if prefs is None:
        return None

    sources = _collect_sources(grid, path, degrees)
    if not sources:
        return None

    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    tie_seen = False
    stats = {
        "preferred_turns": 0,
        "straight_fallbacks": 0,
        "forced_continuations": 0,
    }

    for order, (color, marker, current) in enumerate(sources):
        previous = marker
        time = 0
        seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        step_limit = len(path) + 4

        while True:
            state = (previous, current)
            if state in seen_states:
                return None
            seen_states.add(state)

            prev_best = best.get(current)
            if prev_best is None or time > prev_best[0] or (time == prev_best[0] and order > prev_best[2]):
                if prev_best is not None and time == prev_best[0] and order != prev_best[2]:
                    tie_seen = True
                best[current] = (time, color, order)
            elif prev_best is not None and time == prev_best[0] and order != prev_best[2]:
                tie_seen = True

            exits = [(direction, neighbor) for direction, neighbor in _neighbors4(current) if neighbor in path and neighbor != previous]
            if not exits:
                break

            if degrees[current] >= 3:
                incoming = next((direction for direction, neighbor in _neighbors4(current) if neighbor == previous), None)
                if incoming is None:
                    return None
                available = {direction: neighbor for direction, neighbor in exits}
                preferred = prefs[current]
                straight = OPPOSITE[incoming]
                if preferred in available:
                    nxt = available[preferred]
                    stats["preferred_turns"] += 1
                elif straight in available:
                    nxt = available[straight]
                    stats["straight_fallbacks"] += 1
                elif len(available) == 1:
                    nxt = next(iter(available.values()))
                    stats["forced_continuations"] += 1
                else:
                    return None
            else:
                if len(exits) != 1:
                    return None
                nxt = exits[0][1]

            previous, current = current, nxt
            time += 1
            if time > step_limit:
                return None

    return best, stats, tie_seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each endpoint marker along the grey path network, turning at
    intersections when that node's blue cue points to a usable exit.
    """
    g = np.asarray(grid, dtype=int)
    traced = _trace_signals(g)
    assert traced is not None, "invalid path-network input"
    best, _stats, _tie_seen = traced

    out = np.full_like(g, BG_COLOR)
    out[g == PATH_COLOR] = PATH_COLOR
    for cell, (_time, color, _order) in best.items():
        out[cell] = color
    return out


def _sample_root_segment(shape: tuple[int, int]) -> dict:
    h, w = shape
    orientation = random.choice(("H", "V"))
    if orientation == "V":
        c = random.randint(4, w - 5)
        r0 = random.randint(2, max(2, h // 4))
        r1 = random.randint(min(h - 3, (3 * h) // 4), h - 3)
        return {"orientation": "V", "a": (r0, c), "b": (r1, c)}
    r = random.randint(4, h - 5)
    c0 = random.randint(2, max(2, w // 4))
    c1 = random.randint(min(w - 3, (3 * w) // 4), w - 3)
    return {"orientation": "H", "a": (r, c0), "b": (r, c1)}


def _too_close_to_other_junctions(
    pivot: tuple[int, int],
    junction_branches: dict[tuple[int, int], set[str]],
) -> bool:
    for other in junction_branches:
        if other == pivot:
            continue
        if max(abs(pivot[0] - other[0]), abs(pivot[1] - other[1])) <= PIVOT_CLEARANCE:
            return True
    return False


def _branch_axis_matches(segment: dict, used_dirs: set[str]) -> bool:
    return set(used_dirs).issubset(set(BRANCH_DIRECTIONS[segment["orientation"]]))


def _try_add_branch(
    shape: tuple[int, int],
    path: set[tuple[int, int]],
    segments: list[dict],
    junction_branches: dict[tuple[int, int], set[str]],
) -> bool:
    segment = random.choice(segments)
    cells = _segment_cells(segment)
    if len(cells) < 5:
        return False

    extendable = [
        pivot
        for pivot, used_dirs in junction_branches.items()
        if pivot in cells[1:-1] and len(used_dirs) == 1 and _branch_axis_matches(segment, used_dirs)
    ]

    if extendable and random.random() < 0.35:
        pivot = random.choice(extendable)
    else:
        candidates = [pivot for pivot in cells[2:-2] if not _too_close_to_other_junctions(pivot, junction_branches)]
        if not candidates:
            return False
        pivot = random.choice(candidates)

    perpendicular_dirs = BRANCH_DIRECTIONS[segment["orientation"]]
    used_dirs = junction_branches[pivot]
    available_dirs = [direction for direction in perpendicular_dirs if direction not in used_dirs]
    if not available_dirs:
        return False

    if len(used_dirs) == 1:
        opposite = OPPOSITE[next(iter(used_dirs))]
        if opposite in available_dirs and random.random() < 0.7:
            direction = opposite
        else:
            direction = random.choice(available_dirs)
    else:
        direction = random.choice(available_dirs)

    dr, dc = DIR_OFFSETS[direction]
    r, c = pivot
    max_length = 0
    while True:
        r += dr
        c += dc
        if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
            break
        max_length += 1
    if max_length < BRANCH_LENGTH_RANGE[0]:
        return False

    length = random.randint(BRANCH_LENGTH_RANGE[0], min(BRANCH_LENGTH_RANGE[1], max_length))
    new_cells = [(pivot[0] + dr * step, pivot[1] + dc * step) for step in range(1, length + 1)]
    marker_cell = (pivot[0] + dr * (length + 1), pivot[1] + dc * (length + 1))
    if marker_cell in path or not _in_bounds(marker_cell, shape):
        return False

    for idx, cell in enumerate(new_cells):
        if cell in path:
            return False
        allowed_neighbors = {pivot} if idx == 0 else set()
        for _dir, neighbor in _neighbors4(cell):
            if neighbor in path and neighbor not in allowed_neighbors:
                return False

    for _dir, neighbor in _neighbors4(marker_cell):
        if neighbor in path and neighbor != new_cells[-1]:
            return False

    path.update(new_cells)
    segments.append(
        {
            "orientation": "V" if direction in ("N", "S") else "H",
            "a": pivot,
            "b": new_cells[-1],
        }
    )
    junction_branches[pivot].add(direction)
    return True


def _safe_preferences(neighbor_dirs: set[str]) -> list[str]:
    if len(neighbor_dirs) == 4:
        return list(DIR_ORDER)
    if len(neighbor_dirs) != 3:
        return []
    missing = (set(DIR_ORDER) - neighbor_dirs).pop()
    blocked = OPPOSITE[missing]
    return [direction for direction in DIR_ORDER if direction in neighbor_dirs and direction != blocked]


def _build_candidate() -> np.ndarray | None:
    shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
    segments = [_sample_root_segment(shape)]
    path = set(_segment_cells(segments[0]))
    junction_branches: dict[tuple[int, int], set[str]] = defaultdict(set)

    for _ in range(random.randint(*BRANCH_COUNT_RANGE)):
        for _attempt in range(200):
            if _try_add_branch(shape, path, segments, junction_branches):
                break

    degrees = _path_degrees(path)
    if any(degree == 2 and not _is_straight_degree2(cell, path) for cell, degree in degrees.items()):
        return None

    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    junctions = [cell for cell, degree in degrees.items() if degree >= 3]
    if not junctions or len(endpoints) < 3 or len(endpoints) > len(SOURCE_COLORS):
        return None

    grid = np.full(shape, BG_COLOR, dtype=int)
    for r, c in path:
        grid[r, c] = PATH_COLOR

    source_colors = random.sample(SOURCE_COLORS, len(endpoints))
    for color, endpoint in zip(source_colors, endpoints):
        incoming = [direction for direction, neighbor in _neighbors4(endpoint) if neighbor in path]
        if len(incoming) != 1:
            return None
        outward = OPPOSITE[incoming[0]]
        dr, dc = DIR_OFFSETS[outward]
        marker = (endpoint[0] + dr, endpoint[1] + dc)
        if not _in_bounds(marker, shape) or int(grid[marker]) != BG_COLOR:
            return None
        grid[marker] = color

    for junction in junctions:
        neighbor_dirs = {direction for direction, neighbor in _neighbors4(junction) if neighbor in path}
        choices = _safe_preferences(neighbor_dirs)
        if not choices:
            return None
        preferred = random.choice(choices)
        dr, dc = PREFERENCE_MARKER_OFFSETS[preferred]
        cue = (junction[0] + dr, junction[1] + dc)
        if not _in_bounds(cue, shape) or int(grid[cue]) != BG_COLOR:
            return None
        grid[cue] = BLUE_COLOR

    traced = _trace_signals(grid)
    if traced is None:
        return None
    _best, stats, tie_seen = traced
    if tie_seen:
        return None
    if stats["forced_continuations"] != 0:
        return None
    if stats["preferred_turns"] == 0 or stats["straight_fallbacks"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid routed-path input")
