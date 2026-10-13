from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT_TURN = {DIRS[i]: DIRS[(i - 1) % 4] for i in range(4)}
RIGHT_TURN = {DIRS[i]: DIRS[(i + 1) % 4] for i in range(4)}
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

MIN_HEIGHT = 26
MAX_HEIGHT = 38
MIN_WIDTH = 26
MAX_WIDTH = 38
MIN_COMPONENTS = 2
MAX_COMPONENTS = 3
MIN_TURNS = 1
MAX_TURNS = 3
MIN_SEG_LEN = 2
MAX_SEG_LEN = 5
MAX_STUB_LEN = 2
COMPONENT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _add(pos: tuple[int, int], direction: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return pos[0] + direction[0] * scale, pos[1] + direction[1] * scale


def _neighbors4(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def _translate_cells(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _allowed_attach(cell: tuple[int, int], occupied: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return False
    actual = {nbr for nbr in _neighbors4(cell) if nbr in occupied}
    return actual == allowed


def _find_sources(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    sources: list[tuple[int, tuple[int, int]]] = []
    for r, c in np.argwhere(~np.isin(grid, [BG, GREY, ORANGE])):
        sources.append((int(grid[r, c]), (int(r), int(c))))
    sources.sort(key=lambda item: item[1])
    return sources


def _find_intersections(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    intersections: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != GREY:
                continue
            if any(int(grid[r + dr, c + dc]) != GREY for dr, dc in DIRS):
                continue
            marks = []
            for dr, dc in DIAGONALS:
                if int(grid[r + dr, c + dc]) == ORANGE:
                    marks.append((dr, dc))
            assert len(marks) == 1, "each intersection must have exactly one orange diagonal marker"
            intersections[(r, c)] = marks[0]
    return intersections


def _trace_signal(
    grid: np.ndarray,
    source: tuple[int, int],
    intersections: dict[tuple[int, int], tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    grey_neighbors = [nbr for nbr in _neighbors4(source) if _in_bounds(grid.shape, nbr) and int(grid[nbr]) == GREY]
    assert len(grey_neighbors) == 1, "each source must touch exactly one grey path cell"

    route = [grey_neighbors[0]]
    current = grey_neighbors[0]
    direction = (current[0] - source[0], current[1] - source[1])

    while True:
        if current in intersections:
            dr, dc = intersections[current]
            vertical = (dr, 0)
            horizontal = (0, dc)
            if direction == horizontal:
                direction = vertical
            elif direction == vertical:
                direction = horizontal
            elif direction[0] == 0:
                direction = vertical
            else:
                direction = horizontal
            nxt = _add(current, direction)
            assert _in_bounds(grid.shape, nxt) and int(grid[nxt]) == GREY
            current = nxt
            route.append(current)
            continue

        nxt = _add(current, direction)
        if not _in_bounds(grid.shape, nxt) or int(grid[nxt]) != GREY:
            return route, current, direction
        current = nxt
        route.append(current)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each colored signal along the grey path network.

    Black is background, grey cells are path cells, and orange diagonal markers
    disambiguate turns at true four-way intersections. When a signal reaches a
    grey dead end, it extends outward in its current direction with a bar whose
    length equals the number of grey path cells it traversed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    intersections = _find_intersections(g)
    sources = _find_sources(g)
    assert sources, "expected at least one colored source"

    out = g.copy()
    for color, source in sources:
        route, endpoint, direction = _trace_signal(g, source, intersections)
        length = len(route)
        for step in range(1, length + 1):
            target = _add(endpoint, direction, step)
            if not _in_bounds(g.shape, target):
                break
            out[target] = color
    return out


def _build_component(source_color: int) -> dict | None:
    for _ in range(COMPONENT_ATTEMPTS):
        start_dir = random.choice(DIRS)
        turns = random.randint(MIN_TURNS, MAX_TURNS)

        route = [(0, 0)]
        route_set = {(0, 0)}
        intersections: list[dict[str, tuple[int, int]]] = []
        current = (0, 0)
        direction = start_dir
        valid = True

        for _turn_idx in range(turns):
            seg_len = random.randint(MIN_SEG_LEN, MAX_SEG_LEN)
            for _ in range(seg_len):
                nxt = _add(current, direction)
                if not _allowed_attach(nxt, route_set, {current}):
                    valid = False
                    break
                route.append(nxt)
                route_set.add(nxt)
                current = nxt
            if not valid:
                break
            out_dir = random.choice([LEFT_TURN[direction], RIGHT_TURN[direction]])
            intersections.append({"center": current, "move_in": direction, "move_out": out_dir})
            direction = out_dir
        if not valid:
            continue

        seg_len = random.randint(MIN_SEG_LEN, MAX_SEG_LEN)
        for _ in range(seg_len):
            nxt = _add(current, direction)
            if not _allowed_attach(nxt, route_set, {current}):
                valid = False
                break
            route.append(nxt)
            route_set.add(nxt)
            current = nxt
        if not valid:
            continue

        endpoint = current
        final_dir = direction
        occupied_grey = set(route_set)
        stub_paths: list[list[tuple[int, int]]] = []
        markers: list[tuple[int, int]] = []

        for info in intersections:
            center = info["center"]
            move_in = info["move_in"]
            move_out = info["move_out"]

            marker = _add(_add(center, move_in), move_out)
            if marker in occupied_grey or marker in markers:
                valid = False
                break
            markers.append(marker)

            for stub_dir in (move_in, (-move_out[0], -move_out[1])):
                free_len = 0
                probe_occupied = occupied_grey | set().union(*map(set, stub_paths)) if stub_paths else occupied_grey
                for step in range(1, MAX_STUB_LEN + 1):
                    cell = _add(center, stub_dir, step)
                    prev = center if step == 1 else _add(center, stub_dir, step - 1)
                    if not _allowed_attach(cell, probe_occupied, {prev}):
                        break
                    free_len = step
                    probe_occupied = probe_occupied | {cell}
                if free_len == 0:
                    valid = False
                    break
                stub_len = random.randint(1, free_len)
                stub = []
                for step in range(1, stub_len + 1):
                    cell = _add(center, stub_dir, step)
                    stub.append(cell)
                    occupied_grey.add(cell)
                stub_paths.append(stub)
            if not valid:
                break
        if not valid:
            continue

        source = _add(route[0], (-start_dir[0], -start_dir[1]))
        occupied_input = occupied_grey | set(markers)
        if source in occupied_input:
            continue
        if {nbr for nbr in _neighbors4(source) if nbr in occupied_input} != {route[0]}:
            continue

        extension: list[tuple[int, int]] = []
        occupied_for_extension = occupied_input | {source}
        prev = endpoint
        for _ in range(len(route)):
            cell = _add(prev, final_dir)
            if not _allowed_attach(cell, occupied_for_extension | set(extension), {prev}):
                valid = False
                break
            extension.append(cell)
            prev = cell
        if not valid:
            continue

        component = {
            "source_color": source_color,
            "source": source,
            "route": route,
            "route_set": route_set,
            "endpoint": endpoint,
            "final_dir": final_dir,
            "intersections": intersections,
            "stub_paths": stub_paths,
            "markers": markers,
            "extension": extension,
        }
        if _validate_component(component):
            return component
    return None


def _validate_component(component: dict) -> bool:
    route = component["route"]
    route_set = set(component["route_set"])
    centers = {info["center"] for info in component["intersections"]}
    stub_paths = component["stub_paths"]
    source = component["source"]
    markers = set(component["markers"])
    extension = component["extension"]

    grey = set(route_set)
    stub_endpoints = set()
    stub_internal = set()
    for stub in stub_paths:
        if not stub:
            return False
        stub_endpoints.add(stub[-1])
        stub_internal.update(stub[:-1])
        grey.update(stub)

    start = route[0]
    endpoint = route[-1]
    for cell in grey:
        degree = sum(nbr in grey for nbr in _neighbors4(cell))
        if cell in centers:
            if degree != 4:
                return False
        elif cell == start or cell == endpoint or cell in stub_endpoints:
            if degree != 1:
                return False
        else:
            if degree != 2:
                return False

    input_cells = grey | markers | {source}
    if {nbr for nbr in _neighbors4(source) if nbr in grey} != {start}:
        return False
    if {nbr for nbr in _neighbors4(source) if nbr in input_cells} != {start}:
        return False

    for info in component["intersections"]:
        center = info["center"]
        move_in = info["move_in"]
        move_out = info["move_out"]
        marker = _add(_add(center, move_in), move_out)
        if marker not in markers:
            return False
        expected_neighbors = {
            _add(center, (-move_in[0], -move_in[1])),
            _add(center, move_out),
            _add(center, move_in),
            _add(center, (-move_out[0], -move_out[1])),
        }
        if {nbr for nbr in _neighbors4(center) if nbr in grey} != expected_neighbors:
            return False

    occupied = input_cells.copy()
    prev = endpoint
    for cell in extension:
        if cell in occupied:
            return False
        if {nbr for nbr in _neighbors4(cell) if nbr in occupied} != {prev}:
            return False
        occupied.add(cell)
        prev = cell

    return True


def _padding(cells: Iterable[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for cell in cells:
        for nbr in _neighbors8(cell):
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w:
                padded.add(nbr)
    return padded


def _place_component(component: dict, h: int, w: int, reserved: set[tuple[int, int]]) -> dict | None:
    footprint = (
        set(component["route_set"])
        | set(component["markers"])
        | {component["source"]}
        | set(component["extension"])
    )
    for stub in component["stub_paths"]:
        footprint.update(stub)

    min_r, min_c, max_r, max_c = _bbox(footprint)
    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        dr = random.randint(row_lo, row_hi)
        dc = random.randint(col_lo, col_hi)
        placed_footprint = set(_translate_cells(footprint, dr, dc))
        if placed_footprint & reserved:
            continue
        placed = {
            "source_color": component["source_color"],
            "source": _translate_cells([component["source"]], dr, dc)[0],
            "route": _translate_cells(component["route"], dr, dc),
            "route_set": set(_translate_cells(component["route_set"], dr, dc)),
            "endpoint": _translate_cells([component["endpoint"]], dr, dc)[0],
            "final_dir": component["final_dir"],
            "intersections": [
                {
                    "center": _translate_cells([info["center"]], dr, dc)[0],
                    "move_in": info["move_in"],
                    "move_out": info["move_out"],
                }
                for info in component["intersections"]
            ],
            "stub_paths": [_translate_cells(stub, dr, dc) for stub in component["stub_paths"]],
            "markers": _translate_cells(component["markers"], dr, dc),
            "extension": _translate_cells(component["extension"], dr, dc),
            "reserved": _padding(placed_footprint, h, w),
        }
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    w = random.randint(MIN_WIDTH, MAX_WIDTH)
    n_components = random.randint(MIN_COMPONENTS, MAX_COMPONENTS)
    colors = random.sample(SOURCE_COLORS, n_components)

    raw_components = []
    for color in colors:
        component = _build_component(color)
        if component is None:
            return None
        raw_components.append(component)
    raw_components.sort(
        key=lambda comp: len(comp["route"]) + sum(len(stub) for stub in comp["stub_paths"]) + len(comp["extension"]),
        reverse=True,
    )

    reserved: set[tuple[int, int]] = set()
    placed_components = []
    for component in raw_components:
        placed = _place_component(component, h, w, reserved)
        if placed is None:
            return None
        reserved |= placed["reserved"]
        placed_components.append(placed)

    grid = np.full((h, w), BG, dtype=int)
    for component in placed_components:
        for r, c in component["route_set"]:
            grid[r, c] = GREY
        for stub in component["stub_paths"]:
            for r, c in stub:
                grid[r, c] = GREY
        for r, c in component["markers"]:
            grid[r, c] = ORANGE
        sr, sc = component["source"]
        grid[sr, sc] = component["source_color"]

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if len(_find_sources(grid)) != n_components:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
