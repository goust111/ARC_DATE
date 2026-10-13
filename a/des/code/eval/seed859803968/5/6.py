from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _add(cell: tuple[int, int], direction: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * scale, cell[1] + direction[1] * scale


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _dot(cell: tuple[int, int], direction: tuple[int, int]) -> int:
    return cell[0] * direction[0] + cell[1] * direction[1]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_stations(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    stations: list[dict[str, object]] = []
    station_by_cell: dict[tuple[int, int], int] = {}

    for comp in _components(grid == MAROON):
        cells = sorted(comp)
        cellset = set(cells)

        if len(cells) == 2:
            same_row = cells[0][0] == cells[1][0] and abs(cells[0][1] - cells[1][1]) == 1
            same_col = cells[0][1] == cells[1][1] and abs(cells[0][0] - cells[1][0]) == 1
            assert same_row or same_col, "straight station must be a length-2 line"
            station = {
                "kind": "straight",
                "cells": tuple(cells),
                "axis": "h" if same_row else "v",
            }
        elif len(cells) == 3:
            degrees = {
                cell: sum(_add(cell, d) in cellset for d in DIRS)
                for cell in cells
            }
            corners = [cell for cell, degree in degrees.items() if degree == 2]
            arms = [cell for cell, degree in degrees.items() if degree == 1]
            assert len(corners) == 1 and len(arms) == 2, "L station must have one corner and two arms"
            station = {
                "kind": "turn",
                "cells": tuple(cells),
                "corner": corners[0],
                "arms": tuple(sorted(arms)),
            }
        else:
            raise AssertionError("unexpected maroon component size")

        index = len(stations)
        stations.append(station)
        for cell in cells:
            assert cell not in station_by_cell
            station_by_cell[cell] = index

    assert stations, "expected at least one station"
    return stations, station_by_cell


def _parse_segments(grid: np.ndarray) -> list[dict[str, object]]:
    colored = np.argwhere((grid != BLACK) & (grid != GREY) & (grid != MAROON))
    segments: list[dict[str, object]] = []

    for rr, cc in colored:
        r = int(rr)
        c = int(cc)
        color = int(grid[r, c])
        grey_dirs = []
        for direction in DIRS:
            nr, nc = _add((r, c), direction)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                grey_dirs.append(direction)
        assert len(grey_dirs) == 1, "each colored endpoint must attach to exactly one grey segment"
        direction = grey_dirs[0]

        cells = [(r, c)]
        cur = _add((r, c), direction)
        while _in_bounds(grid.shape, cur[0], cur[1]) and int(grid[cur]) == GREY:
            cells.append(cur)
            cur = _add(cur, direction)

        assert 2 <= len(cells) <= 4, "endpoint segment must have total length 2-4"
        grey_end = cells[-1]
        assert int(grid[grey_end]) == GREY

        for i, cell in enumerate(cells):
            prev_count = 0
            for step in DIRS:
                nei = _add(cell, step)
                if _in_bounds(grid.shape, nei[0], nei[1]) and nei in cells:
                    prev_count += 1
            expected = 1 if i in {0, len(cells) - 1} else 2
            assert prev_count == expected, "endpoint segment must be a simple straight path"

        segments.append(
            {
                "color": color,
                "colored_end": (r, c),
                "grey_end": grey_end,
                "direction": direction,
            }
        )

    assert segments, "expected colored endpoint segments"
    return segments


def _pass_station(
    entry: tuple[int, int],
    direction: tuple[int, int],
    station: dict[str, object],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if station["kind"] == "straight":
        axis = station["axis"]
        assert (axis == "h" and direction[0] == 0) or (axis == "v" and direction[1] == 0)
        cells = list(station["cells"])
        far = max(cells, key=lambda cell: _dot(cell, direction))
        return far, direction

    corner = station["corner"]
    arms = list(station["arms"])
    assert entry in arms, "path must enter an L station via an arm"
    assert _add(entry, direction) == corner, "entry arm must point into the corner"
    other = arms[0] if arms[1] == entry else arms[1]
    new_direction = _sub(other, corner)
    assert new_direction in DIRS
    return other, new_direction


def _trace_pair(
    grid: np.ndarray,
    stations: list[dict[str, object]],
    station_by_cell: dict[tuple[int, int], int],
    start: dict[str, object],
    target: dict[str, object],
) -> list[tuple[int, int]]:
    target_cell = target["grey_end"]
    pos = start["grey_end"]
    direction = start["direction"]
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paint: list[tuple[int, int]] = []

    while True:
        state = (pos, direction)
        assert state not in seen_states, "path tracing looped"
        seen_states.add(state)

        cur = _add(pos, direction)
        while True:
            assert _in_bounds(grid.shape, cur[0], cur[1]), "path left the grid"
            if cur == target_cell:
                return paint

            value = int(grid[cur])
            if value == BLACK:
                paint.append(cur)
                cur = _add(cur, direction)
                continue

            assert value == MAROON, "path must hit a station or the target grey endpoint"
            pos, direction = _pass_station(cur, direction, stations[station_by_cell[cur]])
            break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored endpoints and connect their grey ends through maroon stations.

    Grey endpoint segments set the launch direction. A length-2 maroon station keeps
    the current direction, while a length-3 L-shaped maroon station turns the path
    through its corner. The connecting path colors only the background cells; grey
    segments and maroon stations stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    stations, station_by_cell = _parse_stations(g)
    segments = _parse_segments(g)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for seg in segments:
        by_color[int(seg["color"])].append(seg)

    out = g.copy()
    for color in sorted(by_color):
        pair = by_color[color]
        assert len(pair) == 2, "each endpoint color must appear exactly twice"
        paint = _trace_pair(g, stations, station_by_cell, pair[0], pair[1])
        for r, c in paint:
            assert int(out[r, c]) == BLACK
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_MIN = 24
GRID_MAX = 36
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
TURN_COUNT_CHOICES = [1, 2, 3]
SEGMENT_LEN_MIN = 7
SEGMENT_LEN_MAX = 10
STUB_LEN_MIN = 2
STUB_LEN_MAX = 4
MAX_SHAPE_ATTEMPTS = 150
MAX_LAYOUT_ATTEMPTS = 300


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _inflate(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    inflated: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                inflated.add((r + dr, c + dc))
    return inflated


def _sample_shape(color: int) -> dict[str, object] | None:
    turns = random.choice(TURN_COUNT_CHOICES)
    directions = [random.choice(DIRS)]
    for _ in range(turns):
        directions.append(random.choice(_perpendiculars(directions[-1])))
    lengths = [random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX) for _ in directions]

    route = [(0, 0)]
    vertices = [(0, 0)]
    pos = (0, 0)
    for direction, length in zip(directions, lengths):
        for _ in range(length):
            pos = _add(pos, direction)
            route.append(pos)
        vertices.append(pos)

    if len(route) != len(set(route)):
        return None

    start_stub_len = random.randint(STUB_LEN_MIN, STUB_LEN_MAX)
    end_stub_len = random.randint(STUB_LEN_MIN, STUB_LEN_MAX)
    start_stub = [_add(route[0], directions[0], -i) for i in range(start_stub_len)]
    end_stub = [_add(route[-1], directions[-1], i) for i in range(end_stub_len)]

    route_set = set(route)
    if (set(start_stub[1:]) & route_set) or (set(end_stub[1:]) & route_set):
        return None
    if set(start_stub[1:]) & set(end_stub[1:]):
        return None

    turn_station_cells: set[tuple[int, int]] = set()
    for i in range(1, len(vertices) - 1):
        vertex = vertices[i]
        turn_station_cells.add(_add(vertex, directions[i - 1], -1))
        turn_station_cells.add(vertex)
        turn_station_cells.add(_add(vertex, directions[i]))

    candidates: list[tuple[int, int]] = []
    global_start = 0
    for i, length in enumerate(lengths):
        has_prev = i > 0
        has_next = i < len(lengths) - 1
        min_local = 3 if has_prev else 2
        max_local = length - 4 if has_next else length - 3
        for local in range(min_local, max_local + 1):
            candidates.append((global_start + local, global_start + local + 1))
        global_start += length

    if not candidates:
        return None

    random.shuffle(candidates)
    straight_station: tuple[tuple[int, int], tuple[int, int]] | None = None
    for a_idx, b_idx in candidates:
        cell_a = route[a_idx]
        cell_b = route[b_idx]
        if cell_a in turn_station_cells or cell_b in turn_station_cells:
            continue
        station_cells = {cell_a, cell_b}
        if station_cells & set(start_stub[1:]):
            continue
        if station_cells & set(end_stub[1:]):
            continue
        straight_station = (cell_a, cell_b)
        break
    if straight_station is None:
        return None

    straight_cells = set(straight_station)
    station_cells = turn_station_cells | straight_cells
    if len(station_cells) != len(turn_station_cells) + 2:
        return None

    visible_cells = station_cells | set(start_stub) | set(end_stub)
    if len(visible_cells) != len(station_cells) + len(set(start_stub)) + len(set(end_stub)):
        return None

    occupied = route_set | set(start_stub[1:]) | set(end_stub[1:])
    allowed_edges: set[frozenset[tuple[int, int]]] = set()
    for seq in (route, start_stub, end_stub):
        for a, b in zip(seq, seq[1:]):
            allowed_edges.add(frozenset((a, b)))
    for cell in occupied:
        for direction in DIRS:
            nei = _add(cell, direction)
            if nei in occupied and frozenset((cell, nei)) not in allowed_edges:
                return None

    reserve = _inflate(route_set | set(start_stub) | set(end_stub), radius=1)

    return {
        "color": color,
        "route": route,
        "paint_cells": [cell for cell in route if cell not in station_cells and cell not in {route[0], route[-1]}],
        "turn_station_cells": turn_station_cells,
        "straight_station": straight_station,
        "start_stub": start_stub,
        "end_stub": end_stub,
        "reserve": reserve,
    }


def _feasible_offsets(shape: dict[str, object], blocked: np.ndarray) -> list[tuple[int, int]]:
    reserve = list(shape["reserve"])
    min_r = min(r for r, _c in reserve)
    max_r = max(r for r, _c in reserve)
    min_c = min(c for _r, c in reserve)
    max_c = max(c for _r, c in reserve)

    offsets: list[tuple[int, int]] = []
    for off_r in range(-min_r, blocked.shape[0] - max_r):
        for off_c in range(-min_c, blocked.shape[1] - max_c):
            ok = True
            for r, c in reserve:
                if blocked[r + off_r, c + off_c]:
                    ok = False
                    break
            if ok:
                offsets.append((off_r, off_c))
    return offsets


def _translate_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], offset: tuple[int, int]) -> list[tuple[int, int]]:
    off_r, off_c = offset
    return [(r + off_r, c + off_c) for r, c in cells]


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    colors = random.sample(COLOR_POOL, pair_count)

    shapes: list[dict[str, object]] = []
    for color in colors:
        for _ in range(MAX_SHAPE_ATTEMPTS):
            shape = _sample_shape(color)
            if shape is not None:
                shapes.append(shape)
                break
        else:
            return None

    shapes.sort(key=lambda item: len(item["reserve"]), reverse=True)
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []

    for shape in shapes:
        offsets = _feasible_offsets(shape, blocked)
        if not offsets:
            return None
        offset = random.choice(offsets)

        reserve_cells = _translate_cells(shape["reserve"], offset)
        for r, c in reserve_cells:
            blocked[r, c] = True

        placed.append(
            {
                "color": shape["color"],
                "paint_cells": _translate_cells(shape["paint_cells"], offset),
                "turn_station_cells": _translate_cells(shape["turn_station_cells"], offset),
                "straight_station": _translate_cells(list(shape["straight_station"]), offset),
                "start_stub": _translate_cells(shape["start_stub"], offset),
                "end_stub": _translate_cells(shape["end_stub"], offset),
            }
        )

    inp = np.full((h, w), BLACK, dtype=int)
    for item in placed:
        color = int(item["color"])

        for r, c in item["start_stub"]:
            inp[r, c] = GREY
        for r, c in item["end_stub"]:
            inp[r, c] = GREY

        start_color = item["start_stub"][-1]
        end_color = item["end_stub"][-1]
        inp[start_color] = color
        inp[end_color] = color

        for r, c in item["turn_station_cells"]:
            assert inp[r, c] == BLACK
            inp[r, c] = MAROON
        for r, c in item["straight_station"]:
            assert inp[r, c] == BLACK
            inp[r, c] = MAROON

    out = inp.copy()
    for item in placed:
        color = int(item["color"])
        for r, c in item["paint_cells"]:
            assert out[r, c] == BLACK
            out[r, c] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        scene = _build_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input")
