from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
ROOM = 9
GREEN = 3
ORANGE = 7
SOURCE_COLORS = (1, 2, 4, 6, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    return {
        UP: RIGHT,
        RIGHT: DOWN,
        DOWN: LEFT,
        LEFT: UP,
    }[direction]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    return {
        UP: LEFT,
        LEFT: DOWN,
        DOWN: RIGHT,
        RIGHT: UP,
    }[direction]


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for pts in _components_of_color(grid, ROOM):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = 2 * h + 2 * w - 4
        if len(pts) != expected:
            continue
        border = set(pts)
        ok = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != ((rr, cc) in border):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        source_marker: tuple[int, int, int] | None = None
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                value = int(grid[rr, cc])
                if value == BG:
                    continue
                if source_marker is not None:
                    ok = False
                    break
                source_marker = (rr, cc, value)
            if not ok:
                break
        if not ok:
            continue
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "source_color": None if source_marker is None else int(source_marker[2]),
            }
        )
    return rooms


def _junction_mode(grid: np.ndarray, junction: tuple[int, int]) -> str:
    h, w = grid.shape
    has_green = False
    has_orange = False
    jr, jc = junction
    for nr, nc in _neighbors8(jr, jc):
        if 0 <= nr < h and 0 <= nc < w:
            value = int(grid[nr, nc])
            if value == GREEN:
                has_green = True
            elif value == ORANGE:
                has_orange = True
    if has_green and not has_orange:
        return "green"
    if has_orange and not has_green:
        return "orange"
    return "none"


def _turn_label(incoming: tuple[int, int], outgoing: tuple[int, int]) -> str | None:
    if outgoing == incoming:
        return "straight"
    if outgoing == _turn_right(incoming):
        return "right"
    if outgoing == _turn_left(incoming):
        return "left"
    return None


def _choose_branch(
    mode: str,
    prev_cell: tuple[int, int],
    current_cell: tuple[int, int],
    options: list[tuple[int, int]],
) -> tuple[int, int]:
    incoming = (current_cell[0] - prev_cell[0], current_cell[1] - prev_cell[1])
    labeled: dict[str, tuple[int, int]] = {}
    for nr, nc in sorted(options):
        outgoing = (nr - current_cell[0], nc - current_cell[1])
        label = _turn_label(incoming, outgoing)
        if label is not None:
            labeled[label] = (nr, nc)
    priorities = {
        "green": ("right", "straight", "left"),
        "orange": ("left", "straight", "right"),
        "none": ("straight", "right", "left"),
    }[mode]
    for label in priorities:
        if label in labeled:
            return labeled[label]
    return sorted(options)[0]


def _trace_destination(
    grid: np.ndarray,
    start_endpoint: tuple[int, int],
    source_room_id: int,
    path_neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    endpoint_to_rooms: dict[tuple[int, int], set[int]],
    rooms: list[dict[str, object]],
) -> int:
    prev: tuple[int, int] | None = None
    current = start_endpoint
    seen: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while True:
        state = (prev, current)
        assert state not in seen, "path tracing entered a loop"
        seen.add(state)

        if prev is not None and current in endpoint_to_rooms:
            candidates = [
                room_id
                for room_id in sorted(endpoint_to_rooms[current])
                if room_id != source_room_id and rooms[room_id]["source_color"] is None
            ]
            if candidates:
                return candidates[0]

        neighbors = path_neighbors[current]
        if prev is None:
            options = list(neighbors)
        else:
            options = [cell for cell in neighbors if cell != prev]
        assert options, "path terminated before reaching a destination room"

        if prev is None or len(neighbors) <= 2:
            assert len(options) == 1, "non-junction path cell must have a unique continuation"
            chosen = options[0]
        else:
            chosen = _choose_branch(_junction_mode(grid, current), prev, current, options)
        prev, current = current, chosen


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rooms = _find_rooms(g)
    border_to_room: dict[tuple[int, int], int] = {}
    for room_id, room in enumerate(rooms):
        for cell in room["border"]:
            border_to_room[cell] = room_id

    path_cells = {(r, c) for r in range(h) for c in range(w) if int(g[r, c]) == PATH}
    path_neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in path_cells:
        r, c = cell
        path_neighbors[cell] = [nbr for nbr in _neighbors4(r, c) if nbr in path_cells]

    endpoints = {cell for cell, nbrs in path_neighbors.items() if len(nbrs) == 1}
    endpoint_to_rooms: dict[tuple[int, int], set[int]] = {}
    for endpoint in endpoints:
        room_ids = {border_to_room[nbr] for nbr in _neighbors4(*endpoint) if nbr in border_to_room}
        if room_ids:
            endpoint_to_rooms[endpoint] = room_ids

    for room_id, room in enumerate(rooms):
        source_color = room["source_color"]
        if source_color is None:
            continue
        start_endpoints = [
            endpoint
            for endpoint, room_ids in endpoint_to_rooms.items()
            if room_id in room_ids
        ]
        assert len(start_endpoints) == 1, "each source room must touch exactly one path endpoint"
        dest_room_id = _trace_destination(g, start_endpoints[0], room_id, path_neighbors, endpoint_to_rooms, rooms)
        r0, c0, r1, c1 = rooms[dest_room_id]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(source_color)
    return out


def _room_top_left_from_endpoint(
    endpoint: tuple[int, int],
    side_dir: tuple[int, int],
    height: int,
    width: int,
) -> tuple[int, int]:
    er, ec = endpoint
    if side_dir == UP:
        return er + 1, ec - width // 2
    if side_dir == DOWN:
        return er - height, ec - width // 2
    if side_dir == LEFT:
        return er - height // 2, ec + 1
    return er - height // 2, ec - width


def _room_cells(r0: int, c0: int, height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    full = {(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)}
    border = {
        (r, c)
        for r in range(r0, r0 + height)
        for c in range(c0, c0 + width)
        if r in (r0, r0 + height - 1) or c in (c0, c0 + width - 1)
    }
    interior = [
        (r, c)
        for r in range(r0 + 1, r0 + height - 1)
        for c in range(c0 + 1, c0 + width - 1)
    ]
    return full, border, interior


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in cells:
        expanded.update(_neighbors8(r, c))
    return expanded


def _available_labels(mode: str) -> tuple[str, ...]:
    choices = {
        "green": (
            ("left", "straight", "right"),
            ("straight", "right"),
            ("left", "right"),
        ),
        "orange": (
            ("left", "straight", "right"),
            ("straight", "left"),
            ("left", "right"),
        ),
        "none": (
            ("left", "straight", "right"),
            ("straight", "left"),
            ("straight", "right"),
        ),
    }[mode]
    return random.choice(choices)


def _resolve_selected_label(mode: str, available: tuple[str, ...]) -> str:
    order = {
        "green": ("right", "straight", "left"),
        "orange": ("left", "straight", "right"),
        "none": ("straight", "right", "left"),
    }[mode]
    present = set(available)
    for label in order:
        if label in present:
            return label
    raise AssertionError("available branch labels must not be empty")


def _module_branch_dirs(incoming: tuple[int, int]) -> dict[str, tuple[int, int]]:
    return {
        "straight": incoming,
        "left": _turn_left(incoming),
        "right": _turn_right(incoming),
    }


def _random_room_size() -> tuple[int, int]:
    return random.randint(3, 6), random.randint(3, 6)


def _build_module(mode: str, source_color: int) -> dict[str, object]:
    for _ in range(200):
        incoming = random.choice(DIRS)
        branch_dirs = _module_branch_dirs(incoming)
        available = _available_labels(mode)
        selected = _resolve_selected_label(mode, available)
        source_len = random.randint(2, 5)
        branch_lens = {label: random.randint(2, 5) for label in available}
        source_size = _random_room_size()
        branch_sizes = {label: _random_room_size() for label in available}

        path_cells = {(0, 0)}
        source_endpoint = (-incoming[0] * source_len, -incoming[1] * source_len)
        for step in range(1, source_len + 1):
            path_cells.add((-incoming[0] * step, -incoming[1] * step))

        branch_endpoints: dict[str, tuple[int, int]] = {}
        for label in available:
            dr, dc = branch_dirs[label]
            for step in range(1, branch_lens[label] + 1):
                path_cells.add((dr * step, dc * step))
            branch_endpoints[label] = (dr * branch_lens[label], dc * branch_lens[label])

        room_boxes: list[set[tuple[int, int]]] = []
        expanded_room_boxes: list[set[tuple[int, int]]] = []
        room_defs: list[dict[str, object]] = []

        src_h, src_w = source_size
        src_r0, src_c0 = _room_top_left_from_endpoint(source_endpoint, incoming, src_h, src_w)
        src_full, src_border, src_interior = _room_cells(src_r0, src_c0, src_h, src_w)
        if not src_interior:
            continue
        source_marker = random.choice(src_interior)
        room_boxes.append(src_full)
        expanded_room_boxes.append(_expand8(src_full))
        room_defs.append(
            {
                "full": src_full,
                "border": src_border,
                "interior": src_interior,
                "marker": source_marker,
                "fill_color": None,
            }
        )

        ok = True
        for label in available:
            room_h, room_w = branch_sizes[label]
            endpoint = branch_endpoints[label]
            side_dir = _opposite(branch_dirs[label])
            room_r0, room_c0 = _room_top_left_from_endpoint(endpoint, side_dir, room_h, room_w)
            full, border, interior = _room_cells(room_r0, room_c0, room_h, room_w)
            if not interior:
                ok = False
                break
            if any(full & other for other in room_boxes):
                ok = False
                break
            if any(full & other for other in expanded_room_boxes):
                ok = False
                break
            room_boxes.append(full)
            expanded_room_boxes.append(_expand8(full))
            room_defs.append(
                {
                    "full": full,
                    "border": border,
                    "interior": interior,
                    "marker": None,
                    "fill_color": source_color if label == selected else None,
                }
            )
        if not ok:
            continue

        room_reserved = set().union(*room_boxes)
        if path_cells & room_reserved:
            continue
        if any(
            sum((nbr in path_cells) for cell in room["border"] for nbr in _neighbors4(*cell)) != 1
            for room in room_defs
        ):
            continue

        light_cell: tuple[int, int] | None = None
        light_color = None
        if mode != "none":
            diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            random.shuffle(diagonals)
            light_color = GREEN if mode == "green" else ORANGE
            for dr, dc in diagonals:
                candidate = (dr, dc)
                if candidate not in path_cells and candidate not in room_reserved:
                    light_cell = candidate
                    break
            if light_cell is None:
                continue

        degrees = {
            cell: sum((nbr in path_cells) for nbr in _neighbors4(*cell))
            for cell in path_cells
        }
        if degrees[(0, 0)] != 1 + len(available):
            continue
        if degrees[source_endpoint] != 1:
            continue
        if any(degrees[branch_endpoints[label]] != 1 for label in available):
            continue

        input_cells: dict[tuple[int, int], int] = {}
        output_cells: dict[tuple[int, int], int] = {}
        reserved = set(path_cells)

        for cell in path_cells:
            input_cells[cell] = PATH
            output_cells[cell] = PATH
        if light_cell is not None and light_color is not None:
            input_cells[light_cell] = light_color
            output_cells[light_cell] = light_color
            reserved.add(light_cell)

        for room in room_defs:
            reserved |= room["full"]
            for cell in room["border"]:
                input_cells[cell] = ROOM
                output_cells[cell] = ROOM
            marker = room["marker"]
            if marker is not None:
                input_cells[marker] = source_color
                output_cells[marker] = source_color
            fill_color = room["fill_color"]
            if fill_color is not None:
                for cell in room["interior"]:
                    output_cells[cell] = int(fill_color)

        return {
            "input_cells": input_cells,
            "output_cells": output_cells,
            "reserved": reserved,
        }
    raise AssertionError("failed to build a valid module")


def _module_bbox(module: dict[str, object]) -> tuple[int, int, int, int]:
    return _bbox(module["reserved"])


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _paint_cells(grid: np.ndarray, cells: dict[tuple[int, int], int], dr: int, dc: int) -> None:
    for (r, c), color in cells.items():
        grid[r + dr, c + dc] = int(color)


def _place_modules(
    height: int,
    width: int,
    modules: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()

    for module in sorted(modules, key=lambda mod: len(mod["reserved"]), reverse=True):
        r0, c0, r1, c1 = _module_bbox(module)
        min_dr = 1 - r0
        max_dr = height - 2 - r1
        min_dc = 1 - c0
        max_dc = width - 2 - c1
        if min_dr > max_dr or min_dc > max_dc:
            return None

        translations = [
            (dr, dc)
            for dr in range(min_dr, max_dr + 1)
            for dc in range(min_dc, max_dc + 1)
        ]
        random.shuffle(translations)

        placed = False
        for dr, dc in translations:
            translated = _translate_cells(module["reserved"], dr, dc)
            if any(cell in forbidden for cell in translated):
                continue
            _paint_cells(inp, module["input_cells"], dr, dc)
            _paint_cells(out, module["output_cells"], dr, dc)
            for r, c in translated:
                for nr, nc in _neighbors8(r, c) + [(r, c)]:
                    if 0 <= nr < height and 0 <= nc < width:
                        forbidden.add((nr, nc))
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        height = random.randint(24, 36)
        width = random.randint(24, 36)
        num_modules = random.choice((3, 4))
        modes = ["green", "orange", "none"]
        if num_modules == 4:
            modes.append(random.choice(modes))
        random.shuffle(modes)
        colors = random.sample(SOURCE_COLORS, num_modules)
        modules = [_build_module(mode, color) for mode, color in zip(modes, colors, strict=False)]
        placed = _place_modules(height, width, modules)
        if placed is None:
            continue
        inp, out = placed
        if np.array_equal(main(inp), out):
            return inp, out
    raise AssertionError("generator failed to assemble a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
