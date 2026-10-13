from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SURVIVOR_COLORS = (RED, YELLOW, PINK, ORANGE)
EXTERIOR_MARKER_COLORS = (GREY, GREEN)
ROOM_SLOT_NAMES = ("top_left", "mid_right", "bottom_left")
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _flood_external_non_walls(grid: np.ndarray) -> np.ndarray:
    """Return the boundary-reachable mask when blue is treated as the only wall."""
    h, w = map(int, grid.shape)
    external = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) != BLUE and not external[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) != BLUE and not external[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if not _in_bounds((h, w), r, c) or external[r, c] or int(grid[r, c]) == BLUE:
            continue
        external[r, c] = True
        for nr, nc in _neighbors4(r, c):
            stack.append((nr, nc))
    return external


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _route_cells_horizontal_then_vertical(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []
    if c0 != c1:
        dc = 1 if c1 > c0 else -1
        for c in range(c0 + dc, c1 + dc, dc):
            cell = (r0, c)
            if cell != end:
                cells.append(cell)
    if r0 != r1:
        dr = 1 if r1 > r0 else -1
        for r in range(r0 + dr, r1 + dr, dr):
            cell = (r, c1)
            if cell != end:
                cells.append(cell)
    return cells


def _route_cells_vertical_then_horizontal(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []
    if r0 != r1:
        dr = 1 if r1 > r0 else -1
        for r in range(r0 + dr, r1 + dr, dr):
            cell = (r, c0)
            if cell != end:
                cells.append(cell)
    if c0 != c1:
        dc = 1 if c1 > c0 else -1
        for c in range(c0 + dc, c1 + dc, dc):
            cell = (r1, c)
            if cell != end:
                cells.append(cell)
    return cells


def _route_is_clear(grid: np.ndarray, external: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(
        _in_bounds(grid.shape, r, c) and not external[r, c] and int(grid[r, c]) == BLACK
        for r, c in cells
    )


def _survivor_markers(grid: np.ndarray, external: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != BLUE) & (~external))
    return sorted((int(r), int(c)) for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """Flood from the boundary through non-wall cells, erase exterior markers, then connect surviving enclosed markers with constrained orthogonal paths.

    Input structure:
    - A black background contains one blue 4-connected orthogonal wall object.
    - The wall partitions the grid into boundary-connected exterior space and at least two enclosed regions.
    - Single-pixel colored markers lie in both exterior and enclosed regions.

    Rule:
    - Treat blue pixels as walls and flood from the grid boundary through all non-blue cells to identify the exterior region.
    - Recolor every exterior cell, including exterior markers, to green.
    - Keep only enclosed markers, sort them by row then column, and pair them in that order.
    - For each pair, draw a grey orthogonal route that prefers horizontal-then-vertical through enclosed black cells; if that route would cross a wall or marker, switch to vertical-then-horizontal instead.
    - Do not overwrite marker pixels, and leave the blue wall unchanged.

    Key insight:
    - The blue wall controls two coupled decisions at once: it decides which markers survive the exterior flood and it constrains which Manhattan routing order is valid inside each enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    external = _flood_external_non_walls(g)
    out = g.copy()
    out[external & (g != BLUE)] = GREEN

    survivors = _survivor_markers(g, external)
    assert survivors and len(survivors) % 2 == 0, "surviving enclosed markers must be a positive even count"

    for i in range(0, len(survivors), 2):
        start = survivors[i]
        end = survivors[i + 1]
        hv_cells = _route_cells_horizontal_then_vertical(start, end)
        if _route_is_clear(g, external, hv_cells):
            route = hv_cells
        else:
            vh_cells = _route_cells_vertical_then_horizontal(start, end)
            assert _route_is_clear(g, external, vh_cells), "expected fallback vertical-first route to be valid"
            route = vh_cells
        for r, c in route:
            if out[r, c] == BLACK:
                out[r, c] = GREY
    return out


def _draw_segment(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color


def _room_slots(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "top_left": (3 + random.randint(0, 1), 2 + random.randint(0, 1)),
        "mid_right": (h // 2 - 2 + random.randint(0, 1), w - 8 - random.randint(0, 1)),
        "bottom_left": (h - 7 + random.randint(0, 1), 3 + random.randint(0, 2)),
    }


def _build_rooms(h: int, w: int) -> list[dict]:
    room_count = random.choice((2, 3))
    chosen_slots = ("top_left", "bottom_left") if room_count == 2 else ROOM_SLOT_NAMES
    slots = _room_slots(h, w)
    rooms: list[dict] = []
    for name in chosen_slots:
        r0, c0 = slots[name]
        rh = random.randint(5, 6)
        rw = random.randint(6, 7)
        room = {"slot": name, "r0": r0, "c0": c0, "h": rh, "w": rw, "stub": False}
        rooms.append(room)
    return rooms


def _draw_room_wall(grid: np.ndarray, room: dict) -> None:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    _draw_segment(grid, (r0, c0), (r0, c1), BLUE)
    _draw_segment(grid, (r1, c0), (r1, c1), BLUE)
    _draw_segment(grid, (r0, c0), (r1, c0), BLUE)
    _draw_segment(grid, (r0, c1), (r1, c1), BLUE)

    if room["stub"]:
        stub_col = int(room["stub_col"])
        stub_end = int(room["stub_end"])
        _draw_segment(grid, (r0, stub_col), (stub_end, stub_col), BLUE)


def _paint_wall_layout(rooms: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    trunk_row = 1
    branch_cols = [int(room["c0"]) for room in rooms]
    trunk_c0 = max(1, min(branch_cols) - random.randint(0, 1))
    trunk_c1 = min(w - 2, max(branch_cols) + random.randint(0, 2))
    _draw_segment(grid, (trunk_row, trunk_c0), (trunk_row, trunk_c1), BLUE)

    for room in rooms:
        c0 = int(room["c0"])
        r0 = int(room["r0"])
        _draw_segment(grid, (trunk_row, c0), (r0, c0), BLUE)
        _draw_room_wall(grid, room)
    return grid


def _room_accessible_coords(grid: np.ndarray, room: dict) -> list[tuple[int, int]]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    coords = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) == BLACK:
                coords.append((r, c))
    return coords


def _candidate_pairs(grid: np.ndarray, external: np.ndarray, room: dict) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    coords = _room_accessible_coords(grid, room)
    out = {"hv": [], "vh": []}
    for i, start in enumerate(coords):
        for end in coords[i + 1 :]:
            if start[0] == end[0] or start[1] == end[1]:
                continue
            if abs(start[0] - end[0]) + abs(start[1] - end[1]) < 4:
                continue
            pair = tuple(sorted((start, end)))
            hv_clear = _route_is_clear(grid, external, _route_cells_horizontal_then_vertical(*pair))
            vh_clear = _route_is_clear(grid, external, _route_cells_vertical_then_horizontal(*pair))
            if hv_clear:
                out["hv"].append(pair)
            if (not hv_clear) and vh_clear:
                out["vh"].append(pair)
    return out


def _place_survivor_pairs(
    grid: np.ndarray, rooms: list[dict], external: np.ndarray
) -> tuple[list[tuple[int, int]], dict[str, str]] | None:
    occupied_count = random.choice((1, 2))
    occupied_rooms = random.sample(rooms, k=min(occupied_count, len(rooms)))

    if len(occupied_rooms) == 2:
        desired = ["hv", "vh"]
        random.shuffle(desired)
    else:
        desired = [random.choice(("hv", "vh"))]

    occupied_rooms = sorted(occupied_rooms, key=lambda room: int(room["r0"]))
    mode_by_slot: dict[str, str] = {}
    markers: list[tuple[int, int]] = []
    used_positions: set[tuple[int, int]] = set()

    for room, mode in zip(occupied_rooms, desired):
        candidates = _candidate_pairs(grid, external, room)[mode]
        if not candidates:
            return None
        start, end = random.choice(candidates)
        if start in used_positions or end in used_positions:
            return None
        used_positions.add(start)
        used_positions.add(end)
        markers.extend([start, end])
        mode_by_slot[str(room["slot"])] = mode
    return sorted(markers), mode_by_slot


def _place_exterior_markers(
    grid: np.ndarray, external: np.ndarray, count: int, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    candidates = []
    h, w = map(int, grid.shape)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not external[r, c] or int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if sum(1 for nr, nc in _neighbors4(r, c) if external[nr, nc] and int(grid[nr, nc]) == BLACK) >= 3:
                candidates.append((r, c))
    if len(candidates) < count:
        return None
    return random.sample(candidates, k=count)


def _semantic_checks(
    inp: np.ndarray,
    rooms: list[dict],
    occupied_modes: dict[str, str],
    survivor_coords: list[tuple[int, int]],
    exterior_coords: list[tuple[int, int]],
) -> bool:
    external = _flood_external_non_walls(inp)
    enclosed_mask = (~external) & (inp != BLUE)
    enclosed_regions = _components(enclosed_mask)
    if len(enclosed_regions) < 2:
        return False

    wall_components = _components(inp == BLUE)
    if len(wall_components) != 1:
        return False

    markers = np.argwhere((inp != BLACK) & (inp != BLUE))
    if len(markers) != len(survivor_coords) + len(exterior_coords):
        return False
    if len(markers) < 4 or len(markers) > 6:
        return False

    colors = [int(inp[r, c]) for r, c in map(tuple, markers)]
    if len(set(colors)) != len(colors):
        return False

    survivor_set = set(survivor_coords)
    exterior_set = set(exterior_coords)
    if not survivor_set or len(survivor_set) % 2 != 0 or not exterior_set:
        return False

    for comp in enclosed_regions:
        count = sum((int(r), int(c)) in survivor_set for r, c in comp)
        if count > 2:
            return False

    if survivor_coords != sorted(survivor_coords):
        return False

    pair_modes_found: list[str] = []
    used_route: set[tuple[int, int]] = set()
    wall_only = inp.copy()
    for r, c in exterior_coords:
        wall_only[r, c] = BLACK
    for r, c in survivor_coords:
        wall_only[r, c] = BLACK

    external_wall_only = _flood_external_non_walls(wall_only)
    for i in range(0, len(survivor_coords), 2):
        start = survivor_coords[i]
        end = survivor_coords[i + 1]
        hv = _route_cells_horizontal_then_vertical(start, end)
        vh = _route_cells_vertical_then_horizontal(start, end)
        hv_clear = _route_is_clear(wall_only, external_wall_only, hv)
        vh_clear = _route_is_clear(wall_only, external_wall_only, vh)
        if hv_clear:
            pair_modes_found.append("hv")
            route = hv
        elif vh_clear:
            pair_modes_found.append("vh")
            route = vh
        else:
            return False
        if used_route.intersection(route):
            return False
        used_route.update(route)
        if len(route) < 3:
            return False

    expected_modes = sorted(occupied_modes.values())
    if sorted(pair_modes_found) != expected_modes:
        return False

    out = main(inp)

    recolor_only = inp.copy()
    recolor_only[external & (inp != BLUE)] = GREEN
    if np.array_equal(out, recolor_only):
        return False

    if "vh" in occupied_modes.values():
        always_h = recolor_only.copy()
        for i in range(0, len(survivor_coords), 2):
            for r, c in _route_cells_horizontal_then_vertical(survivor_coords[i], survivor_coords[i + 1]):
                if always_h[r, c] == BLACK:
                    always_h[r, c] = GREY
        if np.array_equal(out, always_h):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(17, 19)
    w = random.randint(17, 20)
    rooms = _build_rooms(h, w)

    for room in rooms:
        if random.random() < 0.55:
            room["stub"] = True
            c0 = int(room["c0"])
            w_room = int(room["w"])
            r0 = int(room["r0"])
            h_room = int(room["h"])
            room["stub_col"] = c0 + random.randint(2, w_room - 3)
            room["stub_end"] = r0 + random.randint(2, h_room - 2)

    grid = _paint_wall_layout(rooms, h, w)
    external = _flood_external_non_walls(grid)
    enclosed_regions = _components((~external) & (grid != BLUE))
    if len(enclosed_regions) < 2:
        return None

    survivor_info = _place_survivor_pairs(grid, rooms, external)
    if survivor_info is None:
        return None
    survivor_coords, occupied_modes = survivor_info

    if len(survivor_coords) == 2:
        exterior_count = 2
    else:
        exterior_count = random.choice((1, 2))
    exterior_coords = _place_exterior_markers(grid, external, exterior_count, set(survivor_coords))
    if exterior_coords is None:
        return None

    if len(survivor_coords) == 2:
        survivor_colors = random.sample(list(SURVIVOR_COLORS), k=2)
    else:
        survivor_colors = random.sample(list(SURVIVOR_COLORS), k=4)
    exterior_colors = list(EXTERIOR_MARKER_COLORS[:exterior_count])

    inp = grid.copy()
    for (r, c), color in zip(survivor_coords, survivor_colors):
        inp[r, c] = color
    for (r, c), color in zip(exterior_coords, exterior_colors):
        inp[r, c] = color

    if not _semantic_checks(inp, rooms, occupied_modes, survivor_coords, exterior_coords):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-flood constrained-routing input")
