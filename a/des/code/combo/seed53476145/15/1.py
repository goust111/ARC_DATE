from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(comp) != 2 * (h + w) - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        for comp in _components_of_mask(grid == color):
            if _is_rect_frame(grid, color, comp):
                frames.append({"color": int(color), "bbox": _bbox(comp)})
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one rectangular wall frame"
    return frames


def _wire_components_in_frame(grid: np.ndarray, frame: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    wall = int(frame["color"])
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    wire_colors = sorted(set(map(int, interior.flat)) - {int(bg), wall})
    wires: list[dict] = []
    for color in wire_colors:
        for comp in _components_of_mask(interior == color):
            abs_comp = [(r + r0 + 1, c + c0 + 1) for r, c in comp]
            wires.append({"color": int(color), "coords": abs_comp})
    wires.sort(key=lambda wire: (wire["color"], wire["coords"]))
    assert wires, "each frame must contain at least one wire"
    return wires


def _path_endpoints(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    endpoints: list[tuple[int, int]] = []
    for cell in coords:
        degree = 0
        for dr, dc in DIRS4:
            if (cell[0] + dr, cell[1] + dc) in coord_set:
                degree += 1
        assert degree in (1, 2), "wire must be a simple one-pixel-wide path"
        if degree == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "wire must have exactly two endpoints"
    endpoints.sort()
    return endpoints


def _endpoint_port(frame_bbox: tuple[int, int, int, int], endpoint: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    r, c = endpoint
    sides: list[str] = []
    if r == r0 + 1:
        sides.append("top")
    if r == r1 - 1:
        sides.append("bottom")
    if c == c0 + 1:
        sides.append("left")
    if c == c1 - 1:
        sides.append("right")
    assert len(sides) == 1, "endpoint must touch exactly one wall side"
    side = sides[0]
    if side == "top":
        return side, (r0 - 1, c)
    if side == "bottom":
        return side, (r1 + 1, c)
    if side == "left":
        return side, (r, c0 - 1)
    return side, (r, c1 + 1)


def _perimeter_cycle(frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    cycle: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        cycle.append((r0 - 1, c))
    for r in range(r0, r1 + 2):
        cycle.append((r, c1 + 1))
    for c in range(c1, c0 - 2, -1):
        cycle.append((r1 + 1, c))
    for r in range(r1, r0 - 1, -1):
        cycle.append((r, c0 - 1))
    return cycle


def _route_options(frame_bbox: tuple[int, int, int, int], a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], int, int]:
    cycle = _perimeter_cycle(frame_bbox)
    index = {cell: i for i, cell in enumerate(cycle)}
    ia = index[a]
    ib = index[b]
    n = len(cycle)
    cw = (ib - ia) % n
    ccw = (ia - ib) % n
    if cw <= ccw:
        route = [cycle[(ia + k) % n] for k in range(cw + 1)]
    else:
        route = [cycle[(ia - k) % n] for k in range(ccw + 1)]
    return route, cw, ccw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each internal wire onto the outside of its grey rectangle by drawing
    the same-colored shortest arc along the rectangle's outer perimeter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        for wire in _wire_components_in_frame(g, frame, bg):
            endpoints = _path_endpoints(wire["coords"])
            _side0, outer0 = _endpoint_port(frame["bbox"], endpoints[0])
            _side1, outer1 = _endpoint_port(frame["bbox"], endpoints[1])
            route, _cw, _ccw = _route_options(frame["bbox"], outer0, outer1)
            for r, c in route:
                out[r, c] = int(wire["color"])

    return out


GEN_BG = 0
GEN_WALL = 5
GEN_WIRE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_HEIGHT_RANGE = (22, 32)
GEN_WIDTH_RANGE = (22, 32)
GEN_RECT_COUNT_CHOICES = [1, 2, 3]
GEN_RECT_COUNT_WEIGHTS = [0.3, 0.45, 0.25]
GEN_RECT_HEIGHT_RANGE = (7, 12)
GEN_RECT_WIDTH_RANGE = (7, 12)
GEN_PLACEMENT_ATTEMPTS = 160
GEN_ROUTE_ATTEMPTS = 80
GEN_RETRIES = 400


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _max_wires_for_rect(frame_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_bbox
    side = min(r1 - r0 + 1, c1 - c0 + 1)
    return min(3, 1 + int(side >= 9) + int(side >= 11))


def _candidate_ports(frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    ports: list[dict] = []
    for c in range(c0 + 2, c1 - 1):
        ports.append({"side": "top", "cell": (r0 + 1, c), "outer": (r0 - 1, c)})
        ports.append({"side": "bottom", "cell": (r1 - 1, c), "outer": (r1 + 1, c)})
    for r in range(r0 + 2, r1 - 1):
        ports.append({"side": "left", "cell": (r, c0 + 1), "outer": (r, c0 - 1)})
        ports.append({"side": "right", "cell": (r, c1 - 1), "outer": (r, c1 + 1)})
    return ports


def _pair_kind(side0: str, side1: str) -> str:
    if side0 == side1:
        return "same"
    if {side0, side1} in ({"top", "bottom"}, {"left", "right"}):
        return "opposite"
    return "adjacent"


def _random_bfs_segment(
    frame_bbox: tuple[int, int, int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = frame_bbox
    q = deque([start])
    parent = {start: None}

    while q:
        cell = q.popleft()
        dirs = DIRS4[:]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cell[0] + dr, cell[1] + dc
            nxt = (nr, nc)
            if not (r0 + 1 <= nr <= r1 - 1 and c0 + 1 <= nc <= c1 - 1):
                continue
            if nxt != start and nxt != goal:
                if nr in (r0 + 1, r1 - 1) or nc in (c0 + 1, c1 - 1):
                    continue
            if nxt in blocked and nxt != goal:
                continue
            if nxt in parent:
                continue
            parent[nxt] = cell
            if nxt == goal:
                path = [goal]
                cur = goal
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _core_cells(frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    return [
        (r, c)
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
    ]


def _build_wire_path(
    frame_bbox: tuple[int, int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    core = [cell for cell in _core_cells(frame_bbox) if cell not in occupied and cell not in {start, end}]
    for _ in range(GEN_ROUTE_ATTEMPTS):
        waypoint_count = random.choices([0, 1, 2], weights=[3, 4, 2])[0]
        waypoint_count = min(waypoint_count, len(core))
        waypoints = random.sample(core, waypoint_count)
        targets = waypoints + [end]

        current = start
        used = {start}
        path = [start]
        ok = True
        for i, target in enumerate(targets):
            reserved = set(targets[i + 1 :])
            blocked = set(occupied) | (used - {current}) | reserved
            segment = _random_bfs_segment(frame_bbox, current, target, blocked)
            if segment is None:
                ok = False
                break
            path.extend(segment[1:])
            used.update(segment[1:])
            current = target
        if ok and len(set(path)) == len(path):
            return path
    return None


def _plan_wires_for_rect(frame_bbox: tuple[int, int, int, int], colors: list[int]) -> list[dict] | None:
    ports = _candidate_ports(frame_bbox)
    used_ports: set[tuple[int, int]] = set()
    used_external: set[tuple[int, int]] = set()
    used_internal: set[tuple[int, int]] = set()
    wires: list[dict] = []

    for color in colors:
        free_ports = [port for port in ports if port["cell"] not in used_ports]
        by_kind: dict[str, list[tuple[dict, dict, list[tuple[int, int]]]]] = {"same": [], "adjacent": [], "opposite": []}
        for p0, p1 in combinations(free_ports, 2):
            route, cw, ccw = _route_options(frame_bbox, p0["outer"], p1["outer"])
            if cw == ccw:
                continue
            if any(cell in used_external for cell in route):
                continue
            kind = _pair_kind(p0["side"], p1["side"])
            by_kind[kind].append((p0, p1, route))

        kinds = [kind for kind, items in by_kind.items() if items]
        if not kinds:
            return None

        kind = random.choice(kinds)
        pair_options = by_kind[kind][:]
        random.shuffle(pair_options)
        chosen = None
        for p0, p1, route in pair_options:
            path = _build_wire_path(frame_bbox, p0["cell"], p1["cell"], used_internal)
            if path is None:
                continue
            chosen = {
                "color": int(color),
                "path": path,
                "route": route,
                "ports": (p0["cell"], p1["cell"]),
            }
            break
        if chosen is None:
            return None

        used_ports.update(chosen["ports"])
        used_external.update(chosen["route"])
        used_internal.update(chosen["path"])
        wires.append(chosen)

    return wires


def _choose_rectangles(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    rect_count = random.choices(GEN_RECT_COUNT_CHOICES, weights=GEN_RECT_COUNT_WEIGHTS)[0]
    frames: list[tuple[int, int, int, int]] = []
    outer_boxes: list[tuple[int, int, int, int]] = []

    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        if len(frames) >= rect_count:
            break
        rh = random.randint(*GEN_RECT_HEIGHT_RANGE)
        rw = random.randint(*GEN_RECT_WIDTH_RANGE)
        if rh > h - 2 or rw > w - 2:
            continue
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        outer = (r0 - 1, c0 - 1, r0 + rh, c0 + rw)
        if all(_boxes_far_enough(outer, prev, gap=1) for prev in outer_boxes):
            frames.append(bbox)
            outer_boxes.append(outer)

    if len(frames) != rect_count:
        return None
    frames.sort()
    return frames


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    frames = _choose_rectangles(h, w)
    if frames is None:
        return None

    wire_counts: list[int] = []
    colors_left = len(GEN_WIRE_COLORS)
    for i, frame_bbox in enumerate(frames):
        max_wires = min(_max_wires_for_rect(frame_bbox), colors_left - (len(frames) - i - 1))
        if max_wires < 1:
            return None
        wire_counts.append(random.randint(1, max_wires))
        colors_left -= wire_counts[-1]

    total_wires = sum(wire_counts)
    wire_palette = random.sample(GEN_WIRE_COLORS, total_wires)

    grid = np.full((h, w), GEN_BG, dtype=int)
    color_cursor = 0

    for frame_bbox, wire_count in zip(frames, wire_counts):
        r0, c0, r1, c1 = frame_bbox
        grid[r0, c0 : c1 + 1] = GEN_WALL
        grid[r1, c0 : c1 + 1] = GEN_WALL
        grid[r0 : r1 + 1, c0] = GEN_WALL
        grid[r0 : r1 + 1, c1] = GEN_WALL

        wires = _plan_wires_for_rect(frame_bbox, wire_palette[color_cursor : color_cursor + wire_count])
        if wires is None:
            return None
        color_cursor += wire_count
        for wire in wires:
            for r, c in wire["path"]:
                if grid[r, c] != GEN_BG:
                    return None
                grid[r, c] = int(wire["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
