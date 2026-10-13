from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Direction = str


BG_COLOR = 8  # azure
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
DIRS: tuple[Direction, ...] = ("tl", "tr", "bl", "br")
MISSING_BY_DIR: dict[Direction, Coord] = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}
DIR_BY_MISSING = {value: key for key, value in MISSING_BY_DIR.items()}

# Generator configuration.
GRID_H_RANGE = (10, 17)
GRID_W_RANGE = (10, 17)
OBJECT_COUNT_RANGE = (2, 4)
SOURCE_MARGIN = 1
MIN_PATH_MANHATTAN = 5
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _occupied_locals(direction: Direction) -> list[Coord]:
    missing = MISSING_BY_DIR[direction]
    return [(r, c) for r in range(2) for c in range(2) if (r, c) != missing]


def _corner_local(direction: Direction) -> Coord:
    mr, mc = MISSING_BY_DIR[direction]
    return 1 - mr, 1 - mc


def _destination_top_left(direction: Direction, h: int, w: int) -> Coord:
    if direction == "tl":
        return 0, 0
    if direction == "tr":
        return 0, w - 2
    if direction == "bl":
        return h - 2, 0
    return h - 2, w - 2


def _target_corner(direction: Direction, h: int, w: int) -> Coord:
    if direction == "tl":
        return 0, 0
    if direction == "tr":
        return 0, w - 1
    if direction == "bl":
        return h - 1, 0
    return h - 1, w - 1


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _path_cells(start: Coord, end: Coord, order: str) -> list[Coord]:
    sr, sc = start
    tr, tc = end
    if order == "hv":
        mid = (sr, tc)
    else:
        mid = (tr, sc)
    first = _segment_cells(start, mid)
    second = _segment_cells(mid, end)
    return first + second[1:]


def _object_from_spec(spec: dict, h: int, w: int) -> dict:
    r0, c0 = map(int, spec["top_left"])
    direction = str(spec["direction"])
    corner_local = _corner_local(direction)
    local_colors: dict[Coord, int] = {}
    cells: set[Coord] = set()
    for local in _occupied_locals(direction):
        color = int(spec["corner_color"] if local == corner_local else spec["arm_color"])
        local_colors[local] = color
        cells.add((int(r0 + local[0]), int(c0 + local[1])))

    dr0, dc0 = _destination_top_left(direction, h, w)
    dest_cells: set[Coord] = {
        (int(dr0 + local[0]), int(dc0 + local[1])) for local in local_colors
    }
    corner_abs = (int(r0 + corner_local[0]), int(c0 + corner_local[1]))
    return {
        "direction": direction,
        "bbox": (int(r0), int(c0), int(r0 + 1), int(c0 + 1)),
        "cells": cells,
        "corner_abs": corner_abs,
        "corner_color": int(spec["corner_color"]),
        "local_colors": local_colors,
        "dest_top_left": (int(dr0), int(dc0)),
        "dest_cells": dest_cells,
        "target_corner": _target_corner(direction, h, w),
    }


def _route_is_clear(route: list[Coord], own_cells: set[Coord], blocked: set[Coord]) -> bool:
    for cell in route:
        if cell in own_cells:
            continue
        if cell in blocked:
            return False
    return True


def _choose_routes(objects: list[dict]) -> tuple[list[str], list[list[Coord]]] | None:
    occupied = set().union(*(obj["cells"] for obj in objects)) if objects else set()
    orders: list[str] = []
    routes: list[list[Coord]] = []
    for obj in objects:
        own = set(obj["cells"])
        blocked = occupied - own
        start = tuple(obj["corner_abs"])
        end = tuple(obj["target_corner"])
        hv = _path_cells(start, end, "hv")
        vh = _path_cells(start, end, "vh")
        hv_clear = _route_is_clear(hv, own, blocked)
        vh_clear = _route_is_clear(vh, own, blocked)
        if not hv_clear and not vh_clear:
            return None
        if hv_clear:
            orders.append("hv")
            routes.append(hv)
        else:
            orders.append("vh")
            routes.append(vh)
    return orders, routes


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    objs: list[dict] = []
    for comp in _components(g != bg):
        assert len(comp) == 3, "every foreground component must be a 3-cell L"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 1 and c1 - c0 == 1, "every object must fit a 2x2 bbox"

        cells = {(int(r), int(c)) for r, c in comp}
        local_cells = {(int(r - r0), int(c - c0)) for r, c in comp}
        missing = next((p for p in ((0, 0), (0, 1), (1, 0), (1, 1)) if p not in local_cells), None)
        assert missing is not None, "expected exactly one missing corner in each 2x2 bbox"
        direction = DIR_BY_MISSING[missing]

        degree2: list[Coord] = []
        for r, c in cells:
            degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, h, w))
            if degree == 2:
                degree2.append((int(r), int(c)))
        assert len(degree2) == 1, "expected a unique corner pixel"
        corner_abs = degree2[0]

        local_colors: dict[Coord, int] = {}
        corner_color = None
        for r, c in comp:
            local = (int(r - r0), int(c - c0))
            color = int(g[int(r), int(c)])
            local_colors[local] = color
            if (int(r), int(c)) == corner_abs:
                corner_color = color
        assert corner_color is not None

        dr0, dc0 = _destination_top_left(direction, h, w)
        dest_cells = {(int(dr0 + lr), int(dc0 + lc)) for lr, lc in local_colors}
        objs.append(
            {
                "direction": direction,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cells": cells,
                "corner_abs": corner_abs,
                "corner_color": int(corner_color),
                "local_colors": local_colors,
                "dest_top_left": (int(dr0), int(dc0)),
                "dest_cells": dest_cells,
                "target_corner": _target_corner(direction, h, w),
            }
        )

    objs.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["direction"]))
    assert objs, "expected at least one L object"
    return bg, objs


def _render_output(shape: tuple[int, int], bg: int, objects: list[dict], routes: list[list[Coord]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj, route in zip(objects, routes):
        for r, c in route:
            out[int(r), int(c)] = int(obj["corner_color"])
    for obj in objects:
        dr0, dc0 = map(int, obj["dest_top_left"])
        for (lr, lc), color in obj["local_colors"].items():
            out[int(dr0 + lr), int(dc0 + lc)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 3-cell L object to the matching outer corner and draw a corner-colored path.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _parse_objects(g)
    chosen = _choose_routes(objects)
    assert chosen is not None, "every object must have a valid L-shaped route to its corner"
    _orders, routes = chosen
    return _render_output(g.shape, bg, objects, routes)


def _boxes_touch(a: Coord, b: Coord) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 1, ac0 + 1
    br1, bc1 = br0 + 1, bc0 + 1
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _candidate_positions(h: int, w: int) -> list[Coord]:
    return [
        (int(r), int(c))
        for r in range(SOURCE_MARGIN, h - 1 - SOURCE_MARGIN)
        for c in range(SOURCE_MARGIN, w - 1 - SOURCE_MARGIN)
    ]


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for spec in specs:
        obj = _object_from_spec(spec, h, w)
        for (lr, lc), color in obj["local_colors"].items():
            r0, c0 = map(int, spec["top_left"])
            grid[int(r0 + lr), int(c0 + lc)] = int(color)
    return grid


def _layout_is_clear(objects: list[dict], routes: list[list[Coord]]) -> bool:
    for obj in objects:
        sr, sc = obj["corner_abs"]
        tr, tc = obj["target_corner"]
        if abs(int(sr) - int(tr)) + abs(int(sc) - int(tc)) < MIN_PATH_MANHATTAN:
            return False

    path_sets = [set(route) for route in routes]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i]["dest_cells"] & objects[j]["dest_cells"]:
                return False
            if path_sets[i] & path_sets[j]:
                return False
            if path_sets[i] & objects[j]["dest_cells"]:
                return False
            if path_sets[j] & objects[i]["dest_cells"]:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    directions = random.sample(list(DIRS), n_objects)
    color_pool = random.sample(PALETTE, 2 * n_objects)

    specs: list[dict] = []
    for i, direction in enumerate(directions):
        specs.append(
            {
                "direction": direction,
                "arm_color": int(color_pool[2 * i]),
                "corner_color": int(color_pool[2 * i + 1]),
                "top_left": None,
            }
        )

    positions = _candidate_positions(h, w)
    if len(positions) < n_objects:
        return None

    forced_idx = random.randrange(n_objects)
    blocker_choices = [i for i in range(n_objects) if i != forced_idx]
    blocker_idx = random.choice(blocker_choices)

    random.shuffle(positions)
    placed: dict[int, Coord] = {}
    guard_v: set[Coord] | None = None
    guard_h: set[Coord] | None = None

    # Place one object so it is forced onto the vertical-first branch, and one
    # blocker whose horizontal-first route stays clear.
    found_pair = False
    for pos_a in positions:
        spec_a = dict(specs[forced_idx])
        spec_a["top_left"] = pos_a
        obj_a = _object_from_spec(spec_a, h, w)
        a_h = set(_path_cells(obj_a["corner_abs"], obj_a["target_corner"], "hv")) - set(obj_a["cells"])
        a_v = set(_path_cells(obj_a["corner_abs"], obj_a["target_corner"], "vh")) - set(obj_a["cells"])
        if abs(obj_a["corner_abs"][0] - obj_a["target_corner"][0]) + abs(obj_a["corner_abs"][1] - obj_a["target_corner"][1]) < MIN_PATH_MANHATTAN:
            continue

        blocker_positions = positions[:]
        random.shuffle(blocker_positions)
        for pos_b in blocker_positions:
            if pos_b == pos_a or _boxes_touch(pos_a, pos_b):
                continue
            spec_b = dict(specs[blocker_idx])
            spec_b["top_left"] = pos_b
            obj_b = _object_from_spec(spec_b, h, w)
            if not (set(obj_b["cells"]) & a_h):
                continue
            if set(obj_b["cells"]) & a_v:
                continue

            chosen = _choose_routes([obj_a, obj_b])
            if chosen is None:
                continue
            orders_ab, routes_ab = chosen
            if orders_ab != ["vh", "hv"]:
                continue
            route_a = set(routes_ab[0])
            route_b = set(routes_ab[1])
            if route_a & route_b:
                continue
            if route_a & obj_b["dest_cells"]:
                continue
            if route_b & obj_a["dest_cells"]:
                continue

            placed[forced_idx] = pos_a
            placed[blocker_idx] = pos_b
            guard_v = route_a
            guard_h = route_b
            found_pair = True
            break
        if found_pair:
            break
    if not found_pair or guard_v is None or guard_h is None:
        return None

    remaining = [i for i in range(n_objects) if i not in placed]
    random.shuffle(remaining)
    for idx in remaining:
        candidate_positions = positions[:]
        random.shuffle(candidate_positions)
        selected = None
        for pos in candidate_positions:
            if any(_boxes_touch(pos, other) for other in placed.values()):
                continue
            spec = dict(specs[idx])
            spec["top_left"] = pos
            obj = _object_from_spec(spec, h, w)
            if set(obj["cells"]) & guard_v:
                continue
            if set(obj["cells"]) & guard_h:
                continue
            selected = pos
            break
        if selected is None:
            return None
        placed[idx] = selected

    for idx, pos in placed.items():
        specs[idx]["top_left"] = pos

    objects = [_object_from_spec(spec, h, w) for spec in specs]
    chosen = _choose_routes(objects)
    if chosen is None:
        return None
    orders, routes = chosen
    if orders[forced_idx] != "vh":
        return None
    if orders[blocker_idx] != "hv":
        return None
    if "vh" not in orders or "hv" not in orders:
        return None
    if not _layout_is_clear(objects, routes):
        return None

    inp = _render_input((h, w), specs)
    expected = _render_output((h, w), BG_COLOR, objects, routes)

    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
