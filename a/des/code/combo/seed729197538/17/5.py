from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
WHITE = 8
GATE_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
START_PORTS = (
    (-1, 0),
    (-1, 1),
    (2, 0),
    (2, 1),
    (0, -1),
    (1, -1),
    (0, 2),
    (1, 2),
)
PAIR_OPTIONS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
)
GEN_RETRIES = 300
ITEM_RETRIES = 200
ROUTE_RETRIES = 80


def _neighbors(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for nr, nc in _neighbors(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    assert mask[seed]
    q = deque([seed])
    seen = {seed}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(mask.shape, r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rot90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, -r) for r, c in cells}


def _flip(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in cells}


def _all_variants(cells: set[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    cur = set(cells)
    out: set[frozenset[tuple[int, int]]] = set()
    for _ in range(4):
        out.add(_normalize(cur))
        out.add(_normalize(_flip(cur)))
        cur = _rot90(cur)
    return sorted(out)


def _peel_layers(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    layers: list[set[tuple[int, int]]] = []
    while remaining:
        shell = {
            cell
            for cell in remaining
            if any((cell[0] + dr, cell[1] + dc) not in remaining for dr, dc in DIR4)
        }
        assert shell
        layers.append(shell)
        remaining -= shell
    return layers


def _count_2x2(cells: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in cells:
        if (r + 1, c) in cells and (r, c + 1) in cells and (r + 1, c + 1) in cells:
            total += 1
    return total


def _branch_count(cells: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in _neighbors((10**9, 10**9), r, c))
        if deg >= 3:
            total += 1
    return total


def _shape_score(cells: set[tuple[int, int]]):
    r0, c0, r1, c1 = _bbox(cells)
    area = len(cells)
    compactness = area / ((r1 - r0 + 1) * (c1 - c0 + 1))
    return (_count_2x2(cells) > 0, compactness, _count_2x2(cells), _branch_count(cells), -area)


def _build_shape_library():
    patterns = (
        ("11", "11"),
        ("111", "111"),
        ("1111", "1111"),
        ("111", "110"),
        ("1110", "1111"),
        ("111", "111", "111"),
        ("1111", "1111", "1111"),
        ("111", "111", "110"),
        ("1110", "1110", "1111", "1111"),
        ("1111", "1111", "1100", "1100"),
        ("1111", "1111", "1111", "1111"),
        ("11111", "11111", "11111", "11111", "11111"),
        ("11111", "11111", "11111", "11110", "11110"),
    )
    by_layers: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    seen: set[frozenset[tuple[int, int]]] = set()
    for pattern in patterns:
        base = {(r, c) for r, row in enumerate(pattern) for c, ch in enumerate(row) if ch == "1"}
        for variant in _all_variants(base):
            if variant in seen:
                continue
            seen.add(variant)
            cells = set(variant)
            layers = _peel_layers(cells)
            layer_count = len(layers)
            if layer_count not in by_layers:
                continue
            r0, c0, r1, c1 = _bbox(cells)
            attachments = []
            for r, c in sorted(cells):
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if (rr, cc) not in cells:
                        attachments.append({"cell": (r, c), "pre": (rr, cc)})
            by_layers[layer_count].append(
                {
                    "cells": cells,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "attachments": attachments,
                }
            )
    return by_layers


SHAPES_BY_LAYERS = _build_shape_library()


def _find_start_squares(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    starts: list[set[tuple[int, int]]] = []
    for comp in _components(grid == WHITE):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0, c1 - c0) != (1, 1):
            continue
        if all((r, c) in comp for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            starts.append(comp)
    return sorted(starts, key=lambda comp: _bbox(comp))


def _adjacency(cells: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        r, c = cell
        adj[cell] = [nb for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if nb in cells]
    return adj


def _reachable_without_edge(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    edge: tuple[tuple[int, int], tuple[int, int]],
) -> set[tuple[int, int]]:
    u, v = edge
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if (cur == u and nb == v) or (cur == v and nb == u):
                continue
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def _find_terminal_object(
    grid: np.ndarray,
    comp: set[tuple[int, int]],
    start_node: tuple[int, int],
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    adj = _adjacency(comp)
    best_cells = None
    best_prev = None
    best_score = None
    for u in sorted(comp):
        for v in adj[u]:
            if not (u < v):
                continue
            reachable = _reachable_without_edge(adj, start_node, (u, v))
            if len(reachable) == len(comp):
                continue
            other = comp - reachable
            if start_node in other:
                continue
            if len(other) < 4:
                continue
            if any(int(grid[r, c]) != GREY for r, c in other):
                continue
            score = _shape_score(other)
            attach_prev = v if u in other else u
            if best_score is None or score > best_score:
                best_score = score
                best_cells = other
                best_prev = attach_prev
    assert best_cells is not None and best_prev is not None, "failed to isolate a terminal grey object"
    return best_cells, best_prev


def _trace_gate_colors(
    grid: np.ndarray,
    comp: set[tuple[int, int]],
    start_node: tuple[int, int],
    object_cells: set[tuple[int, int]],
    attach_prev: tuple[int, int],
) -> list[int]:
    path_cells = comp - object_cells
    assert start_node in path_cells and attach_prev in path_cells
    cur = start_node
    prev = None
    colors: list[int] = []
    while True:
        value = int(grid[cur])
        if value != GREY:
            colors.append(value)
        if cur == attach_prev:
            return colors
        nxts = [nb for nb in ((cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)) if nb in path_cells and nb != prev]
        assert len(nxts) == 1, "path must remain a single chain before the object"
        prev, cur = cur, nxts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each marked path, read the gate colors in traversal order, and paint
    successive onion layers of the terminal grey object with those colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    starts = _find_start_squares(g)
    assert starts, "expected at least one 2x2 white start square"

    walkable = (g != BLACK) & (g != WHITE)
    claimed: set[tuple[int, int]] = set()
    for start_cells in starts:
        adjacent = set()
        for r, c in start_cells:
            for nr, nc in _neighbors(g.shape, r, c):
                if walkable[nr, nc]:
                    adjacent.add((nr, nc))
        assert len(adjacent) == 1, "each start square must touch exactly one path pixel"
        start_node = next(iter(adjacent))
        comp = _component_from_seed(walkable, start_node)
        assert claimed.isdisjoint(comp), "path components must be disjoint"
        claimed |= comp

        object_cells, attach_prev = _find_terminal_object(g, comp, start_node)
        gate_colors = _trace_gate_colors(g, comp, start_node, object_cells, attach_prev)
        layers = _peel_layers(object_cells)
        for idx, layer in enumerate(layers):
            if idx >= len(gate_colors):
                break
            for r, c in layer:
                out[r, c] = gate_colors[idx]

    return out


def _expand4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _bfs_path(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if not free[start] or not free[goal]:
        return None
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        nbrs = list(_neighbors(free.shape, cur[0], cur[1]))
        rng.shuffle(nbrs)
        nbrs.sort(key=lambda rc: abs(rc[0] - goal[0]) + abs(rc[1] - goal[1]))
        for nb in nbrs:
            if free[nb] and nb not in parent:
                parent[nb] = cur
                q.append(nb)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _route_is_simple_chain(route: list[tuple[int, int]]) -> bool:
    if len(route) != len(set(route)):
        return False
    route_set = set(route)
    endpoints = {route[0], route[-1]}
    for cell in route:
        r, c = cell
        deg = sum((r + dr, c + dc) in route_set for dr, dc in DIR4)
        if cell in endpoints:
            if deg != 1:
                return False
        elif deg != 2:
            return False
    return True


def _plan_route(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    free_cells = [tuple(cell) for cell in np.argwhere(free)]
    for _ in range(ROUTE_RETRIES):
        waypoint_count = rng.choices([0, 1, 2], weights=[3, 4, 2])[0]
        waypoints = []
        for _ in range(waypoint_count):
            cands = [
                cell
                for cell in free_cells
                if cell not in {start, goal}
                and abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= 3
                and abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]) >= 3
            ]
            if not cands:
                waypoints = []
                break
            waypoints.append(rng.choice(cands))
        points = [start] + waypoints + [goal]

        path = [start]
        used = {start}
        ok = True
        for nxt in points[1:]:
            seg_free = free.copy()
            for cell in _expand4(used - {path[-1]}, free.shape):
                seg_free[cell] = False
            seg_free[path[-1]] = True
            seg_free[nxt] = True
            seg = _bfs_path(seg_free, path[-1], nxt, rng)
            if seg is None:
                ok = False
                break
            path.extend(seg[1:])
            used.update(seg)
        if ok and _route_is_simple_chain(path):
            return path
    return None


def _sample_spaced_indices(length: int, count: int, rng: random.Random) -> list[int] | None:
    inner = list(range(1, length - 1))
    for _ in range(30):
        chosen = sorted(rng.sample(inner, count))
        if all(b - a >= 2 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _paint_input(shape: tuple[int, int], items: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for item in items:
        for r, c in item["start_cells"]:
            grid[r, c] = WHITE
        for r, c in item["object_cells"]:
            grid[r, c] = GREY
        for idx, (r, c) in enumerate(item["route"]):
            grid[r, c] = GREY
            if idx in item["gate_lookup"]:
                grid[r, c] = item["gate_lookup"][idx]
    return grid


def _expected_output(inp: np.ndarray, items: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for item in items:
        layers = _peel_layers(item["object_cells"])
        for idx, layer in enumerate(layers):
            if idx >= len(item["gate_colors"]):
                break
            color = item["gate_colors"][idx]
            for r, c in layer:
                out[r, c] = color
    return out


def _try_place_object(
    shape_info: dict,
    keepout: set[tuple[int, int]],
    canvas_shape: tuple[int, int],
    rng: random.Random,
):
    h, w = canvas_shape
    sh, sw = shape_info["height"], shape_info["width"]
    for _ in range(80):
        attach = rng.choice(shape_info["attachments"])
        r0 = rng.randint(1, h - sh - 2)
        c0 = rng.randint(1, w - sw - 2)
        object_cells = {(r0 + r, c0 + c) for r, c in shape_info["cells"]}
        attach_cell = (r0 + attach["cell"][0], c0 + attach["cell"][1])
        pre_cell = (r0 + attach["pre"][0], c0 + attach["pre"][1])
        if not (0 <= pre_cell[0] < h and 0 <= pre_cell[1] < w):
            continue
        if any(cell in keepout for cell in object_cells | {pre_cell}):
            continue
        return object_cells, attach_cell, pre_cell
    return None


def _try_place_start(
    canvas_shape: tuple[int, int],
    keepout: set[tuple[int, int]],
    object_keepout: set[tuple[int, int]],
    pre_cell: tuple[int, int],
    rng: random.Random,
):
    h, w = canvas_shape
    for _ in range(80):
        r0 = rng.randint(1, h - 4)
        c0 = rng.randint(1, w - 4)
        start_cells = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
        if any(cell in keepout or cell in object_keepout for cell in start_cells):
            continue
        port = rng.choice(START_PORTS)
        first_cell = (r0 + port[0], c0 + port[1])
        if not (0 <= first_cell[0] < h and 0 <= first_cell[1] < w):
            continue
        if first_cell in keepout or first_cell in object_keepout:
            continue
        if abs(first_cell[0] - pre_cell[0]) + abs(first_cell[1] - pre_cell[1]) < 4:
            continue
        return start_cells, first_cell
    return None


def _place_item(
    canvas_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    keepout: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    gate_count, layer_count = rng.choice(PAIR_OPTIONS)
    shape_info = rng.choice(SHAPES_BY_LAYERS[layer_count])
    obj = _try_place_object(shape_info, keepout, canvas_shape, rng)
    if obj is None:
        return None
    object_cells, attach_cell, pre_cell = obj
    object_keepout = _expand4(object_cells, canvas_shape)

    start_info = _try_place_start(canvas_shape, keepout, object_keepout, pre_cell, rng)
    if start_info is None:
        return None
    start_cells, first_cell = start_info
    start_keepout = _expand4(start_cells, canvas_shape)

    free = np.ones(canvas_shape, dtype=bool)
    for cell in keepout | object_keepout | start_keepout | occupied | object_cells | start_cells:
        free[cell] = False
    free[first_cell] = True
    free[pre_cell] = True

    route = _plan_route(free, first_cell, pre_cell, rng)
    if route is None or len(route) < gate_count + 2:
        return None

    gate_indices = _sample_spaced_indices(len(route), gate_count, rng)
    if gate_indices is None:
        return None
    gate_colors = list(rng.sample(GATE_COLORS, gate_count))
    gate_lookup = {idx: gate_colors[i] for i, idx in enumerate(gate_indices)}

    return {
        "start_cells": start_cells,
        "object_cells": object_cells,
        "attach_cell": attach_cell,
        "pre_cell": pre_cell,
        "route": route,
        "gate_indices": gate_indices,
        "gate_colors": gate_colors,
        "gate_lookup": gate_lookup,
        "reserved": object_cells | start_cells | set(route),
        "keepout": object_keepout | start_keepout | _expand4(set(route), canvas_shape),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    item_count = rng.choices([1, 2, 3], weights=[4, 3, 2])[0]
    h = rng.randint(16 + 4 * item_count, 30)
    w = rng.randint(16 + 4 * item_count, 30)
    occupied: set[tuple[int, int]] = set()
    keepout: set[tuple[int, int]] = set()
    items: list[dict] = []

    for _ in range(item_count):
        placed = None
        for _ in range(ITEM_RETRIES):
            placed = _place_item((h, w), occupied, keepout, rng)
            if placed is not None:
                items.append(placed)
                occupied |= placed["reserved"]
                keepout |= placed["keepout"]
                break
        if placed is None:
            return None

    inp = _paint_input((h, w), items)
    expected = _expected_output(inp, items)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
