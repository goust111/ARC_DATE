import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
NODE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PORT_OFFSETS = {
    "U": (-2, 0),
    "D": (2, 0),
    "L": (0, -2),
    "R": (0, 2),
}


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _port(center: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = PORT_OFFSETS[side]
    return center[0] + dr, center[1] + dc


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        assert r0 == r1 or c0 == c1
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        steps = max(abs(r1 - r0), abs(c1 - c0))
        for step in range(steps + 1):
            cell = (r0 + dr * step, c0 + dc * step)
            if not cells or cell != cells[-1]:
                cells.append(cell)
    return cells


def _draw_node(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = color


def _draw_path(grid: np.ndarray, points: list[tuple[int, int]]) -> None:
    for r, c in _polyline_cells(points):
        assert grid[r, c] in (BLACK, GREY)
        grid[r, c] = GREY


def _find_nodes(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    nodes: list[dict] = []
    cell_to_node: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not _inside(h, w, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in comp]
            cols = [cc for _, cc in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(comp) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue

            node_id = len(nodes)
            center = (r0 + 1, c0 + 1)
            node = {
                "id": node_id,
                "color": color,
                "cells": set(comp),
                "center": center,
                "priority": (r0, c0),
            }
            nodes.append(node)
            for cell in comp:
                cell_to_node[cell] = node_id

    assert len(nodes) >= 2
    assert len({node["color"] for node in nodes}) == len(nodes)
    return nodes, cell_to_node


def _node_starts(
    grid: np.ndarray, node: dict
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    starts: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in node["cells"]:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _inside(h, w, nr, nc) and int(grid[nr, nc]) == GREY:
                starts[(nr, nc)] = (dr, dc)
    return sorted(starts.items())


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    start_node_id: int,
    cell_to_node: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "looping path"
        seen_states.add(state)
        cells.append((r, c))

        fr, fc = r + dr, c + dc
        end_node = cell_to_node.get((fr, fc))
        if end_node is not None:
            assert end_node != start_node_id
            return cells, end_node
        if _inside(h, w, fr, fc) and int(grid[fr, fc]) == GREY:
            r, c = fr, fc
            continue

        back = (-dr, -dc)
        turn_dirs: list[tuple[int, int]] = []
        node_dirs: list[int] = []
        for ndr, ndc in DIRS4:
            if (ndr, ndc) == back:
                continue
            nr, nc = r + ndr, c + ndc
            if not _inside(h, w, nr, nc):
                continue
            if int(grid[nr, nc]) == GREY:
                turn_dirs.append((ndr, ndc))
                continue
            other_node = cell_to_node.get((nr, nc))
            if other_node is not None and other_node != start_node_id:
                node_dirs.append(other_node)

        if len(turn_dirs) == 1:
            dr, dc = turn_dirs[0]
            r += dr
            c += dc
            continue
        assert not turn_dirs and len(node_dirs) == 1, "ambiguous path topology"
        return cells, node_dirs[0]


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    nodes, cell_to_node = _find_nodes(g)
    degrees = {node["id"]: 0 for node in nodes}
    path_map: dict[tuple[tuple[int, int], frozenset[tuple[int, int]]], dict] = {}
    all_starts: set[tuple[int, int]] = set()

    for node in nodes:
        starts = _node_starts(g, node)
        assert 1 <= len(starts) <= 3
        for start, direction in starts:
            all_starts.add(start)
            cells, end_node = _trace_path(g, start, direction, node["id"], cell_to_node)
            key = (tuple(sorted((node["id"], end_node))), frozenset(cells))
            path_map[key] = {
                "nodes": tuple(sorted((node["id"], end_node))),
                "cells": cells,
            }

    for path in path_map.values():
        a, b = path["nodes"]
        degrees[a] += 1
        degrees[b] += 1

    assert all(1 <= deg <= 3 for deg in degrees.values())

    grey_cells = {tuple(pos) for pos in np.argwhere(g == GREY)}
    usage: dict[tuple[int, int], int] = defaultdict(int)
    for path in path_map.values():
        for cell in path["cells"]:
            usage[cell] += 1
    assert grey_cells == set(usage)

    for r, c in grey_cells:
        grey_neighbors = sum(
            1
            for dr, dc in DIRS4
            if _inside(g.shape[0], g.shape[1], r + dr, c + dc)
            and int(g[r + dr, c + dc]) == GREY
        )
        assert grey_neighbors in (1, 2, 4)
        if usage[(r, c)] > 1:
            assert grey_neighbors == 4

    return {
        "nodes": nodes,
        "paths": list(path_map.values()),
        "degrees": degrees,
        "usage": usage,
        "starts": all_starts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey network paths using the color of the higher-priority endpoint node.

    Input structure:
    - The background is black.
    - Each node is a solid 3x3 square in a unique non-black, non-grey color.
    - Grey one-pixel orthogonal paths connect pairs of nodes, with optional right-angle bends.
    - Some grey pixels are true crossings where one horizontal path and one vertical path share a cell.

    Rule:
    - Detect every 3x3 node and use its vertical position as its priority, with higher nodes winning.
    - From each node-adjacent grey start pixel, trace the connected path until it reaches the other endpoint node.
    - Recolor every grey pixel on that path with the color of the higher-priority endpoint node.
    - If multiple traced paths cover the same crossing pixel, keep the color from the globally highest-priority participating node.

    Key insight:
    - Path color depends on graph connectivity plus endpoint ranking by row, not on geometric proximity along the path itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    analysis = _analyze_scene(g)
    best: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    for path in analysis["paths"]:
        a_id, b_id = path["nodes"]
        a = analysis["nodes"][a_id]
        b = analysis["nodes"][b_id]
        winner = a if a["priority"] < b["priority"] else b
        candidate = (winner["priority"], int(winner["color"]))
        for cell in path["cells"]:
            prev = best.get(cell)
            if prev is None or candidate[0] < prev[0]:
                best[cell] = candidate

    for (r, c), (_, color) in best.items():
        out[r, c] = color
    return out


def _random_colors(rng: random.Random, count: int) -> list[int]:
    return rng.sample(NODE_COLORS, count)


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = rng.randint(0, min(3, int(r0)))
    left = rng.randint(0, min(3, int(c0)))
    bottom = rng.randint(0, min(3, int(grid.shape[0] - 1 - r1)))
    right = rng.randint(0, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _augment_scene(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = grid
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return _crop_with_margin(np.asarray(out, dtype=int), rng)


def _build_cross_scene(rng: random.Random, extra_leaf: bool) -> np.ndarray | None:
    x_a = rng.randint(4, 6)
    x_c = x_a + rng.randint(3, 4)
    x_cross = x_c + rng.randint(5, 6)
    x_d = x_cross + rng.randint(4, 5)
    x_b = x_d + rng.randint(3, 4)

    y_a = rng.randint(4, 5)
    y_d = y_a + rng.randint(4, 5)
    y_cross = y_d + rng.randint(4, 5)
    y_b = y_cross + rng.randint(4, 5)
    y_c = y_b + rng.randint(3, 4)

    centers = {
        "A": (y_a, x_a),
        "B": (y_b, x_b),
        "C": (y_c, x_c),
        "D": (y_d, x_d),
    }
    if extra_leaf:
        centers["E"] = (y_a + rng.randint(1, 2), x_b + rng.randint(3, 4))

    for r, c in centers.values():
        if not (1 <= r <= 28 and 1 <= c <= 28):
            return None

    grid = np.zeros((30, 30), dtype=int)
    for center, color in zip(centers.values(), _random_colors(rng, len(centers))):
        _draw_node(grid, center, color)

    _draw_path(
        grid,
        [
            _port(centers["A"], "D"),
            (y_cross, x_a),
            (y_cross, x_b),
            _port(centers["B"], "U"),
        ],
    )
    _draw_path(
        grid,
        [
            _port(centers["C"], "R"),
            (y_c, x_cross),
            (y_d, x_cross),
            _port(centers["D"], "L"),
        ],
    )
    if extra_leaf:
        y_e, x_e = centers["E"]
        _draw_path(
            grid,
            [
                _port(centers["D"], "R"),
                (y_d, x_e),
                _port(centers["E"], "D"),
            ],
        )
    return grid


def _build_star_scene(rng: random.Random) -> np.ndarray | None:
    x_b = rng.randint(5, 7)
    x_a = x_b + rng.randint(7, 8)
    x_d = x_a + rng.randint(7, 8)
    y_a = rng.randint(4, 5)
    y_b = y_a + rng.randint(5, 6)
    y_c = y_b + rng.randint(4, 5)
    y_d = y_c + rng.randint(4, 5)

    centers = {
        "A": (y_a, x_a),
        "B": (y_b, x_b),
        "C": (y_c, x_a),
        "D": (y_d, x_d),
    }
    for r, c in centers.values():
        if not (1 <= r <= 28 and 1 <= c <= 28):
            return None

    grid = np.zeros((30, 30), dtype=int)
    for center, color in zip(centers.values(), _random_colors(rng, len(centers))):
        _draw_node(grid, center, color)

    _draw_path(grid, [_port(centers["A"], "L"), (y_a, x_b), _port(centers["B"], "U")])
    _draw_path(grid, [_port(centers["A"], "D"), _port(centers["C"], "U")])
    _draw_path(grid, [_port(centers["A"], "R"), (y_a, x_d), _port(centers["D"], "U")])
    return grid


def _build_chain_scene(rng: random.Random) -> np.ndarray | None:
    x_0 = rng.randint(5, 6)
    x_2 = x_0 + rng.randint(8, 10)
    x_4 = x_0 + rng.randint(3, 5)

    y_0 = rng.randint(4, 5)
    y_2 = y_0 + rng.randint(3, 4)
    y_1 = y_2 + rng.randint(3, 4)
    y_3 = y_1 + rng.randint(5, 6)
    y_4 = y_3 + rng.randint(5, 6)

    centers = {
        "N0": (y_0, x_0),
        "N1": (y_1, x_0),
        "N2": (y_2, x_2),
        "N3": (y_3, x_2),
        "N4": (y_4, x_4),
    }
    for r, c in centers.values():
        if not (1 <= r <= 28 and 1 <= c <= 28):
            return None

    grid = np.zeros((30, 30), dtype=int)
    for center, color in zip(centers.values(), _random_colors(rng, len(centers))):
        _draw_node(grid, center, color)

    _draw_path(grid, [_port(centers["N0"], "D"), _port(centers["N1"], "U")])
    _draw_path(
        grid,
        [
            _port(centers["N1"], "R"),
            (y_1, x_2 - 2),
            _port(centers["N2"], "L"),
        ],
    )
    _draw_path(grid, [_port(centers["N2"], "D"), _port(centers["N3"], "U")])
    _draw_path(
        grid,
        [
            _port(centers["N3"], "L"),
            (y_3, x_4),
            _port(centers["N4"], "U"),
        ],
    )
    return grid


def _validate_generated(grid: np.ndarray, expect_crossing: bool) -> bool:
    try:
        analysis = _analyze_scene(grid)
    except AssertionError:
        return False

    node_rows = [node["center"][0] for node in analysis["nodes"]]
    if len(set(node_rows)) != len(node_rows):
        return False

    cross_cells = [cell for cell, count in analysis["usage"].items() if count > 1]
    if expect_crossing != bool(cross_cells):
        return False

    out = main(grid)
    if np.any(out == GREY):
        return False
    for node in analysis["nodes"]:
        for r, c in node["cells"]:
            if int(out[r, c]) != int(node["color"]):
                return False
    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    families = [
        ("cross", True),
        ("cross_leaf", True),
        ("star", False),
        ("chain", False),
    ]
    family, expect_crossing = rng.choice(families)
    if family == "cross":
        raw = _build_cross_scene(rng, extra_leaf=False)
    elif family == "cross_leaf":
        raw = _build_cross_scene(rng, extra_leaf=True)
    elif family == "star":
        raw = _build_star_scene(rng)
    else:
        raw = _build_chain_scene(rng)

    if raw is None:
        return None
    grid = _augment_scene(raw, rng)
    if max(grid.shape) > 30:
        return None
    if not _validate_generated(grid, expect_crossing):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-network scene")
