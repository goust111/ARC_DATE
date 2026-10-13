from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PRIMARY_TO_BITS = {BLUE: 1, RED: 2, YELLOW: 4}
BITS_TO_COLOR = {
    1: BLUE,
    2: RED,
    4: YELLOW,
    3: MAGENTA,
    5: GREEN,
    6: ORANGE,
    7: MAROON,
}

GENERATE_ATTEMPTS = 300
LENGTH_RANGE = (2, 5)
MARGIN_RANGE = (1, 4)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    assert a[1] == b[1], "pipe segments must be orthogonal"
    c = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, c) for r in range(r0, r1 + 1)]


def _pipe_degrees(pipe: np.ndarray) -> np.ndarray:
    h, w = pipe.shape
    deg = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not pipe[r, c]:
                continue
            count = 0
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(pipe.shape, rr, cc) and pipe[rr, cc]:
                    count += 1
            deg[r, c] = count
    return deg


def _adjacent_valve_color(grid: np.ndarray, coord: tuple[int, int]) -> int | None:
    r, c = coord
    colors = {
        int(grid[rr, cc])
        for dr, dc in DIRS
        for rr, cc in [(r + dr, c + dc)]
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) in PRIMARY_TO_BITS
    }
    assert len(colors) <= 1, "a pipe endpoint may touch at most one valve marker"
    return next(iter(colors)) if colors else None


def _build_pipe_graph(grid: np.ndarray):
    pipe = grid == GREY
    assert bool(pipe.any()), "expected a pipe network"
    deg = _pipe_degrees(pipe)

    significant = {
        (int(r), int(c))
        for r, c in np.argwhere(pipe)
        if int(deg[r, c]) != 2
    }
    assert significant, "expected at least one endpoint or junction"

    edges: list[dict[str, object]] = []
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {node: [] for node in significant}
    used_segments: set[frozenset[tuple[int, int]]] = set()

    for node in significant:
        r, c = node
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not _in_bounds(grid.shape, rr, cc) or not pipe[rr, cc]:
                continue
            step = frozenset({node, (rr, cc)})
            if step in used_segments:
                continue

            path = [node]
            prev = node
            cur = (rr, cc)
            used_segments.add(step)
            while cur not in significant:
                path.append(cur)
                nxts = []
                for ddr, ddc in DIRS:
                    nr, nc = cur[0] + ddr, cur[1] + ddc
                    if _in_bounds(grid.shape, nr, nc) and pipe[nr, nc] and (nr, nc) != prev:
                        nxts.append((nr, nc))
                assert len(nxts) == 1, "degree-2 pipe cells must continue uniquely"
                nxt = nxts[0]
                used_segments.add(frozenset({cur, nxt}))
                prev, cur = cur, nxt
            path.append(cur)

            edge_id = len(edges)
            edges.append({"a": node, "b": cur, "cells": path})
            adj[node].append((cur, edge_id))
            adj[cur].append((node, edge_id))

    covered = {cell for edge in edges for cell in edge["cells"]}
    assert covered == {tuple(map(int, cell)) for cell in np.argwhere(pipe)}, "graph must cover all pipe pixels"
    return deg, significant, edges, adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate valve colors along a grey pipe tree and mix colors at junctions.

    Input structure:
    - Black background with a one-cell-wide grey orthogonal pipe network.
    - Colored valve markers (red, blue, yellow) touch some dead-end pipe cells.
    - Pipe junctions are T or cross nodes with three or four pipe neighbors.

    Rule:
    - A valve colors its pipe branch until the first junction or dead-end.
    - A junction that receives colors from at least two child branches becomes their mix.
    - That mixed color continues only toward the unique bare dead-end sink.
    - Valve pixels themselves stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    deg, significant, edges, adj = _build_pipe_graph(g)

    endpoints = [node for node in significant if int(deg[node]) <= 1]
    assert endpoints, "expected pipe endpoints"

    valve_bits: dict[tuple[int, int], int] = {}
    sink_candidates: list[tuple[int, int]] = []
    for node in endpoints:
        valve_color = _adjacent_valve_color(g, node)
        if valve_color is None:
            sink_candidates.append(node)
        else:
            valve_bits[node] = PRIMARY_TO_BITS[valve_color]

    assert len(sink_candidates) == 1, "expected exactly one unmarked sink endpoint"
    sink = sink_candidates[0]
    assert 2 <= len(valve_bits) <= 3, "expected two or three distinct valves"
    assert len(set(valve_bits.values())) == len(valve_bits), "valve colors must be distinct"

    parent: dict[tuple[int, int], tuple[int, int] | None] = {sink: None}
    parent_edge: dict[tuple[int, int], int] = {}
    stack = [sink]
    order = [sink]
    while stack:
        node = stack.pop()
        for nxt, edge_id in adj[node]:
            if parent[node] == nxt:
                continue
            if nxt in parent:
                raise AssertionError("pipe network must be a tree")
            parent[nxt] = node
            parent_edge[nxt] = edge_id
            stack.append(nxt)
            order.append(nxt)

    assert len(parent) == len(significant), "pipe network must be connected"
    assert len(edges) == len(significant) - 1, "pipe graph must be acyclic"

    children: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {node: [] for node in significant}
    for node, par in parent.items():
        if par is None:
            continue
        children[par].append((node, parent_edge[node]))

    edge_bits: dict[int, int] = {}
    node_bits: dict[tuple[int, int], int] = {}

    def dfs(node: tuple[int, int]) -> int | None:
        if node in valve_bits:
            assert not children[node], "valves must sit on leaf endpoints"
            return valve_bits[node]

        incoming: list[tuple[int, int]] = []
        for child, edge_id in children[node]:
            msg = dfs(child)
            if msg is None:
                continue
            edge_bits[edge_id] = msg
            incoming.append((edge_id, msg))

        if len(incoming) >= 2:
            bits = 0
            for _edge_id, msg in incoming:
                bits |= msg
            node_bits[node] = bits
            return bits
        return None

    dfs(sink)

    out = g.copy()
    for edge_id, bits in edge_bits.items():
        color = BITS_TO_COLOR[bits]
        for r, c in edges[edge_id]["cells"]:
            out[r, c] = color

    for node, bits in node_bits.items():
        out[node] = BITS_TO_COLOR[bits]

    return out


def _transform_point(point: tuple[int, int], rotation: int, flip: bool) -> tuple[int, int]:
    r, c = point
    if flip:
        c = -c
    for _ in range(rotation % 4):
        r, c = c, -r
    return int(r), int(c)


def _random_length() -> int:
    return random.randint(*LENGTH_RANGE)


def _build_two_mix() -> dict[str, object]:
    colors = random.sample(list(PRIMARY_TO_BITS), 2)
    nodes = {
        "S": (_random_length(), 0),
        "J": (0, 0),
        "A": (0, -_random_length()),
        "B": (0, _random_length()),
    }
    valves = {"A": colors[0], "B": colors[1]}
    valve_points = {
        "A": (nodes["A"][0], nodes["A"][1] - 1),
        "B": (nodes["B"][0], nodes["B"][1] + 1),
    }
    parents = {"J": "S", "A": "J", "B": "J"}
    return {
        "name": "two_mix",
        "root": "S",
        "nodes": nodes,
        "parents": parents,
        "valves": valves,
        "valve_points": valve_points,
    }


def _build_chain_three() -> dict[str, object]:
    colors = random.sample(list(PRIMARY_TO_BITS), 3)
    side_sign = random.choice((-1, 1))
    nodes = {
        "S": (_random_length(), 0),
        "J0": (0, 0),
        "J1": (-_random_length(), 0),
        "A": (0, side_sign * _random_length()),
        "B": (0, 0),  # overwritten below for readability
        "C": (0, 0),  # overwritten below for readability
    }
    upper_row = nodes["J1"][0]
    nodes["B"] = (upper_row, -_random_length())
    nodes["C"] = (upper_row, _random_length())
    valves = {"A": colors[2], "B": colors[0], "C": colors[1]}
    valve_points = {
        "A": (nodes["A"][0], nodes["A"][1] + side_sign),
        "B": (nodes["B"][0], nodes["B"][1] - 1),
        "C": (nodes["C"][0], nodes["C"][1] + 1),
    }
    parents = {"J0": "S", "J1": "J0", "A": "J0", "B": "J1", "C": "J1"}
    return {
        "name": "chain_three",
        "root": "S",
        "nodes": nodes,
        "parents": parents,
        "valves": valves,
        "valve_points": valve_points,
    }


def _build_star_three() -> dict[str, object]:
    colors = random.sample(list(PRIMARY_TO_BITS), 3)
    nodes = {
        "S": (_random_length(), 0),
        "J": (0, 0),
        "A": (0, -_random_length()),
        "B": (0, _random_length()),
        "C": (-_random_length(), 0),
    }
    valves = {"A": colors[0], "B": colors[1], "C": colors[2]}
    valve_points = {
        "A": (nodes["A"][0], nodes["A"][1] - 1),
        "B": (nodes["B"][0], nodes["B"][1] + 1),
        "C": (nodes["C"][0] - 1, nodes["C"][1]),
    }
    parents = {"J": "S", "A": "J", "B": "J", "C": "J"}
    return {
        "name": "star_three",
        "root": "S",
        "nodes": nodes,
        "parents": parents,
        "valves": valves,
        "valve_points": valve_points,
    }


def _transform_scenario(spec: dict[str, object]) -> dict[str, object]:
    rotation = random.randrange(4)
    flip = bool(random.randrange(2))

    nodes = {
        name: _transform_point(coord, rotation, flip)
        for name, coord in spec["nodes"].items()
    }
    valve_points = {
        name: _transform_point(coord, rotation, flip)
        for name, coord in spec["valve_points"].items()
    }

    all_points = list(nodes.values()) + list(valve_points.values())
    min_r = min(r for r, _c in all_points)
    max_r = max(r for r, _c in all_points)
    min_c = min(c for _r, c in all_points)
    max_c = max(c for _r, c in all_points)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    height = (max_r - min_r + 1) + top + bottom
    width = (max_c - min_c + 1) + left + right
    dr = top - min_r
    dc = left - min_c

    shifted_nodes = {name: (r + dr, c + dc) for name, (r, c) in nodes.items()}
    shifted_valves = {name: (r + dr, c + dc) for name, (r, c) in valve_points.items()}

    return {
        "name": spec["name"],
        "root": spec["root"],
        "parents": dict(spec["parents"]),
        "nodes": shifted_nodes,
        "valves": dict(spec["valves"]),
        "valve_points": shifted_valves,
        "shape": (height, width),
    }


def _render_scenario(spec: dict[str, object]) -> np.ndarray:
    grid = np.full(spec["shape"], BLACK, dtype=int)

    for child, parent in spec["parents"].items():
        a = spec["nodes"][child]
        b = spec["nodes"][parent]
        for r, c in _line_cells(a, b):
            grid[r, c] = GREY

    for name, color in spec["valves"].items():
        vr, vc = spec["valve_points"][name]
        grid[vr, vc] = color

    return grid


def _expected_from_spec(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    children: dict[str, list[str]] = {name: [] for name in spec["nodes"]}
    for child, parent in spec["parents"].items():
        children[parent].append(child)

    edge_bits: dict[tuple[str, str], int] = {}
    node_bits: dict[str, int] = {}

    def dfs(node: str) -> int | None:
        if node in spec["valves"]:
            return PRIMARY_TO_BITS[spec["valves"][node]]

        incoming: list[int] = []
        for child in children[node]:
            msg = dfs(child)
            if msg is None:
                continue
            edge_bits[(child, node)] = msg
            incoming.append(msg)

        if len(incoming) >= 2:
            bits = 0
            for msg in incoming:
                bits |= msg
            node_bits[node] = bits
            return bits
        return None

    dfs(spec["root"])

    out = grid.copy()
    for (child, parent), bits in edge_bits.items():
        color = BITS_TO_COLOR[bits]
        for r, c in _line_cells(spec["nodes"][child], spec["nodes"][parent]):
            out[r, c] = color
    for node, bits in node_bits.items():
        out[spec["nodes"][node]] = BITS_TO_COLOR[bits]
    return out


def _scenario_has_expected_branches(spec: dict[str, object], out: np.ndarray) -> bool:
    colors = set(map(int, out.ravel()))
    if spec["name"] == "two_mix":
        bits = 0
        for valve_color in spec["valves"].values():
            bits |= PRIMARY_TO_BITS[valve_color]
        return BITS_TO_COLOR[bits] in colors
    if spec["name"] == "chain_three":
        upper_bits = PRIMARY_TO_BITS[spec["valves"]["B"]] | PRIMARY_TO_BITS[spec["valves"]["C"]]
        return BITS_TO_COLOR[upper_bits] in colors and MAROON in colors
    return MAROON in colors


def _generate_candidate() -> np.ndarray | None:
    builder = random.choice((_build_two_mix, _build_chain_three, _build_star_three))
    spec = _transform_scenario(builder())
    grid = _render_scenario(spec)
    expected = _expected_from_spec(grid, spec)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if not _scenario_has_expected_branches(spec, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid pipe-mixing puzzle")
