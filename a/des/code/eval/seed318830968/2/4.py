from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
FIRST_NODE_TRIES = 48
SECOND_NODE_TRIES = 64
WAYPOINT_TRIES = 24

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _square_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(top, top + 3) for c in range(left, left + 3)]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_nodes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    nodes: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g)):
        if color in (BG, GREY):
            continue
        for comp in _components(g == color):
            if len(comp) != 9:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0, c1 - c0) != (2, 2):
                continue
            solid = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if int(g[rr, cc]) != color:
                        solid = False
                        break
                if not solid:
                    break
            if not solid:
                continue
            nodes.append(
                {
                    "color": color,
                    "cells": _square_cells(r0, c0),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    nodes.sort(key=lambda node: node["bbox"])
    return nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify 3x3 colored node pairs linked by grey orthogonal paths.

    When a path contains a single non-grey gate pixel, the two endpoint node
    colors are swapped and the gate reverts to grey. Paths without a gate keep
    both endpoint colors unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    nodes = _find_nodes(g)
    assert nodes, "expected at least one 3x3 node"

    node_mask = np.zeros((h, w), dtype=bool)
    node_by_pos: dict[tuple[int, int], int] = {}
    for idx, node in enumerate(nodes):
        for cell in node["cells"]:
            node_mask[cell] = True
            node_by_pos[cell] = idx

    path_mask = (g != BG) & (~node_mask)
    path_components = _components(path_mask)
    assert path_components, "expected at least one path"

    out = g.copy()
    for comp in path_components:
        touched: set[int] = set()
        has_gate = False
        for r, c in comp:
            if int(g[r, c]) != GREY:
                has_gate = True
            for nr, nc in _neighbors4(r, c, h, w):
                node_idx = node_by_pos.get((nr, nc))
                if node_idx is not None:
                    touched.add(node_idx)

        assert len(touched) == 2, "each path must connect exactly two nodes"

        for r, c in comp:
            out[r, c] = GREY

        if has_gate:
            a, b = sorted(touched)
            color_a = int(nodes[a]["color"])
            color_b = int(nodes[b]["color"])
            for r, c in nodes[a]["cells"]:
                out[r, c] = color_b
            for r, c in nodes[b]["cells"]:
                out[r, c] = color_a

    return out


def _expand8(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _ports(top: int, left: int) -> dict[str, tuple[int, int]]:
    return {
        "top": (top - 1, left + 1),
        "bottom": (top + 3, left + 1),
        "left": (top + 1, left - 1),
        "right": (top + 1, left + 3),
    }


def _valid_node_positions(forbidden: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for top in range(1, h - 3):
        for left in range(1, w - 3):
            clear = True
            for rr in range(top - 1, top + 4):
                for cc in range(left - 1, left + 4):
                    if (rr, cc) in forbidden:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                positions.append((top, left))
    return positions


def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    if start in blocked or goal in blocked:
        return None

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dirs = list(DIRS4)
    random.shuffle(dirs)

    while q:
        cell = q.popleft()
        if cell == goal:
            break
        r, c = cell
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cell
            q.append(nxt)

    if goal not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _route_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    strategies = ["direct", "waypoint"]
    if random.random() < 0.55:
        strategies.reverse()

    for strategy in strategies:
        if strategy == "direct":
            path = _bfs_path(start, goal, blocked, h, w)
            if path is not None:
                return path
            continue

        free_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in blocked and (r, c) not in (start, goal)
        ]
        random.shuffle(free_cells)
        for waypoint in free_cells[:WAYPOINT_TRIES]:
            if abs(waypoint[0] - start[0]) + abs(waypoint[1] - start[1]) < 3:
                continue
            if abs(waypoint[0] - goal[0]) + abs(waypoint[1] - goal[1]) < 3:
                continue
            first = _bfs_path(start, waypoint, blocked, h, w)
            if first is None:
                continue
            blocked_second = set(blocked)
            blocked_second.update(first[:-1])
            second = _bfs_path(waypoint, goal, blocked_second, h, w)
            if second is None:
                continue
            return first + second[1:]

    return None


def _route_between_nodes(
    node_a: tuple[int, int],
    node_b: tuple[int, int],
    base_forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    blocked = set(base_forbidden)
    blocked.update(_expand8(_square_cells(*node_a), h, w))
    blocked.update(_expand8(_square_cells(*node_b), h, w))

    port_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for port_a in _ports(*node_a).values():
        for port_b in _ports(*node_b).values():
            if port_a == port_b:
                continue
            dist = abs(port_a[0] - port_b[0]) + abs(port_a[1] - port_b[1])
            if dist < 4:
                continue
            port_pairs.append((port_a, port_b))
    random.shuffle(port_pairs)

    for port_a, port_b in port_pairs:
        local_blocked = set(blocked)
        local_blocked.discard(port_a)
        local_blocked.discard(port_b)
        path = _route_path(port_a, port_b, local_blocked, h, w)
        if path is not None:
            return path
    return None


def _sample_gate_flags(pair_count: int) -> list[bool]:
    flags = [random.random() < 0.5 for _ in range(pair_count)]
    if pair_count > 1 and (all(flags) or not any(flags)) and random.random() < 0.5:
        flip_idx = random.randrange(pair_count)
        flags[flip_idx] = not flags[flip_idx]
    return flags


def _pick_gate_color(color_a: int, color_b: int) -> int:
    options = [color for color in NODE_COLORS if color not in (color_a, color_b)]
    return random.choice(options)


def _pick_pair_layout(
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
    gate_required: bool,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    positions = _valid_node_positions(forbidden, h, w)
    if len(positions) < 2:
        return None

    first_choices = random.sample(positions, min(len(positions), FIRST_NODE_TRIES))
    for node_a in first_choices:
        temp_forbidden = set(forbidden)
        temp_forbidden.update(_expand8(_square_cells(*node_a), h, w))
        positions_b = _valid_node_positions(temp_forbidden, h, w)
        if not positions_b:
            continue
        second_choices = random.sample(positions_b, min(len(positions_b), SECOND_NODE_TRIES))
        for node_b in second_choices:
            center_dist = abs((node_a[0] + 1) - (node_b[0] + 1)) + abs((node_a[1] + 1) - (node_b[1] + 1))
            if center_dist < 10:
                continue
            path = _route_between_nodes(node_a, node_b, forbidden, h, w)
            if path is None:
                continue
            if gate_required and len(path) < 3:
                continue
            return node_a, node_b, path

    return None


def _build_expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        for r, c in spec["path"]:
            out[r, c] = GREY
        if spec["gated"]:
            for r, c in _square_cells(*spec["node_a"]):
                out[r, c] = int(spec["color_b"])
            for r, c in _square_cells(*spec["node_b"]):
                out[r, c] = int(spec["color_a"])
    return out


def _max_pairs_for_shape(h: int, w: int) -> int:
    bound = min(h, w)
    if bound >= 26:
        return 4
    if bound >= 22:
        return 3
    if bound >= 18:
        return 2
    return 1


def _attempt_scene() -> np.ndarray | None:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    pair_count = random.randint(1, _max_pairs_for_shape(h, w))

    node_colors = random.sample(list(NODE_COLORS), 2 * pair_count)
    gate_flags = _sample_gate_flags(pair_count)

    grid = np.full((h, w), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for pair_idx in range(pair_count):
        layout = _pick_pair_layout(forbidden, h, w, gate_flags[pair_idx])
        if layout is None:
            return None
        node_a, node_b, path = layout
        color_a = node_colors[2 * pair_idx]
        color_b = node_colors[2 * pair_idx + 1]

        for r, c in _square_cells(*node_a):
            grid[r, c] = color_a
        for r, c in _square_cells(*node_b):
            grid[r, c] = color_b
        for r, c in path:
            grid[r, c] = GREY

        if gate_flags[pair_idx]:
            gate_choices = path[1:-1]
            if not gate_choices:
                return None
            gate_r, gate_c = random.choice(gate_choices)
            grid[gate_r, gate_c] = _pick_gate_color(color_a, color_b)

        pair_cells = _square_cells(*node_a) + _square_cells(*node_b) + path
        forbidden.update(_expand8(pair_cells, h, w))
        specs.append(
            {
                "node_a": node_a,
                "node_b": node_b,
                "color_a": color_a,
                "color_b": color_b,
                "path": list(path),
                "gated": bool(gate_flags[pair_idx]),
            }
        )

    expected = _build_expected_output(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(main(expected), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
