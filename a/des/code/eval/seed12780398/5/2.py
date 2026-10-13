from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


RED = 2
GREY = 5
MAROON = 9

SIDE_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
SIDES = tuple(SIDE_VECS)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _box_border_is(patch: np.ndarray, color: int) -> bool:
    return bool(
        np.all(patch[0, :] == color)
        and np.all(patch[2, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, 2] == color)
    )


def _scan_objects(
    grid: np.ndarray, bg: int
) -> tuple[list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    nodes: list[dict] = []
    terminals: list[dict] = []
    blocked = np.zeros((h, w), dtype=bool)

    for r in range(h - 2):
        for c in range(w - 2):
            patch = g[r : r + 3, c : c + 3]
            center = int(patch[1, 1])
            if _box_border_is(patch, GREY) and center not in {bg, RED, GREY, MAROON}:
                node = {
                    "id": len(nodes),
                    "kind": "node",
                    "top_left": (r, c),
                    "center": (r + 1, c + 1),
                    "priority": center,
                    "connections": [],
                }
                nodes.append(node)
                blocked[r : r + 3, c : c + 3] = True
            elif _box_border_is(patch, MAROON) and center == bg:
                terminal = {
                    "id": len(terminals),
                    "kind": "terminal",
                    "top_left": (r, c),
                    "center": (r + 1, c + 1),
                    "connections": [],
                }
                terminals.append(terminal)
                blocked[r : r + 3, c : c + 3] = True

    return nodes, terminals, blocked


def _port_cell(center: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = SIDE_VECS[side]
    return center[0] + 2 * dr, center[1] + 2 * dc


def _wire_components(grid: np.ndarray, wire_mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comp_id = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not wire_mask[r, c] or comp_id[r, c] != -1:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            comp_id[r, c] = len(components)
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if comp_id[nr, nc] != -1 or not wire_mask[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    comp_id[nr, nc] = comp_id[r, c]
                    q.append((nr, nc))
            components.append({"color": color, "cells": cells, "touches": []})

    return comp_id, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the signal from the red source through colored wires, routing at grey nodes by
    the node-center color, and light the reached terminal with the last wire color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    nodes, terminals, blocked = _scan_objects(g, bg)

    wire_mask = (
        ~blocked
        & (g != bg)
        & (g != RED)
        & (g != GREY)
        & (g != MAROON)
    )
    comp_id, components = _wire_components(g, wire_mask)

    for obj in itertools.chain(nodes, terminals):
        center = obj["center"]
        seen_components = set()
        for side in SIDES:
            pr, pc = _port_cell(center, side)
            if not _in_bounds(g.shape, pr, pc) or not wire_mask[pr, pc]:
                continue
            cid = int(comp_id[pr, pc])
            if cid in seen_components:
                continue
            seen_components.add(cid)
            obj["connections"].append({"component": cid, "side": side})
            components[cid]["touches"].append((obj["kind"], obj["id"]))

    source_pixels = [
        (int(r), int(c))
        for r, c in np.argwhere(g == RED)
        if r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1
    ]
    assert len(source_pixels) == 1
    src_r, src_c = source_pixels[0]
    source_neighbors = [
        (nr, nc)
        for nr, nc in _neighbors4(src_r, src_c)
        if _in_bounds(g.shape, nr, nc) and wire_mask[nr, nc]
    ]
    assert len(source_neighbors) == 1
    start_component = int(comp_id[source_neighbors[0]])
    components[start_component]["touches"].append(("source", 0))

    for component in components:
        deduped: list[tuple[str, int]] = []
        for touch in component["touches"]:
            if touch not in deduped:
                deduped.append(touch)
        component["touches"] = deduped

    node_by_id = {node["id"]: node for node in nodes}
    terminal_by_id = {terminal["id"]: terminal for terminal in terminals}

    out = g.copy()
    previous_touch = ("source", 0)
    current_component = start_component
    visited_components: set[int] = set()

    while current_component not in visited_components:
        visited_components.add(current_component)
        component = components[current_component]
        next_touches = [
            touch for touch in component["touches"] if touch != previous_touch and touch[0] != "source"
        ]
        if len(next_touches) != 1:
            break

        kind, obj_id = next_touches[0]
        if kind == "terminal":
            tr, tc = terminal_by_id[obj_id]["center"]
            out[tr, tc] = int(component["color"])
            break

        node = node_by_id[obj_id]
        matches = []
        for conn in node["connections"]:
            cid = int(conn["component"])
            if cid == current_component:
                continue
            if int(components[cid]["color"]) == int(node["priority"]):
                matches.append(cid)
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            break

        previous_touch = ("node", obj_id)
        current_component = matches[0]

    return out


# --------------- generator ---------------

FREE_COLORS = [0, 1, 3, 4, 6, 7, 8]
LATTICE_SIDE_RANGE = (4, 6)
STEP_CHOICES = [6, 7]
NODE_COUNT_CHOICES = [1, 2, 3]
MAX_GENERATION_ATTEMPTS = 400


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[str, tuple[int, int]]]:
    r, c = cell
    out = []
    for side, (dr, dc) in SIDE_VECS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((side, (nr, nc)))
    return out


def _side_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for side, (sdr, sdc) in SIDE_VECS.items():
        if (dr, dc) == (sdr, sdc):
            return side
    raise ValueError((src, dst))


def _boundary_starts(rows: int, cols: int) -> list[tuple[tuple[int, int], str]]:
    starts: list[tuple[tuple[int, int], str]] = []
    for c in range(cols):
        starts.append(((0, c), "up"))
        starts.append(((rows - 1, c), "down"))
    for r in range(rows):
        starts.append(((r, 0), "left"))
        starts.append(((r, cols - 1), "right"))
    return starts


def _sample_path(
    rows: int, cols: int, node_count: int
) -> tuple[list[tuple[int, int]], str] | None:
    starts = _boundary_starts(rows, cols)
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == node_count:
            return list(path)
        options = [cell for _side, cell in _cell_neighbors(path[-1], rows, cols) if cell not in path]
        random.shuffle(options)
        for nxt in options:
            path.append(nxt)
            result = dfs(path)
            if result is not None:
                return result
            path.pop()
        return None

    for start_cell, source_side in starts:
        result = dfs([start_cell])
        if result is not None:
            return result, source_side
    return None


def _mandatory_options(
    spec: dict, rows: int, cols: int, occupied_cells: set[tuple[int, int]]
) -> list[dict]:
    forbidden_sides = {spec["incoming_side"]}
    if spec["next_node_side"] is not None:
        forbidden_sides.add(spec["next_node_side"])
    available = [
        (side, cell)
        for side, cell in _cell_neighbors(spec["cell"], rows, cols)
        if side not in forbidden_sides and cell not in occupied_cells
    ]

    if spec["kind"] == "continue":
        return [{"main": None, "decoys": [choice]} for choice in available]
    if spec["kind"] == "success":
        out = []
        for main_choice in available:
            for decoy_choice in available:
                if decoy_choice != main_choice:
                    out.append({"main": main_choice, "decoys": [decoy_choice]})
        return out

    return [
        {"main": None, "decoys": list(combo)}
        for combo in itertools.combinations(available, 2)
    ]


def _assign_terminals(
    specs: list[dict], rows: int, cols: int, path_cells: set[tuple[int, int]]
) -> dict[int, dict] | None:
    def backtrack(
        remaining: set[int], occupied: set[tuple[int, int]], assigned: dict[int, dict]
    ) -> dict[int, dict] | None:
        if not remaining:
            return assigned

        best_index = None
        best_options = None
        for index in remaining:
            options = _mandatory_options(specs[index], rows, cols, occupied)
            if not options:
                return None
            random.shuffle(options)
            if best_options is None or len(options) < len(best_options):
                best_index = index
                best_options = options

        assert best_index is not None and best_options is not None
        for option in best_options:
            newly_used = set(cell for _side, cell in option["decoys"])
            if option["main"] is not None:
                newly_used.add(option["main"][1])
            result = backtrack(
                remaining - {best_index},
                occupied | newly_used,
                {**assigned, best_index: option},
            )
            if result is not None:
                return result
        return None

    assigned = backtrack(set(range(len(specs))), set(path_cells), {})
    if assigned is None:
        return None

    occupied = set(path_cells)
    for option in assigned.values():
        for _side, cell in option["decoys"]:
            occupied.add(cell)
        if option["main"] is not None:
            occupied.add(option["main"][1])

    for index, spec in enumerate(specs):
        if random.random() >= 0.4:
            continue
        used_sides = {spec["incoming_side"]}
        if spec["next_node_side"] is not None:
            used_sides.add(spec["next_node_side"])
        for side, _cell in assigned[index]["decoys"]:
            used_sides.add(side)
        if assigned[index]["main"] is not None:
            used_sides.add(assigned[index]["main"][0])

        extras = [
            (side, cell)
            for side, cell in _cell_neighbors(spec["cell"], rows, cols)
            if side not in used_sides and cell not in occupied
        ]
        if extras:
            extra = random.choice(extras)
            assigned[index]["decoys"].append(extra)
            occupied.add(extra[1])

    return assigned


def _assign_edge_colors(edges: list[dict], palette: list[int]) -> bool:
    incident: dict[tuple[str, int], set[int]] = Counter()  # type: ignore[assignment]
    incident = {}
    for edge in edges:
        for endpoint in edge["ends"]:
            incident.setdefault(endpoint, set())

    order = sorted(
        range(len(edges)),
        key=lambda idx: -sum(len(incident[endpoint]) for endpoint in edges[idx]["ends"]),
    )

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        edge_index = order[pos]
        edge = edges[edge_index]
        a, b = edge["ends"]
        blocked = incident[a] | incident[b]
        choices = [color for color in palette if color not in blocked]
        random.shuffle(choices)
        for color in choices:
            edge["color"] = color
            incident[a].add(color)
            incident[b].add(color)
            if backtrack(pos + 1):
                return True
            incident[a].remove(color)
            incident[b].remove(color)
            edge.pop("color", None)
        return False

    return backtrack(0)


def _center_for(cell: tuple[int, int], step: int) -> tuple[int, int]:
    return 3 + step * cell[0], 3 + step * cell[1]


def _source_pos(cell: tuple[int, int], source_side: str, step: int) -> tuple[int, int]:
    cr, cc = _center_for(cell, step)
    dr, dc = SIDE_VECS[source_side]
    return cr + 3 * dr, cc + 3 * dc


def _port_from_cell(cell: tuple[int, int], side: str, step: int) -> tuple[int, int]:
    cr, cc = _center_for(cell, step)
    dr, dc = SIDE_VECS[side]
    return cr + 2 * dr, cc + 2 * dc


def _draw_segment(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int
) -> None:
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            grid[r0, c] = color
    else:
        assert c0 == c1
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            grid[r, c0] = color


def _expected_output(grid: np.ndarray, lit_terminal: tuple[int, int] | None, color: int | None) -> np.ndarray:
    out = np.array(grid, copy=True)
    if lit_terminal is not None and color is not None:
        out[lit_terminal] = int(color)
    return out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(*LATTICE_SIDE_RANGE)
    cols = random.randint(*LATTICE_SIDE_RANGE)
    step = random.choice(STEP_CHOICES)
    node_count = random.choice([n for n in NODE_COUNT_CHOICES if n <= rows * cols])
    success = bool(random.getrandbits(1))
    bg = random.choice(FREE_COLORS)
    wire_palette = [color for color in FREE_COLORS if color != bg]

    path_sample = _sample_path(rows, cols, node_count)
    if path_sample is None:
        return None
    path, source_side = path_sample

    specs = []
    for i, cell in enumerate(path):
        incoming_side = source_side if i == 0 else _side_between(cell, path[i - 1])
        next_node_side = _side_between(cell, path[i + 1]) if i + 1 < len(path) else None
        kind = "continue"
        if i == len(path) - 1:
            kind = "success" if success else "stop"
        specs.append(
            {
                "cell": cell,
                "incoming_side": incoming_side,
                "next_node_side": next_node_side,
                "kind": kind,
            }
        )

    terminal_assignments = _assign_terminals(specs, rows, cols, set(path))
    if terminal_assignments is None:
        return None

    edges: list[dict] = []
    edges.append(
        {
            "kind": "source",
            "ends": (("source", 0), ("node", 0)),
            "node_index": 0,
            "node_side": source_side,
        }
    )
    for i in range(len(path) - 1):
        edges.append(
            {
                "kind": "path",
                "ends": (("node", i), ("node", i + 1)),
                "node_index": i,
                "next_node_index": i + 1,
                "node_side": _side_between(path[i], path[i + 1]),
                "other_side": _side_between(path[i + 1], path[i]),
            }
        )

    terminals: list[dict] = []
    terminal_index = 0
    for node_index, option in terminal_assignments.items():
        if option["main"] is not None:
            side, cell = option["main"]
            terminals.append({"id": terminal_index, "cell": cell, "node_index": node_index, "side": side})
            edges.append(
                {
                    "kind": "main_terminal",
                    "ends": (("node", node_index), ("terminal", terminal_index)),
                    "node_index": node_index,
                    "terminal_index": terminal_index,
                    "node_side": side,
                    "other_side": OPPOSITE[side],
                }
            )
            terminal_index += 1
        for side, cell in option["decoys"]:
            terminals.append({"id": terminal_index, "cell": cell, "node_index": node_index, "side": side})
            edges.append(
                {
                    "kind": "decoy_terminal",
                    "ends": (("node", node_index), ("terminal", terminal_index)),
                    "node_index": node_index,
                    "terminal_index": terminal_index,
                    "node_side": side,
                    "other_side": OPPOSITE[side],
                }
            )
            terminal_index += 1

    if not _assign_edge_colors(edges, wire_palette):
        return None

    priorities: dict[int, int] = {}
    for i in range(len(path)):
        if i < len(path) - 1:
            outgoing = next(edge for edge in edges if edge["kind"] == "path" and edge["node_index"] == i)
            priorities[i] = int(outgoing["color"])
        elif success:
            outgoing = next(
                edge for edge in edges if edge["kind"] == "main_terminal" and edge["node_index"] == i
            )
            priorities[i] = int(outgoing["color"])
        else:
            incident_colors = {
                int(edge["color"])
                for edge in edges
                if ("node", i) in edge["ends"]
            }
            options = [color for color in wire_palette if color not in incident_colors]
            if not options:
                return None
            priorities[i] = random.choice(options)

    height = step * (rows - 1) + 7
    width = step * (cols - 1) + 7
    grid = np.full((height, width), bg, dtype=int)

    for i, cell in enumerate(path):
        cr, cc = _center_for(cell, step)
        r0, c0 = cr - 1, cc - 1
        grid[r0 : r0 + 3, c0 : c0 + 3] = GREY
        grid[cr, cc] = priorities[i]

    terminal_centers: dict[int, tuple[int, int]] = {}
    for terminal in terminals:
        cr, cc = _center_for(terminal["cell"], step)
        r0, c0 = cr - 1, cc - 1
        grid[r0 : r0 + 3, c0 : c0 + 3] = MAROON
        grid[cr, cc] = bg
        terminal_centers[terminal["id"]] = (cr, cc)

    src_r, src_c = _source_pos(path[0], source_side, step)
    if not _in_bounds(grid.shape, src_r, src_c):
        return None
    grid[src_r, src_c] = RED

    for edge in edges:
        color = int(edge["color"])
        if edge["kind"] == "source":
            node_cell = path[edge["node_index"]]
            port = _port_from_cell(node_cell, edge["node_side"], step)
            _draw_segment(grid, port, port, color)
        elif edge["kind"] == "path":
            a = _port_from_cell(path[edge["node_index"]], edge["node_side"], step)
            b = _port_from_cell(path[edge["next_node_index"]], edge["other_side"], step)
            _draw_segment(grid, a, b, color)
        else:
            node_cell = path[edge["node_index"]]
            terminal_cell = terminals[edge["terminal_index"]]["cell"]
            a = _port_from_cell(node_cell, edge["node_side"], step)
            b = _port_from_cell(terminal_cell, edge["other_side"], step)
            _draw_segment(grid, a, b, color)

    lit_terminal = None
    lit_color = None
    if success:
        main_edge = next(edge for edge in edges if edge["kind"] == "main_terminal")
        lit_terminal = terminal_centers[int(main_edge["terminal_index"])]
        lit_color = int(main_edge["color"])

    expected = _expected_output(grid, lit_terminal, lit_color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _attempt_generate()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid signal-routing instance")
