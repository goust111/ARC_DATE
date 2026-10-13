from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
CORRIDOR = 5
ROOM_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
LATTICE_CENTERS = [4, 13, 22]
ROOM_SIDE_CHOICES = [5, 7]
CHAIN_LEN_CHOICES = [3, 4, 5]
MAX_BRANCHES = 3
GEN_RETRIES = 500

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Node = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _nonzero_bbox(grid: np.ndarray) -> BBox:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        raise AssertionError("expected at least one foreground pixel")
    return _bbox_coords(coords)


def _boundary_coords(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _detect_rooms(grid: np.ndarray) -> tuple[list[dict], dict[Coord, int]]:
    colors = sorted(set(map(int, grid.flatten())) - {BG, CORRIDOR})
    rooms: list[dict] = []
    boundary_owner: dict[Coord, int] = {}
    seen_room_colors: set[int] = set()

    for color in colors:
        for comp in _components_mask(grid == color):
            if comp.shape[0] <= 1:
                continue
            bbox = _bbox_coords(comp)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == color):
                continue
            if not np.all(patch[-1, :] == color):
                continue
            if not np.all(patch[:, 0] == color):
                continue
            if not np.all(patch[:, -1] == color):
                continue

            cr = r0 + h // 2
            cc = c0 + w // 2
            key_color = int(grid[cr, cc])
            if key_color in {BG, CORRIDOR, color}:
                continue

            inner = patch[1:-1, 1:-1]
            rel_center = (h // 2 - 1, w // 2 - 1)
            non_center_vals = []
            for rr in range(inner.shape[0]):
                for cc2 in range(inner.shape[1]):
                    if (rr, cc2) == rel_center:
                        continue
                    non_center_vals.append(int(inner[rr, cc2]))

            if not non_center_vals:
                continue
            if any(v not in {BG, color} for v in non_center_vals):
                continue

            filled = all(v == color for v in non_center_vals)
            hollow = all(v == BG for v in non_center_vals)
            if not (filled or hollow):
                continue
            if color in seen_room_colors:
                raise AssertionError("room border colors must be unique")
            seen_room_colors.add(color)

            rid = len(rooms)
            room = {
                "id": rid,
                "bbox": bbox,
                "border_color": int(color),
                "center": (cr, cc),
                "key_color": key_color,
                "filled": filled,
            }
            rooms.append(room)
            for cell in _boundary_coords(bbox):
                boundary_owner[cell] = rid

    if not rooms:
        raise AssertionError("expected at least one room")
    return rooms, boundary_owner


def _detect_corridors(
    grid: np.ndarray, boundary_owner: dict[Coord, int], n_rooms: int
) -> tuple[list[dict], dict[int, list[tuple[int, int]]]]:
    corridors: list[dict] = []
    adjacency: dict[int, list[tuple[int, int]]] = {rid: [] for rid in range(n_rooms)}

    for comp in _components_mask(grid == CORRIDOR):
        rows = {int(r) for r, _ in comp}
        cols = {int(c) for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            raise AssertionError("corridors must be straight horizontal or vertical lines")

        touched: set[int] = set()
        cells = [(int(r), int(c)) for r, c in comp]
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                rid = boundary_owner.get((nr, nc))
                if rid is not None:
                    touched.add(rid)

        if len(touched) != 2:
            raise AssertionError("each corridor component must touch exactly two rooms")

        a, b = sorted(touched)
        eid = len(corridors)
        corridors.append({"id": eid, "rooms": (a, b), "cells": cells})
        adjacency[a].append((b, eid))
        adjacency[b].append((a, eid))

    return corridors, adjacency


def _trace_chain(
    rooms: list[dict], adjacency: dict[int, list[tuple[int, int]]], start_id: int
) -> tuple[list[int], list[int]]:
    visited_rooms = [start_id]
    visited_set = {start_id}
    used_corridors: list[int] = []
    current = start_id

    while True:
        want = int(rooms[current]["key_color"])
        matches = [
            (nbr, eid)
            for nbr, eid in adjacency[current]
            if int(rooms[nbr]["border_color"]) == want
        ]
        if not matches:
            break
        if len(matches) != 1:
            raise AssertionError("adjacent color match must be unique")

        nxt, eid = matches[0]
        if nxt in visited_set:
            raise AssertionError("pointer chain must not loop")
        used_corridors.append(eid)
        visited_rooms.append(nxt)
        visited_set.add(nxt)
        current = nxt

    return visited_rooms, used_corridors


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, list[tuple[int, int]]], int]:
    rooms, boundary_owner = _detect_rooms(grid)
    filled_rooms = [room["id"] for room in rooms if room["filled"]]
    if len(filled_rooms) != 1:
        raise AssertionError("expected exactly one filled start room")
    corridors, adjacency = _detect_corridors(grid, boundary_owner, len(rooms))
    return rooms, corridors, adjacency, filled_rooms[0]


def _render_output(
    shape: tuple[int, int], rooms: list[dict], corridors: list[dict], visited_rooms: list[int], used_corridors: list[int]
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for eid in used_corridors:
        for r, c in corridors[eid]["cells"]:
            out[r, c] = CORRIDOR
    for rid in visited_rooms:
        r0, c0, r1, c1 = rooms[rid]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rooms[rid]["border_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the keyed corridor chain from the pre-filled room and erase every room and corridor outside that chain.

    Input structure:
    - A black background contains several odd-sized hollow rectangular rooms with distinct border colors.
    - Each room has a single center key pixel whose color matches some room border color, and one room is already filled except for that visible center key.
    - Grey width-1 straight corridor segments connect selected room pairs, creating a branching room graph with off-chain side rooms.

    Rule:
    - Identify the unique pre-filled start room, read its center key color, and look only among rooms directly connected to it by a grey corridor.
    - Move to the unique adjacent room whose border color equals that key color, and repeat the same lookup from the new current room using the input keys.
    - Stop when no adjacent room has the requested border color, then keep only the visited rooms and traversed corridors; fill every kept room solid with its border color and erase all other rooms and corridors.

    Key insight:
    - The key color is only half of the instruction: the next room is determined by the conjunction of color equality and direct corridor adjacency, so remote same-color decoys must be ignored and can even force the final halt.
    """
    g = np.asarray(grid, dtype=int)
    rooms, corridors, adjacency, start_id = _analyze_scene(g)
    visited_rooms, used_corridors = _trace_chain(rooms, adjacency, start_id)
    return _render_output(g.shape, rooms, corridors, visited_rooms, used_corridors)


def _node_neighbors(node: Node) -> list[Node]:
    r, c = node
    out: list[Node] = []
    if r > 0:
        out.append((r - 1, c))
    if r < 2:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c < 2:
        out.append((r, c + 1))
    return out


def _sample_chain(length: int) -> list[Node] | None:
    nodes = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        start = random.choice(nodes)
        path = [start]
        used = {start}
        while len(path) < length:
            tail = path[-1]
            cands = [nb for nb in _node_neighbors(tail) if nb not in used]
            if not cands:
                break
            cands.sort(
                key=lambda nb: sum(1 for nn in _node_neighbors(nb) if nn not in used),
                reverse=True,
            )
            pick = random.choice(cands[: max(1, min(2, len(cands)))])
            path.append(pick)
            used.add(pick)

        if len(path) != length:
            continue
        if _branch_options(path):
            return path
    return None


def _branch_options(chain: list[Node]) -> list[tuple[int, Node]]:
    used = set(chain)
    options: list[tuple[int, Node]] = []
    for idx, node in enumerate(chain[:-1]):
        for nb in _node_neighbors(node):
            if nb not in used:
                options.append((idx, nb))
    return options


def _select_branches(chain: list[Node]) -> list[tuple[int, Node]] | None:
    options = _branch_options(chain)
    if not options:
        return None

    max_count = min(MAX_BRANCHES, len(ROOM_COLORS) - len(chain), len({node for _, node in options}))
    if max_count < 1:
        return None
    count = random.randint(1, max_count)

    random.shuffle(options)
    chosen: list[tuple[int, Node]] = []
    used_nodes = set(chain)
    for attach_idx, node in options:
        if node in used_nodes:
            continue
        chosen.append((attach_idx, node))
        used_nodes.add(node)
        if len(chosen) == count:
            break

    if not chosen:
        return None
    return chosen


def _make_room_specs(chain: list[Node], branches: list[tuple[int, Node]]) -> tuple[dict[Node, dict], list[tuple[Node, Node]], Node]:
    nodes = list(chain) + [node for _, node in branches]
    border_colors = random.sample(ROOM_COLORS, len(nodes))
    room_specs: dict[Node, dict] = {}

    for node, border_color in zip(nodes, border_colors, strict=False):
        lr, lc = node
        h = random.choice(ROOM_SIDE_CHOICES)
        w = random.choice(ROOM_SIDE_CHOICES)
        cr = LATTICE_CENTERS[lr]
        cc = LATTICE_CENTERS[lc]
        room_specs[node] = {
            "node": node,
            "border_color": int(border_color),
            "bbox": (cr - h // 2, cc - w // 2, cr + h // 2, cc + w // 2),
            "filled": node == chain[0],
        }

    graph_edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    graph_edges.extend((chain[attach_idx], branch_node) for attach_idx, branch_node in branches)
    graph: dict[Node, set[Node]] = {node: set() for node in nodes}
    for a, b in graph_edges:
        graph[a].add(b)
        graph[b].add(a)

    decoy_branch = random.choice([node for _, node in branches])
    for idx, node in enumerate(chain):
        if idx + 1 < len(chain):
            room_specs[node]["key_color"] = room_specs[chain[idx + 1]]["border_color"]
        else:
            room_specs[node]["key_color"] = room_specs[decoy_branch]["border_color"]

    all_nodes = list(nodes)
    for _, node in branches:
        non_neighbors = [
            other
            for other in all_nodes
            if other != node and other not in graph[node]
        ]
        room_specs[node]["key_color"] = room_specs[random.choice(non_neighbors)]["border_color"]

    return room_specs, graph_edges, decoy_branch


def _corridor_cells(room_a: dict, room_b: dict) -> list[Coord]:
    r0a, c0a, r1a, c1a = room_a["bbox"]
    r0b, c0b, r1b, c1b = room_b["bbox"]
    cra = (r0a + r1a) // 2
    cca = (c0a + c1a) // 2
    crb = (r0b + r1b) // 2
    ccb = (c0b + c1b) // 2

    cells: list[Coord] = []
    if cra == crb:
        row = cra
        if cca < ccb:
            for c in range(c1a + 1, c0b):
                cells.append((row, c))
        else:
            for c in range(c1b + 1, c0a):
                cells.append((row, c))
    elif cca == ccb:
        col = cca
        if cra < crb:
            for r in range(r1a + 1, r0b):
                cells.append((r, col))
        else:
            for r in range(r1b + 1, r0a):
                cells.append((r, col))
    else:
        raise AssertionError("corridor endpoints must share a row or a column")

    if not cells:
        raise AssertionError("corridor length must be positive")
    return cells


def _draw_room_input(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    color = int(room["border_color"])
    key = int(room["key_color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if room["filled"]:
        grid[r0 + 1 : r1, c0 + 1 : c1] = color
    grid[(r0 + r1) // 2, (c0 + c1) // 2] = key


def _draw_room_output(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(room["border_color"])


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _nonzero_bbox(inp)
    top = random.randint(0, min(2, r0))
    left = random.randint(0, min(2, c0))
    bottom = random.randint(0, min(2, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(2, inp.shape[1] - 1 - c1))
    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _validate_generation_spec(
    room_specs: dict[Node, dict],
    graph_edges: list[tuple[Node, Node]],
    chain: list[Node],
    decoy_branch: Node,
) -> None:
    graph: dict[Node, set[Node]] = defaultdict(set)
    for a, b in graph_edges:
        graph[a].add(b)
        graph[b].add(a)

    branch_nodes = [node for node in room_specs if node not in chain]
    if not branch_nodes:
        raise AssertionError("expected at least one off-chain branch room")

    terminal = chain[-1]
    terminal_key = int(room_specs[terminal]["key_color"])
    terminal_adj_matches = [
        nb for nb in graph[terminal] if int(room_specs[nb]["border_color"]) == terminal_key
    ]
    if terminal_adj_matches:
        raise AssertionError("terminal room must halt because no adjacent room matches its key")

    remote_terminal_matches = [
        node
        for node in room_specs
        if node != terminal
        and node not in graph[terminal]
        and int(room_specs[node]["border_color"]) == terminal_key
    ]
    if remote_terminal_matches != [decoy_branch]:
        raise AssertionError("terminal key must point to one non-adjacent decoy branch room")

    for idx, node in enumerate(chain[:-1]):
        want = int(room_specs[node]["key_color"])
        matches = [nb for nb in graph[node] if int(room_specs[nb]["border_color"]) == want]
        if matches != [chain[idx + 1]]:
            raise AssertionError("each chain step must use a unique adjacent color match")

    if np.array_equal(
        np.array([room_specs[n]["border_color"] for n in chain], dtype=int),
        np.array([room_specs[n]["key_color"] for n in chain], dtype=int),
    ):
        raise AssertionError("keys should not collapse into a trivial identity pattern")


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    chain_len = random.choice(CHAIN_LEN_CHOICES)
    chain = _sample_chain(chain_len)
    if chain is None:
        return None

    branches = _select_branches(chain)
    if branches is None:
        return None

    room_specs, graph_edges, decoy_branch = _make_room_specs(chain, branches)
    _validate_generation_spec(room_specs, graph_edges, chain, decoy_branch)

    inp = np.full((27, 27), BG, dtype=int)
    out = np.full((27, 27), BG, dtype=int)

    for room in room_specs.values():
        _draw_room_input(inp, room)
    for a, b in graph_edges:
        cells = _corridor_cells(room_specs[a], room_specs[b])
        for r, c in cells:
            inp[r, c] = CORRIDOR
    for node in chain:
        _draw_room_output(out, room_specs[node])
    for idx in range(len(chain) - 1):
        cells = _corridor_cells(room_specs[chain[idx]], room_specs[chain[idx + 1]])
        for r, c in cells:
            out[r, c] = CORRIDOR

    inp, out = _crop_pair(inp, out)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None

    rooms, corridors, adjacency, start_id = _analyze_scene(inp)
    traced_rooms, traced_corridors = _trace_chain(rooms, adjacency, start_id)
    solved = _render_output(inp.shape, rooms, corridors, traced_rooms, traced_corridors)

    if len(room_specs) != len(rooms):
        return None
    if len(traced_rooms) != len(chain):
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(inp == CORRIDOR) == np.count_nonzero(out == CORRIDOR):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, out = attempt
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid corridor-keychain ARC input")
