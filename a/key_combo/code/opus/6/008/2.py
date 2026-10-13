from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
RED = 2
SEED_COLORS = (1, 2, 3, 4, 7)
DIRS = {
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
GRID_RANGE = (27, 29)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_lists(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _shell_of(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shell: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in cells:
                shell.add((nr, nc))
    return shell


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _port_candidates(
    cells: set[tuple[int, int]], side: str
) -> list[dict[str, tuple[int, int]]]:
    dr, dc = DIRS[side]
    shell = _shell_of(cells)
    candidates = []
    seen_gates = set()
    for r, c in cells:
        gate = (r + dr, c + dc)
        outer = (r + 2 * dr, c + 2 * dc)
        if gate not in shell or outer in cells or outer in shell:
            continue
        if sum((gr, gc) in cells for gr, gc in _neighbors4(*gate)) != 1:
            continue
        if gate in seen_gates:
            continue
        seen_gates.add(gate)
        candidates.append({"gate": gate, "outer": outer, "inner": (r, c)})
    candidates.sort(key=lambda item: (item["gate"][0], item["gate"][1]))
    return candidates


def _double_port_candidates(cells: set[tuple[int, int]], side: str) -> list[dict[str, tuple[int, int] | list[tuple[int, int]]]]:
    singles = _port_candidates(cells, side)
    pairs = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            a = singles[i]
            b = singles[j]
            ga = a["gate"]
            gb = b["gate"]
            if abs(ga[0] - gb[0]) + abs(ga[1] - gb[1]) != 1:
                continue
            oa = a["outer"]
            ob = b["outer"]
            if abs(oa[0] - ob[0]) + abs(oa[1] - ob[1]) != 1:
                continue
            pairs.append(
                {
                    "gates": [ga, gb],
                    "outers": [oa, ob],
                }
            )
    return pairs


def _choose_seed_cell(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = []
    for cell in cells:
        remaining = set(cells)
        remaining.remove(cell)
        if _is_connected(remaining):
            degree = sum(nei in cells for nei in _neighbors4(*cell))
            candidates.append((degree, -abs(cell[0]), -abs(cell[1]), cell))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1]


def _random_room_shape(
    required_sides: set[str], double_side: str | None
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(120):
        h = random.randint(3, 4)
        w = random.randint(4, 5)
        cells = {(r, c) for r in range(h) for c in range(w)}

        if random.random() < 0.9:
            side = random.choice(tuple(DIRS))
            if side in ("up", "down"):
                length = random.randint(1, min(2, w))
                start = random.randint(0, w - length)
                rr = -1 if side == "up" else h
                for cc in range(start, start + length):
                    cells.add((rr, cc))
            else:
                length = random.randint(1, min(2, h))
                start = random.randint(0, h - length)
                cc = -1 if side == "left" else w
                for rr in range(start, start + length):
                    cells.add((rr, cc))

        if random.random() < 0.5 and len(cells) > 12:
            removable = list(cells)
            random.shuffle(removable)
            for cell in removable:
                trial = set(cells)
                trial.remove(cell)
                if len(trial) >= 10 and _is_connected(trial):
                    cells = trial
                    break

        cells = _normalize_cells(cells)
        if not 10 <= len(cells) <= 40:
            continue

        if any(not _port_candidates(cells, side) for side in required_sides):
            continue
        if double_side is not None and not _double_port_candidates(cells, double_side):
            continue

        seed = _choose_seed_cell(cells)
        if seed is None:
            continue
        return cells, seed
    return None


def _slot_neighbors(slot: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = slot
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            out.append((nr, nc))
    return out


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == 1 and dc == 0:
        return "down"
    if dr == -1 and dc == 0:
        return "up"
    if dr == 0 and dc == 1:
        return "right"
    if dr == 0 and dc == -1:
        return "left"
    raise ValueError("slots must be 4-neighbors")


def _random_chain(length: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        cur = (random.randint(0, 2), random.randint(0, 2))
        chain = [cur]
        used = {cur}
        while len(chain) < length:
            choices = [n for n in _slot_neighbors(cur) if n not in used]
            if not choices:
                break
            cur = random.choice(choices)
            chain.append(cur)
            used.add(cur)
        if len(chain) == length:
            return chain
    return None


def _layout_slots():
    for _ in range(120):
        chain_len = random.randint(4, 5)
        chain = _random_chain(chain_len)
        if chain is None:
            continue
        chain_set = set(chain)
        candidates = []
        for slot in [(r, c) for r in range(3) for c in range(3)]:
            if slot in chain_set:
                continue
            parents = [idx for idx, src in enumerate(chain) if slot in _slot_neighbors(src)]
            if parents:
                candidates.append((slot, parents))
        if len(candidates) < 2:
            continue
        dead_count = random.randint(2, min(3, len(candidates), 8 - chain_len))
        chosen = random.sample(candidates, dead_count)
        dead_slots = [slot for slot, _ in chosen]
        attachments = {slot: random.choice(parents) for slot, parents in chosen}
        return chain, dead_slots, attachments
    return None


def _placement_ranges(
    slot: tuple[int, int], shape_cells: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[range, range] | None:
    h, w = shape
    max_r = max(r for r, _ in shape_cells)
    max_c = max(c for _, c in shape_cells)
    row_edges = np.linspace(0, h, 4, dtype=int)
    col_edges = np.linspace(0, w, 4, dtype=int)
    sr, sc = slot
    slot_r0, slot_r1 = int(row_edges[sr]), int(row_edges[sr + 1]) - 1
    slot_c0, slot_c1 = int(col_edges[sc]), int(col_edges[sc + 1]) - 1
    r_lo = max(2, slot_r0 + 1)
    c_lo = max(2, slot_c0 + 1)
    r_hi = min(h - 3 - max_r, slot_r1 - 1 - max_r)
    c_hi = min(w - 3 - max_c, slot_c1 - 1 - max_c)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _bfs_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors4(*cur):
            if not _in_bounds(shape, *nxt) or nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    assert bg == BG

    cavity_comps = [
        comp for comp in _component_lists(g == BG) if not _touches_border(comp, g.shape)
    ]
    assert 6 <= len(cavity_comps) <= 10
    cavity_id = np.full((h, w), -1, dtype=int)
    cavities = []
    for idx, comp in enumerate(cavity_comps):
        for r, c in comp:
            cavity_id[r, c] = idx
        cavities.append(
            {
                "coords": comp,
                "coord_set": set(comp),
                "seed_color": None,
                "seed_pos": None,
                "paths": [],
            }
        )

    comps = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, WALL)):
        for comp in _component_lists(g == color):
            comps.append({"color": color, "coords": comp})

    paths = []
    for comp in comps:
        coords = comp["coords"]
        color = int(comp["color"])
        adj_cavities = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc) and cavity_id[nr, nc] >= 0:
                    adj_cavities.add(int(cavity_id[nr, nc]))

        if len(coords) == 1:
            assert len(adj_cavities) == 1
            cid = next(iter(adj_cavities))
            assert cavities[cid]["seed_color"] is None
            cavities[cid]["seed_color"] = color
            cavities[cid]["seed_pos"] = coords[0]
            continue

        assert len(adj_cavities) == 2
        path_idx = len(paths)
        path = {
            "color": color,
            "coords": coords,
            "coord_set": set(coords),
            "cavities": tuple(sorted(adj_cavities)),
        }
        paths.append(path)
        for cid in path["cavities"]:
            cavities[cid]["paths"].append(path_idx)

    assert all(cav["seed_color"] in SEED_COLORS for cav in cavities)

    start_candidates = []
    for cid, cav in enumerate(cavities):
        for path_idx in cav["paths"]:
            path = paths[path_idx]
            touching = []
            for r, c in path["coords"]:
                if any(
                    _in_bounds(g.shape, nr, nc) and cavity_id[nr, nc] == cid
                    for nr, nc in _neighbors4(r, c)
                ):
                    touching.append((r, c))
            if len(touching) < 2:
                continue
            touching_set = set(touching)
            if any(nei in touching_set for cell in touching for nei in _neighbors4(*cell)):
                start_candidates.append(cid)
                break
    start_candidates = sorted(set(start_candidates))
    assert len(start_candidates) == 1
    start = start_candidates[0]
    assert cavities[start]["seed_color"] == RED

    return cavities, paths, start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade-fill color-seeded cavities by following guide paths whose color must match the destination seed.

    Input structure:
    - The background is black, grey walls enclose several irregular black cavities, and each cavity contains exactly one singleton seed color from {blue, red, green, yellow, orange}.
    - Colored guide paths run through wall gaps between pairs of cavities; one special red-seeded cavity is marked by a unique double-width gap.

    Rule:
    - Find the cavity with the double-width gap and fill its black interior with red while keeping its seed unchanged.
    - From every filled cavity, inspect each incident guide path, identify the cavity at the far end, and compare the path color with that destination cavity's seed color.
    - Only when the colors match, fill that destination cavity's black interior with its seed color, then continue the same process from the newly filled cavity.
    - Leave all walls, guide paths, seed pixels, and every never-activated cavity unchanged.

    Key insight:
    - The paths encode both connectivity and activation permission: topology alone is insufficient because propagation only crosses into cavities whose seed color matches the path color.
    """
    g = np.asarray(grid, dtype=int)
    cavities, paths, start = _parse_scene(g)

    active = {start}
    queue = deque([start])
    while queue:
        cid = queue.popleft()
        for path_idx in cavities[cid]["paths"]:
            a, b = paths[path_idx]["cavities"]
            other = b if a == cid else a
            if other in active:
                continue
            if int(paths[path_idx]["color"]) == int(cavities[other]["seed_color"]):
                active.add(other)
                queue.append(other)

    out = g.copy()
    for cid in active:
        fill_color = int(cavities[cid]["seed_color"])
        for r, c in cavities[cid]["coords"]:
            out[r, c] = fill_color
    return out


def _expected_from_metadata(
    grid: np.ndarray, rooms: list[dict], edges: list[dict]
) -> tuple[np.ndarray, set[int], dict[int, int], int]:
    active = {0}
    depth = {0: 0}
    queue = deque([0])
    dead_edges = 0
    while queue:
        src = queue.popleft()
        for edge in edges:
            if src not in (edge["a"], edge["b"]):
                continue
            dst = edge["b"] if edge["a"] == src else edge["a"]
            if dst in active:
                continue
            if edge["color"] == rooms[dst]["seed_color"]:
                active.add(dst)
                depth[dst] = depth[src] + 1
                queue.append(dst)
            else:
                dead_edges += 1

    out = grid.copy()
    for rid in active:
        for r, c in rooms[rid]["interior"]:
            if (r, c) != rooms[rid]["seed_pos"]:
                out[r, c] = rooms[rid]["seed_color"]
    return out, active, depth, dead_edges


def _route_edge(
    grid: np.ndarray,
    rooms: list[dict],
    edge: dict,
    path_cells: set[tuple[int, int]],
    room_reserved: list[set[tuple[int, int]]],
) -> bool:
    a = edge["a"]
    b = edge["b"]
    side_a = edge["side_a"]
    side_b = edge["side_b"]
    room_a = rooms[a]
    room_b = rooms[b]

    if edge.get("double_start"):
        src_options = []
        for item in room_a["double_ports"][side_a]:
            gates = [tuple(item["gates"][0]), tuple(item["gates"][1])]
            outers = [tuple(item["outers"][0]), tuple(item["outers"][1])]
            if any(cell in room_reserved[a] for cell in gates + outers):
                continue
            src_options.append({"gates": gates, "outers": outers})
    else:
        src_options = []
        for item in room_a["ports"][side_a]:
            gate = tuple(item["gate"])
            outer = tuple(item["outer"])
            if gate in room_reserved[a] or outer in room_reserved[a]:
                continue
            src_options.append({"gates": [gate], "outers": [outer]})

    dst_options = []
    for item in room_b["ports"][side_b]:
        gate = tuple(item["gate"])
        outer = tuple(item["outer"])
        if gate in room_reserved[b] or outer in room_reserved[b]:
            continue
        dst_options.append({"gates": [gate], "outers": [outer]})

    random.shuffle(src_options)
    random.shuffle(dst_options)
    shape = grid.shape

    for src in src_options:
        for dst in dst_options:
            start_outer = min(
                src["outers"],
                key=lambda cell: abs(cell[0] - dst["outers"][0][0]) + abs(cell[1] - dst["outers"][0][1]),
            )
            end_outer = dst["outers"][0]
            allowances = set(src["gates"] + src["outers"] + dst["gates"] + dst["outers"])
            blocked = set()
            for r in range(shape[0]):
                for c in range(shape[1]):
                    if grid[r, c] != BG and (r, c) not in allowances:
                        blocked.add((r, c))
            for cell in path_cells:
                for nbr in _neighbors8(*cell):
                    if _in_bounds(shape, *nbr) and nbr not in allowances:
                        blocked.add(nbr)
                if cell not in allowances:
                    blocked.add(cell)

            path = _bfs_path(shape, blocked, start_outer, end_outer)
            if path is None:
                continue

            full = set(path) | set(src["gates"]) | set(src["outers"]) | set(dst["gates"]) | set(dst["outers"])
            if any(not _in_bounds(shape, *cell) for cell in full):
                continue
            if any(grid[r, c] != BG and cell not in allowances for cell in full for r, c in [cell]):
                continue

            for r, c in full:
                grid[r, c] = edge["color"]
            path_cells.update(full)
            room_reserved[a].update(src["gates"] + src["outers"])
            room_reserved[b].update(dst["gates"] + dst["outers"])
            edge["cells"] = full
            return True
    return False


def _build_candidate_rooms(
    h: int,
    w: int,
    chain: list[tuple[int, int]],
    dead_slots: list[tuple[int, int]],
    attachments: dict[tuple[int, int], int],
) -> tuple[list[dict], list[dict]] | None:
    room_slots = chain + dead_slots
    slot_to_room = {slot: idx for idx, slot in enumerate(room_slots)}
    edges = []
    for idx in range(len(chain) - 1):
        a = idx
        b = idx + 1
        side_a = _dir_between(chain[idx], chain[idx + 1])
        edges.append(
            {
                "a": a,
                "b": b,
                "side_a": side_a,
                "side_b": OPPOSITE[side_a],
                "double_start": idx == 0,
                "kind": "chain",
            }
        )
    for slot in dead_slots:
        a = attachments[slot]
        b = slot_to_room[slot]
        side_a = _dir_between(chain[a], slot)
        edges.append(
            {
                "a": a,
                "b": b,
                "side_a": side_a,
                "side_b": OPPOSITE[side_a],
                "double_start": False,
                "kind": "dead",
            }
        )

    room_incident: list[set[str]] = [set() for _ in room_slots]
    for edge in edges:
        room_incident[edge["a"]].add(edge["side_a"])
        room_incident[edge["b"]].add(edge["side_b"])

    rooms = []
    for rid, slot in enumerate(room_slots):
        double_side = edges[0]["side_a"] if rid == 0 else None
        shape_data = _random_room_shape(room_incident[rid], double_side)
        if shape_data is None:
            return None
        local_cells, local_seed = shape_data
        ranges = _placement_ranges(slot, local_cells, (h, w))
        if ranges is None:
            return None
        r_choices, c_choices = ranges
        placed = None
        options = [(r0, c0) for r0 in r_choices for c0 in c_choices]
        random.shuffle(options)
        for r0, c0 in options:
            global_cells = {(r + r0, c + c0) for r, c in local_cells}
            shell = _shell_of(global_cells)
            if any(not _in_bounds((h, w), r, c) for r, c in shell):
                continue
            placed = (global_cells, shell, (local_seed[0] + r0, local_seed[1] + c0))
            break
        if placed is None:
            return None

        global_cells, shell, seed_pos = placed
        ports = {side: [] for side in DIRS}
        double_ports = {side: [] for side in DIRS}
        for side in DIRS:
            for item in _port_candidates(local_cells, side):
                ports[side].append(
                    {
                        "gate": (item["gate"][0] + r0, item["gate"][1] + c0),
                        "outer": (item["outer"][0] + r0, item["outer"][1] + c0),
                    }
                )
            for item in _double_port_candidates(local_cells, side):
                double_ports[side].append(
                    {
                        "gates": [
                            (item["gates"][0][0] + r0, item["gates"][0][1] + c0),
                            (item["gates"][1][0] + r0, item["gates"][1][1] + c0),
                        ],
                        "outers": [
                            (item["outers"][0][0] + r0, item["outers"][0][1] + c0),
                            (item["outers"][1][0] + r0, item["outers"][1][1] + c0),
                        ],
                    }
                )

        rooms.append(
            {
                "slot": slot,
                "interior": global_cells,
                "shell": shell,
                "seed_pos": seed_pos,
                "ports": ports,
                "double_ports": double_ports,
            }
        )

    occupied = set()
    for room in rooms:
        footprint = room["interior"] | room["shell"]
        if occupied & footprint:
            return None
        occupied |= footprint

    return rooms, edges


def _decorate_and_validate(
    h: int, w: int, rooms: list[dict], edges: list[dict]
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)

    seeds = [RED]
    for _ in range(1, len(rooms)):
        seeds.append(random.choice(SEED_COLORS))
    for rid, room in enumerate(rooms):
        room["seed_color"] = seeds[rid]

    for edge in edges:
        if edge["kind"] == "chain":
            edge["color"] = rooms[edge["b"]]["seed_color"]
        else:
            choices = [color for color in SEED_COLORS if color != rooms[edge["b"]]["seed_color"]]
            edge["color"] = random.choice(choices)

    for room in rooms:
        for r, c in room["shell"]:
            grid[r, c] = WALL
        sr, sc = room["seed_pos"]
        grid[sr, sc] = room["seed_color"]

    path_cells: set[tuple[int, int]] = set()
    room_reserved = [set() for _ in rooms]
    route_order = sorted(edges, key=lambda edge: (not edge.get("double_start", False), edge["a"], edge["b"]))
    for edge in route_order:
        if not _route_edge(grid, rooms, edge, path_cells, room_reserved):
            return None

    expected, active, depth, dead_edges = _expected_from_metadata(grid, rooms, edges)
    if len(active) < 4 or dead_edges < 2 or max(depth.values()) < 3:
        return None

    topo_active = {0}
    topo_queue = deque([0])
    while topo_queue:
        src = topo_queue.popleft()
        for edge in edges:
            if src not in (edge["a"], edge["b"]):
                continue
            dst = edge["b"] if edge["a"] == src else edge["a"]
            if dst not in topo_active:
                topo_active.add(dst)
                topo_queue.append(dst)
    if len(topo_active) <= len(active):
        return None

    direct = {0}
    for edge in edges:
        if 0 not in (edge["a"], edge["b"]):
            continue
        dst = edge["b"] if edge["a"] == 0 else edge["a"]
        if edge["color"] == rooms[dst]["seed_color"]:
            direct.add(dst)
    if len(direct) >= len(active):
        return None

    try:
        parsed_cavities, parsed_paths, start = _parse_scene(grid)
    except Exception:
        return None
    if start != 0 or len(parsed_cavities) != len(rooms) or len(parsed_paths) != len(edges):
        return None

    out = main(grid)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_RANGE)
        w = random.randint(*GRID_RANGE)
        slot_layout = _layout_slots()
        if slot_layout is None:
            continue
        chain, dead_slots, attachments = slot_layout
        built = _build_candidate_rooms(h, w, chain, dead_slots, attachments)
        if built is None:
            continue
        rooms, edges = built
        grid = _decorate_and_validate(h, w, rooms, edges)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade cavity puzzle")
