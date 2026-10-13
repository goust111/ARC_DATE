from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
GEN_WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (2, 4)
ROOM_SIZE_RANGE = (3, 5)
GEN_RETRIES = 400


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            label = len(components)
            queue = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return labels, components


def _region_key(coords: np.ndarray) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in coords)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BACKGROUND if np.any(g == BACKGROUND) else _mode_color(g)
    colors = [int(color) for color in np.unique(g)]
    wall_candidates = [color for color in colors if color != bg and int(np.count_nonzero(g == color)) != 2]
    assert len(wall_candidates) == 1, "expected one wall color plus portal colors that each appear twice"
    wall = wall_candidates[0]

    passable = g != wall
    region_labels, region_cells = _label_components(passable)
    assert region_cells, "expected at least one non-wall region"

    region_keys = [_region_key(coords) for coords in region_cells]
    adjacency = {idx: set() for idx in range(len(region_cells))}
    region_portals: dict[int, list[tuple[tuple[int, int], int]]] = {idx: [] for idx in range(len(region_cells))}
    seeds: list[tuple[tuple[int, int], int, int]] = []

    portal_colors = [color for color in colors if color not in {bg, wall}]
    assert portal_colors, "expected at least one portal pair"
    for color in portal_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each portal color must appear exactly twice"
        first = (int(coords[0, 0]), int(coords[0, 1]))
        second = (int(coords[1, 0]), int(coords[1, 1]))
        region_a = int(region_labels[first])
        region_b = int(region_labels[second])
        assert region_a >= 0 and region_b >= 0
        assert region_a != region_b, "portal endpoints must occupy different regions"

        adjacency[region_a].add(region_b)
        adjacency[region_b].add(region_a)
        region_portals[region_a].append((first, int(color)))
        region_portals[region_b].append((second, int(color)))
        seeds.append((first, region_a, int(color)))
        seeds.append((second, region_b, int(color)))

    direct_fill: dict[int, int] = {}
    for region, entries in region_portals.items():
        if not entries:
            continue
        # If a region hosts multiple portal colors, use the earliest portal cell as its local owner.
        entries.sort(key=lambda item: item[0])
        direct_fill[region] = int(entries[0][1])

    return {
        "grid": g,
        "bg": int(bg),
        "wall": int(wall),
        "region_labels": region_labels,
        "region_cells": region_cells,
        "region_keys": region_keys,
        "adjacency": {region: sorted(neighbors, key=lambda idx: region_keys[idx]) for region, neighbors in adjacency.items()},
        "direct_fill": direct_fill,
        "seeds": sorted(seeds, key=lambda item: item[0]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat same-color pixel pairs as portal links between grey-walled regions, then fill each reachable region
    with the winning portal color while preserving the wall and portal pixels themselves.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = info["bg"]
    region_labels = info["region_labels"]
    adjacency = info["adjacency"]
    direct_fill = info["direct_fill"]
    seeds = info["seeds"]

    reachable_fill: dict[int, int] = {}
    seen_states: set[tuple[int, int]] = set()
    queue = deque()
    for _coord, region, color in seeds:
        state = (int(region), int(color))
        if state in seen_states:
            continue
        seen_states.add(state)
        queue.append(state)

    while queue:
        region, color = queue.popleft()
        if region not in direct_fill and region not in reachable_fill:
            reachable_fill[region] = int(color)
        for neighbor in adjacency[region]:
            state = (int(neighbor), int(color))
            if state in seen_states:
                continue
            seen_states.add(state)
            queue.append(state)

    out = g.copy()
    final_fill = {**reachable_fill, **direct_fill}
    for region, color in final_fill.items():
        mask = (region_labels == int(region)) & (g == bg)
        out[mask] = int(color)
    return out


def _build_room_grid() -> tuple[np.ndarray, list[dict]]:
    room_rows = random.randint(*ROOM_ROW_RANGE)
    room_cols = random.randint(*ROOM_COL_RANGE)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    height = sum(heights) + room_rows + 1
    width = sum(widths) + room_cols + 1
    grid = np.full((height, width), BACKGROUND, dtype=int)

    boundary_rows = [0]
    cursor = 0
    for room_h in heights:
        cursor += room_h + 1
        boundary_rows.append(cursor)
    boundary_cols = [0]
    cursor = 0
    for room_w in widths:
        cursor += room_w + 1
        boundary_cols.append(cursor)

    for row in boundary_rows:
        grid[row, :] = GEN_WALL
    for col in boundary_cols:
        grid[:, col] = GEN_WALL

    rooms: list[dict] = []
    top = 1
    room_id = 0
    for rr, room_h in enumerate(heights):
        left = 1
        for cc, room_w in enumerate(widths):
            rooms.append(
                {
                    "id": room_id,
                    "bbox": (top, left, top + room_h - 1, left + room_w - 1),
                    "grid_pos": (rr, cc),
                }
            )
            room_id += 1
            left += room_w + 1
        top += room_h + 1
    return grid, rooms


def _scenario_pairs(room_ids: list[int]) -> list[tuple[int, int]]:
    pair_cap = max(1, min(3, len(room_ids) // 2))
    pair_count = random.randint(1 if pair_cap == 1 else 2, pair_cap)
    chosen = random.sample(room_ids, 2 * pair_count)
    random.shuffle(chosen)
    return [(chosen[i], chosen[i + 1]) for i in range(0, len(chosen), 2)]


def _scenario_chain(room_ids: list[int]) -> list[tuple[int, int]]:
    edge_count = random.randint(2, min(4, len(room_ids) - 1))
    seq = random.sample(room_ids, edge_count + 1)
    return [(seq[i], seq[i + 1]) for i in range(edge_count)]


def _scenario_star(room_ids: list[int]) -> list[tuple[int, int]]:
    leaf_count = random.randint(2, min(4, len(room_ids) - 1))
    chosen = random.sample(room_ids, leaf_count + 1)
    center = chosen[0]
    return [(center, leaf) for leaf in chosen[1:]]


def _scenario_cycle(room_ids: list[int]) -> list[tuple[int, int]]:
    node_count = random.randint(4, min(6, len(room_ids)))
    seq = random.sample(room_ids, node_count)
    return [(seq[i], seq[(i + 1) % node_count]) for i in range(node_count)]


def _scenario_mixed(room_ids: list[int]) -> list[tuple[int, int]]:
    chain_nodes = 4 if len(room_ids) >= 6 and random.random() < 0.5 else 3
    chosen = random.sample(room_ids, chain_nodes + 2)
    chain = chosen[:chain_nodes]
    pair = chosen[chain_nodes:]
    edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    edges.append((pair[0], pair[1]))
    return edges


def _pick_edges(room_ids: list[int]) -> list[tuple[int, int]]:
    builders = [("pairs", _scenario_pairs)]
    if len(room_ids) >= 3:
        builders.extend([("chain", _scenario_chain), ("star", _scenario_star)])
    if len(room_ids) >= 4:
        builders.append(("cycle", _scenario_cycle))
    if len(room_ids) >= 5:
        builders.append(("mixed", _scenario_mixed))
    _name, builder = random.choice(builders)
    return builder(room_ids)


def _side_candidates(room: dict) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = room["bbox"]
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _assign_portal_cells(rooms: list[dict], edges: list[tuple[int, int]]) -> dict[int, dict[int, tuple[int, int]]]:
    room_map = {room["id"]: room for room in rooms}
    incident: dict[int, list[int]] = {room["id"]: [] for room in rooms}
    for edge_id, (a, b) in enumerate(edges):
        incident[a].append(edge_id)
        incident[b].append(edge_id)

    assignments: dict[int, dict[int, tuple[int, int]]] = {room["id"]: {} for room in rooms}
    for room_id, edge_ids in incident.items():
        if not edge_ids:
            continue
        candidates = _side_candidates(room_map[room_id])
        usable_sides = [side for side, coords in candidates.items() if coords]
        assert len(edge_ids) <= len(usable_sides), "room needs more distinct wall-adjacent portal cells than available sides"
        chosen_sides = random.sample(usable_sides, len(edge_ids))
        random.shuffle(edge_ids)
        for edge_id, side in zip(edge_ids, chosen_sides):
            assignments[room_id][edge_id] = random.choice(candidates[side])
    return assignments


def _gen_attempt() -> np.ndarray | None:
    grid, rooms = _build_room_grid()
    room_ids = [room["id"] for room in rooms]
    edges = _pick_edges(room_ids)
    if len(edges) > len(COLOR_POOL):
        return None

    degrees: dict[int, int] = {room_id: 0 for room_id in room_ids}
    for a, b in edges:
        if a == b:
            return None
        degrees[a] += 1
        degrees[b] += 1
    if any(degree > 4 for degree in degrees.values()):
        return None

    assignments = _assign_portal_cells(rooms, edges)
    colors = random.sample(COLOR_POOL, len(edges))
    for edge_id, (room_a, room_b) in enumerate(edges):
        color = colors[edge_id]
        coord_a = assignments[room_a][edge_id]
        coord_b = assignments[room_b][edge_id]
        if coord_a == coord_b:
            return None
        if grid[coord_a] != BACKGROUND or grid[coord_b] != BACKGROUND:
            return None
        grid[coord_a] = color
        grid[coord_b] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-region input")
