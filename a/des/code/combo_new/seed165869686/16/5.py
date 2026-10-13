from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SCENARIOS = (
    frozenset({1, 2}),
    frozenset({1, 2, 3}),
    frozenset({1, 2, 4}),
    frozenset({1, 2, 3, 4}),
)

MIN_ROOM_SIZE = 3
MAX_ROOM_SIZE = 6
MAX_GENERATION_TRIES = 400


def _border_majority_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = [int(v) for v in grid[0]]
    if h > 1:
        border.extend(int(v) for v in grid[-1])
    if w > 1:
        border.extend(int(v) for v in grid[1:-1, 0])
        border.extend(int(v) for v in grid[1:-1, -1])
    return Counter(border).most_common(1)[0][0]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: dict[int, list[tuple[int, int]]] = {}
    next_label = 1

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            components[next_label] = coords
            next_label += 1
    return labels, components


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _find_doors(grid: np.ndarray, wall_color: int) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    doors: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == wall_color:
                continue
            vertical = (
                int(grid[r - 1, c]) == wall_color
                and int(grid[r + 1, c]) == wall_color
                and int(grid[r, c - 1]) != wall_color
                and int(grid[r, c + 1]) != wall_color
            )
            horizontal = (
                int(grid[r, c - 1]) == wall_color
                and int(grid[r, c + 1]) == wall_color
                and int(grid[r - 1, c]) != wall_color
                and int(grid[r + 1, c]) != wall_color
            )
            if vertical ^ horizontal:
                doors[(int(r), int(c))] = "v" if vertical else "h"
    return doors


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, dict[tuple[int, int], str], dict[int, list[tuple[int, int]]], dict[tuple[int, int], tuple[int, int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_color = _border_majority_color(g)
    door_orientations = _find_doors(g, wall_color)

    sealed = g.copy()
    for r, c in door_orientations:
        sealed[r, c] = wall_color

    labels, components = _label_components(sealed != wall_color)
    rooms = {
        label: coords
        for label, coords in components.items()
        if not _touches_edge(coords, h, w)
    }

    room_doors: dict[int, set[tuple[int, int]]] = {label: set() for label in rooms}
    door_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for (r, c), orient in door_orientations.items():
        if orient == "v":
            left = int(labels[r, c - 1])
            right = int(labels[r, c + 1])
            a, b = left, right
        else:
            up = int(labels[r - 1, c])
            down = int(labels[r + 1, c])
            a, b = up, down
        if a in rooms and b in rooms and a != b:
            door_pairs[(r, c)] = (a, b)
            room_doors[a].add((r, c))
            room_doors[b].add((r, c))

    degrees = {label: len(room_doors[label]) for label in rooms}
    return wall_color, door_orientations, rooms, door_pairs, degrees


def _fill_color_for_degree(degree: int) -> int:
    if degree == 1:
        return RED
    if degree == 2:
        return BLUE
    if degree == 3:
        return GREEN
    if degree >= 4:
        return YELLOW
    return BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room by its door count, then mark dead-end doors in orange.

    Input structure:
    - Black background cells form room interiors and single-pixel door gaps.
    - Grey pixels form a width-1 orthogonal wall network, including a closed grey border.
    - Each room is a rectangular enclosed background region, and each door is a one-cell gap in a wall between two adjacent rooms.

    Rule:
    - Identify every room by temporarily closing the wall-gap door pixels and flood-filling the enclosed background regions.
    - Count how many detected door gaps touch each room boundary.
    - Fill rooms with red for 1 door, blue for 2 doors, green for 3 doors, and yellow for 4 or more doors.
    - Leave walls grey, keep ordinary doors black, and repaint a door orange whenever at least one of its adjacent rooms is a red dead-end room.

    Key insight:
    - The door pixels are background-colored holes inside grey walls, so the puzzle is solved by detecting those single-cell wall gaps first, sealing them to recover the room components, and only then counting room-door adjacencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_color, _, rooms, door_pairs, degrees = _extract_structure(g)
    out = g.copy()

    for label, coords in rooms.items():
        fill_color = _fill_color_for_degree(int(degrees.get(label, 0)))
        for r, c in coords:
            out[r, c] = fill_color

    for (r, c), (a, b) in door_pairs.items():
        if int(degrees.get(a, 0)) == 1 or int(degrees.get(b, 0)) == 1:
            out[r, c] = ORANGE
        else:
            out[r, c] = BLACK

    out[g == wall_color] = wall_color
    return out


def _sample_axis_sizes(rng: random.Random, count: int) -> list[int]:
    while True:
        sizes = [rng.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(count)]
        total = sum(sizes) + count + 1
        if total <= 30:
            return sizes


def _axis_ranges(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges: list[tuple[int, int]] = []
    walls = [0]
    cursor = 1
    for size in sizes:
        start = cursor
        end = cursor + size - 1
        ranges.append((start, end))
        walls.append(end + 1)
        cursor = end + 2
    return ranges, walls, walls[-1] + 1


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _build_adjacency(nrows: int, ncols: int) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:
    adjacency = {idx: [] for idx in range(nrows * ncols)}
    edges: list[tuple[int, int]] = []
    for r in range(nrows):
        for c in range(ncols):
            node = r * ncols + c
            if c + 1 < ncols:
                other = node + 1
                adjacency[node].append(other)
                adjacency[other].append(node)
                edges.append((node, other))
            if r + 1 < nrows:
                other = node + ncols
                adjacency[node].append(other)
                adjacency[other].append(node)
                edges.append((node, other))
    return adjacency, edges


def _random_spanning_tree(adjacency: dict[int, list[int]], rng: random.Random) -> set[tuple[int, int]]:
    start = rng.choice(list(adjacency))
    visited = {start}
    stack = [start]
    chosen: set[tuple[int, int]] = set()

    while stack:
        node = stack[-1]
        options = [nbr for nbr in adjacency[node] if nbr not in visited]
        if not options:
            stack.pop()
            continue
        nbr = rng.choice(options)
        chosen.add(_edge_key(node, nbr))
        visited.add(nbr)
        stack.append(nbr)
    return chosen


def _degrees_for_edges(num_nodes: int, edges: set[tuple[int, int]]) -> dict[int, int]:
    degrees = {node: 0 for node in range(num_nodes)}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees


def _degree_categories(degrees: dict[int, int]) -> set[int]:
    cats: set[int] = set()
    for degree in degrees.values():
        if degree >= 4:
            cats.add(4)
        elif degree >= 1:
            cats.add(degree)
    return cats


def _choose_door_edges(
    nrows: int,
    ncols: int,
    adjacency: dict[int, list[int]],
    all_edges: list[tuple[int, int]],
    scenario: frozenset[int],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    num_nodes = nrows * ncols
    chosen = _random_spanning_tree(adjacency, rng)

    if 4 in scenario:
        hubs = [node for node, nbrs in adjacency.items() if len(nbrs) >= 4]
        if not hubs:
            return None
        hub = rng.choice(hubs)
        for nbr in adjacency[hub]:
            chosen.add(_edge_key(hub, nbr))

    if 3 in scenario:
        candidates = [node for node, nbrs in adjacency.items() if len(nbrs) >= 3]
        rng.shuffle(candidates)
        for node in candidates:
            missing = [nbr for nbr in adjacency[node] if _edge_key(node, nbr) not in chosen]
            rng.shuffle(missing)
            while _degrees_for_edges(num_nodes, chosen)[node] < 3 and missing:
                chosen.add(_edge_key(node, missing.pop()))
            if _degrees_for_edges(num_nodes, chosen)[node] >= 3:
                break

    degrees = _degrees_for_edges(num_nodes, chosen)
    if 2 in scenario and 2 not in degrees.values():
        candidates = [node for node, nbrs in adjacency.items() if len(nbrs) >= 2]
        rng.shuffle(candidates)
        for node in candidates:
            if degrees[node] >= 2:
                continue
            missing = [nbr for nbr in adjacency[node] if _edge_key(node, nbr) not in chosen]
            rng.shuffle(missing)
            while degrees[node] < 2 and missing:
                chosen.add(_edge_key(node, missing.pop()))
                degrees = _degrees_for_edges(num_nodes, chosen)
            if 2 in degrees.values():
                break

    remaining = [edge for edge in all_edges if edge not in chosen]
    rng.shuffle(remaining)
    extra_prob = rng.uniform(0.08, 0.22)
    for edge in remaining:
        if rng.random() < extra_prob:
            chosen.add(edge)

    degrees = _degrees_for_edges(num_nodes, chosen)
    if 1 not in degrees.values():
        return None
    if any(degree == 0 for degree in degrees.values()):
        return None
    if not scenario.issubset(_degree_categories(degrees)):
        return None
    return chosen


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    door_edges: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    row_ranges, wall_rows, height = _axis_ranges(row_sizes)
    col_ranges, wall_cols, width = _axis_ranges(col_sizes)
    grid = np.full((height, width), BLACK, dtype=int)

    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY

    ncols = len(col_sizes)
    for a, b in door_edges:
        ra, ca = divmod(a, ncols)
        rb, cb = divmod(b, ncols)
        if ra == rb:
            room_r0, room_r1 = row_ranges[ra]
            wall_c = col_ranges[min(ca, cb)][1] + 1
            door_r = rng.randint(room_r0 + 1, room_r1 - 1)
            grid[door_r, wall_c] = BLACK
        else:
            room_c0, room_c1 = col_ranges[ca]
            wall_r = row_ranges[min(ra, rb)][1] + 1
            door_c = rng.randint(room_c0 + 1, room_c1 - 1)
            grid[wall_r, door_c] = BLACK

    return grid


def _choose_room_grid(rng: random.Random, scenario: frozenset[int]) -> tuple[int, int]:
    for _ in range(100):
        if 4 in scenario:
            nrows = rng.randint(3, 4)
            ncols = rng.randint(3, 4)
        elif 3 in scenario:
            nrows = rng.randint(2, 4)
            ncols = rng.randint(2, 4)
            if nrows == 2 and ncols == 2:
                continue
        else:
            nrows = rng.randint(2, 4)
            ncols = rng.randint(2, 4)
        if nrows * ncols >= 4:
            return nrows, ncols
    raise RuntimeError("failed to choose a valid room grid")


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_TRIES):
        scenario = rng.choice(SCENARIOS)
        nrows, ncols = _choose_room_grid(rng, scenario)
        row_sizes = _sample_axis_sizes(rng, nrows)
        col_sizes = _sample_axis_sizes(rng, ncols)

        adjacency, all_edges = _build_adjacency(nrows, ncols)
        door_edges = _choose_door_edges(nrows, ncols, adjacency, all_edges, scenario, rng)
        if door_edges is None:
            continue

        grid = _build_grid(row_sizes, col_sizes, door_edges, rng)
        wall_color, _, rooms, door_pairs, degrees = _extract_structure(grid)
        if wall_color != GREY:
            continue
        if len(rooms) != nrows * ncols:
            continue
        if len(door_pairs) != len(door_edges):
            continue
        if any(degree == 0 for degree in degrees.values()):
            continue
        if not scenario.issubset(_degree_categories(degrees)):
            continue
        if 1 not in degrees.values():
            continue

        out = main(grid)
        if int(np.count_nonzero(out == ORANGE)) == 0:
            continue
        return grid
    raise RuntimeError("generate_input: exhausted attempts")
