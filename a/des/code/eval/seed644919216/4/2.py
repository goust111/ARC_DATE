from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("U", "R", "D", "L")
SIDE_STEPS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
TOOTH_OFFSETS = {
    "U": (-2, 0),
    "R": (0, 2),
    "D": (2, 0),
    "L": (0, -2),
}
CW = {"U": "R", "R": "D", "D": "L", "L": "U"}
CCW = {dst: src for src, dst in CW.items()}
MAX_GENERATION_ATTEMPTS = 400


def _body_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _tooth_cell(center: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = center
    dr, dc = TOOTH_OFFSETS[side]
    return r + dr, c + dc


def _extract_gears(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gears = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            ring = [
                int(patch[rr, cc])
                for rr in range(3)
                for cc in range(3)
                if (rr, cc) != (1, 1)
            ]
            body_color = ring[0]
            if body_color == BG or any(value != body_color for value in ring):
                continue

            center_value = int(patch[1, 1])
            if center_value not in (body_color, RED):
                continue

            teeth = {
                side
                for side in SIDES
                if 0 <= _tooth_cell((r, c), side)[0] < h
                and 0 <= _tooth_cell((r, c), side)[1] < w
                and int(g[_tooth_cell((r, c), side)]) == body_color
            }
            gears.append(
                {
                    "center": (r, c),
                    "color": body_color,
                    "teeth": frozenset(teeth),
                    "driver": center_value == RED and body_color != RED,
                }
            )

    gears.sort(key=lambda gear: gear["center"])
    assert gears, "expected at least one gear"

    driver_indices = [idx for idx, gear in enumerate(gears) if gear["driver"]]
    assert len(driver_indices) == 1, "expected exactly one driver gear"

    covered = set()
    for gear in gears:
        covered.update(_body_cells(gear["center"]))
        covered.update(_tooth_cell(gear["center"], side) for side in gear["teeth"])

    non_bg = {tuple(map(int, coord)) for coord in np.argwhere(g != BG)}
    assert covered == non_bg, "all foreground pixels must belong to gears"
    return gears, driver_indices[0]


def _build_mesh_graph(grid: np.ndarray, gears: list[dict]) -> list[list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    by_center = {gear["center"]: idx for idx, gear in enumerate(gears)}
    graph = [set() for _ in gears]

    for idx, gear in enumerate(gears):
        r, c = gear["center"]

        right_idx = by_center.get((r, c + 6))
        if (
            right_idx is not None
            and "R" in gear["teeth"]
            and "L" in gears[right_idx]["teeth"]
            and c + 3 < w
            and int(g[r, c + 3]) == BG
        ):
            graph[idx].add(right_idx)
            graph[right_idx].add(idx)

        down_idx = by_center.get((r + 6, c))
        if (
            down_idx is not None
            and "D" in gear["teeth"]
            and "U" in gears[down_idx]["teeth"]
            and r + 3 < h
            and int(g[r + 3, c]) == BG
        ):
            graph[idx].add(down_idx)
            graph[down_idx].add(idx)

    return [sorted(neighbors) for neighbors in graph]


def _propagate_rotation(graph: list[list[int]], start: int) -> dict[int, int]:
    directions = {start: 1}
    queue = deque([start])

    while queue:
        idx = queue.popleft()
        for neighbor in graph[idx]:
            expected = -directions[idx]
            if neighbor in directions:
                assert directions[neighbor] == expected, "gear graph must be bipartite"
                continue
            directions[neighbor] = expected
            queue.append(neighbor)

    return directions


def _rotate_teeth(teeth: frozenset[str], direction: int) -> frozenset[str]:
    mapping = CW if direction > 0 else CCW
    return frozenset(mapping[side] for side in teeth)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate meshed gears outward from the unique red-centered driver gear.

    Each gear keeps its 3x3 body fixed in place. Only tooth positions move,
    rotating 90 degrees clockwise for the driver, then alternating direction
    across every meshed neighbor in the same connected component.
    """
    g = np.asarray(grid, dtype=int)
    gears, driver_idx = _extract_gears(g)
    graph = _build_mesh_graph(g, gears)
    directions = _propagate_rotation(graph, driver_idx)

    out = g.copy()

    for gear in gears:
        for rr, cc in _body_cells(gear["center"]):
            out[rr, cc] = gear["color"]
        for side in gear["teeth"]:
            rr, cc = _tooth_cell(gear["center"], side)
            out[rr, cc] = BG

    for idx, gear in enumerate(gears):
        final_teeth = gear["teeth"]
        if idx in directions:
            final_teeth = _rotate_teeth(gear["teeth"], directions[idx])
        for side in final_teeth:
            rr, cc = _tooth_cell(gear["center"], side)
            out[rr, cc] = gear["color"]

    return out


def _grid_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    neighbors = []
    for dr, dc in SIDE_STEPS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def _grow_component(rows: int, cols: int, size: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    component = {start}

    while len(component) < size:
        frontier = {
            neighbor
            for node in component
            for neighbor in _grid_neighbors(node, rows, cols)
            if neighbor not in component
        }
        if not frontier:
            return None
        component.add(random.choice(tuple(frontier)))

    return component


def _place_isolated(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    isolated = []
    taken = set(occupied)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]

    while len(isolated) < count:
        candidates = [
            cell
            for cell in all_cells
            if cell not in taken
            and all(
                abs(cell[0] - other[0]) + abs(cell[1] - other[1]) != 1
                for other in taken
            )
        ]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        isolated.append(chosen)
        taken.add(chosen)

    return isolated


def _required_teeth(node: tuple[int, int], component: set[tuple[int, int]]) -> set[str]:
    required = set()
    r, c = node
    for side, (dr, dc) in SIDE_STEPS.items():
        if (r + dr, c + dc) in component:
            required.add(side)
    return required


def _sample_teeth(required: set[str]) -> frozenset[str]:
    teeth = set(required)
    for side in SIDES:
        if side not in teeth and random.random() < 0.5:
            teeth.add(side)
    if not teeth:
        teeth.add(random.choice(SIDES))
    return frozenset(teeth)


def _center_from_node(node: tuple[int, int], top_extra: int, left_extra: int) -> tuple[int, int]:
    return top_extra + 2 + 6 * node[0], left_extra + 2 + 6 * node[1]


def _render_input(
    gears: list[dict],
    lattice_rows: int,
    lattice_cols: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict], dict[tuple[int, int], tuple[int, int]]]:
    top_extra = random.randint(0, 3)
    left_extra = random.randint(0, 3)
    bottom_extra = random.randint(0, 3)
    right_extra = random.randint(0, 3)

    height = top_extra + 6 * (lattice_rows - 1) + 5 + bottom_extra
    width = left_extra + 6 * (lattice_cols - 1) + 5 + right_extra
    grid = np.full((height, width), BG, dtype=int)

    expected = {}
    centers_by_node = {}
    for gear in gears:
        center = _center_from_node(gear["node"], top_extra, left_extra)
        centers_by_node[gear["node"]] = center
        expected[center] = {
            "color": gear["color"],
            "teeth": gear["teeth"],
            "driver": gear["driver"],
        }

        for rr, cc in _body_cells(center):
            grid[rr, cc] = gear["color"]
        for side in gear["teeth"]:
            rr, cc = _tooth_cell(center, side)
            grid[rr, cc] = gear["color"]
        if gear["driver"]:
            grid[center] = RED

    return grid, expected, centers_by_node


def _validate_generated(
    grid: np.ndarray,
    expected: dict[tuple[int, int], dict],
    centers_by_node: dict[tuple[int, int], tuple[int, int]],
    component: set[tuple[int, int]],
) -> bool:
    try:
        parsed, driver_idx = _extract_gears(grid)
        if len(parsed) != len(expected):
            return False

        parsed_by_center = {gear["center"]: gear for gear in parsed}
        if set(parsed_by_center) != set(expected):
            return False

        for center, spec in expected.items():
            parsed_gear = parsed_by_center[center]
            if parsed_gear["color"] != spec["color"]:
                return False
            if parsed_gear["teeth"] != spec["teeth"]:
                return False
            if parsed_gear["driver"] != spec["driver"]:
                return False

        graph = _build_mesh_graph(grid, parsed)
        actual_edges = set()
        for idx, neighbors in enumerate(graph):
            for neighbor in neighbors:
                if idx < neighbor:
                    edge = tuple(sorted((parsed[idx]["center"], parsed[neighbor]["center"])))
                    actual_edges.add(edge)

        expected_edges = set()
        for node in component:
            for neighbor in _grid_neighbors(node, 99, 99):
                if neighbor in component:
                    a = centers_by_node[node]
                    b = centers_by_node[neighbor]
                    expected_edges.add(tuple(sorted((a, b))))

        if actual_edges != expected_edges:
            return False

        directions = _propagate_rotation(graph, driver_idx)
        reachable_centers = {parsed[idx]["center"] for idx in directions}
        expected_reachable = {centers_by_node[node] for node in component}
        if reachable_centers != expected_reachable:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if np.array_equal(out, grid):
            return False
    except AssertionError:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    component_size = random.choice((2, 3, 4, 5))
    isolated_count = 0 if random.random() < 0.5 else random.randint(1, 3)
    total_gears = component_size + isolated_count
    if total_gears > len(PALETTE):
        return None

    min_side = 3 if total_gears <= 4 else 4
    lattice_rows = random.randint(min_side, 5)
    lattice_cols = random.randint(min_side, 5)
    component = _grow_component(lattice_rows, lattice_cols, component_size)
    if component is None:
        return None

    isolated = _place_isolated(lattice_rows, lattice_cols, component, isolated_count)
    if isolated is None:
        return None

    driver_node = random.choice(tuple(component))
    all_nodes = list(component) + isolated
    random.shuffle(all_nodes)
    colors = random.sample(PALETTE, total_gears)

    gears = []
    for node, color in zip(all_nodes, colors):
        required = _required_teeth(node, component) if node in component else set()
        gears.append(
            {
                "node": node,
                "color": color,
                "teeth": _sample_teeth(required),
                "driver": node == driver_node,
            }
        )

    grid, expected, centers_by_node = _render_input(gears, lattice_rows, lattice_cols)
    if not _validate_generated(grid, expected, centers_by_node, component):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-rotation input")
