from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


RED = 2
BLUE = 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _foreground_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    assert len(colors) == 1, "expected exactly one foreground color"
    return colors[0]


def _build_star_graph(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int, str]], list[list[int]]]:
    row_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    col_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (r, c) in enumerate(points):
        row_map[int(r)].append((int(c), idx))
        col_map[int(c)].append((int(r), idx))

    edges: list[tuple[int, int, str]] = []
    adjacency = [[] for _ in points]

    for _row, items in row_map.items():
        items.sort()
        for (_c0, i0), (_c1, i1) in zip(items, items[1:]):
            edges.append((i0, i1, "H"))
            adjacency[i0].append(i1)
            adjacency[i1].append(i0)

    for _col, items in col_map.items():
        items.sort()
        for (_r0, i0), (_r1, i1) in zip(items, items[1:]):
            edges.append((i0, i1, "V"))
            adjacency[i0].append(i1)
            adjacency[i1].append(i0)

    return edges, adjacency


def _connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color pixels into constellations by row/column nearest-neighbor links.

    Input structure:
    - The dominant color is the background.
    - All non-background cells are single-color star pixels.
    - Two stars are directly linked when they share a row or column and there is
      no other star between them on that row or column.
    - Constellations are the connected components of that link relation.

    Rule:
    - For each constellation, connect every directly linked same-row or same-column pair
      with a straight line through the cells between them.
    - If the constellation's total horizontal connector length is at least its total
      vertical connector length, recolor all of that constellation's connector cells red.
    - Otherwise recolor all of that constellation's connector cells blue.
    - Preserve every original star pixel in its original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    fg = _foreground_color(g, bg)
    points = [tuple(map(int, rc)) for rc in np.argwhere(g == fg)]
    assert points, "expected at least one foreground pixel"

    edges, adjacency = _build_star_graph(points)
    components = _connected_components(adjacency)
    point_set = set(points)

    fill_requests: dict[tuple[int, int], int] = {}
    for comp in components:
        comp_set = set(comp)
        comp_edges = [(a, b, axis) for a, b, axis in edges if a in comp_set]

        horizontal_total = 0
        vertical_total = 0
        for a, b, axis in comp_edges:
            r0, c0 = points[a]
            r1, c1 = points[b]
            if axis == "H":
                horizontal_total += abs(c1 - c0) - 1
            else:
                vertical_total += abs(r1 - r0) - 1

        line_color = RED if horizontal_total >= vertical_total else BLUE

        for a, b, axis in comp_edges:
            r0, c0 = points[a]
            r1, c1 = points[b]
            if axis == "H":
                step = 1 if c1 > c0 else -1
                for c in range(c0 + step, c1, step):
                    cell = (r0, c)
                    assert cell not in point_set, "connector path cannot pass through a star pixel"
                    prev = fill_requests.get(cell)
                    assert prev in (None, line_color), "conflicting connector colors"
                    fill_requests[cell] = line_color
            else:
                step = 1 if r1 > r0 else -1
                for r in range(r0 + step, r1, step):
                    cell = (r, c0)
                    assert cell not in point_set, "connector path cannot pass through a star pixel"
                    prev = fill_requests.get(cell)
                    assert prev in (None, line_color), "conflicting connector colors"
                    fill_requests[cell] = line_color

    out = g.copy()
    for (r, c), color in fill_requests.items():
        out[r, c] = color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
COMPONENT_ATTEMPTS = 200
POSITION_ATTEMPTS = 120

COMPONENT_COUNT_CHOICES = (1, 2, 3)
COMPONENT_COUNT_WEIGHTS = (2, 5, 3)
STAR_COUNT_RANGE = (2, 5)

SMALL_GAP_RANGE = (2, 3)
LARGE_GAP_RANGE = (3, 5)
PACK_GAP_RANGE = (1, 3)
OUTER_MARGIN_RANGE = (0, 2)

LOCAL_SIDE_MAX = 12
GRID_SIDE_MIN = 8
GRID_SIDE_MAX = 28

GEN_COLORS = [0, 3, 4, 5, 6, 7, 8, 9]


def _choose_component_majors(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("H", "V"))]
    if random.random() < 0.7:
        majors = ["H", "V"]
        while len(majors) < count:
            majors.append(random.choice(("H", "V")))
        random.shuffle(majors)
        return majors
    return [random.choice(("H", "V")) for _ in range(count)]


def _edge_orientations(edge_count: int, major_axis: str) -> list[str]:
    if edge_count == 1:
        return [major_axis]

    if random.random() < 0.35:
        if major_axis == "H":
            h_count, v_count = edge_count, 0
        else:
            h_count, v_count = 0, edge_count
    else:
        if major_axis == "H":
            h_count = random.randint((edge_count + 1) // 2, edge_count - 1)
            v_count = edge_count - h_count
        else:
            v_count = random.randint((edge_count + 1) // 2, edge_count - 1)
            h_count = edge_count - v_count

    orientations = ["H"] * h_count + ["V"] * v_count
    random.shuffle(orientations)
    return orientations


def _expandable_options(nodes: list[tuple[int, int]], orientation: str) -> list[tuple[int, str]]:
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in nodes:
        row_to_cols[int(r)].append(int(c))
        col_to_rows[int(c)].append(int(r))

    options: list[tuple[int, str]] = []
    if orientation == "H":
        for idx, (r, c) in enumerate(nodes):
            cols = row_to_cols[int(r)]
            if c == min(cols):
                options.append((idx, "left"))
            if c == max(cols):
                options.append((idx, "right"))
    else:
        for idx, (r, c) in enumerate(nodes):
            rows = col_to_rows[int(c)]
            if r == min(rows):
                options.append((idx, "up"))
            if r == max(rows):
                options.append((idx, "down"))
    return options


def _build_component_tree(orientations: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]] | None:
    nodes = [(0, 0)]
    edges: list[tuple[int, int, str]] = []

    for orientation in orientations:
        options = _expandable_options(nodes, orientation)
        if not options:
            return None
        parent_idx, side = random.choice(options)
        parent_r, parent_c = nodes[parent_idx]

        used_rows = [r for r, _c in nodes]
        used_cols = [c for _r, c in nodes]
        if side == "left":
            new_node = (parent_r, min(used_cols) - 1)
        elif side == "right":
            new_node = (parent_r, max(used_cols) + 1)
        elif side == "up":
            new_node = (min(used_rows) - 1, parent_c)
        else:
            new_node = (max(used_rows) + 1, parent_c)

        nodes.append(new_node)
        edges.append((parent_idx, len(nodes) - 1, orientation))

    return nodes, edges


def _sample_axis_positions(levels: list[int], gap_range: tuple[int, int]) -> dict[int, int]:
    positions = {levels[0]: 0}
    current = 0
    for level in levels[1:]:
        current += random.randint(*gap_range)
        positions[level] = current
    return positions


def _measure_totals(
    nodes: list[tuple[int, int]],
    edges: list[tuple[int, int, str]],
    row_positions: dict[int, int],
    col_positions: dict[int, int],
) -> tuple[list[tuple[int, int]], int, int]:
    actual_nodes = [(row_positions[r], col_positions[c]) for r, c in nodes]
    horizontal_total = 0
    vertical_total = 0
    for a, b, axis in edges:
        r0, c0 = actual_nodes[a]
        r1, c1 = actual_nodes[b]
        if axis == "H":
            horizontal_total += abs(c1 - c0) - 1
        else:
            vertical_total += abs(r1 - r0) - 1
    return actual_nodes, horizontal_total, vertical_total


def _make_component(major_axis: str) -> dict[str, object] | None:
    edge_count = random.randint(STAR_COUNT_RANGE[0] - 1, STAR_COUNT_RANGE[1] - 1)
    orientations = _edge_orientations(edge_count, major_axis)
    built = _build_component_tree(orientations)
    if built is None:
        return None
    abstract_nodes, edges = built

    row_levels = sorted({r for r, _c in abstract_nodes})
    col_levels = sorted({c for _r, c in abstract_nodes})

    for _ in range(POSITION_ATTEMPTS):
        row_gaps = LARGE_GAP_RANGE if major_axis == "V" else SMALL_GAP_RANGE
        col_gaps = LARGE_GAP_RANGE if major_axis == "H" else SMALL_GAP_RANGE
        row_positions = _sample_axis_positions(row_levels, row_gaps)
        col_positions = _sample_axis_positions(col_levels, col_gaps)
        actual_nodes, horizontal_total, vertical_total = _measure_totals(
            abstract_nodes, edges, row_positions, col_positions
        )

        if major_axis == "H" and horizontal_total < vertical_total:
            continue
        if major_axis == "V" and vertical_total <= horizontal_total:
            continue

        height = max(r for r, _c in actual_nodes) + 1
        width = max(c for _r, c in actual_nodes) + 1
        if height > LOCAL_SIDE_MAX or width > LOCAL_SIDE_MAX:
            continue

        return {
            "nodes": actual_nodes,
            "edges": edges,
            "major_axis": major_axis,
            "height": height,
            "width": width,
        }

    return None


def _render_expected(grid: np.ndarray, components: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for comp in components:
        color = RED if comp["major_axis"] == "H" else BLUE
        nodes = comp["nodes"]
        for a, b, axis in comp["edges"]:
            r0, c0 = nodes[a]
            r1, c1 = nodes[b]
            if axis == "H":
                lo, hi = sorted((c0, c1))
                out[r0, lo + 1 : hi] = color
            else:
                lo, hi = sorted((r0, r1))
                out[lo + 1 : hi, c0] = color
    return out


def _pack_components(
    local_components: list[dict[str, object]],
    bg: int,
    fg: int,
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    count = len(local_components)
    row_order = random.sample(range(count), k=count)
    col_order = random.sample(range(count), k=count)

    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)
    row_gaps = [random.randint(*PACK_GAP_RANGE) for _ in range(max(0, count - 1))]
    col_gaps = [random.randint(*PACK_GAP_RANGE) for _ in range(max(0, count - 1))]

    height = top_margin + bottom_margin + sum(int(local_components[i]["height"]) for i in row_order) + sum(row_gaps)
    width = left_margin + right_margin + sum(int(local_components[i]["width"]) for i in col_order) + sum(col_gaps)
    if not (GRID_SIDE_MIN <= height <= GRID_SIDE_MAX and GRID_SIDE_MIN <= width <= GRID_SIDE_MAX):
        return None

    row_starts: dict[int, int] = {}
    current = top_margin
    for pos, idx in enumerate(row_order):
        row_starts[idx] = current
        current += int(local_components[idx]["height"])
        if pos < len(row_gaps):
            current += row_gaps[pos]

    col_starts: dict[int, int] = {}
    current = left_margin
    for pos, idx in enumerate(col_order):
        col_starts[idx] = current
        current += int(local_components[idx]["width"])
        if pos < len(col_gaps):
            current += col_gaps[pos]

    grid = np.full((height, width), bg, dtype=int)
    placed_components: list[dict[str, object]] = []
    for idx, comp in enumerate(local_components):
        row0 = row_starts[idx]
        col0 = col_starts[idx]
        shifted_nodes = [(row0 + r, col0 + c) for r, c in comp["nodes"]]
        for r, c in shifted_nodes:
            grid[r, c] = fg
        placed_components.append(
            {
                "nodes": shifted_nodes,
                "edges": comp["edges"],
                "major_axis": comp["major_axis"],
            }
        )

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid != bg):
        return None
    return grid, placed_components


def _gen_attempt() -> np.ndarray | None:
    bg, fg = random.sample(GEN_COLORS, 2)
    component_count = random.choices(COMPONENT_COUNT_CHOICES, weights=COMPONENT_COUNT_WEIGHTS, k=1)[0]
    majors = _choose_component_majors(component_count)

    components: list[dict[str, object]] = []
    for major_axis in majors:
        component = None
        for _ in range(COMPONENT_ATTEMPTS):
            component = _make_component(major_axis)
            if component is not None:
                break
        if component is None:
            return None
        components.append(component)

    packed = _pack_components(components, bg, fg)
    if packed is None:
        return None
    grid, placed_components = packed

    expected = _render_expected(grid, placed_components)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid constellation-linking input")
