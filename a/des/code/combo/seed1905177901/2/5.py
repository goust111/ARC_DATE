from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
NODE_SIZE = 3

# Visible direction legend for path colors. This replaces the invalid
# "color-id closeness" rule with a fixed, learnable color semantics.
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2
TOP_TO_BOTTOM = 3
BOTTOM_TO_TOP = 4
NODE_COLORS = [5, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
MIN_GAP = 2
MAX_GAP = 7

TEMPLATES = [
    {
        "points": {"A": (1, 0), "B": (0, 1), "C": (1, 1), "D": (1, 2)},
        "edges": [("A", "C"), ("B", "C"), ("C", "D")],
    },
    {
        "points": {"A": (1, 0), "B": (0, 1), "C": (1, 1), "D": (1, 2), "E": (2, 2)},
        "edges": [("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")],
    },
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return comps


def _split_nodes_and_paths(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    paths: list[dict] = []

    for comp in _components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if (
            int(comp["area"]) == NODE_SIZE * NODE_SIZE
            and r1 - r0 + 1 == NODE_SIZE
            and c1 - c0 + 1 == NODE_SIZE
            and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))
        ):
            nodes.append(comp)
        else:
            paths.append(comp)

    assert nodes, "expected at least one 3x3 node"
    return nodes, paths


def _path_receiver(comp: dict, node_by_cell: dict[tuple[int, int], int]) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])

    if r0 == r1:
        left = node_by_cell.get((r0, c0 - 1))
        right = node_by_cell.get((r0, c1 + 1))
        assert left is not None and right is not None and left != right, "horizontal path must connect two nodes"
        if color == LEFT_TO_RIGHT:
            receiver = right
        elif color == RIGHT_TO_LEFT:
            receiver = left
        else:
            raise AssertionError("horizontal path has invalid direction color")
        return receiver, int(c1 - c0 + 1)

    if c0 == c1:
        top = node_by_cell.get((r0 - 1, c0))
        bottom = node_by_cell.get((r1 + 1, c0))
        assert top is not None and bottom is not None and top != bottom, "vertical path must connect two nodes"
        if color == TOP_TO_BOTTOM:
            receiver = bottom
        elif color == BOTTOM_TO_TOP:
            receiver = top
        else:
            raise AssertionError("vertical path has invalid direction color")
        return receiver, int(r1 - r0 + 1)

    raise AssertionError("paths must be straight horizontal or vertical segments")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor 3x3 node blocks from the colors of their incoming paths.

    This version uses a fixed visible path-color legend instead of illegal
    color-id ordering:
    - 1: left -> right
    - 2: right -> left
    - 3: top -> bottom
    - 4: bottom -> top

    A node with one incoming path takes that path color. A node with two
    incoming paths takes the longer path's color. Nodes with no incoming paths
    keep their original color. Paths stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    nodes, paths = _split_nodes_and_paths(g, bg)

    node_by_cell: dict[tuple[int, int], int] = {}
    for i, node in enumerate(nodes):
        for r, c in node["coords"]:
            node_by_cell[(int(r), int(c))] = i

    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for path in paths:
        receiver, length = _path_receiver(path, node_by_cell)
        incoming[receiver].append((int(path["color"]), length))

    out = g.copy()
    for i, node in enumerate(nodes):
        choices = incoming.get(i, [])
        color = int(node["color"])
        if len(choices) == 1:
            color = int(choices[0][0])
        elif len(choices) == 2:
            assert choices[0][1] != choices[1][1], "two incoming paths must have distinct lengths"
            color = int(max(choices, key=lambda item: item[1])[0])
        elif len(choices) > 2:
            raise AssertionError("nodes are expected to have at most two incoming paths")

        r0, c0, r1, c1 = node["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _transform_points(points: dict[str, tuple[int, int]], variant: int) -> dict[str, tuple[int, int]]:
    h = max(r for r, _c in points.values()) + 1
    w = max(c for _r, c in points.values()) + 1
    transformed: dict[str, tuple[int, int]] = {}

    for name, (r, c) in points.items():
        if variant == 0:
            nr, nc = r, c
        elif variant == 1:
            nr, nc = c, h - 1 - r
        elif variant == 2:
            nr, nc = h - 1 - r, w - 1 - c
        elif variant == 3:
            nr, nc = w - 1 - c, r
        elif variant == 4:
            nr, nc = r, w - 1 - c
        elif variant == 5:
            nr, nc = h - 1 - r, c
        elif variant == 6:
            nr, nc = c, r
        elif variant == 7:
            nr, nc = w - 1 - c, h - 1 - r
        else:
            raise AssertionError("unknown symmetry")
        transformed[name] = (int(nr), int(nc))

    return transformed


def _axis_positions(levels: list[int]) -> tuple[dict[int, int], int]:
    pos_by_level: dict[int, int] = {}
    cur = random.randint(1, 3)

    for i, level in enumerate(levels):
        pos_by_level[level] = cur
        if i + 1 < len(levels):
            cur += NODE_SIZE + random.randint(MIN_GAP, MAX_GAP)

    total_size = cur + NODE_SIZE + random.randint(1, 3)
    return pos_by_level, total_size


def _instantiate_points(points: dict[str, tuple[int, int]]) -> tuple[dict[str, tuple[int, int]], int, int]:
    row_levels = sorted({r for r, _c in points.values()})
    col_levels = sorted({c for _r, c in points.values()})
    row_pos, height = _axis_positions(row_levels)
    col_pos, width = _axis_positions(col_levels)

    actual = {
        name: (row_pos[r], col_pos[c])
        for name, (r, c) in points.items()
    }
    return actual, height, width


def _edge_color(src: tuple[int, int], dst: tuple[int, int]) -> int:
    sr, sc = src
    dr, dc = dst
    if sr == dr:
        return LEFT_TO_RIGHT if sc < dc else RIGHT_TO_LEFT
    if sc == dc:
        return TOP_TO_BOTTOM if sr < dr else BOTTOM_TO_TOP
    raise AssertionError("edges must be axis-aligned")


def _edge_length(src: tuple[int, int], dst: tuple[int, int]) -> int:
    sr, sc = src
    dr, dc = dst
    if sr == dr:
        return abs(dc - sc) - NODE_SIZE
    if sc == dc:
        return abs(dr - sr) - NODE_SIZE
    raise AssertionError("edges must be axis-aligned")


def _draw_edge(grid: np.ndarray, src: tuple[int, int], dst: tuple[int, int], color: int) -> None:
    sr, sc = src
    dr, dc = dst

    if sr == dr:
        row = sr + 1
        left = min(sc, dc) + NODE_SIZE
        right = max(sc, dc)
        grid[row, left:right] = color
        return

    if sc == dc:
        col = sc + 1
        top = min(sr, dr) + NODE_SIZE
        bottom = max(sr, dr)
        grid[top:bottom, col] = color
        return

    raise AssertionError("edges must be axis-aligned")


def _expected_output(
    grid: np.ndarray,
    points: dict[str, tuple[int, int]],
    edges: list[tuple[str, str]],
    node_colors: dict[str, int],
) -> np.ndarray:
    incoming: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for src, dst in edges:
        incoming[dst].append((_edge_color(points[src], points[dst]), _edge_length(points[src], points[dst])))

    out = grid.copy()
    for name, (r, c) in points.items():
        choices = incoming.get(name, [])
        color = int(node_colors[name])
        if len(choices) == 1:
            color = int(choices[0][0])
        elif len(choices) == 2:
            color = int(max(choices, key=lambda item: item[1])[0])
        out[r : r + NODE_SIZE, c : c + NODE_SIZE] = color

    return out


def _build_candidate() -> np.ndarray | None:
    template = random.choice(TEMPLATES)
    transformed = _transform_points(template["points"], random.randrange(8))
    points, height, width = _instantiate_points(transformed)

    incoming_lengths: dict[str, list[int]] = defaultdict(list)
    for src, dst in template["edges"]:
        length = _edge_length(points[src], points[dst])
        if length < MIN_GAP:
            return None
        incoming_lengths[dst].append(length)

    for lengths in incoming_lengths.values():
        if len(lengths) == 2 and lengths[0] == lengths[1]:
            return None

    colors = random.sample(NODE_COLORS, len(points))
    node_colors = {name: colors[i] for i, name in enumerate(sorted(points))}

    grid = np.full((height, width), BG, dtype=int)
    for name, (r, c) in points.items():
        grid[r : r + NODE_SIZE, c : c + NODE_SIZE] = int(node_colors[name])

    for src, dst in template["edges"]:
        _draw_edge(grid, points[src], points[dst], _edge_color(points[src], points[dst]))

    expected = _expected_output(grid, points, template["edges"], node_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
