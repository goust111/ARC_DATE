from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

PALETTE_LEN = 4
NODE_SIZE = 3
MIN_STEP = 4
MAX_STEP = 6
MAX_DIM = 30
MAX_GENERATION_ATTEMPTS = 400
PATH_COLORS = (BLUE, RED, GREEN, YELLOW)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _palette_cells(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    return {(h - 1, c) for c in range(w - PALETTE_LEN, w)}


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _find_palette(grid: np.ndarray) -> tuple[int, ...]:
    assert grid.shape[1] >= PALETTE_LEN, "grid must be wide enough for the palette"
    palette = tuple(int(x) for x in grid[-1, -PALETTE_LEN:])
    assert len(set(palette)) == PALETTE_LEN, "palette must contain four distinct colors"
    assert all(color != BLACK for color in palette), "palette colors must be non-background"
    return palette


def _find_node_color(grid: np.ndarray, palette: tuple[int, ...]) -> int:
    colors = set(map(int, np.unique(grid)))
    colors.discard(BLACK)
    node_colors = [color for color in colors if color not in palette]
    assert len(node_colors) == 1, "expected exactly one node color outside the palette"
    return int(node_colors[0])


def _find_nodes(grid: np.ndarray, node_color: int) -> list[tuple[int, int, int, int]]:
    components = _connected_components(grid == node_color)
    assert components, "expected at least one node"

    boxes: list[tuple[int, int, int, int]] = []
    for coords in components:
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(coords) == NODE_SIZE * NODE_SIZE, "node must be a 3x3 block"
        assert r1 - r0 + 1 == NODE_SIZE and c1 - c0 + 1 == NODE_SIZE, "node bbox must be 3x3"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == node_color), "node must be solid"
        boxes.append((r0, c0, r1, c1))

    return boxes


def _incident_colors(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    node_color: int,
    palette_set: set[int],
    palette_coords: set[tuple[int, int]],
) -> set[int]:
    r0, c0, r1, c1 = bbox
    colors: set[int] = set()

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if r0 <= nr <= r1 and c0 <= nc <= c1:
                    continue
                if (nr, nc) in palette_coords:
                    continue
                color = int(grid[nr, nc])
                if color == BLACK or color == node_color:
                    continue
                assert color in palette_set, "incident edges must use palette colors"
                colors.add(color)

    assert len(colors) in (2, 3), "each node must have two or three distinct incident path colors"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orange node to the leftmost palette color missing from its incident paths.

    Input structure:
    - The background is black.
    - Several solid 3x3 node blocks share one distinct non-palette color.
    - Nodes are connected by one-cell-wide horizontal or vertical paths whose colors are listed by a four-cell palette on the bottom-right edge.

    Rule:
    - Read the palette from the last four cells of the bottom row; this defines the full ordered set of possible path colors.
    - For each 3x3 node, collect the distinct path colors touching that node.
    - Find the palette colors absent from that incident set and repaint the whole node with the leftmost missing palette color.
    - Leave the paths, palette, and background unchanged.

    Key insight:
    - A node's output color is determined by the visible palette complement at that node, not by preferring any used path color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _find_palette(g)
    palette_set = set(palette)
    palette_coords = _palette_cells(g.shape)
    node_color = _find_node_color(g, palette)
    nodes = _find_nodes(g, node_color)

    out = g.copy()
    for bbox in nodes:
        incident = _incident_colors(g, bbox, node_color, palette_set, palette_coords)
        missing = [color for color in palette if color not in incident]
        assert len(missing) in (1, 2), "node must miss one or two palette colors"
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(missing[0])

    return out


def _graph_shape() -> tuple[int, int]:
    length = random.randint(2, 5)
    if random.random() < 0.5:
        return 2, length
    return length, 2


def _sample_axis_positions(count: int, reserve_bottom_palette: bool) -> tuple[list[int], int]:
    gaps = [random.randint(MIN_STEP, MAX_STEP) for _ in range(count - 1)]
    span = NODE_SIZE + sum(gaps)

    min_trail = 2 if reserve_bottom_palette else 0
    lead_cap = min(3, MAX_DIM - span - min_trail)
    assert lead_cap >= 0
    lead = random.randint(0, lead_cap)

    trail_cap = min(4, MAX_DIM - span - lead)
    trail_lo = min_trail
    assert trail_cap >= trail_lo
    trail = random.randint(trail_lo, trail_cap)

    positions = [lead]
    for gap in gaps:
        positions.append(positions[-1] + gap)

    return positions, lead + span + trail


def _make_nodes(rows: int, cols: int) -> tuple[dict[tuple[int, int], tuple[int, int]], int, int]:
    row_positions, height = _sample_axis_positions(rows, reserve_bottom_palette=True)
    col_positions, width = _sample_axis_positions(cols, reserve_bottom_palette=False)
    nodes = {
        (r, c): (row_positions[r], col_positions[c])
        for r in range(rows)
        for c in range(cols)
    }
    return nodes, height, width


def _make_edges(rows: int, cols: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], list[int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    incident: dict[tuple[int, int], list[int]] = defaultdict(list)

    for r in range(rows):
        for c in range(cols - 1):
            edge = ((r, c), (r, c + 1))
            incident[(r, c)].append(len(edges))
            incident[(r, c + 1)].append(len(edges))
            edges.append(edge)

    for r in range(rows - 1):
        for c in range(cols):
            edge = ((r, c), (r + 1, c))
            incident[(r, c)].append(len(edges))
            incident[(r + 1, c)].append(len(edges))
            edges.append(edge)

    return edges, incident


def _color_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    incident: dict[tuple[int, int], list[int]],
    palette: tuple[int, ...],
) -> dict[int, int] | None:
    used_by_node: dict[tuple[int, int], set[int]] = {node: set() for node in incident}
    assignment: dict[int, int] = {}

    def backtrack() -> bool:
        if len(assignment) == len(edges):
            return True

        best_idx = None
        best_options = None
        for edge_idx, (u, v) in enumerate(edges):
            if edge_idx in assignment:
                continue
            options = [color for color in palette if color not in used_by_node[u] and color not in used_by_node[v]]
            if best_options is None or len(options) < len(best_options):
                best_idx = edge_idx
                best_options = options
            if best_options is not None and len(best_options) <= 1:
                break

        assert best_idx is not None and best_options is not None
        if not best_options:
            return False

        random.shuffle(best_options)
        u, v = edges[best_idx]
        for color in best_options:
            assignment[best_idx] = color
            used_by_node[u].add(color)
            used_by_node[v].add(color)
            if backtrack():
                return True
            used_by_node[u].remove(color)
            used_by_node[v].remove(color)
            del assignment[best_idx]
        return False

    if not backtrack():
        return None
    return assignment


def _node_outputs(
    incident: dict[tuple[int, int], list[int]],
    edge_colors: dict[int, int],
    palette: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    outputs: dict[tuple[int, int], int] = {}
    for node, edge_ids in incident.items():
        colors = {edge_colors[edge_idx] for edge_idx in edge_ids}
        assert len(colors) == len(edge_ids)
        missing = [color for color in palette if color not in colors]
        assert len(missing) in (1, 2)
        outputs[node] = int(missing[0])
    return outputs


def _paint_edge(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    node_positions: dict[tuple[int, int], tuple[int, int]],
    color: int,
) -> None:
    ra, ca = node_positions[a]
    rb, cb = node_positions[b]

    if ra == rb:
        row = ra + 1
        start = min(ca, cb) + NODE_SIZE
        end = max(ca, cb) - 1
        assert end >= start
        grid[row, start : end + 1] = color
        return

    assert ca == cb
    col = ca + 1
    start = min(ra, rb) + NODE_SIZE
    end = max(ra, rb) - 1
    assert end >= start
    grid[start : end + 1, col] = color


def _expected_output(
    grid: np.ndarray,
    node_positions: dict[tuple[int, int], tuple[int, int]],
    outputs: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for node, (r0, c0) in node_positions.items():
        out[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = outputs[node]
    return out


def _valid_scene_features(
    rows: int,
    cols: int,
    edge_colors: dict[int, int],
    outputs: dict[tuple[int, int], int],
) -> bool:
    if len(set(edge_colors.values())) < 3:
        return False
    if len(set(outputs.values())) < 2:
        return False

    degrees = [2 if r in (0, rows - 1) and c in (0, cols - 1) else 3 for r in range(rows) for c in range(cols)]
    if any(deg not in (2, 3) for deg in degrees):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows, cols = _graph_shape()
    node_positions, height, width = _make_nodes(rows, cols)
    edges, incident = _make_edges(rows, cols)
    edge_colors = _color_edges(edges, incident, PATH_COLORS)
    if edge_colors is None:
        return None

    outputs = _node_outputs(incident, edge_colors, PATH_COLORS)
    if not _valid_scene_features(rows, cols, edge_colors, outputs):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r0, c0 in node_positions.values():
        grid[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = ORANGE

    for edge_idx, (a, b) in enumerate(edges):
        _paint_edge(grid, a, b, node_positions, edge_colors[edge_idx])

    grid[-1, -PALETTE_LEN:] = np.array(PATH_COLORS, dtype=int)

    expected = _expected_output(grid, node_positions, outputs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
