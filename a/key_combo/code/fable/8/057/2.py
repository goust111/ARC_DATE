from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DEPTH_TO_COLOR = {1: BLUE, 2: RED, 3: GREEN, 4: YELLOW}
ROOM_ROWS = 3
ROOM_COLS = 4
ROOM_NODES = [(r, c) for r in range(ROOM_ROWS) for c in range(ROOM_COLS)]
EXT = ("ext",)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _detect_doors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    """Return door cells as {(r, c): orientation}, orientation in {'h', 'v'}."""
    h, w = grid.shape
    doors: dict[tuple[int, int], str] = {}
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue

            left_right_wall = 0 < c < w - 1 and int(grid[r, c - 1]) == GREY and int(grid[r, c + 1]) == GREY
            up_down_wall = 0 < r < h - 1 and int(grid[r - 1, c]) == GREY and int(grid[r + 1, c]) == GREY
            if left_right_wall == up_down_wall:
                continue

            if left_right_wall:
                if r == 0 or r == h - 1:
                    continue
                if int(grid[r - 1, c]) == GREY or int(grid[r + 1, c]) == GREY:
                    continue
                doors[(r, c)] = "h"
            else:
                if c == 0 or c == w - 1:
                    continue
                if int(grid[r, c - 1]) == GREY or int(grid[r, c + 1]) == GREY:
                    continue
                doors[(r, c)] = "v"
    return doors


def _label_components(open_mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = open_mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or labels[r, c] != -1:
                continue
            label = len(components)
            cells: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            labels[r, c] = label
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(cells)
    return labels, components


def _room_graph(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[set[int]], set[int], dict[tuple[int, int], str]]:
    doors = _detect_doors(grid)
    door_mask = np.zeros(grid.shape, dtype=bool)
    for r, c in doors:
        door_mask[r, c] = True

    open_mask = (grid == BLACK) & ~door_mask
    labels, components = _label_components(open_mask)
    adjacency = [set() for _ in components]
    boundary_labels: set[int] = set()
    h, w = grid.shape

    for r in range(h):
        for c in (0, w - 1):
            label = int(labels[r, c])
            if label != -1:
                boundary_labels.add(label)
    for c in range(w):
        for r in (0, h - 1):
            label = int(labels[r, c])
            if label != -1:
                boundary_labels.add(label)

    for (r, c), orientation in doors.items():
        if orientation == "h":
            sides = ((r - 1, c), (r + 1, c))
        else:
            sides = ((r, c - 1), (r, c + 1))

        ids = []
        for rr, cc in sides:
            if not (0 <= rr < h and 0 <= cc < w):
                ids = []
                break
            label = int(labels[rr, cc])
            if label == -1:
                ids = []
                break
            ids.append(label)
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        a, b = ids
        adjacency[a].add(b)
        adjacency[b].add(a)

    return labels, components, adjacency, boundary_labels, doors


def _door_depths(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[int, int], dict[tuple[int, int], str]]:
    labels, components, adjacency, boundary_labels, doors = _room_graph(grid)
    distances = {label: 0 for label in boundary_labels}
    queue = deque(boundary_labels)

    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt in distances:
                continue
            distances[nxt] = distances[cur] + 1
            queue.append(nxt)

    return labels, components, distances, doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each room by the minimum number of doors needed to enter it from the exterior.

    Input structure:
    - The grid has a black background and one-cell-thick grey wall segments forming a rectilinear floor plan.
    - Some wall cells are missing as one-pixel door gaps; some rooms have no door path from outside.
    - Only black and grey appear in the input.

    Rule:
    - Treat each door gap as a crossing point through an otherwise solid wall, not as ordinary open floor.
    - Temporarily close the door gaps to split the black area into exterior/room regions, then connect regions back through the doors.
    - Any room at door-depth 1 becomes blue, depth 2 red, depth 3 green, depth 4 yellow; unreachable rooms, exterior, and the door cells stay black, while grey walls stay unchanged.

    Key insight:
    - Geometric position is irrelevant: the right answer comes from the region graph induced by walls and doors, and each door contributes exactly one step on the shortest path from outside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREY}, "input must contain only black background and grey walls"
    assert GREY in colors, "expected at least one grey wall"

    labels, components, distances, _doors = _door_depths(g)
    out = g.copy()

    for label, cells in enumerate(components):
        depth = distances.get(label)
        if depth is None or depth == 0:
            continue
        assert depth in DEPTH_TO_COLOR, "generator must keep reachable room depths within 1..4"
        color = DEPTH_TO_COLOR[depth]
        for r, c in cells:
            out[r, c] = color

    # Door cells remain black because they are connectors, not room interiors.
    return out


def _cell_rects(row_heights: list[int], col_widths: list[int], top: int, left: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rects: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    row_starts = []
    cur_r = top + 1
    for height in row_heights:
        row_starts.append(cur_r)
        cur_r += height + 1

    col_starts = []
    cur_c = left + 1
    for width in col_widths:
        col_starts.append(cur_c)
        cur_c += width + 1

    for r in range(ROOM_ROWS):
        for c in range(ROOM_COLS):
            r0 = row_starts[r]
            c0 = col_starts[c]
            rects[(r, c)] = (r0, c0, r0 + row_heights[r] - 1, c0 + col_widths[c] - 1)
    return rects


def _build_shell(grid: np.ndarray, rects: dict[tuple[int, int], tuple[int, int, int, int]]) -> None:
    outer_r0 = min(r0 for r0, _, _, _ in rects.values()) - 1
    outer_c0 = min(c0 for _, c0, _, _ in rects.values()) - 1
    outer_r1 = max(r1 for _, _, r1, _ in rects.values()) + 1
    outer_c1 = max(c1 for _, _, _, c1 in rects.values()) + 1
    grid[outer_r0 : outer_r1 + 1, outer_c0] = GREY
    grid[outer_r0 : outer_r1 + 1, outer_c1] = GREY
    grid[outer_r0, outer_c0 : outer_c1 + 1] = GREY
    grid[outer_r1, outer_c0 : outer_c1 + 1] = GREY

    for r in range(ROOM_ROWS):
        for c in range(ROOM_COLS - 1):
            r0, _c0, r1, c1 = rects[(r, c)]
            wall_c = c1 + 1
            grid[r0 : r1 + 1, wall_c] = GREY
    for r in range(ROOM_ROWS - 1):
        for c in range(ROOM_COLS):
            _r0, c0, r1, c1 = rects[(r, c)]
            wall_r = r1 + 1
            grid[wall_r, c0 : c1 + 1] = GREY


def _cut_door(grid: np.ndarray, rects: dict[tuple[int, int], tuple[int, int, int, int]], edge: tuple) -> None:
    if edge[0] == EXT:
        _ext, cell, side = edge
        r0, c0, r1, c1 = rects[cell]
        if side == "left":
            rr = random.randint(r0, r1)
            grid[rr, c0 - 1] = BLACK
        elif side == "right":
            rr = random.randint(r0, r1)
            grid[rr, c1 + 1] = BLACK
        elif side == "top":
            cc = random.randint(c0, c1)
            grid[r0 - 1, cc] = BLACK
        else:
            cc = random.randint(c0, c1)
            grid[r1 + 1, cc] = BLACK
        return

    a, b = edge
    ar0, ac0, ar1, ac1 = rects[a]
    br0, bc0, br1, bc1 = rects[b]
    if ar0 == br0 and ar1 == br1:
        wall_c = min(ac1, bc1) + 1
        rr = random.randint(ar0, ar1)
        grid[rr, wall_c] = BLACK
    else:
        wall_r = min(ar1, br1) + 1
        cc = random.randint(ac0, ac1)
        grid[wall_r, cc] = BLACK


def _graph_distances(edges: list[tuple]) -> dict[tuple[int, int], int]:
    adjacency: dict[tuple, set[tuple]] = {EXT: set()}
    for edge in edges:
        if edge[0] == EXT:
            _, node, _side = edge
            adjacency.setdefault(EXT, set()).add(node)
            adjacency.setdefault(node, set()).add(EXT)
        else:
            a, b = edge
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    queue = deque([EXT])
    distances = {EXT: 0}
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt in distances:
                continue
            distances[nxt] = distances[cur] + 1
            queue.append(nxt)

    return {node: dist for node, dist in distances.items() if node != EXT}


def _count_simple_paths(edges: list[tuple], target: tuple[int, int], cutoff: int = 8) -> int:
    adjacency: dict[tuple, set[tuple]] = {EXT: set()}
    for edge in edges:
        if edge[0] == EXT:
            _, node, _side = edge
            adjacency.setdefault(EXT, set()).add(node)
            adjacency.setdefault(node, set()).add(EXT)
        else:
            a, b = edge
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    count = 0

    def dfs(node: tuple, seen: set[tuple], depth: int) -> None:
        nonlocal count
        if depth > cutoff or count >= 3:
            return
        if node == target:
            count += 1
            return
        for nxt in adjacency[node]:
            if nxt in seen:
                continue
            dfs(nxt, seen | {nxt}, depth + 1)

    dfs(EXT, {EXT}, 0)
    return count


def _expected_output(grid: np.ndarray, rects: dict[tuple[int, int], tuple[int, int, int, int]], edges: list[tuple]) -> np.ndarray:
    out = grid.copy()
    distances = _graph_distances(edges)
    for node in ROOM_NODES:
        depth = distances.get(node)
        if depth not in DEPTH_TO_COLOR:
            continue
        r0, c0, r1, c1 = rects[node]
        out[r0 : r1 + 1, c0 : c1 + 1] = DEPTH_TO_COLOR[depth]
    return out


def _apply_symmetry(grid: np.ndarray, k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, k)
    if flip_lr:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _sample_edges() -> list[tuple]:
    edges = [
        (EXT, (0, 0), "left"),
        (EXT, (2, 1), "bottom"),
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((2, 1), (2, 2)),
        ((2, 2), (1, 2)),
        ((1, 2), (0, 2)),
    ]

    optionals = [
        ((2, 0), (2, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 2)),
        ((2, 2), (2, 3)),
    ]
    random.shuffle(optionals)
    for edge in optionals:
        if random.random() < 0.55:
            edges.append(edge)
    return edges


def _sample_layout() -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], list[tuple]]:
    row_heights = [random.randint(3, 5) for _ in range(ROOM_ROWS)]
    col_widths = [random.randint(3, 5) for _ in range(ROOM_COLS)]

    top = random.choice([0, 0, 1, 2])
    right = random.choice([0, 0, 1, 2])
    left = random.randint(1, 3)
    bottom = random.randint(1, 3)
    if random.random() < 0.65:
        if random.random() < 0.5:
            top = 0
        else:
            right = 0

    build_h = sum(row_heights) + ROOM_ROWS + 1
    build_w = sum(col_widths) + ROOM_COLS + 1
    grid_h = top + build_h + bottom
    grid_w = left + build_w + right
    assert grid_h <= 30 and grid_w <= 30

    grid = np.zeros((grid_h, grid_w), dtype=int)
    rects = _cell_rects(row_heights, col_widths, top, left)
    _build_shell(grid, rects)
    edges = _sample_edges()
    for edge in edges:
        _cut_door(grid, rects, edge)
    return grid, rects, edges


def _validate_candidate(inp: np.ndarray, expected: np.ndarray, edges: list[tuple]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) != {BLACK, GREY}:
        return False

    distances = _graph_distances(edges)
    required_depths = {1, 2, 3, 4}
    if not required_depths <= set(distances.values()):
        return False
    if any(depth > 4 for depth in distances.values()):
        return False
    if (1, 3) in distances:
        return False
    if _count_simple_paths(edges, (0, 2)) < 2:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    _labels, _components, solver_distances, doors = _door_depths(inp)
    if len(doors) != len(edges):
        return False
    if max(solver_distances.values(), default=0) > 4:
        return False
    if not np.any(actual == YELLOW):
        return False
    if not np.any(actual == BLACK):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, rects, edges = _sample_layout()
        expected = _expected_output(grid, rects, edges)
        k = random.randrange(4)
        flip_lr = random.random() < 0.5
        grid = _apply_symmetry(grid, k, flip_lr)
        expected = _apply_symmetry(expected, k, flip_lr)
        if _validate_candidate(grid, expected, edges):
            return grid
    raise RuntimeError("failed to generate a valid door-depth puzzle")
