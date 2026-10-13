"""ARC task 8: relay colors through yellow doors between red-walled regions."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
PAINT_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _label_open_regions(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Label 4-connected cells that are not red walls or yellow doors."""
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if labels[sr, sc] >= 0 or int(grid[sr, sc]) in (RED, YELLOW):
                continue
            label = len(regions)
            cells: list[tuple[int, int]] = []
            labels[sr, sc] = label
            queue = deque([(sr, sc)])
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and labels[nr, nc] < 0
                            and int(grid[nr, nc]) not in (RED, YELLOW)):
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            regions.append(cells)
    return labels, regions


def _read_strip(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Return the unique horizontal strip and its left-to-right visible sequence."""
    mask = ~np.isin(grid, (BLACK, RED, YELLOW))
    points = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(points) >= 2, "expected a color strip of length at least two"
    rows = {r for r, _ in points}
    assert len(rows) == 1, "all paint pixels must form one horizontal strip"
    row = next(iter(rows))
    cols = sorted(c for _, c in points)
    assert cols == list(range(cols[0], cols[-1] + 1)), "strip must be contiguous"
    colors = [int(grid[row, c]) for c in cols]
    assert len(colors) == len(set(colors)), "strip colors must be mutually distinct"
    return colors, [(row, c) for c in cols]


def _region_graph(grid: np.ndarray, labels: np.ndarray, count: int) -> list[set[int]]:
    """Make one undirected graph edge from each valid single-pixel yellow door."""
    h, w = grid.shape
    graph = [set() for _ in range(count)]
    for r, c in np.argwhere(grid == YELLOW):
        r, c = int(r), int(c)
        vertical = []
        horizontal = []
        if r > 0 and r + 1 < h:
            vertical = [int(labels[r - 1, c]), int(labels[r + 1, c])]
        if c > 0 and c + 1 < w:
            horizontal = [int(labels[r, c - 1]), int(labels[r, c + 1])]
        pairs = [pair for pair in (vertical, horizontal)
                 if len(pair) == 2 and pair[0] >= 0 and pair[1] >= 0
                 and pair[0] != pair[1]]
        assert len(pairs) == 1, "each yellow door must join exactly two opposite regions"
        a, b = pairs[0]
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _distances(graph: list[set[int]], source: int) -> list[int]:
    dist = [-1] * len(graph)
    dist[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if dist[nxt] < 0:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Fill red-walled regions cyclically by their shortest yellow-door distance from a color strip.

    Input structure:
    - Black open regions are completely separated by red horizontal and vertical walls.
    - Single yellow pixels embedded in walls are doors, each joining exactly the two
      regions on opposite sides; exactly one region contains a horizontal strip of
      mutually distinct paint colors.

    Rule:
    - Regard each open region as a graph node and each yellow door as one edge.
    - If the strip has colors left-to-right ``c[0]...c[k-1]``, fill every region at
      minimum door distance ``d`` from the strip region with ``c[d mod k]``.
      Cover the strip itself, preserve red walls and yellow doors, and leave regions
      unreachable through doors black.

    Key insight:
    - Flooding occurs layer by layer on the region-adjacency graph, not through wall
      pixels; the paint phase is the shortest number of crossed doors modulo the
      visibly ordered strip length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"

    colors, strip_cells = _read_strip(g)
    labels, regions = _label_open_regions(g)
    source_labels = {int(labels[r, c]) for r, c in strip_cells}
    assert len(source_labels) == 1, "the entire strip must lie in one region"
    source = next(iter(source_labels))
    graph = _region_graph(g, labels, len(regions))
    dist = _distances(graph, source)

    out = g.copy()
    for region, cells in enumerate(regions):
        d = dist[region]
        if d < 0:
            assert all(int(g[r, c]) == BLACK for r, c in cells), (
                "only the source strip may contain nonblack open cells")
            continue
        color = colors[d % len(colors)]
        for r, c in cells:
            out[r, c] = color
    return out


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if r + 1 < rows:
                edges.append((node, (r + 1) * cols + c))
            if c + 1 < cols:
                edges.append((node, node + 1))
    return edges


def _graph_from_edges(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    graph = [set() for _ in range(n)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _choose_door_graph(rows: int, cols: int, k: int):
    """Sample a nondegenerate graph: cycle, branch, long tail, and unreachable node."""
    all_edges = _grid_edges(rows, cols)
    n = rows * cols
    for _ in range(500):
        edges = [edge for edge in all_edges if random.random() < random.uniform(0.38, 0.62)]
        graph = _graph_from_edges(n, edges)
        candidates = []
        for source in range(n):
            dist = _distances(graph, source)
            reached = [i for i, d in enumerate(dist) if d >= 0]
            unreachable = [i for i, d in enumerate(dist) if d < 0]
            internal_edges = sum(len(graph[i]) for i in reached) // 2
            has_cycle = internal_edges >= len(reached)
            has_branch = any(len(graph[i]) >= 3 for i in reached)
            if (unreachable and max(dist) >= k and has_cycle and has_branch
                    and len(reached) >= k + 3):
                # At least one door must affect distances/fills beyond its own pixel.
                critical = []
                for edge in edges:
                    a, b = edge
                    if a not in reached:
                        continue
                    reduced = [e for e in edges if e != edge]
                    d2 = _distances(_graph_from_edges(n, reduced), source)
                    if any(d2[i] != dist[i] for i in reached):
                        critical.append(edge)
                if critical:
                    candidates.append((source, dist, critical))
        if candidates:
            source, dist, critical = random.choice(candidates)
            return edges, source, dist, random.choice(critical)
    return None


def _validate_generated(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Check the stated semantics independently from merely calling main on the input."""
    try:
        labels, regions = _label_open_regions(grid)
        colors, strip = _read_strip(grid)
        graph = _region_graph(grid, labels, len(regions))
        source = int(labels[strip[0]])
        dist = _distances(graph, source)
        if colors != meta["colors"] or source != meta["source"] or dist != meta["dist"]:
            return False
        if not any(d < 0 for d in dist) or max(dist) < len(colors):
            return False
        if not all(any(d == phase for d in dist) for phase in range(len(colors))):
            return False
        if not np.array_equal(expected[grid == RED], grid[grid == RED]):
            return False
        if not np.array_equal(expected[grid == YELLOW], grid[grid == YELLOW]):
            return False
        for i, cells in enumerate(regions):
            values = {int(expected[r, c]) for r, c in cells}
            wanted = BLACK if dist[i] < 0 else colors[dist[i] % len(colors)]
            if values != {wanted}:
                return False

        # Removing the selected critical door must change at least one region's
        # semantic fill, ignoring the changed door pixel itself.
        door = meta["door_cells"][meta["critical"]]
        altered = grid.copy()
        altered[door] = RED
        changed = main(altered)
        mask = np.ones(grid.shape, dtype=bool)
        mask[door] = False
        if np.array_equal(changed[mask], expected[mask]):
            return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.choice((3, 4))
    room_cols = random.choice((3, 4))
    heights = [random.randint(3, 5) for _ in range(room_rows)]
    widths = [random.randint(4, 6) for _ in range(room_cols)]
    h = sum(heights) + room_rows - 1
    w = sum(widths) + room_cols - 1
    if max(h, w) > 30:
        return None

    k = random.choice((2, 3))
    chosen = _choose_door_graph(room_rows, room_cols, k)
    if chosen is None:
        return None
    edges, source, dist, critical = chosen

    row_starts = []
    pos = 0
    for size in heights:
        row_starts.append(pos)
        pos += size + 1
    col_starts = []
    pos = 0
    for size in widths:
        col_starts.append(pos)
        pos += size + 1

    grid = np.zeros((h, w), dtype=int)
    for r in range(room_rows - 1):
        grid[row_starts[r] + heights[r], :] = RED
    for c in range(room_cols - 1):
        grid[:, col_starts[c] + widths[c]] = RED

    door_cells: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b in edges:
        ar, ac = divmod(a, room_cols)
        br, bc = divmod(b, room_cols)
        if ar == br:  # vertical wall between left and right rooms
            wall_c = col_starts[min(ac, bc)] + widths[min(ac, bc)]
            door = (random.randrange(row_starts[ar], row_starts[ar] + heights[ar]), wall_c)
        else:  # horizontal wall between upper and lower rooms
            wall_r = row_starts[min(ar, br)] + heights[min(ar, br)]
            door = (wall_r, random.randrange(col_starts[ac], col_starts[ac] + widths[ac]))
        grid[door] = YELLOW
        door_cells[(a, b)] = door
        door_cells[(b, a)] = door

    sr, sc = divmod(source, room_cols)
    colors = random.sample(PAINT_COLORS, k)
    strip_r = random.randrange(row_starts[sr], row_starts[sr] + heights[sr])
    first_c = random.randrange(col_starts[sc], col_starts[sc] + widths[sc] - k + 1)
    for offset, color in enumerate(colors):
        grid[strip_r, first_c + offset] = color

    expected = grid.copy()
    for node, d in enumerate(dist):
        rr, cc = divmod(node, room_cols)
        rs = slice(row_starts[rr], row_starts[rr] + heights[rr])
        cs = slice(col_starts[cc], col_starts[cc] + widths[cc])
        expected[rs, cs] = BLACK if d < 0 else colors[d % k]

    meta = {"colors": colors, "source": source, "dist": dist,
            "critical": critical, "door_cells": door_cells}
    if not np.array_equal(main(grid), expected):
        return None
    if not _validate_generated(grid, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, semantically checked red-wall/yellow-door relay instances."""
    for _ in range(200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a nondegenerate yellow-door relay input")
