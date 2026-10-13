from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 2
GREEN = 3
YELLOW = 4
FRAME = 5

GEN_ATTEMPTS = 400
ROW_CELL_COUNTS = (3, 4, 5)
COL_CELL_COUNTS = (3, 4, 5)
CELL_SIZE_MIN = 2
CELL_SIZE_MAX = 4
OPEN_PROBABILITY = 0.0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one gray rectangular frame"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have non-empty interior"
    assert (r1 - r0) != (c1 - c0), "frame must be non-square"
    assert 0 < r0 < r1 < grid.shape[0] - 1, "frame cannot lie on the outer border"
    assert 0 < c0 < c1 < grid.shape[1] - 1, "frame cannot lie on the outer border"

    frame = np.zeros_like(mask, dtype=bool)
    frame[r0, c0 : c1 + 1] = True
    frame[r1, c0 : c1 + 1] = True
    frame[r0 : r1 + 1, c0] = True
    frame[r0 : r1 + 1, c1] = True
    assert np.array_equal(mask, frame), "gray pixels must form exactly one 1-pixel rectangular frame"
    return r0, c0, r1, c1


def _label_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    regions: list[np.ndarray] = [np.empty((0, 2), dtype=int)]
    next_id = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or labels[r, c] != 0:
                continue

            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))

            regions.append(np.array(coords, dtype=int))

    assert next_id >= 2, "walls must split the background into multiple regions"
    return labels, regions


def _build_region_graph(grid: np.ndarray, labels: np.ndarray, n_regions: int) -> list[set[int]]:
    h, w = grid.shape
    graph = [set() for _ in range(n_regions + 1)]

    for r in range(h):
        for c in range(w - 2):
            left = int(labels[r, c])
            right = int(labels[r, c + 2])
            if left > 0 and right > 0 and left != right and grid[r, c + 1] != BG:
                graph[left].add(right)
                graph[right].add(left)

    for r in range(h - 2):
        for c in range(w):
            top = int(labels[r, c])
            bottom = int(labels[r + 2, c])
            if top > 0 and bottom > 0 and top != bottom and grid[r + 1, c] != BG:
                graph[top].add(bottom)
                graph[bottom].add(top)

    assert any(graph[rid] for rid in range(1, n_regions + 1)), "expected wall-induced region adjacencies"
    return graph


def _bipartite_parity(graph: list[set[int]], start_region: int) -> list[int]:
    parity = [-1] * len(graph)
    parity[start_region] = 0
    q = deque([start_region])

    while q:
        region = q.popleft()
        for neighbor in graph[region]:
            expected = 1 - parity[region]
            if parity[neighbor] == -1:
                parity[neighbor] = expected
                q.append(neighbor)
            else:
                assert parity[neighbor] == expected, "region adjacency graph must be bipartite"

    assert all(value != -1 for value in parity[1:]), "all regions must share one wall-adjacency graph"
    return parity


def _region_strictly_inside(coords: np.ndarray, frame_box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame_box
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()) > r0 and int(rows.max()) < r1 and int(cols.min()) > c0 and int(cols.max()) < c1


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[np.ndarray], list[set[int]], list[int], set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_box = _find_frame(g)
    labels, regions = _label_black_regions(g)
    n_regions = len(regions) - 1

    start_region = int(labels[g.shape[0] - 1, 0])
    assert start_region > 0, "the lower-left corner must belong to a black background region"

    graph = _build_region_graph(g, labels, n_regions)
    parity = _bipartite_parity(graph, start_region)

    inside = {
        region_id
        for region_id in range(1, n_regions + 1)
        if _region_strictly_inside(regions[region_id], frame_box)
    }
    assert len(inside) >= 2, "the frame interior must contain at least two complete regions"
    return g, frame_box, regions, graph, parity, inside


def _render_coloring(
    grid: np.ndarray, regions: list[np.ndarray], parity: list[int], inside: set[int], *, flip_inside: bool
) -> np.ndarray:
    out = grid.copy()
    for region_id in range(1, len(regions)):
        color = GREEN if parity[region_id] == 0 else YELLOW
        if flip_inside and region_id in inside:
            color = YELLOW if color == GREEN else GREEN
        coords = regions[region_id]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Alternately color wall-separated regions, then flip the coloring polarity only for regions strictly inside the gray frame.

    Input structure:
    - The grid uses black background cells, red one-cell-thick wall segments, and one gray one-cell-thick rectangular frame.
    - Red walls split the black background into 4-connected regions whose wall-adjacency graph is bipartite.
    - The lower-left corner belongs to one black region, and the gray frame sits away from the outer border with at least two full regions strictly inside it.

    Rule:
    - Treat red wall pixels and gray frame pixels as impassable, and identify all 4-connected black regions.
    - Build the region adjacency graph across one-cell-thick wall separators, then color the lower-left region green and every wall-adjacent region with alternating green/yellow parity.
    - Keep only whole regions whose every cell lies strictly inside the gray frame interior, and flip their parity colors after the global coloring.
    - Preserve every red wall pixel and every gray frame pixel unchanged.

    Key insight:
    - The walls determine the global bipartite coloring, while the rectangle acts only as a local polarity switch for the complete regions it encloses.
    """
    g, _frame_box, regions, _graph, parity, inside = _analyze(np.asarray(grid, dtype=int))
    return _render_coloring(g, regions, parity, inside, flip_inside=True)


def _sample_cell_sizes(count: int) -> list[int]:
    return [random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(count)]


def _partition_from_sizes(sizes: list[int]) -> tuple[int, list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    walls: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx != len(sizes) - 1:
            walls.append(cursor)
            cursor += 1
    return cursor, spans, walls


def _paint_base_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    for row in wall_rows:
        grid[row, :] = WALL
    for col in wall_cols:
        grid[:, col] = WALL


def _carve_openings(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
) -> None:
    for col in wall_cols:
        for r0, r1 in row_spans:
            if random.random() < OPEN_PROBABILITY:
                grid[r0 : r1 + 1, col] = BG

    for row in wall_rows:
        for c0, c1 in col_spans:
            if random.random() < OPEN_PROBABILITY:
                grid[row, c0 : c1 + 1] = BG


def _compute_partition_analysis(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[set[int]], list[int]]:
    labels, regions = _label_black_regions(grid)
    start_region = int(labels[grid.shape[0] - 1, 0])
    assert start_region > 0
    graph = _build_region_graph(grid, labels, len(regions) - 1)
    parity = _bipartite_parity(graph, start_region)
    return labels, regions, graph, parity


def _frame_candidates(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    regions: list[np.ndarray],
    parity: list[int],
) -> list[tuple[int, int, int, int, set[int]]]:
    candidates: list[tuple[int, int, int, int, set[int]]] = []
    n_regions = len(regions) - 1

    for top_idx, top in enumerate(wall_rows):
        for bottom in wall_rows[top_idx + 1 :]:
            for left_idx, left in enumerate(wall_cols):
                for right in wall_cols[left_idx + 1 :]:
                    if (bottom - top) == (right - left):
                        continue
                    if not np.all(grid[top, left : right + 1] == WALL):
                        continue
                    if not np.all(grid[bottom, left : right + 1] == WALL):
                        continue
                    if not np.all(grid[top : bottom + 1, left] == WALL):
                        continue
                    if not np.all(grid[top : bottom + 1, right] == WALL):
                        continue

                    inside = {
                        region_id
                        for region_id in range(1, n_regions + 1)
                        if _region_strictly_inside(regions[region_id], (top, left, bottom, right))
                    }
                    if len(inside) < 2 or len(inside) >= n_regions:
                        continue
                    if {parity[region_id] for region_id in inside} != {0, 1}:
                        continue
                    candidates.append((top, left, bottom, right, inside))
    return candidates


def _edge_crosses_frame(
    a_coords: np.ndarray, b_coords: np.ndarray, frame_box: tuple[int, int, int, int]
) -> bool:
    a_inside = _region_strictly_inside(a_coords, frame_box)
    b_inside = _region_strictly_inside(b_coords, frame_box)
    return a_inside != b_inside


def _has_same_color_cross_frame_edge(
    regions: list[np.ndarray],
    graph: list[set[int]],
    parity: list[int],
    inside: set[int],
    frame_box: tuple[int, int, int, int],
) -> bool:
    for region_id in range(1, len(regions)):
        for neighbor in graph[region_id]:
            if neighbor <= region_id:
                continue
            if not _edge_crosses_frame(regions[region_id], regions[neighbor], frame_box):
                continue

            first = parity[region_id]
            second = parity[neighbor]
            if region_id in inside:
                first = 1 - first
            if neighbor in inside:
                second = 1 - second
            if first == second:
                return True
    return False


def _attempt_input() -> np.ndarray | None:
    row_count = random.choice(ROW_CELL_COUNTS)
    col_count = random.choice(COL_CELL_COUNTS)

    h, row_spans, wall_rows = _partition_from_sizes(_sample_cell_sizes(row_count))
    w, col_spans, wall_cols = _partition_from_sizes(_sample_cell_sizes(col_count))
    if not wall_rows or not wall_cols:
        return None

    grid = np.zeros((h, w), dtype=int)
    _paint_base_walls(grid, wall_rows, wall_cols)
    _carve_openings(grid, row_spans, col_spans, wall_rows, wall_cols)

    try:
        _labels, regions, graph, parity = _compute_partition_analysis(grid)
    except AssertionError:
        return None

    candidates = _frame_candidates(grid, wall_rows, wall_cols, regions, parity)
    if not candidates:
        return None

    top, left, bottom, right, inside = random.choice(candidates)
    inp = grid.copy()
    inp[top, left : right + 1] = FRAME
    inp[bottom, left : right + 1] = FRAME
    inp[top : bottom + 1, left] = FRAME
    inp[top : bottom + 1, right] = FRAME

    try:
        analyzed = _analyze(inp)
        out = main(inp)
        base_out = _render_coloring(analyzed[0], analyzed[2], analyzed[4], analyzed[5], flip_inside=False)
    except AssertionError:
        return None

    g, frame_box, analyzed_regions, analyzed_graph, analyzed_parity, analyzed_inside = analyzed
    if analyzed_inside != inside:
        return None

    changed = (out != base_out) & (g == BG)
    if not changed.any():
        return None

    inside_mask = np.zeros(g.shape, dtype=bool)
    for region_id in analyzed_inside:
        coords = analyzed_regions[region_id]
        inside_mask[coords[:, 0], coords[:, 1]] = True
    if not np.array_equal(changed, inside_mask):
        return None

    if set(np.unique(base_out[inside_mask])) != {GREEN, YELLOW}:
        return None
    if not _has_same_color_cross_frame_edge(
        analyzed_regions, analyzed_graph, analyzed_parity, analyzed_inside, frame_box
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 9")
