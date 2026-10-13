from __future__ import annotations

import random

import numpy as np


BLACK = 0
ROAD = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MARKER_COLORS = tuple(color for color in range(10) if color not in {BLACK, ROAD})

MIN_LINES = 3
MAX_LINES = 4
ODD_GAPS = (3, 5)
EVEN_GAPS = (4,)
ALL_GAPS = ODD_GAPS + EVEN_GAPS
MAX_EXTENSION = 3
GEN_ATTEMPTS = 400


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def _neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.append((int(rr), int(cc)))
    return out


def _is_marker(value: int) -> bool:
    return int(value) not in {BLACK, ROAD}


def _network_data(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], list[tuple[int, int]]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    road = grid != BLACK
    road_cells = [tuple(map(int, rc)) for rc in np.argwhere(road)]
    neighbor_map = {cell: _neighbors(road, cell) for cell in road_cells}

    nodes: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    for cell in road_cells:
        neighbors = neighbor_map[cell]
        degree = len(neighbors)
        value = int(grid[cell])
        if _is_marker(value):
            nodes.add(cell)
            protected.add(cell)
            continue
        if degree != 2:
            nodes.add(cell)
            if degree >= 3:
                protected.add(cell)
            continue
        a, b = neighbors
        if a[0] != b[0] and a[1] != b[1]:
            nodes.add(cell)

    return road, neighbor_map, nodes, protected


def _extract_segments(
    grid: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    _road, neighbor_map, nodes, protected = _network_data(grid)
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []

    for start in sorted(nodes):
        for nxt in neighbor_map[start]:
            edge = _edge_key(start, nxt)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [start]
            prev = start
            curr = nxt
            while True:
                path.append(curr)
                if curr in nodes:
                    break
                options = [cell for cell in neighbor_map[curr] if cell != prev]
                assert len(options) == 1, "channel interiors must form simple segments"
                nxt2 = options[0]
                edge = _edge_key(curr, nxt2)
                if edge in visited_edges:
                    raise AssertionError("unexpected loop while tracing a channel segment")
                visited_edges.add(edge)
                prev, curr = curr, nxt2
            segments.append(path)

    return segments, nodes, protected


def _paintable_cells(path: list[tuple[int, int]], protected: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = list(path)
    if path[0] in protected:
        cells = cells[1:]
    if path[-1] in protected:
        cells = cells[:-1]
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject each intersection marker color into the connected grey channels and split marked-to-marked corridors at their midpoint.

    Input structure:
    - The background is black.
    - Grey pixels form single-cell horizontal and vertical channels, with cross intersections and occasional dead-end stubs.
    - Some intersection centers are replaced by colored marker pixels; unmarked intersections stay grey.

    Rule:
    - Treat every non-black pixel as part of one orthogonal road network and decompose it into channel segments between topological nodes.
    - Colored intersection markers push their color through every incident segment until the next intersection or a dead end.
    - If both endpoint intersections of a segment are marked, color the half nearer each endpoint with that endpoint's color; when the number of paintable cells is odd, leave the middle cell grey.
    - If exactly one endpoint is marked, color every non-intersection cell on that segment with the marker color, including a dead-end tip but excluding an unmarked intersection endpoint.
    - Keep marked intersections colored and keep unmarked intersections grey.

    Key insight:
    - The propagation is not a global flood fill: each straight channel segment is solved locally from the marker status of its two endpoints, while intersections themselves stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    segments, _nodes, protected = _extract_segments(g)

    for path in segments:
        start = path[0]
        end = path[-1]
        start_marked = _is_marker(int(g[start]))
        end_marked = _is_marker(int(g[end]))
        paint_cells = _paintable_cells(path, protected)
        if not paint_cells:
            continue

        if start_marked and end_marked:
            half = len(paint_cells) // 2
            for cell in paint_cells[:half]:
                out[cell] = int(g[start])
            for cell in paint_cells[len(paint_cells) - half :]:
                out[cell] = int(g[end])
        elif start_marked or end_marked:
            color = int(g[start] if start_marked else g[end])
            for cell in paint_cells:
                out[cell] = color

    return out


def _gap_sequence(count: int, first_choices: tuple[int, ...]) -> list[int]:
    gaps = [random.choice(first_choices)]
    for _ in range(count - 1):
        gaps.append(random.choice(ALL_GAPS))
    return gaps


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.asarray(out, dtype=int)


def _segment_case_flags(grid: np.ndarray) -> tuple[bool, bool, bool, bool]:
    odd_split = False
    even_split = False
    marked_to_unmarked = False
    marked_to_dead = False

    segments, _nodes, protected = _extract_segments(grid)
    for path in segments:
        start = path[0]
        end = path[-1]
        start_marked = _is_marker(int(grid[start]))
        end_marked = _is_marker(int(grid[end]))
        paint_cells = _paintable_cells(path, protected)
        if not paint_cells:
            continue

        if start_marked and end_marked:
            if len(paint_cells) % 2 == 1:
                odd_split = True
            else:
                even_split = True
        elif start_marked ^ end_marked:
            if start in protected and end in protected:
                marked_to_unmarked = True
            else:
                marked_to_dead = True

    return odd_split, even_split, marked_to_unmarked, marked_to_dead


def _gen_attempt() -> np.ndarray | None:
    num_rows = random.randint(MIN_LINES, MAX_LINES)
    num_cols = random.randint(MIN_LINES, MAX_LINES)

    row_gaps = _gap_sequence(num_rows - 1, ODD_GAPS)
    col_gaps = _gap_sequence(num_cols - 1, EVEN_GAPS)
    top_ext = [random.randint(1, MAX_EXTENSION)] + [random.randint(0, MAX_EXTENSION) for _ in range(num_cols - 1)]
    bottom_ext = [random.randint(0, MAX_EXTENSION) for _ in range(num_cols)]
    left_ext = [random.randint(0, MAX_EXTENSION) for _ in range(num_rows)]
    right_ext = [random.randint(0, MAX_EXTENSION) for _ in range(num_rows)]

    row_base = [0]
    for gap in row_gaps:
        row_base.append(row_base[-1] + gap)
    col_base = [0]
    for gap in col_gaps:
        col_base.append(col_base[-1] + gap)

    top_pad = max(top_ext) + random.randint(1, 2)
    left_pad = max(left_ext) + random.randint(1, 2)
    rows = [top_pad + value for value in row_base]
    cols = [left_pad + value for value in col_base]

    height = rows[-1] + max(bottom_ext) + random.randint(2, 3)
    width = cols[-1] + max(right_ext) + random.randint(2, 3)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for idx, c in enumerate(cols):
        r0 = rows[0] - top_ext[idx]
        r1 = rows[-1] + bottom_ext[idx]
        grid[r0 : r1 + 1, c] = ROAD
    for idx, r in enumerate(rows):
        c0 = cols[0] - left_ext[idx]
        c1 = cols[-1] + right_ext[idx]
        grid[r, c0 : c1 + 1] = ROAD

    blocked = {
        (rows[-1], cols[1]),
    }
    markers = [
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[0], cols[-1]),
        (rows[1], cols[-1]),
        (rows[-1], cols[0]),
    ]
    blocked.update(markers)

    all_intersections = [(r, c) for r in rows for c in cols]
    extras = [cell for cell in all_intersections if cell not in blocked]
    random.shuffle(extras)
    extra_count = random.randint(0, min(2, len(extras)))
    markers.extend(extras[:extra_count])

    colors = random.sample(MARKER_COLORS, len(markers))
    for cell, color in zip(markers, colors, strict=False):
        grid[cell] = color

    grid = _apply_random_symmetry(grid)

    odd_split, even_split, marked_to_unmarked, marked_to_dead = _segment_case_flags(grid)
    if not (odd_split and even_split and marked_to_unmarked and marked_to_dead):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-injection puzzle")
