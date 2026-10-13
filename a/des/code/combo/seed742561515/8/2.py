from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


AZURE = 8
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

TRACK_COLORS = (RED, GREEN, YELLOW, MAGENTA)
SPECIAL_COLORS = {MAROON, ORANGE, GREY}

# --------------- solver ---------------

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components_for_colors(grid: np.ndarray, colors: set[int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) not in colors:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _scan_to_marker(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        color = int(grid[r, c])
        if color == AZURE:
            r += dr
            c += dc
            continue
        assert color in SPECIAL_COLORS, "segment extensions may only end at special markers"
        return (r, c)
    raise AssertionError("segment extension must hit a marker before leaving the grid")


def _edge_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells: list[tuple[int, int]] = []
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac + step, bc, step):
            cells.append((ar, c))
    else:
        assert ac == bc
        step = 1 if br > ar else -1
        for r in range(ar + step, br, step):
            cells.append((r, ac))
    return cells


def _extract_edges(grid: np.ndarray) -> list[dict]:
    edges: list[dict] = []
    for coords in _components_for_colors(grid, set(TRACK_COLORS)):
        color = int(grid[coords[0]])
        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        assert len(rows) == 1 or len(cols) == 1, "track components must be straight"
        if len(rows) == 1:
            row = next(iter(rows))
            ordered = sorted(c for _r, c in coords)
            assert ordered == list(range(ordered[0], ordered[-1] + 1)), "horizontal segment must be contiguous"
            left = (row, ordered[0])
            right = (row, ordered[-1])
            a = _scan_to_marker(grid, left, (0, -1))
            b = _scan_to_marker(grid, right, (0, 1))
        else:
            col = next(iter(cols))
            ordered = sorted(r for r, _c in coords)
            assert ordered == list(range(ordered[0], ordered[-1] + 1)), "vertical segment must be contiguous"
            top = (ordered[0], col)
            bottom = (ordered[-1], col)
            a = _scan_to_marker(grid, top, (-1, 0))
            b = _scan_to_marker(grid, bottom, (1, 0))
        assert a != b
        between = _edge_cells(a, b)
        assert between, "each track must span at least one interior cell"
        segment_set = set(coords)
        for cell in between:
            cell_color = int(grid[cell])
            if cell in segment_set:
                assert cell_color == color
            else:
                assert cell_color == AZURE
        edges.append({"a": a, "b": b, "color": color, "cells": between})
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each maroon source through colored straight guides, switching color at orange turn nodes.

    Input structure
    - The background is azure.
    - Red/green/yellow/magenta monochrome straight segments sit on rows or columns.
    - Maroon endpoints are sources, grey endpoints are terminals, and orange pixels are turn nodes.

    Rule
    - Each visible straight segment defines the color and orientation of one path leg.
    - Extend that leg through intervening azure cells until it reaches the special markers at both ends.
    - Starting from every maroon source, fill only azure cells along its full route, recolor the source to the
      first leg color, switch to the next leg color at each orange node, and stop at grey.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h > 0 and w > 0
    assert np.all(np.isin(g, [AZURE, *TRACK_COLORS, ORANGE, GREY, MAROON]))

    markers = {tuple(map(int, coord)): int(g[tuple(coord)]) for coord in np.argwhere(np.isin(g, list(SPECIAL_COLORS)))}
    assert markers, "expected at least one special marker"

    edges = _extract_edges(g)
    assert edges, "expected at least one track segment"

    adjacency: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        adjacency[edge["a"]].append(i)
        adjacency[edge["b"]].append(i)

    sources = [coord for coord, color in markers.items() if color == MAROON]
    terminals = [coord for coord, color in markers.items() if color == GREY]
    turns = [coord for coord, color in markers.items() if color == ORANGE]
    assert sources and terminals

    for coord in sources + terminals:
        assert len(adjacency[coord]) == 1, "sources and terminals must be degree-1"
    for coord in turns:
        assert len(adjacency[coord]) == 2, "turn nodes must be degree-2"

    out = g.copy()
    used_edges: set[int] = set()
    for source in sources:
        prev_edge = None
        cur = source
        while True:
            incident = adjacency[cur]
            if prev_edge is None:
                assert len(incident) == 1
                edge_id = incident[0]
            else:
                next_edges = [edge_id for edge_id in incident if edge_id != prev_edge]
                assert len(next_edges) == 1
                edge_id = next_edges[0]
            edge = edges[edge_id]
            used_edges.add(edge_id)
            for cell in edge["cells"]:
                if int(out[cell]) == AZURE:
                    out[cell] = edge["color"]
            if int(out[cur]) == MAROON:
                out[cur] = edge["color"]
            nxt = edge["b"] if edge["a"] == cur else edge["a"]
            if markers[nxt] == GREY:
                break
            assert markers[nxt] == ORANGE
            prev_edge = edge_id
            cur = nxt

    assert len(used_edges) == len(edges), "all segments should belong to some source-to-terminal path"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
PATH_COUNT_CHOICES = (1, 2, 2, 3)
TURN_COUNT_CHOICES = (0, 1, 1, 2, 2, 3)
EDGE_LEN_RANGE = (4, 6)
PLACEMENT_PADDING = 1
GEN_RETRIES = 300


def _sample_lengths(turns: int) -> tuple[int, list[int], list[int]]:
    start_axis = random.randrange(2)
    nseg = turns + 1
    horiz = (nseg + (1 if start_axis == 0 else 0)) // 2
    vert = nseg - horiz
    lengths = [random.randint(*EDGE_LEN_RANGE) for _ in range(nseg)]
    return start_axis, lengths, [horiz, vert]


def _build_polyline(turns: int) -> list[tuple[int, int]]:
    start_axis, lengths, _counts = _sample_lengths(turns)
    vertices = [(0, 0)]
    r = c = 0
    for i, length in enumerate(lengths):
        axis = (start_axis + i) % 2
        if axis == 0:
            c += length
        else:
            r += length
        vertices.append((r, c))

    max_r = max(r for r, _c in vertices)
    max_c = max(c for _r, c in vertices)
    if random.random() < 0.5:
        vertices = [(max_r - r, c) for r, c in vertices]
    if random.random() < 0.5:
        vertices = [(r, max_c - c) for r, c in vertices]
    return vertices


def _bbox_of_points(points: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in points) + 1, max(c for _r, c in points) + 1


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(mask.shape[0], top + height + PLACEMENT_PADDING)
    c1 = min(mask.shape[1], left + width + PLACEMENT_PADDING)
    mask[r0:r1, c0:c1] = True


def _place_polyline(
    occupied: np.ndarray,
    vertices: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = occupied.shape
    box_h, box_w = _bbox_of_points(vertices)
    if box_h > h or box_w > w:
        return None
    positions = [(top, left) for top in range(h - box_h + 1) for left in range(w - box_w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if occupied[max(0, top - PLACEMENT_PADDING) : min(h, top + box_h + PLACEMENT_PADDING),
                    max(0, left - PLACEMENT_PADDING) : min(w, left + box_w + PLACEMENT_PADDING)].any():
            continue
        translated = [(top + r, left + c) for r, c in vertices]
        _reserve(occupied, top, left, box_h, box_w)
        return translated
    return None


def _edge_interior(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return _edge_cells(a, b)


def _draw_path(grid: np.ndarray, vertices: list[tuple[int, int]]) -> None:
    colors: list[int] = []
    for i in range(len(vertices) - 1):
        options = [color for color in TRACK_COLORS if not colors or color != colors[-1]]
        colors.append(random.choice(options))

    for i, coord in enumerate(vertices):
        if i == 0:
            grid[coord] = MAROON
        elif i == len(vertices) - 1:
            grid[coord] = GREY
        else:
            grid[coord] = ORANGE

    for i, (a, b) in enumerate(zip(vertices, vertices[1:])):
        interior = _edge_interior(a, b)
        assert len(interior) >= 3
        seg_len = random.randint(2, len(interior) - 1)
        start = random.randint(0, len(interior) - seg_len)
        for cell in interior[start : start + seg_len]:
            grid[cell] = colors[i]


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), AZURE, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_paths = random.choice(PATH_COUNT_CHOICES)
    if min(h, w) < 18 and n_paths == 3:
        n_paths = 2

    for _path_idx in range(n_paths):
        placed = None
        for _ in range(80):
            turns = random.choice(TURN_COUNT_CHOICES)
            if min(h, w) < 17:
                turns = min(turns, 2)
            polyline = _build_polyline(turns)
            placed = _place_polyline(occupied, polyline)
            if placed is not None:
                _draw_path(grid, placed)
                break
        if placed is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
