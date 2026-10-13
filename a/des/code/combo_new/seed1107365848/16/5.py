from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ALL_COLORS = list(range(10))
MIN_SIZE = 12
MAX_SIZE = 24
MIN_TURNS = 1
MAX_TURNS = 6
MIN_SEG_LEN = 2
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(counts.most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4((h, w), rr, cc):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_edge(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _orthogonal_sides(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _path_candidate(
    grid: np.ndarray, bg: int
) -> tuple[int, set[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    candidates: list[tuple[int, set[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        for comp in _components_of_color(grid, int(color)):
            if len(comp) < 2:
                continue
            cells = set(comp)
            neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
            valid = True
            endpoints: list[tuple[int, int]] = []
            for cell in comp:
                nbrs = [nbr for nbr in _neighbors4(grid.shape, *cell) if nbr in cells]
                neighbor_map[cell] = nbrs
                if len(nbrs) not in (1, 2):
                    valid = False
                    break
                if len(nbrs) == 1:
                    endpoints.append(cell)
            if valid and len(endpoints) == 2 and all(_is_edge(cell, grid.shape) for cell in endpoints):
                candidates.append((int(color), cells, neighbor_map))
    assert len(candidates) == 1, "expected exactly one simple edge-to-edge path"
    return candidates[0]


def _ordered_path(
    start: tuple[int, int], neighbor_map: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        nxts = [nbr for nbr in neighbor_map[cur] if nbr != prev]
        if not nxts:
            return order
        assert len(nxts) == 1, "path must be unbranched"
        prev, cur = cur, nxts[0]


def _extract_controls(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int], dict[int, int]]:
    bg = _dominant_color(grid)
    _, path_cells, neighbor_map = _path_candidate(grid, bg)

    endpoints = [cell for cell, nbrs in neighbor_map.items() if len(nbrs) == 1]
    assert len(endpoints) == 2

    probe_path = _ordered_path(endpoints[0], neighbor_map)
    turn_marker_positions = set(_turn_marker_cells(probe_path))

    start_endpoint: tuple[int, int] | None = None
    start_marker: tuple[int, int] | None = None
    for endpoint in endpoints:
        next_cell = neighbor_map[endpoint][0]
        direction = (next_cell[0] - endpoint[0], next_cell[1] - endpoint[1])
        markers: list[tuple[int, int]] = []
        for sr, sc in _orthogonal_sides(direction):
            rr, cc = endpoint[0] + sr, endpoint[1] + sc
            if _in_bounds(grid.shape, rr, cc) and (rr, cc) not in path_cells and int(grid[rr, cc]) != bg:
                markers.append((rr, cc))
        usable = [marker for marker in markers if marker not in turn_marker_positions]
        if usable:
            assert len(usable) == 1, "start endpoint must have exactly one non-turn side marker"
            assert start_endpoint is None, "only one endpoint may carry the start marker"
            start_endpoint = endpoint
            start_marker = usable[0]

    assert start_endpoint is not None and start_marker is not None, "failed to locate start marker"
    path = probe_path if start_endpoint == endpoints[0] else list(reversed(probe_path))

    turn_colors: dict[int, int] = {}
    for idx in range(1, len(path) - 1):
        prev = path[idx - 1]
        cur = path[idx]
        nxt = path[idx + 1]
        v_prev = (cur[0] - prev[0], cur[1] - prev[1])
        v_next = (nxt[0] - cur[0], nxt[1] - cur[1])
        if v_prev == v_next:
            continue
        mr = cur[0] + v_prev[0] - v_next[0]
        mc = cur[1] + v_prev[1] - v_next[1]
        assert _in_bounds(grid.shape, mr, mc), "turn marker must stay inside the grid"
        assert (mr, mc) not in path_cells, "turn marker may not lie on the path"
        assert int(grid[mr, mc]) != bg, "turn marker must be present"
        turn_colors[idx] = int(grid[mr, mc])

    return path, start_marker, turn_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a grey polyline path segment by segment using the colored marker beside the start and beside each outer turn corner.

    Input structure:
    - The background is black, and one non-background color forms a single-pixel-wide 4-connected orthogonal path with no branches.
    - The path starts on one edge pixel and ends on another edge pixel.
    - Exactly one endpoint has a side-adjacent single-pixel marker that gives the initial repaint color.
    - Every turn pixel has one single-pixel marker on the outer diagonal corner of that bend, while the terminal endpoint has no marker.

    Rule:
    - Identify the unique edge-to-edge simple path and determine which endpoint is the start by finding the side marker next to it.
    - Traverse the path from that start endpoint to the other endpoint.
    - Repaint path pixels with the start-marker color until a turn is reached; at each turn, including the turn pixel itself, switch to the color of that turn's outer-corner marker.
    - Leave all non-path marker pixels unchanged.

    Key insight:
    - The puzzle is a path-tracing problem: geometry identifies the travel order and the turns, while nearby marker pixels act as local color-change instructions.
    """
    g = np.asarray(grid, dtype=int)
    path, start_marker, turn_colors = _extract_controls(g)

    out = g.copy()
    current_color = int(g[start_marker])
    for idx, (r, c) in enumerate(path):
        if idx in turn_colors:
            current_color = int(turn_colors[idx])
        out[r, c] = current_color
    return out


def _cells_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [vertices[0]]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        rr, cc = r0, c0
        while (rr, cc) != (r1, c1):
            rr += dr
            cc += dc
            cells.append((rr, cc))
    return cells


def _build_canonical_vertices(h: int, w: int, turns: int) -> list[tuple[int, int]]:
    end_side = random.choice(["left", "right"]) if turns % 2 == 1 else None
    start_col = random.randint(2, w - 3)
    vertices = [(0, start_col)]
    row, col = 0, start_col
    segment_count = turns + 1

    for seg_idx in range(segment_count):
        is_vertical = seg_idx % 2 == 0
        is_last = seg_idx == segment_count - 1

        if is_vertical:
            future_verticals = sum(1 for idx in range(seg_idx + 1, segment_count) if idx % 2 == 0)
            if is_last:
                row = h - 1
            else:
                upper = h - 1 - MIN_SEG_LEN * future_verticals if turns % 2 == 0 else h - 2 - MIN_SEG_LEN * future_verticals
                lower = row + MIN_SEG_LEN
                assert lower <= upper
                row = random.randint(lower, upper)
            vertices.append((row, col))
            continue

        if is_last:
            assert end_side is not None
            target_col = 0 if end_side == "left" else w - 1
            assert abs(target_col - col) >= MIN_SEG_LEN
            col = target_col
            vertices.append((row, col))
            continue

        min_col, max_col = 1, w - 2
        if turns % 2 == 1:
            future_nonfinal_h = sum(1 for idx in range(seg_idx + 2, segment_count - 1) if idx % 2 == 1)
            if future_nonfinal_h == 0:
                if end_side == "left":
                    min_col = 2
                else:
                    max_col = w - 3
        candidates = [cand for cand in range(min_col, max_col + 1) if abs(cand - col) >= MIN_SEG_LEN]
        assert candidates
        col = random.choice(candidates)
        vertices.append((row, col))

    return vertices


def _start_marker_cell(
    path: list[tuple[int, int]], path_set: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[int, int]:
    start = path[0]
    nxt = path[1]
    direction = (nxt[0] - start[0], nxt[1] - start[1])
    candidates: list[tuple[int, int]] = []
    for sr, sc in _orthogonal_sides(direction):
        rr, cc = start[0] + sr, start[1] + sc
        if _in_bounds(shape, rr, cc) and (rr, cc) not in path_set:
            candidates.append((rr, cc))
    assert candidates
    return random.choice(candidates)


def _turn_marker_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for idx in range(1, len(path) - 1):
        prev = path[idx - 1]
        cur = path[idx]
        nxt = path[idx + 1]
        v_prev = (cur[0] - prev[0], cur[1] - prev[1])
        v_next = (nxt[0] - cur[0], nxt[1] - cur[1])
        if v_prev == v_next:
            continue
        markers.append((cur[0] + v_prev[0] - v_next[0], cur[1] + v_prev[1] - v_next[1]))
    return markers


def _endpoint_side_markers(
    path: list[tuple[int, int]], marker_cells: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[int, int]:
    total = 0
    for endpoint, neighbor in ((path[0], path[1]), (path[-1], path[-2])):
        direction = (neighbor[0] - endpoint[0], neighbor[1] - endpoint[1])
        for sr, sc in _orthogonal_sides(direction):
            rr, cc = endpoint[0] + sr, endpoint[1] + sc
            if _in_bounds(shape, rr, cc) and (rr, cc) in marker_cells:
                total += 1
    return total


def _paint_expected(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start_marker: tuple[int, int],
    turn_markers: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    turn_iter = iter(turn_markers)
    turn_lookup: dict[int, int] = {}
    for idx in range(1, len(path) - 1):
        prev = path[idx - 1]
        cur = path[idx]
        nxt = path[idx + 1]
        if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            mr, mc = next(turn_iter)
            turn_lookup[idx] = int(grid[mr, mc])

    current_color = int(grid[start_marker])
    for idx, (r, c) in enumerate(path):
        if idx in turn_lookup:
            current_color = int(turn_lookup[idx])
        out[r, c] = current_color
    return out


def _apply_symmetry(arr: np.ndarray, sym_id: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    if sym_id >= 4:
        out = np.fliplr(out)
    return np.rot90(out, sym_id % 4)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    turns = random.randint(MIN_TURNS, MAX_TURNS)

    path_color = random.choice(ALL_COLORS[1:])
    marker_palette = [color for color in ALL_COLORS[1:] if color != path_color]
    if turns + 1 > len(marker_palette):
        return None
    marker_colors = random.sample(marker_palette, turns + 1)

    vertices = _build_canonical_vertices(h, w, turns)
    path = _cells_from_vertices(vertices)
    path_set = set(path)
    if len(path_set) != len(path):
        return None

    start_marker = _start_marker_cell(path, path_set, (h, w))
    turn_markers = _turn_marker_cells(path)
    marker_cells = [start_marker] + turn_markers

    if len(set(marker_cells)) != len(marker_cells):
        return None
    if any(cell in path_set for cell in marker_cells):
        return None
    if any(not _in_bounds((h, w), *cell) for cell in marker_cells):
        return None
    if _endpoint_side_markers(path, set(marker_cells), (h, w)) != 1:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in path:
        grid[r, c] = path_color
    for cell, color in zip(marker_cells, marker_colors):
        grid[cell] = color

    expected = _paint_expected(grid, path, start_marker, turn_markers)
    sym_id = random.randrange(8)
    grid = _apply_symmetry(grid, sym_id)
    expected = _apply_symmetry(expected, sym_id)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
