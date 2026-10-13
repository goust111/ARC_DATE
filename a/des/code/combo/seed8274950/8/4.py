from __future__ import annotations

import heapq
import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _edge_positions(h: int, w: int):
    for c in range(w):
        yield 0, c
        yield h - 1, c
    for r in range(1, h - 1):
        yield r, 0
        yield r, w - 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _wall_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _bg_and_markers(grid: np.ndarray, wall: int):
    h, w = grid.shape
    edge_set = set(_edge_positions(h, w))
    edge_candidates = [(r, c) for r, c in edge_set if int(grid[r, c]) != wall]

    interior_colors = [
        int(grid[r, c])
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) != wall and (r, c) not in edge_set
    ]
    assert interior_colors, "expected corridor cells away from the border"
    bg = Counter(interior_colors).most_common(1)[0][0]

    markers = sorted(
        (r, c, int(grid[r, c]))
        for r, c in edge_candidates
        if int(grid[r, c]) != bg
    )
    assert markers, "expected at least one colored border marker"
    return bg, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grid with a thin maze wall and border entry markers, flood each corridor
    from its marker color until the path ends.

    When a corridor has markers on both ends, the fill is the simultaneous
    multi-source result; an exact tie is resolved by marker reading order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall = _wall_color(g)
    bg, markers = _bg_and_markers(g, wall)

    walkable = g != wall
    dist = np.full((h, w), -1, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    pq: list[tuple[int, int, int, int]] = []

    for idx, (r, c, _color) in enumerate(markers):
        dist[r, c] = 0
        owner[r, c] = idx
        heapq.heappush(pq, (0, idx, r, c))

    while pq:
        d, idx, r, c = heapq.heappop(pq)
        if d != int(dist[r, c]) or idx != int(owner[r, c]):
            continue
        for nr, nc in _neighbors4(r, c, h, w):
            if not walkable[nr, nc]:
                continue
            nd = d + 1
            better = int(dist[nr, nc]) == -1 or nd < int(dist[nr, nc])
            tie_better = nd == int(dist[nr, nc]) and idx < int(owner[nr, nc])
            if better or tie_better:
                dist[nr, nc] = nd
                owner[nr, nc] = idx
                heapq.heappush(pq, (nd, idx, nr, nc))

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg and int(owner[r, c]) >= 0:
                out[r, c] = markers[int(owner[r, c])][2]
    return out


def _is_boundary_cell(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == rows - 1 or c == cols - 1


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int):
    r, c = cell
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _center_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return 2 * r + 1, 2 * c + 1


def _mid_pixel(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    return ar + br + 1, ac + bc + 1


def _marker_options(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    options: list[tuple[int, int]] = []
    if r == 0:
        options.append((0, 2 * c + 1))
    if r == rows - 1:
        options.append((2 * rows, 2 * c + 1))
    if c == 0:
        options.append((2 * r + 1, 0))
    if c == cols - 1:
        options.append((2 * r + 1, 2 * cols))
    return options


def _free_neighbors(
    cell: tuple[int, int],
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    local: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        nxt
        for nxt in _cell_neighbors(cell, rows, cols)
        if nxt not in blocked and nxt not in local
    ]


def _grow_path(
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    start: tuple[int, int],
    mode: str,
) -> list[tuple[int, int]] | None:
    free_cells = rows * cols - len(blocked)
    required_len = 3 if mode == "dead" else 4
    if free_cells < required_len:
        return None
    if mode == "dead":
        min_len = random.randint(required_len, min(6, free_cells))
    else:
        min_len = random.randint(required_len, min(7, free_cells))
    max_len = random.randint(min_len, min(free_cells, min_len + 5))

    path = [start]
    local = {start}

    def endpoint_ok(cell: tuple[int, int]) -> bool:
        if mode == "dead":
            return len(path) >= min_len and not _is_boundary_cell(cell, rows, cols)
        return (
            len(path) >= min_len
            and cell != start
            and _is_boundary_cell(cell, rows, cols)
        )

    def dfs() -> list[tuple[int, int]] | None:
        cur = path[-1]
        neighbors = _free_neighbors(cur, rows, cols, blocked, local)
        if endpoint_ok(cur):
            if not neighbors or len(path) == max_len or random.random() < 0.45:
                return list(path)
        if len(path) >= max_len:
            return None

        random.shuffle(neighbors)
        inward = [cell for cell in neighbors if not _is_boundary_cell(cell, rows, cols)]
        boundary = [cell for cell in neighbors if _is_boundary_cell(cell, rows, cols)]

        if mode == "dead":
            ordered = inward + boundary
        elif len(path) < min_len:
            ordered = inward + boundary
        else:
            ordered = boundary + inward

        for nxt in ordered:
            local.add(nxt)
            path.append(nxt)
            result = dfs()
            if result is not None:
                return result
            path.pop()
            local.remove(nxt)

        if endpoint_ok(cur):
            return list(path)
        return None

    return dfs()


def _walkable_components_ok(grid: np.ndarray, wall: int, bg: int, markers: set[tuple[int, int]]) -> bool:
    comps = _connected_components(grid != wall)
    for comp in comps:
        comp_set = set(comp)
        degrees = {
            cell: sum((nr, nc) in comp_set for nr, nc in _neighbors4(cell[0], cell[1], *grid.shape))
            for cell in comp
        }
        if any(deg not in (1, 2) for deg in degrees.values()):
            return False
        if sum(deg == 1 for deg in degrees.values()) != 2:
            return False
        marker_count = sum(cell in markers for cell in comp)
        if marker_count not in (1, 2):
            return False
        if any(cell in markers and degrees[cell] != 1 for cell in comp):
            return False
        bg_endpoints = sum(
            degrees[cell] == 1 and int(grid[cell]) == bg
            for cell in comp
        )
        if marker_count == 1 and bg_endpoints != 1:
            return False
        if marker_count == 2 and bg_endpoints != 0:
            return False
    return True


def _render_grid(
    rows: int,
    cols: int,
    paths: list[tuple[str, list[tuple[int, int]]]],
    bg: int,
    wall: int,
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.full((2 * rows + 1, 2 * cols + 1), wall, dtype=int)
    marker_idx = 0
    marker_positions: set[tuple[int, int]] = set()

    for mode, path in paths:
        for cell in path:
            cr, cc = _center_pixel(cell)
            grid[cr, cc] = bg
        for a, b in zip(path, path[1:]):
            mr, mc = _mid_pixel(a, b)
            grid[mr, mc] = bg

        endpoints = [path[0]]
        if mode == "pair":
            endpoints.append(path[-1])
        for endpoint in endpoints:
            marker_pos = random.choice(_marker_options(endpoint, rows, cols))
            assert marker_pos not in marker_positions
            marker_positions.add(marker_pos)
            grid[marker_pos] = marker_colors[marker_idx]
            marker_idx += 1

    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(4, 7)
    cols = random.randint(4, 7)
    n_components = random.randint(1, 3)

    if n_components == 1:
        modes = [random.choice(["dead", "pair"])]
    else:
        modes = [random.choice(["dead", "pair"]) for _ in range(n_components)]
        if "dead" not in modes:
            modes[0] = "dead"
        if "pair" not in modes:
            modes[-1] = "pair"
        random.shuffle(modes)

    used: set[tuple[int, int]] = set()
    paths: list[tuple[str, list[tuple[int, int]]]] = []
    boundary_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if _is_boundary_cell((r, c), rows, cols)
    ]

    for mode in modes:
        starts = [cell for cell in boundary_cells if cell not in used]
        random.shuffle(starts)
        chosen = None
        for start in starts:
            for _ in range(12):
                path = _grow_path(rows, cols, used, start, mode)
                if path is not None:
                    chosen = path
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None
        used.update(chosen)
        paths.append((mode, chosen))

    marker_total = sum(1 if mode == "dead" else 2 for mode, _path in paths)
    if marker_total > 8:
        return None

    bg, wall = random.sample(range(10), 2)
    marker_palette = [color for color in range(10) if color not in (bg, wall)]
    marker_colors = random.sample(marker_palette, marker_total)
    grid = _render_grid(rows, cols, paths, bg, wall, marker_colors)

    wall_comps = _connected_components(grid == wall)
    if len(wall_comps) != 1:
        return None

    marker_positions = {
        (r, c)
        for r, c in _edge_positions(*grid.shape)
        if int(grid[r, c]) not in (bg, wall)
    }
    if len(marker_positions) != marker_total:
        return None

    counts = Counter(int(v) for v in grid.ravel())
    if any(counts[color] != 1 for color in marker_colors):
        return None

    if not _walkable_components_ok(grid, wall, bg, marker_positions):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[grid == wall], grid[grid == wall]):
        return None
    for r, c in marker_positions:
        if int(out[r, c]) != int(grid[r, c]):
            return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid maze-corridor input")
