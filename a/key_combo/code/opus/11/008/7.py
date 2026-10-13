from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLOCKER = 1
RAY = 2
PATH = 3
FRAME = 5

MAX_GENERATION_ATTEMPTS = 500
_PATTERN_CACHE: dict[int, list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {}


def _draw_diamond(grid: np.ndarray, center_r: int, center_c: int, radius: int) -> None:
    for r in range(center_r - radius, center_r + radius + 1):
        dc = radius - abs(r - center_r)
        grid[r, center_c - dc] = FRAME
        grid[r, center_c + dc] = FRAME


def _diamond_info(grid: np.ndarray) -> tuple[int, int, int, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a grey diamond frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h % 2 == 1, "diamond frame bbox must be square with odd side"
    radius = h // 2
    center_r = r0 + radius
    center_c = c0 + radius
    dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
    assert coords.shape[0] == 4 * radius and np.all(dists == radius), "expected a complete 1px diamond"
    return center_r, center_c, radius, (r0, c0, r1, c1)


def _interior_mask(shape: tuple[int, int], center_r: int, center_c: int, radius: int) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return np.abs(rows - center_r) + np.abs(cols - center_c) < radius


def _marker_positions(center_r: int, center_c: int, radius: int) -> dict[str, tuple[int, int]]:
    return {
        "left": (center_r, center_c - radius - 1),
        "right": (center_r, center_c + radius + 1),
        "top": (center_r - radius - 1, center_c),
        "bottom": (center_r + radius + 1, center_c),
    }


def _lex_neighbors(coord: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = coord
    out = []
    for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            out.append((rr, cc))
    out.sort()
    return out


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    center_r: int,
    center_c: int,
    radius: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    rr = r + dr
    cc = c + dc
    entered = False
    painted: list[tuple[int, int]] = []

    while 0 <= rr < h and 0 <= cc < w:
        inside = abs(rr - center_r) + abs(cc - center_c) < radius
        if inside:
            entered = True
            if int(grid[rr, cc]) == BG:
                painted.append((rr, cc))
            else:
                break
        elif entered:
            break
        rr += dr
        cc += dc
    return painted


def _entry_inside(
    marker: tuple[int, int],
    step: tuple[int, int],
    center_r: int,
    center_c: int,
    radius: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    r, c = marker
    dr, dc = step
    rr = r + dr
    cc = c + dc
    while 0 <= rr < shape[0] and 0 <= cc < shape[1]:
        if abs(rr - center_r) + abs(cc - center_c) < radius:
            return rr, cc
        rr += dr
        cc += dc
    raise AssertionError("marker does not lead into the diamond interior")


def _simulate_red_phase(
    grid: np.ndarray, center_r: int, center_c: int, radius: int
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    markers = _marker_positions(center_r, center_c, radius)
    planned = {
        "left": _trace_ray(grid, markers["left"], (0, 1), center_r, center_c, radius),
        "right": _trace_ray(grid, markers["right"], (0, -1), center_r, center_c, radius),
    }
    out = grid.copy()
    for cells in planned.values():
        for r, c in cells:
            out[r, c] = RAY
    return out, planned


def _bfs_dist(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    if not passable[start]:
        return dist
    q = deque([start])
    dist[start] = 0
    while q:
        cur = q.popleft()
        for nxt in _lex_neighbors(cur, passable.shape):
            if passable[nxt] and dist[nxt] < 0:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _shortest_lex_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    dist_start = _bfs_dist(passable, start)
    if dist_start[end] < 0:
        return None
    dist_end = _bfs_dist(passable, end)
    target = int(dist_start[end])

    path = [start]
    cur = start
    while cur != end:
        next_step = None
        for nxt in _lex_neighbors(cur, passable.shape):
            if not passable[nxt]:
                continue
            if dist_start[nxt] != dist_start[cur] + 1:
                continue
            if dist_end[nxt] < 0 or dist_start[nxt] + dist_end[nxt] != target:
                continue
            next_step = nxt
            break
        assert next_step is not None, "shortest-path reconstruction failed"
        path.append(next_step)
        cur = next_step
    return path


def _count_shortest_paths(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int], limit: int = 3
) -> int:
    dist_start = _bfs_dist(passable, start)
    if dist_start[end] < 0:
        return 0
    dist_end = _bfs_dist(passable, end)
    target = int(dist_start[end])

    ways = np.zeros(passable.shape, dtype=int)
    ways[start] = 1
    for depth in range(target):
        coords = np.argwhere(dist_start == depth)
        coords = sorted((int(r), int(c)) for r, c in coords)
        for cur in coords:
            if ways[cur] == 0:
                continue
            if dist_start[cur] + dist_end[cur] != target:
                continue
            for nxt in _lex_neighbors(cur, passable.shape):
                if not passable[nxt]:
                    continue
                if dist_start[nxt] != depth + 1:
                    continue
                if dist_start[nxt] + dist_end[nxt] != target:
                    continue
                ways[nxt] = min(limit, ways[nxt] + ways[cur])
    return int(ways[end])


def _path_on_grid(
    grid: np.ndarray, center_r: int, center_c: int, radius: int
) -> tuple[list[tuple[int, int]] | None, int]:
    interior = _interior_mask(grid.shape, center_r, center_c, radius)
    markers = _marker_positions(center_r, center_c, radius)
    start = _entry_inside(markers["top"], (1, 0), center_r, center_c, radius, grid.shape)
    end = _entry_inside(markers["bottom"], (-1, 0), center_r, center_c, radius, grid.shape)
    passable = (grid == BG) & interior
    path = _shortest_lex_path(passable, start, end)
    count = _count_shortest_paths(passable, start, end) if path is not None else 0
    return path, count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the red side rays inside a diamond, then route the lexicographically smallest shortest green path through the remaining interior gap.

    Input structure
    - The grid is black background with one complete grey 1-pixel diamond frame.
    - A red marker sits one cell outside the left midpoint of the frame bbox and another one cell outside the right midpoint; a green marker sits one cell above the top midpoint and another one cell below the bottom midpoint.
    - The diamond interior contains blue blocker pixels, while the strict interior black cells are the only cells that can be filled or traversed.

    Rule
    - First emit a horizontal red ray inward from each red marker, crossing the frame but painting only strict-interior black cells; each ray stops before the first blue blocker or the opposite frame.
    - Treat every newly painted red cell as an obstacle together with the grey frame and blue blockers.
    - Then find the 4-connected shortest path between the top and bottom green markers through the strict interior black cells only, and if several shortest paths exist choose the lexicographically smallest coordinate sequence.
    - Paint every cell on that chosen path green and leave all other cells unchanged.

    Key insight
    - The red rays are not the answer by themselves: they reshape the interior topology, so the green shortest path can only be determined after the wall-laying stage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center_r, center_c, radius, _bbox = _diamond_info(g)
    markers = _marker_positions(center_r, center_c, radius)
    assert g[markers["left"]] == RAY and g[markers["right"]] == RAY
    assert g[markers["top"]] == PATH and g[markers["bottom"]] == PATH

    blocker_coords = np.argwhere(g == BLOCKER)
    assert 2 <= blocker_coords.shape[0] <= 3, "expected 2-3 blocker pixels"
    interior = _interior_mask(g.shape, center_r, center_c, radius)
    assert np.all(interior[tuple(blocker_coords.T)]), "blockers must lie strictly inside the diamond"

    after_rays, _planned = _simulate_red_phase(g, center_r, center_c, radius)
    path, _count = _path_on_grid(after_rays, center_r, center_c, radius)
    assert path is not None, "green path must exist"

    out = after_rays.copy()
    for r, c in path:
        out[r, c] = PATH
    return out


def _ray_stop_reason(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    center_r: int,
    center_c: int,
    radius: int,
) -> str:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    rr = r + dr
    cc = c + dc
    entered = False
    while 0 <= rr < h and 0 <= cc < w:
        inside = abs(rr - center_r) + abs(cc - center_c) < radius
        if inside:
            entered = True
            if int(grid[rr, cc]) != BG:
                return "blocker" if int(grid[rr, cc]) == BLOCKER else "other"
        elif entered:
            return "frame"
        rr += dr
        cc += dc
    return "none"


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        center_r, center_c, radius, _bbox = _diamond_info(grid)
        markers = _marker_positions(center_r, center_c, radius)

        if grid[markers["left"]] != RAY or grid[markers["right"]] != RAY:
            return False
        if grid[markers["top"]] != PATH or grid[markers["bottom"]] != PATH:
            return False

        top_entry = _entry_inside(markers["top"], (1, 0), center_r, center_c, radius, grid.shape)
        bottom_entry = _entry_inside(markers["bottom"], (-1, 0), center_r, center_c, radius, grid.shape)
        left_entry = _entry_inside(markers["left"], (0, 1), center_r, center_c, radius, grid.shape)
        right_entry = _entry_inside(markers["right"], (0, -1), center_r, center_c, radius, grid.shape)

        blocker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLOCKER)]
        if len(blocker_coords) != 3:
            return False
        if top_entry in blocker_coords or bottom_entry in blocker_coords:
            return False
        if left_entry in blocker_coords or right_entry in blocker_coords:
            return False

        if _ray_stop_reason(grid, markers["left"], (0, 1), center_r, center_c, radius) != "blocker":
            return False
        if _ray_stop_reason(grid, markers["right"], (0, -1), center_r, center_c, radius) != "blocker":
            return False

        after_rays, planned = _simulate_red_phase(grid, center_r, center_c, radius)
        if len(planned["left"]) == 0 or len(planned["right"]) == 0:
            return False

        pre_path, pre_count = _path_on_grid(grid, center_r, center_c, radius)
        post_path, post_count = _path_on_grid(after_rays, center_r, center_c, radius)
        if pre_path is None or post_path is None:
            return False
        if pre_path == post_path:
            return False
        if post_count < 2:
            return False
        if pre_count < 2:
            return False

        if not any(c != center_c for _, c in post_path):
            return False
        if np.count_nonzero(after_rays == RAY) <= 2:
            return False
        if len(post_path) < 5:
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if np.count_nonzero(out == PATH) <= 2:
            return False
        return True
    except AssertionError:
        return False


def _canonical_grid(radius: int) -> tuple[np.ndarray, int, int]:
    size = 2 * radius + 9
    center = size // 2
    grid = np.full((size, size), BG, dtype=int)
    _draw_diamond(grid, center, center, radius)
    markers = _marker_positions(center, center, radius)
    grid[markers["left"]] = RAY
    grid[markers["right"]] = RAY
    grid[markers["top"]] = PATH
    grid[markers["bottom"]] = PATH
    return grid, center, center


def _valid_patterns_for_radius(radius: int) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    cached = _PATTERN_CACHE.get(radius)
    if cached is not None:
        return cached

    base, center_r, center_c = _canonical_grid(radius)
    markers = _marker_positions(center_r, center_c, radius)
    top_entry = _entry_inside(markers["top"], (1, 0), center_r, center_c, radius, base.shape)
    bottom_entry = _entry_inside(markers["bottom"], (-1, 0), center_r, center_c, radius, base.shape)
    left_entry = _entry_inside(markers["left"], (0, 1), center_r, center_c, radius, base.shape)
    right_entry = _entry_inside(markers["right"], (0, -1), center_r, center_c, radius, base.shape)
    forbidden = {top_entry, bottom_entry, left_entry, right_entry}

    row_cols = list(range(center_c - radius + 1, center_c + radius))
    interior_cells = [
        (r, c)
        for r in range(base.shape[0])
        for c in range(base.shape[1])
        if abs(r - center_r) + abs(c - center_c) < radius
    ]

    patterns: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for left_col in row_cols:
        for right_col in row_cols:
            if left_col >= right_col:
                continue
            if (center_r, left_col) in forbidden or (center_r, right_col) in forbidden:
                continue

            for block_r, block_c in interior_cells:
                if (block_r, block_c) in forbidden:
                    continue
                if block_r == center_r:
                    continue

                grid = base.copy()
                grid[center_r, left_col] = BLOCKER
                grid[center_r, right_col] = BLOCKER
                grid[block_r, block_c] = BLOCKER
                if not _validate_instance(grid):
                    continue

                pattern = (
                    (0, left_col - center_c),
                    (0, right_col - center_c),
                    (block_r - center_r, block_c - center_c),
                )
                patterns.append(pattern)

    assert patterns, f"no valid blocker patterns found for radius {radius}"
    _PATTERN_CACHE[radius] = patterns
    return patterns


def _generate_attempt() -> np.ndarray | None:
    radius = int(np.random.randint(4, 8))
    min_size = 2 * radius + 3
    h = int(np.random.randint(min_size, min(30, min_size + 5) + 1))
    w = int(np.random.randint(min_size, min(30, min_size + 5) + 1))
    center_r = int(np.random.randint(radius + 1, h - radius - 1))
    center_c = int(np.random.randint(radius + 1, w - radius - 1))

    grid = np.full((h, w), BG, dtype=int)
    _draw_diamond(grid, center_r, center_c, radius)

    markers = _marker_positions(center_r, center_c, radius)
    grid[markers["left"]] = RAY
    grid[markers["right"]] = RAY
    grid[markers["top"]] = PATH
    grid[markers["bottom"]] = PATH

    patterns = _valid_patterns_for_radius(radius)
    blocker_pattern = patterns[int(np.random.randint(len(patterns)))]
    for dr, dc in blocker_pattern:
        grid[center_r + dr, center_c + dc] = BLOCKER

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-wall pathfinding instance")
