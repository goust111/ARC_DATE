from __future__ import annotations

import heapq
from collections import Counter, deque

import numpy as np


BG = 0
PIPE = 5
VALVE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, -1), (0, 1))
BRANCH_MODES = ("up", "down", "both")
ODD_GAPS = (3, 5)
ODD_LENGTHS = (3, 5, 7)
GENERATION_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    return sum(1 for rr, cc in _neighbors4(r, c) if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]))


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start_r, start_c = map(int, points[0])
    seen = {((start_r), (start_c))}
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(mask.shape, rr, cc) or not bool(mask[rr, cc]) or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return len(seen) == int(points.shape[0])


def _draw_segment(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
        return
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
        return
    raise ValueError("segments must be axis-aligned")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each valve color through the grey pipe network until competing expansions meet.

    Input structure:
    - The background is black, with one connected single-pixel-wide pipe network on top.
    - Most pipe cells are grey, while some pipe endpoints or T/cross junctions are replaced
      by single colored valve pixels.
    - Pipes run only horizontally or vertically, and every non-background pixel belongs to
      the same network.

    Rule:
    - Treat every valve as a source and expand along the pipe graph through grey cells only.
    - A grey pipe cell is recolored by the valve that reaches it through the shortest path
      inside the network; valve pixels themselves keep their original colors.
    - This makes each valve-to-valve pipe segment split at its midpoint, while valve-to-dead-end
      segments are fully claimed by the nearest valve.
    - If two valves can reach a grey cell in the same number of steps, break the tie by the
      row-major order of the valve positions.

    Key insight:
    - The task is a multi-source shortest-path fill on the pipe graph: grey cells belong to
      the nearest valve measured along connected pipe distance, not by straight-line geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert fg_colors, "expected at least one foreground color"

    counts = {color: int(np.count_nonzero(g == color)) for color in fg_colors}
    pipe_color = max(fg_colors, key=lambda color: counts[color])
    valve_cells = [
        (int(r), int(c), int(g[int(r), int(c)]))
        for r, c in np.argwhere((g != int(bg)) & (g != int(pipe_color)))
    ]
    assert valve_cells, "expected at least one valve"

    valve_cells.sort()
    mask = g != int(bg)
    out = g.copy()

    best_dist = np.full(g.shape, np.iinfo(np.int32).max, dtype=np.int32)
    best_owner = np.full(g.shape, len(valve_cells), dtype=np.int32)
    heap: list[tuple[int, int, int, int]] = []

    for owner, (r, c, _color) in enumerate(valve_cells):
        best_dist[r, c] = 0
        best_owner[r, c] = owner
        heapq.heappush(heap, (0, owner, r, c))

    while heap:
        dist, owner, r, c = heapq.heappop(heap)
        if dist != int(best_dist[r, c]) or owner != int(best_owner[r, c]):
            continue
        for rr, cc in _neighbors4(r, c):
            if not _in_bounds(g.shape, rr, cc) or not bool(mask[rr, cc]):
                continue
            next_dist = dist + 1
            if next_dist < int(best_dist[rr, cc]) or (
                next_dist == int(best_dist[rr, cc]) and owner < int(best_owner[rr, cc])
            ):
                best_dist[rr, cc] = next_dist
                best_owner[rr, cc] = owner
                heapq.heappush(heap, (next_dist, owner, rr, cc))

    for r, c in np.argwhere(g == int(pipe_color)):
        owner = int(best_owner[int(r), int(c)])
        out[int(r), int(c)] = int(valve_cells[owner][2])

    return out


def _pad_and_rotate(rng: np.random.Generator, grid: np.ndarray) -> np.ndarray:
    top = int(rng.choice([0, 0, 1, 2]))
    bottom = int(rng.choice([0, 0, 1, 2]))
    left = int(rng.choice([0, 0, 1, 2]))
    right = int(rng.choice([0, 0, 1, 2]))

    h, w = grid.shape
    padded = np.full((h + top + bottom, w + left + right), BG, dtype=int)
    padded[top : top + h, left : left + w] = grid

    k = int(rng.integers(0, 4))
    padded = np.rot90(padded, k)
    if bool(rng.integers(0, 2)):
        padded = np.fliplr(padded)
    if bool(rng.integers(0, 2)):
        padded = np.flipud(padded)
    return np.asarray(padded, dtype=int)


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    n_junctions = int(rng.integers(2, 5))
    left_gap = int(rng.choice(ODD_GAPS))
    right_gap = int(rng.choice(ODD_GAPS))
    middle_gaps = [int(rng.choice(ODD_GAPS)) for _ in range(n_junctions - 1)]

    branch_modes = [str(rng.choice(BRANCH_MODES)) for _ in range(n_junctions)]
    if "both" not in branch_modes and bool(rng.integers(0, 2)):
        branch_modes[int(rng.integers(0, n_junctions))] = "both"

    up_lengths = []
    down_lengths = []
    for mode in branch_modes:
        up_lengths.append(int(rng.choice(ODD_LENGTHS)) if mode in ("up", "both") else 0)
        down_lengths.append(int(rng.choice(ODD_LENGTHS)) if mode in ("down", "both") else 0)

    max_up = max(up_lengths) if up_lengths else 0
    max_down = max(down_lengths) if down_lengths else 0
    trunk_row = max_up

    junction_cols = [left_gap]
    for gap in middle_gaps:
        junction_cols.append(junction_cols[-1] + gap)
    right_col = junction_cols[-1] + right_gap

    local_h = max_up + max_down + 1
    local_w = right_col + 1
    if local_h > 26 or local_w > 26:
        raise RuntimeError("local canvas too large")

    cells: set[tuple[int, int]] = set()
    _draw_segment(cells, trunk_row, 0, trunk_row, right_col)

    endpoints: list[tuple[int, int]] = [(trunk_row, 0), (trunk_row, right_col)]
    junctions: list[tuple[int, int]] = []

    for col, mode, up_len, down_len in zip(junction_cols, branch_modes, up_lengths, down_lengths):
        junctions.append((trunk_row, col))
        if mode in ("up", "both"):
            _draw_segment(cells, trunk_row - up_len, col, trunk_row, col)
            endpoints.append((trunk_row - up_len, col))
        if mode in ("down", "both"):
            _draw_segment(cells, trunk_row, col, trunk_row + down_len, col)
            endpoints.append((trunk_row + down_len, col))

    grid = np.full((local_h, local_w), BG, dtype=int)
    for r, c in cells:
        grid[r, c] = PIPE

    max_extra = min(len(VALVE_COLORS) - len(junctions), len(endpoints), 4)
    extra_count = int(rng.integers(0, max_extra + 1))
    if extra_count == len(endpoints) and endpoints:
        extra_count -= 1
    if extra_count == 0 and max_extra > 0 and bool(rng.integers(0, 2)):
        extra_count = 1

    endpoints_copy = list(endpoints)
    rng.shuffle(endpoints_copy)
    extra_valves = endpoints_copy[:extra_count]
    valve_positions = junctions + extra_valves

    colors = list(rng.choice(np.array(VALVE_COLORS, dtype=int), size=len(valve_positions), replace=False))
    for color, (r, c) in zip(colors, valve_positions):
        grid[r, c] = int(color)

    return _pad_and_rotate(rng, grid)


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _most_common_color(grid)
    fg_colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    if len(fg_colors) < 2:
        return False

    counts = Counter({color: int(np.count_nonzero(grid == color)) for color in fg_colors})
    pipe_color, pipe_count = counts.most_common(1)[0]
    if pipe_count <= 0:
        return False

    mask = grid != int(bg)
    if not _connected(mask):
        return False

    valve_mask = (grid != int(bg)) & (grid != int(pipe_color))
    valve_points = [tuple(map(int, pt)) for pt in np.argwhere(valve_mask)]
    if len(valve_points) < 2 or len(valve_points) > len(VALVE_COLORS):
        return False
    if len({int(grid[r, c]) for r, c in valve_points}) != len(valve_points):
        return False

    grey_endpoints = 0
    has_branch_junction = False
    for r, c in np.argwhere(mask):
        deg = _degree(mask, int(r), int(c))
        if valve_mask[int(r), int(c)]:
            if deg not in (1, 3, 4):
                return False
        elif int(grid[int(r), int(c)]) == int(pipe_color):
            if deg == 1:
                grey_endpoints += 1
            if deg >= 3:
                has_branch_junction = True

    if grey_endpoints == 0:
        return False
    if has_branch_junction:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _build_candidate(rng)
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid valve-pipe puzzle")
