from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

MAX_GEN_ATTEMPTS = 400
TURN_CHOICES = (2, 3, 4)
YELLOW_PATTERNS = (
    ((-1, -1), (1, 0)),
    ((-1, 1), (0, -1), (1, -1)),
    ((-1, 0), (1, 1)),
    ((-1, -1), (-1, 1), (1, 0)),
    ((0, 1), (1, -1), (1, 1)),
    ((-1, 0), (0, 1), (1, -1), (1, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _count_turns(path: list[tuple[int, int]]) -> int:
    return sum(
        (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
        for a, b, c in zip(path, path[1:], path[2:])
    )


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment_cells(a, b)
        if cells and seg[0] == cells[-1]:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _unique_shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) == BLUE:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))

    assert goal in dist, "red endpoints must be connected"

    path = [goal]
    cur = goal
    while cur != start:
        predecessors = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) in dist and dist[(nr, nc)] == dist[cur] - 1
        ]
        assert len(predecessors) == 1, "expected a unique shortest path"
        cur = predecessors[0]
        path.append(cur)
    path.reverse()
    return path


def _unique_shortest_path_open(
    open_cells: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        cell = q.popleft()
        for nxt in _neighbors4(*cell):
            if nxt not in open_cells or nxt in dist:
                continue
            dist[nxt] = dist[cell] + 1
            q.append(nxt)

    if goal not in dist:
        return None

    path = [goal]
    cur = goal
    while cur != start:
        predecessors = [
            prev
            for prev in _neighbors4(*cur)
            if prev in dist and dist[prev] == dist[cur] - 1
        ]
        if len(predecessors) != 1:
            return None
        cur = predecessors[0]
        path.append(cur)
    path.reverse()
    return path


def _connected_without_cell(
    open_cells: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: tuple[int, int],
) -> bool:
    if blocked == start or blocked == goal or blocked not in open_cells:
        return False

    allowed = set(open_cells)
    allowed.remove(blocked)
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}

    while q:
        cell = q.popleft()
        if cell == goal:
            return True
        for nxt in _neighbors4(*cell):
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _target_endpoint(grid: np.ndarray, red_coords: np.ndarray) -> tuple[int, int]:
    targets = []
    for r, c in map(tuple, red_coords.tolist()):
        window = grid[r - 1 : r + 2, c - 1 : c + 2]
        if int(np.count_nonzero(window == YELLOW)) > 0:
            targets.append((int(r), int(c)))
    assert len(targets) == 1, "exactly one red endpoint should have nearby yellow pixels"
    return targets[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Find the unique shortest path, count its turns, and scale the yellow-marked endpoint neighborhood by that turn count.

    Input structure
    - The grid has black background, blue wall pixels, and exactly two red endpoint pixels.
    - One red endpoint sits at the center of a wall-free 3x3 neighborhood that also contains a small asymmetric yellow pattern; the other endpoint has no nearby yellow.

    Rule
    - Treat blue as blocked and find the unique shortest 4-neighbor path between the two red endpoints.
    - Recolor only the black interior cells of that path to gray, then count the number of direction changes along the full path.
    - Identify the yellow-marked endpoint, crop its centered 3x3 neighborhood from the path-marked grid, and enlarge that crop by the turn count.

    Key insight
    - The shortest path is only an intermediate object: its turn count encodes the scale factor that determines how large the endpoint neighborhood becomes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red_coords[0]))
    goal = tuple(map(int, red_coords[1]))

    target = _target_endpoint(g, red_coords)
    tr, tc = target
    assert 1 <= tr < g.shape[0] - 1 and 1 <= tc < g.shape[1] - 1, "target endpoint must support a 3x3 crop"
    assert not np.any(g[tr - 1 : tr + 2, tc - 1 : tc + 2] == BLUE), "target 3x3 neighborhood must be wall-free"

    path = _unique_shortest_path(g, start, goal)
    turns = _count_turns(path)
    assert 2 <= turns <= 4, "this puzzle expects a turn count between 2 and 4"

    marked = g.copy()
    for r, c in path[1:-1]:
        if int(marked[r, c]) == BLACK:
            marked[r, c] = GREY

    crop = marked[tr - 1 : tr + 2, tc - 1 : tc + 2]
    return np.kron(crop, np.ones((turns, turns), dtype=int))


def _build_main_points(turns: int) -> list[tuple[int, int]] | None:
    horizontal_lengths = [random.randint(3, 5) for _ in range((turns + 2) // 2)]
    vertical_lengths = [random.randint(3, 5) for _ in range((turns + 1) // 2)]
    vertical_signs = [random.choice((-1, 1)) for _ in range(len(vertical_lengths))]

    points = [(0, 0)]
    r = 0
    c = 0
    h_idx = 0
    v_idx = 0
    for step in range(turns + 1):
        if step % 2 == 0:
            c += horizontal_lengths[h_idx]
            h_idx += 1
        else:
            r += vertical_signs[v_idx] * vertical_lengths[v_idx]
            v_idx += 1
        points.append((r, c))

    cells = _polyline_cells(points)
    if len(cells) != len(set(cells)):
        return None
    return points


def _horizontal_detour_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    side: int,
) -> list[tuple[int, int]]:
    r = start[0]
    c0, c1 = sorted((start[1], end[1]))
    bridge_row = r + side
    mid_row = r + (1 if side > 0 else -1)

    cells = [(mid_row, c0), (bridge_row, c0)]
    cells.extend((bridge_row, c) for c in range(c0 + 1, c1))
    cells.extend(((bridge_row, c1), (mid_row, c1)))
    return cells


def _choose_yellow_cells(
    target: tuple[int, int],
    entry: tuple[int, int],
) -> list[tuple[int, int]]:
    tr, tc = target
    entry_rel = (entry[0] - tr, entry[1] - tc)
    candidates = [
        pattern
        for pattern in YELLOW_PATTERNS
        if entry_rel not in pattern
    ]
    pattern = random.choice(candidates)
    return [(tr + dr, tc + dc) for dr, dc in pattern]


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _build_layout(turns: int):
    for _ in range(200):
        points = _build_main_points(turns)
        if points is None:
            continue

        start = points[0]
        target = points[-1]
        room = {(target[0] + dr, target[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        main_cells = _polyline_cells(points)
        base_open = set(main_cells) | room

        detour_candidates = []
        for idx, (a, b) in enumerate(zip(points, points[1:])):
            if a[0] != b[0]:
                continue
            if a == target or b == target:
                continue
            seg = _segment_cells(a, b)
            if len(seg) < 4:
                continue
            detour_candidates.append((idx, a, b, seg))
        random.shuffle(detour_candidates)

        for _idx, a, b, seg in detour_candidates:
            for side in random.sample((-2, 2), 2):
                detour = _horizontal_detour_cells(a, b, side)
                if any(cell in base_open and cell not in {a, b} for cell in detour):
                    continue

                open_cells = set(base_open)
                open_cells.update(detour)
                path = _unique_shortest_path_open(open_cells, start, target)
                if path is None or _count_turns(path) != turns:
                    continue

                block_cell = seg[len(seg) // 2]
                if not _connected_without_cell(open_cells, start, target, block_cell):
                    continue

                entry = path[-2]
                yellow_cells = _choose_yellow_cells(target, entry)
                if any(cell not in room for cell in yellow_cells):
                    continue

                all_cells = open_cells | set(yellow_cells)
                rows = [r for r, _ in all_cells]
                cols = [c for _, c in all_cells]
                span_h = max(rows) - min(rows) + 1
                span_w = max(cols) - min(cols) + 1
                margin_top = random.randint(1, 4)
                margin_bottom = random.randint(1, 4)
                margin_left = random.randint(1, 4)
                margin_right = random.randint(1, 4)
                full_h = span_h + margin_top + margin_bottom
                full_w = span_w + margin_left + margin_right
                if max(full_h, full_w) > 24:
                    continue

                shift_r = margin_top - min(rows)
                shift_c = margin_left - min(cols)
                return {
                    "open_cells": {(r + shift_r, c + shift_c) for r, c in open_cells},
                    "start": (start[0] + shift_r, start[1] + shift_c),
                    "target": (target[0] + shift_r, target[1] + shift_c),
                    "yellow_cells": [(r + shift_r, c + shift_c) for r, c in yellow_cells],
                    "shape": (full_h, full_w),
                    "turns": turns,
                }
    return None


def _render_layout(layout: dict) -> np.ndarray:
    h, w = layout["shape"]
    open_cells = layout["open_cells"]
    start = layout["start"]
    target = layout["target"]
    yellow_cells = layout["yellow_cells"]

    grid = np.full((h, w), BLACK, dtype=int)
    walls = set()
    for r, c in open_cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and (nr, nc) not in open_cells:
                walls.add((nr, nc))
    for r, c in walls:
        grid[r, c] = BLUE

    for r, c in yellow_cells:
        grid[r, c] = YELLOW
    grid[start] = RED
    grid[target] = RED
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        turns = random.choice(TURN_CHOICES)
        layout = _build_layout(turns)
        if layout is None:
            continue

        grid = _render_layout(layout)
        grid = _random_transform(grid)

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.shape != (3 * turns, 3 * turns):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
