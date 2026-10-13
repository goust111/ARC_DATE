from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import Color, in_bounds, neighbors4
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import Color, in_bounds, neighbors4


GREEN = int(Color.GREEN)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
YELLOW = int(Color.YELLOW)
BLACK = int(Color.BLACK)

GEN_RETRY_LIMIT = 300


def _trace_cycle(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in np.argwhere(mask)}
    assert pts, "expected a nonempty loop"
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, "green object must be a single-pixel orthogonal cycle"

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        next_cells = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) in pts and (nr, nc) != prev
        ]
        assert len(next_cells) == 1
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _turn_partition(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    counts: Counter[int] = Counter()
    turn_signs: dict[tuple[int, int], int] = {}
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        counts[sign] += 1
        turn_signs[(r, c)] = sign

    assert turn_signs, "loop must contain bends"
    assert counts[1] != counts[-1], "convex and concave counts must differ"
    convex_sign = 1 if counts[1] > counts[-1] else -1
    convex = {cell for cell, sign in turn_signs.items() if sign == convex_sign}
    concave = set(turn_signs) - convex
    assert len(convex) - len(concave) == 4, "simple rectilinear cycle should have four more convex bends"
    return convex, concave


def _erase_convex_bends(g: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.asarray(g, dtype=int).copy()
    path = _trace_cycle(out == GREEN)
    convex, _concave = _turn_partition(path)
    for r, c in convex:
        out[r, c] = BLACK
    return out, convex


def _unique_shortest_path(
    g: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = map(int, g.shape)
    dist = {start: 0}
    ways = {start: 1}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if int(g[nr, nc]) in (GREEN, BLUE):
                continue
            nd = dist[(r, c)] + 1
            nxt = (int(nr), int(nc))
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[(r, c)]
                q.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] += ways[(r, c)]

    if goal not in dist or ways[goal] != 1:
        return None

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if (nr, nc) in dist
            and dist[(nr, nc)] == dist[cur] - 1
            and int(g[nr, nc]) not in (GREEN, BLUE)
        ]
        if len(preds) != 1:
            return None
        cur = preds[0]
        if cur != start:
            path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the cycle's majority-turn bends to open wall gaps, then paint the unique shortest route between the red endpoints.

    Input structure
    - The grid uses black background, one green one-pixel-wide orthogonal cycle, blue blocker pixels, and exactly two red endpoints.
    - The green cycle is 4-connected and every green pixel has degree 2 inside the green set.
    - Blue and green pixels are walls; red endpoints sit on traversable cells.

    Rule
    - Trace the green cycle in order and inspect the turn direction at each bend pixel.
    - Treat the majority turn sign on that cycle as the convex class and the minority turn sign as the concave class.
    - Erase every convex bend to black, while keeping concave bends and straight green pixels unchanged.
    - In the modified grid, find the unique shortest 4-neighbor path between the two red endpoints through non-wall cells.
    - Recolor the interior cells of that path yellow and preserve every other cell.

    Key insight
    - The bend classification is structural rather than decorative: deleting the majority-turn bends changes the wall topology, and only the resulting openings make the final shortest-path trace available.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}, "unexpected input colors"

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red_coords[0]))
    goal = tuple(map(int, red_coords[1]))

    transformed, _deleted = _erase_convex_bends(g)
    path = _unique_shortest_path(transformed, start, goal)
    assert path is not None, "expected a unique shortest path after deleting convex bends"

    out = transformed.copy()
    out[start] = RED
    out[goal] = RED
    for r, c in path:
        assert int(out[r, c]) == BLACK, "path interior must traverse only passable background cells"
        out[r, c] = YELLOW
    return out


def _build_top_finger_loop(height: int, width: int, left_col: int, right_col: int, depth: int) -> np.ndarray:
    pts = [
        (1, 1),
        (1, left_col),
        (depth, left_col),
        (depth, right_col),
        (1, right_col),
        (1, width),
        (height, width),
        (height, 1),
        (1, 1),
    ]

    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            cells = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 >= r0 else -1
            cells = [(r, c0) for r in range(r0, r1 + step, step)]
        if not path:
            path.extend(cells)
        elif path[-1] == cells[0]:
            path.extend(cells[1:])
        else:
            path.extend(cells)
    if path and path[0] == path[-1]:
        path.pop()

    grid = np.zeros((height + 2, width + 2), dtype=int)
    for r, c in path:
        grid[r, c] = GREEN

    traced = _trace_cycle(grid == GREEN)
    assert len(traced) >= 20
    convex, concave = _turn_partition(traced)
    assert convex and concave
    return grid


def _adjacent_to_loop(loop_grid: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    return any(
        in_bounds(loop_grid.shape, nr, nc) and int(loop_grid[nr, nc]) == GREEN
        for nr, nc in neighbors4(r, c)
    )


def _random_transform(g: np.ndarray) -> np.ndarray:
    out = np.rot90(g, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(10, 14)
    width = random.randint(10, 14)
    left_col = random.randint(3, 5)
    right_col = random.randint(left_col + 2, min(width - 3, left_col + 4))
    depth = random.randint(max(5, height // 2), height - 3)

    base = _build_top_finger_loop(height, width, left_col, right_col, depth)
    transformed, deleted = _erase_convex_bends(base)
    empty_cells = [tuple(map(int, rc)) for rc in np.argwhere(base == BLACK)]
    blocker_slots = [cell for cell in empty_cells if not _adjacent_to_loop(base, cell)]

    for _ in range(2500):
        start, goal = random.sample(empty_cells, 2)
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 8:
            continue

        g = base.copy()
        g[start] = RED
        g[goal] = RED

        blockers = random.sample(blocker_slots, random.randint(3, min(12, len(blocker_slots))))
        for cell in blockers:
            if cell not in (start, goal):
                g[cell] = BLUE

        after_grid, deleted = _erase_convex_bends(g)
        after_path = _unique_shortest_path(after_grid, start, goal)
        if after_path is None or len(after_path) < 8:
            continue
        if not any(cell in deleted for cell in after_path):
            continue

        before_path = _unique_shortest_path(g, start, goal)
        if before_path is not None and len(before_path) <= len(after_path):
            continue

        blocker_touch = any(
            any(
                in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLUE
                for nr, nc in neighbors4(r, c)
            )
            for r, c in after_path
        )
        if not blocker_touch:
            continue

        out = main(g)
        yellow_count = int(np.count_nonzero(out == YELLOW))
        if yellow_count != len(after_path):
            continue

        return _random_transform(g)

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid loop-opening path puzzle")
