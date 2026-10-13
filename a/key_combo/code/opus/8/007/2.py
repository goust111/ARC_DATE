from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _find_marker_block(grid: np.ndarray) -> tuple[int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.shape == (4, 2), "expected exactly one 2x2 grey marker block"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "grey marker must be 2x2"
    expected = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
    assert {tuple(map(int, rc)) for rc in grey} == expected, "grey marker must be a solid 2x2 block"
    return int(r0), int(c0)


def _apply_cross_erase(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0 = _find_marker_block(out)
    rows = (r0, r0 + 1)
    cols = (c0, c0 + 1)

    for r in rows:
        for c in range(out.shape[1]):
            if int(out[r, c]) in (BLUE, GREY):
                out[r, c] = BLACK
    for c in cols:
        for r in range(out.shape[0]):
            if int(out[r, c]) in (BLUE, GREY):
                out[r, c] = BLACK

    return out


def _distances(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int] | None = None) -> dict[tuple[int, int], int]:
    h, w = map(int, grid.shape)
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        if goal is not None and (r, c) == goal:
            break
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) in (BLUE, GREY):
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _unique_shortest_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    dist = _distances(grid, start, goal)
    assert goal in dist, "expected the red endpoints to be connected after erasing the cross"

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in dist and dist[(nr, nc)] == dist[cur] - 1
        ]
        assert len(preds) == 1, "expected a unique shortest path after erasing the cross"
        cur = preds[0]
        if cur != start:
            path.append(cur)
    path.reverse()
    return path


def _path_length(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    dist = _distances(grid, start, goal)
    return dist.get(goal)


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in path:
        assert int(out[r, c]) == BLACK, "only black interior cells should be repainted"
        out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the grey block's two rows and two columns from the maze walls, then paint the unique shortest route between the red endpoints.

    Input structure:
    - The grid is a blue-wall maze on a black background with exactly two red endpoint pixels.
    - Exactly one 2x2 grey marker block sits on wall cells and specifies two adjacent rows and two adjacent columns.
    - The maze topology changes only where those marked rows and columns are cleared.

    Rule:
    - Find the 2x2 grey block and take its two rows and two columns.
    - Turn every blue wall on those rows or columns black, and also turn the four grey marker cells black; keep any red endpoints unchanged.
    - On the modified maze, treat blue as blocked and find the unique shortest 4-neighbor path between the two red endpoints.
    - Recolor only the black interior cells of that path yellow, preserving every other cell.

    Key insight:
    - The grey 2x2 marker is a topology modifier: you must open its full row-and-column cross before solving the maze, because the intended shortest path exists only in the erased maze.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    red = np.argwhere(g == RED)
    assert red.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, red[0]))
    goal = tuple(map(int, red[1]))

    erased = _apply_cross_erase(g)
    path = _unique_shortest_path(erased, start, goal)
    return _paint_path(erased, path)


def _carve_h(grid: np.ndarray, r: int, c0: int, c1: int, color: int = BLACK) -> None:
    if c0 > c1:
        c0, c1 = c1, c0
    grid[r, c0 : c1 + 1] = color


def _carve_v(grid: np.ndarray, c: int, r0: int, r1: int, color: int = BLACK) -> None:
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, c] = color


def _apply_transform(grid: np.ndarray, k: int, flip_lr: bool, flip_ud: bool) -> np.ndarray:
    out = np.rot90(grid, k)
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _add_spur(grid: np.ndarray, anchor: tuple[int, int], blocked_rows: set[int], blocked_cols: set[int]) -> None:
    h, w = grid.shape
    ar, ac = anchor
    options: list[list[tuple[int, int]]] = []

    if ac >= 2:
        end_r = random.choice([r for r in range(h) if r not in blocked_rows])
        mid_c = random.randint(0, max(0, ac - 2))
        options.append([(ar, ac), (ar, mid_c), (end_r, mid_c)])
    if ac + 2 < w:
        end_r = random.choice([r for r in range(h) if r not in blocked_rows])
        mid_c = random.randint(ac + 2, w - 1)
        options.append([(ar, ac), (ar, mid_c), (end_r, mid_c)])
    if ar >= 2:
        end_c = random.choice([c for c in range(w) if c not in blocked_cols])
        mid_r = random.randint(0, max(0, ar - 2))
        options.append([(ar, ac), (mid_r, ac), (mid_r, end_c)])
    if ar + 2 < h:
        end_c = random.choice([c for c in range(w) if c not in blocked_cols])
        mid_r = random.randint(ar + 2, h - 1)
        options.append([(ar, ac), (mid_r, ac), (mid_r, end_c)])

    if not options:
        return

    p0, p1, p2 = random.choice(options)
    _carve_h(grid, p0[0], p0[1], p1[1])
    _carve_v(grid, p1[1], p0[0], p2[0])
    _carve_h(grid, p2[0], p1[1], p2[1])


def _build_candidate() -> np.ndarray:
    h = random.randint(13, 22)
    w = random.randint(13, 22)
    grid = np.full((h, w), BLUE, dtype=int)

    r0 = random.randint(3, h - 6)
    c0 = random.randint(3, w - 6)
    r1 = r0 + 1
    c1 = c0 + 1
    rr = random.choice([r0, r1])
    cc = random.choice([c0, c1])

    left_attach_c = random.randint(1, c0 - 2)
    below_attach_r = random.randint(r1 + 2, h - 2)

    start_on_erased = random.random() < 0.35
    goal_on_erased = random.random() < 0.35

    if start_on_erased:
        start_r = rr
        start_c = random.randint(0, left_attach_c)
    else:
        start_r = random.choice([r for r in range(h) if r not in (r0, r1)])
        start_c = random.randint(0, left_attach_c)

    if goal_on_erased:
        goal_r = random.randint(below_attach_r, h - 1)
        goal_c = cc
    else:
        goal_r = random.randint(below_attach_r, h - 1)
        goal_c = random.randint(c1 + 2, w - 1)

    _carve_v(grid, start_c, start_r, rr)
    _carve_h(grid, rr, start_c, left_attach_c)

    _carve_v(grid, goal_c, below_attach_r, goal_r)
    _carve_h(grid, below_attach_r, cc, goal_c)

    _carve_h(grid, rr, left_attach_c, c0 - 1)
    _carve_v(grid, cc, r1 + 1, below_attach_r)

    spur_count = random.randint(0, 2)
    anchors = [(rr, left_attach_c), (below_attach_r, goal_c)]
    blocked_rows = {r0, r1}
    blocked_cols = {c0, c1}
    for _ in range(spur_count):
        _add_spur(grid, random.choice(anchors), blocked_rows, blocked_cols)

    grid[r0 : r0 + 2, c0 : c0 + 2] = GREY
    grid[start_r, start_c] = RED
    grid[goal_r, goal_c] = RED

    return _apply_transform(
        grid,
        k=random.randrange(4),
        flip_lr=bool(random.getrandbits(1)),
        flip_ud=bool(random.getrandbits(1)),
    )


def _is_valid_candidate(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    red = np.argwhere(grid == RED)
    if red.shape != (2, 2):
        return False
    start = tuple(map(int, red[0]))
    goal = tuple(map(int, red[1]))

    try:
        erased = _apply_cross_erase(grid)
        path = _unique_shortest_path(erased, start, goal)
    except AssertionError:
        return False

    before = _path_length(grid, start, goal)
    after = _path_length(erased, start, goal)
    if after is None or len(path) == 0:
        return False
    if before is not None and before <= after:
        return False

    solved = main(grid)
    yellow = np.argwhere(solved == YELLOW)
    return yellow.shape[0] == len(path)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        if _is_valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid cross-erase maze input")
