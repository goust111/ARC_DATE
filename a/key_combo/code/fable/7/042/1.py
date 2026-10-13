from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5


def _mirror_col(col: int, width: int) -> int:
    return width - 1 - col


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_roles(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    grey = grid == GREY
    paired = grey & np.fliplr(grey)
    endpoints = np.argwhere(grey & ~np.fliplr(grey))
    assert endpoints.shape == (2, 2), "expected exactly two unmatched grey endpoints"
    start = tuple(map(int, endpoints[0]))
    goal = tuple(map(int, endpoints[1]))
    return paired, start, goal


def _unique_shortest_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    sr, sc = start
    dist[sr, sc] = 0
    ways[sr, sc] = 1
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) != goal and int(grid[nr, nc]) != BLACK:
                continue
            nd = int(dist[r, c]) + 1
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
            elif dist[nr, nc] == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    gr, gc = goal
    assert dist[gr, gc] >= 0, "expected a path between the two unmatched grey endpoints"
    assert ways[gr, gc] == 1, "expected a unique shortest path"

    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the interior of the unique shortest corridor between the two unmatched grey endpoints.

    Input structure:
    - The grid has black background cells and grey foreground cells, and the width is even.
    - A grey cell whose vertical-mirror partner is also grey belongs to a wall pair.
    - Exactly two grey cells lack grey mirror partners; these are the path endpoints.

    Rule:
    - Treat every mirror-paired grey cell as an impassable wall.
    - Treat the two unmatched grey cells as the start and goal and search through 4-neighbor black cells.
    - Recolor only the black interior cells of the unique shortest path to yellow, keeping every grey cell unchanged.

    Key insight:
    - Grey pixels are not classified by color alone; mirror pairing separates wall pixels from the two endpoint pixels before pathfinding begins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 1 <= h <= 30 and 1 <= w <= 30, "grid must satisfy ARC size bounds"
    assert w % 2 == 0, "expected an even-width grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "unexpected input colors"

    walls, start, goal = _extract_roles(g)
    solved = g.copy()
    blocked = np.where(walls, GREY, g)
    path = _unique_shortest_path(blocked, start, goal)
    for r, c in path[1:-1]:
        assert int(solved[r, c]) == BLACK, "path interior must pass through black cells only"
        solved[r, c] = YELLOW
    return solved


MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (9, 17)
HALF_WIDTH_RANGE = (6, 13)
MIN_INTERIOR_PATH = 5
TARGET_BLACK_FRACTION = 0.24


def _extend_straight(path: list[tuple[int, int]], dst: tuple[int, int]) -> None:
    r0, c0 = path[-1]
    r1, c1 = dst
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            path.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            path.append((r, c0))


def _build_left_path(height: int, half_width: int) -> list[tuple[int, int]] | None:
    if half_width < 4 or height < 5:
        return None

    start_col = random.randint(0, min(2, half_width - 4))
    start_row = random.randint(1, height - 2)
    max_turn_cols = list(range(start_col + 1, half_width - 1))
    if not max_turn_cols:
        return None

    turn_count = random.randint(2, min(5, len(max_turn_cols)))
    turn_cols = sorted(random.sample(max_turn_cols, turn_count))

    path = [(start_row, start_col)]
    current_row = start_row
    for col in turn_cols:
        _extend_straight(path, (current_row, col))
        if col == turn_cols[-1]:
            continue
        row_choices = [r for r in range(1, height - 1) if r != current_row]
        if not row_choices:
            return None
        next_row = random.choice(row_choices)
        _extend_straight(path, (next_row, col))
        current_row = next_row

    _extend_straight(path, (current_row, half_width - 1))
    return path


def _ordered_full_path(left_path: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    mirrored = [(r, _mirror_col(c, width)) for r, c in reversed(left_path)]
    return list(left_path) + mirrored


def _mask_from_cells(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _add_vertical_branch(mask: np.ndarray, attach: tuple[int, int]) -> bool:
    h, w = map(int, mask.shape)
    r0, c0 = attach
    if c0 >= w // 2 - 1:
        return False
    direction = random.choice((-1, 1))
    target_len = random.randint(1, 4)

    left_cells: list[tuple[int, int]] = []
    r = r0
    for _ in range(target_len):
        r += direction
        if not (0 <= r < h):
            break
        if mask[r, c0]:
            break
        prev = attach if not left_cells else left_cells[-1]
        black_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c0) if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]]
        if black_neighbors != [prev] and set(black_neighbors) != {prev}:
            break
        left_cells.append((r, c0))

    if not left_cells:
        return False

    mirror_cells = [(r, _mirror_col(c, w)) for r, c in left_cells]
    for cell in mirror_cells:
        if mask[cell]:
            return False

    for idx, (r, c) in enumerate(mirror_cells):
        prev = (r0, _mirror_col(c0, w)) if idx == 0 else mirror_cells[idx - 1]
        black_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]]
        if black_neighbors != [prev] and set(black_neighbors) != {prev}:
            return False

    for cell in left_cells + mirror_cells:
        mask[cell] = True
    return True


def _add_room(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    half_width = w // 2
    room_h = random.randint(1, min(4, h - 2))
    room_w = random.randint(1, min(3, half_width - 2))
    r0 = random.randint(1, h - room_h - 1)
    c0 = random.randint(0, half_width - room_w - 1)

    left = [(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)]
    if any(c >= half_width - 1 for _r, c in left):
        return False
    right = [(r, _mirror_col(c, w)) for r, c in left]
    candidate = set(left + right)
    if any(mask[cell] for cell in candidate):
        return False

    for r, c in candidate:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return False

    for cell in candidate:
        mask[cell] = True
    return True


def _choose_endpoints(full_path: list[tuple[int, int]], width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    n = len(full_path)
    left_limit = n // 2 - 1
    right_start = n // 2 + 1
    if left_limit < 1 or right_start >= n - 1:
        return None

    for _ in range(80):
        i = random.randint(0, left_limit)
        j = random.randint(right_start, n - 1)
        if j - i - 1 < MIN_INTERIOR_PATH:
            continue
        a = full_path[i]
        b = full_path[j]
        if b == (a[0], _mirror_col(a[1], width)):
            continue
        return a, b
    return None


def _build_grid() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    half_width = random.randint(*HALF_WIDTH_RANGE)
    width = half_width * 2

    left_path = _build_left_path(height, half_width)
    if left_path is None:
        return None

    full_path = _ordered_full_path(left_path, width)
    endpoints = _choose_endpoints(full_path, width)
    if endpoints is None:
        return None
    start, goal = endpoints

    black_mask = _mask_from_cells((height, width), full_path)

    branch_anchors = left_path[1:-1]
    random.shuffle(branch_anchors)
    branch_target = random.randint(2, min(6, len(branch_anchors))) if branch_anchors else 0
    branch_count = 0
    for attach in branch_anchors:
        if _add_vertical_branch(black_mask, attach):
            branch_count += 1
        if branch_count >= branch_target:
            break

    room_attempts = 0
    while black_mask.mean() < TARGET_BLACK_FRACTION and room_attempts < 20:
        _add_room(black_mask)
        room_attempts += 1

    grid = np.full((height, width), GREY, dtype=int)
    grid[black_mask] = BLACK
    grid[start] = GREY
    grid[goal] = GREY

    unmatched = np.argwhere((grid == GREY) & ~np.fliplr(grid == GREY))
    if unmatched.shape != (2, 2):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
