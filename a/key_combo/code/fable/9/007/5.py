from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEPARATOR_COLORS = (5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

PANEL_HEIGHT_RANGE = (9, 13)
PANEL_WIDTH_RANGE = (10, 14)
MAX_GENERATION_ATTEMPTS = 400


def _append_segment(path: list[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1

    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))

    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _append_module(
    path: list[tuple[int, int]], center_row: int, c0: int, c1: int, use_top_branch: bool
) -> None:
    bend_row = center_row - 2 if use_top_branch else center_row + 2
    _append_segment(path, path[-1], (bend_row, c0))
    _append_segment(path, path[-1], (bend_row, c1))
    _append_segment(path, path[-1], (center_row, c1))


def _validate_path_geometry(path: list[tuple[int, int]]) -> bool:
    if len(set(path)) != len(path):
        return False

    cells = set(path)
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False

    for i, (r, c) in enumerate(path):
        intended = set()
        if i > 0:
            intended.add(path[i - 1])
        if i + 1 < len(path):
            intended.add(path[i + 1])
        actual = {(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in cells}
        if actual != intended:
            return False

    return True


def _segment_len(segment: tuple[tuple[int, int], tuple[int, int]]) -> int:
    (r0, c0), (r1, c1) = segment
    return abs(r1 - r0) + abs(c1 - c0)


def _straight_runs(path: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    runs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    i = 0
    while i < len(path) - 1:
        j = i + 1
        direction = (path[j][0] - path[i][0], path[j][1] - path[i][1])
        while j + 1 < len(path):
            next_direction = (path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1])
            if next_direction != direction:
                break
            j += 1
        runs.append((path[i], path[j]))
        i = j
    return runs


def _bfs_counts(
    panel: np.ndarray, start: tuple[int, int]
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    height, width = map(int, panel.shape)
    dist = {start: 0}
    count = {start: 1}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if int(panel[nr, nc]) == BLUE:
                continue
            nxt = (nr, nc)
            if nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                count[nxt] = count[(r, c)]
                queue.append(nxt)
            elif dist[nxt] == dist[(r, c)] + 1:
                count[nxt] = min(2, count[nxt] + count[(r, c)])

    return dist, count


def _solve_panel(
    panel: np.ndarray,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    reds = np.argwhere(panel == RED)
    assert reds.shape == (2, 2), "each panel must contain exactly two red endpoints"
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))

    dist_start, count_start = _bfs_counts(panel, start)
    assert goal in dist_start, "endpoints must be connected"
    assert count_start[goal] == 1, "shortest path must be unique"
    dist_goal, _ = _bfs_counts(panel, goal)

    path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        predecessors = [
            (nr, nc)
            for nr, nc in ((cur[0] + dr, cur[1] + dc) for dr, dc in DIRS4)
            if (nr, nc) in dist_start and dist_start[(nr, nc)] == dist_start[cur] - 1
        ]
        assert len(predecessors) == 1, "unique path reconstruction failed"
        cur = predecessors[0]
        if cur != start:
            path.append(cur)
    path.reverse()
    return start, goal, path, dist_start, dist_goal


def _find_separator_column(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    colors, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}

    candidates = []
    for col in range(1, width - 1):
        color = int(grid[0, col])
        if np.all(grid[:, col] == color) and count_by_color[color] == height and 2 * col + 1 == width:
            candidates.append(col)

    assert len(candidates) == 1, "expected exactly one unique full-height separator column"
    return candidates[0]


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    separator_col = _find_separator_column(grid)
    left = np.asarray(grid[:, :separator_col], dtype=int)
    right = np.asarray(grid[:, separator_col + 1 :], dtype=int)
    assert left.shape == right.shape and left.size > 0

    left_colors = set(map(int, np.unique(left)))
    right_colors = set(map(int, np.unique(right)))
    assert left_colors <= {BLACK, BLUE, RED}
    assert right_colors <= {BLACK, BLUE, RED}

    left_reds = np.argwhere(left == RED)
    right_reds = np.argwhere(right == RED)
    assert left_reds.shape == (2, 2) and right_reds.shape == (2, 2)
    assert np.array_equal(left_reds, right_reds), "the two panels must align their endpoints"

    return left, right, separator_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the hidden shortest routes through two separator-split mazes and paint where they agree or diverge.

    Input structure:
    - One full-height separator column of a unique color splits the grid into equal-width left and right panels.
    - Each panel is a black-and-blue maze with exactly two red endpoint pixels, and the two panels place those endpoints at identical panel coordinates.
    - In each panel, the red endpoints are connected by a unique shortest 4-neighbor route through non-blue cells.

    Rule:
    - Ignore the separator and solve the left and right mazes independently.
    - In each panel, take the unique shortest path between the two red endpoints, excluding the endpoints themselves.
    - Output a single panel-sized black canvas with the red endpoints preserved: paint green where both hidden paths pass, yellow where exactly one path passes, and leave all other cells black.

    Key insight:
    - The compared objects are not the visible walls but the two unpainted maze solutions, so the task is shortest-path solving followed by an aligned xor/intersection overlay of those solved routes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left, right, _separator_col = _split_input(g)
    start_l, goal_l, path_l, _dist_l_s, _dist_l_g = _solve_panel(left)
    start_r, goal_r, path_r, _dist_r_s, _dist_r_g = _solve_panel(right)
    assert start_l == start_r and goal_l == goal_r

    out = np.full(left.shape, BLACK, dtype=int)
    out[start_l] = RED
    out[goal_l] = RED

    left_set = set(path_l)
    right_set = set(path_r)
    for r, c in left_set & right_set:
        out[r, c] = GREEN
    for r, c in left_set ^ right_set:
        out[r, c] = YELLOW
    return out


def _build_path_pair(
    height: int, width: int, module_count: int
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
] | None:
    for _ in range(200):
        if module_count == 1:
            center = random.randint(2, height - 3)
            goal_row = random.randint(1, height - 2)
            c_left = random.randint(1, 2)
            max_gap = min(5, width - 6)
            if max_gap < 3:
                continue
            c_right = c_left + random.randint(3, max_gap)
            post_col = min(width - 3, c_right + random.randint(1, 2))
            if post_col <= c_right or width - 1 - post_col < 2:
                continue

            left_top = random.choice((True, False))
            right_top = random.choice((True, False))
            if left_top == right_top:
                right_top = not left_top

            shared_segments = [
                ((center, 0), (center, c_left)),
                ((center, c_right), (center, post_col)),
                ((center, post_col), (goal_row, post_col)),
                ((goal_row, post_col), (goal_row, width - 1)),
            ]

            paths = []
            for use_top in (left_top, right_top):
                path: list[tuple[int, int]] = []
                _append_segment(path, (center, 0), (center, c_left))
                _append_module(path, center, c_left, c_right, use_top)
                _append_segment(path, path[-1], (center, post_col))
                _append_segment(path, path[-1], (goal_row, post_col))
                _append_segment(path, path[-1], (goal_row, width - 1))
                paths.append(path)
        else:
            center1 = random.randint(2, height - 3)
            center2_candidates = [r for r in range(2, height - 2) if abs(r - center1) >= 4]
            if not center2_candidates:
                continue
            center2 = random.choice(center2_candidates)
            goal_row = random.randint(1, height - 2)

            c1 = random.randint(1, 2)
            c2 = c1 + random.randint(2, 4)
            c3 = c2 + random.randint(1, 2)
            c4 = c3 + random.randint(1, 2)
            c5 = c4 + random.randint(2, 4)
            c6 = c5 + random.randint(1, 2)
            if c6 > width - 3 or width - 1 - c6 < 1:
                continue

            left_bits = [random.choice((True, False)), random.choice((True, False))]
            right_bits = [random.choice((True, False)), random.choice((True, False))]
            if left_bits == right_bits:
                right_bits[1] = not right_bits[1]

            shared_segments = [
                ((center1, 0), (center1, c1)),
                ((center1, c2), (center1, c3)),
                ((center2, c3), (center2, c4)),
                ((center2, c5), (center2, c6)),
                ((center2, c6), (goal_row, c6)),
                ((goal_row, c6), (goal_row, width - 1)),
            ]

            paths = []
            for top1, top2 in (tuple(left_bits), tuple(right_bits)):
                path = []
                _append_segment(path, (center1, 0), (center1, c1))
                _append_module(path, center1, c1, c2, top1)
                _append_segment(path, path[-1], (center1, c3))
                _append_segment(path, path[-1], (center2, c3))
                _append_segment(path, path[-1], (center2, c4))
                _append_module(path, center2, c4, c5, top2)
                _append_segment(path, path[-1], (center2, c6))
                _append_segment(path, path[-1], (goal_row, c6))
                _append_segment(path, path[-1], (goal_row, width - 1))
                paths.append(path)

        if not all(_validate_path_geometry(path) for path in paths):
            continue

        overlap = len(set(paths[0][1:-1]) & set(paths[1][1:-1]))
        exclusive = len(set(paths[0][1:-1]) ^ set(paths[1][1:-1]))
        if overlap < 4 or exclusive < 4:
            continue
        return paths[0], paths[1], shared_segments

    return None


def _make_base_panel(path: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    panel = np.full((height, width), BLUE, dtype=int)
    for r, c in path:
        panel[r, c] = BLACK
    panel[path[0]] = RED
    panel[path[-1]] = RED
    return panel


def _bypass_cells(
    segment: tuple[tuple[int, int], tuple[int, int]], sign: int
) -> list[tuple[int, int]] | None:
    (r0, c0), (r1, c1) = segment
    if r0 == r1:
        if sign not in (-1, 1):
            return None
        c0, c1 = sorted((c0, c1))
        r_mid = r0 + sign
        r_far = r0 + 2 * sign
        cells = [(r_mid, c0), (r_far, c0), (r_mid, c1), (r_far, c1)]
        cells.extend((r_far, c) for c in range(c0, c1 + 1))
        return list(dict.fromkeys(cells))

    if c0 == c1:
        if sign not in (-1, 1):
            return None
        r0, r1 = sorted((r0, r1))
        c_mid = c0 + sign
        c_far = c0 + 2 * sign
        cells = [(r0, c_mid), (r0, c_far), (r1, c_mid), (r1, c_far)]
        cells.extend((r, c_far) for r in range(r0, r1 + 1))
        return list(dict.fromkeys(cells))

    return None


def _try_add_bypass(panel: np.ndarray, segment: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray | None:
    height, width = map(int, panel.shape)
    signs = [-1, 1]
    random.shuffle(signs)
    for sign in signs:
        cells = _bypass_cells(segment, sign)
        if cells is None:
            continue
        if not all(0 <= r < height and 0 <= c < width for r, c in cells):
            continue
        if not all(int(panel[r, c]) == BLUE for r, c in cells):
            continue

        trial = panel.copy()
        for r, c in cells:
            trial[r, c] = BLACK
        try:
            _solve_panel(trial)
        except AssertionError:
            continue
        return trial
    return None


def _has_longer_route(
    panel: np.ndarray,
    path_set: set[tuple[int, int]],
    dist_start: dict[tuple[int, int], int],
    dist_goal: dict[tuple[int, int], int],
    goal: tuple[int, int],
) -> bool:
    best = dist_start[goal]
    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            if int(panel[r, c]) == BLUE or int(panel[r, c]) == RED:
                continue
            cell = (r, c)
            if cell in path_set:
                continue
            if cell in dist_start and cell in dist_goal and dist_start[cell] + dist_goal[cell] > best:
                return True
    return False


def _naive_open_overlay(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full(left.shape, BLACK, dtype=int)
    red_mask = left == RED
    out[red_mask] = RED

    left_open = left != BLUE
    right_open = right != BLUE
    both = left_open & right_open & ~red_mask
    exactly_one = (left_open ^ right_open) & ~red_mask
    out[both] = GREEN
    out[exactly_one] = YELLOW
    return out


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        left, right, _ = _split_input(grid)
        out = main(grid)
        start_l, goal_l, path_l, dist_ls, dist_lg = _solve_panel(left)
        start_r, goal_r, path_r, dist_rs, dist_rg = _solve_panel(right)
    except AssertionError:
        return False

    if start_l != start_r or goal_l != goal_r:
        return False

    set_l = set(path_l)
    set_r = set(path_r)
    if not set_l or not set_r or set_l == set_r:
        return False

    if np.sum(out == GREEN) < 4 or np.sum(out == YELLOW) < 4:
        return False

    if not _has_longer_route(left, set_l, dist_ls, dist_lg, goal_l):
        return False
    if not _has_longer_route(right, set_r, dist_rs, dist_rg, goal_r):
        return False

    shared_nonpath = ((left != BLUE) & (right != BLUE) & (out == BLACK)).copy()
    shared_nonpath[left == RED] = False
    if not np.any(shared_nonpath):
        return False

    if np.array_equal(out, _naive_open_overlay(left, right)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*PANEL_HEIGHT_RANGE)
        width = random.randint(*PANEL_WIDTH_RANGE)
        module_count = 2 if width >= 12 and height >= 11 and random.random() < 0.6 else 1

        built = _build_path_pair(height, width, module_count)
        if built is None:
            continue
        left_path, right_path, shared_segments = built

        left_panel = _make_base_panel(left_path, height, width)
        right_panel = _make_base_panel(right_path, height, width)

        common_candidates = [segment for segment in shared_segments if _segment_len(segment) >= 3]
        random.shuffle(common_candidates)
        common_added = False
        for segment in common_candidates:
            next_left = _try_add_bypass(left_panel, segment)
            if next_left is None:
                continue
            next_right = _try_add_bypass(right_panel, segment)
            if next_right is None:
                continue
            left_panel = next_left
            right_panel = next_right
            common_added = True
            break
        if not common_added:
            continue

        for panel_name, path in (("left", left_path), ("right", right_path)):
            panel = left_panel if panel_name == "left" else right_panel
            runs = [segment for segment in _straight_runs(path) if _segment_len(segment) >= 3]
            random.shuffle(runs)
            target_additions = random.randint(0, 2)
            added = 0
            for segment in runs:
                if added >= target_additions:
                    break
                trial = _try_add_bypass(panel, segment)
                if trial is None:
                    continue
                try:
                    start, goal, path_cells, dist_s, dist_g = _solve_panel(trial)
                except AssertionError:
                    continue
                if not _has_longer_route(trial, set(path_cells), dist_s, dist_g, goal):
                    continue
                panel = trial
                added += 1
            if panel_name == "left":
                left_panel = panel
            else:
                right_panel = panel

        separator_color = random.choice(SEPARATOR_COLORS)
        grid = np.full((height, 2 * width + 1), separator_color, dtype=int)
        grid[:, :width] = left_panel
        grid[:, width + 1 :] = right_panel

        if _validate_generated_grid(grid):
            return grid

    raise RuntimeError("failed to generate a valid dual-maze disagreement puzzle")
