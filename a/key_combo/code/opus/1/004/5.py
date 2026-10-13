from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GRAY = 5
CODE_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 800


def _color_set(*colors: int) -> set[int]:
    return {int(color) for color in colors}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords.sort()
            result.append(np.array(coords, dtype=int))

    return result


def _find_separator_row(grid: np.ndarray) -> int:
    h, _w = grid.shape
    allowed = _color_set(BG, *CODE_COLORS)
    candidates: list[int] = []

    for r in range(h - 1):
        if not np.all(grid[r] == GRAY):
            continue
        below = grid[r + 1]
        below_colors = {int(color) for color in np.unique(below)}
        if below_colors <= allowed and any(int(color) in CODE_COLORS for color in below_colors):
            candidates.append(r)

    assert len(candidates) == 1, "expected exactly one separator row above the code row"
    return int(candidates[0])


def _path_depths(path: list[tuple[int, int]]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for _r, c in path:
        depths[int(c)] = depths.get(int(c), 0) + 1
    return depths


def _lexicographic_shortest_path(
    starts: np.ndarray,
    goals: np.ndarray,
    passable: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in goals:
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))

    best = min(int(dist[int(r), int(c)]) for r, c in starts if int(dist[int(r), int(c)]) >= 0)
    start_candidates = sorted(
        (int(r), int(c)) for r, c in starts if int(dist[int(r), int(c)]) == best
    )
    assert start_candidates

    path: list[tuple[int, int]] = [start_candidates[0]]
    current = start_candidates[0]
    while int(dist[current[0], current[1]]) > 0:
        next_cells: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr = current[0] + dr
            nc = current[1] + dc
            if 0 <= nr < h and 0 <= nc < w and int(dist[nr, nc]) == int(dist[current[0], current[1]]) - 1:
                next_cells.append((int(nr), int(nc)))
        assert next_cells, "shortest-path predecessor missing"
        current = min(next_cells)
        path.append(current)

    return path


def _canonical_shortest_path(components: list[np.ndarray], passable: np.ndarray) -> list[tuple[int, int]]:
    assert len(components) == 2
    path_a = _lexicographic_shortest_path(components[0], components[1], passable)
    path_b = _lexicographic_shortest_path(components[1], components[0], passable)
    return min(path_a, path_b)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    sep = _find_separator_row(g)
    code_row = sep + 1
    assert code_row < h
    assert np.all(np.isin(g[code_row], np.array((BG,) + CODE_COLORS, dtype=int)))
    assert np.any(np.isin(g[code_row], np.array(CODE_COLORS, dtype=int)))

    endpoint_components = _components(g == RED)
    assert len(endpoint_components) == 2, "expected exactly two red endpoint components"

    passable = g != GRAY
    path = _canonical_shortest_path(endpoint_components, passable)
    assert path
    assert all(int(r) < sep for r, _c in path), "the shortest path must remain above the separator"
    depths = _path_depths(path)

    return {
        "grid": g,
        "separator": int(sep),
        "code_row": int(code_row),
        "components": endpoint_components,
        "path": path,
        "depths": depths,
    }


def _draw_histogram(grid: np.ndarray, separator: int, code_row: int, depths: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for c, depth in depths.items():
        color = int(out[code_row, int(c)])
        if color == BG:
            continue
        top = int(separator) - int(depth)
        assert top >= 0
        assert np.all(out[top:int(separator), int(c)] == BG), "bars must occupy blank space only"
        out[top:int(separator), int(c)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the hidden shortest-path column profile into a colored histogram above the separator row.

    Input structure:
    - The grid has black background, gray wall pixels in the upper area, and exactly two small red endpoint components embedded near the left and right sides of the maze.
    - A full gray separator row appears below the maze, and the row immediately under it is a code row containing isolated non-black color pixels in some columns.
    - The rows below the code row are otherwise empty background.

    Rule:
    - Treat gray pixels as blocked and every other pixel as passable, then find the shortest 4-connected path from either red endpoint component to the other.
    - If several shortest paths exist, choose the lexicographically smallest coordinate sequence.
    - For each column touched by that chosen path, count how many path cells lie in the column.
    - For each such column whose code-row cell is non-black, draw a vertical bar of that counted height directly above the separator using the code-row color; skip path columns whose code-row cell is black.
    - Preserve the original maze, endpoints, separator row, and code row; the path itself is never drawn.

    Key insight:
    - The visible output is not the path but a per-column measurement of the path, with the code row supplying colors and the hidden shortest path supplying the bar heights.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _draw_histogram(info["grid"], info["separator"], info["code_row"], info["depths"])


def _sample_turn_columns(start_col: int, end_col: int, turn_count: int) -> list[int] | None:
    if turn_count <= 0:
        return []
    candidates = list(range(start_col + 2, end_col - 1))
    if len(candidates) < turn_count:
        return None

    for _ in range(200):
        cols = sorted(random.sample(candidates, turn_count))
        checkpoints = [start_col] + cols + [end_col]
        if min(b - a for a, b in zip(checkpoints, checkpoints[1:])) >= 2:
            return cols
    return None


def _sample_rows(usable_rows: list[int], turn_count: int) -> list[int]:
    assert usable_rows
    for _ in range(200):
        rows = [random.choice(usable_rows)]
        for _step in range(turn_count):
            choices = [row for row in usable_rows if row != rows[-1]]
            rows.append(random.choice(choices))
        if any(abs(a - b) >= 1 for a, b in zip(rows, rows[1:])):
            return rows
    raise RuntimeError("failed to sample a row sequence")


def _polyline_path(start_col: int, end_col: int, rows: list[int], turn_cols: list[int]) -> list[tuple[int, int]]:
    assert len(rows) == len(turn_cols) + 1
    path: list[tuple[int, int]] = [(int(rows[0]), int(start_col))]
    current_r = int(rows[0])
    current_c = int(start_col)

    for turn_c, next_r in zip(turn_cols, rows[1:]):
        for c in range(current_c + 1, int(turn_c) + 1):
            path.append((current_r, int(c)))
        step = 1 if int(next_r) > current_r else -1
        for r in range(current_r + step, int(next_r) + step, step):
            path.append((int(r), int(turn_c)))
        current_r = int(next_r)
        current_c = int(turn_c)

    for c in range(current_c + 1, int(end_col) + 1):
        path.append((current_r, int(c)))

    return path


def _carve_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _try_vertical_branch(
    open_mask: np.ndarray,
    root: tuple[int, int],
    direction: int,
    length: int,
    maze_height: int,
) -> list[tuple[int, int]] | None:
    temp_open = np.array(open_mask, copy=True)
    rr, cc = root
    branch: list[tuple[int, int]] = []

    for step in range(1, int(length) + 1):
        nr = int(rr) + int(direction) * step
        nc = int(cc)
        if not (1 <= nr <= maze_height - 2):
            return None
        if temp_open[nr, nc]:
            return None

        neighbors = 0
        for dr, dc in DIR4:
            ar = nr + dr
            ac = nc + dc
            if 0 <= ar < temp_open.shape[0] and 0 <= ac < temp_open.shape[1] and temp_open[ar, ac]:
                neighbors += 1
        if neighbors != 1:
            return None

        temp_open[nr, nc] = True
        branch.append((int(nr), int(nc)))

    return branch


def _canonicalize_path(path: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    forward = tuple((int(r), int(c)) for r, c in path)
    backward = tuple(reversed(forward))
    return min(forward, backward)


def _maze_open_depths(grid: np.ndarray, maze_height: int) -> dict[int, int]:
    depths: dict[int, int] = {}
    for c in range(grid.shape[1]):
        depths[int(c)] = int(np.count_nonzero(grid[:maze_height, int(c)] != GRAY))
    return depths


def _wrong_open_depth_output(
    grid: np.ndarray,
    separator: int,
    code_row: int,
    maze_height: int,
) -> np.ndarray:
    open_depths = _maze_open_depths(grid, maze_height)
    return _draw_histogram(grid, separator, code_row, open_depths)


def _wrong_unit_height_output(
    grid: np.ndarray,
    separator: int,
    code_row: int,
    path_cols: set[int],
) -> np.ndarray:
    depths = {int(c): 1 for c in path_cols}
    return _draw_histogram(grid, separator, code_row, depths)


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(16, 24)
    bar_height = random.randint(4, 6)
    maze_height = random.randint(bar_height + 2, bar_height + 4)
    bottom_rows = random.randint(1, 3)
    height = maze_height + bar_height + 2 + bottom_rows
    if height > 30:
        return None

    separator = maze_height + bar_height
    code_row = separator + 1

    left_len = random.choice((2, 3))
    right_len = random.choice((2, 3))
    start_col = left_len - 1
    end_col = width - right_len
    usable_rows = list(range(1, min(maze_height - 2, bar_height) + 1))
    if len(usable_rows) < 3:
        return None

    max_turns = min(5, (end_col - start_col - 2) // 2)
    if max_turns < 2:
        return None
    turn_count = random.randint(2, max_turns)
    turn_cols = _sample_turn_columns(start_col, end_col, turn_count)
    if turn_cols is None:
        return None
    rows = _sample_rows(usable_rows, turn_count)
    path = _polyline_path(start_col, end_col, rows, turn_cols)
    path_depths = _path_depths(path)

    grid = np.full((height, width), BG, dtype=int)
    grid[:maze_height, :] = GRAY
    grid[separator, :] = GRAY

    _carve_cells(grid, path, BG)
    left_cells = [(int(rows[0]), c) for c in range(left_len)]
    right_cells = [(int(rows[-1]), c) for c in range(width - right_len, width)]
    _carve_cells(grid, left_cells + right_cells, RED)

    open_mask = grid[:maze_height, :] != GRAY
    straight_roots = [
        (int(r), int(c))
        for r, c in path
        if int(path_depths[int(c)]) == 1 and int(start_col) + 1 <= int(c) <= int(end_col) - 1 and int(grid[int(r), int(c)]) == BG
    ]
    random.shuffle(straight_roots)

    desired_branches = random.randint(1, min(3, len(straight_roots)))
    branch_cols: list[int] = []
    used_root_cols: set[int] = set()

    for root in straight_roots:
        if len(branch_cols) >= desired_branches:
            break
        rr, cc = root
        if int(cc) in used_root_cols:
            continue

        placements: list[list[tuple[int, int]]] = []
        directions = [-1, 1]
        random.shuffle(directions)
        for direction in directions:
            max_len = (rr - 1) if direction < 0 else (maze_height - 2 - rr)
            for length in random.sample(list(range(1, min(3, max_len) + 1)), k=min(3, max_len)):
                branch = _try_vertical_branch(open_mask, root, direction, length, maze_height)
                if branch is not None:
                    placements.append(branch)

        if not placements:
            continue

        branch = random.choice(placements)
        _carve_cells(grid, branch, BG)
        for br, bc in branch:
            open_mask[int(br), int(bc)] = True
        branch_cols.append(int(cc))
        used_root_cols.add(int(cc))

    if not branch_cols:
        return None

    path_cols = sorted(int(c) for c in path_depths)
    turn_cols_used = [int(c) for c, depth in path_depths.items() if int(depth) > 1]
    straight_cols = [int(c) for c, depth in path_depths.items() if int(depth) == 1]
    if not turn_cols_used or not straight_cols:
        return None

    nonpath_open_cols = [
        c
        for c in range(width)
        if c not in path_depths and int(np.count_nonzero(grid[:maze_height, int(c)] != GRAY)) > 0
    ]
    if not nonpath_open_cols:
        return None

    colored_cols: list[int] = []
    mandatory = [
        random.choice(turn_cols_used),
        random.choice(branch_cols),
        random.choice(nonpath_open_cols),
    ]
    for col in mandatory:
        if int(col) not in colored_cols:
            colored_cols.append(int(col))

    extra_pool = [c for c in range(width) if c not in colored_cols]
    target_count = random.randint(len(colored_cols) + 2, min(len(CODE_COLORS), len(colored_cols) + 5))
    while len(colored_cols) < target_count and extra_pool:
        choice = random.choice(extra_pool)
        extra_pool.remove(choice)
        colored_cols.append(int(choice))

    random.shuffle(colored_cols)
    colors = random.sample(CODE_COLORS, len(colored_cols))
    for c, color in zip(colored_cols, colors):
        grid[code_row, int(c)] = int(color)

    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid == GRAY):
        return None

    hidden_path = _canonicalize_path(path)
    expected = _draw_histogram(grid, separator, code_row, path_depths)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    analysis = _analyze(grid)
    if tuple(analysis["path"]) != hidden_path:
        return None
    if analysis["depths"] != path_depths:
        return None

    colored_path_cols = [c for c in path_cols if int(grid[code_row, int(c)]) != BG]
    if not colored_path_cols:
        return None
    if not any(int(grid[code_row, int(c)]) == BG for c in path_cols):
        return None
    if not any(c not in path_depths and int(grid[code_row, int(c)]) != BG for c in range(width)):
        return None
    if not any(path_depths[int(c)] > 1 and int(grid[code_row, int(c)]) != BG for c in turn_cols_used):
        return None

    maze_open_depths = _maze_open_depths(grid, maze_height)
    if not any(
        int(grid[code_row, int(c)]) != BG and maze_open_depths[int(c)] > path_depths[int(c)]
        for c in branch_cols
    ):
        return None
    if len({int(path_depths[int(c)]) for c in colored_path_cols}) < 2:
        return None

    wrong_open = _wrong_open_depth_output(grid, separator, code_row, maze_height)
    wrong_unit = _wrong_unit_height_output(grid, separator, code_row, set(path_cols))
    if np.array_equal(wrong_open, expected):
        return None
    if np.array_equal(wrong_unit, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
