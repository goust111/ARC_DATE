from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4, tile_fill


BG = Color.BLACK
WALL = Color.TEAL
ENDPOINT = Color.RED
PATH = Color.YELLOW

GEN_MAX_ATTEMPTS = 400


def _infer_period(grid: np.ndarray) -> int:
    wall_mask = np.asarray(grid == WALL, dtype=bool)
    h = int(wall_mask.shape[0])
    candidates: list[int] = []
    for period in range(1, h // 2 + 1):
        if wall_mask[2 * period :].any():
            continue
        if not wall_mask[:period].any():
            continue
        if np.array_equal(wall_mask[:period], wall_mask[period : 2 * period]):
            candidates.append(period)
    assert candidates, "failed to infer the visible wall period"
    return min(candidates)


def _reconstruct_maze(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    period = _infer_period(g)
    period_tile = np.where(g[:period] == WALL, WALL, BG)
    maze = tile_fill(g.shape, period_tile)
    for r, c in np.argwhere(g == ENDPOINT):
        maze[int(r), int(c)] = ENDPOINT
    return maze.astype(int), period


def _unique_shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    dist: dict[tuple[int, int], int] = {start: 0}
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) == WALL:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))

    assert goal in dist, "the red endpoints must be connected after wall extrapolation"

    rev_path: list[tuple[int, int]] = []
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if (nr, nc) in dist and dist[(nr, nc)] == dist[cur] - 1
        ]
        assert len(preds) == 1, "expected a unique shortest path between the endpoints"
        cur = preds[0]
        if cur != start:
            rev_path.append(cur)
    return list(reversed(rev_path))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the hidden vertical wall period, then paint the unique shortest route between the red endpoints.

    Input structure:
    - The background is black.
    - Cyan wall pixels appear only in the top part of the grid and form exactly two full repeats of one vertical period.
    - Two red endpoint pixels lie below the drawn wall region, where the implied maze has been omitted.

    Rule:
    - Read the wall period from the visible top two repeats and continue that same row pattern downward to the bottom edge.
    - Treat both the visible and extrapolated cyan cells as blocking maze walls.
    - Find the unique shortest 4-neighbor path between the two red endpoints and paint only its black interior cells yellow.

    Key insight:
    - The lower empty area is incomplete, not open space: the walls must be mentally repeated downward before the shortest path is computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    maze, _period = _reconstruct_maze(g)
    reds = np.argwhere(g == ENDPOINT)
    assert reds.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))

    out = maze.copy()
    for r, c in _unique_shortest_path(maze, start, goal):
        assert int(out[r, c]) == BG, "path interior must lie on background cells"
        out[r, c] = PATH
    return out


def _path_from_columns(columns: list[int]) -> list[tuple[int, int]]:
    period = len(columns) - 1
    path = [(0, int(columns[0]))]
    for r in range(period):
        c0 = int(columns[r])
        c1 = int(columns[r + 1])
        if c0 != c1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1 + step, step):
                path.append((r, c))
        if r < period - 1:
            path.append((r + 1, c1))
    assert len(path) == len(set(path)), "period path must be simple"
    return path


def _split_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cut = random.randint(1, total - 1)
    return [int(cut), int(total - cut)]


def _columns_chain_ok(columns: list[int], width: int) -> bool:
    for a, b in zip(columns, columns[1:]):
        if abs(a - b) >= width:
            return False
        if min(a, b) == 0 and max(a, b) == width - 1:
            return False

    for r in range(len(columns) - 2):
        a = int(columns[r])
        b = int(columns[r + 1])
        c = int(columns[r + 2])
        lo1, hi1 = sorted((a, b))
        lo2, hi2 = sorted((b, c))
        inter_lo = max(lo1, lo2)
        inter_hi = min(hi1, hi2)
        if inter_lo != inter_hi or inter_lo != b:
            return False

    a = int(columns[-2])
    b = int(columns[-1])
    c = int(columns[1])
    lo1, hi1 = sorted((a, b))
    lo2, hi2 = sorted((b, c))
    inter_lo = max(lo1, lo2)
    inter_hi = min(hi1, hi2)
    if inter_lo != inter_hi or inter_lo != b:
        return False
    return True


def _make_period_path(period: int, width: int) -> tuple[list[int], list[tuple[int, int]]]:
    for _ in range(300):
        start_col = random.randint(1, width - 2)
        if start_col < 1 or start_col > width - 2:
            continue

        left_col = random.randint(0, start_col - 1)
        right_col = random.randint(start_col + 1, width - 1)
        if right_col - left_col >= width - 1:
            continue
        if max(start_col - left_col, right_col - start_col) < 2:
            continue

        run_rows = [1, 1, 1]
        if period == 6:
            choices = []
            if right_col - start_col >= 2:
                choices.append(0)
            if right_col - left_col >= 2:
                choices.append(1)
            if start_col - left_col >= 2:
                choices.append(2)
            if not choices:
                continue
            run_rows[random.choice(choices)] = 2

        out_steps = _split_positive(right_col - start_col, run_rows[0])
        cross_steps = _split_positive(right_col - left_col, run_rows[1])
        back_steps = _split_positive(start_col - left_col, run_rows[2])
        diffs = out_steps + [0] + [-step for step in cross_steps] + [0] + back_steps
        if len(diffs) != period:
            continue
        columns = [start_col]
        cur = start_col
        for diff in diffs:
            cur += diff
            columns.append(int(cur))
        if columns[-1] != start_col:
            continue
        if not _columns_chain_ok(columns, width):
            continue
        if len(set(columns)) < 3:
            continue
        path = _path_from_columns(columns)
        return columns, path
    raise RuntimeError("failed to build a non-degenerate period path")


def _period_grid(period: int, width: int, block_path: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((period, width), WALL, dtype=int)
    for r, c in block_path:
        grid[int(r), int(c)] = BG
    assert np.all(np.any(grid == WALL, axis=1)), "every visible period row should contain a wall"
    return grid


def _stack_path(
    block_path: list[tuple[int, int]], period: int, height: int
) -> list[tuple[int, int]]:
    full_path: list[tuple[int, int]] = []
    for base_r in range(0, height, period):
        for r, c in block_path:
            rr = base_r + int(r)
            if rr >= height:
                break
            full_path.append((rr, int(c)))
    return full_path


def _path_graph_ok(full_grid: np.ndarray) -> bool:
    open_cells = {tuple(map(int, rc)) for rc in np.argwhere(full_grid != WALL)}
    if not open_cells:
        return False

    start = next(iter(open_cells))
    seen = {start}
    stack = [start]
    endpoint_count = 0

    while stack:
        r, c = stack.pop()
        deg = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in open_cells:
                continue
            deg += 1
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
        if deg > 2:
            return False
        if deg == 0 and len(open_cells) > 1:
            return False
        if deg == 1:
            endpoint_count += 1

    return seen == open_cells and endpoint_count == 2


def _render_expected_output(
    full_grid: np.ndarray,
    full_path: list[tuple[int, int]],
    idx_a: int,
    idx_b: int,
) -> np.ndarray:
    lo, hi = sorted((idx_a, idx_b))
    out = full_grid.copy()
    a = full_path[idx_a]
    b = full_path[idx_b]
    out[a] = ENDPOINT
    out[b] = ENDPOINT
    for r, c in full_path[lo + 1 : hi]:
        out[int(r), int(c)] = PATH
    return out


def _choose_height(period: int) -> int:
    exact_multiple = random.random() < 0.5
    if exact_multiple:
        max_periods = min(4, 30 // period)
        total_periods = random.randint(3, max_periods)
        return total_periods * period

    max_full_periods = min(4, (30 - 1) // period)
    total_periods = random.randint(3, max_full_periods)
    max_remainder = min(period - 1, 30 - total_periods * period)
    if max_remainder <= 0:
        return total_periods * period
    return total_periods * period + random.randint(1, max_remainder)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    period = random.choice((5, 6))
    width = random.randint(8, 14)
    height = _choose_height(period)

    _columns, block_path = _make_period_path(period, width)
    period_grid = _period_grid(period, width, block_path)
    full_grid = tile_fill((height, width), period_grid).astype(int)
    full_path = _stack_path(block_path, period, height)

    assert _path_graph_ok(full_grid), "constructed maze must expose a single corridor chain"

    hidden_indices = [i for i, (r, _c) in enumerate(full_path) if r >= 2 * period]
    min_sep = max(period, 5)
    if len(hidden_indices) <= min_sep:
        raise RuntimeError("not enough hidden corridor cells for two separated endpoints")

    pos_a = random.randint(0, len(hidden_indices) - min_sep - 1)
    pos_b = random.randint(pos_a + min_sep, len(hidden_indices) - 1)
    idx_a = hidden_indices[pos_a]
    idx_b = hidden_indices[pos_b]
    a = full_path[idx_a]
    b = full_path[idx_b]

    inp = np.zeros((height, width), dtype=int)
    inp[: 2 * period] = full_grid[: 2 * period]
    inp[a] = ENDPOINT
    inp[b] = ENDPOINT

    expected = _render_expected_output(full_grid, full_path, idx_a, idx_b)
    return inp, expected


def _case_is_valid(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    reds = np.argwhere(inp == ENDPOINT)
    if reds.shape != (2, 2):
        return False

    try:
        period = _infer_period(inp)
        solved = main(inp)
    except Exception:
        return False

    if np.any(inp[2 * period :] == WALL):
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(solved, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp, expected = _generate_case()
        if _case_is_valid(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid periodic wall maze")
