from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LOCK_COLORS = (1, 2, 3, 4, 6, 7, 8)

REGION_COUNT_CHOICES = (2, 3, 4)
REGION_SIZE_MIN = 3
REGION_SIZE_MAX = 5
LOCK_PRESENT_PROB = 0.8
GEN_RETRIES = 500


@dataclass(frozen=True)
class Lock:
    row: int
    col: int
    color: int
    region_a: int
    region_b: int


@dataclass
class Puzzle:
    row_spans: list[tuple[int, int]]
    col_spans: list[tuple[int, int]]
    region_keys: list[set[int]]
    start: tuple[int, int]
    start_region: int
    locks: list[Lock]


def _region_id(row_idx: int, col_idx: int, n_cols: int) -> int:
    return row_idx * n_cols + col_idx


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if np.all(grid[r, :] != BLACK) and np.any(grid[r, :] == GREY)]
    wall_cols = [c for c in range(w) if np.all(grid[:, c] != BLACK) and np.any(grid[:, c] == GREY)]

    assert wall_rows and wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols and wall_cols[0] == 0 and wall_cols[-1] == w - 1
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:]))
    assert all(b - a >= 2 for a, b in zip(wall_cols, wall_cols[1:]))
    for r in wall_rows:
        for c in wall_cols:
            assert int(grid[r, c]) == GREY
    return wall_rows, wall_cols


def _analyze_puzzle(grid: np.ndarray) -> Puzzle:
    g = np.asarray(grid, dtype=int)
    wall_rows, wall_cols = _find_wall_lines(g)
    row_spans = [(wall_rows[i], wall_rows[i + 1]) for i in range(len(wall_rows) - 1)]
    col_spans = [(wall_cols[j], wall_cols[j + 1]) for j in range(len(wall_cols) - 1)]

    n_row_regions = len(row_spans)
    n_col_regions = len(col_spans)
    n_regions = n_row_regions * n_col_regions
    region_keys: list[set[int]] = [set() for _ in range(n_regions)]

    start: tuple[int, int] | None = None
    start_region = -1
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            rid = _region_id(i, j, n_col_regions)
            sub = g[r0 + 1 : r1, c0 + 1 : c1]
            assert sub.size > 0

            keys = np.unique(sub[(sub != BLACK) & (sub != MAROON)])
            region_keys[rid] = {int(color) for color in keys}

            maroons = np.argwhere(sub == MAROON)
            if maroons.size:
                assert maroons.shape[0] == 1
                assert start is None
                rr, cc = map(int, maroons[0])
                start = (r0 + 1 + rr, c0 + 1 + cc)
                start_region = rid

    assert start is not None and start_region >= 0

    locks: list[Lock] = []
    for i in range(n_row_regions - 1):
        wall_r = row_spans[i][1]
        for j, (c0, c1) in enumerate(col_spans):
            top = _region_id(i, j, n_col_regions)
            bottom = _region_id(i + 1, j, n_col_regions)
            for c in range(c0 + 1, c1):
                color = int(g[wall_r, c])
                if color not in (BLACK, GREY):
                    locks.append(Lock(wall_r, c, color, top, bottom))

    for i, (r0, r1) in enumerate(row_spans):
        for j in range(n_col_regions - 1):
            wall_c = col_spans[j][1]
            left = _region_id(i, j, n_col_regions)
            right = _region_id(i, j + 1, n_col_regions)
            for r in range(r0 + 1, r1):
                color = int(g[r, wall_c])
                if color not in (BLACK, GREY):
                    locks.append(Lock(r, wall_c, color, left, right))

    return Puzzle(row_spans, col_spans, region_keys, start, start_region, locks)


def _opened_locks(puzzle: Puzzle) -> list[bool]:
    opened: list[bool] = []
    for lock in puzzle.locks:
        keys_a = puzzle.region_keys[lock.region_a]
        keys_b = puzzle.region_keys[lock.region_b]
        opened.append(lock.color in keys_a or lock.color in keys_b)
    return opened


def _flood_mask(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    passable = (grid == BLACK) | (grid == MAROON)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _reachable_regions(puzzle: Puzzle, opened: list[bool]) -> set[int]:
    adj = [[] for _ in range(len(puzzle.region_keys))]
    for lock, is_open in zip(puzzle.locks, opened):
        if not is_open:
            continue
        adj[lock.region_a].append(lock.region_b)
        adj[lock.region_b].append(lock.region_a)

    reachable = {puzzle.start_region}
    q: deque[int] = deque([puzzle.start_region])
    while q:
        rid = q.popleft()
        for nbr in adj[rid]:
            if nbr not in reachable:
                reachable.add(nbr)
                q.append(nbr)
    return reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-matched wall locks, then flood maroon from the unique start region.

    Input structure:
    - Black rectangular regions are separated by full grey wall rows/columns.
    - Non-grey pixels on internal wall segments are locks.
    - Non-black, non-maroon pixels inside regions are keys.
    - Exactly one maroon pixel marks the flood-fill start.

    Rule:
    - A lock opens when either adjacent region contains a key of the same color.
    - Opened locks become black before the flood step.
    - Flood-fill from the maroon start through black cells and opened locks only.
    - Preserve keys, grey walls, and unopened locks.
    """
    g = np.asarray(grid, dtype=int)
    puzzle = _analyze_puzzle(g)

    out = g.copy()
    for lock, is_open in zip(puzzle.locks, _opened_locks(puzzle)):
        if is_open:
            out[lock.row, lock.col] = BLACK

    reached = _flood_mask(out, puzzle.start)
    out[reached] = MAROON
    return out


def _spans_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        spans.append((cursor, cursor + size + 1))
        cursor += size + 1
    return spans


def _build_base_grid(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _spans_from_sizes(heights)
    col_spans = _spans_from_sizes(widths)
    h = row_spans[-1][1] + 1
    w = col_spans[-1][1] + 1
    grid = np.full((h, w), GREY, dtype=int)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    return grid, row_spans, col_spans


def _candidate_lock_specs(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[tuple[list[tuple[int, int]], int, int]]:
    specs: list[tuple[list[tuple[int, int]], int, int]] = []
    n_cols = len(col_spans)
    for i in range(len(row_spans) - 1):
        wall_r = row_spans[i][1]
        for j, (c0, c1) in enumerate(col_spans):
            coords = [(wall_r, c) for c in range(c0 + 1, c1)]
            specs.append((coords, _region_id(i, j, n_cols), _region_id(i + 1, j, n_cols)))
    for i, (r0, r1) in enumerate(row_spans):
        for j in range(len(col_spans) - 1):
            wall_c = col_spans[j][1]
            coords = [(r, wall_c) for r in range(r0 + 1, r1)]
            specs.append((coords, _region_id(i, j, n_cols), _region_id(i, j + 1, n_cols)))
    return specs


def _sample_region_keys(rng: random.Random, n_regions: int, color_pool: list[int]) -> list[set[int]]:
    region_keys: list[set[int]] = []
    for _ in range(n_regions):
        count = rng.choices((0, 1, 2), weights=(3, 5, 2))[0]
        count = min(count, len(color_pool))
        region_keys.append(set(rng.sample(color_pool, count)))
    return region_keys


def _lock_color_for_edge(
    rng: random.Random,
    color_pool: list[int],
    keys_a: set[int],
    keys_b: set[int],
) -> int:
    want_open = rng.random() < 0.5
    open_colors = sorted(keys_a | keys_b)
    closed_colors = [color for color in color_pool if color not in open_colors]

    if want_open and open_colors:
        return rng.choice(open_colors)
    if (not want_open) and closed_colors:
        return rng.choice(closed_colors)
    return rng.choice(color_pool)


def _place_region_markers(
    grid: np.ndarray,
    rng: random.Random,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    start_region: int,
    region_keys: list[set[int]],
) -> None:
    n_cols = len(col_spans)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            rid = _region_id(i, j, n_cols)
            cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            rng.shuffle(cells)
            cursor = 0
            if rid == start_region:
                sr, sc = cells[cursor]
                grid[sr, sc] = MAROON
                cursor += 1
            for color in sorted(region_keys[rid]):
                kr, kc = cells[cursor]
                grid[kr, kc] = color
                cursor += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_row_regions = rng.choice(REGION_COUNT_CHOICES)
    n_col_regions = rng.choice(REGION_COUNT_CHOICES)
    heights = [rng.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(n_row_regions)]
    widths = [rng.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(n_col_regions)]
    grid, row_spans, col_spans = _build_base_grid(heights, widths)

    n_regions = n_row_regions * n_col_regions
    start_region = rng.randrange(n_regions)

    palette_size = rng.randint(2, 4)
    color_pool = rng.sample(list(LOCK_COLORS), palette_size)
    region_keys = _sample_region_keys(rng, n_regions, color_pool)
    if sum(len(keys) for keys in region_keys) < 2:
        return None

    lock_specs = _candidate_lock_specs(row_spans, col_spans)
    rng.shuffle(lock_specs)
    locks: list[Lock] = []
    for coords, region_a, region_b in lock_specs:
        if rng.random() >= LOCK_PRESENT_PROB:
            continue
        color = _lock_color_for_edge(rng, color_pool, region_keys[region_a], region_keys[region_b])
        row, col = rng.choice(coords)
        locks.append(Lock(row, col, color, region_a, region_b))

    if len(locks) < 3:
        return None

    if rng.random() < 0.7:
        keyed_regions = [rid for rid, keys in enumerate(region_keys) if keys]
        rng.shuffle(keyed_regions)
        for rid in keyed_regions:
            incident = [idx for idx, lock in enumerate(locks) if lock.region_a == rid or lock.region_b == rid]
            if len(incident) < 2:
                continue
            color = rng.choice(sorted(region_keys[rid]))
            for idx in rng.sample(incident, 2):
                lock = locks[idx]
                locks[idx] = Lock(lock.row, lock.col, color, lock.region_a, lock.region_b)
            break

    for lock in locks:
        grid[lock.row, lock.col] = lock.color

    _place_region_markers(grid, rng, row_spans, col_spans, start_region, region_keys)

    try:
        puzzle = _analyze_puzzle(grid)
    except AssertionError:
        return None

    opened = _opened_locks(puzzle)
    reachable = _reachable_regions(puzzle, opened)
    if sum(opened) == 0 or sum(opened) == len(opened):
        return None
    if len(reachable) <= 1 or len(reachable) >= n_regions:
        return None
    if not any(len(keys) == 0 for keys in puzzle.region_keys):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == MAROON) <= np.count_nonzero(grid == MAROON):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
