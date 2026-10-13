from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _foreground_component(grid: np.ndarray, bg: int) -> list[Coord]:
    cells = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != bg)
    }
    assert cells, "expected non-background path"

    start = next(iter(cells))
    stack = [start]
    seen = {start}
    order = []
    while stack:
        cur = stack.pop()
        order.append(cur)
        for nxt in _neighbors4(*cur):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    assert seen == cells, "expected exactly one foreground component"
    return order


def _ordered_path_from_cells(cells: set[Coord]) -> list[Coord]:
    degrees = {
        cell: sum(nbr in cells for nbr in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    path = []
    prev = None
    cur = endpoints[0]
    while True:
        path.append(cur)
        next_cells = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cells), "path traversal must visit every cell"
    return path


def _ordered_path(grid: np.ndarray, bg: int) -> list[Coord]:
    cells = set(_foreground_component(grid, bg))
    return _ordered_path_from_cells(cells)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    path = _ordered_path(g, bg)
    colors = [int(g[r, c]) for r, c in path]

    out = g.copy()
    for (r, c), color in zip(path, reversed(colors)):
        out[r, c] = color
    return out


def _turn_choices(direction: Coord | None) -> list[Coord]:
    if direction is None:
        dirs = DIRS4[:]
    else:
        dr, dc = direction
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _sample_run_lengths(total_steps: int, segments: int) -> list[int]:
    runs = []
    remaining = total_steps
    for idx in range(segments):
        slots_left = segments - idx - 1
        lo = max(1, remaining - 5 * slots_left)
        hi = min(5, remaining - slots_left)
        if lo > hi:
            return []
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan(height: int, width: int) -> list[int]:
    max_len = min((height * width) // 3, 30)
    min_len = 7
    if max_len < min_len:
        max_len = min_len
    target_len = random.randint(min_len, max_len)
    total_steps = target_len - 1

    max_segments = min(8, total_steps)
    valid_segments = [
        seg
        for seg in range(2, max_segments + 1)
        if seg <= total_steps <= 5 * seg
    ]
    segments = random.choice(valid_segments)
    return _sample_run_lengths(total_steps, segments)


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    height: int,
    width: int,
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_path(height: int, width: int, run_lengths: list[int]) -> list[Coord] | None:
    starts = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(starts)

    def search(
        seg_idx: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if seg_idx == len(run_lengths):
            return path
        for direction in _turn_choices(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_idx], height, width)
            if extended is None:
                continue
            found = search(seg_idx + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        built = search(0, None, [start], {start})
        if built is not None:
            return built
    return None


def _sample_colors(length: int, bg: int) -> list[int]:
    palette = [c for c in range(10) if c != bg]
    palette_size = random.randint(2, min(6, len(palette)))
    chosen = random.sample(palette, palette_size)
    colors = [random.choice(chosen) for _ in range(length)]

    if len(set(colors)) == 1:
        alt = random.choice([c for c in chosen if c != colors[0]])
        colors[random.randrange(length)] = alt
    if colors == colors[::-1]:
        for i in range(1, length):
            if colors[i] != colors[0]:
                colors[0], colors[i] = colors[i], colors[0]
                break
    return colors


def _oracle_valid(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        path = _ordered_path(grid, bg)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    expected = grid.copy()
    values = [int(grid[r, c]) for r, c in path]
    for (r, c), color in zip(path, reversed(values)):
        expected[r, c] = color
    return bool(np.array_equal(out, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(8, 16)
    width = random.randint(8, 16)
    bg = random.randint(0, 9)

    runs = _sample_run_plan(height, width)
    path = _build_path(height, width, runs)
    if path is None:
        return None

    area = height * width
    if len(path) * 2 >= area:
        return None

    colors = _sample_colors(len(path), bg)
    grid = np.full((height, width), bg, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    counts = Counter(int(v) for v in grid.flat)
    if counts[bg] <= len(path):
        return None
    if not _oracle_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
