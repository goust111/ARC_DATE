from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PATH_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 10
GRID_MAX = 28
PATH_ATTEMPTS = 180
GEN_ATTEMPTS = 320
MAX_SEG_LEN = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_path(coord_set: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(*cur) if p in coord_set and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path must have a unique continuation"
        prev, cur = cur, nxts[0]
        path.append(cur)
    assert len(path) == len(coord_set), "path traversal must cover the whole component"
    return path


def _marker_cells(
    shape: tuple[int, int],
    exit_cell: tuple[int, int],
    direction: tuple[int, int],
    turns: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, turns + 1):
        rr = exit_cell[0] + dr * step
        cc = exit_cell[1] + dc * step
        if not _in_bounds(shape, rr, cc):
            break
        cells.append((rr, cc))
    return cells


def _turn_count(path: list[tuple[int, int]]) -> tuple[int, tuple[int, int]]:
    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    assert directions, "path must contain at least two cells"
    turns = sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))
    return turns, directions[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each pipe path and mark its exit with a red scale whose length equals the path's turn count.

    Input structure:
    - The background is black.
    - Several disjoint monochromatic 4-connected pipe paths appear on the grid; each path has width 1 and a unique color.
    - Each path has exactly two endpoints, with one endpoint on the grid edge and the other in the interior; isolated grey pixels are decoys and do not belong to any path.

    Rule:
    - For each non-grey foreground component, verify that it is a simple path and identify the unique edge-touching endpoint as the entry.
    - Trace the path from the entry to the interior exit and count how many times the step direction changes by 90 degrees.
    - Starting one cell beyond the exit, draw that many red cells along the continuation of the final segment, truncating the ray at the grid boundary; keep all original path pixels and grey decoys unchanged.

    Key insight:
    - The visible output length is not tied to path size but to the number of right-angle turns encountered when following the pipe from its boundary entry to its interior exit.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]

    for color in colors:
        for comp in _components(g == color):
            coord_set = set(comp)
            degrees = {
                cell: sum(nei in coord_set for nei in _neighbors4(*cell))
                for cell in coord_set
            }
            assert all(deg in (1, 2) for deg in degrees.values()), "pipe must be branch-free"
            endpoints = [cell for cell, deg in degrees.items() if deg == 1]
            assert len(endpoints) == 2, "pipe must have exactly two endpoints"
            entry_candidates = [cell for cell in endpoints if _is_border(g.shape, *cell)]
            assert len(entry_candidates) == 1, "exactly one endpoint must touch the grid edge"

            path = _trace_path(coord_set, entry_candidates[0])
            exit_cell = path[-1]
            assert exit_cell != path[0] and not _is_border(g.shape, *exit_cell), "exit must be interior"

            turns, final_dir = _turn_count(path)
            for rr, cc in _marker_cells(g.shape, exit_cell, final_dir, turns):
                assert int(g[rr, cc]) == BLACK, "red marker ray must extend only through background"
                out[rr, cc] = RED

    return out


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _boundary_starts(shape: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    starts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(w):
        starts.append(((0, c), (1, 0)))
        starts.append(((h - 1, c), (-1, 0)))
    for r in range(h):
        starts.append(((r, 0), (0, 1)))
        starts.append(((r, w - 1), (0, -1)))
    return starts


def _moat_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _valid_prefixes(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    used: set[tuple[int, int]],
    tip: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    prefixes: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    cur = tip
    seen = set(used)
    built: list[tuple[int, int]] = []
    for _ in range(MAX_SEG_LEN):
        nr = cur[0] + direction[0]
        nc = cur[1] + direction[1]
        nxt = (nr, nc)
        if not _in_bounds(shape, nr, nc) or nxt in blocked or nxt in seen:
            break
        adj = [p for p in _neighbors4(nr, nc) if p in seen]
        if any(p != cur for p in adj):
            break
        built.append(nxt)
        seen.add(nxt)
        cur = nxt
        prefixes.append((built.copy(), cur))
    random.shuffle(prefixes)
    return prefixes


def _marker_is_clear(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    path_used: set[tuple[int, int]],
    exit_cell: tuple[int, int],
    marker: list[tuple[int, int]],
) -> bool:
    seen_marker: set[tuple[int, int]] = set()
    prev = exit_cell
    for cell in marker:
        if cell in blocked or cell in path_used or cell in seen_marker:
            return False
        adj_path = [p for p in _neighbors4(*cell) if p in path_used]
        allowed_path = {exit_cell} if cell == marker[0] else set()
        if any(p not in allowed_path for p in adj_path):
            return False
        adj_marker = [p for p in _neighbors4(*cell) if p in seen_marker]
        if any(p != prev for p in adj_marker):
            return False
        seen_marker.add(cell)
        prev = cell
    return True


def _final_options(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    used: set[tuple[int, int]],
    tip: tuple[int, int],
    direction: tuple[int, int],
    turns: int,
    want_truncation: bool,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    min_len = 2 if turns == 0 else 1
    options: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for cells, end in _valid_prefixes(shape, blocked, used, tip, direction):
        if len(cells) < min_len or _is_border(shape, *end):
            continue
        marker = _marker_cells(shape, end, direction, turns)
        if turns > 0:
            if want_truncation and len(marker) >= turns:
                continue
            if not want_truncation and len(marker) < turns:
                continue
        if _marker_is_clear(shape, blocked, used | set(cells), end, marker):
            options.append((cells, marker))
    random.shuffle(options)
    return options


def _sample_turn_plan(path_count: int) -> list[tuple[int, bool]]:
    plans: list[tuple[int, bool]] = []
    if path_count == 1:
        turns = random.randint(1, 5)
        return [(turns, bool(random.getrandbits(1)))]

    if path_count >= 2 and random.random() < 0.7:
        plans.append((random.randint(1, 5), True))
        plans.append((random.randint(1, 5), False))

    while len(plans) < path_count:
        if random.random() < 0.2:
            plans.append((0, False))
        else:
            turns = random.randint(1, 5)
            plans.append((turns, bool(random.getrandbits(1))))

    random.shuffle(plans)
    return plans


def _sample_one_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    turns: int,
    want_truncation: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    starts = _boundary_starts(shape)
    for _ in range(PATH_ATTEMPTS):
        entry, direction = random.choice(starts)
        if entry in blocked:
            continue

        path = [entry]
        used = {entry}
        tip = entry
        cur_dir = direction
        success = True
        marker: list[tuple[int, int]] = []

        for seg_idx in range(turns + 1):
            if seg_idx == turns:
                finals = _final_options(shape, blocked, used, tip, cur_dir, turns, want_truncation)
                if not finals:
                    success = False
                    break
                cells, marker = random.choice(finals)
                path.extend(cells)
                used.update(cells)
                tip = path[-1]
                break

            candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
            for cells, end in _valid_prefixes(shape, blocked, used, tip, cur_dir):
                next_used = used | set(cells)
                next_dirs = _perpendiculars(cur_dir)
                random.shuffle(next_dirs)
                for next_dir in next_dirs:
                    if seg_idx == turns - 1:
                        if _final_options(shape, blocked, next_used, end, next_dir, turns, want_truncation):
                            candidates.append((cells, next_dir))
                    elif _valid_prefixes(shape, blocked, next_used, end, next_dir):
                        candidates.append((cells, next_dir))
            if not candidates:
                success = False
                break

            cells, cur_dir = random.choice(candidates)
            path.extend(cells)
            used.update(cells)
            tip = path[-1]

        if success and marker == _marker_cells(shape, tip, cur_dir, turns):
            return path, marker
    return None


def _add_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _free_cells(shape: tuple[int, int], blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    area = h * w

    max_paths = 4 if area >= 300 else 3 if area >= 180 else 2
    path_count = random.randint(1, max_paths)
    if len(PATH_COLORS) < path_count:
        return None

    plans = _sample_turn_plan(path_count)
    colors = random.sample(list(PATH_COLORS), path_count)

    grid = np.full(shape, BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()

    for color, (turns, want_truncation) in zip(colors, plans):
        sample = _sample_one_path(shape, blocked, turns, want_truncation)
        if sample is None:
            return None
        path, marker = sample
        _add_cells(grid, path, color)
        blocked |= _moat_cells(shape, set(path) | set(marker))

    free = _free_cells(shape, blocked)
    if not free:
        return None
    target_noise = random.randint(1, min(8, len(free)))
    random.shuffle(free)
    placed_noise = 0
    for cell in free:
        if cell in blocked:
            continue
        grid[cell] = GREY
        blocked |= _moat_cells(shape, {cell})
        placed_noise += 1
        if placed_noise == target_noise:
            break
    if placed_noise == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if not np.any(out == RED):
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
