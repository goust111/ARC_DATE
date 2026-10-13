from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
PATH_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 900
PATH_RETRIES = 500


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        nr, nc = int(r) + int(dr), int(c) + int(dc)
        if _in_bounds(nr, nc, h, w):
            yield nr, nc


def _same_color_components(g: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color in (BG, YELLOW) or seen[sr, sc]:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                cells.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append((color, cells))

    return comps


def _trace_from_yellow_marker(
    g: np.ndarray, cells: list[tuple[int, int]], color: int
) -> list[tuple[int, int]]:
    h, w = g.shape
    cell_set = set(cells)
    assert len(cell_set) == len(cells), "path component must not repeat cells"

    def same_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = cell
        return [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in cell_set]

    degrees = {cell: len(same_neighbors(cell)) for cell in cell_set}
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    starts: list[tuple[int, int]] = []
    for r, c in endpoints:
        if any(int(g[nr, nc]) == YELLOW for nr, nc in _neighbors4(r, c, h, w)):
            starts.append((int(r), int(c)))
    assert len(starts) == 1, "exactly one path endpoint must touch a yellow marker"

    ordered: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = starts[0]
    while True:
        ordered.append(cur)
        next_cells = [cell for cell in same_neighbors(cur) if cell != prev]
        if prev is None:
            assert len(next_cells) == 1, "start endpoint must have one path neighbor"
        elif not next_cells:
            break
        else:
            assert len(next_cells) == 1, "path must have a unique continuation"

        prev, cur = cur, next_cells[0]
        assert len(ordered) <= len(cell_set), "path trace must not cycle"

    assert len(ordered) == len(cell_set), "yellow marker must identify the whole path"
    assert all(int(g[r, c]) == int(color) for r, c in ordered)
    return ordered


def _turn_indices(cells: list[tuple[int, int]]) -> list[int]:
    turns: list[int] = []
    for i in range(1, len(cells) - 1):
        pr, pc = cells[i - 1]
        r, c = cells[i]
        nr, nc = cells[i + 1]
        before = (int(r - pr), int(c - pc))
        after = (int(nr - r), int(nc - c))
        if before != after:
            turns.append(i)
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color outside branches at each turn of every yellow-marked path.

    Input structure:
    - The background is black.
    - One or more single-pixel-wide orthogonal colored paths are drawn in
      distinct non-yellow colors, and paths do not touch or cross.
    - Each path has exactly two endpoints, with one endpoint 4-adjacent to a
      single yellow marker pixel that identifies the walking start.

    Rule:
    - For each path, start from the endpoint next to its yellow marker and walk
      through the path one pixel at a time, counting the start as step 1.
    - Whenever the walk reaches a 90-degree turn pixel, continue the incoming
      direction as the outside direction of the bend.
    - From that turn pixel, draw a branch in the outside direction whose length
      equals the cumulative step count at the turn.
    - Branches use the path color, stop at the grid boundary, and do not overwrite
      any non-black pixel. Yellow markers are kept in the output.

    Key insight:
    - The yellow marker fixes which endpoint is step 1, so the order of turns
      determines both the outside branch direction and its length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    for color, comp in _same_color_components(g):
        path = _trace_from_yellow_marker(g, comp, int(color))
        for i in _turn_indices(path):
            pr, pc = path[i - 1]
            r, c = path[i]
            dr, dc = int(r - pr), int(c - pc)
            rr, cc = int(r), int(c)
            for _ in range(i + 1):
                rr += dr
                cc += dc
                if not _in_bounds(rr, cc, h, w):
                    break
                if int(out[rr, cc]) != BG:
                    break
                out[rr, cc] = int(color)

    return out


def _path_degrees(cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cell_set = set(cells)
    return {
        cell: sum((cell[0] + dr, cell[1] + dc) in cell_set for dr, dc in DIRS)
        for cell in cell_set
    }


def _is_simple_ordered_path(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(cells) < 5 or len(set(cells)) != len(cells):
        return False
    if any(not _in_bounds(r, c, h, w) for r, c in cells):
        return False

    degrees = _path_degrees(cells)
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2 or cells[0] not in endpoints or cells[-1] not in endpoints:
        return False
    return all(degree in (1, 2) for degree in degrees.values())


def _branch_cells_for_path(
    cells: list[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]]:
    branch_cells: list[tuple[int, int]] = []
    for i in _turn_indices(cells):
        pr, pc = cells[i - 1]
        r, c = cells[i]
        dr, dc = int(r - pr), int(c - pc)
        rr, cc = int(r), int(c)
        before_count = len(branch_cells)
        for _ in range(i + 1):
            rr += dr
            cc += dc
            if not _in_bounds(rr, cc, h, w):
                break
            branch_cells.append((int(rr), int(cc)))
        if len(branch_cells) == before_count:
            return []
    return branch_cells


def _perpendicular_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(int(dc), int(-dr)), (int(-dc), int(dr))]


def _build_random_path(h: int, w: int) -> list[tuple[int, int]] | None:
    turn_count = random.randint(1, 4)
    directions = [random.choice(DIRS)]
    for _ in range(turn_count):
        directions.append(random.choice(_perpendicular_dirs(directions[-1])))

    lengths = [random.randint(2, 6) for _ in directions]
    start = (random.randrange(h), random.randrange(w))
    cells = [start]
    r, c = start
    for (dr, dc), length in zip(directions, lengths):
        for _ in range(length):
            r += int(dr)
            c += int(dc)
            cells.append((int(r), int(c)))

    if not _is_simple_ordered_path(cells, h, w):
        return None
    if not _turn_indices(cells):
        return None
    return cells


def _near_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _marker_candidates(
    cells: list[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    start = cells[0]
    first = cells[1]
    first_dir = (int(first[0] - start[0]), int(first[1] - start[1]))
    dirs = [(-first_dir[0], -first_dir[1])] + _perpendicular_dirs(first_dir)
    random.shuffle(dirs[1:])

    path_set = set(cells)
    candidates: list[tuple[int, int]] = []
    for dr, dc in dirs:
        mr, mc = int(start[0] + dr), int(start[1] + dc)
        marker = (mr, mc)
        if not _in_bounds(mr, mc, h, w) or marker in forbidden or marker in path_set:
            continue
        path_neighbors = sum((nr, nc) in path_set for nr, nc in _neighbors4(mr, mc, h, w))
        if path_neighbors == 1:
            candidates.append(marker)
    return candidates


def _candidate_path(
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]] | None:
    for _ in range(PATH_RETRIES):
        cells = _build_random_path(h, w)
        if cells is None:
            continue

        path_set = set(cells)
        branches = _branch_cells_for_path(cells, h, w)
        branch_set = set(branches)
        if not branches or len(branch_set) != len(branches):
            continue
        if branch_set & path_set:
            continue
        if (path_set | branch_set) & occupied_output:
            continue
        if _near_existing(path_set, occupied_input):
            continue

        forbidden = occupied_output | branch_set
        markers = _marker_candidates(cells, h, w, forbidden)
        random.shuffle(markers)
        for marker in markers:
            candidate_input = path_set | {marker}
            if candidate_input & occupied_output:
                continue
            if _near_existing(candidate_input, occupied_input):
                continue
            return cells, marker, branches

    return None


def _render_expected(
    grid: np.ndarray, branches_by_color: list[tuple[int, list[tuple[int, int]]]]
) -> np.ndarray:
    out = grid.copy()
    for color, cells in branches_by_color:
        for r, c in cells:
            if int(out[r, c]) == BG:
                out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    path_count = random.randint(1, 3)
    colors = random.sample(PATH_COLORS, path_count)

    grid = np.full((h, w), BG, dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    branches_by_color: list[tuple[int, list[tuple[int, int]]]] = []

    for color in colors:
        candidate = _candidate_path(h, w, occupied_input, occupied_output)
        if candidate is None:
            return None

        cells, marker, branches = candidate
        for r, c in cells:
            grid[int(r), int(c)] = int(color)
        grid[int(marker[0]), int(marker[1])] = YELLOW

        path_set = set(cells)
        branch_set = set(branches)
        occupied_input |= path_set | {marker}
        occupied_output |= path_set | branch_set | {marker}
        branches_by_color.append((int(color), branches))

    expected = _render_expected(grid, branches_by_color)
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    if np.array_equal(predicted, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid yellow-started path input")
