from __future__ import annotations

import random

import numpy as np


BLACK = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATE_ATTEMPTS = 400

_GENERATION_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _seed_list(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seeds = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != BLACK) & (grid != WALL))
    ]
    colors = [color for _r, _c, color in seeds]
    assert len(colors) == len(set(colors)), "seed colors must be unique"
    seeds.sort(key=lambda item: (item[0], item[1]))
    return seeds


def _trace_to_wall(
    grid: np.ndarray, start: tuple[int, int], dr: int, dc: int
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == WALL:
            return (r, c)
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Seeds cast invisible orthogonal rays and color the first wall pixel they hit.

    If several seeds hit the same wall pixel, the seed with smaller (row, col)
    priority determines the final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    best_hits: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    for r, c, color in _seed_list(g):
        priority = (r, c)
        for dr, dc in DIRS:
            hit = _trace_to_wall(g, (r, c), dr, dc)
            if hit is None:
                continue
            prev = best_hits.get(hit)
            if prev is None or priority < prev[0]:
                best_hits[hit] = (priority, color)

    for (r, c), (_priority, color) in best_hits.items():
        out[r, c] = color
    return out


def _line_cells(
    orientation: str, start: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r0, c0 = start
    if orientation == "v":
        return [(r0 + k, c0) for k in range(length)]
    return [(r0, c0 + k) for k in range(length)]


def _between_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    out = []
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac + step, bc, step):
            out.append((ar, c))
    else:
        assert ac == bc
        step = 1 if br > ar else -1
        for r in range(ar + step, br, step):
            out.append((r, ac))
    return out


def _path_clear(
    start: tuple[int, int],
    hit: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    for cell in _between_cells(start, hit):
        if cell in wall_cells or cell in seed_cells or cell in protected_empty:
            return False
    return True


def _can_place_wall(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    candidate = set(cells)
    if len(candidate) != len(cells):
        return False
    for r, c in candidate:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in wall_cells or (r, c) in seed_cells or (r, c) in protected_empty:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in wall_cells and (nr, nc) not in candidate:
                return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in seed_cells:
                return False
    return True


def _can_place_seed(
    shape: tuple[int, int],
    cell: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c):
        return False
    if cell in wall_cells or cell in seed_cells or cell in protected_empty:
        return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in wall_cells:
            return False
    for nr, nc in _neighbors8(r, c):
        if (nr, nc) in seed_cells:
            return False
    return True


def _place_conflict_feature(
    shape: tuple[int, int],
    orientation: str,
    colors: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(160):
        if orientation == "v":
            if h < 5 or w < 8:
                return False
            length = random.randint(3, min(7, h - 2))
            r0 = random.randint(1, h - length - 1)
            c = random.randint(3, w - 4)
            cells = _line_cells("v", (r0, c), length)
            if not _can_place_wall(shape, cells, wall_cells, seed_cells, protected_empty):
                continue
            hit = (random.randint(r0, r0 + length - 1), c)
            left = (hit[0], random.randint(0, c - 2))
            right = (hit[0], random.randint(c + 2, w - 1))
            if not _can_place_seed(shape, left, wall_cells, seed_cells, protected_empty):
                continue
            if not _can_place_seed(shape, right, wall_cells, seed_cells | {left}, protected_empty):
                continue
            if not _path_clear(left, hit, wall_cells, seed_cells, protected_empty):
                continue
            if not _path_clear(right, hit, wall_cells, seed_cells | {left}, protected_empty):
                continue
            wall_cells.update(cells)
            seed_cells.update({left, right})
            seeds.append((left[0], left[1], colors[0]))
            seeds.append((right[0], right[1], colors[1]))
            protected_empty.update(_between_cells(left, hit))
            protected_empty.update(_between_cells(right, hit))
            return True

        if h < 8 or w < 5:
            return False
        length = random.randint(3, min(7, w - 2))
        r = random.randint(3, h - 4)
        c0 = random.randint(1, w - length - 1)
        cells = _line_cells("h", (r, c0), length)
        if not _can_place_wall(shape, cells, wall_cells, seed_cells, protected_empty):
            continue
        hit = (r, random.randint(c0, c0 + length - 1))
        top = (random.randint(0, r - 2), hit[1])
        bottom = (random.randint(r + 2, h - 1), hit[1])
        if not _can_place_seed(shape, top, wall_cells, seed_cells, protected_empty):
            continue
        if not _can_place_seed(shape, bottom, wall_cells, seed_cells | {top}, protected_empty):
            continue
        if not _path_clear(top, hit, wall_cells, seed_cells, protected_empty):
            continue
        if not _path_clear(bottom, hit, wall_cells, seed_cells | {top}, protected_empty):
            continue
        wall_cells.update(cells)
        seed_cells.update({top, bottom})
        seeds.append((top[0], top[1], colors[0]))
        seeds.append((bottom[0], bottom[1], colors[1]))
        protected_empty.update(_between_cells(top, hit))
        protected_empty.update(_between_cells(bottom, hit))
        return True
    return False


def _place_single_hit_feature(
    shape: tuple[int, int],
    orientation: str,
    color: int,
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    h, w = shape
    used_rows = {r for r, _c, _color in seeds}
    used_cols = {c for _r, c, _color in seeds}

    for _ in range(180):
        if orientation == "v":
            if h < 5 or w < 7:
                return False
            length = random.randint(3, min(7, h - 2))
            r0 = random.randint(1, h - length - 1)
            c = random.randint(2, w - 3)
            cells = _line_cells("v", (r0, c), length)
            if not _can_place_wall(shape, cells, wall_cells, seed_cells, protected_empty):
                continue
            hit_r = random.randint(r0 + 1, r0 + length - 2)
            if hit_r in used_rows:
                continue
            side = random.choice((-1, 1))
            if side < 0:
                seed = (hit_r, random.randint(0, c - 2))
            else:
                seed = (hit_r, random.randint(c + 2, w - 1))
            hit = (hit_r, c)
        else:
            if h < 7 or w < 5:
                return False
            length = random.randint(3, min(7, w - 2))
            r = random.randint(2, h - 3)
            c0 = random.randint(1, w - length - 1)
            cells = _line_cells("h", (r, c0), length)
            if not _can_place_wall(shape, cells, wall_cells, seed_cells, protected_empty):
                continue
            hit_c = random.randint(c0 + 1, c0 + length - 2)
            if hit_c in used_cols:
                continue
            side = random.choice((-1, 1))
            if side < 0:
                seed = (random.randint(0, r - 2), hit_c)
            else:
                seed = (random.randint(r + 2, h - 1), hit_c)
            hit = (r, hit_c)

        if not _can_place_seed(shape, seed, wall_cells, seed_cells, protected_empty):
            continue
        if not _path_clear(seed, hit, wall_cells, seed_cells, protected_empty):
            continue

        wall_cells.update(cells)
        seed_cells.add(seed)
        seeds.append((seed[0], seed[1], color))
        protected_empty.update(_between_cells(seed, hit))
        return True
    return False


def _place_extra_wall(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(120):
        orientation = random.choice(("v", "h"))
        if orientation == "v":
            if h < 4:
                return False
            length = random.randint(2, min(7, h - 2))
            r0 = random.randint(1, h - length - 1)
            c = random.randint(1, w - 2)
            cells = _line_cells("v", (r0, c), length)
        else:
            if w < 4:
                return False
            length = random.randint(2, min(7, w - 2))
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            cells = _line_cells("h", (r, c0), length)
        if _can_place_wall(shape, cells, wall_cells, seed_cells, protected_empty):
            wall_cells.update(cells)
            return True
    return False


def _place_extra_seed(
    shape: tuple[int, int],
    color: int,
    wall_cells: set[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    used_rows = {r for r, _c, _color in seeds}
    used_cols = {c for _r, c, _color in seeds}
    h, w = shape
    for _ in range(200):
        cell = (random.randint(0, h - 1), random.randint(0, w - 1))
        if cell[0] in used_rows or cell[1] in used_cols:
            continue
        if _can_place_seed(shape, cell, wall_cells, seed_cells, protected_empty):
            seed_cells.add(cell)
            seeds.append((cell[0], cell[1], color))
            return True
    return False


def _render_grid(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL
    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _wall_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(grid == WALL):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == WALL:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _hit_records(
    grid: np.ndarray,
) -> tuple[dict[tuple[int, int], list[tuple[tuple[int, int], int, tuple[int, int]]]], int]:
    hits: dict[tuple[int, int], list[tuple[tuple[int, int], int, tuple[int, int]]]] = {}
    miss_count = 0
    for r, c, color in _seed_list(grid):
        for dr, dc in DIRS:
            hit = _trace_to_wall(grid, (r, c), dr, dc)
            if hit is None:
                miss_count += 1
                continue
            hits.setdefault(hit, []).append(((r, c), color, (dr, dc)))
    return hits, miss_count


def _valid_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    seeds = _seed_list(grid)
    if len(seeds) < 3:
        return False

    components = _wall_components(grid)
    if len(components) < 2:
        return False

    for comp in components:
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        if not (len(rows) == 1 or len(cols) == 1):
            return False
        if len(rows) == 1:
            cols_sorted = sorted(cols)
            if cols_sorted != list(range(cols_sorted[0], cols_sorted[-1] + 1)):
                return False
        else:
            rows_sorted = sorted(rows)
            if rows_sorted != list(range(rows_sorted[0], rows_sorted[-1] + 1)):
                return False

    seed_cells = {(r, c) for r, c, _color in seeds}
    wall_cells = {tuple(map(int, cell)) for cell in np.argwhere(grid == WALL)}
    for r, c, _color in seeds:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in wall_cells:
                return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in seed_cells and (nr, nc) != (r, c):
                return False

    hits, miss_count = _hit_records(grid)
    if miss_count == 0:
        return False

    single_hit = 0
    directional_conflict = 0
    for entries in hits.values():
        unique_seeds = {seed for seed, _color, _direction in entries}
        unique_dirs = {direction for _seed, _color, direction in entries}
        if len(unique_seeds) == 1:
            single_hit += 1
        if len(unique_seeds) >= 2 and len(unique_dirs) >= 2:
            directional_conflict += 1

    if single_hit == 0 or directional_conflict == 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    return True


def _generate_attempt(mode: int) -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    shape = (h, w)

    colors = list(SEED_COLORS)
    random.shuffle(colors)

    wall_cells: set[tuple[int, int]] = set()
    seed_cells: set[tuple[int, int]] = set()
    protected_empty: set[tuple[int, int]] = set()
    seeds: list[tuple[int, int, int]] = []

    conflict_orientation = "v" if mode in (0, 2) else "h"
    single_orientation = "h" if mode in (0, 3) else "v"

    if not _place_conflict_feature(
        shape,
        conflict_orientation,
        (colors[0], colors[1]),
        wall_cells,
        seed_cells,
        seeds,
        protected_empty,
    ):
        return None

    if not _place_single_hit_feature(
        shape,
        single_orientation,
        colors[2],
        wall_cells,
        seed_cells,
        seeds,
        protected_empty,
    ):
        return None

    extra_wall_goal = random.randint(0, 2)
    for _ in range(extra_wall_goal):
        _place_extra_wall(shape, wall_cells, seed_cells, protected_empty)

    extra_seed_goal = random.randint(0, min(3, len(colors) - 3))
    for color in colors[3 : 3 + extra_seed_goal]:
        _place_extra_seed(shape, color, wall_cells, seed_cells, seeds, protected_empty)

    grid = _render_grid(shape, wall_cells, seeds)
    if not _valid_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATION_COUNT

    mode = _GENERATION_COUNT % 4
    _GENERATION_COUNT += 1

    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-and-wall puzzle")
