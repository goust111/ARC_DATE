import random
from typing import Callable

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_PATCH = {
    (-1, 0): (0, 1),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
    (0, 1): (1, 2),
}
GEN_RETRIES = 300


def _detect_grid_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            color = int(grid[r, 0])
            row_counts[color] = row_counts.get(color, 0) + 1
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            color = int(grid[0, c])
            col_counts[color] = col_counts.get(color, 0) + 1
    candidates = set(row_counts) & set(col_counts)
    assert candidates, "expected uniform grid-line rows and columns"
    return max(candidates, key=lambda color: row_counts[color] + col_counts[color])


def _board_size(grid: np.ndarray, grid_color: int) -> int:
    h, w = grid.shape
    assert h == w and (h - 1) % 4 == 0, "expected square board with 3x3 cells"
    rows = [r for r in range(h) if np.all(grid[r, :] == grid_color)]
    cols = [c for c in range(w) if np.all(grid[:, c] == grid_color)]
    expected = list(range(0, h, 4))
    assert rows == expected and cols == expected, "expected evenly spaced 1-pixel grid lines"
    return (h - 1) // 4


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    return 4 * cell[0] + 1, 4 * cell[1] + 1


def _patch(grid: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    r0, c0 = _cell_origin(cell)
    return grid[r0 : r0 + 3, c0 : c0 + 3]


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, set[int], dict[tuple[int, int], tuple[int, tuple[int, int]]], set[tuple[int, int]]]:
    grid_color = _detect_grid_color(grid)
    n = _board_size(grid, grid_color)
    source_colors: set[int] = set()
    pointers: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    empty_cells: set[tuple[int, int]] = set()

    for br in range(n):
        for bc in range(n):
            cell = (br, bc)
            p = _patch(grid, cell)
            non_black = np.argwhere(p != BLACK)
            if non_black.size == 0:
                empty_cells.add(cell)
                continue
            coords = [tuple(map(int, rc)) for rc in non_black]
            colors = {int(p[r, c]) for r, c in coords}
            assert grid_color not in colors, "grid line color may not appear inside a cell"
            if len(coords) == 1 and coords[0] == (1, 1):
                source_colors.add(int(p[1, 1]))
                continue
            assert len(coords) == 2 and (1, 1) in coords and len(colors) == 1, "invalid pointer/source encoding"
            other = coords[0] if coords[1] == (1, 1) else coords[1]
            direction = (other[0] - 1, other[1] - 1)
            assert direction in DIR_TO_PATCH, "pointer must be cardinal"
            pointers[cell] = (int(p[1, 1]), direction)

    assert source_colors, "expected at least one source"
    assert pointers, "expected at least one pointer"
    return n, source_colors, pointers, empty_cells


def _fill_pointer_cell(out: np.ndarray, inp: np.ndarray, cell: tuple[int, int], fill_color: int) -> None:
    r0, c0 = _cell_origin(cell)
    out_patch = out[r0 : r0 + 3, c0 : c0 + 3]
    in_patch = inp[r0 : r0 + 3, c0 : c0 + 3]
    out_patch[in_patch == BLACK] = fill_color


def _fill_empty_cell(out: np.ndarray, cell: tuple[int, int], fill_color: int) -> None:
    r0, c0 = _cell_origin(cell)
    out[r0 : r0 + 3, c0 : c0 + 3] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, source_colors, pointers, empty_cells = _parse_scene(g)

    out = g.copy()
    painted: set[tuple[int, int]] = set()

    for source_color in sorted(source_colors):
        starters = sorted(cell for cell, (color, _direction) in pointers.items() if color == source_color)
        for starter in starters:
            current = starter
            while True:
                _color, direction = pointers[current]
                nxt = (current[0] + direction[0], current[1] + direction[1])
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
                    break
                if nxt in painted:
                    break
                if nxt in pointers:
                    _fill_pointer_cell(out, g, nxt, source_color)
                    painted.add(nxt)
                    current = nxt
                    continue
                if nxt in empty_cells:
                    _fill_empty_cell(out, nxt, source_color)
                    painted.add(nxt)
                break

    return out


def _neighbors(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS:
        nr, nc = cell[0] + dr, cell[1] + dc
        if 0 <= nr < n and 0 <= nc < n:
            out.append((nr, nc))
    return out


def _on_boundary(cell: tuple[int, int], n: int) -> bool:
    return cell[0] == 0 or cell[1] == 0 or cell[0] == n - 1 or cell[1] == n - 1


def _dist_to_boundary(cell: tuple[int, int], n: int) -> int:
    return min(cell[0], cell[1], n - 1 - cell[0], n - 1 - cell[1])


def _grow_walk(
    start: tuple[int, int],
    edges: int,
    blocked: set[tuple[int, int]],
    n: int,
    rng: random.Random,
    final_ok: Callable[[list[tuple[int, int]]], bool],
) -> list[tuple[int, int]] | None:
    path = [start]
    used = set(blocked)
    used.add(start)

    def dfs(remaining: int) -> bool:
        if remaining == 0:
            return final_ok(path)
        choices = [nb for nb in _neighbors(path[-1], n) if nb not in used]
        rng.shuffle(choices)
        for nb in choices:
            used.add(nb)
            path.append(nb)
            if dfs(remaining - 1):
                return True
            path.pop()
            used.remove(nb)
        return False

    if dfs(edges):
        return path.copy()
    return None


def _loop_targets(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    last = path[-1]
    return [cell for cell in path[1:-1] if abs(cell[0] - last[0]) + abs(cell[1] - last[1]) == 1]


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr, dc) in DIR_TO_PATCH, "cells must be cardinal neighbors"
    return dr, dc


def _free_cells(n: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]


def _build_empty_chain(
    n: int, occupied: set[tuple[int, int]], rng: random.Random
) -> dict[str, object] | None:
    free = _free_cells(n, occupied)
    if len(free) < 2:
        return None
    max_pointers = 2 if n <= 4 else 4
    pointer_count = rng.randint(0, max_pointers)
    edges = pointer_count + 1
    candidates = free[:]
    rng.shuffle(candidates)
    for start in candidates[:80]:
        path = _grow_walk(start, edges, occupied, n, rng, lambda _path: True)
        if path is None:
            continue
        return {
            "branch": "empty",
            "starter": path[0],
            "downstream": path[1:-1],
            "terminal": path[-1],
        }
    return None


def _build_boundary_chain(
    n: int, occupied: set[tuple[int, int]], rng: random.Random
) -> dict[str, object] | None:
    free = _free_cells(n, occupied)
    if len(free) < 2:
        return None
    max_pointers = 2 if n <= 4 else 4
    pointer_count = rng.randint(1, max_pointers)
    candidates = [cell for cell in free if _dist_to_boundary(cell, n) <= pointer_count]
    rng.shuffle(candidates)
    for start in candidates[:100]:
        path = _grow_walk(start, pointer_count, occupied, n, rng, lambda walk: _on_boundary(walk[-1], n))
        if path is None:
            continue
        end = path[-1]
        outward = [d for d in DIRS if not (0 <= end[0] + d[0] < n and 0 <= end[1] + d[1] < n)]
        if not outward:
            continue
        return {
            "branch": "boundary",
            "starter": path[0],
            "downstream": path[1:],
            "end_dir": rng.choice(outward),
        }
    return None


def _build_loop_chain(
    n: int, occupied: set[tuple[int, int]], rng: random.Random
) -> dict[str, object] | None:
    free = _free_cells(n, occupied)
    if len(free) < 3:
        return None
    max_pointers = 3 if n <= 4 else 5
    pointer_count = rng.randint(2, max_pointers)
    candidates = free[:]
    rng.shuffle(candidates)
    for start in candidates[:120]:
        path = _grow_walk(start, pointer_count, occupied, n, rng, lambda walk: bool(_loop_targets(walk)))
        if path is None:
            continue
        targets = _loop_targets(path)
        if not targets:
            continue
        return {
            "branch": "loop",
            "starter": path[0],
            "downstream": path[1:],
            "loop_target": rng.choice(targets),
        }
    return None


def _build_chain(branch: str, n: int, occupied: set[tuple[int, int]], rng: random.Random) -> dict[str, object] | None:
    if branch == "empty":
        return _build_empty_chain(n, occupied, rng)
    if branch == "boundary":
        return _build_boundary_chain(n, occupied, rng)
    assert branch == "loop"
    return _build_loop_chain(n, occupied, rng)


def _blank_board(n: int) -> np.ndarray:
    size = 4 * n + 1
    grid = np.full((size, size), BLACK, dtype=int)
    grid[::4, :] = GREY
    grid[:, ::4] = GREY
    return grid


def _draw_source(grid: np.ndarray, cell: tuple[int, int], color: int) -> None:
    r0, c0 = _cell_origin(cell)
    grid[r0 + 1, c0 + 1] = color


def _draw_pointer(grid: np.ndarray, cell: tuple[int, int], color: int, direction: tuple[int, int]) -> None:
    r0, c0 = _cell_origin(cell)
    grid[r0 + 1, c0 + 1] = color
    pr, pc = DIR_TO_PATCH[direction]
    grid[r0 + pr, c0 + pc] = color


def _render_scene(
    n: int,
    source_cells: list[tuple[int, int]],
    source_colors: list[int],
    chains: list[dict[str, object]],
    rng: random.Random,
) -> np.ndarray:
    grid = _blank_board(n)
    source_color_by_cell = {cell: color for cell, color in zip(source_cells, source_colors)}
    source_palette = set(source_colors)
    relay_palette = [color for color in COLOR_POOL if color not in source_palette]

    for cell, color in source_color_by_cell.items():
        _draw_source(grid, cell, color)

    for chain in chains:
        source_color = int(chain["source_color"])
        starter = chain["starter"]
        downstream = list(chain["downstream"])

        if chain["branch"] == "empty":
            terminal = chain["terminal"]
            first_target = terminal if not downstream else downstream[0]
            _draw_pointer(grid, starter, source_color, _direction(starter, first_target))
            for i, cell in enumerate(downstream):
                nxt = terminal if i == len(downstream) - 1 else downstream[i + 1]
                relay_color = rng.choice(relay_palette)
                _draw_pointer(grid, cell, relay_color, _direction(cell, nxt))
        elif chain["branch"] == "boundary":
            first_target = downstream[0]
            _draw_pointer(grid, starter, source_color, _direction(starter, first_target))
            for i, cell in enumerate(downstream):
                relay_color = rng.choice(relay_palette)
                if i == len(downstream) - 1:
                    direction = chain["end_dir"]
                else:
                    direction = _direction(cell, downstream[i + 1])
                _draw_pointer(grid, cell, relay_color, direction)
        else:
            first_target = downstream[0]
            loop_target = chain["loop_target"]
            _draw_pointer(grid, starter, source_color, _direction(starter, first_target))
            for i, cell in enumerate(downstream):
                relay_color = rng.choice(relay_palette)
                if i == len(downstream) - 1:
                    direction = _direction(cell, loop_target)
                else:
                    direction = _direction(cell, downstream[i + 1])
                _draw_pointer(grid, cell, relay_color, direction)

    occupied = set(source_cells)
    for chain in chains:
        occupied.add(chain["starter"])
        occupied.update(chain["downstream"])
        if chain["branch"] == "empty":
            occupied.add(chain["terminal"])

    free = _free_cells(n, occupied)
    rng.shuffle(free)
    distractor_count = min(len(free), rng.randint(0, max(1, n // 2)))
    for cell in free[:distractor_count]:
        _draw_pointer(grid, cell, rng.choice(relay_palette), rng.choice(DIRS))

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(4, 7)
    area = n * n
    max_chains = min(6, max(2, area // 10 + 2))
    num_sources = rng.randint(1, min(3, max_chains))
    min_chains = min(max_chains, max(2, num_sources))
    num_chains = rng.randint(min_chains, max_chains)

    branches = ["empty", "boundary", "loop"]
    branch_plan = branches[:]
    while len(branch_plan) < num_chains:
        branch_plan.append(rng.choice(branches))
    rng.shuffle(branch_plan)

    source_colors = rng.sample(COLOR_POOL, num_sources)

    chains_per_source = [1] * num_sources
    for _ in range(num_chains - num_sources):
        chains_per_source[rng.randrange(num_sources)] += 1
    source_indices: list[int] = []
    for idx, count in enumerate(chains_per_source):
        source_indices.extend([idx] * count)
    rng.shuffle(source_indices)

    all_cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(all_cells)
    source_cells = all_cells[:num_sources]
    occupied = set(source_cells)

    chains: list[dict[str, object]] = []
    for branch, source_idx in zip(branch_plan, source_indices):
        chain = _build_chain(branch, n, occupied, rng)
        if chain is None:
            return None
        chain["source_color"] = source_colors[source_idx]
        chains.append(chain)
        occupied.add(chain["starter"])
        occupied.update(chain["downstream"])
        if branch == "empty":
            occupied.add(chain["terminal"])

    grid = _render_scene(n, source_cells, source_colors, chains, rng)
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
