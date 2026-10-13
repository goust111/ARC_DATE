from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, neighbors4

        return Color, neighbors4
    except Exception:
        arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("_arc_utils_fable_1_006_4", arc_utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Color, module.neighbors4


Color, neighbors4 = _load_arc_utils()

Coord = tuple[int, int]


WALL_COLOR = int(Color.TEAL)
CORRIDOR_COLOR = int(Color.BLACK)
SEAM_COLOR = int(Color.YELLOW)
SOURCE_COLORS = tuple(
    int(c)
    for c in (
        Color.BLUE,
        Color.RED,
        Color.GREEN,
        Color.GREY,
        Color.PINK,
        Color.ORANGE,
        Color.MAROON,
    )
)

MAZE_SIZE_MIN = 9
MAZE_SIZE_MAX = 17
MIN_SOURCE_DEPTH = 2
MAX_EQUIDISTANT_DISTANCE = 6
MIN_SEAM_CELLS = 2
MAX_SEAM_CELLS = 10
MIN_SIDE_CELLS = 4
MAX_ATTEMPTS = 400


def _walkable_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid) != WALL_COLOR


def _walkable_neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    r, c = cell
    out: list[Coord] = []
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
            out.append((int(nr), int(nc)))
    return out


def _distance_map(mask: np.ndarray, start: Coord) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _walkable_neighbors(mask, (r, c)):
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _find_sources(grid: np.ndarray) -> list[Coord]:
    colors = sorted(set(map(int, np.unique(grid))) - {CORRIDOR_COLOR, WALL_COLOR})
    assert SEAM_COLOR not in colors, "input should not already contain yellow seam cells"
    assert len(colors) == 2, "expected exactly two source colors"

    coords: list[Coord] = []
    for color in colors:
        locs = np.argwhere(grid == int(color))
        assert locs.shape[0] == 1, "each source color must appear exactly once"
        coords.append((int(locs[0, 0]), int(locs[0, 1])))
    return coords


def _equidistant_corridor_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Coord]]:
    walkable = _walkable_mask(grid)
    sources = _find_sources(grid)
    d1 = _distance_map(walkable, sources[0])
    d2 = _distance_map(walkable, sources[1])
    assert d1[sources[1]] >= 0, "sources must lie in the same corridor component"
    seam = (np.asarray(grid) == CORRIDOR_COLOR) & (d1 == d2) & (d1 >= 0)
    return seam, d1, d2, sources


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    seam, _d1, _d2, _sources = _equidistant_corridor_mask(grid)
    out = np.asarray(grid, dtype=int).copy()
    out[seam] = SEAM_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark corridor cells whose shortest corridor distance to two colored sources is exactly equal.

    Input structure:
    - Teal cells are maze walls and black cells are corridor floor.
    - Exactly two distinct non-black, non-teal single-pixel source markers lie in the same 4-connected corridor network.
    - No yellow pixels appear in the input.

    Rule:
    - Treat every non-teal cell as walkable and compute 4-neighbor shortest-path distances from each source through the corridor.
    - Recolor each black corridor cell whose two corridor distances are equal to yellow.
    - Leave both source pixels, all teal walls, and every non-equidistant corridor cell unchanged.

    Key insight:
    - The yellow seam is a geodesic equal-distance set inside the maze, so walls can bend it away from any straight geometric bisector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    return _oracle_output(g)


def _carve_perfect_maze(rows: int, cols: int) -> np.ndarray:
    grid = np.full((rows, cols), WALL_COLOR, dtype=int)
    coarse_h = (rows - 1) // 2
    coarse_w = (cols - 1) // 2

    start = (random.randrange(coarse_h), random.randrange(coarse_w))
    visited = {start}
    stack = [start]
    grid[2 * start[0] + 1, 2 * start[1] + 1] = CORRIDOR_COLOR

    while stack:
        cr, cc = stack[-1]
        options = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w and (nr, nc) not in visited:
                options.append((nr, nc, dr, dc))

        if not options:
            stack.pop()
            continue

        nr, nc, dr, dc = random.choice(options)
        visited.add((nr, nc))
        gr, gc = 2 * cr + 1, 2 * cc + 1
        grid[gr + dr, gc + dc] = CORRIDOR_COLOR
        grid[2 * nr + 1, 2 * nc + 1] = CORRIDOR_COLOR
        stack.append((nr, nc))

    return grid


def _one_path(mask: np.ndarray, start: Coord, goal: Coord) -> list[Coord] | None:
    q = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path: list[Coord] = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nxt in _walkable_neighbors(mask, cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    return None


def _loop_openings(grid: np.ndarray) -> list[tuple[Coord, Coord, Coord]]:
    openings: list[tuple[Coord, Coord, Coord]] = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            if int(grid[r, c]) != WALL_COLOR:
                continue
            if int(grid[r, c - 1]) == CORRIDOR_COLOR and int(grid[r, c + 1]) == CORRIDOR_COLOR:
                openings.append(((r, c), (r, c - 1), (r, c + 1)))
            elif int(grid[r - 1, c]) == CORRIDOR_COLOR and int(grid[r + 1, c]) == CORRIDOR_COLOR:
                openings.append(((r, c), (r - 1, c), (r + 1, c)))
    return openings


def _carve_route_loop(grid: np.ndarray, a: Coord, b: Coord) -> np.ndarray | None:
    """Add a cycle that gives the two sources a genuine alternate route."""
    mask = grid == CORRIDOR_COLOR
    source_path = _one_path(mask, a, b)
    if source_path is None:
        return None
    source_edges = {frozenset(edge) for edge in zip(source_path, source_path[1:])}

    candidates = _loop_openings(grid)
    random.shuffle(candidates)
    suitable: list[tuple[Coord, Coord, Coord]] = []
    for opening, left, right in candidates:
        cycle_path = _one_path(mask, left, right)
        if cycle_path is None:
            continue
        cycle_edges = {frozenset(edge) for edge in zip(cycle_path, cycle_path[1:])}
        if source_edges & cycle_edges:
            suitable.append((opening, left, right))
    if not suitable:
        return None

    out = grid.copy()
    out[random.choice(suitable)[0]] = CORRIDOR_COLOR
    remaining = [item for item in _loop_openings(out)]
    random.shuffle(remaining)
    for opening, _left, _right in remaining[: random.randint(0, 3)]:
        out[opening] = CORRIDOR_COLOR
    return out


def _has_alternate_route(mask: np.ndarray, a: Coord, b: Coord) -> bool:
    shortest = _one_path(mask, a, b)
    if shortest is None:
        return False
    for edge_a, edge_b in zip(shortest, shortest[1:]):
        q = deque([a])
        seen = {a}
        banned = frozenset((edge_a, edge_b))
        while q:
            cur = q.popleft()
            if cur == b:
                return True
            for nxt in _walkable_neighbors(mask, cur):
                if nxt in seen or frozenset((cur, nxt)) == banned:
                    continue
                seen.add(nxt)
                q.append(nxt)
    return False


def _branch_info(mask: np.ndarray, midpoint: Coord, start: Coord) -> dict:
    cells: list[Coord] = []
    by_depth: dict[int, list[Coord]] = defaultdict(list)
    parent = {start: midpoint}
    seen = {start}
    stack = [(start, 1)]

    while stack:
        cell, depth = stack.pop()
        cells.append(cell)
        by_depth[int(depth)].append(cell)
        for nb in _walkable_neighbors(mask, cell):
            if nb == midpoint or nb == parent[cell] or nb in seen:
                continue
            seen.add(nb)
            parent[nb] = cell
            stack.append((nb, depth + 1))

    return {"cells": cells, "by_depth": dict(by_depth)}


def _preferred_depth_cells(mask: np.ndarray, branch: dict, depth: int) -> list[Coord]:
    cells = list(branch["by_depth"][int(depth)])
    leaves = [cell for cell in cells if len(_walkable_neighbors(mask, cell)) == 1]
    return leaves or cells


def _manhattan_equal_mask(shape: tuple[int, int], a: Coord, b: Coord) -> np.ndarray:
    rows = np.abs(np.arange(shape[0], dtype=int)[:, None] - int(a[0]))
    cols = np.abs(np.arange(shape[1], dtype=int)[None, :] - int(a[1]))
    da = rows + cols

    rows = np.abs(np.arange(shape[0], dtype=int)[:, None] - int(b[0]))
    cols = np.abs(np.arange(shape[1], dtype=int)[None, :] - int(b[1]))
    db = rows + cols
    return da == db


def _pair_is_semantic(mask: np.ndarray, a: Coord, b: Coord) -> bool:
    d1 = _distance_map(mask, a)
    d2 = _distance_map(mask, b)
    if d1[b] < 2 * MIN_SOURCE_DEPTH or d1[b] % 2 != 0:
        return False
    if not np.all(d1[mask] >= 0) or not np.all(d2[mask] >= 0):
        return False

    seam = mask & (d1 == d2) & (d1 >= 0)
    seam_cells = int(np.count_nonzero(seam))
    if not (MIN_SEAM_CELLS <= seam_cells <= MAX_SEAM_CELLS):
        return False
    nearest_seam_distance = int(np.min(d1[seam]))
    if not (MIN_SOURCE_DEPTH <= nearest_seam_distance <= MAX_EQUIDISTANT_DISTANCE):
        return False

    coords = np.argwhere(seam)
    if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
        return False

    if int(np.count_nonzero(mask & (d1 < d2))) < MIN_SIDE_CELLS:
        return False
    if int(np.count_nonzero(mask & (d2 < d1))) < MIN_SIDE_CELLS:
        return False

    manhattan = _manhattan_equal_mask(mask.shape, a, b)
    if not np.any(seam & ~manhattan):
        return False

    return True


def _pick_sources(mask: np.ndarray) -> tuple[Coord, Coord] | None:
    midpoints = [tuple(map(int, coord)) for coord in np.argwhere(mask) if len(_walkable_neighbors(mask, tuple(map(int, coord)))) >= 3]
    random.shuffle(midpoints)

    for midpoint in midpoints:
        neighbors = _walkable_neighbors(mask, midpoint)
        if len(neighbors) < 3:
            continue

        branches = [_branch_info(mask, midpoint, nb) for nb in neighbors]
        branch_pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
        random.shuffle(branch_pairs)

        for i, j in branch_pairs:
            seam_size = 1 + sum(len(branches[k]["cells"]) for k in range(len(branches)) if k not in (i, j))
            if not (MIN_SEAM_CELLS <= seam_size <= MAX_SEAM_CELLS):
                continue

            common_depths = sorted(set(branches[i]["by_depth"]) & set(branches[j]["by_depth"]), reverse=True)
            common_depths = [depth for depth in common_depths if depth >= MIN_SOURCE_DEPTH]
            if not common_depths:
                continue

            for depth in common_depths:
                left_cells = _preferred_depth_cells(mask, branches[i], depth)
                right_cells = _preferred_depth_cells(mask, branches[j], depth)
                random.shuffle(left_cells)
                random.shuffle(right_cells)
                a = left_cells[0]
                b = right_cells[0]
                if _pair_is_semantic(mask, a, b):
                    return a, b

    return None


def _paint_sources(grid: np.ndarray, a: Coord, b: Coord) -> np.ndarray:
    out = grid.copy()
    color_a, color_b = random.sample(SOURCE_COLORS, 2)
    out[a] = int(color_a)
    out[b] = int(color_b)
    return out


def _semantic_validation(grid: np.ndarray) -> bool:
    try:
        sources = _find_sources(grid)
        seam, d1, d2, parsed_sources = _equidistant_corridor_mask(grid)
        assert sources == parsed_sources
        walkable = _walkable_mask(grid)
        assert np.all(d1[walkable] >= 0), "walkable corridor must be connected to source 1"
        assert np.all(d2[walkable] >= 0), "walkable corridor must be connected to source 2"
        assert int(d1[sources[1]]) >= 2 * MIN_SOURCE_DEPTH, (
            "the final looped maze must not make the sources trivially close"
        )
        assert _has_alternate_route(walkable, sources[0], sources[1]), (
            "sources must have more than one corridor route"
        )

        seam_cells = int(np.count_nonzero(seam))
        assert MIN_SEAM_CELLS <= seam_cells <= MAX_SEAM_CELLS
        nearest_seam_distance = int(np.min(d1[seam]))
        assert MIN_SOURCE_DEPTH <= nearest_seam_distance <= MAX_EQUIDISTANT_DISTANCE
        coords = np.argwhere(seam)
        assert len(set(map(int, coords[:, 0]))) >= 2 and len(set(map(int, coords[:, 1]))) >= 2
        assert int(np.count_nonzero((grid == CORRIDOR_COLOR) & (d1 < d2))) >= MIN_SIDE_CELLS
        assert int(np.count_nonzero((grid == CORRIDOR_COLOR) & (d2 < d1))) >= MIN_SIDE_CELLS

        manhattan = _manhattan_equal_mask(grid.shape, sources[0], sources[1])
        oracle = grid.copy()
        oracle[seam] = SEAM_COLOR
        naive = grid.copy()
        naive[(grid == CORRIDOR_COLOR) & manhattan] = SEAM_COLOR

        assert np.any(seam & ~manhattan), "maze routing must matter for at least one seam cell"
        assert not np.array_equal(oracle, naive), "output must not collapse to a simple Manhattan bisector"
        assert not np.array_equal(oracle, grid), "output must change at least one corridor cell"
        assert np.array_equal(main(grid), oracle), "main must match the independently checked oracle"
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randrange(MAZE_SIZE_MIN, MAZE_SIZE_MAX + 1, 2)
    cols = random.randrange(MAZE_SIZE_MIN, MAZE_SIZE_MAX + 1, 2)
    maze = _carve_perfect_maze(rows, cols)
    corridor = maze == CORRIDOR_COLOR
    pair = _pick_sources(corridor)
    if pair is None:
        return None

    maze = _carve_route_loop(maze, *pair)
    if maze is None:
        return None

    grid = _paint_sources(maze, *pair)
    if not _semantic_validation(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-source equidistant seam puzzle")
