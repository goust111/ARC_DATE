from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    """Yield orthogonal neighbors in the common ARC (row, column) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _distances(passable: np.ndarray, start: Coord) -> np.ndarray:
    """Four-connected unweighted shortest-path distances; -1 means unreachable."""
    height, width = passable.shape
    dist = np.full((height, width), -1, dtype=int)
    sr, sc = start
    if not (0 <= sr < height and 0 <= sc < width and bool(passable[sr, sc])):
        return dist

    dist[sr, sc] = 0
    queue: deque[Coord] = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        next_distance = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not bool(passable[nr, nc]) or int(dist[nr, nc]) >= 0:
                continue
            dist[nr, nc] = next_distance
            queue.append((nr, nc))
    return dist


def _find_sources(grid: np.ndarray) -> list[tuple[Coord, int]]:
    source_colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY)]
    assert len(source_colors) == 2, "expected exactly two non-black, non-gray source colors"

    sources: list[tuple[Coord, int]] = []
    for color in source_colors:
        coords = np.argwhere(grid == color)
        assert len(coords) == 1, "each source color must occur at exactly one pixel"
        r, c = map(int, coords[0])
        sources.append(((r, c), color))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """Let two colored sources compete along corridors by four-connected geodesic distance.

    Input structure:
    - A black background is divided into corridor regions by gray wall pixels.
    - Exactly two differently colored, single-pixel sources lie in the non-wall space.

    Rule:
    - Compute each cell's four-connected shortest-path distance to both sources without crossing gray walls.
    - Recolor a black cell with the strictly nearer source color; equal-distance and doubly unreachable cells stay black.
    - Preserve gray walls and the two source pixels.

    Key insight:
    - Ownership is determined by geodesic distance through the corridor network, so wall-forced detours—not straight-line distance—locate the black tie seam.
    """
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30x30"
    assert bool(np.any(g == GRAY)), "expected gray wall pixels"

    sources = _find_sources(g)
    passable = g != GRAY
    (source_a, color_a), (source_b, color_b) = sources
    dist_a = _distances(passable, source_a)
    dist_b = _distances(passable, source_b)

    background = g == BLACK
    reached_a = dist_a >= 0
    reached_b = dist_b >= 0
    owned_a = background & reached_a & (~reached_b | (dist_a < dist_b))
    owned_b = background & reached_b & (~reached_a | (dist_b < dist_a))

    out = g.copy()
    out[owned_a] = color_a
    out[owned_b] = color_b
    return out


def _carve_perfect_maze(height: int, width: int) -> tuple[np.ndarray, Coord]:
    """Carve a randomized tree maze whose corridors occupy odd lattice cells."""
    grid = np.full((height, width), GRAY, dtype=int)
    lattice = [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]
    start = random.choice(lattice)
    visited = {start}
    stack = [start]
    grid[start] = BLACK

    while stack:
        r, c = stack[-1]
        choices = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in visited:
                continue
            if 1 <= nxt[0] < height - 1 and 1 <= nxt[1] < width - 1:
                choices.append(nxt)
        if not choices:
            stack.pop()
            continue

        nr, nc = random.choice(choices)
        grid[(r + nr) // 2, (c + nc) // 2] = BLACK
        grid[nr, nc] = BLACK
        visited.add((nr, nc))
        stack.append((nr, nc))
    return grid, start


def _add_unreachable_pocket(grid: np.ndarray) -> Coord | None:
    """Turn one wall pixel enclosed on four sides by walls into an isolated black pocket."""
    height, width = grid.shape
    candidates = []
    for r in range(2, height - 2):
        for c in range(2, width - 2):
            if int(grid[r, c]) != GRAY:
                continue
            if all(int(grid[nr, nc]) == GRAY for nr, nc in _neighbors4(r, c)):
                candidates.append((r, c))
    if not candidates:
        return None
    pocket = random.choice(candidates)
    grid[pocket] = BLACK
    return pocket


def _main_component(grid: np.ndarray, start: Coord) -> tuple[list[Coord], np.ndarray]:
    dist = _distances(grid != GRAY, start)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(dist >= 0)]
    return cells, dist


def _choose_sources(grid: np.ndarray, maze_start: Coord) -> tuple[Coord, Coord] | None:
    """Choose sources that expose ties and a wall-induced Manhattan-distance trap."""
    cells, _ = _main_component(grid, maze_start)
    if len(cells) < 35:
        return None

    distance_cache: dict[Coord, np.ndarray] = {}

    def distances_from(cell: Coord) -> np.ndarray:
        if cell not in distance_cache:
            distance_cache[cell] = _distances(grid != GRAY, cell)
        return distance_cache[cell]

    attempts = min(450, max(180, len(cells) * 2))
    for _ in range(attempts):
        source_a, source_b = random.sample(cells, 2)
        dist_a = distances_from(source_a)
        route_length = int(dist_a[source_b])
        direct_length = abs(source_a[0] - source_b[0]) + abs(source_a[1] - source_b[1])
        if route_length < 10 or route_length % 2 != 0 or route_length < direct_length + 4:
            continue

        dist_b = distances_from(source_b)
        reachable = dist_a >= 0
        owned_a = reachable & (dist_a < dist_b)
        owned_b = reachable & (dist_b < dist_a)
        ties = reachable & (dist_a == dist_b)
        # Subtract the source pixel itself when measuring visible recolored territory.
        if int(owned_a.sum()) < 8 or int(owned_b.sum()) < 8:
            continue
        tie_count = int(ties.sum())
        if not (1 <= tie_count <= max(10, len(cells) // 4)):
            continue

        rows, cols = np.indices(grid.shape)
        manhattan_a = np.abs(rows - source_a[0]) + np.abs(cols - source_a[1])
        manhattan_b = np.abs(rows - source_b[0]) + np.abs(cols - source_b[1])
        reversed_owner = reachable & (
            ((dist_a < dist_b) & (manhattan_a > manhattan_b))
            | ((dist_b < dist_a) & (manhattan_b > manhattan_a))
        )
        if int(reversed_owner.sum()) < 1:
            continue
        return source_a, source_b
    return None


def _semantic_profile(grid: np.ndarray) -> dict[str, int | bool]:
    """Measure the non-degeneracy properties required of every generated puzzle."""
    sources = _find_sources(grid)
    (source_a, _), (source_b, _) = sources
    passable = grid != GRAY
    dist_a = _distances(passable, source_a)
    dist_b = _distances(passable, source_b)
    background = grid == BLACK
    both = background & (dist_a >= 0) & (dist_b >= 0)
    owned_a = both & (dist_a < dist_b)
    owned_b = both & (dist_b < dist_a)
    ties = both & (dist_a == dist_b)
    unreachable = background & (dist_a < 0) & (dist_b < 0)

    rows, cols = np.indices(grid.shape)
    manhattan_a = np.abs(rows - source_a[0]) + np.abs(cols - source_a[1])
    manhattan_b = np.abs(rows - source_b[0]) + np.abs(cols - source_b[1])
    reversed_owner = both & (
        ((dist_a < dist_b) & (manhattan_a > manhattan_b))
        | ((dist_b < dist_a) & (manhattan_b > manhattan_a))
    )
    return {
        "territory_a": int(owned_a.sum()),
        "territory_b": int(owned_b.sum()),
        "ties": int(ties.sum()),
        "unreachable": int(unreachable.sum()),
        "geodesic_manhattan_reversals": int(reversed_owner.sum()),
        "sources_connected": bool(dist_a[source_b] >= 0),
    }


def _valid_generated_input(grid: np.ndarray) -> bool:
    """Validate structure, unique semantics, key dependencies, and non-degeneracy."""
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        profile = _semantic_profile(grid)
        out = main(grid)
    except (AssertionError, ValueError):
        return False

    if not profile["sources_connected"]:
        return False
    if min(int(profile["territory_a"]), int(profile["territory_b"])) < 8:
        return False
    if int(profile["ties"]) < 1 or int(profile["unreachable"]) < 1:
        return False
    if int(profile["geodesic_manhattan_reversals"]) < 1:
        return False

    # Direct semantic invariants: only reachable black cells may change; walls and sources persist.
    sources = _find_sources(grid)
    (source_a, color_a), (source_b, color_b) = sources
    dist_a = _distances(grid != GRAY, source_a)
    dist_b = _distances(grid != GRAY, source_b)
    expected = grid.copy()
    for r, c in np.argwhere(grid == BLACK):
        r, c = int(r), int(c)
        da, db = int(dist_a[r, c]), int(dist_b[r, c])
        if da >= 0 and (db < 0 or da < db):
            expected[r, c] = color_a
        elif db >= 0 and (da < 0 or db < da):
            expected[r, c] = color_b
    return bool(np.array_equal(out, expected))


def generate_input() -> np.ndarray:
    """Generate varied mazes containing every important semantic branch."""
    odd_sizes = (15, 17, 19, 21, 23, 25, 27, 29)
    for _ in range(350):
        height = random.choice(odd_sizes)
        width = random.choice(odd_sizes)
        grid, maze_start = _carve_perfect_maze(height, width)
        if _add_unreachable_pocket(grid) is None:
            continue

        sources = _choose_sources(grid, maze_start)
        if sources is None:
            continue
        source_a, source_b = sources
        color_a, color_b = random.sample(SOURCE_COLORS, 2)
        grid[source_a] = color_a
        grid[source_b] = color_b

        if _valid_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a non-degenerate dual-source geodesic puzzle")
