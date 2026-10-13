from __future__ import annotations

from collections import deque
import random

import numpy as np


# The stable visible palette across examples makes every role unambiguous.
BACKGROUND_COLOR = 0
WALL_COLOR = 5
FRONTIER_COLOR = 2
SOURCE_COLORS = (1, 3, 4, 6, 7, 8, 9)

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10**9

MAZE_SIZES = tuple(range(15, 30, 2))
MIN_SOURCE_DISTANCE = 8
MIN_EXCLUSIVE_CELLS = 6
MAX_FRONTIER_CELLS = 30
SOURCE_SAMPLES_PER_MAZE = 100
GENERATION_ATTEMPTS = 500

# Consecutive calls cover a bent two-source frontier, a three-source case, and
# a forked frontier.  Geometry and colors remain freshly random in every call.
_generation_counter = 0


def _neighbors(shape: tuple[int, int], r: int, c: int):
    """Yield in-bounds 4-neighbors in the canonical (row, column) frame."""
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _distance_map(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Return 4-connected shortest-path distances through passable cells."""
    dist = np.full(passable.shape, INF, dtype=int)
    sr, sc = start
    assert bool(passable[sr, sc]), "a source must be on passable ground"
    dist[sr, sc] = 0
    queue: deque[tuple[int, int]] = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        next_distance = int(dist[r, c]) + 1
        for nr, nc in _neighbors(passable.shape, r, c):
            if bool(passable[nr, nc]) and int(dist[nr, nc]) == INF:
                dist[nr, nc] = next_distance
                queue.append((nr, nc))
    return dist


def _find_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    """Read the differently colored singleton source pixels from the input."""
    g = np.asarray(grid, dtype=int)
    assert not np.any(g == FRONTIER_COLOR), "the red frontier color must be absent from the input"

    source_mask = ~np.isin(g, (BACKGROUND_COLOR, WALL_COLOR))
    source_cells = [tuple(map(int, cell)) for cell in np.argwhere(source_mask)]
    assert 2 <= len(source_cells) <= len(SOURCE_COLORS), "expected at least two isolated source pixels"

    source_values = [int(g[cell]) for cell in source_cells]
    assert len(set(source_values)) == len(source_values), "every source must have its own visible color"
    for color in source_values:
        assert color in SOURCE_COLORS, "unexpected non-background color"
        assert int(np.count_nonzero(g == color)) == 1, "each source color must occur exactly once"

    return source_cells


def _distance_stack(passable: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([_distance_map(passable, source) for source in sources], axis=0)


def _tie_mask_from_distances(distances: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    minimum = np.min(distances, axis=0)
    nearest_count = np.sum(distances == minimum[None, :, :], axis=0)
    return eligible & (minimum < INF) & (nearest_count >= 2)


def _frontier_analysis(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sources = _find_sources(g)
    passable = g != WALL_COLOR
    distances = _distance_stack(passable, sources)
    frontier = _tie_mask_from_distances(distances, g == BACKGROUND_COLOR)
    return {
        "sources": sources,
        "passable": passable,
        "distances": distances,
        "frontier": frontier,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every passable cell tied for the smallest wall-avoiding distance to the colored sources.

    Input structure:
    - Black cells are passable floor and gray cells form one-pixel-thick maze-wall segments.
    - Two or three differently colored isolated source pixels lie in the connected floor network.
    - Red, the output-only contact-frontier color, is absent from the input.

    Rule:
    - From every source, measure shortest-path distance in the original grid using only 4-connected moves and never crossing gray walls.
    - Recolor a black floor cell red exactly when two or more sources attain the finite minimum distance there.
    - Preserve every other floor cell, every wall cell, and every source pixel.

    Key insight:
    - The marked cells are ties between simultaneous geodesic wavefronts, so maze detours can bend, split, or fork the frontier away from a geometric bisector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    analysis = _frontier_analysis(g)
    out = g.copy()
    out[analysis["frontier"]] = FRONTIER_COLOR
    return out


def _carve_prim_maze(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    """Carve a random connected, acyclic corridor network inside a wall border."""
    assert rows % 2 == 1 and cols % 2 == 1
    room_rows = (rows - 1) // 2
    room_cols = (cols - 1) // 2
    passable = np.zeros((rows, cols), dtype=bool)

    start = (rng.randrange(room_rows), rng.randrange(room_cols))
    visited = {start}
    passable[2 * start[0] + 1, 2 * start[1] + 1] = True
    edges: list[tuple[int, int, int, int, int, int]] = []

    def add_edges(room: tuple[int, int]) -> None:
        r, c = room
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < room_rows and 0 <= nc < room_cols and (nr, nc) not in visited:
                edges.append((r, c, nr, nc, dr, dc))

    add_edges(start)
    while edges:
        index = rng.randrange(len(edges))
        r, c, nr, nc, dr, dc = edges.pop(index)
        if (nr, nc) in visited:
            continue
        visited.add((nr, nc))
        passable[2 * r + 1 + dr, 2 * c + 1 + dc] = True
        passable[2 * nr + 1, 2 * nc + 1] = True
        add_edges((nr, nc))

    return passable


def _degree(passable: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(bool(passable[nr, nc]) for nr, nc in _neighbors(passable.shape, r, c))


def _component_count(mask: np.ndarray) -> int:
    remaining = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors(mask.shape, r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
    return count


def _manhattan_tie_mask(
    shape: tuple[int, int],
    sources: list[tuple[int, int]],
    eligible: np.ndarray,
) -> np.ndarray:
    rr, cc = np.indices(shape)
    distances = np.stack([np.abs(rr - r) + np.abs(cc - c) for r, c in sources], axis=0)
    minimum = np.min(distances, axis=0)
    return eligible & (np.sum(distances == minimum[None, :, :], axis=0) >= 2)


def _source_layout_is_semantic(
    passable: np.ndarray,
    sources: list[tuple[int, int]],
    mode: int,
) -> bool:
    """Reject layouts that permit shortcuts or make an object semantically idle."""
    distances = _distance_stack(passable, sources)
    if np.any(distances[:, passable] >= INF):
        return False

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if int(distances[i][sources[j]]) < MIN_SOURCE_DISTANCE:
                return False

    eligible = passable.copy()
    for source in sources:
        eligible[source] = False

    minimum = np.min(distances, axis=0)
    nearest = distances == minimum[None, :, :]
    nearest_count = np.sum(nearest, axis=0)
    frontier = eligible & (minimum < INF) & (nearest_count >= 2)
    frontier_count = int(np.count_nonzero(frontier))

    minimum_frontier = (3, 4, 4)[mode]
    if not (minimum_frontier <= frontier_count <= MAX_FRONTIER_CELLS):
        return False
    if frontier_count > int(np.count_nonzero(passable)) // 5:
        return False

    # Every source owns a visible territory and participates in a marked tie.
    for i in range(len(sources)):
        exclusive = eligible & nearest[i] & (nearest_count == 1)
        if int(np.count_nonzero(exclusive)) < MIN_EXCLUSIVE_CELLS:
            return False
        if not np.any(frontier & nearest[i]):
            return False

        others = [source for j, source in enumerate(sources) if j != i]
        reduced_distances = _distance_stack(passable, others)
        reduced_frontier = _tie_mask_from_distances(reduced_distances, eligible)
        if np.array_equal(reduced_frontier, frontier):
            return False

    # Removing the walls (Manhattan distance) must give a genuinely different answer.
    manhattan_frontier = _manhattan_tie_mask(passable.shape, sources, eligible)
    if np.array_equal(frontier, manhattan_frontier):
        return False
    if not np.any(frontier & ~manhattan_frontier):
        return False

    coords = np.argwhere(frontier)
    if mode == 0:
        # A visibly bent frontier rules out a straight row/column bisector.
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            return False
    elif mode == 1:
        # Three sources should pinch the frontier into separate pairwise contacts.
        if len(sources) != 3 or _component_count(frontier) < 2:
            return False
    else:
        # A degree-three frontier cell makes the wall-induced fork unmistakable.
        forked = any(
            sum(bool(frontier[nr, nc]) for nr, nc in _neighbors(frontier.shape, int(r), int(c))) >= 3
            for r, c in coords
        )
        if not forked:
            return False

    return True


def _render_input(
    passable: np.ndarray,
    sources: list[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full(passable.shape, WALL_COLOR, dtype=int)
    grid[passable] = BACKGROUND_COLOR
    colors = rng.sample(SOURCE_COLORS, len(sources))
    rng.shuffle(colors)
    for source, color in zip(sources, colors):
        grid[source] = int(color)
    return grid


def _semantic_validation(grid: np.ndarray, mode: int) -> bool:
    """Check role uniqueness and every cell of the stated wavefront rule."""
    try:
        g = np.asarray(grid, dtype=int)
        assert g.ndim == 2 and max(g.shape) <= 30
        assert set(map(int, np.unique(g))) <= {
            BACKGROUND_COLOR,
            WALL_COLOR,
            *SOURCE_COLORS,
        }

        analysis = _frontier_analysis(g)
        sources = analysis["sources"]
        expected_count = 3 if mode == 1 else 2
        assert len(sources) == expected_count
        assert _source_layout_is_semantic(analysis["passable"], sources, mode)

        distances = analysis["distances"]
        minimum = np.min(distances, axis=0)
        nearest_count = np.sum(distances == minimum[None, :, :], axis=0)
        frontier = analysis["frontier"]

        # These assertions quantify the rule directly instead of merely closing
        # the generate_input -> main loop.
        assert np.all((nearest_count[frontier] >= 2) & (minimum[frontier] < INF))
        ordinary_floor = (g == BACKGROUND_COLOR) & ~frontier
        assert np.all((nearest_count[ordinary_floor] == 1) | (minimum[ordinary_floor] >= INF))

        expected = g.copy()
        expected[frontier] = FRONTIER_COLOR
        solved = main(g)
        assert np.array_equal(solved, expected)
        assert np.array_equal(solved[g == WALL_COLOR], g[g == WALL_COLOR])
        source_mask = ~np.isin(g, (BACKGROUND_COLOR, WALL_COLOR))
        assert np.array_equal(solved[source_mask], g[source_mask])
        assert not np.array_equal(solved, g)
        return True
    except (AssertionError, ValueError):
        return False


def _gen_attempt(mode: int, rng: random.Random) -> np.ndarray | None:
    rows = rng.choice(MAZE_SIZES)
    cols = rng.choice(MAZE_SIZES)
    passable = _carve_prim_maze(rows, cols, rng)

    leaves = [
        tuple(map(int, cell))
        for cell in np.argwhere(passable)
        if _degree(passable, tuple(map(int, cell))) == 1
    ]
    source_count = 3 if mode == 1 else 2
    if len(leaves) < source_count:
        return None

    for _ in range(SOURCE_SAMPLES_PER_MAZE):
        sources = rng.sample(leaves, source_count)
        if not _source_layout_is_semantic(passable, sources, mode):
            continue
        grid = _render_input(passable, sources, rng)
        if _semantic_validation(grid, mode):
            return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, semantically non-degenerate wavefront-frontier input."""
    global _generation_counter

    mode = _generation_counter % 3
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode, rng)
        if grid is not None:
            _generation_counter += 1
            return grid
    raise RuntimeError(f"failed to generate wavefront-frontier mode {mode}")
