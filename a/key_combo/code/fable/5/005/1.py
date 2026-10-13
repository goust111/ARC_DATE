from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 250
_MAX_SEED_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) >= 4, "expected floor, wall, and two seed colors"
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], (
        "the floor background must be uniquely most common"
    )
    return int(ordered[0][0])


def _roles(grid: np.ndarray) -> tuple[int, int, tuple[int, int], tuple[Coord, Coord]]:
    """Identify the visible roles without assigning meaning to color ids."""
    bg = _background_color(grid)
    colors = {int(value) for value in np.unique(grid)}
    assert len(colors) == 4, "expected exactly four visible colors"

    seed_colors = tuple(
        color
        for color in colors - {bg}
        if int(np.count_nonzero(grid == color)) == 1
    )
    assert len(seed_colors) == 2, "exactly two non-background colors must be singleton seeds"

    wall_colors = colors - {bg, *seed_colors}
    assert len(wall_colors) == 1, "the remaining non-background color must be the wall"
    wall = int(wall_colors.pop())
    assert int(np.count_nonzero(grid == wall)) > 1, "a wall must contain multiple pixels"

    # Sort positions spatially only to make the returned tuple deterministic.  The
    # transformation itself treats the two seed/color pairs symmetrically.
    pairs = []
    for color in seed_colors:
        locations = np.argwhere(grid == color)
        assert locations.shape == (1, 2)
        pos = tuple(map(int, locations[0]))
        pairs.append((pos, int(color)))
    pairs.sort(key=lambda item: item[0])
    positions = (pairs[0][0], pairs[1][0])
    colors_by_position = (pairs[0][1], pairs[1][1])
    return bg, wall, colors_by_position, positions


def _distances(shape: tuple[int, int], wall_mask: np.ndarray, start: Coord) -> np.ndarray:
    """Four-connected shortest-path lengths through every non-wall cell."""
    h, w = shape
    distance = np.full((h, w), -1, dtype=np.int16)
    assert not bool(wall_mask[start])
    distance[start] = 0
    queue: deque[Coord] = deque([start])
    while queue:
        r, c = queue.popleft()
        next_distance = int(distance[r, c]) + 1
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if wall_mask[rr, cc] or distance[rr, cc] >= 0:
                continue
            distance[rr, cc] = next_distance
            queue.append((rr, cc))
    return distance


def main(grid: np.ndarray) -> np.ndarray:
    """Color every maze-floor cell by its nearer seed in four-connected geodesic distance.

    Input structure:
    - A uniquely dominant background color is the maze floor, and one other
      non-singleton color forms wall barriers inside the grid.
    - Exactly two remaining colors each occur once, marking two distinct seed
      pixels on the same four-connected floor region.

    Rule:
    - For each background floor cell, compute the shortest four-connected path
      to each seed without crossing a wall.
    - Recolor it with the nearer seed's color; if the two path lengths are equal,
      leave it background.  Preserve all wall and seed pixels.

    Key insight:
    - Ownership uses distance along the winding maze, not straight-line distance;
      the unchanged floor cells trace the maze's geodesic bisector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    bg, wall, seed_colors, seeds = _roles(g)
    wall_mask = g == wall
    distance_a = _distances(g.shape, wall_mask, seeds[0])
    distance_b = _distances(g.shape, wall_mask, seeds[1])

    traversable = ~wall_mask
    assert bool(np.all(distance_a[traversable] >= 0)), (
        "all floor cells and both seeds must share one four-connected region"
    )
    assert bool(np.all(distance_b[traversable] >= 0))

    out = g.copy()
    floor = g == bg
    out[floor & (distance_a < distance_b)] = seed_colors[0]
    out[floor & (distance_b < distance_a)] = seed_colors[1]
    # Equal-distance floor cells, walls, and original seeds remain unchanged.
    return out


def _make_slalom(h: int, w: int) -> np.ndarray | None:
    """Return a boolean wall mask with alternating edge-attached barriers."""
    walls = np.zeros((h, w), dtype=bool)
    vertical = random.random() < 0.5
    axial = w if vertical else h
    cross = h if vertical else w

    barrier_count = random.randint(2, min(5, max(2, (axial - 5) // 4)))
    candidates = list(range(3, axial - 3))
    choices: list[tuple[int, ...]] = []

    # Keep neighboring barriers far enough apart that each channel is legible.
    for _ in range(80):
        positions = tuple(sorted(random.sample(candidates, barrier_count)))
        if all(b - a >= 3 for a, b in zip(positions, positions[1:])):
            choices.append(positions)
            break
    if not choices:
        return None

    opening_at_start = random.random() < 0.5
    for index, position in enumerate(choices[0]):
        opening = random.randint(2, min(4, cross - 5))
        gap_at_start = opening_at_start ^ bool(index % 2)
        if vertical:
            if gap_at_start:
                walls[opening:, position] = True
            else:
                walls[: h - opening, position] = True
        else:
            if gap_at_start:
                walls[position, opening:] = True
            else:
                walls[position, : w - opening] = True

        # Add a short perpendicular foot at some barrier tips.  This changes
        # route lengths while retaining a wide, visually readable passage.
        if random.random() < 0.45:
            foot = random.randint(1, 2)
            if vertical:
                tip_r = opening if gap_at_start else h - opening - 1
                direction = random.choice((-1, 1))
                c0, c1 = sorted((position, position + direction * foot))
                if 0 <= c0 and c1 < w:
                    walls[tip_r, c0 : c1 + 1] = True
            else:
                tip_c = opening if gap_at_start else w - opening - 1
                direction = random.choice((-1, 1))
                r0, r1 = sorted((position, position + direction * foot))
                if 0 <= r0 and r1 < h:
                    walls[r0 : r1 + 1, tip_c] = True

    return walls


def _all_floor_connected(walls: np.ndarray) -> bool:
    floor = np.argwhere(~walls)
    if floor.size == 0:
        return False
    start = tuple(map(int, floor[0]))
    distances = _distances(walls.shape, walls, start)
    return bool(np.all(distances[~walls] >= 0))


def _ownership(distance_a: np.ndarray, distance_b: np.ndarray) -> np.ndarray:
    """Encode A ownership as -1, ties as 0, and B ownership as 1."""
    return np.sign(distance_a.astype(np.int32) - distance_b.astype(np.int32)).astype(np.int8)


def _seed_pair_has_required_evidence(walls: np.ndarray, a: Coord, b: Coord) -> bool:
    """Reject easy/ambiguous pairs and retain explicit anti-shortcut evidence."""
    distance_a = _distances(walls.shape, walls, a)
    distance_b = _distances(walls.shape, walls, b)
    floor = ~walls
    if np.any(distance_a[floor] < 0) or np.any(distance_b[floor] < 0):
        return False

    ownership = _ownership(distance_a, distance_b)
    usable_floor = floor.copy()
    usable_floor[a] = False
    usable_floor[b] = False
    owned_a = usable_floor & (ownership < 0)
    owned_b = usable_floor & (ownership > 0)
    ties = usable_floor & (ownership == 0)
    if int(owned_a.sum()) < 12 or int(owned_b.sum()) < 12 or int(ties.sum()) < 2:
        return False

    rr, cc = np.indices(walls.shape)
    manhattan_a = np.abs(rr - a[0]) + np.abs(cc - a[1])
    manhattan_b = np.abs(rr - b[0]) + np.abs(cc - b[1])
    manhattan_ownership = _ownership(manhattan_a, manhattan_b)

    # A decisive counterexample is owned strictly by opposite seeds under the
    # visible straight-line shortcut and the actual corridor metric.
    opposite = usable_floor & (ownership * manhattan_ownership == -1)
    if int(opposite.sum()) < 2:
        return False

    # Both seeds must need the walls: each route map differs somewhere from its
    # obstacle-free Manhattan distance.
    detour_a = usable_floor & (distance_a > manhattan_a)
    detour_b = usable_floor & (distance_b > manhattan_b)
    if not bool(detour_a.any() and detour_b.any()):
        return False

    # Prevent a trivial full-grid half-plane: the tie set must span both axes or
    # contain a bend/disconnected cross-section.
    tie_coords = np.argwhere(ties)
    if len(np.unique(tie_coords[:, 0])) < 2 or len(np.unique(tie_coords[:, 1])) < 2:
        return False
    return True


def _sample_seed_pair(walls: np.ndarray) -> tuple[Coord, Coord] | None:
    floor_cells = [tuple(map(int, cell)) for cell in np.argwhere(~walls)]
    h, w = walls.shape
    interior = [
        cell
        for cell in floor_cells
        if 1 <= cell[0] < h - 1 and 1 <= cell[1] < w - 1
    ]
    if len(interior) < 2:
        return None

    # Bias toward separated seeds so both colors own a substantial visible area.
    minimum_separation = max(6, min(h, w) // 2)
    for _ in range(_MAX_SEED_ATTEMPTS):
        a, b = random.sample(interior, 2)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < minimum_separation:
            continue
        if _seed_pair_has_required_evidence(walls, a, b):
            return a, b
    return None


def _render_input(walls: np.ndarray, seeds: tuple[Coord, Coord]) -> np.ndarray:
    bg, wall, seed_a, seed_b = random.sample(_PALETTE, 4)
    grid = np.full(walls.shape, bg, dtype=int)
    grid[walls] = wall
    grid[seeds[0]] = seed_a
    grid[seeds[1]] = seed_b
    return grid


def _generator_semantic_check(grid: np.ndarray) -> bool:
    """Check the described rule and the generator's non-degeneracy promises."""
    try:
        bg, wall, seed_colors, seeds = _roles(grid)
        walls = grid == wall
        if not _seed_pair_has_required_evidence(walls, seeds[0], seeds[1]):
            return False

        distance_a = _distances(grid.shape, walls, seeds[0])
        distance_b = _distances(grid.shape, walls, seeds[1])
        expected = grid.copy()
        floor = grid == bg
        expected[floor & (distance_a < distance_b)] = seed_colors[0]
        expected[floor & (distance_b < distance_a)] = seed_colors[1]
        actual = main(grid)

        if not np.array_equal(actual, expected):
            return False
        if not np.array_equal(actual[walls], grid[walls]):
            return False
        if any(int(actual[pos]) != color for pos, color in zip(seeds, seed_colors)):
            return False
        if not bool(np.any(floor & (distance_a == distance_b) & (actual == bg))):
            return False
        if not bool(np.any(floor & (actual == seed_colors[0]))):
            return False
        if not bool(np.any(floor & (actual == seed_colors[1]))):
            return False
        return True
    except (AssertionError, ValueError, IndexError):
        return False


def generate_input() -> np.ndarray:
    """Generate a varied, semantically audited two-seed slalom maze."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        h = random.randint(15, 25)
        w = random.randint(15, 25)
        walls = _make_slalom(h, w)
        if walls is None or int(walls.sum()) < 8:
            continue
        if int((~walls).sum()) <= int(walls.sum()) or not _all_floor_connected(walls):
            continue

        seeds = _sample_seed_pair(walls)
        if seeds is None:
            continue
        grid = _render_input(walls, seeds)
        if _generator_semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid geodesic maze-coloring input")
