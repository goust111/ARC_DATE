from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBSTACLE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GENERATION_ATTEMPTS = 600

Coord = tuple[int, int]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the shared ARC convention: shape=(rows, cols), coord=(r, c)."""
    height, width = map(int, shape)
    return r in (0, height - 1) or c in (0, width - 1)


def _turn(direction: Coord, counterclockwise: bool) -> Coord:
    """Rotate a screen-grid (dr, dc) vector by one quarter-turn."""
    dr, dc = direction
    return (-dc, dr) if counterclockwise else (dc, -dr)


def _read_legend(grid: np.ndarray, background: int) -> tuple[list[int], set[Coord]]:
    """Read adjacent 2x2 solid swatches from the upper-left, in visible order."""
    height, width = map(int, grid.shape)
    assert height >= 4 and width >= 6
    colors: list[int] = []
    cells: set[Coord] = set()
    c = 0
    while c + 1 < width:
        patch = grid[0:2, c : c + 2]
        color = int(patch[0, 0])
        if color in (background, OBSTACLE) or not bool(np.all(patch == color)):
            break
        colors.append(color)
        cells.update((r, cc) for r in (0, 1) for cc in (c, c + 1))
        c += 2
    assert 4 <= len(colors) <= 5, "expected four or five upper-left 2x2 color swatches"
    assert len(set(colors)) == len(colors), "legend swatches must have distinct colors"
    return colors, cells


def _read_start(grid: np.ndarray, background: int, legend_cells: set[Coord]) -> tuple[Coord, Coord]:
    """Find the sole non-grey edge singleton outside the legend and point inward."""
    height, width = map(int, grid.shape)
    candidates: list[Coord] = []
    for r in range(height):
        for c in range(width):
            if not _is_edge(grid.shape, r, c) or (r, c) in legend_cells:
                continue
            if int(grid[r, c]) not in (background, OBSTACLE):
                candidates.append((r, c))
    assert len(candidates) == 1, "expected one edge start marker outside the legend"
    r, c = candidates[0]
    assert r not in (0, height - 1) or c not in (0, width - 1), "start marker cannot be a corner"
    if r == 0:
        direction = (1, 0)
    elif r == height - 1:
        direction = (-1, 0)
    elif c == 0:
        direction = (0, 1)
    else:
        direction = (0, -1)
    return (r, c), direction


def _simulate(
    shape: tuple[int, int],
    obstacles: set[Coord],
    start: Coord,
    direction: Coord,
    color_count: int,
    *,
    count_boundaries: bool = True,
    alternate: bool = True,
) -> tuple[list[Coord], int]:
    """Return obstacle hits and boundary-hit count; stop when colors are exhausted."""
    position = start
    active_obstacles = set(obstacles)
    hits: list[Coord] = []
    boundary_hits = 0
    turn_index = 0
    seen: set[tuple[Coord, Coord, int, int]] = set()

    # A valid puzzle stops after color_count obstacle impacts.  State detection
    # gives malformed inputs a clear failure instead of an infinite loop.
    while len(hits) < color_count:
        state = (position, direction, turn_index % 2, len(hits))
        assert state not in seen, "trajectory cycles before the color sequence is exhausted"
        seen.add(state)

        nr = position[0] + direction[0]
        nc = position[1] + direction[1]
        # A visible grey obstacle has precedence when it occupies a perimeter
        # cell; otherwise that perimeter cell is the boundary wall.
        if (nr, nc) in active_obstacles:
            hits.append((nr, nc))
            # Recoloring removes the cell from the set of grey obstacles; the
            # invisible particle can cross that non-grey cell on a later leg.
            active_obstacles.remove((nr, nc))
            if len(hits) == color_count:
                break
            ccw = True if not alternate else (turn_index % 2 == 0)
            direction = _turn(direction, ccw)
            turn_index += 1
            continue

        if _is_edge(shape, nr, nc):
            # The perimeter is the visible boundary wall.  The edge start is
            # the only exception: its first inward step enters the playfield.
            if _is_edge(shape, position[0], position[1]):
                height, width = shape
                assert 0 < nr < height - 1 and 0 < nc < width - 1
                position = (nr, nc)
                continue
            boundary_hits += 1
            ccw = True if not alternate else (turn_index % 2 == 0)
            direction = _turn(direction, ccw)
            if count_boundaries:
                turn_index += 1
            continue

        assert 0 < nr < shape[0] - 1 and 0 < nc < shape[1] - 1
        position = (nr, nc)

    return hits, boundary_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey obstacles in the order hit by an invisible, alternately turning particle.

    Input structure:
    - The dominant background contains three to five adjacent 2x2 solid color
      swatches at upper-left, one non-corner edge start marker, and separated
      grey single-cell obstacles in the field (occasionally on the perimeter).
    - The start's edge determines its unique inward cardinal direction; the
      outermost row and column form the visible boundary wall.

    Rule:
    - Move straight from the start without drawing the path.  On a grey obstacle,
      recolor that cell with the next left-to-right legend color and turn at
      the preceding cell; once non-grey, that cell no longer blocks a later leg.
      A visible grey obstacle takes precedence on a perimeter cell; every other
      perimeter cell is a boundary that turns without using color.
    - Starting with counterclockwise, alternate counterclockwise and clockwise
      over every collision, including boundary collisions, and stop immediately
      when the last legend color is used.  Preserve all other cells.

    Key insight:
    - Obstacle colors encode invisible path-visit order, and unpainted boundary
      bounces still advance the shared alternating-turn rhythm.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    values, counts = np.unique(g, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert background != OBSTACLE

    sequence, legend_cells = _read_legend(g, background)
    start, direction = _read_start(g, background, legend_cells)
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)}
    assert len(obstacles) > len(sequence), "expected both target and untouched grey obstacles"

    hits, boundary_hits = _simulate(g.shape, obstacles, start, direction, len(sequence))
    assert boundary_hits > 0, "the trajectory must include a boundary collision"

    out = g.copy()
    for coord, color in zip(hits, sequence):
        out[coord] = color
    return out


def _open_ray(shape: tuple[int, int], position: Coord, direction: Coord, obstacles: set[Coord]) -> tuple[list[Coord], Coord | None]:
    """List free interior cells up to the nearest obstacle or perimeter wall."""
    free: list[Coord] = []
    r, c = position
    while True:
        r += direction[0]
        c += direction[1]
        if (r, c) in obstacles:
            return free, (r, c)
        if _is_edge(shape, r, c):
            return free, None
        free.append((r, c))


def _plan_path(shape: tuple[int, int], start: Coord, direction: Coord, color_count: int) -> tuple[set[Coord], list[Coord], set[Coord]] | None:
    """Construct O,B,O,O(edge),O[,O(edge)], making the boundary beat necessary."""
    assert color_count in (4, 5)
    events = ["obstacle", "boundary", "obstacle", "edge_obstacle", "obstacle"]
    if color_count == 5:
        events.append("edge_obstacle")
    obstacles: set[Coord] = set()
    targets: list[Coord] = []
    swept: set[Coord] = {start}
    position = start
    turn_index = 0

    for event in events:
        # Earlier targets have already been recolored, so only as-yet-unhit
        # obstacles would block.  Targets are introduced in visit order here.
        free, blocker = _open_ray(shape, position, direction, set())
        if event == "boundary":
            if blocker is not None:
                return None
            swept.update(free)
            if free:
                position = free[-1]
        elif event == "edge_obstacle":
            target = (position[0] + direction[0], position[1] + direction[1])
            if not _is_edge(shape, *target) or target in obstacles or target == start:
                return None
            obstacles.add(target)
            targets.append(target)
        else:
            # Leave at least one visible empty step before each target.  Do not
            # put a later target on an already travelled path.
            candidates = [cell for cell in free[1:] if cell not in swept]
            if not candidates:
                return None
            # Bias away from very long empty corridors while retaining variety.
            limit = max(1, (2 * len(candidates)) // 3)
            target = random.choice(candidates[:limit])
            idx = free.index(target)
            swept.update(free[:idx])
            position = free[idx - 1]
            obstacles.add(target)
            targets.append(target)

        direction = _turn(direction, turn_index % 2 == 0)
        turn_index += 1

    return obstacles, targets, swept


def _sample_start(height: int, width: int, legend_width: int) -> tuple[Coord, Coord]:
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        if side == "top":
            choices = list(range(max(legend_width + 1, 2), width - 1))
            if choices:
                c = random.choice(choices)
                return (0, c), (1, 0)
        elif side == "bottom":
            c = random.randint(1, width - 2)
            return (height - 1, c), (-1, 0)
        elif side == "left":
            r = random.randint(3, height - 2)
            return (r, 0), (0, 1)
        else:
            r = random.randint(2, height - 2)
            return (r, width - 1), (0, -1)
    raise AssertionError("no valid start location")


def _wrong_rule_differs(shape: tuple[int, int], obstacles: set[Coord], start: Coord, direction: Coord, targets: list[Coord]) -> bool:
    """Reject samples that fail to disprove either major shortcut."""
    count = len(targets)
    try:
        wrong_boundary, _ = _simulate(
            shape, obstacles, start, direction, count, count_boundaries=False
        )
    except AssertionError:
        wrong_boundary = []
    try:
        always_left, _ = _simulate(
            shape, obstacles, start, direction, count, alternate=False
        )
    except AssertionError:
        always_left = []
    return wrong_boundary != targets and always_left != targets


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(4, 5)
    height = random.randint(16, 25)
    width = random.randint(max(16, 2 * color_count + 5), 26)
    sequence = random.sample(PALETTE, color_count)
    start_color = random.choice([color for color in PALETTE if color not in sequence])
    start, direction = _sample_start(height, width, 2 * color_count)

    planned = _plan_path((height, width), start, direction, color_count)
    if planned is None:
        return None
    obstacles, targets, swept = planned
    if len(targets) != color_count or len(set(targets)) != color_count:
        return None

    forbidden = swept | obstacles | {start}
    distractor_candidates = [
        (r, c)
        for r in range(2, height - 2)
        for c in range(1, width - 1)
        if (r, c) not in forbidden
        and all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in obstacles)
    ]
    random.shuffle(distractor_candidates)
    distractors: list[Coord] = []
    wanted = random.randint(2, min(7, max(2, len(distractor_candidates))))
    for cell in distractor_candidates:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in distractors):
            distractors.append(cell)
            if len(distractors) == wanted:
                break
    if len(distractors) < 2:
        return None
    obstacles.update(distractors)

    try:
        hits, boundary_hits = _simulate((height, width), obstacles, start, direction, color_count)
    except AssertionError:
        return None
    if hits != targets or boundary_hits < 1:
        return None
    if not _wrong_rule_differs((height, width), obstacles, start, direction, targets):
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for i, color in enumerate(sequence):
        grid[0:2, 2 * i : 2 * i + 2] = color
    grid[start] = start_color
    for obstacle in obstacles:
        grid[obstacle] = OBSTACLE

    expected = grid.copy()
    for target, color in zip(targets, sequence):
        expected[target] = color
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(np.count_nonzero(actual == OBSTACLE)) != len(distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked examples with necessary boundary timing."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pendulum-calling input")
