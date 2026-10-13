from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRAY = 5
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GENERATION_PROFILE = 0


def _background(grid: np.ndarray) -> int:
    """Return the dominant ARC color, following arc_utils.get_bg."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    """Yield cardinal neighbors in the common (row, column) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _read_scene(grid: np.ndarray):
    """Decode the visible compass, snake tape, and isolated start pixel."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)

    # Gray is the visible center marker.  Its four colored arms, not color ids,
    # define the direction code.
    compass_candidates = []
    for rr, cc in np.argwhere(g == GRAY):
        r, c = int(rr), int(cc)
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        arms = {
            (-1, 0): int(g[r - 1, c]),
            (1, 0): int(g[r + 1, c]),
            (0, -1): int(g[r, c - 1]),
            (0, 1): int(g[r, c + 1]),
        }
        colors = list(arms.values())
        if all(v not in (bg, GRAY) for v in colors) and len(set(colors)) == 4:
            compass_candidates.append(((r, c), arms))
    assert len(compass_candidates) == 1, "the compass center must be unique"

    center, direction_to_color = compass_candidates[0]
    color_to_direction = {color: direction for direction, color in direction_to_color.items()}
    instruction_colors = set(color_to_direction)
    compass_cells = {center}
    for dr, dc in DIRECTIONS:
        compass_cells.add((center[0] + dr, center[1] + dc))

    # Removing the four legend arms leaves exactly the serial instruction tape.
    snake_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(g, list(instruction_colors)))
        if (int(r), int(c)) not in compass_cells
    }
    assert len(snake_cells) >= 2, "the instruction snake is missing"

    degree = {
        p: sum(q in snake_cells for q in _neighbors4(*p))
        for p in snake_cells
    }
    endpoints = [p for p, d in degree.items() if d == 1]
    assert len(endpoints) == 2, "the snake must have exactly two endpoints"
    assert all(d in (1, 2) for d in degree.values()), "the snake must be unbranched"

    # This is the stated global-grid endpoint reference frame: row first, then
    # column only as the tie breaker.
    current = min(endpoints, key=lambda p: (p[0], p[1]))
    ordered_cells = []
    previous = None
    while True:
        ordered_cells.append(current)
        next_cells = [
            q for q in _neighbors4(*current)
            if q in snake_cells and q != previous
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "the snake traversal is ambiguous"
        previous, current = current, next_cells[0]
    assert len(ordered_cells) == len(snake_cells), "the snake must be connected"

    occupied = snake_cells | compass_cells
    starts = [
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in occupied
    ]
    assert len(starts) == 1, "there must be one isolated start pixel"
    start = starts[0]
    start_color = int(g[start])
    assert start_color not in instruction_colors, "start color cannot be an instruction color"

    colors = [int(g[p]) for p in ordered_cells]
    return bg, compass_cells, snake_cells, start, start_color, color_to_direction, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute a multicolored snake as compass-coded moves and draw the resulting trail.

    Input structure:
    - A dominant background contains one 4-connected, unbranched multicolor snake
      with two endpoints, a 3x3 compass whose gray center has four differently
      colored cardinal arms, and one isolated start pixel of another color.
    - Every snake cell uses one of the compass-arm colors.

    Rule:
    - Start reading at the snake endpoint with the smallest row, breaking an
      endpoint-row tie by the smallest column, and traverse every snake cell.
    - Each read color means the cardinal direction shown by that color around
      the gray compass center.  Move one cell from the start for every command
      and paint the landing cell with its command color; later visits overwrite
      earlier ones.  Erase the snake and compass from the output.

    Key insight:
    - The snake is a serial instruction tape: its endpoint fixes instruction
      order, the compass supplies the color-to-direction code, and the isolated
      pixel supplies the trajectory's origin.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, _legend, _snake, start, start_color, color_to_direction, colors = _read_scene(g)

    out = np.full(g.shape, bg, dtype=int)
    r, c = start
    out[r, c] = start_color
    for color in colors:
        dr, dc = color_to_direction[color]
        r, c = r + dr, c + dc
        assert 0 <= r < h and 0 <= c < w, "instruction trail leaves the grid"
        out[r, c] = color
    return out


def _walk_positions(commands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    positions = [(0, 0)]
    r = c = 0
    for dr, dc in commands:
        r, c = r + dr, c + dc
        positions.append((r, c))
    return positions


def _grow_new_steps(
    commands: list[tuple[int, int]],
    positions: list[tuple[int, int]],
    target_length: int,
    *,
    box_radius: int = 4,
) -> bool:
    """Extend a walk through new cells only, preserving existing crossings."""
    visited = set(positions)
    while len(commands) < target_length:
        r, c = positions[-1]
        choices = []
        for step in DIRECTIONS:
            q = (r + step[0], c + step[1])
            if q in visited or q == (0, 0):
                continue
            if max(abs(q[0]), abs(q[1])) > box_radius:
                continue
            choices.append((step, q))
        if not choices:
            return False
        step, q = random.choice(choices)
        commands.append(step)
        positions.append(q)
        visited.add(q)
    return True


def _sample_commands(want_crossing: bool) -> list[tuple[int, int]] | None:
    """Create a compact walk containing all four directions and the requested branch."""
    for _ in range(400):
        length = random.randint(12, 18)
        if want_crossing:
            # A five-command hook revisits its first landing cell, never the
            # origin.  Rotation and reflection are implicit in the random
            # perpendicular basis.
            e1 = random.choice(DIRECTIONS)
            perpendicular = [(dr, dc) for dr, dc in DIRECTIONS if dr * e1[0] + dc * e1[1] == 0]
            e2 = random.choice(perpendicular)
            commands = [e1, e1, e2, (-e1[0], -e1[1]), (-e2[0], -e2[1])]
            positions = _walk_positions(commands)
            if not _grow_new_steps(commands, positions, length):
                continue
        else:
            commands = []
            positions = [(0, 0)]
            if not _grow_new_steps(commands, positions, length):
                continue

        positions = _walk_positions(commands)
        landings = positions[1:]
        has_crossing = len(set(landings)) < len(landings)
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        if has_crossing != want_crossing:
            continue
        if (0, 0) in landings:
            continue
        if set(commands) != set(DIRECTIONS):
            continue
        if len(set(landings)) < 9:
            continue
        if not (3 <= max(rows) - min(rows) <= 8 and 3 <= max(cols) - min(cols) <= 8):
            continue
        return commands
    return None


def _spaced_from(cell: tuple[int, int], occupied: set[tuple[int, int]], gap: int) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > gap for r, c in occupied)


def _place_walk(
    shape: tuple[int, int], commands: list[tuple[int, int]]
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = shape
    rel = _walk_positions(commands)
    rmin, rmax = min(r for r, _ in rel), max(r for r, _ in rel)
    cmin, cmax = min(c for _, c in rel), max(c for _, c in rel)
    if h - 2 - rmax < 1 - rmin or w - 2 - cmax < 1 - cmin:
        return None
    start = (
        random.randint(1 - rmin, h - 2 - rmax),
        random.randint(1 - cmin, w - 2 - cmax),
    )
    footprint = {(start[0] + r, start[1] + c) for r, c in rel}
    return start, footprint


def _place_compass(
    shape: tuple[int, int], forbidden: set[tuple[int, int]]
) -> tuple[int, int] | None:
    h, w = shape
    choices = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            patch = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
            if all(_spaced_from(p, forbidden, 2) for p in patch):
                choices.append((r, c))
    return random.choice(choices) if choices else None


def _sample_snake_path(
    shape: tuple[int, int],
    length: int,
    forbidden: set[tuple[int, int]],
    want_tied_endpoints: bool,
) -> list[tuple[int, int]] | None:
    """Grow an induced 4-connected path away from the compass and output trail."""
    h, w = shape
    free = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _spaced_from((r, c), forbidden, 2)
    ]
    for _ in range(1200):
        if not free:
            return None
        path = [random.choice(free)]
        used = {path[0]}
        while len(path) < length:
            r, c = path[-1]
            candidates = []
            for q in _neighbors4(r, c):
                if not (1 <= q[0] < h - 1 and 1 <= q[1] < w - 1):
                    continue
                if q in used or not _spaced_from(q, forbidden, 2):
                    continue
                # No side contacts: the visible component remains a path and
                # never acquires an implicit branch or shortcut.
                if sum(n in used for n in _neighbors4(*q)) != 1:
                    continue
                candidates.append(q)
            if not candidates:
                break
            q = random.choice(candidates)
            path.append(q)
            used.add(q)
        if len(path) != length:
            continue
        tied = path[0][0] == path[-1][0]
        if tied != want_tied_endpoints:
            continue
        if tied and path[0][1] == path[-1][1]:
            continue
        rows = [r for r, _ in path]
        cols = [c for _, c in path]
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        turns = sum(
            (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            != (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(1, len(path) - 1)
        )
        if turns < 3:
            continue
        selected = min((path[0], path[-1]), key=lambda p: (p[0], p[1]))
        return path if path[0] == selected else list(reversed(path))
    return None


def _execute_colors(
    shape: tuple[int, int],
    bg: int,
    start: tuple[int, int],
    start_color: int,
    colors: list[int],
    color_to_direction: dict[int, tuple[int, int]],
) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    r, c = start
    out[r, c] = start_color
    for color in colors:
        dr, dc = color_to_direction[color]
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return None
        out[r, c] = color
    return out


def _semantic_counterexamples(
    shape: tuple[int, int],
    bg: int,
    start: tuple[int, int],
    start_color: int,
    path: list[tuple[int, int]],
    colors: list[int],
    mapping: dict[int, tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    """Reject examples that fail to distinguish common shortcut hypotheses."""
    wrong_orders = [list(reversed(colors))]
    row_major_colors = [color for _p, color in sorted(zip(path, colors), key=lambda x: x[0])]
    wrong_orders.append(row_major_colors)
    for order in wrong_orders:
        wrong = _execute_colors(shape, bg, start, start_color, order, mapping)
        if wrong is not None and np.array_equal(wrong, expected):
            return False

    # Every pairwise swap of compass meanings must visibly change the result.
    keys = list(mapping)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            altered = dict(mapping)
            altered[keys[i]], altered[keys[j]] = altered[keys[j]], altered[keys[i]]
            wrong = _execute_colors(shape, bg, start, start_color, colors, altered)
            if wrong is not None and np.array_equal(wrong, expected):
                return False
    return True


def _generate_attempt(profile: int) -> np.ndarray | None:
    want_tied_endpoints = profile in (0, 3)
    want_crossing = profile in (0, 2)

    h = random.randint(20, 28)
    w = random.randint(20, 28)
    shape = (h, w)
    bg = random.choice([c for c in range(10) if c != GRAY])
    available = [c for c in range(10) if c not in (bg, GRAY)]
    instruction_colors = random.sample(available, 4)
    start_color = random.choice([c for c in available if c not in instruction_colors])

    commands = _sample_commands(want_crossing)
    if commands is None:
        return None
    placed = _place_walk(shape, commands)
    if placed is None:
        return None
    start, trail_footprint = placed

    center = _place_compass(shape, trail_footprint)
    if center is None:
        return None
    legend_patch = {
        (r, c)
        for r in range(center[0] - 1, center[0] + 2)
        for c in range(center[1] - 1, center[1] + 2)
    }
    snake = _sample_snake_path(
        shape,
        len(commands),
        trail_footprint | legend_patch,
        want_tied_endpoints,
    )
    if snake is None:
        return None

    shuffled = instruction_colors[:]
    random.shuffle(shuffled)
    direction_to_color = dict(zip(DIRECTIONS, shuffled))
    color_to_direction = {color: direction for direction, color in direction_to_color.items()}
    colors = [direction_to_color[d] for d in commands]

    grid = np.full(shape, bg, dtype=int)
    grid[center] = GRAY
    for direction, color in direction_to_color.items():
        grid[center[0] + direction[0], center[1] + direction[1]] = color
    for p, color in zip(snake, colors):
        grid[p] = color
    grid[start] = start_color

    expected = _execute_colors(shape, bg, start, start_color, colors, color_to_direction)
    if expected is None:
        return None
    # Keep all four directions legible even when one crossing is overwritten.
    if not set(instruction_colors).issubset(set(map(int, expected.ravel()))):
        return None
    if int(expected[start]) != start_color:
        return None
    if np.array_equal(grid, expected):
        return None
    if not _semantic_counterexamples(
        shape, bg, start, start_color, snake, colors, color_to_direction, expected
    ):
        return None

    try:
        decoded = _read_scene(grid)
        if decoded[-1] != colors:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except (AssertionError, KeyError):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse, oracle-checked compass-snake examples up to 28x28."""
    global _GENERATION_PROFILE
    profile = _GENERATION_PROFILE % 4
    for _ in range(300):
        grid = _generate_attempt(profile)
        if grid is not None:
            _GENERATION_PROFILE += 1
            return grid
    raise RuntimeError("failed to generate a valid compass-instruction snake")
