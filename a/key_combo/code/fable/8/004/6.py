from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_ATTEMPTS = 600
PATH_ATTEMPTS = 160


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _on_edge(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = map(int, cell)
    h, w = map(int, shape)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _add(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0] + direction[0]), int(cell[1] + direction[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _perp_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    idx = DIR_ORDER.index(direction)
    return [DIR_ORDER[(idx - 1) % 4], DIR_ORDER[(idx + 1) % 4]]


def _edge_inward_direction(cell: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, cell)
    h, w = map(int, shape)
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    if c == w - 1:
        return (0, -1)
    raise AssertionError("particle must lie on the grid edge")


def _same_color_degree(grid: np.ndarray, cell: tuple[int, int], color: int) -> int:
    r, c = map(int, cell)
    count = 0
    for dr, dc in DIR_ORDER:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(color):
            count += 1
    return count


def _find_particles(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    candidates: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not _on_edge((r, c), grid.shape):
                continue
            color = int(grid[r, c])
            if color == int(bg):
                continue
            if _same_color_degree(grid, (r, c), color) == 0:
                candidates.add((r, c))
    particles = sorted(candidates)
    assert len(particles) >= 2, "expected at least two edge particles"
    return particles


def _lane_cell_ok(
    grid: np.ndarray,
    cell: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    bg: int,
) -> bool:
    r, c = map(int, cell)
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != int(bg):
        return False
    if _on_edge((r, c), grid.shape):
        return False
    dr, dc = direction
    if dr != 0:
        return (
            _in_bounds(grid.shape, r, c - 1)
            and _in_bounds(grid.shape, r, c + 1)
            and int(grid[r, c - 1]) == int(color)
            and int(grid[r, c + 1]) == int(color)
        )
    return (
        _in_bounds(grid.shape, r - 1, c)
        and _in_bounds(grid.shape, r + 1, c)
        and int(grid[r - 1, c]) == int(color)
        and int(grid[r + 1, c]) == int(color)
    )


def _trace_particle(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    color = int(grid[start])
    direction = _edge_inward_direction(start, grid.shape)
    path = [start]
    current = start
    seen = {(start, direction)}

    while True:
        ahead = _add(current, direction)
        if not _in_bounds(grid.shape, *ahead):
            break

        if int(grid[ahead]) == int(bg):
            if _on_edge(ahead, grid.shape):
                path.append(ahead)
                break
            if _lane_cell_ok(grid, ahead, direction, color, bg):
                current = ahead
                path.append(current)
                state = (current, direction)
                assert state not in seen, "looping corridor is invalid"
                seen.add(state)
                continue
            break

        turns = []
        for nd in _perp_dirs(direction):
            turn_cell = _add(ahead, nd)
            if _lane_cell_ok(grid, turn_cell, nd, color, bg):
                turns.append((nd, turn_cell))
        assert len(turns) <= 1, "collision produced multiple legal turns"
        if not turns:
            break

        direction, current = turns[0]
        path.append(current)
        state = (current, direction)
        assert state not in seen, "looping corridor is invalid"
        seen.add(state)

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Trace every edge particle through same-color flanked corridors and keep only the uniquely longest complete path.

    Input structure:
    - The background contains several colored corridor systems made of parallel
      one-cell-wide bars; the background lane between two same-colored bars is a
      traversable corridor.
    - Multiple single-pixel particles sit on the grid edge, each in the color
      of the corridor system it enters.
    - Corridor cells themselves are background; only the flanking bars and the
      edge particles are explicitly colored in the input.

    Rule:
    - For each edge particle, move inward along its corridor. A forward step is
      valid only when the next background cell is flanked on both sides by bars
      of the particle's color.
    - If the forward cell is colored, turn into the unique perpendicular
      background corridor whose two flanking bars have that same color; if no
      such perpendicular corridor exists, or the particle reaches the grid
      edge, the trajectory stops.
    - Compare the number of visited cells for all particles. Keep only the
      uniquely longest trajectory by painting every visited cell in that
      particle's color, and erase every losing particle start back to
      background while leaving the corridor bars unchanged.

    Key insight:
    - The compared objects are not visible paths in the input; each candidate
      must first be reconstructed by simulating collision-triggered turns inside
      an implicit background corridor network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    particles = _find_particles(g, bg)
    traces = [_trace_particle(g, start) for start in particles]
    lengths = [len(path) for path in traces]
    best = max(lengths)
    winners = [i for i, length in enumerate(lengths) if length == best]
    assert len(winners) == 1, "expected exactly one longest trajectory"
    winner_i = winners[0]
    winner_color = int(g[particles[winner_i]])

    out = g.copy()
    for start in particles:
        out[start] = int(bg)
    for r, c in traces[winner_i]:
        out[r, c] = winner_color
    return out


def _path_cell_ok(
    cell: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    *,
    allow_edge_terminal: bool = False,
) -> bool:
    r, c = map(int, cell)
    h, w = map(int, shape)
    if not _in_bounds(shape, r, c):
        return False
    if allow_edge_terminal and _on_edge(cell, shape):
        return True
    if _on_edge(cell, shape):
        return False

    dr, dc = direction
    if dr != 0:
        return 1 <= r <= h - 2 and 2 <= c <= w - 3
    return 2 <= r <= h - 3 and 1 <= c <= w - 2


def _can_append(
    path: list[tuple[int, int]],
    cell: tuple[int, int],
    *,
    allow_edge_terminal: bool = False,
) -> bool:
    if cell in path:
        return False
    protected = {len(path) - 1}
    if len(path) >= 2:
        protected.add(len(path) - 2)
    for i, other in enumerate(path):
        if i in protected:
            continue
        if max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1:
            return False
    if allow_edge_terminal:
        return True
    return True


def _rand_start(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        return (0, random.randint(2, w - 3))
    if side == "bottom":
        return (h - 1, random.randint(2, w - 3))
    if side == "left":
        return (random.randint(2, h - 3), 0)
    if side == "right":
        return (random.randint(2, h - 3), w - 1)
    raise AssertionError("unknown side")


def _feasible_turn_lengths(
    path: list[tuple[int, int]],
    direction: tuple[int, int],
    next_direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[int]:
    temp = list(path)
    lengths: list[int] = []
    steps = 0
    while True:
        nxt = _add(temp[-1], direction)
        if not _path_cell_ok(nxt, direction, shape) or not _can_append(temp, nxt):
            break
        temp.append(nxt)
        steps += 1
        blocker = _add(temp[-1], direction)
        turn_cell = _add(blocker, next_direction)
        if (
            steps >= 2
            and _in_bounds(shape, *blocker)
            and not _on_edge(blocker, shape)
            and _path_cell_ok(turn_cell, next_direction, shape)
            and _can_append(temp, turn_cell)
        ):
            lengths.append(steps)
    return lengths


def _feasible_dead_lengths(
    path: list[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[int]:
    temp = list(path)
    lengths: list[int] = []
    steps = 0
    while True:
        nxt = _add(temp[-1], direction)
        if not _path_cell_ok(nxt, direction, shape) or not _can_append(temp, nxt):
            break
        temp.append(nxt)
        steps += 1
        blocker = _add(temp[-1], direction)
        if steps >= 2 and _in_bounds(shape, *blocker) and not _on_edge(blocker, shape):
            lengths.append(steps)
    return lengths


def _extend_with_steps(
    path: list[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]]:
    out = list(path)
    for _ in range(int(steps)):
        out.append(_add(out[-1], direction))
    return out


def _extend_to_edge(
    path: list[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    temp = list(path)
    steps = 0
    while True:
        nxt = _add(temp[-1], direction)
        if not _in_bounds(shape, *nxt):
            return None
        if _on_edge(nxt, shape):
            if not _can_append(temp, nxt, allow_edge_terminal=True):
                return None
            temp.append(nxt)
            return temp if steps + 1 >= 2 else None
        if not _path_cell_ok(nxt, direction, shape) or not _can_append(temp, nxt):
            return None
        temp.append(nxt)
        steps += 1


def _build_random_path(
    shape: tuple[int, int],
    turn_count: int,
    final_mode: str,
) -> list[tuple[int, int]] | None:
    side = random.choice(("top", "bottom", "left", "right"))
    path = [_rand_start(side, shape)]
    direction = _edge_inward_direction(path[0], shape)

    for _ in range(int(turn_count)):
        options: list[tuple[tuple[int, int], list[int]]] = []
        for next_direction in _perp_dirs(direction):
            lengths = _feasible_turn_lengths(path, direction, next_direction, shape)
            if lengths:
                options.append((next_direction, lengths))
        if not options:
            return None
        next_direction, lengths = random.choice(options)
        steps = random.choice(lengths)
        path = _extend_with_steps(path, direction, steps)
        path.append(_add(_add(path[-1], direction), next_direction))
        direction = next_direction

    if final_mode == "exit":
        return _extend_to_edge(path, direction, shape)

    lengths = _feasible_dead_lengths(path, direction, shape)
    if not lengths:
        return None
    return _extend_with_steps(path, direction, random.choice(lengths))


def _flank_cells(
    cell: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = map(int, cell)
    dr, dc = direction
    if dr != 0:
        return (r, c - 1), (r, c + 1)
    return (r - 1, c), (r + 1, c)


def _render_path_features(
    path: list[tuple[int, int]],
    color: int,
    shape: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    features: dict[tuple[int, int], int] = {path[0]: int(color)}
    path_cells = set(path)
    directions = [_edge_inward_direction(path[0], shape)]
    for i in range(1, len(path)):
        delta = _sub(path[i], path[i - 1])
        prev_direction = directions[-1]
        if delta == prev_direction:
            directions.append(prev_direction)
            continue
        new_direction = (delta[0] - prev_direction[0], delta[1] - prev_direction[1])
        if new_direction not in _perp_dirs(prev_direction):
            return None
        directions.append(new_direction)

    for i in range(1, len(path)):
        if _on_edge(path[i], shape):
            continue
        for flank in _flank_cells(path[i], directions[i]):
            if flank in path_cells:
                return None
            if not _in_bounds(shape, *flank):
                return None
            features[flank] = int(color)

    for i in range(1, len(path) - 1):
        delta = _sub(path[i + 1], path[i])
        if delta == directions[i]:
            continue
        blocker = _add(path[i], directions[i])
        if blocker in path_cells:
            return None
        if not _in_bounds(shape, *blocker):
            return None
        features[blocker] = int(color)

    if not _on_edge(path[-1], shape):
        blocker = _add(path[-1], directions[-1])
        if blocker in path_cells or not _in_bounds(shape, *blocker):
            return None
        features[blocker] = int(color)

    for cell in features:
        if _on_edge(cell, shape) and cell != path[0]:
            return None
    return features


def _expanded_footprint(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _place_paths(shape: tuple[int, int]) -> list[tuple[int, list[tuple[int, int]], dict[tuple[int, int], int]]] | None:
    path_count = random.randint(2, 4)
    colors = random.sample(PALETTE, path_count)

    turn_choices = [random.choices([0, 1, 2, 3], weights=[1, 4, 3, 1], k=1)[0] for _ in range(path_count)]
    if max(turn_choices) == 0:
        turn_choices[random.randrange(path_count)] = random.choice([1, 2])

    exit_count = random.randint(0, min(2, path_count))
    final_modes = ["exit"] * exit_count + ["dead"] * (path_count - exit_count)
    random.shuffle(final_modes)

    specs = list(zip(colors, turn_choices, final_modes))
    random.shuffle(specs)

    placed: list[tuple[int, list[tuple[int, int]], dict[tuple[int, int], int]]] = []
    forbidden: set[tuple[int, int]] = set()
    for color, turn_count, final_mode in specs:
        candidate = None
        for _ in range(PATH_ATTEMPTS):
            path = _build_random_path(shape, turn_count, final_mode)
            if path is None or len(path) < 5:
                continue
            features = _render_path_features(path, color, shape)
            if features is None:
                continue
            footprint = set(path) | set(features)
            if footprint & forbidden:
                continue
            candidate = (color, path, features)
            forbidden |= _expanded_footprint(footprint, shape)
            break
        if candidate is None:
            return None
        placed.append(candidate)
    return placed


def _oracle_output(
    grid: np.ndarray,
    particles: list[tuple[int, int]],
    traces: dict[tuple[int, int], list[tuple[int, int]]],
) -> np.ndarray:
    lengths = {start: len(path) for start, path in traces.items()}
    best = max(lengths.values())
    winners = [start for start, length in lengths.items() if length == best]
    if len(winners) != 1:
        raise AssertionError("longest path must be unique")
    winner = winners[0]
    color = int(grid[winner])

    out = grid.copy()
    for start in particles:
        out[start] = BG
    for r, c in traces[winner]:
        out[r, c] = color
    return out


def _validate_generated_grid(
    grid: np.ndarray,
    intended: list[tuple[int, list[tuple[int, int]], dict[tuple[int, int], int]]],
) -> bool:
    try:
        bg = _get_bg(grid)
        particles = _find_particles(grid, bg)
        intended_paths = {path[0]: path for _color, path, _features in intended}
        if set(particles) != set(intended_paths):
            return False

        traced = {start: _trace_particle(grid, start) for start in particles}
        if any(traced[start] != intended_paths[start] for start in particles):
            return False

        lengths = sorted(len(path) for path in traced.values())
        if len(lengths) != len(set(lengths)):
            return False

        turn_counts = []
        for start, path in traced.items():
            direction = _edge_inward_direction(start, grid.shape)
            count = 0
            for i in range(len(path) - 1):
                delta = _sub(path[i + 1], path[i])
                if delta == direction:
                    continue
                new_direction = (delta[0] - direction[0], delta[1] - direction[1])
                if new_direction not in _perp_dirs(direction):
                    return False
                direction = new_direction
                count += 1
            turn_counts.append(count)
        if max(turn_counts) == 0:
            return False

        expected = _oracle_output(grid, particles, traced)
        actual = main(grid)
        return np.array_equal(actual, expected)
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 29), random.randint(18, 29))
    placed = _place_paths(shape)
    if placed is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    for _color, _path, features in placed:
        for (r, c), value in features.items():
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = int(value)

    if _validate_generated_grid(grid, placed):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid longest-corridor input")
