"""ARC task: transfer an incident bar's length through a colored block chain."""

from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
ARC_COLORS = tuple(range(1, 10))
MIN_LINES = 2
MAX_LINES = 3
MAX_GENERATION_ATTEMPTS = 400


def _find_wall(grid: np.ndarray) -> tuple[str, int, int]:
    """Return (orientation, index, color) for the unique full-span wall."""
    candidates: list[tuple[str, int, int]] = []
    height, width = grid.shape
    for r in range(height):
        color = int(grid[r, 0])
        if color != BACKGROUND and bool(np.all(grid[r, :] == color)):
            candidates.append(("horizontal", r, color))
    for c in range(width):
        color = int(grid[0, c])
        if color != BACKGROUND and bool(np.all(grid[:, c] == color)):
            candidates.append(("vertical", c, color))
    assert len(candidates) == 1, "expected exactly one monochrome full-span wall"
    return candidates[0]


def _runs(values: list[int]) -> list[tuple[int, ...]]:
    """Split sorted integer positions into maximal consecutive runs."""
    if not values:
        return []
    result: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == result[-1][-1] + 1:
            result[-1].append(value)
        else:
            result.append([value])
    return [tuple(run) for run in result]


def _coord(
    wall_orientation: str, wall_index: int, line_index: int, side: int, distance: int
) -> tuple[int, int]:
    """Map wall-relative (line, side, distance) to an absolute (row, col)."""
    assert side in (-1, 1) and distance >= 0
    if wall_orientation == "vertical":
        return line_index, wall_index + side * distance
    return wall_index + side * distance, line_index


def _line_values(
    grid: np.ndarray, orientation: str, wall_index: int, line_index: int
) -> np.ndarray:
    """Read one complete work line perpendicular to the wall."""
    if orientation == "vertical":
        return grid[line_index, :]
    return grid[:, line_index]


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer each incident bar's length through a block chain and eject that many far-end blocks to the wall.

    Input structure:
    - A black grid contains one monochrome full-span horizontal or vertical wall.
    - On each of several perpendicular work lines, all on one side of the wall,
      a nearer multicolored contiguous block chain and a farther monochrome
      incident bar are separated by black space; the chain is longer than the bar.
    - Incident-bar lengths are distinct values from 1 through 3, and neither a
      chain nor an incident bar uses the wall color.

    Rule:
    - Slide each incident bar toward its chain until it stops immediately against
      the chain head, without moving the surviving middle of the chain.
    - If the incident length is L, remove the L chain blocks nearest the wall and
      slide them into the L cells against the wall. Preserve their wall-to-outside
      color order, so the block formerly nearest the wall remains nearest the wall.
    - Transform every work line independently and preserve the wall.

    Key insight:
    - Collision displacement passes through the stationary chain: incident length
      is a visible count code for equal ejection at the opposite end, while the
      ejected block colors retain their order rather than the whole chain moving.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2-D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    orientation, wall_index, wall_color = _find_wall(g)
    out = g.copy()
    perpendicular_size = g.shape[0] if orientation == "vertical" else g.shape[1]
    lengths: list[int] = []

    for line_index in range(perpendicular_size):
        line = _line_values(g, orientation, wall_index, line_index)
        occupied = [
            position
            for position, value in enumerate(line)
            if position != wall_index and int(value) != BACKGROUND
        ]
        if not occupied:
            continue

        assert all(int(line[p]) != wall_color for p in occupied), (
            "the wall color must be reserved for the full-span wall"
        )
        sides = {-1 if position < wall_index else 1 for position in occupied}
        assert len(sides) == 1, "one work line must lie wholly on one side of the wall"
        side = sides.pop()

        by_distance = sorted(abs(position - wall_index) for position in occupied)
        runs = _runs(by_distance)
        assert len(runs) == 2, "each work line needs one chain and one separated incident bar"
        chain_distances, incident_distances = runs
        assert chain_distances[-1] < incident_distances[0], (
            "the chain must be nearer the wall than the incident bar"
        )

        chain_colors = [
            int(g[_coord(orientation, wall_index, line_index, side, distance)])
            for distance in chain_distances
        ]
        incident_colors = [
            int(g[_coord(orientation, wall_index, line_index, side, distance)])
            for distance in incident_distances
        ]
        incident_color = incident_colors[0]
        assert all(color == incident_color for color in incident_colors), (
            "the incident bar must be monochrome"
        )

        length = len(incident_distances)
        assert 1 <= length <= 3, "incident length L must be between 1 and 3"
        assert len(chain_distances) > length, "the block chain must be longer than L"
        assert incident_color not in chain_colors, (
            "chain-block colors must differ from their incident-bar color"
        )
        assert chain_distances == tuple(
            range(chain_distances[0], chain_distances[0] + len(chain_distances))
        )
        assert incident_distances == tuple(
            range(incident_distances[0], incident_distances[0] + length)
        )
        lengths.append(length)

        # Clear the old incident bar and the transmitted/ejected chain tail.
        for distance in incident_distances:
            out[_coord(orientation, wall_index, line_index, side, distance)] = BACKGROUND
        ejected_colors = chain_colors[:length]
        for distance in chain_distances[:length]:
            out[_coord(orientation, wall_index, line_index, side, distance)] = BACKGROUND

        # The incident bar stops against the chain head on its outside end.
        first_incident_destination = chain_distances[-1] + 1
        for offset in range(length):
            destination = _coord(
                orientation,
                wall_index,
                line_index,
                side,
                first_incident_destination + offset,
            )
            assert int(out[destination]) == BACKGROUND, "incident destination must be empty"
            out[destination] = incident_color

        # Distance one is adjacent to the wall; order is copied, never color-sorted.
        for distance, color in enumerate(ejected_colors, 1):
            destination = _coord(orientation, wall_index, line_index, side, distance)
            assert int(out[destination]) == BACKGROUND, "wall-side stack destination must be empty"
            out[destination] = color

    assert MIN_LINES <= len(lengths) <= MAX_LINES, "expected two or three independent work lines"
    assert len(lengths) == len(set(lengths)), "incident-bar lengths must differ across work lines"
    return out


def _spaced_indices(size: int, count: int) -> tuple[int, ...]:
    """Choose work lines with at least two empty separator lines between them."""
    choices = [
        indices
        for indices in itertools.combinations(range(2, size - 2), count)
        if all(b - a >= 3 for a, b in zip(indices, indices[1:]))
    ]
    assert choices, "canvas is too narrow for separated work lines"
    return random.choice(choices)


def _geometry_options(length: int, capacity: int) -> list[tuple[int, int, int]]:
    """Return (near-chain distance, chain length, black incident gap) choices."""
    return [
        (near, chain_length, gap)
        for near in range(length + 2, length + 5)
        for chain_length in range(length + 2, length + 6)
        for gap in range(2, 5)
        if near + chain_length + gap + length - 1 <= capacity
    ]


def _side_assignments(
    lengths: tuple[int, ...], negative_capacity: int, positive_capacity: int
) -> list[tuple[int, ...]]:
    """List feasible assignments that visibly exercise both sides of the wall."""
    result = []
    for sides in itertools.product((-1, 1), repeat=len(lengths)):
        if set(sides) != {-1, 1}:
            continue
        if all(
            _geometry_options(
                length,
                negative_capacity if side == -1 else positive_capacity,
            )
            for length, side in zip(lengths, sides)
        ):
            result.append(sides)
    return result


def _chain_colors(pool: tuple[int, ...], length: int, chain_length: int) -> tuple[int, ...]:
    """Make an ordered, visibly segmented chain whose first L colors are distinct."""
    colors = list(random.sample(pool, length))
    while len(colors) < chain_length:
        choices = [color for color in pool if color != colors[-1]]
        colors.append(random.choice(choices))
    assert len(set(colors)) >= 2
    return tuple(colors)


def _independent_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Construct the output directly from generation specs, independently of parsing."""
    expected = grid.copy()
    for spec in specs:
        orientation = spec["orientation"]
        wall = spec["wall"]
        line = spec["line"]
        side = spec["side"]
        length = spec["length"]
        near = spec["near"]
        chain = spec["chain"]
        incident_start = spec["incident_start"]
        incident_color = spec["incident_color"]

        for distance in range(incident_start, incident_start + length):
            expected[_coord(orientation, wall, line, side, distance)] = BACKGROUND
        for distance in range(near, near + length):
            expected[_coord(orientation, wall, line, side, distance)] = BACKGROUND
        for offset in range(length):
            expected[
                _coord(orientation, wall, line, side, near + len(chain) + offset)
            ] = incident_color
        for distance, color in enumerate(chain[:length], 1):
            expected[_coord(orientation, wall, line, side, distance)] = color
    return expected


def _shortcut_outputs(grid: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, ...]:
    """Build common wrong-rule outputs used to reject degenerate examples."""
    stop_only = grid.copy()
    fixed_one = grid.copy()
    reversed_order = _independent_expected(grid, specs)
    whole_chain_push = grid.copy()

    for spec in specs:
        orientation = spec["orientation"]
        wall = spec["wall"]
        line = spec["line"]
        side = spec["side"]
        length = spec["length"]
        near = spec["near"]
        chain = spec["chain"]
        incident_start = spec["incident_start"]
        incident_color = spec["incident_color"]

        old_incident = range(incident_start, incident_start + length)
        final_incident = range(near + len(chain), near + len(chain) + length)
        for canvas in (stop_only, fixed_one, whole_chain_push):
            for distance in old_incident:
                canvas[_coord(orientation, wall, line, side, distance)] = BACKGROUND

        for distance in final_incident:
            stop_only[_coord(orientation, wall, line, side, distance)] = incident_color

        # Wrong count rule: always eject exactly one, though the bar itself still slides.
        fixed_one[_coord(orientation, wall, line, side, near)] = BACKGROUND
        fixed_one[_coord(orientation, wall, line, side, 1)] = chain[0]
        for distance in final_incident:
            fixed_one[_coord(orientation, wall, line, side, distance)] = incident_color

        # Wrong order rule: reverse the ejected sequence at the wall.
        if length >= 2:
            for distance, color in enumerate(reversed(chain[:length]), 1):
                reversed_order[_coord(orientation, wall, line, side, distance)] = color

        # Wrong collision rule: shift the complete chain wallward by L.
        for offset in range(len(chain)):
            whole_chain_push[
                _coord(orientation, wall, line, side, near + offset)
            ] = BACKGROUND
        for offset, color in enumerate(chain):
            whole_chain_push[
                _coord(orientation, wall, line, side, near - length + offset)
            ] = color
        shifted_head = near - length + len(chain) - 1
        for offset in range(length):
            whole_chain_push[
                _coord(orientation, wall, line, side, shifted_head + 1 + offset)
            ] = incident_color

    return stop_only, fixed_one, reversed_order, whole_chain_push


def _semantically_valid(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    """Check uniqueness, every semantic dependency, and tempting counter-rules."""
    if grid.ndim != 2 or max(grid.shape) > 30 or np.array_equal(grid, expected):
        return False
    if len(specs) not in (2, 3):
        return False
    lengths = [int(spec["length"]) for spec in specs]
    if len(lengths) != len(set(lengths)) or not any(length >= 2 for length in lengths):
        return False
    if {int(spec["side"]) for spec in specs} != {-1, 1}:
        return False

    for spec in specs:
        length = int(spec["length"])
        near = int(spec["near"])
        chain = tuple(spec["chain"])
        if len(chain) <= length or len(chain) - length < 2:
            return False
        if near < length + 2 or len(set(chain[:length])) != length:
            return False
        if int(spec["gap"]) < 2:
            return False
        # Each transmitted block genuinely leaves its old cell and has a distinct destination.
        for offset in range(length):
            if near + offset == 1 + offset:
                return False

    try:
        solved = main(grid)
    except (AssertionError, IndexError):
        return False
    if not np.array_equal(solved, expected):
        return False
    return all(not np.array_equal(wrong, expected) for wrong in _shortcut_outputs(grid, specs))


def _generate_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    motion_size = random.randint(26, 30)
    line_size = random.randint(12, 22)
    shape = (line_size, motion_size) if orientation == "vertical" else (motion_size, line_size)
    wall = random.randint(10, motion_size - 11)
    negative_capacity = wall
    positive_capacity = motion_size - 1 - wall

    line_count = random.choices((2, 3), weights=(2, 3), k=1)[0]
    lengths = tuple(random.sample((1, 2, 3), line_count))
    feasible_sides = _side_assignments(lengths, negative_capacity, positive_capacity)
    if not feasible_sides:
        return None
    sides = random.choice(feasible_sides)
    line_indices = _spaced_indices(line_size, line_count)

    role_colors = random.sample(ARC_COLORS, line_count + 1)
    wall_color = role_colors[0]
    incident_colors = tuple(role_colors[1:])
    chain_pool = tuple(
        color for color in ARC_COLORS if color not in {wall_color, *incident_colors}
    )

    grid = np.zeros(shape, dtype=int)
    if orientation == "vertical":
        grid[:, wall] = wall_color
    else:
        grid[wall, :] = wall_color

    specs: list[dict] = []
    for line, side, length, incident_color in zip(
        line_indices, sides, lengths, incident_colors
    ):
        capacity = negative_capacity if side == -1 else positive_capacity
        near, chain_length, gap = random.choice(_geometry_options(length, capacity))
        chain = _chain_colors(chain_pool, length, chain_length)
        incident_start = near + chain_length + gap

        for offset, color in enumerate(chain):
            grid[_coord(orientation, wall, line, side, near + offset)] = color
        for offset in range(length):
            grid[_coord(orientation, wall, line, side, incident_start + offset)] = incident_color

        specs.append(
            {
                "orientation": orientation,
                "wall": wall,
                "line": line,
                "side": side,
                "length": length,
                "near": near,
                "chain": chain,
                "gap": gap,
                "incident_start": incident_start,
                "incident_color": incident_color,
            }
        )

    expected = _independent_expected(grid, specs)
    return grid if _semantically_valid(grid, expected, specs) else None


def generate_input() -> np.ndarray:
    """Generate a varied instance whose visible structure forces every rule step."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid through-chain collision instance")
