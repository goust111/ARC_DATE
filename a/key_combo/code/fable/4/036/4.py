from __future__ import annotations

import random

import numpy as np


# Cardinal directions use the shared ARC convention: (row, column), U/R/D/L.
DIRS4: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

# A filled four-cell triangle, expressed relative to its tip (the sprite head).
TRIANGLE_OFFSETS: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((0, 0), (1, -1), (1, 0), (1, 1)),
    1: ((0, 0), (-1, -1), (0, -1), (1, -1)),
    2: ((0, 0), (-1, -1), (-1, 0), (-1, 1)),
    3: ((0, 0), (-1, 1), (0, 1), (1, 1)),
}


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """Return the cardinal half-line from a to b, or None when unaligned."""
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_sprite(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int], dict[tuple[int, int], int]]:
    """Identify the repeated-color triangle and all uniquely colored markers."""
    colors, counts = np.unique(grid, return_counts=True)
    foreground = {
        int(color): int(count)
        for color, count in zip(colors, counts)
        if int(color) != 0
    }
    sprite_colors = [color for color, count in foreground.items() if count > 1]
    assert len(sprite_colors) == 1, "expected one repeated-color triangle sprite"
    sprite_color = sprite_colors[0]

    sprite_cells = set(map(tuple, np.argwhere(grid == sprite_color)))
    assert len(sprite_cells) == 4, "expected a four-cell filled triangle"
    matches: list[tuple[int, tuple[int, int]]] = []
    for head in sprite_cells:
        hr, hc = map(int, head)
        for facing, offsets in TRIANGLE_OFFSETS.items():
            placed = {(hr + dr, hc + dc) for dr, dc in offsets}
            if placed == sprite_cells:
                matches.append((facing, (hr, hc)))
    assert len(matches) == 1, "triangle tip must define a unique initial facing"
    facing, head = matches[0]

    markers: dict[tuple[int, int], int] = {}
    for color, count in foreground.items():
        if color == sprite_color:
            continue
        assert count == 1, "marker colors must be distinct singleton pixels"
        coords = np.argwhere(grid == color)
        r, c = map(int, coords[0])
        markers[(r, c)] = color
    assert markers, "expected at least one marker"
    return sprite_color, facing, head, markers


def _trace_journey(
    head: tuple[int, int],
    facing: int,
    markers: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], int, int]]:
    """Return consumed (position, color, travel_direction) triples in order."""
    remaining = dict(markers)
    journey: list[tuple[tuple[int, int], int, int]] = []
    while remaining:
        allowed = (facing, (facing - 1) % 4, (facing + 1) % 4)
        rays: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for pos in remaining:
            direction = _direction(head, pos)
            if direction in allowed:
                rays.setdefault(int(direction), []).append((_distance(head, pos), pos))

        assert len(rays) == 1, (
            "from each sprite state exactly one non-backward direction must contain markers"
        )
        travel_direction = next(iter(rays))
        candidates = sorted(rays[travel_direction])
        distance, target = candidates[0]
        assert distance > 0
        assert sum(d == distance for d, _ in candidates) == 1, "nearest target must be unique"

        color = remaining.pop(target)
        journey.append((target, color, travel_direction))
        head = target
        facing = travel_direction
    return journey


def _paint_stripes(
    shape: tuple[int, int], journey: list[tuple[tuple[int, int], int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (r, c), color, travel_direction in journey:
        if travel_direction in (1, 3):
            out[r, :] = color
        else:
            out[:, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Replay the triangle sprite's journey and replace consumed markers with ordered full-grid stripes.

    Input structure:
    - A black background contains one monochrome filled triangular sprite whose tip
      gives its initial facing, plus several differently colored singleton markers.
    - From every journey state, exactly one of the forward, left, and right cardinal
      half-lines contains any remaining marker; the directly backward ray is ignored.

    Rule:
    - Repeatedly take the unique non-backward ray containing markers, move to its
      nearest marker, consume it, and face that travel direction.
    - On a fresh black grid, process markers in consumption order: horizontal travel
      paints the marker's entire row and vertical travel paints its entire column.
      Each later stripe overwrites earlier stripes at all intersections.

    Key insight:
    - A marker's stripe axis and its intersection priority are not intrinsic to its
      color or position; both are recovered only by replaying the hidden journey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a two-dimensional grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"
    assert np.any(g == 0), "expected a black background"

    _sprite_color, facing, head, markers = _find_sprite(g)
    journey = _trace_journey(head, facing, markers)
    return _paint_stripes(g.shape, journey)


# ------------------------------ generator ------------------------------

_GENERATION_ATTEMPTS = 300
_GEN_COUNTER = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _transform_point(
    point: tuple[int, int], quarter_turns: int, reflected: bool
) -> tuple[int, int]:
    """Apply a column reflection, then clockwise quarter turns about the origin."""
    r, c = point
    if reflected:
        c = -c
    for _ in range(quarter_turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _build_path(
    marker_count: int, quarter_turns: int, reflected: bool
) -> tuple[list[tuple[int, int]], int]:
    """Build a sparse spiral whose last base marker starts behind the sprite."""
    a = random.randint(3, 6)
    b = random.randint(3, 6)
    c = random.randint(3, 6)
    d = random.randint(3, 6)

    # p5 is directly behind p0's initial right-facing direction, but becomes
    # reachable only after the four intervening turns.
    points = [(0, 0), (a, 0), (a, b), (-c, b), (-c, -d), (0, -d)]
    if marker_count >= 6:
        e = random.randint(3, 6)
        points.append((0, -d - e))
    if marker_count >= 7:
        f_choices = [value for value in range(3, 7) if value != a]
        f = random.choice(f_choices)
        points.append((f, points[-1][1]))

    transformed = [
        _transform_point(point, quarter_turns, reflected) for point in points
    ]
    facing_vector = _transform_point((0, 1), quarter_turns, reflected)
    initial_facing = _direction((0, 0), facing_vector)
    assert initial_facing is not None
    return transformed, initial_facing


def _translate_case(
    points: list[tuple[int, int]], initial_facing: int
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    directions = [_direction(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if any(direction is None for direction in directions):
        return None
    state_facings = [initial_facing] + [int(direction) for direction in directions]

    occupied: set[tuple[int, int]] = set(points)
    for head, facing in zip(points, state_facings):
        occupied.update(
            (head[0] + dr, head[1] + dc) for dr, dc in TRIANGLE_OFFSETS[facing]
        )
    min_r = min(r for r, _ in occupied)
    max_r = max(r for r, _ in occupied)
    min_c = min(c for _, c in occupied)
    max_c = max(c for _, c in occupied)
    margins = [random.randint(2, 4) for _ in range(4)]
    height = max_r - min_r + 1 + margins[0] + margins[1]
    width = max_c - min_c + 1 + margins[2] + margins[3]
    if height > 30 or width > 30:
        return None

    shift = (margins[0] - min_r, margins[2] - min_c)
    shifted = [(r + shift[0], c + shift[1]) for r, c in points]
    return shifted[0], shifted[1:], (height, width)


def _semantic_case_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    sprite_color: int,
    initial_head: tuple[int, int],
    initial_facing: int,
    marker_positions: list[tuple[int, int]],
    marker_colors: list[int],
) -> bool:
    """Check the stated rule's dependencies, not merely solver round-tripping."""
    if grid.ndim != 2 or max(grid.shape) > 30 or grid.shape != expected.shape:
        return False
    if len(set(marker_colors + [sprite_color])) != len(marker_colors) + 1:
        return False

    planned_directions = [
        _direction(initial_head if i == 0 else marker_positions[i - 1], pos)
        for i, pos in enumerate(marker_positions)
    ]
    if any(direction is None for direction in planned_directions):
        return False
    directions = [int(direction) for direction in planned_directions]
    if not ({0, 2} & set(directions) and {1, 3} & set(directions)):
        return False
    if directions[0] in (initial_facing, (initial_facing + 2) % 4):
        return False  # first move must visibly use a side choice
    if _direction(initial_head, marker_positions[4]) != (initial_facing + 2) % 4:
        return False  # an initially backward marker must really be deferred

    horizontal_rows = [
        pos[0] for pos, direction in zip(marker_positions, directions) if direction in (1, 3)
    ]
    vertical_cols = [
        pos[1] for pos, direction in zip(marker_positions, directions) if direction in (0, 2)
    ]
    if len(horizontal_rows) != len(set(horizontal_rows)):
        return False
    if len(vertical_cols) != len(set(vertical_cols)):
        return False

    markers = dict(zip(marker_positions, marker_colors))
    try:
        traced = _trace_journey(initial_head, initial_facing, markers)
    except AssertionError:
        return False
    if [item[0] for item in traced] != marker_positions:
        return False
    if [item[2] for item in traced] != directions:
        return False
    if not all(color in set(map(int, expected.flat)) for color in marker_colors):
        return False  # no stripe may disappear completely through degeneracy

    reversed_paint = _paint_stripes(grid.shape, list(reversed(traced)))
    if np.array_equal(reversed_paint, expected):
        return False  # intersection colors must make chronological overpaint necessary
    wrong_axes = np.zeros(grid.shape, dtype=int)
    for (r, c), color, direction in traced:
        if direction in (1, 3):
            wrong_axes[:, c] = color
        else:
            wrong_axes[r, :] = color
    if np.array_equal(wrong_axes, expected):
        return False

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(actual, expected) and sprite_color not in set(map(int, expected.flat))


def _generate_attempt(variant: int) -> tuple[np.ndarray, np.ndarray] | None:
    marker_count = 5 + variant % 3
    quarter_turns = variant % 4
    # Cycle all four initial facings; flip chirality on each four-case block.
    reflected = bool((variant // 4) % 2)
    points, initial_facing = _build_path(marker_count, quarter_turns, reflected)
    placed = _translate_case(points, initial_facing)
    if placed is None:
        return None
    initial_head, marker_positions, shape = placed

    palette = random.sample(range(1, 10), marker_count + 1)
    sprite_color = palette[0]
    marker_colors = palette[1:]
    grid = np.zeros(shape, dtype=int)
    for dr, dc in TRIANGLE_OFFSETS[initial_facing]:
        grid[initial_head[0] + dr, initial_head[1] + dc] = sprite_color
    for pos, color in zip(marker_positions, marker_colors):
        if grid[pos] != 0:
            return None
        grid[pos] = color

    directions = [
        _direction(initial_head if i == 0 else marker_positions[i - 1], pos)
        for i, pos in enumerate(marker_positions)
    ]
    journey = [
        (pos, color, int(direction))
        for pos, color, direction in zip(marker_positions, marker_colors, directions)
        if direction is not None
    ]
    if len(journey) != marker_count:
        return None
    expected = _paint_stripes(shape, journey)
    if not _semantic_case_valid(
        grid,
        expected,
        sprite_color,
        initial_head,
        initial_facing,
        marker_positions,
        marker_colors,
    ):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked journey-stripe inputs of at most 30x30."""
    global _GEN_COUNTER
    variant = _GEN_COUNTER
    for _ in range(_GENERATION_ATTEMPTS):
        case = _generate_attempt(variant)
        if case is None:
            continue
        grid, expected = case
        output_key = (tuple(map(int, expected.shape)), expected.tobytes())
        if output_key in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(output_key)
        _GEN_COUNTER += 1
        return grid
    raise RuntimeError("failed to generate a valid journey-stripe input")
