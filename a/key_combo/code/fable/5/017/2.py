"""ARC puzzle: multiplicity-defined portals and walls on a straight beam."""

from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BACKGROUND = 0
START_COLOR = 3
COLOR_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600

# Cycling only the two important branches makes any three consecutive generated
# examples include both a singleton-wall ending and an edge ending.  Everything
# geometric and all color assignments remain random.
_generation_index = 0


def _on_boundary(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    """Whether an (r, c) cell lies on any grid boundary."""
    h, w = shape
    r, c = cell
    return r in (0, h - 1) or c in (0, w - 1)


def _parse_scene(
    grid: np.ndarray,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    dict[tuple[int, int], tuple[tuple[int, int], int]],
    set[tuple[int, int]],
]:
    """Find the unique start bar and classify all other colors by count."""
    assert grid.ndim == 2 and grid.size > 0, "expected a non-empty 2D grid"
    assert np.issubdtype(grid.dtype, np.number), "grid values must be numeric"

    values = set(map(int, np.unique(grid)))
    assert values <= set(range(10)), "expected ARC palette colors"
    assert BACKGROUND in values, "expected black background"

    green = [tuple(map(int, p)) for p in np.argwhere(grid == START_COLOR)]
    assert len(green) == 2, "the green start bar must contain exactly two cells"
    a, b = green
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "start bar must be cardinal and contiguous"

    boundary = [_on_boundary(grid.shape, cell) for cell in green]
    assert sum(boundary) == 1, "exactly one endpoint of the start bar must touch the boundary"
    outer, inner = (a, b) if boundary[0] else (b, a)
    direction = (inner[0] - outer[0], inner[1] - outer[1])

    # Because only the outer endpoint is on the boundary, this vector is the
    # uniquely visible inward normal rather than a color-id convention.
    assert not _on_boundary(grid.shape, inner)
    dr, dc = direction
    assert (dr, dc) in {(-1, 0), (1, 0), (0, -1), (0, 1)}

    portals: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    walls: set[tuple[int, int]] = set()
    counts = Counter(map(int, grid.ravel()))
    semantic_colors = values - {BACKGROUND, START_COLOR}
    assert semantic_colors, "expected colored portal or wall pixels"

    for color in semantic_colors:
        count = int(counts[color])
        points = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        assert count in (1, 2), "every non-start color must occur once or twice"
        if count == 1:
            walls.add(points[0])
        else:
            p, q = points
            portals[p] = (q, color)
            portals[q] = (p, color)

    assert portals, "expected at least one same-color portal pair"
    assert walls, "expected at least one singleton wall color"
    return inner, direction, portals, walls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace an inward beam whose paired colors teleport and recolor it while singleton colors stop it.

    Input structure:
    - A black grid contains one two-cell green bar with exactly one endpoint on the boundary; its inward alignment gives the beam direction.
    - Every other foreground color consists only of isolated pixels and occurs either exactly twice or exactly once.

    Rule:
    - Starting beyond the inner green endpoint, paint traversed black cells with green, preserving every existing foreground pixel.
    - On a color occurring exactly twice, teleport to its mate, keep the original cardinal direction, and continue beyond that mate using the portal color as the new paint color.
    - On a color occurring exactly once, stop before painting it; also stop when the next step leaves the grid.

    Key insight:
    - Global color multiplicity, not color identity, uniquely assigns the behavior: a pair is a direction-preserving recoloring portal and a singleton is a terminal wall.
    """
    g = np.asarray(grid)
    inner, (dr, dc), portals, walls = _parse_scene(g)
    out = g.copy()
    h, w = g.shape

    r, c = inner
    brush = START_COLOR
    used_portal_colors: set[int] = set()
    seen_states: set[tuple[int, int, int]] = set()

    while True:
        state = (r, c, brush)
        assert state not in seen_states, "beam geometry must not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        cell = (nr, nc)
        if int(g[cell]) == BACKGROUND:
            out[cell] = brush
            r, c = cell
            continue

        if cell in walls:
            break

        if cell in portals:
            mate, portal_color = portals[cell]
            assert portal_color not in used_portal_colors, "a portal pair may be crossed at most once"
            used_portal_colors.add(portal_color)
            r, c = mate
            brush = portal_color
            continue

        raise AssertionError("beam must not collide with the green start bar")

    return out


def _to_grid(
    depth: int,
    lateral: int,
    direction: int,
    depth_size: int,
    lateral_size: int,
) -> tuple[int, int]:
    """Map inward depth/lateral coordinates to canonical (row, col)."""
    if direction == 0:  # top -> down
        return depth, lateral
    if direction == 1:  # bottom -> up
        return depth_size - 1 - depth, lateral
    if direction == 2:  # left -> right
        return lateral, depth
    if direction == 3:  # right -> left
        return lateral, depth_size - 1 - depth
    raise ValueError("invalid direction")


def _spaced_lanes(lateral_size: int, count: int) -> list[int] | None:
    """Choose distinct lanes with a full black column/row between beams."""
    candidates = range(2, lateral_size - 2)
    choices = [
        combo
        for combo in itertools.combinations(candidates, count)
        if all(b - a >= 3 for a, b in zip(combo, combo[1:]))
    ]
    if not choices:
        return None
    lanes = list(random.choice(choices))
    random.shuffle(lanes)
    return lanes


def _expected_from_plan(grid: np.ndarray, plan: dict) -> np.ndarray:
    """Render the constructive oracle directly from planned open segments."""
    expected = grid.copy()
    for color, cells in plan["segments"]:
        for cell in cells:
            assert int(grid[cell]) == BACKGROUND
            expected[cell] = color
    return expected


def _well_separated(
    cell: tuple[int, int],
    reserved: set[tuple[int, int]],
    distance: int = 1,
) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > distance for r, c in reserved)


def _validate_generated(grid: np.ndarray, plan: dict) -> bool:
    """Check semantics, uniqueness, branch evidence, and an independent oracle."""
    try:
        inner, direction, portals, walls = _parse_scene(grid)
        if inner != plan["inner"] or direction != plan["direction"]:
            return False

        counts = Counter(map(int, grid.ravel()))
        if any(counts[color] != 2 for color in plan["portal_colors"]):
            return False
        if any(counts[color] != 1 for color in plan["wall_colors"]):
            return False
        if len(portals) != 2 * len(plan["portal_colors"]):
            return False
        if len(walls) != len(plan["wall_colors"]):
            return False

        # Every compositional step has visible evidence: initial green paint,
        # every portal is entered once, every post-portal segment is nonempty
        # and visibly uses that portal's color on a new lane.
        segments = plan["segments"]
        if len(segments) != len(plan["portal_colors"]) + 1:
            return False
        if any(len(cells) < 2 for _color, cells in segments):
            return False
        if segments[0][0] != START_COLOR:
            return False
        if [color for color, _cells in segments[1:]] != plan["portal_colors"]:
            return False
        if len({plan["lane_of_segment"][i] for i in range(len(segments))}) != len(segments):
            return False

        expected = _expected_from_plan(grid, plan)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        changed = actual != grid
        if int(changed.sum()) < 2 * len(segments):
            return False
        if {int(actual[cell]) for cell in map(tuple, np.argwhere(changed))} != {
            START_COLOR,
            *plan["portal_colors"],
        }:
            return False

        # A shortcut that stops at the first colored pixel cannot create the
        # later segments; a shortcut that never recolors also cannot match.
        first_only = grid.copy()
        for cell in segments[0][1]:
            first_only[cell] = START_COLOR
        if np.array_equal(first_only, actual):
            return False
        all_green = grid.copy()
        for _color, cells in segments:
            for cell in cells:
                all_green[cell] = START_COLOR
        if np.array_equal(all_green, actual):
            return False

        end_depth = plan["terminal_depth"]
        end_lane = plan["lane_of_segment"][-1]
        if plan["terminal_wall"]:
            terminal = plan["terminal_cell"]
            if terminal not in walls or actual[terminal] != grid[terminal]:
                return False
            behind = _to_grid(
                end_depth + 1,
                end_lane,
                plan["orientation"],
                plan["depth_size"],
                plan["lateral_size"],
            )
            if int(actual[behind]) != BACKGROUND:
                return False
        else:
            boundary_cell = _to_grid(
                plan["depth_size"] - 1,
                end_lane,
                plan["orientation"],
                plan["depth_size"],
                plan["lateral_size"],
            )
            if int(actual[boundary_cell]) != plan["portal_colors"][-1]:
                return False
        return True
    except (AssertionError, IndexError, ValueError, TypeError):
        return False


def _gen_attempt(terminal_wall: bool, n_portals: int) -> np.ndarray | None:
    depth_size = random.randint(17, 28)
    lateral_size = random.randint(max(18, 3 * (n_portals + 1) + 4), 28)
    orientation = random.randrange(4)
    shape = (
        (depth_size, lateral_size)
        if orientation in (0, 1)
        else (lateral_size, depth_size)
    )
    grid = np.full(shape, BACKGROUND, dtype=int)

    lanes = _spaced_lanes(lateral_size, n_portals + 1)
    if lanes is None:
        return None

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    extra_singletons = 1 if terminal_wall else 2
    if terminal_wall and random.random() < 0.5:
        extra_singletons = 2
    needed = n_portals + int(terminal_wall) + extra_singletons
    if needed > len(colors):
        return None
    portal_colors = colors[:n_portals]
    wall_colors = colors[n_portals:needed]
    terminal_color = wall_colors[0] if terminal_wall else None
    distractor_colors = wall_colors[int(terminal_wall):]

    to_grid = lambda d, x: _to_grid(d, x, orientation, depth_size, lateral_size)
    outer = to_grid(0, lanes[0])
    inner = to_grid(1, lanes[0])
    grid[outer] = START_COLOR
    grid[inner] = START_COLOR

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    marker_cells: set[tuple[int, int]] = {outer, inner}
    source_depths = [1]
    entry_depths: list[int] = []
    portal_points: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for i, color in enumerate(portal_colors):
        source_depth = source_depths[i]
        low = max(source_depth + 3, depth_size // 2 + 1)
        if low > depth_size - 2:
            return None
        entry_depth = random.randint(low, depth_size - 2)
        entry_depths.append(entry_depth)
        entry = to_grid(entry_depth, lanes[i])

        next_source = random.randint(1, max(1, depth_size // 2 - 2))
        exit_cell = to_grid(next_source, lanes[i + 1])
        if not _well_separated(entry, marker_cells) or not _well_separated(exit_cell, marker_cells | {entry}):
            return None

        segment_cells = [to_grid(d, lanes[i]) for d in range(source_depth + 1, entry_depth)]
        segments.append((START_COLOR if i == 0 else portal_colors[i - 1], segment_cells))
        grid[entry] = color
        grid[exit_cell] = color
        marker_cells.update({entry, exit_cell})
        portal_points.append((entry, exit_cell))
        source_depths.append(next_source)

    final_source = source_depths[-1]
    final_lane = lanes[-1]
    terminal_depth = depth_size - 1
    terminal_cell: tuple[int, int] | None = None
    if terminal_wall:
        low = max(final_source + 3, depth_size // 2 + 1)
        if low > depth_size - 2:
            return None
        terminal_depth = random.randint(low, depth_size - 2)
        terminal_cell = to_grid(terminal_depth, final_lane)
        if not _well_separated(terminal_cell, marker_cells):
            return None
        grid[terminal_cell] = int(terminal_color)
        marker_cells.add(terminal_cell)
        final_range = range(final_source + 1, terminal_depth)
    else:
        final_range = range(final_source + 1, depth_size)
    segments.append((portal_colors[-1], [to_grid(d, final_lane) for d in final_range]))

    path_cells = {cell for _color, cells in segments for cell in cells}
    reserved = marker_cells | path_cells
    candidates = [
        (r, c)
        for r in range(1, shape[0] - 1)
        for c in range(1, shape[1] - 1)
        if (r, c) not in reserved and _well_separated((r, c), reserved)
    ]
    random.shuffle(candidates)
    distractors: list[tuple[int, int]] = []
    for color in distractor_colors:
        cell = next((p for p in candidates if _well_separated(p, marker_cells | set(distractors))), None)
        if cell is None:
            return None
        grid[cell] = color
        distractors.append(cell)
        marker_cells.add(cell)
        candidates.remove(cell)

    plan = {
        "inner": inner,
        "direction": (inner[0] - outer[0], inner[1] - outer[1]),
        "orientation": orientation,
        "depth_size": depth_size,
        "lateral_size": lateral_size,
        "portal_colors": portal_colors,
        "wall_colors": wall_colors,
        "portal_points": portal_points,
        "segments": segments,
        "lane_of_segment": lanes,
        "terminal_wall": terminal_wall,
        "terminal_depth": terminal_depth,
        "terminal_cell": terminal_cell,
    }
    return grid if _validate_generated(grid, plan) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically checked puzzle instance no larger than 28x28."""
    global _generation_index
    terminal_wall = _generation_index % 2 == 0
    n_portals = 2 + (_generation_index % 3)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(terminal_wall=terminal_wall, n_portals=n_portals)
        if grid is not None:
            _generation_index += 1
            return grid
    raise RuntimeError("failed to generate a valid multiplicity-portal puzzle")
