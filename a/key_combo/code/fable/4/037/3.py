"""ARC task: move colored pixels using a cross-shaped direction-and-distance legend."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
DIRECTIONS = (
    ("up", -1, 0),
    ("down", 1, 0),
    ("left", 0, -1),
    ("right", 0, 1),
)
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
Move = tuple[Coord, Coord, int, int, int, str]


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: Coord):
    r, c = cell
    for _name, dr, dc in DIRECTIONS:
        yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[set[Coord]]:
    """Return 4-connected components using the common (row, column) convention."""
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in _neighbors4(cell):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, set[Coord], Coord, dict[int, tuple[int, int, int]], list[tuple[Coord, int]]]:
    """Identify the unique cross legend and decode color -> (dr, dc, length)."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg == BACKGROUND, "the puzzle uses a black background"

    components = _components4(g != bg)
    legends = [component for component in components if len(component) > 1]
    assert len(legends) == 1, "expected one nontrivial foreground component"
    assert all(component is legends[0] or len(component) == 1 for component in components)
    legend = legends[0]

    center_candidates = [
        cell for cell in legend if all(neighbor in legend for neighbor in _neighbors4(cell))
    ]
    assert len(center_candidates) == 1, "the cross must have one four-way center"
    center = center_candidates[0]

    decoded: dict[int, tuple[int, int, int]] = {}
    reconstructed = {center}
    cr, cc = center
    for _name, dr, dc in DIRECTIONS:
        first = (cr + dr, cc + dc)
        assert first in legend, "all four cross arms must be present"
        color = int(g[first])
        length = 0
        cell = first
        while cell in legend:
            assert int(g[cell]) == color, "each arm must be monochromatic"
            reconstructed.add(cell)
            length += 1
            cell = (cell[0] + dr, cell[1] + dc)
        assert color not in decoded, "the four arm colors must be distinct"
        decoded[color] = (dr, dc, length)

    assert reconstructed == legend, "the legend component must be an unbranched cross"
    assert len({length for _dr, _dc, length in decoded.values()}) == 4, (
        "the four arm lengths must be different"
    )

    scattered: list[tuple[Coord, int]] = []
    for component in components:
        if component == legend:
            continue
        cell = next(iter(component))
        color = int(g[cell])
        assert color in decoded, "every isolated point color must occur on an arm"
        scattered.append((cell, color))
    assert scattered, "expected isolated colored points"
    assert set(color for _cell, color in scattered) == set(decoded), (
        "every arm color must have at least one moving point"
    )
    return bg, legend, center, decoded, scattered


def main(grid: np.ndarray) -> np.ndarray:
    """Move every isolated colored point by the direction and length encoded by its matching cross arm.

    Input structure:
    - A black background contains one multicolored cross component and otherwise
      isolated single pixels. The cross has a unique center and four straight,
      monochromatic arms whose colors and pairwise-distinct lengths are visible.
    - Every isolated pixel has one of the four arm colors.

    Rule:
    - Keep the cross fixed. For each isolated pixel, use its matching arm's
      center-relative direction and translate the pixel by exactly that arm's
      length. Clear all original point positions and paint only their destinations.

    Key insight:
    - Each arm is simultaneously a compass arrow and a ruler: its side selects
      the displacement direction, while its run length selects the distance.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend, _center, decoded, scattered = _parse_scene(g)

    destinations: list[tuple[Coord, int]] = []
    for (r, c), color in scattered:
        dr, dc, length = decoded[color]
        destination = (r + dr * length, c + dc * length)
        assert _in_bounds(g.shape, destination), "a moved point must stay in bounds"
        assert destination not in legend, "a moved point must not overlap the legend"
        destinations.append((destination, color))

    destination_cells = [cell for cell, _color in destinations]
    assert len(destination_cells) == len(set(destination_cells)), (
        "moved points must have distinct destinations"
    )

    out = g.copy()
    for cell, _color in scattered:
        out[cell] = bg
    for cell, color in destinations:
        out[cell] = color
    return out


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _path(source: Coord, dr: int, dc: int, length: int) -> tuple[Coord, ...]:
    return tuple((source[0] + dr * k, source[1] + dc * k) for k in range(length + 1))


def _candidate_moves(
    shape: tuple[int, int],
    legend: set[Coord],
    direction_index: int,
    color: int,
    length: int,
    kind: str,
) -> list[Move]:
    """Enumerate direct-rule moves having the requested misleading endpoint type."""
    h, w = shape
    _name, dr, dc = DIRECTIONS[direction_index]
    candidates: list[Move] = []
    for dest_r in range(h):
        for dest_c in range(w):
            destination = (dest_r, dest_c)
            source = (dest_r - dr * length, dest_c - dc * length)
            if not _in_bounds(shape, source) or destination in legend or source in legend:
                continue
            if any(_chebyshev(source, cell) <= 1 for cell in legend):
                continue
            travel = _path(source, dr, dc, length)
            if any(cell in legend for cell in travel):
                continue

            forward = (dest_r + dr, dest_c + dc)
            if kind == "edge":
                on_forward_edge = (
                    (dr == -1 and dest_r == 0)
                    or (dr == 1 and dest_r == h - 1)
                    or (dc == -1 and dest_c == 0)
                    or (dc == 1 and dest_c == w - 1)
                )
                if not on_forward_edge:
                    continue
            elif kind == "legend":
                if forward not in legend:
                    continue
            elif kind == "float":
                if not (2 <= dest_r < h - 2 and 2 <= dest_c < w - 2):
                    continue
                if not _in_bounds(shape, forward) or forward in legend:
                    continue
                if any(abs(dest_r - r) + abs(dest_c - c) == 1 for r, c in legend):
                    continue
            else:
                raise ValueError(f"unknown endpoint kind: {kind}")
            candidates.append((source, destination, color, dr, dc, kind))
    random.shuffle(candidates)
    return candidates


def _compatible(candidate: Move, chosen: list[Move], length_by_color: dict[int, int]) -> bool:
    source, destination, color, dr, dc, _kind = candidate
    travel = set(_path(source, dr, dc, length_by_color[color]))
    for old_source, old_destination, old_color, old_dr, old_dc, _old_kind in chosen:
        if _chebyshev(source, old_source) <= 1:
            return False
        if _chebyshev(destination, old_destination) <= 1:
            return False
        if source == old_destination or destination == old_source:
            return False
        old_travel = set(
            _path(old_source, old_dr, old_dc, length_by_color[old_color])
        )
        if travel & old_travel:
            return False
    return True


def _select_moves(
    slots: list[tuple[int, str]],
    pools: dict[tuple[int, str], list[Move]],
    length_by_color: dict[int, int],
) -> list[Move] | None:
    """Small randomized backtracking search for separated points and trajectories."""
    slots = sorted(slots, key=lambda slot: len(pools[slot]))

    def search(index: int, chosen: list[Move]) -> list[Move] | None:
        if index == len(slots):
            return chosen.copy()
        pool = pools[slots[index]]
        # Limiting each level keeps retries fast while retaining substantial variety.
        for candidate in pool[:80]:
            if _compatible(candidate, chosen, length_by_color):
                chosen.append(candidate)
                result = search(index + 1, chosen)
                if result is not None:
                    return result
                chosen.pop()
        return None

    return search(0, [])


def _gravity_destination(
    source: Coord, dr: int, dc: int, shape: tuple[int, int], legend: set[Coord]
) -> Coord:
    """Deliberately wrong slide-until-blocked oracle used only for anti-shortcut checks."""
    cell = source
    while True:
        nxt = (cell[0] + dr, cell[1] + dc)
        if not _in_bounds(shape, nxt) or nxt in legend:
            return cell
        cell = nxt


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    legend: set[Coord],
    center: Coord,
    decoded: dict[int, tuple[int, int, int]],
    moves: list[Move],
) -> bool:
    """Check rule semantics, cue necessity, uniqueness, and the gravity counterexample."""
    try:
        bg, found_legend, found_center, found_decoded, scattered = _parse_scene(inp)
        if bg != BACKGROUND or found_legend != legend or found_center != center:
            return False
        if found_decoded != decoded:
            return False

        sources = {source for source, _dest, _color, _dr, _dc, _kind in moves}
        if sources != {cell for cell, _color in scattered}:
            return False
        if len(moves) != len(sources):
            return False
        if set(color for _source, _dest, color, _dr, _dc, _kind in moves) != set(decoded):
            return False

        anchored = sum(kind in {"edge", "legend"} for *_rest, kind in moves)
        floating = sum(kind == "float" for *_rest, kind in moves)
        if not (anchored > len(moves) / 2 and floating >= 1):
            return False

        direct = inp.copy()
        for source, _destination, _color, _dr, _dc, _kind in moves:
            direct[source] = BACKGROUND
        for _source, destination, color, _dr, _dc, _kind in moves:
            direct[destination] = color
        if not np.array_equal(direct, expected):
            return False
        if np.count_nonzero(inp != expected) != 2 * len(moves):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        # A one-cell shortcut must fail because every distinct arm length is used.
        unit = inp.copy()
        for source in sources:
            unit[source] = BACKGROUND
        for source, _destination, color, dr, dc, _kind in moves:
            unit[source[0] + dr, source[1] + dc] = color
        if np.array_equal(unit, expected):
            return False

        # Most points support the gravity bait, but at least one visibly suspended
        # point has an open cell ahead and makes the slide-until-blocked result wrong.
        gravity = inp.copy()
        for source in sources:
            gravity[source] = BACKGROUND
        for source, _destination, color, dr, dc, _kind in moves:
            gravity[_gravity_destination(source, dr, dc, inp.shape, legend)] = color
        if np.array_equal(gravity, expected):
            return False

        occupied_output = legend | {
            destination for _source, destination, _color, _dr, _dc, _kind in moves
        }
        for _source, destination, _color, dr, dc, kind in moves:
            if kind == "float" and (destination[0] + dr, destination[1] + dc) in occupied_output:
                return False
        return True
    except (AssertionError, IndexError):
        return False


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    lengths = random.sample(range(1, 6), 4)

    up, down, left, right = lengths
    center_r_lo, center_r_hi = up + 3, h - down - 4
    center_c_lo, center_c_hi = left + 3, w - right - 4
    if center_r_lo > center_r_hi or center_c_lo > center_c_hi:
        return None
    center = (
        random.randint(center_r_lo, center_r_hi),
        random.randint(center_c_lo, center_c_hi),
    )

    arm_colors = random.sample(range(1, 10), 4)
    center_color = random.choice([color for color in range(1, 10) if color not in arm_colors])
    inp = np.full((h, w), BACKGROUND, dtype=int)
    inp[center] = center_color
    legend = {center}
    decoded: dict[int, tuple[int, int, int]] = {}
    for direction_index, (_name, dr, dc) in enumerate(DIRECTIONS):
        color = arm_colors[direction_index]
        length = lengths[direction_index]
        decoded[color] = (dr, dc, length)
        for k in range(1, length + 1):
            cell = (center[0] + dr * k, center[1] + dc * k)
            inp[cell] = color
            legend.add(cell)

    point_count = random.randint(7, 10)
    floating_count = random.choice((1, 2))
    kinds = ["float"] * floating_count + ["edge", "legend"]
    kinds.extend(random.choice(("edge", "legend")) for _ in range(point_count - len(kinds)))
    random.shuffle(kinds)

    direction_slots = list(range(4))
    direction_slots.extend(random.randrange(4) for _ in range(point_count - 4))
    random.shuffle(direction_slots)
    slots = [(arm_colors[index], kind) for index, kind in zip(direction_slots, kinds)]

    pools: dict[tuple[int, str], list[Move]] = {}
    length_by_color = {color: decoded[color][2] for color in decoded}
    for color, kind in set(slots):
        direction_index = arm_colors.index(color)
        pools[(color, kind)] = _candidate_moves(
            inp.shape, legend, direction_index, color, length_by_color[color], kind
        )
        if not pools[(color, kind)]:
            return None

    moves = _select_moves(slots, pools, length_by_color)
    if moves is None:
        return None

    expected = inp.copy()
    for source, _destination, color, _dr, _dc, _kind in moves:
        if expected[source] != BACKGROUND:
            return None
        expected[source] = color
    inp = expected.copy()
    for source, _destination, _color, _dr, _dc, _kind in moves:
        expected[source] = BACKGROUND
    for _source, destination, color, _dr, _dc, _kind in moves:
        if expected[destination] != BACKGROUND:
            return None
        expected[destination] = color

    if not _validate_generated(inp, expected, legend, center, decoded, moves):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate compass-and-ruler cross instance."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-and-ruler cross input")
