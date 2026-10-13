from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


# Coordinates follow the shared ARC toolbox convention: every coordinate is
# (row, column).  Each direction points from an implied center toward its bar.
_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_DIRECTION_NAMES = tuple(_DIRECTIONS)
_GAPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    """Return the most frequent color, as in arc_utils.get_bg."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """Small self-contained equivalent of arc_utils.components_mask(..., 4)."""
    height, width = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[set[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DIRECTIONS.values():
                neighbor = (r + dr, c + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _side_cells(center: tuple[int, int], side: str) -> frozenset[tuple[int, int]]:
    """The two bar cells at distances two and three from a hidden center."""
    r, c = center
    dr, dc = _DIRECTIONS[side]
    return frozenset(((r + 2 * dr, c + 2 * dc), (r + 3 * dr, c + 3 * dc)))


def _split_foreground(
    grid: np.ndarray, bg: int
) -> tuple[set[frozenset[tuple[int, int]]], set[tuple[int, int]]]:
    """Separate valid isolated 1x2 bars from valid isolated singleton pixels."""
    bars: set[frozenset[tuple[int, int]]] = set()
    dots: set[tuple[int, int]] = set()
    components = _components4(grid != bg)
    assert components, "expected foreground bars and isolated pixels"

    for component in components:
        if len(component) == 1:
            dots.add(next(iter(component)))
            continue
        assert len(component) == 2, "every non-singleton component must be a 1x2 bar"
        (r0, c0), (r1, c1) = tuple(component)
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "bar pixels must share an edge"
        bars.add(frozenset(component))

    foreground = grid != bg
    height, width = grid.shape
    for r, c in dots:
        neighbors = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and 0 <= r + dr < height
            and 0 <= c + dc < width
        ]
        assert all(not foreground[cell] for cell in neighbors), (
            "singleton pixels must also be isolated under 8-connectivity"
        )
    return bars, dots


def _find_centers(
    grid: np.ndarray,
    bg: int,
    bars: set[frozenset[tuple[int, int]]],
) -> dict[tuple[int, int], frozenset[str]]:
    """Find exactly the hidden centers supported by three or four real bars."""
    height, width = grid.shape
    centers: dict[tuple[int, int], frozenset[str]] = {}
    for r in range(3, height - 3):
        for c in range(3, width - 3):
            if int(grid[r, c]) != bg:
                continue
            if any(int(grid[r + dr, c + dc]) != bg for dr, dc in _GAPS):
                continue
            center = (r, c)
            sides = frozenset(
                side for side in _DIRECTION_NAMES if _side_cells(center, side) in bars
            )
            if len(sides) in (3, 4):
                centers[center] = sides
    return centers


def _membership(
    dot: tuple[int, int],
    centers: dict[tuple[int, int], frozenset[str]],
) -> list[tuple[tuple[int, int], str, int]]:
    """List structurally valid center/sightline memberships for one singleton."""
    r, c = dot
    matches: list[tuple[tuple[int, int], str, int]] = []
    for center, sides in centers.items():
        cr, cc = center
        if r == cr:
            if c <= cc - 4 and "left" in sides:
                matches.append((center, "left", cc - c))
            elif c >= cc + 4 and "right" in sides:
                matches.append((center, "right", c - cc))
        if c == cc:
            if r <= cr - 4 and "up" in sides:
                matches.append((center, "up", cr - r))
            elif r >= cr + 4 and "down" in sides:
                matches.append((center, "down", r - cr))
    return matches


def _analyze(
    grid: np.ndarray,
) -> tuple[
    int,
    int,
    set[frozenset[tuple[int, int]]],
    set[tuple[int, int]],
    dict[tuple[int, int], frozenset[str]],
    dict[tuple[tuple[int, int], str], list[tuple[int, tuple[int, int]]]],
]:
    """Parse the full input and assert all uniqueness promises in the task."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    bg = _get_bg(g)
    assert bg == 0, "the puzzle uses a black background"
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 1, "all foreground objects must share one color"
    fg = colors[0]

    bars, dots = _split_foreground(g, bg)
    centers = _find_centers(g, bg, bars)
    complete = [center for center, sides in centers.items() if len(sides) == 4]
    incomplete = [center for center, sides in centers.items() if len(sides) == 3]
    assert complete, "expected at least one complete implied cross"
    assert len(incomplete) == 1, "expected exactly one three-bar incomplete cross"

    bar_use: defaultdict[frozenset[tuple[int, int]], int] = defaultdict(int)
    for center, sides in centers.items():
        for side in sides:
            bar_use[_side_cells(center, side)] += 1
    assert set(bar_use) == bars, "every bar must belong to an implied cross"
    assert all(count == 1 for count in bar_use.values()), "each bar must identify one center only"

    groups: defaultdict[
        tuple[tuple[int, int], str], list[tuple[int, tuple[int, int]]]
    ] = defaultdict(list)
    for dot in dots:
        matches = _membership(dot, centers)
        assert len(matches) == 1, "every isolated pixel needs one unique sightline owner"
        center, side, distance = matches[0]
        assert distance >= 5, "a scattered pixel must be separated from its bar"
        groups[(center, side)].append((distance, dot))
    assert dots, "expected scattered sightline pixels"
    return bg, fg, bars, dots, centers, dict(groups)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reclaim sightline pixels beside complete hidden crosses and erase those assigned to the incomplete cross.

    Input structure:
    - A black grid has one foreground color, isolated 1x2 bars, and isolated pixels.
    - Four bars at center-relative distances two and three imply a complete unpainted cross center; exactly one other center has only three bars.
    - Every singleton lies beyond one existing bar on the row or column of exactly one implied center.

    Rule:
    - Erase all singleton pixels first.  For each four-bar center, sort the pixels on each cardinal sightline by distance and repack them into consecutive cells beginning just outside that side's bar.
    - Do not repaint pixels owned by the three-bar center.  Preserve every bar, hidden center, and one-cell center gap.

    Key insight:
    - The anchor is invisible and must be inferred from the surrounding micro-bars; only a complete four-bar cross attracts its aligned pixels, while completeness failure turns otherwise identical pixels into removable noise.
    """
    g = np.asarray(grid, dtype=int)
    bg, fg, bars, dots, centers, groups = _analyze(g)

    out = g.copy()
    for r, c in dots:
        out[r, c] = bg

    dot_owner = {
        dot: (center, side)
        for (center, side), entries in groups.items()
        for _distance, dot in entries
    }
    painted: set[tuple[int, int]] = set()
    for (center, side), entries in sorted(groups.items()):
        if len(centers[center]) != 4:
            continue
        cr, cc = center
        dr, dc = _DIRECTIONS[side]
        # Sorting implements the stated near-to-far order even though all dots
        # share one visible color.
        ordered = sorted(entries)
        for index, (_distance, source) in enumerate(ordered):
            destination = (cr + (4 + index) * dr, cc + (4 + index) * dc)
            r, c = destination
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert destination not in painted, "packed sightlines must not overlap"
            assert out[r, c] == bg, "a packed pixel may not overwrite a fixed bar"
            assert destination not in dot_owner or dot_owner[destination] == (center, side), (
                "one sightline may not pack through another sightline's source"
            )
            out[r, c] = fg
            painted.add(destination)

    for bar in bars:
        assert all(out[cell] == fg for cell in bar), "bars must remain unchanged"
    for (r, c), _sides in centers.items():
        assert out[r, c] == bg
        assert all(out[r + dr, c + dc] == bg for dr, dc in _GAPS)
    return out


# ---------------------------------------------------------------------------
# Random generator with an independent, metadata-based semantic oracle.
# ---------------------------------------------------------------------------


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _touches_any(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(
        (r + dr, c + dc) in occupied
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    )


def _distance_cell(center: tuple[int, int], side: str, distance: int) -> tuple[int, int]:
    dr, dc = _DIRECTIONS[side]
    return center[0] + distance * dr, center[1] + distance * dc


def _max_distance(shape: tuple[int, int], center: tuple[int, int], side: str) -> int:
    r, c = center
    if side == "up":
        return r
    if side == "down":
        return shape[0] - 1 - r
    if side == "left":
        return c
    return shape[1] - 1 - c


def _candidate_center_is_clear(
    center: tuple[int, int],
    sides: frozenset[str],
    occupied: set[tuple[int, int]],
) -> bool:
    core = set().union(*(_side_cells(center, side) for side in sides))
    return all(not _touches_any(cell, occupied) for cell in core)


def _place_centers(
    shape: tuple[int, int],
    specifications: list[frozenset[str]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    """Place sparse cross cores while varying positions rather than using slots."""
    height, width = shape
    centers: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for sides in specifications:
        placed = False
        for _ in range(600):
            center = (
                int(_GEN_RNG.integers(4, height - 4)),
                int(_GEN_RNG.integers(4, width - 4)),
            )
            if center[0] in {r for r, _ in centers} or center[1] in {c for _, c in centers}:
                continue
            if not _candidate_center_is_clear(center, sides, occupied):
                continue
            core = set().union(*(_side_cells(center, side) for side in sides))
            occupied.update(core)
            centers.append(center)
            placed = True
            break
        if not placed:
            return None
    return centers, occupied


def _visible_to_one_center(
    cell: tuple[int, int],
    centers: list[tuple[int, int]],
    specifications: list[frozenset[str]],
) -> bool:
    mapping = {center: sides for center, sides in zip(centers, specifications)}
    return len(_membership(cell, mapping)) == 1


def _choose_group_cells(
    shape: tuple[int, int],
    center: tuple[int, int],
    side: str,
    count: int,
    centers: list[tuple[int, int]],
    specifications: list[frozenset[str]],
    input_occupied: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    """Choose separated source dots and their compacted consecutive targets."""
    destinations = {_distance_cell(center, side, 4 + index) for index in range(count)}
    if not all(_in_bounds(shape, cell) for cell in destinations):
        return None
    if destinations & output_reserved:
        return None
    # A source may coincide with its own eventual target, but not with another
    # group's target; targets may not cover another group's source.
    if any(cell in input_occupied and cell not in _side_cells(center, side) for cell in destinations):
        return None

    possible = [
        distance
        for distance in range(5, _max_distance(shape, center, side) + 1)
        if _distance_cell(center, side, distance) not in output_reserved
    ]
    if len(possible) < count:
        return None

    for _ in range(160):
        distances = sorted(
            int(value)
            for value in _GEN_RNG.choice(possible, size=count, replace=False)
        )
        if any(b - a < 2 for a, b in zip(distances, distances[1:])):
            continue
        cells = [_distance_cell(center, side, distance) for distance in distances]
        if any(_touches_any(cell, input_occupied) for cell in cells):
            continue
        if any(not _visible_to_one_center(cell, centers, specifications) for cell in cells):
            continue
        if destinations & (input_occupied - set(_side_cells(center, side))):
            continue
        return cells, destinations
    return None


def _build_expected(
    grid: np.ndarray,
    dots_by_group: dict[tuple[tuple[int, int], str], list[tuple[int, int]]],
    center_sides: dict[tuple[int, int], frozenset[str]],
    fg: int,
) -> np.ndarray:
    expected = grid.copy()
    all_dots = {cell for cells in dots_by_group.values() for cell in cells}
    for cell in all_dots:
        expected[cell] = 0
    for (center, side), cells in dots_by_group.items():
        if len(center_sides[center]) != 4:
            continue
        for index in range(len(cells)):
            expected[_distance_cell(center, side, 4 + index)] = fg
    return expected


def _semantic_oracle(
    grid: np.ndarray,
    expected: np.ndarray,
    centers: list[tuple[int, int]],
    specifications: list[frozenset[str]],
    dots_by_group: dict[tuple[tuple[int, int], str], list[tuple[int, int]]],
) -> bool:
    """Check rule meaning and anti-shortcut properties, not only main's closure."""
    intended = {center: sides for center, sides in zip(centers, specifications)}
    try:
        bg, fg, _bars, dots, parsed, groups = _analyze(grid)
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if bg != 0 or parsed != intended or not np.array_equal(actual, expected):
        return False

    intended_dots = {cell for cells in dots_by_group.values() for cell in cells}
    if dots != intended_dots:
        return False
    parsed_groups = {
        key: {dot for _distance, dot in entries} for key, entries in groups.items()
    }
    if parsed_groups != {key: set(cells) for key, cells in dots_by_group.items()}:
        return False

    complete = [center for center, sides in intended.items() if len(sides) == 4]
    incomplete = [center for center, sides in intended.items() if len(sides) == 3]
    if len(complete) < 2 or len(incomplete) != 1:
        return False
    if not all(any(key[0] == center for key in dots_by_group) for center in intended):
        return False
    if not any(
        len(cells) >= 2 and len(intended[center]) == 4
        for (center, _side), cells in dots_by_group.items()
    ):
        return False

    erased = grid.copy()
    for cell in intended_dots:
        erased[cell] = 0
    if np.array_equal(expected, erased):
        return False  # rejects the shortcut "erase every singleton"

    keep_complete = grid.copy()
    for (center, _side), cells in dots_by_group.items():
        if len(intended[center]) == 3:
            for cell in cells:
                keep_complete[cell] = 0
    if np.array_equal(expected, keep_complete):
        return False  # complete-cross dots must really move and compact

    attract_all = erased.copy()
    for (center, side), cells in dots_by_group.items():
        for index in range(len(cells)):
            attract_all[_distance_cell(center, side, 4 + index)] = fg
    if np.array_equal(expected, attract_all):
        return False  # the missing fourth bar must change the result
    return True


def _generation_attempt() -> np.ndarray | None:
    shape = (
        int(_GEN_RNG.integers(25, 31)),
        int(_GEN_RNG.integers(25, 31)),
    )
    complete_count = int(_GEN_RNG.choice((2, 3), p=(0.65, 0.35)))
    missing = str(_GEN_RNG.choice(_DIRECTION_NAMES))
    specifications = [frozenset(_DIRECTION_NAMES) for _ in range(complete_count)]
    specifications.append(frozenset(side for side in _DIRECTION_NAMES if side != missing))
    order = _GEN_RNG.permutation(len(specifications))
    specifications = [specifications[int(index)] for index in order]

    placement = _place_centers(shape, specifications)
    if placement is None:
        return None
    centers, core_occupied = placement

    # Core bars alone must imply exactly the intended centers; this rejects
    # accidental crosses assembled from bars belonging to different motifs.
    probe = np.zeros(shape, dtype=int)
    for cell in core_occupied:
        probe[cell] = 1
    probe_bars, probe_dots = _split_foreground(probe, 0)
    if probe_dots or _find_centers(probe, 0, probe_bars) != {
        center: sides for center, sides in zip(centers, specifications)
    }:
        return None

    fg = int(_GEN_RNG.integers(1, 10))
    grid = np.zeros(shape, dtype=int)
    for cell in core_occupied:
        grid[cell] = fg

    # Every center receives dots.  A randomly selected complete center always
    # has at least two dots on one side, exercising same-side ordering/packing.
    complete_indices = [i for i, sides in enumerate(specifications) if len(sides) == 4]
    multi_index = int(_GEN_RNG.choice(complete_indices))
    group_plan: list[tuple[int, str, int]] = []
    for index, sides in enumerate(specifications):
        shuffled_sides = [str(side) for side in _GEN_RNG.permutation(tuple(sides))]
        if index == multi_index:
            group_plan.append((index, shuffled_sides[0], 2))
            if _GEN_RNG.random() < 0.65:
                group_plan.append((index, shuffled_sides[1], 1))
        else:
            group_plan.append((index, shuffled_sides[0], 1))
            group_plan.append((index, shuffled_sides[1], 1))
            if _GEN_RNG.random() < 0.35:
                group_plan.append((index, shuffled_sides[2], 1))
    _GEN_RNG.shuffle(group_plan)

    input_occupied = set(core_occupied)
    output_reserved = set(core_occupied)
    dots_by_group: dict[tuple[tuple[int, int], str], list[tuple[int, int]]] = {}
    for index, side, count in group_plan:
        center = centers[index]
        chosen = _choose_group_cells(
            shape,
            center,
            side,
            count,
            centers,
            specifications,
            input_occupied,
            output_reserved,
        )
        if chosen is None:
            return None
        cells, destinations = chosen
        dots_by_group[(center, side)] = cells
        input_occupied.update(cells)
        output_reserved.update(destinations)
        for cell in cells:
            grid[cell] = fg

    center_sides = {center: sides for center, sides in zip(centers, specifications)}
    expected = _build_expected(grid, dots_by_group, center_sides, fg)
    if not _semantic_oracle(
        grid, expected, centers, specifications, dots_by_group
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with both semantic branches visible."""
    for _ in range(500):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hidden-anchor instance")
