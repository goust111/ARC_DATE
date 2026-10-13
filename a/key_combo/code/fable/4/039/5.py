from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SLASH_MIRROR = 8
BACKSLASH_MIRROR = 9
MIRROR_COLORS = (SLASH_MIRROR, BACKSLASH_MIRROR)
FILTER_COLORS = (1, 2, 3, 4, 5, 6, 7)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the common ARC (row, column) coordinate convention."""
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _source_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the unique non-corner colored border source and its inward direction."""
    height, width = grid.shape
    border = []
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                if int(grid[r, c]) != BACKGROUND:
                    border.append((r, c))

    assert len(border) == 1, "expected exactly one colored border source"
    r, c = border[0]
    inward = []
    if r == 0:
        inward.append((1, 0))
    if r == height - 1:
        inward.append((-1, 0))
    if c == 0:
        inward.append((0, 1))
    if c == width - 1:
        inward.append((0, -1))
    assert len(inward) == 1, "the source must not occupy a corner"
    return (r, c), inward[0]


def _reflected(direction: tuple[int, int], mirror: int) -> tuple[int, int]:
    """Return the cardinal reflection for a teal '/' or maroon '\\' mirror."""
    dr, dc = direction
    if mirror == SLASH_MIRROR:
        return -dc, -dr
    assert mirror == BACKSLASH_MIRROR
    return dc, dr


def _trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Trace against the immutable input so newly painted trail cells are never components."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors 0..9"

    source, direction = _source_and_direction(g)
    beam_color = int(g[source])
    out = g.copy()
    r, c = source
    path = [source]
    interactions = []
    states = set()

    # Position plus direction determines the future geometry; including it catches loops
    # even if filters would keep changing the beam color inside the loop.
    step_limit = 4 * g.size + 1
    for _ in range(step_limit):
        state = (r, c, direction[0], direction[1])
        assert state not in states, "the optical path must not form a loop"
        states.add(state)

        nr = r + direction[0]
        nc = c + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            details = {
                "source": source,
                "initial_direction": _source_and_direction(g)[1],
                "path": path,
                "interactions": interactions,
                "exit_direction": direction,
            }
            return out, details

        r, c = nr, nc
        path.append((r, c))
        component = int(g[r, c])
        if component == BACKGROUND:
            out[r, c] = beam_color
            continue

        before_direction = direction
        before_color = beam_color
        if component in MIRROR_COLORS:
            direction = _reflected(direction, component)
            kind = "mirror"
        else:
            beam_color = component
            kind = "filter"
        interactions.append(
            {
                "kind": kind,
                "coord": (r, c),
                "component_color": component,
                "direction_before": before_direction,
                "direction_after": direction,
                "beam_before": before_color,
                "beam_after": beam_color,
                "path_index": len(path) - 1,
            }
        )

    raise AssertionError("optical trace exceeded the finite non-loop step bound")


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a source beam through diagonal mirrors and recoloring filters.

    Input structure:
    - A black grid has exactly one non-corner colored source pixel on its border.
    - Interior single pixels are teal and maroon diagonal mirrors, or other-colored filters.
    - The source fires orthogonally inward, and the component arrangement gives a finite, non-looping path.

    Rule:
    - Paint each empty path cell with the current beam color while preserving the source and every component.
    - Teal reflects as "/" and maroon reflects as "\\"; neither mirror changes the beam color.
    - Every other interior colored component is a straight-through filter whose color applies after that pixel.
    - Stop after the ray crosses the opposite or side boundary of the grid.

    Key insight:
    - Mirror color determines each geometric turn, while filter color persists along all later path cells until another filter is reached.
    """
    g = np.asarray(grid, dtype=int)
    out, details = _trace(g)
    hit_mirrors = {
        item["component_color"]
        for item in details["interactions"]
        if item["kind"] == "mirror"
    }
    hit_filters = [item for item in details["interactions"] if item["kind"] == "filter"]
    assert hit_mirrors == set(MIRROR_COLORS), "the path must encounter both mirror orientations"
    assert hit_filters, "the path must encounter at least one filter"
    return out


# ------------------------------- generator ---------------------------------

GEN_HEIGHT_RANGE = (14, 24)
GEN_WIDTH_RANGE = (14, 24)
GEN_MIRROR_COUNT_RANGE = (3, 6)
GEN_FILTER_COUNT_RANGE = (2, 4)
GEN_MIN_SEGMENT = 3
GEN_MAX_ATTEMPTS = 500


def _random_source(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(2, width - 3)), (1, 0)
    if side == "bottom":
        return (height - 1, random.randint(2, width - 3)), (-1, 0)
    if side == "left":
        return (random.randint(2, height - 3), 0), (0, 1)
    return (random.randint(2, height - 3), width - 1), (0, -1)


def _mirror_sequence(count: int) -> list[int]:
    mirrors = [random.choice(MIRROR_COLORS) for _ in range(count)]
    if len(set(mirrors)) == 1:
        mirrors[random.randrange(count)] = (
            BACKSLASH_MIRROR if mirrors[0] == SLASH_MIRROR else SLASH_MIRROR
        )
    random.shuffle(mirrors)
    return mirrors


def _interior(shape: tuple[int, int], coord: tuple[int, int]) -> bool:
    height, width = shape
    r, c = coord
    return 1 <= r < height - 1 and 1 <= c < width - 1


def _available_forward_cells(
    shape: tuple[int, int], position: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    cells = []
    r, c = position
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _route_is_visually_clear(path: list[tuple[int, int]]) -> bool:
    """Reject crossings and side-touching nonconsecutive segments that resemble thick lines."""
    index = {coord: i for i, coord in enumerate(path)}
    if len(index) != len(path):
        return False
    for i, (r, c) in enumerate(path):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = index.get((r + dr, c + dc))
            if j is not None and abs(i - j) != 1:
                return False
    return True


def _build_route(
    shape: tuple[int, int], mirror_colors: list[int]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]] | None:
    source, direction = _random_source(shape)
    path = [source]
    used = {source}
    mirrors: dict[tuple[int, int], int] = {}
    position = source

    for index, mirror_color in enumerate(mirror_colors):
        forward = _available_forward_cells(shape, position, direction)
        candidates = []
        for distance in range(GEN_MIN_SEGMENT, len(forward) + 1):
            segment = forward[:distance]
            endpoint = segment[-1]
            if not _interior(shape, endpoint) or any(cell in used for cell in segment):
                continue
            next_direction = _reflected(direction, mirror_color)
            next_forward = _available_forward_cells(shape, endpoint, next_direction)
            required = GEN_MIN_SEGMENT if index + 1 < len(mirror_colors) else GEN_MIN_SEGMENT + 1
            if len([cell for cell in next_forward if _interior(shape, cell)]) < required:
                continue
            candidates.append((segment, endpoint, next_direction))

        if not candidates:
            return None
        segment, endpoint, next_direction = random.choice(candidates)
        path.extend(segment)
        used.update(segment)
        mirrors[endpoint] = mirror_color
        position = endpoint
        direction = next_direction

    tail = _available_forward_cells(shape, position, direction)
    if len(tail) < GEN_MIN_SEGMENT + 1 or any(cell in used for cell in tail):
        return None
    path.extend(tail)
    if not _route_is_visually_clear(path):
        return None
    return path, mirrors


def _choose_filter_indices(
    shape: tuple[int, int], path: list[tuple[int, int]], mirror_indices: set[int]
) -> list[int] | None:
    eligible = [
        i
        for i, coord in enumerate(path)
        if _interior(shape, coord)
        and i < len(path) - 2
        and all(abs(i - j) >= 2 for j in mirror_indices)
    ]
    eligible = [i for i in eligible if i > 1 and i + 1 not in mirror_indices]
    if len(eligible) < GEN_FILTER_COUNT_RANGE[0]:
        return None

    desired = random.randint(*GEN_FILTER_COUNT_RANGE)
    desired = min(desired, len(eligible))
    first_mirror = min(mirror_indices)
    last_mirror = max(mirror_indices)
    for _ in range(120):
        chosen = sorted(random.sample(eligible, desired))
        if any(b - a < 3 for a, b in zip(chosen, chosen[1:])):
            continue
        # Force filter/mirror composition in both orders, ruling out examples where all
        # recoloring happens only before or only after the entire reflected route.
        if not any(i < last_mirror for i in chosen):
            continue
        if not any(i > first_mirror for i in chosen):
            continue
        return chosen
    return None


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    path: list[tuple[int, int]],
    mirrors: dict[tuple[int, int], int],
    filters: dict[tuple[int, int], int],
) -> bool:
    """Check puzzle meaning, component necessity, uniqueness, and non-degeneracy."""
    try:
        actual, details = _trace(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(path) != len(set(path)) or details["path"] != path:
        return False

    source = details["source"]
    component_coords = set(mirrors) | set(filters)
    visible_coords = {tuple(map(int, rc)) for rc in np.argwhere(inp != BACKGROUND)}
    if visible_coords != component_coords | {source}:
        return False
    hit_coords = {item["coord"] for item in details["interactions"]}
    if hit_coords != component_coords:
        return False

    mirror_hits = [item for item in details["interactions"] if item["kind"] == "mirror"]
    filter_hits = [item for item in details["interactions"] if item["kind"] == "filter"]
    if {item["component_color"] for item in mirror_hits} != set(MIRROR_COLORS):
        return False
    if len(filter_hits) != len(filters) or len(filter_hits) < GEN_FILTER_COUNT_RANGE[0]:
        return False

    # Each mirror must make a real 90-degree turn. Each filter must change color and
    # expose that change on a following empty path cell.
    for item in mirror_hits:
        if item["direction_after"] != _reflected(
            item["direction_before"], item["component_color"]
        ):
            return False
        if item["direction_after"] == item["direction_before"]:
            return False
    for item in filter_hits:
        i = item["path_index"]
        if item["beam_before"] == item["beam_after"]:
            return False
        if i + 1 >= len(path) or int(inp[path[i + 1]]) != BACKGROUND:
            return False
        if int(expected[path[i + 1]]) != item["beam_after"]:
            return False

    original_added = (expected != inp) & (inp == BACKGROUND)
    if int(original_added.sum()) < 8:
        return False

    # Toggling either mirror orientation must alter the geometric trail, not merely
    # recolor the component cell itself.
    for coord, color in mirrors.items():
        changed = inp.copy()
        changed[coord] = BACKSLASH_MIRROR if color == SLASH_MIRROR else SLASH_MIRROR
        try:
            changed_out, _ = _trace(changed)
        except AssertionError:
            continue
        changed_added = (changed_out != changed) & (changed == BACKGROUND)
        if np.array_equal(changed_added, original_added):
            return False

    # Changing a filter to another non-mirror hue must visibly alter a later empty
    # trail cell, proving that filters are not decorative waypoints.
    for coord, color in filters.items():
        replacement = next(candidate for candidate in FILTER_COLORS if candidate != color)
        changed = inp.copy()
        changed[coord] = replacement
        try:
            changed_out, _ = _trace(changed)
        except AssertionError:
            return False
        if not np.any((changed_out != expected) & (inp == BACKGROUND)):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_HEIGHT_RANGE),
        random.randint(*GEN_WIDTH_RANGE),
    )
    mirror_count = random.randint(*GEN_MIRROR_COUNT_RANGE)
    built = _build_route(shape, _mirror_sequence(mirror_count))
    if built is None:
        return None
    path, mirrors = built

    path_to_index = {coord: i for i, coord in enumerate(path)}
    mirror_indices = {path_to_index[coord] for coord in mirrors}
    filter_indices = _choose_filter_indices(shape, path, mirror_indices)
    if filter_indices is None:
        return None

    source = path[0]
    source_color = random.choice(FILTER_COLORS)
    current_color = source_color
    filters: dict[tuple[int, int], int] = {}
    for index in filter_indices:
        color = random.choice([candidate for candidate in FILTER_COLORS if candidate != current_color])
        filters[path[index]] = color
        current_color = color

    inp = np.zeros(shape, dtype=int)
    inp[source] = source_color
    for coord, color in mirrors.items():
        inp[coord] = color
    for coord, color in filters.items():
        inp[coord] = color

    try:
        expected = main(inp)
    except AssertionError:
        return None
    if not _semantic_validation(inp, expected, path, mirrors, filters):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, fully exercised mirror-and-filter ray-tracing instances."""
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-and-filter optical path")
