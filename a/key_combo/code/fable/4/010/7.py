from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Axis = tuple[str, int]

_BACKGROUND = 0
_PALETTE = tuple(range(1, 10))
_DIRECTIONS = ["horizontal", "vertical", "diagonal"]
_direction_cycle: list[str] = []


def _neighbors(r: int, c: int, connectivity: int = 4):
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    for dr, dc in deltas:
        yield r + dr, c + dc


def _components(cells: set[Coord], connectivity: int = 4) -> list[set[Coord]]:
    """Return connected components using the common ARC (row, column) convention."""
    unseen = set(cells)
    result: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for q in _neighbors(r, c, connectivity):
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _axis_from_markers(a: Coord, b: Coord) -> Axis:
    r1, c1 = a
    r2, c2 = b
    assert a != b, "the two markers must be distinct"
    if r1 == r2:
        return "horizontal", r1
    if c1 == c2:
        return "vertical", c1
    assert abs(r2 - r1) == abs(c2 - c1), "markers must define a 45-degree axis"
    if r2 - r1 == c2 - c1:
        return "diag_down", r1 - c1  # line r-c=k
    return "diag_up", r1 + c1  # line r+c=s


def _side_value(p: Coord, axis: Axis) -> int:
    r, c = p
    kind, value = axis
    if kind == "horizontal":
        return r - value
    if kind == "vertical":
        return c - value
    if kind == "diag_down":
        return r - c - value
    if kind == "diag_up":
        return r + c - value
    raise ValueError(f"unknown axis kind: {kind}")


def _reflect(p: Coord, axis: Axis) -> Coord:
    """Reflect one pixel center across an H/V/45-degree line in grid coordinates."""
    r, c = p
    kind, value = axis
    if kind == "horizontal":
        return 2 * value - r, c
    if kind == "vertical":
        return r, 2 * value - c
    if kind == "diag_down":  # r-c=value
        return c + value, r - value
    if kind == "diag_up":  # r+c=value
        return value - c, value - r
    raise ValueError(f"unknown axis kind: {kind}")


def _isolated_marker_candidates(g: np.ndarray) -> list[tuple[int, tuple[Coord, Coord]]]:
    """Find colors occurring as exactly two foreground-isolated single pixels."""
    height, width = g.shape
    candidates: list[tuple[int, tuple[Coord, Coord]]] = []
    for color in map(int, np.unique(g)):
        if color == _BACKGROUND:
            continue
        coords = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        if len(coords) != 2:
            continue
        isolated = True
        for r, c in coords:
            for rr, cc in _neighbors(r, c, 8):
                if 0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) != _BACKGROUND:
                    isolated = False
                    break
            if not isolated:
                break
        if isolated:
            candidates.append((color, (coords[0], coords[1])))
    return candidates


def _analyze(g: np.ndarray) -> tuple[int, tuple[Coord, Coord], Axis, list[Coord]]:
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "ARC grid must be at most 30x30"
    assert np.issubdtype(g.dtype, np.integer), "ARC colors must be integers"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors 0..9"
    assert np.count_nonzero(g == _BACKGROUND) > g.size // 2, "black must be the background"

    candidates = _isolated_marker_candidates(g)
    assert len(candidates) == 1, "exactly one color must form the two isolated markers"
    marker_color, markers = candidates[0]
    axis = _axis_from_markers(*markers)

    marker_set = set(markers)
    pattern = [
        tuple(map(int, p))
        for p in np.argwhere(g != _BACKGROUND)
        if tuple(map(int, p)) not in marker_set
    ]
    assert len(pattern) >= 3, "expected a nontrivial foreground pattern"
    assert len(_components(set(pattern), connectivity=4)) == 1, "the pattern must be one 4-connected object"

    sides = [_side_value(p, axis) for p in pattern]
    assert all(value != 0 for value in sides), "the pattern must not touch the invisible axis"
    assert all(value > 0 for value in sides) or all(value < 0 for value in sides), (
        "the whole pattern must lie strictly on one side of the axis"
    )

    reflected = [_reflect(p, axis) for p in pattern]
    height, width = g.shape
    assert len(set(reflected)) == len(reflected), "reflection must be one-to-one"
    for source, target in zip(pattern, reflected):
        rr, cc = target
        assert 0 <= rr < height and 0 <= cc < width, "reflected pattern must remain in bounds"
        assert target not in marker_set, "reflected pattern must not cover a marker"
        assert int(g[target]) == _BACKGROUND, "the opposite side must be empty"
        assert _reflect(target, axis) == source, "reflection must be an involution"
        assert _side_value(target, axis) == -_side_value(source, axis), (
            "reflection must use the marker-defined axis"
        )
    return marker_color, markers, axis, pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the foreground pattern by reflecting it across the invisible line generated by two markers.

    Input structure:
    - A black grid contains one 4-connected, possibly multicolored foreground pattern wholly on one side of a line.
    - Exactly two same-colored single pixels are isolated from all foreground; they lie on one horizontal,
      vertical, or 45-degree diagonal line, and the foreground does not meet that line.

    Rule:
    - Use the full line through the two isolated markers as the reflection axis and copy every pattern pixel,
      with its color unchanged, to its reflected coordinate on the empty opposite side.
    - Preserve the original pattern and both markers, and never draw the unmarked parts of the axis.

    Key insight:
    - The two isolated pixels generate an invisible mirror axis rather than endpoints of a segment; diagonal
      marker alignments therefore require reflection across r-c or r+c, not an ordinary up/down or left/right flip.
    """
    g = np.asarray(grid)
    _marker_color, _markers, axis, pattern = _analyze(g)
    out = g.copy()
    for p in pattern:
        out[_reflect(p, axis)] = g[p]
    return out


def _next_direction() -> str:
    # Every three consecutive generated examples cover H, V, and diagonal axes.
    global _direction_cycle
    if not _direction_cycle:
        _direction_cycle = _DIRECTIONS.copy()
        random.shuffle(_direction_cycle)
    return _direction_cycle.pop()


def _sample_axis(height: int, width: int, direction: str) -> tuple[Axis, tuple[Coord, Coord]]:
    if direction == "horizontal":
        row = random.randint(3, height - 4)
        choices = list(range(1, width - 1))
        pairs = [(a, b) for a in choices for b in choices if b - a >= 3]
        c1, c2 = random.choice(pairs)
        return ("horizontal", row), ((row, c1), (row, c2))

    if direction == "vertical":
        col = random.randint(3, width - 4)
        choices = list(range(1, height - 1))
        pairs = [(a, b) for a in choices for b in choices if b - a >= 3]
        r1, r2 = random.choice(pairs)
        return ("vertical", col), ((r1, col), (r2, col))

    kind = random.choice(("diag_down", "diag_up"))
    possible: list[tuple[Axis, list[Coord]]] = []
    if kind == "diag_down":
        for value in range(-(width - 1), height):
            points = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if r - c == value]
            if len(points) >= 6:
                possible.append(((kind, value), points))
    else:
        for value in range(height + width - 1):
            points = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if r + c == value]
            if len(points) >= 6:
                possible.append(((kind, value), points))
    axis, points = random.choice(possible)
    i, j = sorted(random.sample(range(len(points)), 2))
    if j - i < 3:
        eligible = [(a, b) for a in range(len(points)) for b in range(a + 3, len(points))]
        i, j = random.choice(eligible)
    return axis, (points[i], points[j])


def _valid_region(height: int, width: int, axis: Axis, markers: tuple[Coord, Coord], sign: int) -> set[Coord]:
    valid: set[Coord] = set()
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            p = (r, c)
            q = _reflect(p, axis)
            if sign * _side_value(p, axis) <= 0:
                continue
            if not (1 <= q[0] < height - 1 and 1 <= q[1] < width - 1):
                continue
            # Keep both the original and the future copy visually separate from markers.
            if any(max(abs(x[0] - m[0]), abs(x[1] - m[1])) <= 1 for x in (p, q) for m in markers):
                continue
            valid.add(p)
    return valid


def _grow_shape(region: set[Coord], target: int) -> set[Coord] | None:
    large_components = [comp for comp in _components(region, 4) if len(comp) >= target]
    if not large_components:
        return None
    allowed = random.choice(large_components)
    start = random.choice(tuple(allowed))
    shape = {start}
    frontier = {q for q in _neighbors(*start, 4) if q in allowed}
    while len(shape) < target:
        if not frontier:
            return None
        # Mostly grow irregularly, sometimes favor a compact attachment.
        if random.random() < 0.35:
            best = max(sum(n in shape for n in _neighbors(*q, 4)) for q in frontier)
            options = [q for q in frontier if sum(n in shape for n in _neighbors(*q, 4)) == best]
            cell = random.choice(options)
        else:
            cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        shape.add(cell)
        frontier.update(q for q in _neighbors(*cell, 4) if q in allowed and q not in shape)
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return shape


def _normalized_colored(coords: dict[Coord, int]) -> tuple[tuple[int, int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c, int(color)) for (r, c), color in coords.items()))


def _nondegenerate_transform(colored: dict[Coord, int], axis: Axis) -> bool:
    reflected = {_reflect(p, axis): color for p, color in colored.items()}
    reflected_norm = _normalized_colored(reflected)
    original_norm = _normalized_colored(colored)
    if reflected_norm == original_norm:  # a mere translation could make the same copy
        return False

    rows = [r for r, _ in colored]
    cols = [c for _, c in colored]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    flip_ud = {(r0 + r1 - r, c): color for (r, c), color in colored.items()}
    flip_lr = {(r, c0 + c1 - c): color for (r, c), color in colored.items()}
    kind = axis[0]
    if kind.startswith("diag") and reflected_norm in {
        _normalized_colored(flip_ud),
        _normalized_colored(flip_lr),
    }:
        return False
    if kind == "horizontal" and reflected_norm == _normalized_colored(flip_lr):
        return False
    if kind == "vertical" and reflected_norm == _normalized_colored(flip_ud):
        return False
    return True


def _paint_pattern(shape: set[Coord], colors: list[int]) -> dict[Coord, int]:
    cells = list(shape)
    random.shuffle(cells)
    assignments = [colors[i] for i in range(len(colors))]
    assignments.extend(random.choice(colors) for _ in range(len(cells) - len(colors)))
    random.shuffle(assignments)
    return dict(zip(cells, assignments))


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    marker_color: int,
    markers: tuple[Coord, Coord],
    axis: Axis,
    colored: dict[Coord, int],
) -> bool:
    """Check the visible rule and anti-shortcut properties, not only solver closure."""
    try:
        found_color, found_markers, found_axis, found_pattern = _analyze(inp)
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    if found_color != marker_color or set(found_markers) != set(markers) or found_axis != axis:
        return False
    if set(found_pattern) != set(colored) or not np.array_equal(solved, expected):
        return False
    if not _nondegenerate_transform(colored, axis):
        return False

    added = set(map(tuple, np.argwhere((expected != inp) & (inp == _BACKGROUND))))
    reflected = {_reflect(p, axis) for p in colored}
    if added != reflected or not reflected:
        return False
    if any(int(expected[m]) != marker_color for m in markers):
        return False
    if any(int(expected[p]) != color or int(expected[_reflect(p, axis)]) != color for p, color in colored.items()):
        return False
    # Every unmarked pixel on the invisible axis remains black.
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if _side_value((r, c), axis) == 0 and (r, c) not in markers and int(expected[r, c]) != _BACKGROUND:
                return False
    return True


def _generate_attempt(direction: str) -> np.ndarray | None:
    height = random.randint(13, 27)
    width = random.randint(13, 27)
    axis, markers = _sample_axis(height, width, direction)
    sign = random.choice((-1, 1))
    region = _valid_region(height, width, axis, markers, sign)
    if len(region) < 8:
        return None
    target = random.randint(8, min(22, len(region)))
    shape = _grow_shape(region, target)
    if shape is None:
        return None

    palette = random.sample(_PALETTE, random.randint(1, 3) + 1)
    marker_color = palette[0]
    colored = _paint_pattern(shape, palette[1:])
    if not _nondegenerate_transform(colored, axis):
        return None

    inp = np.zeros((height, width), dtype=int)
    for marker in markers:
        inp[marker] = marker_color
    for p, color in colored.items():
        inp[p] = color

    expected = inp.copy()
    for p, color in colored.items():
        expected[_reflect(p, axis)] = color
    if not _semantic_validation(inp, expected, marker_color, markers, axis, colored):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with balanced axis categories."""
    direction = _next_direction()
    for _ in range(400):
        grid = _generate_attempt(direction)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {direction} mirror-axis instance")
