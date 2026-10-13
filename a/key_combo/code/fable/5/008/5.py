from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]

# Directions are ordered clockwise so the same index can rotate the sprite.
DIRS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

# An upward-pointing, filled-outline triangle.  Its center is [1, 1].
_TRIANGLE_UP = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
    ],
    dtype=bool,
)
_TRIANGLES = tuple(np.rot90(_TRIANGLE_UP, -k) for k in range(4))


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert ordered, "grid must not be empty"
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], (
        "background color must be uniquely most common"
    )
    return int(ordered[0][0])


def _direction_between(source: Coord, target: Coord) -> Coord:
    """Direction from source to an axis-aligned target."""
    dr = int(target[0] - source[0])
    dc = int(target[1] - source[1])
    assert (dr == 0) != (dc == 0), "successive stations must be axis-aligned"
    return (0 if dr == 0 else (1 if dr > 0 else -1),
            0 if dc == 0 else (1 if dc > 0 else -1))


def _triangle_direction(mask: np.ndarray) -> Coord:
    assert mask.shape == (3, 3)
    matches = [k for k, template in enumerate(_TRIANGLES) if np.array_equal(mask, template)]
    assert len(matches) == 1, "sprite must be exactly one oriented 3x3 triangle"
    return DIRS[matches[0]]


def _clock_at(patch: np.ndarray) -> Coord | None:
    """Recognize a 3x3 clock and return the direction of its hand."""
    if patch.shape != (3, 3):
        return None
    values, counts = np.unique(patch, return_counts=True)
    if len(values) != 2 or sorted(map(int, counts)) != [2, 7]:
        return None

    hand_color = int(patch[1, 1])
    if int(np.sum(patch == hand_color)) != 2:
        return None
    hand_cells = {tuple(map(int, p)) for p in np.argwhere(patch == hand_color)}
    if (1, 1) not in hand_cells:
        return None
    tip = next(iter(hand_cells - {(1, 1)}), None)
    if tip is None:
        return None
    offset = (tip[0] - 1, tip[1] - 1)
    return offset if offset in DIRS else None


def _parse(grid: np.ndarray) -> tuple[int, int, Coord, Coord, dict[Coord, Coord]]:
    """Parse visible clock and triangle structures without color-id semantics."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    bg = _background(g)
    H, W = map(int, g.shape)

    clocks: dict[Coord, Coord] = {}
    clock_cells: set[Coord] = set()
    for r0 in range(H - 2):
        for c0 in range(W - 2):
            patch = g[r0:r0 + 3, c0:c0 + 3]
            if np.any(patch == bg):
                continue
            direction = _clock_at(patch)
            if direction is None:
                continue
            cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
            assert not (cells & clock_cells), "clock widgets must not overlap"
            center = (r0 + 1, c0 + 1)
            clocks[center] = direction
            clock_cells |= cells

    assert clocks, "at least one 3x3 clock is required"

    remaining = [
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in clock_cells
    ]
    assert len(remaining) == int(_TRIANGLE_UP.sum()), (
        "all non-clock foreground cells must belong to one triangle sprite"
    )
    rows = [p[0] for p in remaining]
    cols = [p[1] for p in remaining]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)

    sprite_colors = {int(g[p]) for p in remaining}
    assert len(sprite_colors) == 1, "triangle sprite must be monochrome"
    sprite_color = next(iter(sprite_colors))
    sprite_mask = g[r0:r1 + 1, c0:c1 + 1] == sprite_color
    sprite_direction = _triangle_direction(sprite_mask)
    sprite_center = (r0 + 1, c0 + 1)

    assert all(1 <= r < H - 1 and 1 <= c < W - 1 for r, c in clocks), (
        "clock centers must leave room for their 3x3 widgets"
    )
    return bg, sprite_color, sprite_center, sprite_direction, clocks


def _candidates(position: Coord, direction: Coord,
                clocks: dict[Coord, Coord]) -> list[tuple[int, Coord]]:
    """Remaining stations strictly ahead on the current grid row/column."""
    r, c = position
    dr, dc = direction
    found = []
    for center in clocks:
        rr, cc = center
        distance = (rr - r) * dr + (cc - c) * dc
        if distance > 0 and (rr - r) * dc == (cc - c) * dr:
            found.append((int(distance), center))
    return found


def _edge_center(shape: tuple[int, int], position: Coord, direction: Coord) -> Coord:
    """Move a 3x3 sprite until its bounding box touches the indicated edge."""
    H, W = map(int, shape)
    r, c = position
    if direction == (-1, 0):
        return 1, c
    if direction == (1, 0):
        return H - 2, c
    if direction == (0, -1):
        return r, 1
    assert direction == (0, 1)
    return r, W - 2


def _travel(shape: tuple[int, int], start: Coord, direction: Coord,
            clocks: dict[Coord, Coord], *, nearest: bool = True
            ) -> tuple[Coord, Coord, list[Coord]]:
    """Consume clocks in route order, then return the final edge position."""
    remaining = dict(clocks)
    position = start
    facing = direction
    visited: list[Coord] = []
    while True:
        choices = _candidates(position, facing, remaining)
        if not choices:
            return _edge_center(shape, position, facing), facing, visited
        _, station = (min(choices) if nearest else max(choices))
        position = station
        facing = remaining.pop(station)
        visited.append(station)


def _paint_triangle(out: np.ndarray, center: Coord, direction: Coord,
                    color: int, bg: int) -> None:
    """Paint the oriented triangle around center into an otherwise clean grid."""
    k = DIRS.index(direction)
    r0, c0 = center[0] - 1, center[1] - 1
    patch = np.full((3, 3), bg, dtype=out.dtype)
    patch[_TRIANGLES[k]] = int(color)
    out[r0:r0 + 3, c0:c0 + 3] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move an arrow sprite through pointer-directed 3x3 clocks, consume the clocks, and draw only its final state.

    Input structure:
    - The uniquely most common color is the background. One monochrome 3x3 triangular sprite has a visible tip that gives its initial cardinal direction.
    - Every clock is a filled 3x3 widget: seven border pixels share one color, while its center and exactly one cardinal border pixel share the hand color.

    Rule:
    - From the sprite center, look only forward on its current grid row or column and move to the nearest remaining clock center on that ray.
    - Consume that clock and rotate the sprite to the absolute direction shown by its hand; repeat until no clock remains ahead, then move until the 3x3 sprite touches that grid edge.
    - Return a clean background grid containing only the sprite at that final center and orientation.

    Key insight:
    - Facing selects the next search ray, nearest distance selects the station on it, and that station's visible hand selects the following ray, so the route and cumulative rotation must be followed in order.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprite_color, start, direction, clocks = _parse(g)
    final_center, final_direction, _ = _travel(g.shape, start, direction, clocks)
    out = np.full(g.shape, bg, dtype=int)
    _paint_triangle(out, final_center, final_direction, sprite_color, bg)
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_MAX_GENERATION_ATTEMPTS = 300


def _transform_point(point: Coord, quarter_turns: int, reflect: bool) -> Coord:
    r, c = point
    if reflect:
        c = -c
    for _ in range(quarter_turns):
        r, c = -c, r
    return int(r), int(c)


def _paint_clock(grid: np.ndarray, center: Coord, direction: Coord,
                 frame_color: int, hand_color: int) -> None:
    r, c = center
    grid[r - 1:r + 2, c - 1:c + 2] = int(frame_color)
    grid[r, c] = int(hand_color)
    grid[r + direction[0], c + direction[1]] = int(hand_color)


def _planned_input() -> tuple[
    np.ndarray, np.ndarray, list[Coord], list[Coord], dict[str, int], tuple[Coord, Coord]
]:
    """Build a routed sample and its independent expected output."""
    # P1 is the routed nearest station.  D is a farther clock on the same
    # initial ray, but is not on the correct route; jumping to D changes the
    # terminal state, making nearest-first observable rather than decorative.
    lattice = [
        (0, 0), (0, 1), (1, 1), (1, 2), (2, 2),
        (2, 3), (3, 3), (3, 4),
    ]
    n_route_clocks = random.randint(4, 7)
    lattice = lattice[:n_route_clocks + 1]
    decoy_lattice = (0, 3)

    row_gap = random.randint(4, 6)
    col_gap = random.randint(4, 6)
    scaled = [(r * row_gap, c * col_gap) for r, c in lattice]
    turns = random.randrange(4)
    reflect = bool(random.getrandbits(1))
    points = [_transform_point(p, turns, reflect) for p in scaled]
    decoy_scaled = (decoy_lattice[0] * row_gap, decoy_lattice[1] * col_gap)
    decoy = _transform_point(decoy_scaled, turns, reflect)

    all_centers = points + [decoy]
    min_r = min(r for r, _ in all_centers)
    max_r = max(r for r, _ in all_centers)
    min_c = min(c for _, c in all_centers)
    max_c = max(c for _, c in all_centers)
    span_h, span_w = max_r - min_r, max_c - min_c
    min_h = max(12, span_h + 3)
    min_w = max(12, span_w + 3)
    if min_h > 30 or min_w > 30:
        raise ValueError("route does not fit ARC bounds")
    H = random.randint(min_h, min(30, min_h + 7))
    W = random.randint(min_w, min(30, min_w + 7))
    top_slack = H - (span_h + 3)
    left_slack = W - (span_w + 3)
    off_r = 1 - min_r + random.randint(0, top_slack)
    off_c = 1 - min_c + random.randint(0, left_slack)
    points = [(r + off_r, c + off_c) for r, c in points]
    decoy = (decoy[0] + off_r, decoy[1] + off_c)

    bg, sprite_color, frame_color, hand_color = random.sample(range(10), 4)
    grid = np.full((H, W), bg, dtype=int)
    initial_direction = _direction_between(points[0], points[1])
    _paint_triangle(grid, points[0], initial_direction, sprite_color, bg)

    hands: list[Coord] = []
    for i, center in enumerate(points[1:], start=1):
        if i < len(points) - 1:
            hand = _direction_between(center, points[i + 1])
        else:
            incoming = _direction_between(points[i - 1], center)
            forbidden = {incoming}
            dr_decoy = decoy[0] - center[0]
            dc_decoy = decoy[1] - center[1]
            if (dr_decoy == 0) != (dc_decoy == 0):
                forbidden.add(_direction_between(center, decoy))
            hand = random.choice([d for d in DIRS if d not in forbidden])
        hands.append(hand)
        _paint_clock(grid, center, hand, frame_color, hand_color)

    final_center = _edge_center(grid.shape, points[-1], hands[-1])
    expected = np.full(grid.shape, bg, dtype=int)
    _paint_triangle(expected, final_center, hands[-1], sprite_color, bg)

    # Select a decoy hand for which the explicit wrong strategy "take the
    # farthest clock" provably has a different visible result.
    route_clocks = {p: hand for p, hand in zip(points[1:], hands)}
    valid_decoy_hands = []
    for decoy_hand in DIRS:
        probe_clocks = dict(route_clocks)
        probe_clocks[decoy] = decoy_hand
        wrong_center, wrong_facing, _ = _travel(
            grid.shape, points[0], initial_direction, probe_clocks, nearest=False
        )
        wrong = np.full(grid.shape, bg, dtype=int)
        _paint_triangle(wrong, wrong_center, wrong_facing, sprite_color, bg)
        if not np.array_equal(wrong, expected):
            valid_decoy_hands.append(decoy_hand)
    if not valid_decoy_hands:
        raise ValueError("decoy does not distinguish nearest from farthest")
    decoy_hand = random.choice(valid_decoy_hands)
    _paint_clock(grid, decoy, decoy_hand, frame_color, hand_color)

    colors = {
        "bg": bg,
        "sprite": sprite_color,
        "frame": frame_color,
        "hand": hand_color,
    }
    return grid, expected, points, hands, colors, (decoy, decoy_hand)


def _semantic_checks(grid: np.ndarray, expected: np.ndarray,
                     points: list[Coord], hands: list[Coord],
                     colors: dict[str, int], decoy_spec: tuple[Coord, Coord]) -> bool:
    """Check semantics, dependencies, uniqueness, and anti-shortcut cases."""
    try:
        bg, sprite_color, start, initial, clocks = _parse(grid)
        decoy, decoy_hand = decoy_spec
        if bg != colors["bg"] or sprite_color != colors["sprite"]:
            return False
        if start != points[0] or set(clocks) != set(points[1:]) | {decoy}:
            return False
        if any(clocks[p] != hand for p, hand in zip(points[1:], hands)):
            return False
        if clocks[decoy] != decoy_hand:
            return False

        final, facing, visited = _travel(grid.shape, start, initial, clocks)
        if visited != points[1:] or decoy in visited:
            return False
        if final != _edge_center(grid.shape, points[-1], hands[-1]) or facing != hands[-1]:
            return False
        if not np.array_equal(main(grid), expected):
            return False

        # The initial ray must visibly contain multiple stations, with P1 unique nearest.
        initial_choices = sorted(_candidates(start, initial, clocks))
        if len(initial_choices) < 2 or initial_choices[0][1] != points[1]:
            return False
        far_final, far_facing, _ = _travel(grid.shape, start, initial, clocks, nearest=False)
        far_out = np.full(grid.shape, bg, dtype=int)
        _paint_triangle(far_out, far_final, far_facing, sprite_color, bg)
        if np.array_equal(far_out, expected):
            return False

        # Removing the nearest station must change the result (nearest-first is necessary).
        removed = grid.copy()
        r, c = points[1]
        removed[r - 1:r + 2, c - 1:c + 2] = bg
        if np.array_equal(main(removed), expected):
            return False

        # Initial facing must matter.
        initial_depends = False
        for alt in DIRS:
            if alt == initial:
                continue
            changed = grid.copy()
            changed[points[0][0] - 1:points[0][0] + 2,
                    points[0][1] - 1:points[0][1] + 2] = bg
            _paint_triangle(changed, points[0], alt, sprite_color, bg)
            if not np.array_equal(main(changed), expected):
                initial_depends = True
                break
        if not initial_depends:
            return False

        # Every clock hand is causally relevant: at least one alternative visible
        # hand direction must alter the final state for each station.
        for center, original_hand in zip(points[1:], hands):
            changes_result = False
            for alt in DIRS:
                if alt == original_hand:
                    continue
                changed = grid.copy()
                _paint_clock(changed, center, alt, colors["frame"], colors["hand"])
                if not np.array_equal(main(changed), expected):
                    changes_result = True
                    break
            if not changes_result:
                return False
    except (AssertionError, ValueError, IndexError):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked clock-station input of at most 30x30."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected, points, hands, colors, decoy_spec = _planned_input()
        except ValueError:
            continue
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        if _semantic_checks(grid, expected, points, hands, colors, decoy_spec):
            return grid
    raise RuntimeError("failed to generate a valid clock-station input")
