from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _doors_on_line(line: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int]] | None:
    """Return (center, name, pointer) triples, or None if this is not a wall."""
    n = len(line)
    doors: list[tuple[int, int, int]] = []
    used: set[int] = set()
    for x in range(1, n - 1):
        if (int(line[x]) == bg and int(line[x - 1]) not in (bg, wall)
                and int(line[x + 1]) not in (bg, wall)):
            if any(y in used for y in (x - 1, x, x + 1)):
                return None
            doors.append((x, int(line[x - 1]), int(line[x + 1])))
            used.update((x - 1, x, x + 1))
    if len(doors) < 2:
        return None
    if any(int(line[x]) != wall for x in range(n) if x not in used):
        return None
    names = [name for _x, name, _pointer in doors]
    if len(names) != len(set(names)):
        return None
    return doors


def _wall_candidates(g: np.ndarray, axis: int, bg: int):
    """Find horizontal (axis=0) or vertical (axis=1) encoded wall lines."""
    count = g.shape[axis]
    span = g.shape[1 - axis]
    found = []
    for pos in range(count):
        line = g[pos, :] if axis == 0 else g[:, pos]
        non_bg = [int(v) for v in line if int(v) != bg]
        if not non_bg:
            continue
        wall, amount = Counter(non_bg).most_common(1)[0]
        if amount < max(4, int(np.ceil(0.30 * span))):
            continue
        doors = _doors_on_line(line, bg, wall)
        if doors is not None:
            found.append((pos, wall, doors))
    return found


def _scene(g: np.ndarray):
    bg = _background(g)
    horizontal = _wall_candidates(g, 0, bg)
    vertical = _wall_candidates(g, 1, bg)
    choices = [("horizontal", horizontal), ("vertical", vertical)]
    choices = [(orientation, walls) for orientation, walls in choices if len(walls) >= 2]
    assert len(choices) == 1, "expected one unambiguous set of parallel walls"
    orientation, walls = choices[0]
    walls.sort(key=lambda item: item[0])
    wall_colors = {wall for _pos, wall, _doors in walls}
    assert len(wall_colors) == 1, "all parallel walls must share one visible wall color"

    wall_positions = {pos for pos, _wall, _doors in walls}
    outside = []
    for r, c in np.argwhere(g != bg):
        r, c = int(r), int(c)
        if (orientation == "horizontal" and r not in wall_positions) or (
                orientation == "vertical" and c not in wall_positions):
            outside.append((r, c))
    assert len(outside) == 2, "expected exactly two pixels outside the wall lines"
    marker = int(g[outside[0]])
    assert int(g[outside[1]]) == marker, "the two outside markers must have one color"

    if orientation == "horizontal":
        outside.sort(key=lambda p: p[0])
        assert outside[0][0] < walls[0][0] and outside[1][0] > walls[-1][0]
    else:
        outside.sort(key=lambda p: p[1])
        assert outside[0][1] < walls[0][0] and outside[1][1] > walls[-1][0]
    return bg, marker, orientation, walls, outside[0], outside[1]


def _paint(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int,
           allowed: set[int]) -> None:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1, "route segments must be orthogonal"
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    r, c = r0, c0
    while True:
        assert int(out[r, c]) in allowed, "route met a wall or an unselected nameplate"
        out[r, c] = color
        if (r, c) == (r1, c1):
            return
        r, c = r + dr, c + dc


def _render_route(g: np.ndarray, orientation: str, centers: list[tuple[int, int]],
                  start: tuple[int, int], end: tuple[int, int], bg: int,
                  marker: int) -> np.ndarray:
    out = g.copy()
    allowed = {bg, marker}
    current = start
    if orientation == "horizontal":
        for door_r, door_c in centers:
            approach = (door_r - 1, current[1])
            _paint(out, current, approach, marker, allowed)
            _paint(out, approach, (door_r - 1, door_c), marker, allowed)
            _paint(out, (door_r - 1, door_c), (door_r, door_c), marker, allowed)
            current = (door_r, door_c)
        bend = (end[0], current[1])
    else:
        for door_r, door_c in centers:
            approach = (current[0], door_c - 1)
            _paint(out, current, approach, marker, allowed)
            _paint(out, approach, (door_r, door_c - 1), marker, allowed)
            _paint(out, (door_r, door_c - 1), (door_r, door_c), marker, allowed)
            current = (door_r, door_c)
        bend = (current[0], end[1])
    _paint(out, current, bend, marker, allowed)
    _paint(out, bend, end, marker, allowed)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the marker-colored route selected by the door-name pointer chain.

    Input structure:
    - A dominant background contains several parallel full-span walls, each with
      multiple three-cell doors: name color, background center, pointer color.
    - Two same-colored marker pixels lie on opposite outer sides of the walls;
      door triples run left-to-right, or top-to-bottom after transposition.

    Rule:
    - Use the marker color as the required name on the first wall.  On every
      wall choose the unique door with that name, then use its pointer color as
      the required name on the next wall; the final pointer returns to marker color.
    - In wall order, draw the canonical orthogonal corridor route through every
      selected background center and join the two markers, preserving all plates.

    Key insight:
    - Geometric proximity and alignment are decoys: the visible color-pointer
      chain, interpreted in the parallel-wall coordinate system, selects the doors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, marker, orientation, walls, start, end = _scene(g)

    wanted = marker
    centers: list[tuple[int, int]] = []
    for wall_pos, _wall_color, doors in walls:
        matches = [(x, pointer) for x, name, pointer in doors if name == wanted]
        assert len(matches) == 1, "each chain color must name exactly one door on its wall"
        x, wanted = matches[0]
        centers.append((wall_pos, x) if orientation == "horizontal" else (x, wall_pos))
    assert wanted == marker, "the last selected pointer must return to marker color"
    return _render_route(g, orientation, centers, start, end, bg, marker)


def _spaced_positions(n: int, lo: int, hi: int) -> list[int]:
    # Sample in a compressed coordinate system, then restore three mandatory
    # gap cells after each preceding pick.  This has no rejection tail.
    compressed_count = hi - lo + 1 - 3 * (n - 1)
    if compressed_count < n:
        raise RuntimeError("could not place separated doors")
    picked = sorted(random.sample(range(compressed_count), n))
    return [lo + x + 3 * i for i, x in enumerate(picked)]


def _generated_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    across = h if orientation == "horizontal" else w
    along = w if orientation == "horizontal" else h
    wall_slots = 1 + (across - 7) // 4
    n_walls = random.randint(3, min(5, wall_slots))
    n_doors = random.randint(3, min(4, (along - 2) // 4))

    wall_positions = _spaced_positions(n_walls, 3, across - 4)
    if any(b - a < 3 for a, b in zip(wall_positions, wall_positions[1:])):
        return None
    door_positions = [_spaced_positions(n_doors, 2, along - 3) for _ in wall_positions]
    selected_indices = [random.randrange(n_doors) for _ in wall_positions]
    if len(set(door_positions[i][selected_indices[i]] for i in range(n_walls))) == 1:
        selected_indices[-1] = (selected_indices[-1] + 1) % n_doors

    bg, wall_color = random.sample(range(10), 2)
    usable = [c for c in range(10) if c not in (bg, wall_color)]
    marker = random.choice(usable)
    middle = random.sample([c for c in usable if c != marker], n_walls - 1)
    chain = [marker, *middle, marker]

    grid = np.full((h, w), bg, dtype=int)
    selected_centers: list[tuple[int, int]] = []
    all_centers: list[list[tuple[int, int]]] = []
    for i, wall_pos in enumerate(wall_positions):
        if orientation == "horizontal":
            grid[wall_pos, :] = wall_color
        else:
            grid[:, wall_pos] = wall_color
        selected_name = chain[i]
        other_names = random.sample([c for c in usable if c != selected_name], n_doors - 1)
        name_iter = iter(other_names)
        row_centers = []
        for j, x in enumerate(door_positions[i]):
            name = selected_name if j == selected_indices[i] else next(name_iter)
            pointer = chain[i + 1] if j == selected_indices[i] else random.choice(usable)
            if orientation == "horizontal":
                grid[wall_pos, x - 1:x + 2] = (name, bg, pointer)
                center = (wall_pos, x)
            else:
                grid[x - 1:x + 2, wall_pos] = (name, bg, pointer)
                center = (x, wall_pos)
            row_centers.append(center)
        all_centers.append(row_centers)
        selected_centers.append(row_centers[selected_indices[i]])

    # Align the start marker with an unselected first-wall door: a strong
    # counterexample to nearest-door and straight-ahead heuristics.
    decoy_index = random.choice([j for j in range(n_doors) if j != selected_indices[0]])
    decoy_tangent = door_positions[0][decoy_index]
    end_decoy_index = random.choice([j for j in range(n_doors) if j != selected_indices[-1]])
    end_tangent = door_positions[-1][end_decoy_index]
    if orientation == "horizontal":
        start = (random.randint(0, wall_positions[0] - 2), decoy_tangent)
        end = (random.randint(wall_positions[-1] + 2, h - 1), end_tangent)
    else:
        start = (decoy_tangent, random.randint(0, wall_positions[0] - 2))
        end = (end_tangent, random.randint(wall_positions[-1] + 2, w - 1))
    grid[start] = marker
    grid[end] = marker

    expected = _render_route(grid, orientation, selected_centers, start, end, bg, marker)
    try:
        actual = main(grid)
        parsed = _scene(grid)
    except (AssertionError, IndexError):
        return None
    if parsed[2] != orientation or not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected) or selected_centers[0] == all_centers[0][decoy_index]:
        return None
    # Every unselected opening remains untouched, so the route cannot be
    # explained as simply crossing every visible door.
    for i, centers in enumerate(all_centers):
        for j, center in enumerate(centers):
            if j != selected_indices[i] and int(expected[center]) != bg:
                return None
    # The selected first door is deliberately farther than the aligned decoy.
    selected_tangent = door_positions[0][selected_indices[0]]
    if abs(selected_tangent - decoy_tangent) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate horizontal and vertical pointer mazes."""
    for _ in range(300):
        result = _generated_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid door-name chain puzzle")
