from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

UP: Direction = (-1, 0)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
RIGHT: Direction = (0, 1)
DIRECTIONS: tuple[Direction, ...] = (UP, DOWN, LEFT, RIGHT)
DIR_SET = set(DIRECTIONS)


def _outer_band(shape: tuple[int, int]) -> np.ndarray:
    """Mask the outer two rings of a grid."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:2, :] = True
    mask[-2:, :] = True
    mask[:, :2] = True
    mask[:, -2:] = True
    return mask


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as (N, 2) arrays of (r, c)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue: deque[Coord] = deque([(r0, c0)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _parse_examples(g: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[Direction, int]]]:
    """Read isolated two-pixel examples from the outer two rings."""
    h, w = g.shape
    assert h >= 7 and w >= 7, "the outer examples need a nonempty interior"
    band = _outer_band(g.shape)
    example_mask = band & (g != 0)
    components = _mask_components(example_mask)
    assert len(components) >= 2, "expected several edge examples"

    # The source endpoint is visibly identifiable: its color occurs on an
    # internal object, whereas the landing color occurs only in the examples.
    interior_colors = set(map(int, np.unique(g[2:-2, 2:-2]))) - {0}
    rules: dict[int, tuple[Direction, int]] = {}
    target_colors: set[int] = set()
    for coords in components:
        assert len(coords) == 2, "each edge example must be one isolated adjacent pair"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        ca, cb = int(g[a]), int(g[b])
        assert ca != cb, "the two example colors must differ"
        a_is_source = ca in interior_colors
        b_is_source = cb in interior_colors
        assert a_is_source != b_is_source, "exactly one endpoint color must occur internally"
        src_pos, dst_pos = (a, b) if a_is_source else (b, a)
        source, target = int(g[src_pos]), int(g[dst_pos])
        direction = (dst_pos[0] - src_pos[0], dst_pos[1] - src_pos[1])
        assert direction in DIR_SET, "example endpoints must be orthogonally adjacent"
        assert source not in rules, "each source color must have one unambiguous example"
        assert target not in interior_colors, "landing colors must not masquerade as object colors"
        rules[source] = (direction, target)
        target_colors.add(target)

    assert len(target_colors) == len(rules), "landing colors must pair one-to-one with source colors"
    return example_mask, rules


def _monochrome_objects(g: np.ndarray) -> list[dict]:
    """Find all nonblack monochrome 4-connected objects."""
    objects: list[dict] = []
    for color in set(map(int, np.unique(g))) - {0}:
        for coords in _mask_components(g == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return objects


def _terminal_order(obj: dict, direction: Direction, shape: tuple[int, int]) -> tuple[int, ...]:
    """Order by distance to the directed grid edge, then only by visible coordinates."""
    r0, c0, r1, c1 = obj["bbox"]
    h, w = shape
    if direction == UP:
        distance = r0
        tie = (c0, c1, r0, r1)
    elif direction == DOWN:
        distance = h - 1 - r1
        tie = (c0, c1, -r1, -r0)
    elif direction == LEFT:
        distance = c0
        tie = (r0, r1, c0, c1)
    else:
        assert direction == RIGHT
        distance = w - 1 - c1
        tie = (r0, r1, -c1, -c0)
    return (int(distance), *map(int, tie))


def _settle_direction(
    objects: list[dict], direction: Direction, fixed: np.ndarray
) -> list[tuple[dict, np.ndarray]]:
    """Slide one direction group against fixed and already settled objects."""
    h, w = fixed.shape
    dr, dc = direction
    occupied = fixed.copy()
    placements: list[tuple[dict, np.ndarray]] = []
    for obj in sorted(objects, key=lambda x: _terminal_order(x, direction, fixed.shape)):
        coords = np.asarray(obj["coords"], dtype=int)
        shift = 0
        while True:
            candidate = coords + np.array([(shift + 1) * dr, (shift + 1) * dc], dtype=int)
            rr, cc = candidate[:, 0], candidate[:, 1]
            if rr.min() < 0 or cc.min() < 0 or rr.max() >= h or cc.max() >= w:
                break
            if bool(occupied[rr, cc].any()):
                break
            shift += 1
        placed = coords + np.array([shift * dr, shift * dc], dtype=int)
        assert shift > 0, "every example-controlled object must visibly slide"
        assert not bool(occupied[placed[:, 0], placed[:, 1]].any())
        occupied[placed[:, 0], placed[:, 1]] = True
        placements.append((obj, placed))
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase edge examples, slide their matching objects in the indicated directions, and recolor them with the paired colors.

    Input structure:
    - Black is the background; monochrome 4-connected objects lie outside the outer two rings.
    - Each isolated adjacent pair in the outer two rings contains one color used by internal objects and one color absent internally.
    - Unpaired internal colors are stationary obstacles.

    Rule:
    - In each pair, the internal-object color is X, the other color is Y, and the step from X to Y gives the slide direction.
    - Remove all example pixels and all moving objects, then, within each direction, slide nearer-to-that-edge objects first until the next step would cross the grid or hit a fixed/already settled object.
    - Preserve every object rigidly, repaint each settled X object as Y, and leave unpaired objects fixed.

    Key insight:
    - The edge pairs are executable example sentences: endpoint identity supplies the subject color, adjacency supplies the verb direction, and the second color supplies the final paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"

    example_mask, rules = _parse_examples(g)
    body = g.copy()
    body[example_mask] = 0
    assert not bool((body[_outer_band(g.shape)] != 0).any()), "only examples may occupy the outer two rings"

    objects = _monochrome_objects(body)
    source_colors = set(rules)
    present_sources = {int(obj["color"]) for obj in objects if int(obj["color"]) in source_colors}
    assert present_sources == source_colors, "every example source color needs an internal object"

    moving = [obj for obj in objects if int(obj["color"]) in source_colors]
    fixed = body != 0
    for obj in moving:
        coords = obj["coords"]
        fixed[coords[:, 0], coords[:, 1]] = False

    out = body.copy()
    for obj in moving:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = 0

    all_placed = fixed.copy()
    moved_count = 0
    for direction in DIRECTIONS:
        group = [obj for obj in moving if rules[int(obj["color"])][0] == direction]
        if not group:
            continue
        # Different direction groups are required to have independent corridors;
        # only same-direction settled objects enter this group's collision order.
        for obj, placed in _settle_direction(group, direction, fixed):
            rr, cc = placed[:, 0], placed[:, 1]
            assert not bool(all_placed[rr, cc].any()), "different direction corridors must not overlap"
            target = int(rules[int(obj["color"])][1])
            out[rr, cc] = target
            all_placed[rr, cc] = True
            moved_count += 1

    assert moved_count == len(moving)
    return out


# ------------------------------ generator ------------------------------

GRID_MIN = 19
GRID_MAX = 27
INNER_MARGIN = 3
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _expanded(cells: set[Coord], shape: tuple[int, int], padding: int = 1) -> set[Coord]:
    result: set[Coord] = set()
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                cell = (r + dr, c + dc)
                if _in_bounds(shape, cell):
                    result.add(cell)
    return result


def _cross_cells(axis: int, orth0: int, length: int, direction: Direction) -> set[Coord]:
    if direction[0] == 0:
        return {(orth0 + k, axis) for k in range(length)}
    return {(axis, orth0 + k) for k in range(length)}


def _translate_cells(cells: set[Coord], direction: Direction, steps: int) -> set[Coord]:
    dr, dc = direction
    return {(r + steps * dr, c + steps * dc) for r, c in cells}


def _group_candidate(
    shape: tuple[int, int], direction: Direction, stop_kind: str, reserved: set[Coord]
) -> dict | None:
    """Construct one two-object corridor whose correct order is visibly necessary."""
    h, w = shape
    horizontal = direction[0] == 0
    axis_len = w if horizontal else h
    orth_len = h if horizontal else w
    step = direction[1] if horizontal else direction[0]

    front_len = random.choice((2, 2, 3))
    orth_choices = list(range(INNER_MARGIN, orth_len - INNER_MARGIN - front_len + 1))
    random.shuffle(orth_choices)
    if not orth_choices:
        return None

    if stop_kind == "edge":
        terminal = axis_len - 1 if step == 1 else 0
        axis_choices = [terminal]
    else:
        if step == 1:
            axis_choices = list(range(INNER_MARGIN + 6, axis_len - INNER_MARGIN))
        else:
            axis_choices = list(range(INNER_MARGIN, axis_len - INNER_MARGIN - 6))
        random.shuffle(axis_choices)

    for orth0 in orth_choices:
        rear_offset = random.randrange(front_len)
        for terminal in axis_choices:
            if stop_kind == "edge":
                front_final_axis = terminal
                obstacle: set[Coord] = set()
            else:
                obstacle = _cross_cells(terminal, orth0, front_len, direction)
                front_final_axis = terminal - step

            front_initial_axis = terminal - 3 * step
            rear_initial_axis = terminal - 6 * step
            rear_final_axis = terminal - 2 * step if stop_kind == "block" else terminal - step

            front_initial = _cross_cells(front_initial_axis, orth0, front_len, direction)
            front_final = _cross_cells(front_final_axis, orth0, front_len, direction)
            rear_initial = _cross_cells(rear_initial_axis, orth0 + rear_offset, 1, direction)
            rear_final = _cross_cells(rear_final_axis, orth0 + rear_offset, 1, direction)

            all_key_cells = front_initial | front_final | rear_initial | rear_final | obstacle
            if not all(_in_bounds(shape, cell) for cell in all_key_cells):
                continue
            # Input objects and fixed blockers stay outside the outer two rings.
            input_cells = front_initial | rear_initial | obstacle
            if any(
                min(r, c, h - 1 - r, w - 1 - c) < 2
                for r, c in input_cells
            ):
                continue

            front_steps = abs(front_final_axis - front_initial_axis)
            rear_steps = abs(rear_final_axis - rear_initial_axis)
            sweep = obstacle.copy()
            for k in range(front_steps + 1):
                sweep |= _translate_cells(front_initial, direction, k)
            for k in range(rear_steps + 1):
                sweep |= _translate_cells(rear_initial, direction, k)
            if sweep & reserved:
                continue

            # Reversing near/far order swaps the long bar and singleton positions.
            wrong = _cross_cells(rear_final_axis, orth0, front_len, direction)
            wrong |= _cross_cells(front_final_axis, orth0 + rear_offset, 1, direction)
            correct = front_final | rear_final
            assert wrong != correct
            return {
                "direction": direction,
                "stop_kind": stop_kind,
                "front_initial": front_initial,
                "rear_initial": rear_initial,
                "front_final": front_final,
                "rear_final": rear_final,
                "obstacle": obstacle,
                "sweep": sweep,
                "wrong_final": wrong,
            }
    return None


def _marker_candidates(
    shape: tuple[int, int], direction: Direction, forbidden: set[Coord]
) -> list[tuple[Coord, Coord]]:
    h, w = shape
    band = _outer_band(shape)
    dr, dc = direction
    candidates: list[tuple[Coord, Coord]] = []
    for r in range(h):
        for c in range(w):
            y = (r + dr, c + dc)
            if not _in_bounds(shape, y):
                continue
            if not band[r, c] or not band[y]:
                continue
            if min(r, c, h - 1 - r, w - 1 - c, y[0], y[1], h - 1 - y[0], w - 1 - y[1]) != 0:
                continue
            if (r, c) in forbidden or y in forbidden:
                continue
            candidates.append(((r, c), y))
    random.shuffle(candidates)
    return candidates


def _semantic_accepts(inp: np.ndarray, expected: np.ndarray, groups: list[dict]) -> bool:
    """Check rule identity, branch coverage, order dependence, and oracle output."""
    try:
        marker_mask, rules = _parse_examples(inp)
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    if {g["stop_kind"] for g in groups} != {"edge", "block"}:
        return False
    if len({g["direction"] for g in groups}) < 2:
        return False
    if any((g["front_final"] | g["rear_final"]) == g["wrong_final"] for g in groups):
        return False
    if any(not g["obstacle"] for g in groups if g["stop_kind"] == "block"):
        return False
    if any(int(expected[r, c]) != 0 for r, c in np.argwhere(marker_mask)):
        return False
    expected_rules = {
        int(g["source"]): (g["direction"], int(g["target"])) for g in groups
    }
    return rules == expected_rules


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    n_rules = random.randint(2, 4)

    directions = random.sample(list(DIRECTIONS), n_rules)
    stop_kinds = ["edge", "block"] + [random.choice(("edge", "block")) for _ in range(n_rules - 2)]
    random.shuffle(stop_kinds)

    palette = random.sample(list(range(1, 10)), 2 * n_rules + 1)
    sources = palette[:n_rules]
    targets = palette[n_rules : 2 * n_rules]
    obstacle_color = palette[-1]

    groups: list[dict] = []
    reserved: set[Coord] = set()
    for direction, stop_kind, source, target in zip(directions, stop_kinds, sources, targets):
        group = _group_candidate(shape, direction, stop_kind, reserved)
        if group is None:
            return None
        group["source"] = int(source)
        group["target"] = int(target)
        groups.append(group)
        reserved |= _expanded(group["sweep"], shape, padding=1)

    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    for group in groups:
        source = int(group["source"])
        target = int(group["target"])
        for r, c in group["front_initial"] | group["rear_initial"]:
            inp[r, c] = source
        for r, c in group["front_final"] | group["rear_final"]:
            expected[r, c] = target
        for r, c in group["obstacle"]:
            inp[r, c] = obstacle_color
            expected[r, c] = obstacle_color

    # Keep examples isolated from one another and away from the visible landing cells.
    output_foreground = {tuple(map(int, rc)) for rc in np.argwhere(expected != 0)}
    marker_forbidden = _expanded(output_foreground, shape, padding=1)
    used_marker_cells: set[Coord] = set()
    for group in groups:
        forbidden = marker_forbidden | _expanded(used_marker_cells, shape, padding=1)
        candidates = _marker_candidates(shape, group["direction"], forbidden)
        if not candidates:
            return None
        x, y = candidates[0]
        inp[x] = int(group["source"])
        inp[y] = int(group["target"])
        used_marker_cells |= {x, y}

    if np.count_nonzero(inp == 0) <= inp.size // 2:
        return None
    return inp if _semantic_accepts(inp, expected, groups) else None


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance of the edge-example task."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid edge-example dictionary input")
