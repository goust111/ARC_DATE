from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


TEAL = 8
MAROON = 9
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following the ARC background convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    """Monochromatic 4-connected foreground components in (row, column) coordinates."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if seen[r0, c0] or color == bg:
                continue
            seen[r0, c0] = True
            queue = deque([(r0, c0)])
            cells: set[tuple[int, int]] = set()
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr, nc]
                            and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append({"color": color, "cells": cells})
    return result


def _launcher_direction(
    grid: np.ndarray, cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], int]:
    """Infer the outward direction and marker color from one marked side."""
    h, w = grid.shape
    contacts: set[tuple[tuple[int, int], int]] = set()
    for r, c in cells:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                marker = int(grid[nr, nc])
                if marker in (TEAL, MAROON):
                    contacts.add(((-dr, -dc), marker))
    assert contacts, "each object must touch a fixed marker"
    directions = {direction for direction, _ in contacts}
    colors = {color for _, color in contacts}
    assert len(directions) == 1, "an object marker must touch exactly one cardinal side"
    assert len(colors) == 1, "one object cannot have two marker colors"
    return next(iter(directions)), next(iter(colors))


def _edge_steps(
    shape: tuple[int, int], cells: set[tuple[int, int]], direction: tuple[int, int]
) -> int:
    h, w = shape
    dr, dc = direction
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if dr < 0:
        return min(rows)
    if dr > 0:
        return h - 1 - max(rows)
    if dc < 0:
        return min(cols)
    return w - 1 - max(cols)


def _shift(
    cells: set[tuple[int, int]], direction: tuple[int, int], steps: int
) -> set[tuple[int, int]]:
    dr, dc = direction
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _relay(
    shape: tuple[int, int], objects: list[dict], source: int
) -> tuple[list[set[tuple[int, int]]], list[int], list[int]]:
    """Simulate the collision relay and return positions, visit order, and moves."""
    positions = [set(obj["cells"]) for obj in objects]
    order: list[int] = []
    moves: list[int] = []
    active = source
    visited: set[int] = set()

    while True:
        assert active not in visited, "the collision relay must not contain a cycle"
        visited.add(active)
        order.append(active)
        direction = objects[active]["direction"]
        limit = _edge_steps(shape, positions[active], direction)
        collision: tuple[int, int] | None = None

        for step in range(1, limit + 1):
            trial = _shift(positions[active], direction, step)
            blockers = [
                i for i, occupied in enumerate(positions)
                if i != active and bool(trial & occupied)
            ]
            if blockers:
                assert len(blockers) == 1, "the first collision must identify one struck object"
                collision = (step, blockers[0])
                break

        if collision is None:
            positions[active] = _shift(positions[active], direction, limit)
            moves.append(limit)
            break

        hit_step, struck = collision
        travel = hit_step - 1
        positions[active] = _shift(positions[active], direction, travel)
        moves.append(travel)
        assert not any(
            positions[active] & positions[i]
            for i in range(len(positions)) if i != active
        )
        active = struck

    return positions, order, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay a push from the uniquely teal-marked object through the objects it strikes.

    Input structure:
    - The background is the most frequent color on a grid of monochromatic 4-connected objects.
    - Every object has a fixed teal or maroon marker touching exactly one cardinal side; exactly one object's marker is teal.
    - The marker contact side defines the object's potential launch direction, directly in grid row-column coordinates.

    Rule:
    - Launch only the teal-marked object away from its marker and translate its complete shape one cell at a time.
    - At the first object in its path, stop one step before overlap and launch only that struck object in the direction defined by its own marker.
    - Continue the collision relay until the active object meets no object and reaches the grid edge; keep every marker and every unstruck object fixed.

    Key insight:
    - Marker color selects the unique first mover, while each struck object's marker side redirects the relay; maroon-marked objects outside that collision chain are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 by 30"
    bg = _background(g)
    assert bg not in (TEAL, MAROON), "marker colors cannot be the background"

    objects: list[dict] = []
    for comp in _components(g, bg):
        if comp["color"] in (TEAL, MAROON):
            continue
        direction, marker = _launcher_direction(g, comp["cells"])
        objects.append({
            "color": int(comp["color"]),
            "cells": set(comp["cells"]),
            "direction": direction,
            "marker": marker,
        })

    assert objects, "expected marked foreground objects"
    sources = [i for i, obj in enumerate(objects) if obj["marker"] == TEAL]
    assert len(sources) == 1, "exactly one object must have a teal marker"
    positions, _, _ = _relay(g.shape, objects, sources[0])

    out = g.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = bg

    occupied: set[tuple[int, int]] = set()
    for obj, cells in zip(objects, positions):
        assert not (occupied & cells), "final objects must not overlap"
        occupied |= cells
        for r, c in cells:
            assert int(out[r, c]) not in (TEAL, MAROON), "moving objects cannot cover fixed markers"
            out[r, c] = obj["color"]
    return out


# ---------------------------- generator ----------------------------


def _rect(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)}


def _inside(cells: set[tuple[int, int]], h: int, w: int, margin: int = 0) -> bool:
    return all(margin <= r < h - margin and margin <= c < w - margin for r, c in cells)


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 1) -> bool:
    """Whether two masks are closer than the requested Manhattan-space gap."""
    return any(abs(ar - br) + abs(ac - bc) <= gap for ar, ac in a for br, bc in b)


def _next_rect(
    current: set[tuple[int, int]], direction: tuple[int, int], shape: tuple[int, int],
    distance: int, rng: random.Random,
) -> set[tuple[int, int]]:
    """Place a rectangle so current first overlaps it at exactly distance steps."""
    oh, ow = shape
    rows = [r for r, _ in current]
    cols = [c for _, c in current]
    dr, dc = direction
    if dr:
        overlap_col = rng.randint(min(cols), max(cols))
        c0 = overlap_col - rng.randrange(ow)
        r0 = max(rows) + distance if dr > 0 else min(rows) - distance - oh + 1
    else:
        overlap_row = rng.randint(min(rows), max(rows))
        r0 = overlap_row - rng.randrange(oh)
        c0 = max(cols) + distance if dc > 0 else min(cols) - distance - ow + 1
    return _rect(r0, c0, oh, ow)


def _marker_candidates(
    cells: set[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    """Cells directly behind the object's launch-facing opposite side."""
    dr, dc = direction
    if dr < 0:
        edge = max(r for r, _ in cells)
        return [(edge + 1, c) for r, c in cells if r == edge]
    if dr > 0:
        edge = min(r for r, _ in cells)
        return [(edge - 1, c) for r, c in cells if r == edge]
    if dc < 0:
        edge = max(c for _, c in cells)
        return [(r, edge + 1) for r, c in cells if c == edge]
    edge = min(c for _, c in cells)
    return [(r, edge - 1) for r, c in cells if c == edge]


def _swept_positions(
    cells: set[tuple[int, int]], direction: tuple[int, int], steps: int
) -> set[tuple[int, int]]:
    swept: set[tuple[int, int]] = set()
    for step in range(1, steps + 1):
        swept |= _shift(cells, direction, step)
    return swept


def _render(
    h: int, w: int, bg: int, objects: list[dict], markers: list[tuple[int, int]]
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = obj["color"]
    for i, (r, c) in enumerate(markers):
        grid[r, c] = TEAL if i == 0 else MAROON
    return grid


def _semantic_checks(
    grid: np.ndarray, objects: list[dict], chain_length: int,
    expected_positions: list[set[tuple[int, int]]], order: list[int], moves: list[int],
) -> bool:
    """Check rule dependencies, uniqueness, non-degeneracy, and an independent render oracle."""
    if order != list(range(chain_length)) or len(order) != chain_length:
        return False
    if any(step < 1 for step in moves):
        return False
    chain_dirs = [objects[i]["direction"] for i in range(chain_length)]
    if not any(a != b for a, b in zip(chain_dirs, chain_dirs[1:])):
        return False
    if not any(a[0] != b[0] and a[1] != b[1] for a, b in zip(chain_dirs, chain_dirs[1:])):
        return False

    expected = grid.copy()
    bg = _background(grid)
    for obj in objects:
        for r, c in obj["cells"]:
            expected[r, c] = bg
    used: set[tuple[int, int]] = set()
    for obj, cells in zip(objects, expected_positions):
        if used & cells:
            return False
        used |= cells
        for r, c in cells:
            if int(expected[r, c]) in (TEAL, MAROON):
                return False
            expected[r, c] = obj["color"]
    if not np.array_equal(main(grid), expected):
        return False

    # A source-only shortcut must miss at least the second and third movers.
    if any(expected_positions[i] == objects[i]["cells"] for i in range(chain_length)):
        return False
    # An "all marked objects move" shortcut must alter a chain-external distractor.
    distractors = objects[chain_length:]
    if len(distractors) < 2 or not any(
        _edge_steps(grid.shape, obj["cells"], obj["direction"]) >= 2
        for obj in distractors
    ):
        return False
    # The teal choice is visibly necessary: moving it to a distractor changes the result.
    teal = tuple(map(int, np.argwhere(grid == TEAL)[0]))
    distractor_marker = None
    for r, c in np.argwhere(grid == MAROON):
        rc = (int(r), int(c))
        if any(abs(rc[0] - rr) + abs(rc[1] - cc) == 1
               for obj in distractors for rr, cc in obj["cells"]):
            distractor_marker = rc
            break
    if distractor_marker is None:
        return False
    alternate = grid.copy()
    alternate[teal] = MAROON
    alternate[distractor_marker] = TEAL
    try:
        if np.array_equal(main(alternate), expected):
            return False
    except AssertionError:
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 25)
    w = rng.randint(16, 25)
    bg = rng.choice((0, 5, 6, 7))
    chain_length = rng.randint(3, 4)
    distractor_count = rng.randint(2, 3)
    total = chain_length + distractor_count
    palette = [color for color in range(8) if color != bg]
    rng.shuffle(palette)
    colors = palette[:total]

    directions: list[tuple[int, int]] = [rng.choice(DIRECTIONS)]
    for _ in range(1, chain_length):
        previous = directions[-1]
        choices = [d for d in DIRECTIONS if d != (-previous[0], -previous[1])]
        directions.append(rng.choice(choices))
    if not any(a[0] != b[0] and a[1] != b[1]
               for a, b in zip(directions, directions[1:])):
        return None

    sh, sw = rng.choice(SHAPES)
    r0 = rng.randint(4, h - sh - 5)
    c0 = rng.randint(4, w - sw - 5)
    chain = [_rect(r0, c0, sh, sw)]
    for i in range(chain_length - 1):
        placed = None
        for _ in range(50):
            shape = rng.choice(SHAPES)
            distance = rng.randint(3, 6)
            candidate = _next_rect(chain[-1], directions[i], shape, distance, rng)
            if not _inside(candidate, h, w, margin=2):
                continue
            if any(_touches(candidate, old, gap=1) for old in chain):
                continue
            placed = candidate
            break
        if placed is None:
            return None
        chain.append(placed)

    objects = [
        {"color": colors[i], "cells": chain[i], "direction": directions[i],
         "marker": TEAL if i == 0 else MAROON}
        for i in range(chain_length)
    ]

    # Validate the designed chain before adding objects that must remain irrelevant.
    try:
        chain_positions, order, moves = _relay((h, w), objects, 0)
    except AssertionError:
        return None
    if order != list(range(chain_length)) or len(moves) != chain_length:
        return None
    if any(step < 1 for step in moves):
        return None
    swept: set[tuple[int, int]] = set()
    for i in range(chain_length):
        swept |= _swept_positions(chain[i], directions[i], moves[i])

    # Place marked distractors away from every real relay trajectory.
    for i in range(distractor_count):
        placed_obj = None
        for _ in range(150):
            oh, ow = rng.choice(SHAPES)
            rr = rng.randint(2, h - oh - 3)
            cc = rng.randint(2, w - ow - 3)
            cells = _rect(rr, cc, oh, ow)
            direction = rng.choice(DIRECTIONS)
            if cells & swept:
                continue
            if any(_touches(cells, obj["cells"], gap=2) for obj in objects):
                continue
            if _edge_steps((h, w), cells, direction) < 2:
                continue
            placed_obj = {
                "color": colors[chain_length + i], "cells": cells,
                "direction": direction, "marker": MAROON,
            }
            break
        if placed_obj is None:
            return None
        objects.append(placed_obj)

    all_object_cells = set().union(*(obj["cells"] for obj in objects))
    markers: list[tuple[int, int]] = []
    for owner, obj in enumerate(objects):
        candidates = _marker_candidates(obj["cells"], obj["direction"])
        rng.shuffle(candidates)
        chosen = None
        for marker in candidates:
            if not _inside({marker}, h, w):
                continue
            if marker in all_object_cells or marker in markers or marker in swept:
                continue
            adjacent_owners = {
                j for j, other in enumerate(objects)
                if any(abs(marker[0] - r) + abs(marker[1] - c) == 1
                       for r, c in other["cells"])
            }
            if adjacent_owners == {owner}:
                chosen = marker
                break
        if chosen is None:
            return None
        markers.append(chosen)

    # Fixed markers must also stay clear of every designed final object.
    try:
        expected_positions, order, moves = _relay((h, w), objects, 0)
    except AssertionError:
        return None
    if any(marker in cells for marker in markers for cells in expected_positions):
        return None
    grid = _render(h, w, bg, objects, markers)

    # Parsing must recover exactly one direction and marker color per object.
    try:
        parsed = []
        for comp in _components(grid, bg):
            if comp["color"] not in (TEAL, MAROON):
                parsed.append(_launcher_direction(grid, comp["cells"]))
        if len(parsed) != total:
            return None
        if sum(marker == TEAL for _, marker in parsed) != 1:
            return None
        if not _semantic_checks(
            grid, objects, chain_length, expected_positions, order, moves
        ):
            return None
    except (AssertionError, IndexError):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, unambiguous collision relay with chain-external distractors."""
    rng = random.Random()
    for _ in range(1000):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teal collision-relay input")
