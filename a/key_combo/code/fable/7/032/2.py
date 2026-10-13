"""ARC task: use scattered pixels as measured liquid for matching cups."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components using the common ARC (row, column) convention."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[list[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        component = [start]
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    component.append(nxt)
        result.append(component)
    return result


def _cup_geometry(
    closed_end: tuple[int, int], opening_direction: tuple[int, int], capacity: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Return (border, cavity), with the cavity ordered closed-end to opening."""
    qr, qc = closed_end
    dr, dc = opening_direction
    pr, pc = -dc, dr
    cavity = [(qr + i * dr, qc + i * dc) for i in range(capacity)]
    border = {
        (r + side * pr, c + side * pc)
        for r, c in cavity
        for side in (-1, 1)
    }
    cap_center = (qr - dr, qc - dc)
    border.update(
        (cap_center[0] + side * pr, cap_center[1] + side * pc)
        for side in (-1, 0, 1)
    )
    return border, cavity


def _decode_cup(component: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Decode an exact one-cell-wide U border into its ordered cavity and opening vector."""
    cells = set(component)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)

    candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    for direction in DIRECTIONS:
        dr, dc = direction
        if dr:
            if c1 - c0 + 1 != 3:
                continue
            capacity = r1 - r0
            closed = (r1 - 1, (c0 + c1) // 2) if dr < 0 else (r0 + 1, (c0 + c1) // 2)
        else:
            if r1 - r0 + 1 != 3:
                continue
            capacity = c1 - c0
            closed = ((r0 + r1) // 2, c1 - 1) if dc < 0 else ((r0 + r1) // 2, c0 + 1)
        if capacity < 2:
            continue
        border, cavity = _cup_geometry(closed, direction, capacity)
        if border == cells:
            candidates.append((cavity, direction))

    assert len(candidates) == 1, "each non-singleton color object must be one unambiguous cup"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Pour each color's scattered-pixel count into its matching oriented cup.

    Input structure:
    - A black grid contains several differently colored cup borders and isolated
      pixels in the same colors.  Each cup is an exact one-cell-wide U shape
      whose straight cavity has a uniquely visible opening direction.
    - Every border color occurs on exactly one cup; its other occurrences are
      isolated scattered pixels, and all required overflow paths remain clear.

    Rule:
    - Count and erase the isolated pixels of each color.  Starting at the closed
      end, paint that many cells along the matching cup's cavity toward its
      opening.  Continue on the same axis beyond the opening for any excess,
      while preserving every border pixel.

    Key insight:
    - The scattered pixels encode liquid amount, while the U border independently
      encodes capacity, closed-end start position, and overflow direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    out = g.copy()
    colors = [int(x) for x in np.unique(g) if int(x) != BACKGROUND]
    assert colors, "expected at least one cup"

    decoded: list[tuple[int, list[tuple[int, int]], tuple[int, int], int]] = []
    for color in colors:
        comps = _components(g == color)
        cups = [comp for comp in comps if len(comp) > 1]
        dots = [comp[0] for comp in comps if len(comp) == 1]
        assert len(cups) == 1, "each color must identify exactly one cup"
        assert dots, "each cup color must have at least one measured pixel"
        cavity, direction = _decode_cup(cups[0])
        decoded.append((color, cavity, direction, len(dots)))
        for r, c in dots:
            out[r, c] = BACKGROUND

    # Pour only after all counters have been erased, so every target cell must
    # be genuinely empty rather than another color's measurement marker.
    for color, cavity, (dr, dc), amount in decoded:
        path = list(cavity)
        r, c = cavity[-1]
        for step in range(1, max(0, amount - len(cavity)) + 1):
            path.append((r + step * dr, c + step * dc))
        assert len(path) >= amount
        for r, c in path[:amount]:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "overflow leaves grid"
            assert out[r, c] == BACKGROUND, "liquid path must be unobstructed"
            out[r, c] = color
    return out


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    return {
        (r + dr, c + dc)
        for r, c in cells
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 0 <= r + dr < h and 0 <= c + dc < w
    }


def _expected(
    grid: np.ndarray,
    specs: list[tuple[int, set[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]],
    dots: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    """Independent construction oracle used to validate the generated semantics."""
    out = grid.copy()
    for points in dots.values():
        for r, c in points:
            out[r, c] = BACKGROUND
    for color, _border, cavity, (dr, dc), amount in specs:
        path = list(cavity)
        r, c = cavity[-1]
        path.extend((r + i * dr, c + i * dc) for i in range(1, amount - len(cavity) + 1))
        for r, c in path[:amount]:
            out[r, c] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    shape = (height, width)
    cup_count = random.choice((3, 4))
    colors = random.sample(PALETTE, cup_count)

    # Every grid contains all three semantic branches.  Their assignment to
    # colors and positions is shuffled, so no color or location carries meaning.
    branches = ["under", "equal", "over"]
    if cup_count == 4:
        branches.append(random.choice(branches))
    random.shuffle(branches)

    specs: list[tuple[int, set[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]] = []
    forbidden: set[tuple[int, int]] = set()
    for color, branch in zip(colors, branches):
        placed = False
        for _ in range(400):
            capacity = random.randint(2, 6)
            if branch == "under":
                amount = random.randint(1, capacity - 1)
            elif branch == "equal":
                amount = capacity
            else:
                amount = capacity + random.randint(1, 3)
            direction = random.choice(DIRECTIONS)
            closed = (random.randrange(height), random.randrange(width))
            border, cavity = _cup_geometry(closed, direction, capacity)
            dr, dc = direction
            opening = cavity[-1]
            spill = {
                (opening[0] + i * dr, opening[1] + i * dc)
                for i in range(1, max(0, amount - capacity) + 1)
            }
            zone = border | set(cavity) | spill
            if all(0 <= r < height and 0 <= c < width for r, c in zone) and not (zone & forbidden):
                specs.append((color, border, cavity, direction, amount))
                forbidden |= _halo(zone, shape, radius=1)
                placed = True
                break
        if not placed:
            return None

    grid = np.zeros(shape, dtype=int)
    for color, border, _cavity, _direction, _amount in specs:
        for r, c in border:
            grid[r, c] = color

    dots: dict[int, list[tuple[int, int]]] = {}
    dot_forbidden = set(forbidden)
    for color, _border, _cavity, _direction, amount in specs:
        points: list[tuple[int, int]] = []
        for _ in range(amount):
            choices = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if (r, c) not in dot_forbidden
            ]
            if not choices:
                return None
            point = random.choice(choices)
            points.append(point)
            dot_forbidden |= _halo({point}, shape, radius=1)
            grid[point] = color
        dots[color] = points

    # Semantic checks: exact counts, three branches, nontrivial erase and fill,
    # unique cup decoding, and equality to an independently rendered oracle.
    capacities = {color: len(cavity) for color, _b, cavity, _d, _a in specs}
    relations = {
        "under" if amount < capacities[color] else "equal" if amount == capacities[color] else "over"
        for color, _b, _c, _d, amount in specs
    }
    if relations != {"under", "equal", "over"}:
        return None
    expected = _expected(grid, specs, dots)
    try:
        actual = main(grid)
    except (AssertionError, IndexError):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked multi-cup instance."""
    for _ in range(200):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-and-overflow instance")
