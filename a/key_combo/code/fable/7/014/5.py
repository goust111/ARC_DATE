"""ARC task: rotate each holed object around its own grey pivot."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
GREY = 5
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask in (row, column) coordinates."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[set[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        result.append(comp)
    return result


def _holes(shape: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Find background regions enclosed by one component, using only that object as reference."""
    r0 = min(r for r, _ in shape) - 1
    r1 = max(r for r, _ in shape) + 1
    c0 = min(c for _, c in shape) - 1
    c1 = max(c for _, c in shape) + 1
    empty = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in shape
    }
    regions: list[set[tuple[int, int]]] = []
    while empty:
        seed = empty.pop()
        region = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if q in empty:
                    empty.remove(q)
                    region.add(q)
                    queue.append(q)
        regions.append(region)
    return [
        region
        for region in regions
        if not any(r in (r0, r1) or c in (c0, c1) for r, c in region)
    ]


def _rotate_point(point: tuple[int, int], pivot: tuple[int, int], turns: int) -> tuple[int, int]:
    """Rotate a point by signed quarter-turns; positive is clockwise in row/column space."""
    r, c = point
    pr, pc = pivot
    dr, dc = r - pr, c - pc
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return pr + dr, pc + dc


def _pair_objects(
    shapes: list[set[tuple[int, int]]], pivots: set[tuple[int, int]]
) -> list[tuple[set[tuple[int, int]], tuple[int, int], int, bool]]:
    """Pair shapes and pivots by hole containment or unique outside 4-contact."""
    candidates: list[list[tuple[tuple[int, int], int, bool]]] = []
    for shape in shapes:
        hole_regions = _holes(shape)
        inside = {p for region in hole_regions for p in region}
        matches = []
        for pivot in pivots:
            internal = pivot in inside
            external = (not internal) and any(
                abs(pivot[0] - r) + abs(pivot[1] - c) == 1 for r, c in shape
            )
            if internal or external:
                matches.append((pivot, len(hole_regions), internal))
        candidates.append(matches)

    # The stated inputs make the pairing unique.  A small matching search avoids
    # accidentally resolving geometric ambiguity by scan order.
    solutions: list[list[tuple[set[tuple[int, int]], tuple[int, int], int, bool]]] = []

    def search(i: int, used: set[tuple[int, int]], acc: list) -> None:
        if len(solutions) > 1:
            return
        if i == len(shapes):
            if used == pivots:
                solutions.append(acc.copy())
            return
        for pivot, n_holes, internal in candidates[i]:
            if pivot not in used:
                used.add(pivot)
                acc.append((shapes[i], pivot, n_holes, internal))
                search(i + 1, used, acc)
                acc.pop()
                used.remove(pivot)

    search(0, set(), [])
    assert len(solutions) == 1, "each shape must have exactly one unambiguous grey pivot"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every colored shape around its grey pivot by a hole-count-controlled angle and direction.

    Input structure:
    - A black background contains several mutually separate monochrome 4-connected shapes.
    - Each shape encloses zero to three background holes and has exactly one grey pivot pixel.
    - Its pivot is either inside one of its holes or 4-adjacent to the shape on the outside.

    Rule:
    - Count a shape's enclosed holes and rotate all of its cells around its paired grey pivot
      by that many quarter-turns: counterclockwise for an in-hole pivot, clockwise for an
      outside pivot. Clear the old cells, retain every pivot, and leave zero-hole shapes fixed.

    Key insight:
    - Hole count determines the angle, while the pivot's inside/outside relation independently
      determines direction; both cues must be interpreted in the coordinate frame of that shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette values"
    assert np.any(g == GREY), "expected grey pivots"

    shape_mask = (g != BACKGROUND) & (g != GREY)
    shapes = _components(shape_mask)
    pivots = {tuple(map(int, p)) for p in np.argwhere(g == GREY)}
    assert shapes and len(shapes) == len(pivots)
    for shape in shapes:
        colors = {int(g[p]) for p in shape}
        assert len(colors) == 1, "each shape must be monochrome"

    assignments = _pair_objects(shapes, pivots)
    out = g.copy()
    out[shape_mask] = BACKGROUND
    occupied: set[tuple[int, int]] = set(pivots)
    h, w = g.shape
    for shape, pivot, n_holes, internal in assignments:
        assert 0 <= n_holes <= 3
        turns = -n_holes if internal else n_holes
        moved = {_rotate_point(p, pivot, turns) for p in shape}
        assert all(0 <= r < h and 0 <= c < w for r, c in moved), "rotation must stay in bounds"
        assert not (moved & occupied), "rotated objects and pivots must not overlap"
        color = int(g[next(iter(shape))])
        for p in moved:
            out[p] = color
        occupied |= moved
    return out


def _base_shape(n_holes: int) -> set[tuple[int, int]]:
    """An asymmetric connected glyph with exactly n one-cell holes."""
    if n_holes == 0:
        return {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 1)}
    width = 2 * n_holes + 1
    cells = {(1, c) for c in range(width)} | {(3, c) for c in range(width)}
    cells |= {(2, c) for c in range(0, width, 2)}
    cells.add((0, 0))  # visible asymmetry prevents rotational shortcuts
    return cells


def _orient(cells: set[tuple[int, int]], turns: int, reflect: bool) -> set[tuple[int, int]]:
    pts = {(r, -c if reflect else c) for r, c in cells}
    for _ in range(turns % 4):
        pts = {(c, -r) for r, c in pts}
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return {(r - r0, c - c0) for r, c in pts}


def _outside_pivots(shape: set[tuple[int, int]]) -> list[tuple[int, int]]:
    holes = {p for region in _holes(shape) for p in region}
    result = set()
    for r, c in shape:
        for dr, dc in DIR4:
            p = (r + dr, c + dc)
            if p not in shape and p not in holes:
                result.add(p)
    return list(result)


def _dilate(points: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    return {
        (r + dr, c + dc)
        for r, c in points
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
    }


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    input_shapes: list[set[tuple[int, int]]] = []
    output_shapes: list[set[tuple[int, int]]] = []
    pivot_cells: set[tuple[int, int]] = set()
    specs = []

    # Every generated example exposes the entire angle vocabulary.  For the three
    # hole-bearing objects, both pivot modes are forced to occur in each example.
    internal_flags = [False, random.choice((False, True)), False, False]
    internal_flags[2] = not internal_flags[1]
    internal_flags[3] = random.choice((False, True))
    colors = [random.choice(SHAPE_COLORS) for _ in range(4)]  # repeats are intentional

    for n_holes in range(4):
        internal = internal_flags[n_holes] if n_holes else False
        local = _orient(_base_shape(n_holes), random.randrange(4), random.choice((False, True)))
        hole_cells = [p for region in _holes(local) for p in region]
        pivot_options = hole_cells if internal else _outside_pivots(local)
        random.shuffle(pivot_options)
        placed = False
        for local_pivot in pivot_options:
            for _ in range(350):
                shift = (random.randrange(h), random.randrange(w))
                shape = {(r + shift[0], c + shift[1]) for r, c in local}
                pivot = (local_pivot[0] + shift[0], local_pivot[1] + shift[1])
                turns = -n_holes if internal else n_holes
                moved = {_rotate_point(p, pivot, turns) for p in shape}
                if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in shape | moved | {pivot}):
                    continue
                # Input objects have a visible one-cell moat; output objects are
                # disjoint from all other final objects and every pivot.
                if any(shape & _dilate(old) for old in input_shapes):
                    continue
                if input_shapes and pivot in _dilate(set().union(*input_shapes), 1):
                    continue
                if any(old_pivot in _dilate(shape, 1) for old_pivot in pivot_cells):
                    continue
                if moved & (pivot_cells | {pivot}):
                    continue
                if any(moved & old for old in input_shapes):
                    continue
                if any(shape & old for old in output_shapes):
                    continue
                if any(moved & old for old in output_shapes):
                    continue
                input_shapes.append(shape)
                output_shapes.append(moved)
                pivot_cells.add(pivot)
                specs.append((shape, pivot, n_holes, internal, moved, colors[n_holes]))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    for shape, pivot, _, _, _, color in specs:
        for p in shape:
            grid[p] = color
        grid[pivot] = GREY

    # Semantic checks use construction metadata, not merely main(generate_input()).
    try:
        assignments = _pair_objects(_components((grid != 0) & (grid != GREY)), pivot_cells)
    except AssertionError:
        return None
    observed = sorted((n, internal) for _, _, n, internal in assignments)
    expected_modes = sorted((n, internal) for _, _, n, internal, _, _ in specs)
    if observed != expected_modes or {n for n, _ in observed} != {0, 1, 2, 3}:
        return None
    if not ({inside for n, inside in observed if n > 0} == {False, True}):
        return None

    expected = grid.copy()
    for shape, _, _, _, _, _ in specs:
        for p in shape:
            expected[p] = BACKGROUND
    for _, _, _, _, moved, color in specs:
        for p in moved:
            expected[p] = color
    if not np.array_equal(main(grid), expected) or np.array_equal(grid, expected):
        return None

    # Reject samples where the visible geometry would fail to disprove either
    # a fixed-one-turn angle shortcut or an ignored-direction shortcut.
    angle_witness = any(n in (2, 3) and moved != {_rotate_point(p, pivot, 1) for p in shape}
                        for shape, pivot, n, _, moved, _ in specs)
    direction_witness = any(n in (1, 3) and internal and
                            moved != {_rotate_point(p, pivot, n) for p in shape}
                            for shape, pivot, n, internal, moved, _ in specs)
    if not (angle_witness and direction_witness):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle- and semantics-checked valid puzzle input."""
    for _ in range(500):
        result = _attempt_generate()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hole-clock input")
