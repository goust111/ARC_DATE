"""ARC task: counter-rotate two concentric rectangular rings."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following the shared ARC toolbox convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _perimeter(box: BBox) -> list[Coord]:
    """Clockwise perimeter coordinates, beginning at the upper-left corner."""
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1
    coords: list[Coord] = []
    coords.extend((r0, c) for c in range(c0, c1 + 1))
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _foreground_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """Find 4-connected foreground components as (row, column) arrays."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(grid != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        points: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            points.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                        and int(grid[rr, cc]) != bg):
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(points, dtype=int))
    return result


def _component_box(points: np.ndarray) -> BBox:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rings(grid: np.ndarray, bg: int) -> tuple[BBox, BBox]:
    """Locate the unique pair of complete rectangular perimeters two cells apart."""
    candidates: list[tuple[BBox, BBox]] = []
    components = _foreground_components(grid, bg)
    component_sets = [set(map(tuple, points.tolist())) for points in components]
    boxes = [_component_box(points) for points in components]

    for i, outer in enumerate(boxes):
        r0, c0, r1, c1 = outer
        if r1 - r0 < 6 or c1 - c0 < 6:
            continue
        inner = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        outer_cells = set(_perimeter(outer))
        if component_sets[i] != outer_cells:
            continue
        for j, box in enumerate(boxes):
            if i == j or box != inner:
                continue
            inner_cells = set(_perimeter(inner))
            if component_sets[j] != inner_cells:
                continue

            # The specified reference frame is the two ring bboxes: their one-cell
            # separating perimeter and the inner cavity must both be background.
            allowed = outer_cells | inner_cells
            valid = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r, c) not in allowed and int(grid[r, c]) != bg:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidates.append((outer, inner))

    assert len(candidates) == 1, "expected one unambiguous concentric double ring"
    return candidates[0]


def _rotate_values(out: np.ndarray, source: np.ndarray, coords: list[Coord], steps: int) -> None:
    values = np.asarray([source[r, c] for r, c in coords], dtype=source.dtype)
    shifted = np.roll(values, int(steps))
    for (r, c), value in zip(coords, shifted):
        out[r, c] = value


def main(grid: np.ndarray) -> np.ndarray:
    """Counter-rotate two concentric rings by the number of exterior dots.

    Input structure:
    - A dominant background surrounds two complete, multicolored concentric
      rectangular rings, with exactly one background perimeter between them.
    - A group of same-colored scattered dots lies strictly outside the outer
      ring's bounding box; all other cells inside that box are ring or background.

    Rule:
    - Count the exterior scatter pixels n in the coordinate frame outside the
      outer ring bounding box. Move every outer-ring color n perimeter cells
      clockwise, and every inner-ring color n perimeter cells counterclockwise.
    - Preserve the background and all scatter pixels exactly where they are.

    Key insight:
    - The exterior dots encode a rotation distance, while the one-cell gap marks
      two meshed concentric gears whose equally stepped rotations have opposite directions.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional grid"
    bg = _background(g)
    outer, inner = _find_rings(g, bg)
    r0, c0, r1, c1 = outer

    outside = np.ones(g.shape, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    scatter = np.argwhere(outside & (g != bg))
    assert len(scatter) > 0, "expected exterior scatter pixels"
    scatter_colors = {int(g[r, c]) for r, c in scatter}
    assert len(scatter_colors) == 1, "scatter pixels must have one common color"

    n = int(len(scatter))
    out = g.copy()
    _rotate_values(out, g, _perimeter(outer), n)   # positive = clockwise
    _rotate_values(out, g, _perimeter(inner), -n)  # negative = counterclockwise
    return out


def _has_dihedral_symmetry(values: np.ndarray) -> bool:
    """Reject every non-identity rotational or reflected symmetry."""
    size = len(values)
    if any(np.array_equal(values, np.roll(values, k)) for k in range(1, size)):
        return True
    reversed_values = values[::-1]
    return any(np.array_equal(values, np.roll(reversed_values, k)) for k in range(size))


def _random_ring_values(length: int, palette: list[int], required: int, n: int) -> np.ndarray:
    for _ in range(200):
        values = np.asarray(random.choices(palette, k=length), dtype=int)
        values[random.randrange(length)] = required
        if len(set(map(int, values))) < 3:
            continue
        if _has_dihedral_symmetry(values):
            continue
        if np.array_equal(np.roll(values, n), np.roll(values, -n)):
            continue
        return values
    raise RuntimeError("could not make an asymmetric ring sequence")


def _paint_values(grid: np.ndarray, coords: list[Coord], values: np.ndarray) -> None:
    for (r, c), value in zip(coords, values):
        grid[r, c] = int(value)


def _expected(grid: np.ndarray, outer: BBox, inner: BBox, n: int) -> np.ndarray:
    """Independent generator oracle using explicit source-to-destination movement."""
    result = grid.copy()
    for coords, delta in ((_perimeter(outer), n), (_perimeter(inner), -n)):
        length = len(coords)
        for source_index, (r, c) in enumerate(coords):
            dr, dc = coords[(source_index + delta) % length]
            result[dr, dc] = grid[r, c]
    return result


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    outer_h = random.randint(9, min(17, h - 6))
    outer_w = random.randint(9, min(17, w - 6))
    r0 = random.randint(3, h - outer_h - 3)
    c0 = random.randint(3, w - outer_w - 3)
    outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    inner = (r0 + 2, c0 + 2, r0 + outer_h - 3, c0 + outer_w - 3)

    bg = random.randrange(10)
    available = [color for color in range(10) if color != bg]
    palette = random.sample(available, random.randint(4, 6))
    scatter_color = random.choice(palette)
    max_steps = min(9, len(_perimeter(inner)) // 3)
    n = random.randint(2, max_steps)

    grid = np.full((h, w), bg, dtype=int)
    outer_values = _random_ring_values(len(_perimeter(outer)), palette, scatter_color, n)
    inner_values = _random_ring_values(len(_perimeter(inner)), palette, scatter_color, n)
    _paint_values(grid, _perimeter(outer), outer_values)
    _paint_values(grid, _perimeter(inner), inner_values)

    # Eligible dots are outside the outer bbox, at least one cell away from it.
    candidates: list[Coord] = []
    r1, c1 = outer[2], outer[3]
    for r in range(h):
        for c in range(w):
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    dots: list[Coord] = []
    for point in candidates:
        if all(max(abs(point[0] - rr), abs(point[1] - cc)) >= 2 for rr, cc in dots):
            dots.append(point)
            if len(dots) == n:
                break
    if len(dots) != n:
        return None
    for r, c in dots:
        grid[r, c] = scatter_color

    # Semantic validation: unique parsing, independent oracle, preservation,
    # opposite-direction necessity, count dependence, and non-degenerate change.
    try:
        found_outer, found_inner = _find_rings(grid, bg)
        solved = main(grid)
    except AssertionError:
        return None
    expected = _expected(grid, outer, inner, n)
    if (found_outer, found_inner) != (outer, inner) or not np.array_equal(solved, expected):
        return None
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    if not np.array_equal(solved[outside], grid[outside]):
        return None
    if np.array_equal(solved, grid):
        return None
    wrong_same_direction = grid.copy()
    _rotate_values(wrong_same_direction, grid, _perimeter(outer), n)
    _rotate_values(wrong_same_direction, grid, _perimeter(inner), n)
    if np.array_equal(solved, wrong_same_direction):
        return None
    if np.array_equal(expected, _expected(grid, outer, inner, n - 1)):
        return None
    if Counter(map(int, solved.ravel())) != Counter(map(int, grid.ravel())):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate valid double-ring instance."""
    for _ in range(100):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid double-ring input")

