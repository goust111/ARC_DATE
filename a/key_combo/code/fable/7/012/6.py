"""ARC task 6: conserve material while extruding it through an aperture."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        queue = deque([(start_r, start_c)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _container_from_component(g: np.ndarray, cells: np.ndarray) -> dict:
    """Decode one rectangular wall and its sole aperture in grid coordinates."""
    r0, c0 = cells.min(axis=0).tolist()
    r1, c1 = cells.max(axis=0).tolist()
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "container must have a visible cavity"

    component_cells = {tuple(map(int, p)) for p in cells}
    perimeter = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )
    assert component_cells <= perimeter, "gray component must be a one-cell rectangular wall"
    missing = perimeter - component_cells
    assert missing, "container must have an aperture"

    sides = []
    if all(r == r0 and c0 < c < c1 for r, c in missing):
        sides.append(0)
    if all(c == c1 and r0 < r < r1 for r, c in missing):
        sides.append(1)
    if all(r == r1 and c0 < c < c1 for r, c in missing):
        sides.append(2)
    if all(c == c0 and r0 < r < r1 for r, c in missing):
        sides.append(3)
    assert len(sides) == 1, "aperture must lie on exactly one wall side"
    side = sides[0]
    aperture = sorted(missing)
    if side in (0, 2):
        values = [c for _r, c in aperture]
    else:
        values = [r for r, _c in aperture]
    assert values == list(range(values[0], values[-1] + 1)), "aperture must be contiguous"
    assert component_cells == perimeter - missing, "wall perimeter must otherwise be complete"

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    colors = set(map(int, np.unique(interior))) - {BACKGROUND}
    assert len(colors) == 1, "each cavity must contain exactly one material color"
    color = colors.pop()
    assert color != WALL
    count = int(np.count_nonzero(interior == color))
    aperture_width = len(aperture)
    assert count > 0 and count % aperture_width == 0
    return {
        "bbox": (r0, c0, r1, c1),
        "aperture": aperture,
        "side": side,
        "color": color,
        "count": count,
        "width": aperture_width,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extrude each container's conserved contents into an aperture-aligned solid rectangle.

    Input structure:
    - A black background holds several separated one-cell gray rectangular container walls.
    - Each wall has exactly one contiguous, non-corner aperture, and aperture widths are distinct.
    - Each cavity contains scattered pixels of one material color; containers use different material colors.

    Rule:
    - Clear every material pixel inside its own container cavity.
    - Starting immediately outside its aperture, paint a solid rectangle aligned with the aperture.
    - Its width is the aperture width and its outward length is material count divided by that width.
      Preserve every wall pixel and every aperture cell.

    Key insight:
    - The aperture fixes one rectangle dimension, while conservation of the counted cavity pixels fixes the other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(np.count_nonzero(g == BACKGROUND)) > g.size // 2, "black must be dominant background"
    wall_components = _components4(g == WALL)
    assert len(wall_components) >= 2, "expected several containers"
    containers = [_container_from_component(g, cells) for cells in wall_components]
    assert len({x["width"] for x in containers}) == len(containers), "aperture widths must differ"
    assert len({x["color"] for x in containers}) == len(containers), "material colors must differ"

    out = g.copy()
    for item in containers:
        r0, c0, r1, c1 = item["bbox"]
        color = item["color"]
        cavity = out[r0 + 1 : r1, c0 + 1 : c1]
        cavity[cavity == color] = BACKGROUND

    height, width = g.shape
    claimed: set[tuple[int, int]] = set()
    for item in containers:
        dr, dc = DIRS[item["side"]]
        length = item["count"] // item["width"]
        target = []
        for aperture_r, aperture_c in item["aperture"]:
            for distance in range(1, length + 1):
                r = aperture_r + dr * distance
                c = aperture_c + dc * distance
                assert 0 <= r < height and 0 <= c < width, "extrusion must fit on canvas"
                target.append((r, c))
        assert len(set(target)) == item["count"]
        assert not (set(target) & claimed), "extrusions must not collide"
        assert all(int(out[r, c]) == BACKGROUND for r, c in target), "extrusion path must be empty"
        for r, c in target:
            out[r, c] = item["color"]
        claimed.update(target)
    return out


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    height, width = shape
    return {
        (r + dr, c + dc)
        for r, c in cells
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 0 <= r + dr < height and 0 <= c + dc < width
    }


def _try_generate() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)
    shape = (height, width)
    n = random.choice((2, 3))
    aperture_widths = random.sample((1, 2, 3, 4), n)
    colors = random.sample(PALETTE, n)
    sides = random.sample(range(4), n)  # exposes distinct outward directions in every grid
    grid = np.zeros(shape, dtype=int)
    occupied: set[tuple[int, int]] = set()
    specs = []

    for aperture_width, color, side in zip(aperture_widths, colors, sides):
        placed = False
        for _ in range(250):
            outer_h = random.randint(max(5, aperture_width + 2 if side in (1, 3) else 5), 9)
            outer_w = random.randint(max(5, aperture_width + 2 if side in (0, 2) else 5), 9)
            cavity_area = (outer_h - 2) * (outer_w - 2)
            possible_lengths = [x for x in range(2, 7) if aperture_width * x <= cavity_area - 1]
            if not possible_lengths:
                continue
            length = random.choice(possible_lengths)
            # Leave a background cell beyond the extrusion: length cannot be inferred from the canvas edge.
            top_margin = length + 2 if side == 0 else 2
            bottom_margin = length + 2 if side == 2 else 2
            left_margin = length + 2 if side == 3 else 2
            right_margin = length + 2 if side == 1 else 2
            r_lo, r_hi = top_margin, height - outer_h - bottom_margin
            c_lo, c_hi = left_margin, width - outer_w - right_margin
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0, c0 = random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
            r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1

            if side in (0, 2):
                start = random.randint(c0 + 1, c1 - aperture_width)
                aperture = [(r0 if side == 0 else r1, c) for c in range(start, start + aperture_width)]
            else:
                start = random.randint(r0 + 1, r1 - aperture_width)
                aperture = [(r, c1 if side == 1 else c0) for r in range(start, start + aperture_width)]
            perimeter = (
                {(r0, c) for c in range(c0, c1 + 1)}
                | {(r1, c) for c in range(c0, c1 + 1)}
                | {(r, c0) for r in range(r0, r1 + 1)}
                | {(r, c1) for r in range(r0, r1 + 1)}
            )
            walls = perimeter - set(aperture)
            cavity_cells = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
            dr, dc = DIRS[side]
            extrusion = {
                (r + dr * distance, c + dc * distance)
                for r, c in aperture
                for distance in range(1, length + 1)
            }
            footprint = walls | set(aperture) | cavity_cells | extrusion
            if _expanded(footprint, shape, 1) & occupied:
                continue

            material_count = aperture_width * length
            material = set(random.sample(sorted(cavity_cells), material_count))
            # Scattered input cannot already be mistaken for its packed output rectangle.
            rs = {r for r, _c in material}
            cs = {c for _r, c in material}
            if len(rs) * len(cs) == material_count:
                continue
            for r, c in walls:
                grid[r, c] = WALL
            for r, c in material:
                grid[r, c] = color
            occupied.update(_expanded(footprint, shape, 1))
            specs.append((walls, set(aperture), cavity_cells, extrusion, color, material_count, aperture_width, length))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for _walls, _aperture, cavity, extrusion, color, _count, _w, _length in specs:
        for r, c in cavity:
            if expected[r, c] == color:
                expected[r, c] = BACKGROUND
        for r, c in extrusion:
            expected[r, c] = color

    # Independent semantic checks, beyond solver self-consistency.
    if len(specs) != n or len({s[6] for s in specs}) != n:
        return None
    if len({s[7] for s in specs}) < 2:  # content count must contribute visible information
        return None
    for walls, aperture, cavity, extrusion, color, count, aperture_width, length in specs:
        if not (np.all(expected[tuple(zip(*walls))] == WALL) and np.all(expected[tuple(zip(*aperture))] == BACKGROUND)):
            return None
        if np.any(expected[tuple(zip(*cavity))] == color):
            return None
        if len(extrusion) != count or count != aperture_width * length:
            return None
        if not all(expected[r, c] == color for r, c in extrusion):
            return None
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    return grid if np.array_equal(actual, expected) and not np.array_equal(grid, expected) else None


def generate_input() -> np.ndarray:
    """Generate varied, non-overlapping multi-container examples."""
    for _ in range(500):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid aperture-conservation puzzle")
