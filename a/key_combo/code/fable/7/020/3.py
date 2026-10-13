"""ARC task: stamp a multicolor sprite over a two-generator translation lattice."""

from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
GRAY = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _read_input(grid: np.ndarray):
    """Identify the connected sprite, its internal anchor, and the two markers."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    assert np.count_nonzero(g == GRAY) == 3, "expected one anchor and two gray generators"

    comps = _components(g != BLACK)
    sprites = [comp for comp in comps if len(comp) > 1]
    singles = [comp for comp in comps if len(comp) == 1]
    assert len(sprites) == 1 and len(singles) == 2
    sprite = sprites[0]
    assert all(g[tuple(comp[0])] == GRAY for comp in singles)

    anchor_cells = [(int(r), int(c)) for r, c in sprite if g[r, c] == GRAY]
    assert len(anchor_cells) == 1
    anchor = anchor_cells[0]
    markers = sorted(tuple(map(int, comp[0])) for comp in singles)
    vectors = [(r - anchor[0], c - anchor[1]) for r, c in markers]
    det = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
    assert det != 0, "translation generators must be linearly independent"

    sprite_cells = [(int(r - anchor[0]), int(c - anchor[1]), int(g[r, c])) for r, c in sprite]
    return g, anchor, vectors, sprite_cells


def _coefficient_bounds(shape, anchor, vectors, sprite_cells):
    """Bound all lattice anchors whose sprite bounding box can meet the canvas."""
    h, w = shape
    ars = [r for r, _c, _color in sprite_cells]
    acs = [c for _r, c, _color in sprite_cells]
    # A translated anchor can matter only inside this expanded canvas rectangle.
    row_bounds = (-max(ars), h - 1 - min(ars))
    col_bounds = (-max(acs), w - 1 - min(acs))
    v1, v2 = vectors
    det = v1[0] * v2[1] - v1[1] * v2[0]
    aa, bb = [], []
    for pr in row_bounds:
        for pc in col_bounds:
            qr, qc = pr - anchor[0], pc - anchor[1]
            aa.append((qr * v2[1] - qc * v2[0]) / det)
            bb.append((v1[0] * qc - v1[1] * qr) / det)
    return (
        range(math.floor(min(aa)) - 1, math.ceil(max(aa)) + 2),
        range(math.floor(min(bb)) - 1, math.ceil(max(bb)) + 2),
    )


def _render(shape, anchor, vectors, sprite_cells):
    h, w = shape
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    a_values, b_values = _coefficient_bounds(shape, anchor, vectors, sprite_cells)
    for a in a_values:
        for b in b_values:
            ar = anchor[0] + a * vectors[0][0] + b * vectors[1][0]
            ac = anchor[1] + a * vectors[0][1] + b * vectors[1][1]
            visible = []
            for dr, dc, color in sprite_cells:
                r, c = ar + dr, ac + dc
                if 0 <= r < h and 0 <= c < w:
                    visible.append((r, c, color))
            if not visible:
                continue
            assert not any(occupied[r, c] for r, c, _ in visible), "lattice copies overlap"
            for r, c, color in visible:
                occupied[r, c] = True
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the anchored multicolor sprite across the canvas on its generated lattice.

    Input structure:
    - The background is black. One 4-connected multicolor sprite contains exactly
      one gray anchor pixel, and two further gray pixels are isolated from it.
    - The two anchor-to-isolated-point vectors are linearly independent, and
      translated sprite copies do not overlap.

    Rule:
    - Use the sprite anchor as the origin and the two vectors from it to the isolated
      gray pixels as translation generators. Paint a copy at every integer linear
      combination of those vectors, clipping each copy at the canvas boundary.
    - The gray anchor is copied with the sprite, so the two isolated markers become
      anchors of their corresponding copies.

    Key insight:
    - The isolated gray pixels are not destinations or noise: together they define
      the two-generator lattice that must be extended over the entire canvas.
    """
    g, anchor, vectors, sprite_cells = _read_input(grid)
    return _render(g.shape, anchor, vectors, sprite_cells)


def _random_sprite():
    """Make a connected, asymmetric sprite with the anchor away from its bbox edge."""
    side = random.choice((3, 3, 4))
    anchor = (1, 1) if side == 3 or random.random() < 0.5 else (2, 2)
    cells = {anchor}
    for dr, dc in NEIGHBORS4:
        cells.add((anchor[0] + dr, anchor[1] + dc))
    target = random.randint(6, min(10, side * side))
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                q = (r + dr, c + dc)
                if 0 <= q[0] < side and 0 <= q[1] < side and q not in cells:
                    frontier.append(q)
        if not frontier:
            break
        cells.add(random.choice(frontier))

    non_anchor = list(cells - {anchor})
    random.shuffle(non_anchor)
    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.choice((2, 2, 3)))
    colors = {anchor: GRAY}
    for i, cell in enumerate(non_anchor):
        colors[cell] = palette[i] if i < len(palette) else random.choice(palette)
    return anchor, colors


def _known_render(shape, anchor, vectors, sprite_cells):
    """Generator oracle using a deliberately simple bounded enumeration."""
    h, w = shape
    out = np.zeros(shape, dtype=int)
    used = set()
    records = []
    for a in range(-12, 13):
        for b in range(-12, 13):
            ar = anchor[0] + a * vectors[0][0] + b * vectors[1][0]
            ac = anchor[1] + a * vectors[0][1] + b * vectors[1][1]
            visible = []
            for dr, dc, color in sprite_cells:
                r, c = ar + dr, ac + dc
                if 0 <= r < h and 0 <= c < w:
                    visible.append((r, c, color))
            if not visible:
                continue
            coords = {(r, c) for r, c, _ in visible}
            if used & coords:
                return None
            used |= coords
            for r, c, color in visible:
                out[r, c] = color
            records.append((a, b, len(visible)))
    return out, records, used


def _generate_attempt():
    h, w = random.randint(20, 28), random.randint(20, 28)
    local_anchor, colored_cells = _random_sprite()
    anchor = (random.randint(7, h - 8), random.randint(7, w - 8))

    vertical = (random.choice((-1, 1)) * random.randint(5, 7), random.randint(-2, 2))
    horizontal = (random.randint(-2, 2), random.choice((-1, 1)) * random.randint(5, 7))
    vectors = [vertical, horizontal]
    if random.random() < 0.5:
        vectors.reverse()
    det = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
    if abs(det) < 20:
        return None

    markers = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in vectors]
    if not all(0 <= r < h and 0 <= c < w for r, c in markers):
        return None

    grid = np.zeros((h, w), dtype=int)
    sprite_cells = []
    sprite_global = set()
    for (lr, lc), color in colored_cells.items():
        r = anchor[0] + lr - local_anchor[0]
        c = anchor[1] + lc - local_anchor[1]
        grid[r, c] = color
        sprite_global.add((r, c))
        sprite_cells.append((r - anchor[0], c - anchor[1], color))
    for marker in markers:
        if marker in sprite_global:
            return None
        grid[marker] = GRAY

    # Both markers must be genuinely isolated 4-components in the visible input.
    if any(abs(r1 - r2) + abs(c1 - c2) <= 1 for r1, c1 in markers for r2, c2 in sprite_global):
        return None
    if abs(markers[0][0] - markers[1][0]) + abs(markers[0][1] - markers[1][1]) <= 1:
        return None

    oracle = _known_render(grid.shape, anchor, vectors, sprite_cells)
    if oracle is None:
        return None
    expected, records, used = oracle
    full_size = len(sprite_cells)
    if len(records) < 8 or not any(n < full_size for _a, _b, n in records):
        return None
    if not any(a != 0 and b != 0 for a, b, _n in records):
        return None
    # A union of the two one-dimensional progressions is a strict shortcut and
    # must miss some of the required two-dimensional lattice.
    axial = {(a, b) for a, b, _n in records if a == 0 or b == 0}
    if len(axial) == len(records):
        return None
    rows = [r for r, _c in used]
    cols = [c for _r, c in used]
    if min(rows) > 2 or max(rows) < h - 3 or min(cols) > 2 or max(cols) < w - 3:
        return None

    try:
        parsed = _read_input(grid)
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if parsed[1] != anchor or set(parsed[2]) != set(vectors):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return None
    if not all(actual[marker] == GRAY for marker in markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance of the lattice task."""
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-stamping input")
