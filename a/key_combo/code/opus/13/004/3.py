from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7)
GENERATION_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_values(values: list[int], shift: int) -> list[int]:
    n = len(values)
    shift %= n
    if shift == 0:
        return list(values)
    return list(values[-shift:] + values[:-shift])


def _ring_positions(center: tuple[int, int], radius: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cr, cc = center
    k = int(radius)
    corners = [
        (cr - k, cc - k),
        (cr - k, cc + k),
        (cr + k, cc + k),
        (cr + k, cc - k),
    ]
    sides = [
        (cr - k, cc),
        (cr, cc - k),
        (cr + k, cc),
        (cr, cc + k),
    ]
    return corners, sides


def _extract_structure(grid: np.ndarray) -> tuple[int, tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    fg = np.argwhere(g != bg)
    assert fg.size > 0, "expected foreground rings"

    r0, c0, r1, c1 = _bbox(fg)
    assert r1 - r0 == c1 - c0, "outer structure must be square"
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "ring center must be a cell"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    assert g[center] == bg, "shared center pixel must stay background"

    rings: dict[int, set[tuple[int, int]]] = {}
    for r, c in fg:
        rr, cc = int(r), int(c)
        dr = abs(rr - center[0])
        dc = abs(cc - center[1])
        radius = max(dr, dc)
        assert radius > 0, "center pixel cannot be foreground"
        assert dr in (0, radius) and dc in (0, radius), "foreground must be corner or side midpoint"
        rings.setdefault(radius, set()).add((rr, cc))

    assert 2 <= len(rings) <= 4, "expected 2-4 nested rings"
    for radius, cells in rings.items():
        corners, sides = _ring_positions(center, radius)
        assert cells == set(corners + sides), "each ring must contain exactly 8 sparse square-ring cells"

    return bg, center, sorted(rings, reverse=True)


def _apply_cycle(out: np.ndarray, src: np.ndarray, positions: list[tuple[int, int]], shift: int) -> None:
    rotated = _rotate_values([int(src[r, c]) for r, c in positions], shift)
    for (r, c), value in zip(positions, rotated):
        out[r, c] = value


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each sparse concentric square ring by a depth-dependent amount.

    Input structure:
    - Black is the dominant background color, and one central background cell is shared by all rings.
    - The foreground consists of 2-4 concentric sparse square rings.
    - Each ring keeps only 8 cells: its 4 corners and the midpoints of its 4 sides.

    Rule:
    - Order the rings from outside to inside and assign depths 1, 2, 3, ...
    - For the ring at depth D, cycle the 4 corner colors D steps clockwise in the order
      top-left, top-right, bottom-right, bottom-left.
    - For that same ring, cycle the 4 side-midpoint colors D steps counterclockwise in the
      order top, left, bottom, right.
    - Leave the center and every background cell unchanged.

    Key insight:
    - Containment is not a yes/no trigger here: each ring's nesting depth becomes the exact rotation step count.
    """
    g = np.asarray(grid, dtype=int)
    _bg, center, radii = _extract_structure(g)
    out = g.copy()

    for depth, radius in enumerate(radii, start=1):
        corners, sides = _ring_positions(center, radius)
        _apply_cycle(out, g, corners, depth)
        _apply_cycle(out, g, sides, depth)

    return out


def _candidate_radii(count: int, min_outer: int = 3, max_outer: int = 6) -> list[list[int]]:
    candidates: list[list[int]] = []

    def dfs(path: list[int]) -> None:
        if len(path) == count:
            if min_outer <= path[-1] <= max_outer:
                candidates.append(path.copy())
            return
        for step in (1, 2):
            nxt = path[-1] + step
            if nxt > max_outer:
                continue
            path.append(nxt)
            dfs(path)
            path.pop()

    dfs([1])
    return candidates


def _non_invariant_cycle(values: list[int], shift: int) -> bool:
    return _rotate_values(values, shift) != values


def _make_ring_colors(depth: int) -> tuple[list[int], list[int]]:
    shift = depth % 4
    for _ in range(200):
        unique_count = random.randint(3, 6)
        chosen = random.sample(PALETTE, unique_count)
        corners = [random.choice(chosen) for _ in range(4)]
        sides = [random.choice(chosen) for _ in range(4)]
        if len(set(corners)) < 2 or len(set(sides)) < 2:
            continue
        if shift and not (_non_invariant_cycle(corners, shift) and _non_invariant_cycle(sides, shift)):
            continue
        return corners, sides
    raise RuntimeError("failed to sample a visible ring coloring")


def _build_grid(radii_desc: list[int]) -> np.ndarray:
    outer = radii_desc[0]
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)

    rows = 2 * outer + 1 + top_pad + bottom_pad
    cols = 2 * outer + 1 + left_pad + right_pad
    grid = np.full((rows, cols), BG, dtype=int)
    center = (top_pad + outer, left_pad + outer)

    for depth, radius in enumerate(radii_desc, start=1):
        corners, sides = _ring_positions(center, radius)
        corner_values, side_values = _make_ring_colors(depth)
        for (r, c), value in zip(corners, corner_values):
            grid[r, c] = value
        for (r, c), value in zip(sides, side_values):
            grid[r, c] = value

    return grid


def _oracle_accepts(grid: np.ndarray) -> bool:
    try:
        _extract_structure(grid)
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        ring_count = random.randint(2, 4)
        radii = random.choice(_candidate_radii(ring_count))
        grid = _build_grid(sorted(radii, reverse=True))
        if _oracle_accepts(grid):
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
