from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
GRAY = 5
RING_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
LAYOUT_ATTEMPTS = 200
PATTERN_ATTEMPTS = 200

RING_DRAW_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CLOCKWISE_CORNERS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
COUNTERCLOCKWISE_SIDES = ((-1, 0), (0, -1), (1, 0), (0, 1))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _ring_values(grid: np.ndarray, center: tuple[int, int]) -> tuple[int, ...]:
    r, c = center
    return tuple(int(grid[r + dr, c + dc]) for dr, dc in RING_DRAW_OFFSETS)


def _place_ring_values(grid: np.ndarray, center: tuple[int, int], values: tuple[int, ...]) -> None:
    r, c = center
    for value, (dr, dc) in zip(values, RING_DRAW_OFFSETS):
        grid[r + dr, c + dc] = int(value)
    grid[r, c] = BLACK


def _rotate_ring_values(values: tuple[int, ...], steps: int) -> tuple[int, ...]:
    steps %= 4
    patch = np.array(
        [
            [values[0], values[1], values[2]],
            [values[3], BLACK, values[4]],
            [values[5], values[6], values[7]],
        ],
        dtype=int,
    )
    out = patch.copy()

    corner_vals = [int(patch[r + 1, c + 1]) for r, c in CLOCKWISE_CORNERS]
    side_vals = [int(patch[r + 1, c + 1]) for r, c in COUNTERCLOCKWISE_SIDES]

    for i, value in enumerate(corner_vals):
        dr, dc = CLOCKWISE_CORNERS[(i + steps) % 4]
        out[dr + 1, dc + 1] = value
    for i, value in enumerate(side_vals):
        dr, dc = COUNTERCLOCKWISE_SIDES[(i + steps) % 4]
        out[dr + 1, dc + 1] = value

    return (
        int(out[0, 0]),
        int(out[0, 1]),
        int(out[0, 2]),
        int(out[1, 0]),
        int(out[1, 2]),
        int(out[2, 0]),
        int(out[2, 1]),
        int(out[2, 2]),
    )


def _corners_only(values: tuple[int, ...], steps: int) -> tuple[int, ...]:
    steps %= 4
    patch = np.array(
        [
            [values[0], values[1], values[2]],
            [values[3], BLACK, values[4]],
            [values[5], values[6], values[7]],
        ],
        dtype=int,
    )
    out = patch.copy()
    corner_vals = [int(patch[r + 1, c + 1]) for r, c in CLOCKWISE_CORNERS]
    for i, value in enumerate(corner_vals):
        dr, dc = CLOCKWISE_CORNERS[(i + steps) % 4]
        out[dr + 1, dc + 1] = value
    return (
        int(out[0, 0]),
        int(out[0, 1]),
        int(out[0, 2]),
        int(out[1, 0]),
        int(out[1, 2]),
        int(out[2, 0]),
        int(out[2, 1]),
        int(out[2, 2]),
    )


def _sides_only(values: tuple[int, ...], steps: int) -> tuple[int, ...]:
    steps %= 4
    patch = np.array(
        [
            [values[0], values[1], values[2]],
            [values[3], BLACK, values[4]],
            [values[5], values[6], values[7]],
        ],
        dtype=int,
    )
    out = patch.copy()
    side_vals = [int(patch[r + 1, c + 1]) for r, c in COUNTERCLOCKWISE_SIDES]
    for i, value in enumerate(side_vals):
        dr, dc = COUNTERCLOCKWISE_SIDES[(i + steps) % 4]
        out[dr + 1, dc + 1] = value
    return (
        int(out[0, 0]),
        int(out[0, 1]),
        int(out[0, 2]),
        int(out[1, 0]),
        int(out[1, 2]),
        int(out[2, 0]),
        int(out[2, 1]),
        int(out[2, 2]),
    )


def _find_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict] = []
    occupied_centers: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            cells = [(r + dr, c + dc) for dr, dc in RING_DRAW_OFFSETS]
            vals = [int(g[rr, cc]) for rr, cc in cells]
            if any(v in (BLACK, GRAY) for v in vals):
                continue
            center = (r, c)
            assert center not in occupied_centers
            occupied_centers.add(center)
            rings.append(
                {
                    "center": center,
                    "cells": tuple(cells),
                    "values": tuple(vals),
                }
            )
    return rings


def _decode_bridge(comp: np.ndarray, ring_cell_to_ring: dict[tuple[int, int], int]) -> tuple[int, int]:
    coords = {tuple(map(int, pt)) for pt in comp}
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "gray bridge must be straight"

    if len(rows) == 1:
        r = next(iter(rows))
        c0 = min(c for _, c in coords)
        c1 = max(c for _, c in coords)
        assert coords == {(r, c) for c in range(c0, c1 + 1)}, "horizontal bridge must be contiguous"
        left = ring_cell_to_ring.get((r, c0 - 1))
        right = ring_cell_to_ring.get((r, c1 + 1))
        assert left is not None and right is not None and left != right, "bridge endpoints must touch two rings"
        for c in range(c0, c1 + 1):
            assert (r - 1, c) not in ring_cell_to_ring and (r + 1, c) not in ring_cell_to_ring
            if c != c0:
                assert (r, c - 1) not in ring_cell_to_ring
            if c != c1:
                assert (r, c + 1) not in ring_cell_to_ring
        return int(left), int(right)

    c = next(iter(cols))
    r0 = min(r for r, _ in coords)
    r1 = max(r for r, _ in coords)
    assert coords == {(r, c) for r in range(r0, r1 + 1)}, "vertical bridge must be contiguous"
    top = ring_cell_to_ring.get((r0 - 1, c))
    bottom = ring_cell_to_ring.get((r1 + 1, c))
    assert top is not None and bottom is not None and top != bottom, "bridge endpoints must touch two rings"
    for r in range(r0, r1 + 1):
        assert (r, c - 1) not in ring_cell_to_ring and (r, c + 1) not in ring_cell_to_ring
        if r != r0:
            assert (r - 1, c) not in ring_cell_to_ring
        if r != r1:
            assert (r + 1, c) not in ring_cell_to_ring
    return int(top), int(bottom)


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    rings = _find_rings(g)
    assert 3 <= len(rings) <= 5, "expected 3-5 rings"

    ring_cell_to_ring: dict[tuple[int, int], int] = {}
    for idx, ring in enumerate(rings):
        for cell in ring["cells"]:
            assert cell not in ring_cell_to_ring, "rings must not overlap"
            ring_cell_to_ring[cell] = idx

    degrees = [0] * len(rings)
    bridges: list[tuple[int, int]] = []
    for comp in _components_of_color(g, GRAY):
        a, b = _decode_bridge(comp, ring_cell_to_ring)
        bridges.append((a, b))
        degrees[a] += 1
        degrees[b] += 1

    for deg in degrees:
        assert 0 <= deg <= 3, "ring degree out of allowed range"
    return rings, degrees, bridges


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored 3x3 ring by a number of steps encoded by how many gray bridges touch it.

    Input structure:
    - The grid is a black background containing 3-5 complete 3x3 rings, each with a black center and eight non-black, non-gray perimeter cells.
    - Some pairs of rings are connected by straight gray bridge bars that run horizontally or vertically through background cells.
    - Bridges never branch: each gray component is one 4-connected line segment whose two ends touch two different rings.

    Rule:
    - Detect every 3x3 ring from its black center and its eight surrounding colored cells.
    - Count how many gray bridge bars are incident to each ring.
    - For a ring with count N, move its four corner colors N steps clockwise and move its four side-center colors N steps counterclockwise.
    - Keep every gray bridge cell, every black background cell, and every black ring center unchanged.

    Key insight:
    - The gray bridges are not drawn in the output; they act as visible counters whose per-ring degree directly sets the two opposite color cycles on that ring.
    """
    g = np.asarray(grid, dtype=int)
    rings, degrees, _bridges = _analyze_scene(g)
    out = g.copy()
    for ring, steps in zip(rings, degrees):
        _place_ring_values(out, ring["center"], _rotate_ring_values(ring["values"], int(steps)))
    return out


def _spanning_trees(slot_points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    n = len(slot_points)
    idx_edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(slot_points[i][0] - slot_points[j][0]) + abs(slot_points[i][1] - slot_points[j][1]) == 1
    ]
    trees: list[list[tuple[int, int]]] = []
    for chosen in itertools.combinations(idx_edges, n - 1):
        deg = [0] * n
        graph = [[] for _ in range(n)]
        for a, b in chosen:
            deg[a] += 1
            deg[b] += 1
            graph[a].append(b)
            graph[b].append(a)
        if max(deg) > 3 or max(deg) < 2:
            continue
        seen = {0}
        q = deque([0])
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(seen) != n:
            continue
        trees.append([(slot_points[a], slot_points[b]) for a, b in chosen])
    return trees


def _choose_layout() -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    slot_universe = [(r, c) for r in range(4) for c in range(4)]
    for _ in range(LAYOUT_ATTEMPTS):
        ring_count = random.choice((4, 5, 5))
        chosen = random.sample(slot_universe, ring_count)
        isolate = random.randrange(ring_count)
        active = [pt for i, pt in enumerate(chosen) if i != isolate]
        trees = _spanning_trees(active)
        if not trees:
            continue

        maxdeg3 = []
        other = []
        for tree in trees:
            deg: dict[tuple[int, int], int] = {pt: 0 for pt in active}
            for a, b in tree:
                deg[a] += 1
                deg[b] += 1
            if max(deg.values()) == 3:
                maxdeg3.append(tree)
            else:
                other.append(tree)

        if maxdeg3 and random.random() < 0.6:
            tree = random.choice(maxdeg3)
        else:
            tree = random.choice(trees)

        min_r = min(r for r, _ in chosen)
        min_c = min(c for _, c in chosen)
        norm = [(r - min_r, c - min_c) for r, c in chosen]
        remap = {old: new for old, new in zip(chosen, norm)}
        norm_tree = [(remap[a], remap[b]) for a, b in tree]
        return norm, norm_tree
    raise RuntimeError("failed to choose a valid slot layout")


def _bridge_degrees(slot_centers: list[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[int]:
    idx = {pt: i for i, pt in enumerate(slot_centers)}
    deg = [0] * len(slot_centers)
    for a, b in edges:
        deg[idx[a]] += 1
        deg[idx[b]] += 1
    return deg


def _sample_ring_pattern() -> tuple[int, ...]:
    for _ in range(PATTERN_ATTEMPTS):
        unique_count = random.choices((2, 3, 4), weights=(1, 2, 4), k=1)[0]
        palette = random.sample(RING_COLORS, unique_count)
        vals = tuple(random.choice(palette) for _ in range(8))
        if len(set(vals)) != unique_count:
            continue
        orbit = {_rotate_ring_values(vals, s) for s in range(4)}
        if len(orbit) != 4:
            continue
        corner_orbit = {_corners_only(vals, s) for s in range(4)}
        side_orbit = {_sides_only(vals, s) for s in range(4)}
        if len(corner_orbit) != 4 or len(side_orbit) != 4:
            continue
        return vals
    raise RuntimeError("failed to sample a non-degenerate ring pattern")


def _draw_bridge(
    grid: np.ndarray,
    center_a: tuple[int, int],
    center_b: tuple[int, int],
) -> None:
    ra, ca = center_a
    rb, cb = center_b
    if ra == rb:
        c0, c1 = sorted((ca, cb))
        for c in range(c0 + 2, c1 - 1):
            grid[ra, c] = GRAY
        return
    assert ca == cb
    r0, r1 = sorted((ra, rb))
    for r in range(r0 + 2, r1 - 1):
        grid[r, ca] = GRAY


def _semantic_accepts(
    grid: np.ndarray,
    intended_degrees: list[int],
    duplicate_pair: tuple[int, int],
) -> bool:
    try:
        rings, degrees, bridges = _analyze_scene(grid)
    except AssertionError:
        return False

    if sorted(degrees) != sorted(intended_degrees):
        return False
    if not any(d == 0 for d in degrees):
        return False
    if not any(d >= 2 for d in degrees):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    ring_by_center = {ring["center"]: ring for ring in rings}
    ordered_centers = sorted(ring_by_center)
    ordered_degrees = [degrees[[ring["center"] for ring in rings].index(center)] for center in ordered_centers]
    ordered_values = [_ring_values(grid, center) for center in ordered_centers]

    dup_a, dup_b = duplicate_pair
    if ordered_values[dup_a] != ordered_values[dup_b]:
        return False
    if ordered_degrees[dup_a] == ordered_degrees[dup_b]:
        return False

    same_count_different_pattern = False
    for i in range(len(ordered_centers)):
        for j in range(i + 1, len(ordered_centers)):
            if ordered_degrees[i] == ordered_degrees[j] and ordered_values[i] != ordered_values[j]:
                same_count_different_pattern = True
    if not same_count_different_pattern:
        return False

    for center, true_steps, values in zip(ordered_centers, ordered_degrees, ordered_values):
        output_values = _ring_values(out, center)
        if output_values != _rotate_ring_values(values, true_steps):
            return False
        if true_steps > 0 and output_values == values:
            return False
        if true_steps > 0 and (_corners_only(values, true_steps) == output_values or _sides_only(values, true_steps) == output_values):
            return False
        for alt_steps in range(4):
            if alt_steps == true_steps:
                continue
            if _rotate_ring_values(values, alt_steps) == output_values:
                return False

    bridge_mask = grid == GRAY
    if not np.array_equal(out[bridge_mask], grid[bridge_mask]):
        return False
    if len(bridges) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    slots, edges = _choose_layout()
    degrees = _bridge_degrees(slots, edges)
    if not any(d == 0 for d in degrees) or not any(d >= 2 for d in degrees):
        return None

    step = random.choice((4, 5, 6))
    top_pad = random.randint(1, 2)
    left_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    right_pad = random.randint(1, 2)

    max_slot_r = max(r for r, _ in slots)
    max_slot_c = max(c for _, c in slots)
    h = top_pad + max_slot_r * step + 3 + bottom_pad
    w = left_pad + max_slot_c * step + 3 + right_pad
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    centers: list[tuple[int, int]] = []
    for sr, sc in slots:
        r0 = top_pad + sr * step
        c0 = left_pad + sc * step
        centers.append((r0 + 1, c0 + 1))

    low = min(range(len(degrees)), key=lambda i: (degrees[i], i))
    high = max(range(len(degrees)), key=lambda i: (degrees[i], -i))
    if degrees[low] == degrees[high]:
        return None
    duplicate_pair = (low, high)

    patterns: list[tuple[int, ...] | None] = [None] * len(centers)
    base = _sample_ring_pattern()
    patterns[low] = base
    patterns[high] = base

    used = {base}
    for i in range(len(centers)):
        if patterns[i] is not None:
            continue
        for _ in range(PATTERN_ATTEMPTS):
            vals = _sample_ring_pattern()
            if vals not in used:
                patterns[i] = vals
                used.add(vals)
                break
        if patterns[i] is None:
            return None

    for center, vals in zip(centers, patterns):
        _place_ring_values(grid, center, vals)

    center_by_slot = {slot: center for slot, center in zip(slots, centers)}
    for a, b in edges:
        _draw_bridge(grid, center_by_slot[a], center_by_slot[b])

    try:
        rings, parsed_degrees, _bridges = _analyze_scene(grid)
    except AssertionError:
        return None

    center_to_degree = {ring["center"]: deg for ring, deg in zip(rings, parsed_degrees)}
    ordered_centers = sorted(center_to_degree)
    ordered_degrees = [center_to_degree[c] for c in ordered_centers]
    center_index = {center: i for i, center in enumerate(ordered_centers)}
    duplicate_pair_sorted = tuple(center_index[centers[i]] for i in duplicate_pair)

    if not _semantic_accepts(grid, ordered_degrees, duplicate_pair_sorted):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-count ring sample")
