from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 8  # azure
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
TRIANGLE_COUNT_RANGE = (1, 4)
SIDE_RANGE = (3, 8)
MAX_GEN_ATTEMPTS = 400


def _bresenham(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p0
    r1, c1 = p1
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    pts: list[tuple[int, int]] = []
    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return pts


def _triangle_area2(vertices: list[tuple[int, int]]) -> int:
    (r1, c1), (r2, c2), (r3, c3) = vertices
    return abs((r2 - r1) * (c3 - c1) - (r3 - r1) * (c2 - c1))


def _point_in_triangle(
    p: tuple[int, int], a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]
) -> bool:
    pr, pc = p
    ax, ay = a
    bx, by = b
    cx, cy = c
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = pr - ax, pc - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    eps = 1e-9
    return u >= -eps and v >= -eps and u + v <= 1.0 + eps


def _paint_triangle(out: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    rows = [r for r, _ in vertices]
    cols = [c for _, c in vertices]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    a, b, c = vertices
    for r in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if _point_in_triangle((r, cc), a, b, c):
                out[r, cc] = color

    for p0, p1 in ((a, b), (b, c), (c, a)):
        for r, cc in _bresenham(p0, p1):
            out[r, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group three same-colored markers into triangles, then fill each triangle.

    Triangles are processed from larger area to smaller area so later triangles
    overwrite earlier ones where they overlap. Original marker pixels are
    restored at the end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(g != BG):
        positions[int(g[r, c])].append((int(r), int(c)))

    triangles: list[tuple[int, int, list[tuple[int, int]]]] = []
    for color, pts in positions.items():
        assert len(pts) == 3, "each foreground color must appear exactly three times"
        area2 = _triangle_area2(pts)
        assert area2 > 0, "triangle vertices must be non-collinear"
        triangles.append((area2, int(color), pts))

    out = g.copy()
    triangles.sort(key=lambda item: item[0], reverse=True)
    for _area2, color, pts in triangles:
        _paint_triangle(out, pts, color)

    # Preserve the original single-pixel markers even if they fall inside later triangles.
    out[g != BG] = g[g != BG]
    return out


def _make_right_triangle(
    h: int, w: int, side: int, dr: int, dc: int
) -> list[tuple[int, int]] | None:
    if dr > 0:
        r_corner = random.randint(1, h - side - 2)
    else:
        r_corner = random.randint(side + 1, h - 2)
    if dc > 0:
        c_corner = random.randint(1, w - side - 2)
    else:
        c_corner = random.randint(side + 1, w - 2)

    pts = [
        (r_corner, c_corner),
        (r_corner + dr * side, c_corner),
        (r_corner, c_corner + dc * side),
    ]
    if len(set(pts)) != 3:
        return None
    if not all(0 <= r < h and 0 <= c < w for r, c in pts):
        return None
    return pts


def _triangles_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    rows_a = [r for r, _ in a]
    cols_a = [c for _, c in a]
    rows_b = [r for r, _ in b]
    cols_b = [c for _, c in b]
    if max(rows_a) < min(rows_b) or max(rows_b) < min(rows_a):
        return False
    if max(cols_a) < min(cols_b) or max(cols_b) < min(cols_a):
        return False

    ra0, ra1 = min(rows_a), max(rows_a)
    ca0, ca1 = min(cols_a), max(cols_a)
    rb0, rb1 = min(rows_b), max(rows_b)
    cb0, cb1 = min(cols_b), max(cols_b)
    for r in range(max(ra0, rb0), min(ra1, rb1) + 1):
        for c in range(max(ca0, cb0), min(ca1, cb1) + 1):
            if _point_in_triangle((r, c), *a) and _point_in_triangle((r, c), *b):
                return True
    return False


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    triangle_count = random.randint(*TRIANGLE_COUNT_RANGE)
    colors = random.sample(PALETTE, triangle_count)
    target_overlap = triangle_count >= 2 and random.random() < 0.5

    triangles: list[tuple[int, int, list[tuple[int, int]]]] = []
    used_markers: set[tuple[int, int]] = set()
    used_areas: set[int] = set()
    found_overlap = False

    for idx, color in enumerate(colors):
        placed = False
        for _ in range(120):
            side = random.randint(*SIDE_RANGE)
            pts = _make_right_triangle(h, w, side, random.choice((-1, 1)), random.choice((-1, 1)))
            if pts is None:
                continue
            if any(pt in used_markers for pt in pts):
                continue
            area2 = _triangle_area2(pts)
            if area2 in used_areas:
                continue

            overlaps_existing = any(_triangles_overlap(pts, old_pts) for _a, _c, old_pts in triangles)
            if target_overlap and idx == triangle_count - 1 and not found_overlap and not overlaps_existing:
                continue
            if not target_overlap and overlaps_existing:
                continue

            triangles.append((area2, color, pts))
            used_markers.update(pts)
            used_areas.add(area2)
            found_overlap = found_overlap or overlaps_existing
            placed = True
            break
        if not placed:
            return None

    if target_overlap and not found_overlap:
        return None

    for _area2, color, pts in triangles:
        for r, c in pts:
            grid[r, c] = color

    expected = grid.copy()
    for area2, color, pts in sorted(triangles, key=lambda item: item[0], reverse=True):
        _paint_triangle(expected, pts, color)
    expected[grid != BG] = grid[grid != BG]

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _generate_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid triangle-fill input")
