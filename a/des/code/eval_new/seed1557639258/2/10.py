from __future__ import annotations

import math
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

COLOR_POOL = list(range(10))
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
POLYGON_COUNT_RANGE = (1, 3)
VERTEX_COUNT_CHOICES = (3, 4, 5, 6)
PLACE_MARGIN = 1
POLYGON_GAP = 2
LOCAL_GEN_RETRIES = 160
GLOBAL_GEN_RETRIES = 320


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bresenham_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(int(r1 - r0))
    dc = abs(int(c1 - c0))
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    pts: list[tuple[int, int]] = []
    while True:
        pts.append((int(r0), int(c0)))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return pts


def _cyclic_vertices(markers: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    pts = np.array([(r, c) for r, c, _ in markers], dtype=float)
    center = pts.mean(axis=0)
    ordered = sorted(
        markers,
        key=lambda item: math.atan2(float(item[0]) - float(center[0]), float(item[1]) - float(center[1])),
    )
    anchor_idx = min(range(len(ordered)), key=lambda i: (ordered[i][0], ordered[i][1]))
    return ordered[anchor_idx:] + ordered[:anchor_idx]


def _region_marker_colors(local: np.ndarray, region: list[tuple[int, int]], bg: int, border: int) -> list[int]:
    dists: dict[int, int] = {}
    h, w = local.shape
    for r, c in region:
        for dr, dc in DIR8:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            val = int(local[nr, nc])
            if val == bg or val == border:
                continue
            dist = max(abs(int(dr)), abs(int(dc)))
            prev = dists.get(val)
            if prev is None or dist < prev:
                dists[val] = dist
    touching = [color for color, dist in dists.items() if dist == 1]
    if len(touching) >= 3:
        return sorted(touching)
    return [color for color, _ in sorted(dists.items(), key=lambda kv: (kv[1], kv[0]))[:3]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Triangulate each marked convex polygon from its top-left vertex and fill the resulting triangles.

    Input structure:
    - bg = background color. The grid contains one or more closed convex polygon outlines.
    - Each polygon uses one border color on all edge pixels, while every vertex pixel is replaced by a distinct marker color.
    - Polygon interiors are background and polygons are separated from one another.

    Rule:
    - For each polygon, recover the cyclic vertex order from the colored marker pixels.
    - Choose the top-left vertex as the fan anchor and draw diagonals from it to every non-adjacent vertex, using the polygon's border color.
    - Each enclosed triangular region is filled with the smallest marker color among the three vertices on that triangle.
    - The original outer outline and all added diagonals end up in the border color, so the vertex markers are converted back to border pixels.

    Key insight:
    - The marker pixels identify the polygon vertices directly; once their cyclic order is known, the entire output is a deterministic fan triangulation from the top-left anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for comp in _components(g != bg, connectivity=8):
        vals, cnts = np.unique([int(g[r, c]) for r, c in comp], return_counts=True)
        border = int(vals[int(cnts.argmax())])
        markers = [(int(r), int(c), int(g[r, c])) for r, c in comp if int(g[r, c]) != border]
        if len(markers) < 3:
            continue

        ordered = _cyclic_vertices(markers)
        anchor_r, anchor_c, _anchor_color = ordered[0]

        r0, c0, r1, c1 = _bbox(comp)
        cr0 = max(0, int(r0 - 1))
        cc0 = max(0, int(c0 - 1))
        cr1 = min(g.shape[0] - 1, int(r1 + 1))
        cc1 = min(g.shape[1] - 1, int(c1 + 1))
        local = np.full((cr1 - cr0 + 1, cc1 - cc0 + 1), bg, dtype=int)
        for r, c in comp:
            local[int(r - cr0), int(c - cc0)] = int(g[r, c])

        for vr, vc, _vcolor in ordered[2:-1]:
            for rr, cc in _bresenham_line((anchor_r, anchor_c), (vr, vc))[1:-1]:
                lr = int(rr - cr0)
                lc = int(cc - cc0)
                if int(local[lr, lc]) == bg:
                    local[lr, lc] = border

        for region in _components(local == bg, connectivity=4):
            if any(
                r == 0 or c == 0 or r == local.shape[0] - 1 or c == local.shape[1] - 1
                for r, c in region
            ):
                continue
            tri_colors = _region_marker_colors(local, region, bg, border)
            if len(tri_colors) < 3:
                continue
            fill_color = min(tri_colors)
            for rr, cc in region:
                out[int(cr0 + rr), int(cc0 + cc)] = fill_color

        for r, c in comp:
            out[int(r), int(c)] = border
        for vr, vc, _vcolor in ordered[2:-1]:
            for rr, cc in _bresenham_line((anchor_r, anchor_c), (vr, vc)):
                out[int(rr), int(cc)] = border

    return out


def _normalize_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(points)
    return [(int(r - r0), int(c - c0)) for r, c in points]


def _polygon_cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    ax, ay = int(a[1]), int(a[0])
    bx, by = int(b[1]), int(b[0])
    cx, cy = int(c[1]), int(c[0])
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _is_strictly_convex(points: list[tuple[int, int]]) -> bool:
    n = len(points)
    if n < 3 or len(set(points)) != n:
        return False
    sign = 0
    for i in range(n):
        cross = _polygon_cross(points[i], points[(i + 1) % n], points[(i + 2) % n])
        if cross == 0:
            return False
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif sign != cur:
            return False
    return True


def _render_polygon(vertices: list[tuple[int, int]], border: int, marker_colors: list[int], bg: int) -> np.ndarray:
    h = max(r for r, _ in vertices) + 1
    w = max(c for _, c in vertices) + 1
    patch = np.full((h, w), bg, dtype=int)
    n = len(vertices)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        for r, c in _bresenham_line(a, b):
            patch[int(r), int(c)] = border
    for (r, c), color in zip(vertices, marker_colors, strict=True):
        patch[int(r), int(c)] = int(color)
    return patch


def _interior_area(patch: np.ndarray, bg: int) -> int:
    area = 0
    for region in _components(patch == bg, connectivity=4):
        if any(r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1 for r, c in region):
            continue
        area += len(region)
    return int(area)


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = patch
    if random.random() < 0.5:
        out = np.fliplr(out)
    rot = random.randint(0, 3)
    if rot:
        out = np.rot90(out, rot)
    return np.array(out, dtype=int, copy=True)


def _sample_convex_vertices(n_vertices: int) -> list[tuple[int, int]] | None:
    min_radius = 3 if n_vertices <= 4 else 4
    max_radius = 6
    for _ in range(LOCAL_GEN_RETRIES):
        ry = random.randint(min_radius, max_radius)
        rx = random.randint(min_radius, max_radius)
        base = random.random() * 2.0 * math.pi
        step = 2.0 * math.pi / float(n_vertices)
        jitter = step * 0.22
        points: list[tuple[int, int]] = []
        cy = float(ry + 1)
        cx = float(rx + 1)
        for i in range(n_vertices):
            ang = base + i * step + random.uniform(-jitter, jitter)
            rr = int(round(cy + ry * math.sin(ang)))
            cc = int(round(cx + rx * math.cos(ang)))
            points.append((int(rr), int(cc)))
        points = _normalize_points(points)
        if not _is_strictly_convex(points):
            continue
        border = 1
        marker_colors = list(range(2, 2 + n_vertices))
        patch = _render_polygon(points, border=border, marker_colors=marker_colors, bg=0)
        if min(patch.shape) < 6 or max(patch.shape) > 14:
            continue
        if _interior_area(patch, bg=0) < max(3, n_vertices - 1):
            continue
        return points
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], gap: int) -> bool:
    h, w = occupied.shape
    ph, pw = shape
    r0 = max(0, int(top - gap))
    c0 = max(0, int(left - gap))
    r1 = min(h, int(top + ph + gap))
    c1 = min(w, int(left + pw + gap))
    return not bool(occupied[r0:r1, c0:c1].any())


def _paint_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    mask = patch != bg
    grid[top : top + ph, left : left + pw][mask] = patch[mask]
    occupied[top : top + ph, left : left + pw] |= mask


def _build_candidate_polygon(bg: int) -> np.ndarray | None:
    n_vertices = random.choice(VERTEX_COUNT_CHOICES)
    vertices = _sample_convex_vertices(n_vertices)
    if vertices is None:
        return None
    colors = [color for color in COLOR_POOL if color != bg]
    border = random.choice(colors)
    markers = random.sample([color for color in colors if color != border], n_vertices)
    patch = _render_polygon(vertices, border=border, marker_colors=markers, bg=bg)
    return _transform_patch(patch)


def generate_input() -> np.ndarray:
    for _ in range(GLOBAL_GEN_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        bg = random.choice(COLOR_POOL)
        grid = np.full((h, w), bg, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        target_count = random.randint(*POLYGON_COUNT_RANGE)
        placed = 0
        for _poly_idx in range(target_count):
            patch = _build_candidate_polygon(bg)
            if patch is None:
                break
            ph, pw = patch.shape
            if ph + 2 * PLACE_MARGIN > h or pw + 2 * PLACE_MARGIN > w:
                break

            spots: list[tuple[int, int]] = []
            for top in range(PLACE_MARGIN, h - ph - PLACE_MARGIN + 1):
                for left in range(PLACE_MARGIN, w - pw - PLACE_MARGIN + 1):
                    if _can_place(occupied, top, left, patch.shape, POLYGON_GAP):
                        spots.append((int(top), int(left)))
            if not spots:
                break

            top, left = random.choice(spots)
            _paint_patch(grid, occupied, patch, top, left, bg)
            placed += 1

        if placed != target_count:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if int(np.count_nonzero(out != bg)) <= int(np.count_nonzero(grid != bg)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid triangulation puzzle input")
