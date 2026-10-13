from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
FILL_BY_SIDES = {3: 2, 4: 1, 5: 3}
LINE_COLOR = 5
BG_CHOICES = (0, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 32)
GRID_W_RANGE = (18, 32)
POLY_COUNT_CHOICES = (1, 2, 3, 4)
POLY_COUNT_WEIGHTS = (2, 3, 3, 2)
PATCH_RETRIES = 120
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400
RECT_PAD = 2

POLYGON_TEMPLATES: dict[int, list[tuple[list[tuple[int, int]], int, int]]] = {
    3: [
        ([(0, 0), (0, 4), (2, 2)], 2, 4),
        ([(0, 0), (4, 0), (2, 2)], 2, 4),
        ([(0, 2), (2, 0), (2, 4)], 2, 4),
    ],
    4: [
        ([(0, 0), (0, 4), (3, 4), (3, 0)], 1, 4),
        ([(0, 2), (2, 4), (4, 2), (2, 0)], 1, 4),
        ([(0, 0), (0, 4), (2, 6), (2, 2)], 2, 4),
    ],
    5: [
        ([(0, 2), (2, 4), (4, 4), (4, 0), (2, 0)], 1, 4),
        ([(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], 1, 4),
    ],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _closed_bg_regions(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _region_side_count(region: list[tuple[int, int]]) -> int:
    # Use cell centers on a doubled lattice so the convex hull retains the visible polygon geometry.
    hull = _convex_hull([(2 * c + 1, 2 * r + 1) for r, c in region])
    return len(hull)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background region by its polygon side count.

    Regions with 3, 4, and 5 sides become red, blue, and green respectively.
    Open background that touches the outer border is left unchanged.
    """
    g = np.asarray(grid, dtype=int)
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])

    out = g.copy()
    for region in _closed_bg_regions(g, bg):
        side_count = _region_side_count(region)
        fill = FILL_BY_SIDES.get(side_count)
        if fill is None:
            continue
        for r, c in region:
            out[r, c] = fill
    return out


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(r - min_r, c - min_c) for r, c in vertices]


def _transform_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [(r, c) for r, c in vertices]
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    for _ in range(random.randrange(4)):
        pts = [(c, -r) for r, c in pts]
    return _normalize_vertices(pts)


def _draw_outline(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    outline: set[tuple[int, int]] = set()
    n = len(vertices)
    for i in range(n):
        r0, c0 = vertices[i]
        r1, c1 = vertices[(i + 1) % n]
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        span_r = abs(r1 - r0)
        span_c = abs(c1 - c0)
        assert span_r == 0 or span_c == 0 or span_r == span_c
        steps = max(span_r, span_c)
        for k in range(steps + 1):
            outline.add((r0 + dr * k, c0 + dc * k))
    return outline


def _make_patch(side_count: int, max_h: int, max_w: int, scale_cap: int) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        template, min_scale, max_scale = random.choice(POLYGON_TEMPLATES[side_count])
        hi = min(max_scale, scale_cap)
        if hi < min_scale:
            continue

        scale = random.randint(min_scale, hi)
        vertices = [(r * scale, c * scale) for r, c in template]
        vertices = _transform_vertices(vertices)
        outline = _draw_outline(vertices)

        max_r = max(r for r, _ in outline)
        max_c = max(c for _, c in outline)
        patch = np.zeros((max_r + 3, max_c + 3), dtype=int)
        for r, c in outline:
            patch[r + 1, c + 1] = LINE_COLOR

        if patch.shape[0] > max_h or patch.shape[1] > max_w:
            continue

        regions = _closed_bg_regions(patch, 0)
        if len(regions) != 1:
            continue
        if _region_side_count(regions[0]) != side_count:
            continue
        return patch
    return None


def _choose_side_counts(n_polys: int) -> list[int]:
    if n_polys == 1:
        return [random.choice((3, 4, 5))]

    if n_polys >= 3 and random.random() < 0.55:
        counts = [3, 4, 5] + [random.choice((3, 4, 5)) for _ in range(n_polys - 3)]
        random.shuffle(counts)
        return counts

    counts = [random.choice((3, 4, 5)) for _ in range(n_polys)]
    if n_polys == 2 and counts[0] == counts[1] and random.random() < 0.6:
        counts[1] = random.choice([v for v in (3, 4, 5) if v != counts[0]])
    random.shuffle(counts)
    return counts


def _paint_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        if blocked[top : top + ph, left : left + pw].any():
            continue

        view = grid[top : top + ph, left : left + pw]
        if np.any((patch == LINE_COLOR) & (view != bg)):
            continue

        view[patch == LINE_COLOR] = LINE_COLOR

        r0 = max(0, top - RECT_PAD)
        c0 = max(0, left - RECT_PAD)
        r1 = min(h, top + ph + RECT_PAD)
        c1 = min(w, left + pw + RECT_PAD)
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _valid_generated_grid(grid: np.ndarray, expected_counts: list[int], bg: int) -> bool:
    if int(np.sum(grid == bg)) <= int(np.sum(grid == LINE_COLOR)):
        return False

    regions = _closed_bg_regions(grid, bg)
    if len(regions) != len(expected_counts):
        return False

    actual_counts = sorted(_region_side_count(region) for region in regions)
    if actual_counts != sorted(expected_counts):
        return False

    out = main(grid)
    if not np.array_equal(out[grid == LINE_COLOR], grid[grid == LINE_COLOR]):
        return False

    for region in regions:
        fill = FILL_BY_SIDES[_region_side_count(region)]
        for r, c in region:
            if int(out[r, c]) != fill:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        n_polys = random.choices(POLY_COUNT_CHOICES, weights=POLY_COUNT_WEIGHTS, k=1)[0]
        bg = random.choice(BG_CHOICES)

        grid = np.full((h, w), bg, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        side_counts = _choose_side_counts(n_polys)

        scale_cap = 4 if n_polys <= 2 else 3
        if min(h, w) >= 28 and random.random() < 0.35:
            scale_cap += 1

        patches: list[np.ndarray] = []
        ok = True
        for side_count in side_counts:
            patch = _make_patch(side_count, h - 2, w - 2, scale_cap)
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
        for patch in patches:
            if not _paint_patch(grid, blocked, patch, bg):
                ok = False
                break
        if not ok:
            continue

        if _valid_generated_grid(grid, side_counts, bg):
            return grid

    raise RuntimeError("failed to generate a valid ARC input")
