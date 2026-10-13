from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
TEAL = 8

GENERATOR_RETRIES = 500
MIN_GRID_SIZE = 17
MAX_GRID_SIZE = 21
MIN_REGION_AREA = 9


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _find_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components_of_mask(grid == GREEN)
    centers: list[tuple[int, int]] = []
    for comp in comps:
        assert comp.shape[0] == 5, "every green object must be a 5-cell plus"
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        assert r1 - r0 == 2 and c1 - c0 == 2, "plus bbox must be 3x3"
        center = (r0 + 1, c0 + 1)
        assert set(map(tuple, comp.tolist())) == _plus_cells(center), "expected exact plus shape"
        centers.append(center)
    centers.sort()
    assert len(centers) >= 2, "expected multiple pluses"
    return centers


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _connect_diagonals(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    centers = _find_plus_centers(out)
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) != abs(dc):
                continue
            step_r = _sign(dr)
            step_c = _sign(dc)
            gap = [(r1 + k * step_r, c1 + k * step_c) for k in range(1, abs(dr))]
            if all(out[r, c] == BLACK for r, c in gap):
                for r, c in gap:
                    out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal red barriers between aligned pluses, then recolor the smallest and largest reachable background regions.

    Input structure:
    - The grid uses black background and contains several green 5-pixel plus objects.
    - Each plus has a single center cell with four cardinal arms, forming a 3x3 bounding box.
    - Some plus centers lie on clear 45-degree diagonals, while the rest act as distractors or barrier anchors.

    Rule:
    - Preserve every input pixel.
    - Scan plus-center pairs in row-major order; whenever two centers are diagonally aligned and every interior cell on that diagonal is black, paint those interior cells red.
    - Treat all non-black cells after this connection step as walls and find the 4-connected black regions.
    - Recolor the unique smallest black region orange and the unique largest black region teal, leaving all other black regions black.

    Key insight:
    - The diagonal connectors are structural walls rather than decoration: they change the topology of the remaining black space, so region sizes must be measured only after drawing them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN}, "input must contain only black background and green pluses"

    walls = _connect_diagonals(g)
    regions = _components_of_mask(walls == BLACK)
    assert len(regions) >= 3, "expected at least three background regions after adding connectors"

    areas = [int(region.shape[0]) for region in regions]
    assert len(set(areas)) == len(areas), "region areas must be distinct"

    smallest_index = int(np.argmin(areas))
    largest_index = int(np.argmax(areas))
    assert smallest_index != largest_index

    out = walls.copy()
    small = regions[smallest_index]
    large = regions[largest_index]
    out[small[:, 0], small[:, 1]] = ORANGE
    out[large[:, 0], large[:, 1]] = TEAL
    return out


def _transform_point(n: int, point: tuple[int, int], code: int) -> tuple[int, int]:
    r, c = point
    if code == 0:
        return r, c
    if code == 1:
        return r, n - 1 - c
    if code == 2:
        return n - 1 - r, c
    if code == 3:
        return n - 1 - r, n - 1 - c
    if code == 4:
        return c, r
    if code == 5:
        return c, n - 1 - r
    if code == 6:
        return n - 1 - c, r
    return n - 1 - c, n - 1 - r


def _transform_centers(n: int, centers: list[tuple[int, int]], code: int) -> list[tuple[int, int]]:
    return [_transform_point(n, center, code) for center in centers]


def _valid_plus_layout(n: int, centers: list[tuple[int, int]]) -> bool:
    if len(set(centers)) != len(centers):
        return False
    pluses = []
    for center in centers:
        r, c = center
        if not (1 <= r < n - 1 and 1 <= c < n - 1):
            return False
        pluses.append(_plus_cells(center))

    for i, first in enumerate(pluses):
        for second in pluses[i + 1 :]:
            if first & second:
                return False
            for r1, c1 in first:
                for r2, c2 in second:
                    if abs(r1 - r2) + abs(c1 - c2) == 1:
                        return False
    return True


def _add_distractors(
    n: int,
    centers: list[tuple[int, int]],
    target_count: int,
) -> list[tuple[int, int]] | None:
    out = list(centers)
    for _ in range(2000):
        if len(out) >= target_count:
            return out
        candidate = (random.randint(1, n - 2), random.randint(1, n - 2))
        if candidate in out:
            continue
        good = True
        for other in out:
            if max(abs(candidate[0] - other[0]), abs(candidate[1] - other[1])) < 3:
                good = False
                break
            if abs(candidate[0] - other[0]) == abs(candidate[1] - other[1]):
                good = False
                break
        if good and _valid_plus_layout(n, out + [candidate]):
            out.append(candidate)
    return None


def _build_parallel_family(n: int) -> list[tuple[int, int]] | None:
    max_offset = max(3, n // 3)
    for _ in range(200):
        offset_a = random.randint(2, max_offset)
        offset_b = random.randint(2, max_offset)
        if offset_a == offset_b:
            continue
        centers = [
            (1, 1 + offset_a),
            (n - 2 - offset_a, n - 2),
            (1 + offset_b, 1),
            (n - 2, n - 2 - offset_b),
        ]
        centers = _transform_centers(n, centers, random.randrange(8))
        target_count = random.randint(4, 6)
        centers = _add_distractors(n, centers, target_count)
        if centers is not None and _valid_plus_layout(n, centers):
            return centers
    return None


def _build_diamond_family(n: int) -> list[tuple[int, int]] | None:
    max_radius = min(5, (n - 5) // 2)
    for _ in range(400):
        radius = random.randint(3, max_radius)
        row_lo, row_hi = radius + 4, n - radius - 2
        col_lo, col_hi = radius + 4, n - radius - 2
        if row_lo > row_hi or col_lo > col_hi:
            continue
        center_r = random.randint(row_lo, row_hi)
        center_c = random.randint(col_lo, col_hi)

        top = (center_r - radius, center_c)
        left = (center_r, center_c - radius)
        right = (center_r, center_c + radius)
        bottom = (center_r + radius, center_c)

        tail_len = min(top[0] - 1, top[1] - 1)
        if tail_len < 3:
            continue
        tail = (top[0] - tail_len, top[1] - tail_len)

        centers = [top, left, right, bottom, tail]
        centers = _transform_centers(n, centers, random.randrange(8))
        target_count = random.randint(5, 6)
        centers = _add_distractors(n, centers, target_count)
        if centers is not None and _valid_plus_layout(n, centers):
            return centers
    return None


def _render_input(n: int, centers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((n, n), BLACK, dtype=int)
    for center in centers:
        for r, c in _plus_cells(center):
            grid[r, c] = GREEN
    return grid


def _generated_instance_ok(grid: np.ndarray) -> bool:
    try:
        centers = _find_plus_centers(grid)
        if not (4 <= len(centers) <= 6):
            return False
        walls = _connect_diagonals(grid)
    except AssertionError:
        return False

    if np.count_nonzero(walls == RED) == 0:
        return False

    regions = _components_of_mask(walls == BLACK)
    areas = sorted(int(region.shape[0]) for region in regions)
    if not (3 <= len(areas) <= 4):
        return False
    if len(set(areas)) != len(areas):
        return False
    if min(areas) < MIN_REGION_AREA:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return (
        out.shape == grid.shape
        and np.count_nonzero(out == ORANGE) == min(areas)
        and np.count_nonzero(out == TEAL) == max(areas)
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        n = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        family = random.choice(("parallel", "diamond"))
        centers = _build_parallel_family(n) if family == "parallel" else _build_diamond_family(n)
        if centers is None:
            continue
        grid = _render_input(n, centers)
        if _generated_instance_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
