from __future__ import annotations

from collections import deque
import random

import numpy as np

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    dr = r1 - r0
    dc = c1 - c0
    if dr != 0 and dc != 0 and abs(dr) != abs(dc):
        raise ValueError("line must be horizontal, vertical or diagonal")
    if dr == 0 and dc == 0:
        return [(r0, c0)]
    sr = _sign(dr)
    sc = _sign(dc)
    dist = max(abs(dr), abs(dc))
    pts: list[tuple[int, int]] = []
    r, c = r0, c0
    for _ in range(dist + 1):
        pts.append((r, c))
        r += sr
        c += sc
    return pts


def _find_blue_rect(g: np.ndarray) -> tuple[int, int, int, int]:
    blue_cells = np.argwhere(g == BLUE)
    if blue_cells.size == 0:
        raise ValueError("no blue rectangle found")
    r0, c0, r1, c1 = _bbox([(int(r), int(c)) for r, c in blue_cells.tolist()])
    H = r1 - r0 + 1
    W = c1 - c0 + 1
    if H < 2 or W < 2:
        raise ValueError("rectangle too small")
    block = g[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(block == BLUE):
        raise ValueError("non-solid blue region")
    if blue_cells.shape[0] != H * W:
        raise ValueError("non-rectangular blue component")
    return int(r0), int(c0), int(r1), int(c1)


def _find_red_centers(g: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(g == RED)
    centers: list[tuple[int, int]] = []
    for comp in comps:
        if not comp:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        centers.append(((r0 + r1) // 2, (c0 + c1) // 2))
    return centers


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colors to determine scale, scale grid by nearest-neighbor replication,
    hollow the scaled blue rectangle, then draw a red-line path between marker
    centers with region-based recolor.
    """
    g = np.asarray(grid, dtype=int)
    shape = g.shape
    scale = len({int(v) for v in g.flatten() if int(v) != BG})
    if scale <= 0:
        return g.copy()

    scaled = np.kron(g, np.ones((scale, scale), dtype=int))

    r0, c0, r1, c1 = _find_blue_rect(scaled)
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    if rect_h < 3 or rect_w < 3:
        # cannot create hollow interior for this sample.
        return scaled.copy()

    out = scaled.copy()
    interior = np.zeros_like(scaled, dtype=bool)
    interior[r0 + 1 : r1, c0 + 1 : c1] = True
    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    rr = np.arange(scaled.shape[0])[:, None]
    cc = np.arange(scaled.shape[1])[None, :]
    in_rect = (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)
    border = (scaled == BLUE) & in_rect & ((rr == r0) | (rr == r1) | (cc == c0) | (cc == c1))

    centers = _find_red_centers(scaled)
    if len(centers) != 2:
        raise ValueError("expected two red marker components")
    p0, p1 = centers[0], centers[1]
    path = _line_points(p0, p1)

    for r, c in path:
        if not _in_bounds(out.shape, r, c):
            raise ValueError("line out of bounds")
        if (r, c) == p0 or (r, c) == p1:
            continue
        if border[r, c]:
            out[r, c] = YELLOW
        elif interior[r, c]:
            out[r, c] = GREEN
        else:
            out[r, c] = RED

    # Keep markers red.
    out[scaled == RED] = RED
    return out


def _random_choice_points(H: int, W: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (random.randint(0, H - 1), random.randint(0, W - 1)),
        (random.randint(0, H - 1), random.randint(0, W - 1)),
    )


def _point_in_rect(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    return r0 <= r <= r1 and c0 <= c <= c1


def _interior_point(pts: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> bool:
    return any(r0 < r < r1 and c0 < c < c1 for r, c in pts)


def generate_input() -> np.ndarray:
    for _ in range(6000):
        H = random.randint(6, 8)
        W = random.randint(6, 8)
        g = np.full((H, W), BG, dtype=int)

        # Exactly three non-black colors (blue, red, and one boundary color):
        # keeps scale S manageable so scaled outputs stay <= 30.
        boundary_color = random.choice([3, 5, 6, 7, 8, 9])

        rect_h = random.randint(2, max(2, H - 3))
        rect_w = random.randint(2, max(2, W - 3))
        r0 = random.randint(1, H - rect_h - 1)
        c0 = random.randint(1, W - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if rect_h < 2 or rect_w < 2:
            continue
        if r1 >= H - 1 or c1 >= W - 1:
            continue
        g[r0 : r1 + 1, c0 : c1 + 1] = BLUE

        # Place two red markers such that the connecting line crosses rectangle
        # interior (after path expansion by any valid orientation).
        chosen: tuple[tuple[int, int], tuple[int, int]] | None = None
        for _ in range(2000):
            p1, p2 = _random_choice_points(H, W)
            if p1 == p2:
                continue
            if _point_in_rect(p1[0], p1[1], r0, c0, r1, c1):
                continue
            if _point_in_rect(p2[0], p2[1], r0, c0, r1, c1):
                continue
            try:
                path = _line_points(p1, p2)
            except ValueError:
                continue
            if not any(_point_in_rect(r, c, r0, c0, r1, c1) for r, c in path):
                continue
            if not _interior_point(path, r0, c0, r1, c1):
                continue
            chosen = (p1, p2)
            break
        if chosen is None:
            continue

        p1, p2 = chosen
        g[p1[0], p1[1]] = RED
        g[p2[0], p2[1]] = RED

        # Boundary color-band: at least one colored pixel on bottom row or right column.
        if random.random() < 0.6:
            c = random.randint(0, W - 1)
            if g[H - 1, c] == BG:
                g[H - 1, c] = boundary_color
        else:
            r = random.randint(0, H - 1)
            if g[r, W - 1] == BG:
                g[r, W - 1] = boundary_color

        if random.random() < 0.5:
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if g[r, c] == BG and (r, c) not in (p1, p2):
                g[r, c] = boundary_color

        # Ensure color set size exactly three so scale == 3.
        if len({int(v) for v in g.flatten() if int(v) != BG}) != 3:
            continue

        try:
            out = main(g)
        except Exception:
            continue
        if not np.any(out == GREEN):
            continue
        if out.shape[0] > 30 or out.shape[1] > 30:
            continue
        return g

    raise RuntimeError("failed to generate valid puzzle 8 input")
