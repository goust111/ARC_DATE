from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    assert 2 <= len(comp) <= 4, "input fragments must have length 2-4"
    if len(rows) == 1:
        r = next(iter(rows))
        cs = sorted(c for _, c in comp)
        assert cs == list(range(cs[0], cs[-1] + 1))
        return (r, cs[0]), (r, cs[-1])
    if len(cols) == 1:
        c = next(iter(cols))
        rs = sorted(r for r, _ in comp)
        assert rs == list(range(rs[0], rs[-1] + 1))
        return (rs[0], c), (rs[-1], c)
    raise AssertionError("fragment must be horizontal or vertical")


def _line(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    if r0 != r1 and c0 != c1:
        return None
    dr = (r1 > r0) - (r1 < r0)
    dc = (c1 > c0) - (c1 < c0)
    n = max(abs(r1 - r0), abs(c1 - c0))
    return {(r0 + dr * i, c0 + dc * i) for i in range(n + 1)}


def _valid_loop(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        deg = 0
        for dr, dc in DIRS.values():
            nb = (r + dr, c + dc)
            if nb in cells:
                deg += 1
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if deg != 2:
            return False
    return len(seen) == len(cells)


def _complete_color(grid: np.ndarray, color: int, blocked: np.ndarray) -> set[tuple[int, int]]:
    comps = _components(grid == color)
    assert 3 <= len(comps) <= 5, "each color should have 3-5 fragments"
    endpoints: list[tuple[int, int]] = []
    endpoint_comp: list[int] = []
    base: set[tuple[int, int]] = set()
    for i, comp in enumerate(comps):
        a, b = _segment_endpoints(comp)
        endpoints.extend([a, b])
        endpoint_comp.extend([i, i])
        base.update(comp)

    candidates: dict[int, list[tuple[int, set[tuple[int, int]]]]] = {i: [] for i in range(len(endpoints))}
    endpoint_set = set(endpoints)
    for i, a in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            if endpoint_comp[i] == endpoint_comp[j]:
                continue
            cells = _line(a, endpoints[j])
            if cells is None or len(cells) <= 2:
                continue
            middle = cells - {a, endpoints[j]}
            if any(blocked[r, c] for r, c in middle):
                continue
            if middle & endpoint_set:
                continue
            candidates[i].append((j, cells))
            candidates[j].append((i, cells))

    solutions: list[set[tuple[int, int]]] = []

    def rec(unused: set[int], added: set[tuple[int, int]]) -> None:
        if len(solutions) > 8:
            return
        if not unused:
            loop = base | added
            if _valid_loop(loop):
                solutions.append(loop)
            return
        i = min(unused)
        for j, cells in candidates[i]:
            if j not in unused:
                continue
            middle = cells - {endpoints[i], endpoints[j]}
            if middle & added:
                continue
            rec(unused - {i, j}, added | middle)

    rec(set(range(len(endpoints))), set())
    assert solutions, "no closed path completion found"
    return min(solutions, key=len)


def _fill_inside(out: np.ndarray, boundary: set[tuple[int, int]], color: int) -> None:
    h, w = out.shape
    wall = np.zeros((h, w), dtype=bool)
    for r, c in boundary:
        wall[r, c] = True
        out[r, c] = color

    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    interior = (~seen) & (~wall) & (out == BG)
    out[interior] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    blocked = g != BG
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        boundary = _complete_color(g, color, blocked & (g != color))
        _fill_inside(out, boundary, color)
    return out


def _edge_cells(start: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    return [(r + dr * i, c + dc * i) for i in range(length + 1)]


def _valid_polygon(seq: list[str], lengths: list[int]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = [(0, 0)]
    r = c = 0
    for direction, length in zip(seq, lengths):
        dr, dc = DIRS[direction]
        r += dr * length
        c += dc * length
        pts.append((r, c))
    if pts[-1] != (0, 0):
        return None
    edges = [_line(pts[i], pts[i + 1]) for i in range(len(seq))]
    assert all(edge is not None for edge in edges)
    edge_sets = [set(edge) for edge in edges if edge is not None]
    for i, a in enumerate(edge_sets):
        for j in range(i + 1, len(edge_sets)):
            inter = a & edge_sets[j]
            adjacent = j == i + 1 or (i == 0 and j == len(edge_sets) - 1)
            if adjacent:
                if len(inter) != 1:
                    return None
            elif inter:
                return None
    cells = set().union(*edge_sets)
    if not _valid_loop(cells):
        return None
    return pts, cells


POLYGON_CATALOG = [
    (["N", "E", "S", "W", "N", "W"], [2, 7, 3, 5, 1, 2]),
    (["S", "W", "S", "W", "N", "E"], [2, 3, 1, 2, 3, 5]),
    (["S", "W", "S", "W", "N", "E", "N", "E"], [3, 6, 1, 3, 3, 3, 1, 6]),
    (["E", "S", "E", "N", "W", "N", "W", "S"], [1, 5, 2, 6, 1, 3, 2, 4]),
    (["W", "S", "W", "N", "W", "N", "E", "N", "E", "S"], [1, 7, 2, 4, 1, 4, 1, 4, 3, 5]),
    (["E", "S", "W", "S", "W", "N", "E", "N", "E", "S"], [1, 7, 3, 3, 3, 7, 3, 5, 2, 2]),
]


def _transform_cells(cells: set[tuple[int, int]], variant: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        if variant == 0:
            nr, nc = r, c
        elif variant == 1:
            nr, nc = c, -r
        elif variant == 2:
            nr, nc = -r, -c
        elif variant == 3:
            nr, nc = -c, r
        elif variant == 4:
            nr, nc = r, -c
        elif variant == 5:
            nr, nc = -r, c
        elif variant == 6:
            nr, nc = c, r
        else:
            nr, nc = -c, -r
        out.add((nr, nc))
    return out


def _sample_polygon(rng: random.Random) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    seq, lengths = rng.choice(POLYGON_CATALOG)
    pts, full = _valid_polygon(list(seq), list(lengths)) or ([], set())
    assert pts and full
    visible: set[tuple[int, int]] = set()
    for i in range(0, len(seq), 2):
        visible.update(_line(pts[i], pts[i + 1]) or set())
    variant = rng.randrange(8)
    full = _transform_cells(full, variant)
    visible = _transform_cells(visible, variant)
    return pts, full, visible


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    arr = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    r0, c0 = min(rs), min(cs)
    for r, c in cells:
        arr[r - r0, c - c0] = True
    return arr


def _place_shape(grid: np.ndarray, occupied: np.ndarray, cells: set[tuple[int, int]], color: int, rng: random.Random) -> bool:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    shape_h = rmax - rmin + 1
    shape_w = cmax - cmin + 1
    h, w = grid.shape
    if shape_h + 2 > h or shape_w + 2 > w:
        return False
    for _ in range(200):
        tr = rng.randint(1, h - shape_h - 1)
        tc = rng.randint(1, w - shape_w - 1)
        placed = {(r - rmin + tr, c - cmin + tc) for r, c in cells}
        rs2 = [r for r, _ in placed]
        cs2 = [c for _, c in placed]
        if occupied[min(rs2) - 1 : max(rs2) + 2, min(cs2) - 1 : max(cs2) + 2].any():
            continue
        for r, c in placed:
            grid[r, c] = color
        occupied[min(rs2) - 1 : max(rs2) + 2, min(cs2) - 1 : max(cs2) + 2] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = rng.sample(PALETTE, rng.randint(1, 3))
    for color in colors:
        _pts, _full, visible = _sample_polygon(rng)
        if not _place_shape(grid, occupied, visible, color, rng):
            return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
