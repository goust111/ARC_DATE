import numpy as np
from collections import deque


BG = 0
MARKER = 9
LOOP_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _signed_area(path: list[tuple[int, int]]) -> int:
    area = 0
    for i, (r0, c0) in enumerate(path):
        r1, c1 = path[(i + 1) % len(path)]
        area += c0 * r1 - r0 * c1
    return int(area)


def _trace_cycle(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    for r, c in cell_set:
        deg = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
        assert deg == 2, "loop pixels must form a simple 4-connected cycle"

    start = min(cell_set)
    start_neighbors = sorted((p for p in _neighbors4(*start) if p in cell_set))
    assert len(start_neighbors) == 2

    def walk(first: tuple[int, int]) -> list[tuple[int, int]]:
        order = [start]
        prev = start
        cur = first
        while True:
            order.append(cur)
            nxts = [p for p in _neighbors4(*cur) if p in cell_set and p != prev]
            assert len(nxts) == 1
            nxt = nxts[0]
            if nxt == start:
                break
            prev, cur = cur, nxt
        assert len(order) == len(cell_set)
        return order

    a = walk(start_neighbors[0])
    b = walk(start_neighbors[1])
    area_a = _signed_area(a)
    area_b = _signed_area(b)
    assert area_a != 0 and area_b != 0
    return a if area_a > 0 else b


def _loop_interior_mask(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    wall = np.zeros((local_h, local_w), dtype=bool)
    for r, c in cells:
        wall[r - r0, c - c0] = True

    ext = np.zeros((local_h, local_w), dtype=bool)
    q = deque()
    for lr in range(local_h):
        for lc in range(local_w):
            if lr not in (0, local_h - 1) and lc not in (0, local_w - 1):
                continue
            if not bool(wall[lr, lc]) and not bool(ext[lr, lc]):
                ext[lr, lc] = True
                q.append((lr, lc))

    while q:
        lr, lc = q.popleft()
        for nlr, nlc in _neighbors4(lr, lc):
            if 0 <= nlr < local_h and 0 <= nlc < local_w and not bool(wall[nlr, nlc]) and not bool(ext[nlr, nlc]):
                ext[nlr, nlc] = True
                q.append((nlr, nlc))

    interior = np.zeros(shape, dtype=bool)
    for lr in range(local_h):
        for lc in range(local_w):
            if not bool(wall[lr, lc]) and not bool(ext[lr, lc]):
                interior[r0 + lr, c0 + lc] = True
    return interior


def _extract_loops(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    loops: list[dict] = []
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, MARKER))
    for color in colors:
        for comp in _components(g == color):
            order = _trace_cycle(comp)
            interior = _loop_interior_mask(g.shape, comp)
            markers = [tuple(map(int, p)) for p in np.argwhere(interior & (g == MARKER))]
            loops.append(
                {
                    "color": color,
                    "cells": comp,
                    "order": order,
                    "start": min(comp),
                    "markers": markers,
                }
            )
    loops.sort(key=lambda item: item["start"])
    return loops


def _nearest_boundary_index(marker: tuple[int, int], order: list[tuple[int, int]]) -> int:
    mr, mc = marker
    best = min(abs(mr - r) + abs(mc - c) for r, c in order)
    candidates = [i for i, (r, c) in enumerate(order) if abs(mr - r) + abs(mc - c) == best]
    return min(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll each closed loop into a horizontal segment whose length equals the loop perimeter.
    Interior maroon markers are projected to the nearest loop pixel and mapped to that pixel's
    clockwise index from the top-leftmost starting point.
    """
    g = np.asarray(grid, dtype=int)
    loops = _extract_loops(g)
    assert loops, "expected at least one closed loop"

    width = max(len(loop["order"]) for loop in loops)
    out = np.full((len(loops), width), BG, dtype=int)

    for row, loop in enumerate(loops):
        order = loop["order"]
        out[row, : len(order)] = loop["color"]
        for marker in loop["markers"]:
            out[row, _nearest_boundary_index(marker, order)] = MARKER

    return out


def _boundary_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(region[r, c]):
                continue
            if any(not (0 <= nr < h and 0 <= nc < w and bool(region[nr, nc])) for nr, nc in _neighbors4(r, c)):
                boundary[r, c] = True
    return boundary


def _shape_rectangle() -> np.ndarray:
    h = int(RNG.integers(5, 11))
    w = int(RNG.integers(5, 11))
    return np.ones((h, w), dtype=bool)


def _shape_cut_corner() -> np.ndarray:
    h = int(RNG.integers(6, 12))
    w = int(RNG.integers(6, 12))
    cut_h = int(RNG.integers(1, h - 3))
    cut_w = int(RNG.integers(1, w - 3))
    region = np.ones((h, w), dtype=bool)
    corner = int(RNG.integers(4))
    if corner == 0:
        region[:cut_h, :cut_w] = False
    elif corner == 1:
        region[:cut_h, w - cut_w :] = False
    elif corner == 2:
        region[h - cut_h :, :cut_w] = False
    else:
        region[h - cut_h :, w - cut_w :] = False
    return region


def _shape_side_notch() -> np.ndarray:
    h = int(RNG.integers(7, 13))
    w = int(RNG.integers(7, 13))
    region = np.ones((h, w), dtype=bool)
    side = int(RNG.integers(4))
    if side in (0, 1):
        notch_w = int(RNG.integers(1, w - 4))
        start = int(RNG.integers(2, w - notch_w - 1))
        depth = int(RNG.integers(1, h - 3))
        if side == 0:
            region[:depth, start : start + notch_w] = False
        else:
            region[h - depth :, start : start + notch_w] = False
    else:
        notch_h = int(RNG.integers(1, h - 4))
        start = int(RNG.integers(2, h - notch_h - 1))
        depth = int(RNG.integers(1, w - 3))
        if side == 2:
            region[start : start + notch_h, :depth] = False
        else:
            region[start : start + notch_h, w - depth :] = False
    return region


def _random_region() -> np.ndarray:
    for _ in range(80):
        family = int(RNG.integers(3))
        if family == 0:
            region = _shape_rectangle()
        elif family == 1:
            region = _shape_cut_corner()
        else:
            region = _shape_side_notch()

        if len(_components(region)) != 1:
            continue

        boundary = _boundary_from_region(region)
        interior = region & (~boundary)
        if int(interior.sum()) < 3:
            continue

        patch = np.full(region.shape, BG, dtype=int)
        patch[boundary] = 1
        try:
            _trace_cycle([tuple(map(int, p)) for p in np.argwhere(patch == 1)])
        except AssertionError:
            continue
        return region
    raise RuntimeError("failed to sample a valid loop shape")


def _marker_candidates(boundary_order: list[tuple[int, int]], interior_cells: list[tuple[int, int]]):
    by_index: dict[int, list[tuple[int, int]]] = {}
    for cell in interior_cells:
        mr, mc = cell
        dists = [abs(mr - r) + abs(mc - c) for r, c in boundary_order]
        best = min(dists)
        indices = [i for i, d in enumerate(dists) if d == best]
        if len(indices) != 1:
            continue
        by_index.setdefault(indices[0], []).append(cell)
    return by_index


def _make_patch(color: int, marker_count: int) -> np.ndarray:
    for _ in range(100):
        region = _random_region()
        boundary = _boundary_from_region(region)
        patch = np.full(region.shape, BG, dtype=int)
        patch[boundary] = color
        order = _trace_cycle([tuple(map(int, p)) for p in np.argwhere(patch == color)])
        interior_cells = [tuple(map(int, p)) for p in np.argwhere(region & (~boundary))]
        candidates = _marker_candidates(order, interior_cells)
        if len(candidates) < marker_count:
            continue

        if marker_count:
            chosen_indices = list(RNG.choice(sorted(candidates), size=marker_count, replace=False))
            for idx in chosen_indices:
                cells = candidates[int(idx)]
                cell = cells[int(RNG.integers(len(cells)))]
                patch[cell] = MARKER
        return patch
    raise RuntimeError("failed to build loop patch")


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    h, w = grid.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    return bool(np.all(grid[r0:r1, c0:c1] == BG))


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_lo = 1
    row_hi = h - ph - 1
    col_lo = 1
    col_hi = w - pw - 1
    if row_hi < row_lo or col_hi < col_lo:
        return False

    for _ in range(120):
        top = int(RNG.integers(row_lo, row_hi + 1))
        left = int(RNG.integers(col_lo, col_hi + 1))
        if not _can_place(grid, patch, top, left):
            continue
        mask = patch != BG
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_loops = int(RNG.integers(2, 5))
        colors = list(RNG.choice(LOOP_COLORS, size=n_loops, replace=False))
        marker_counts = [int(RNG.integers(0, 4)) for _ in range(n_loops)]
        patches: list[np.ndarray] = []
        try:
            for color, marker_count in zip(colors, marker_counts):
                patches.append(_make_patch(int(color), marker_count))
        except RuntimeError:
            continue

        max_h = max(p.shape[0] for p in patches)
        max_w = max(p.shape[1] for p in patches)
        area_budget = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches)
        base = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(area_budget * 1.8))))
        h = int(RNG.integers(base, base + 9))
        w = int(RNG.integers(base, base + 9))
        grid = np.full((h, w), BG, dtype=int)

        order = list(RNG.permutation(len(patches)))
        ok = True
        for idx in order:
            if not _place_patch(grid, patches[int(idx)]):
                ok = False
                break
        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape[0] != n_loops:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
