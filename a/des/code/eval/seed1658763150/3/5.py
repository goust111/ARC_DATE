import random
from collections import deque

import numpy as np


BG = 0
RED = 2
AZURE = 8
RING_COLOR_POOL = (1, 3, 4, 5, 6, 7, 9)

GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


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
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_ring_component(comp: list[tuple[int, int]]) -> bool:
    cells = set(comp)
    if len(cells) < 8:
        return False
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if degree != 2:
            return False
    return True


def _trace_cycle(comp: list[tuple[int, int]], first: tuple[int, int]) -> list[tuple[int, int]] | None:
    cells = set(comp)
    start = min(cells)
    order = [start, first]
    prev = start
    cur = first
    while True:
        nbrs = [(nr, nc) for nr, nc in _neighbors4(cur[0], cur[1]) if (nr, nc) in cells]
        if len(nbrs) != 2:
            return None
        nxt_choices = [p for p in nbrs if p != prev]
        if len(nxt_choices) != 1:
            return None
        nxt = nxt_choices[0]
        if nxt == start:
            if len(order) == len(cells):
                return order
            return None
        if nxt in order:
            return None
        order.append(nxt)
        prev, cur = cur, nxt


def _clockwise_cycle(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(comp)
    start = min(cells)
    nbrs = [(nr, nc) for nr, nc in _neighbors4(start[0], start[1]) if (nr, nc) in cells]
    if len(nbrs) != 2:
        raise AssertionError("ring start must have degree two")

    preferred = (start[0], start[1] + 1)
    trial_order = []
    if preferred in nbrs:
        trial_order.append(preferred)
    trial_order.extend(p for p in nbrs if p != preferred)

    order = None
    for first in trial_order:
        order = _trace_cycle(comp, first)
        if order is not None:
            break
    if order is None:
        raise AssertionError("failed to trace ring cycle")

    signed = 0
    n = len(order)
    for i, (r1, c1) in enumerate(order):
        r2, c2 = order[(i + 1) % n]
        signed += c1 * r2 - c2 * r1

    if signed < 0:
        order = [order[0]] + list(reversed(order[1:]))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored ring by the number of azure dots in its cavity.

    The red 2-pixel interior marker touches one inner wall cell and points inward.
    Touching the top or left wall means clockwise rotation; touching the bottom or
    right wall means counterclockwise rotation. All interior markers are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    ring_mask = (g != BG) & (g != RED) & (g != AZURE)
    ring_components = [comp for comp in _components(ring_mask) if _is_ring_component(comp)]

    ring_id = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(ring_components):
        for r, c in comp:
            ring_id[r, c] = idx

    out = np.full_like(g, BG)
    for comp in ring_components:
        for r, c in comp:
            out[r, c] = g[r, c]

    cavity_mask = ring_id < 0
    processed_rings: set[int] = set()
    for cavity in _components(cavity_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cavity):
            continue

        adjacent_rings = set()
        for r, c in cavity:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and ring_id[nr, nc] >= 0:
                    adjacent_rings.add(int(ring_id[nr, nc]))
        if len(adjacent_rings) != 1:
            continue

        ring_idx = next(iter(adjacent_rings))
        ring_comp = ring_components[ring_idx]
        cycle = _clockwise_cycle(ring_comp)

        azure_count = sum(int(g[r, c]) == AZURE for r, c in cavity)
        red_cells = [(r, c) for r, c in cavity if int(g[r, c]) == RED]
        assert len(red_cells) == 2, "each cavity must contain exactly one 2-pixel red arrow"
        assert abs(red_cells[0][0] - red_cells[1][0]) + abs(red_cells[0][1] - red_cells[1][1]) == 1

        touching_dirs: list[tuple[int, int]] = []
        for r, c in red_cells:
            dirs = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(ring_id[nr, nc]) == ring_idx:
                    dirs.append((nr - r, nc - c))
            if dirs:
                touching_dirs.extend(dirs)

        assert touching_dirs, "red arrow must touch its ring"
        wall_dir = touching_dirs[0]
        clockwise = wall_dir in {(-1, 0), (0, -1)}

        values = np.array([int(g[r, c]) for r, c in cycle], dtype=int)
        shifted = np.roll(values, azure_count if clockwise else -azure_count)
        for (r, c), value in zip(cycle, shifted):
            out[r, c] = int(value)

        processed_rings.add(ring_idx)

    for ring_idx, comp in enumerate(ring_components):
        if ring_idx in processed_rings:
            continue
        for r, c in comp:
            out[r, c] = int(g[r, c])

    return out


def _rectangle_cycle(r0: int, c0: int, height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c0 + width):
        cells.append((r0, c))
    for r in range(r0 + 1, r0 + height):
        cells.append((r, c0 + width - 1))
    for c in range(c0 + width - 2, c0 - 1, -1):
        cells.append((r0 + height - 1, c))
    for r in range(r0 + height - 2, r0, -1):
        cells.append((r, c0))
    return cells


def _side_cells(r0: int, c0: int, height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0 + 1, c) for c in range(c0 + 2, c0 + width - 2)]
    if side == "bottom":
        return [(r0 + height - 2, c) for c in range(c0 + 2, c0 + width - 2)]
    if side == "left":
        return [(r, c0 + 1) for r in range(r0 + 2, r0 + height - 2)]
    if side == "right":
        return [(r, c0 + width - 2) for r in range(r0 + 2, r0 + height - 2)]
    raise ValueError(side)


def _arrow_cells(r0: int, c0: int, height: int, width: int, side: str) -> list[tuple[int, int]]:
    touch = random.choice(_side_cells(r0, c0, height, width, side))
    tr, tc = touch
    if side == "top":
        return [(tr, tc), (tr + 1, tc)]
    if side == "bottom":
        return [(tr, tc), (tr - 1, tc)]
    if side == "left":
        return [(tr, tc), (tr, tc + 1)]
    return [(tr, tc), (tr, tc - 1)]


def _interior_cells(r0: int, c0: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r0 + height - 1) for c in range(c0 + 1, c0 + width - 1)]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _paint_ring(grid: np.ndarray, r0: int, c0: int, height: int, width: int, n_colors: int) -> int:
    cycle = _rectangle_cycle(r0, c0, height, width)
    ring_len = len(cycle)
    assert ring_len >= 16

    pattern = random.sample(RING_COLOR_POOL, n_colors)
    values = [pattern[i % n_colors] for i in range(ring_len)]
    assert all(values[i] != values[(i + 1) % ring_len] for i in range(ring_len))
    for (r, c), value in zip(cycle, values):
        grid[r, c] = int(value)
    return ring_len


def _candidate_azure_counts(max_count: int, n_colors: int) -> list[int]:
    return [n for n in range(1, max_count + 1) if n % n_colors != 0]


def _place_ring(grid: np.ndarray, existing: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        height = random.randint(5, min(10, h - 2))
        width = random.randint(5, min(10, w - 2))
        if h - height - 1 < 1 or w - width - 1 < 1:
            continue

        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, w - width - 1)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        if any(_boxes_overlap(bbox, other, gap=2) for other in existing):
            continue

        n_colors = random.choice((2, 3, 4))
        ring_len = 2 * (height + width) - 4
        if ring_len % n_colors == 1:
            continue

        interior = _interior_cells(r0, c0, height, width)
        free_after_arrow = len(interior) - 2
        if free_after_arrow < 1:
            continue

        side = random.choice(("top", "left", "bottom", "right"))
        arrow = _arrow_cells(r0, c0, height, width, side)
        azure_choices = _candidate_azure_counts(min(6, free_after_arrow), n_colors)
        if not azure_choices:
            continue
        azure_count = random.choice(azure_choices)

        _paint_ring(grid, r0, c0, height, width, n_colors)
        for r, c in arrow:
            grid[r, c] = RED

        remaining = [cell for cell in interior if cell not in set(arrow)]
        for r, c in random.sample(remaining, azure_count):
            grid[r, c] = AZURE

        existing.append(bbox)
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    n_rings = random.randint(1, 3)
    min_side = {1: 11, 2: 16, 3: 20}[n_rings]
    h = random.randint(min_side, min_side + 8)
    w = random.randint(min_side, min_side + 8)
    grid = np.full((h, w), BG, dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_rings):
        if not _place_ring(grid, boxes):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue

        ring_mask = (grid != BG) & (grid != RED) & (grid != AZURE)
        if not np.any((grid != out) & ring_mask):
            continue
        if np.any(out == RED) or np.any(out == AZURE):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
