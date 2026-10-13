from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
LEFT = {(-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0)}


def _components(mask: np.ndarray, *, eight: bool = False) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in (DIRS8 if eight else DIRS):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _diamond_from_comp(comp: list[tuple[int, int]], color: int) -> tuple[int, int, int, int]:
    r = round(sum(p[0] for p in comp) / len(comp))
    c = round(sum(p[1] for p in comp) / len(comp))
    radius = max(abs(rr - r) + abs(cc - c) for rr, cc in comp)
    radius = max(0, radius - 1 if color == 2 else radius + 1)
    return r, c, radius, color


def _diamond_cells(center_r: int, center_c: int, radius: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    for r in range(center_r - radius, center_r + radius + 1):
        for c in range(center_c - radius, center_c + radius + 1):
            if 0 <= r < shape[0] and 0 <= c < shape[1] and abs(r - center_r) + abs(c - center_c) == radius:
                cells.append((r, c))
    return cells or [(center_r, center_c)]


def _arc_order(cells: list[tuple[int, int]], cr: int, cc: int) -> list[tuple[int, int]]:
    radius = max(abs(r - cr) + abs(c - cc) for r, c in cells)
    if radius == 0:
        return cells

    def key(p: tuple[int, int]) -> int:
        r, c = p
        if r <= cr and c >= cc:
            return c - cc
        if r >= cr and c >= cc:
            return radius + (r - cr)
        if r >= cr and c <= cc:
            return 2 * radius + (cc - c)
        return 3 * radius + (cr - r)

    return sorted(cells, key=key)


def _draw_short_arc(out: np.ndarray, order: list[tuple[int, int]], a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    if a not in order or b not in order:
        return
    ia, ib = order.index(a), order.index(b)
    n = len(order)
    if (ib - ia) % n <= (ia - ib) % n:
        path = [order[i % n] for i in range(ia, ib + 1 if ib >= ia else ib + n + 1)]
    else:
        path = [order[i % n] for i in range(ia, ib - n - 1 if ib <= ia else ib - 1, -1)]
    for r, c in path:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Resize red and grey diamond outlines, cast trigger rays to them, and connect paired hits along diamond arcs."""
    out = np.array(grid, copy=True)
    out[(out == 2) | (out == 5)] = 7
    diamonds: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
    cell_owner: dict[tuple[int, int], int] = {}
    for color in (2, 5):
        for comp in _components(grid == color, eight=True):
            cr, cc, radius, new_color = _diamond_from_comp(comp, color)
            cells = _diamond_cells(cr, cc, radius, grid.shape)
            idx = len(diamonds)
            diamonds.append((cr, cc, radius, new_color, cells))
            for r, c in cells:
                out[r, c] = new_color
                cell_owner[(r, c)] = idx

    hits: dict[int, list[tuple[tuple[int, int], int, tuple[int, int]]]] = {i: [] for i in range(len(diamonds))}
    for sr, sc in map(tuple, np.argwhere((grid == 1) | (grid == 3))):
        sr, sc = int(sr), int(sc)
        color = int(grid[sr, sc])
        for dr, dc in DIRS:
            r, c = sr, sc
            while True:
                out[r, c] = color
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    break
                owner = cell_owner.get((nr, nc))
                if owner is not None:
                    hits[owner].append(((nr, nc), color, (sr, sc)))
                    break
                r, c = nr, nc

    for idx, items in hits.items():
        if len(items) < 2:
            continue
        cr, cc, _, _, cells = diamonds[idx]
        order = _arc_order(cells, cr, cc)
        items.sort(key=lambda item: order.index(item[0]) if item[0] in order else 10**6)
        for i in range(0, len(items) - 1, 2):
            if items[i][2] == items[i + 1][2]:
                continue
            color = min(items[i][1], items[i + 1][1])
            _draw_short_arc(out, order, items[i][0], items[i + 1][0], color)
    return out


def _draw_diamond(grid: np.ndarray, cr: int, cc: int, radius: int, color: int) -> None:
    for r, c in _diamond_cells(cr, cc, radius, grid.shape):
        grid[r, c] = color


def generate_input() -> np.ndarray:
    grid = np.full((22, 22), 7, dtype=int)
    _draw_diamond(grid, 10, 10, random.choice([4, 5]), 2)
    _draw_diamond(grid, 6, 16, random.choice([3, 4]), 5)
    triggers = [(10, 3, 1), (17, 10, 3), (6, 3, 1), (18, 16, 3), (3, 10, random.choice([1, 3]))]
    random.shuffle(triggers)
    for r, c, color in triggers[: random.randint(2, 4)]:
        grid[r, c] = color
    return grid
