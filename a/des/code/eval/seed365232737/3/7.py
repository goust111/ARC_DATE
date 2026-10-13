from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5

DIRS8 = {
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
}


def _neighbors(mask: np.ndarray, r: int, c: int, connectivity: int):
    steps = DIRS8 if connectivity == 8 else {(-1, 0), (1, 0), (0, -1), (0, 1)}
    h, w = mask.shape
    for dr, dc in steps:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            yield rr, cc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(mask, rr, cc, connectivity):
                    if not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _edge_order(cell: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = cell
    if r == 0:
        return 0, c
    if c == w - 1:
        return 1, r
    if r == h - 1:
        return 2, w - 1 - c
    return 3, h - 1 - r


def _arrow_direction(cells: np.ndarray) -> tuple[int, int]:
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    if r1 - r0 != 1 or c1 - c0 != 1 or len(cells) != 3:
        raise AssertionError("turn marker must be a 3-cell L in a 2x2 box")
    present = {tuple(map(int, p)) for p in cells}
    for rr in (int(r0), int(r1)):
        for cc in (int(c0), int(c1)):
            if (rr, cc) not in present:
                dr = -1 if rr == r0 else 1
                dc = -1 if cc == c0 else 1
                return dr, dc
    raise AssertionError("invalid turn marker")


def _arrow_lookup(grid: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], set[tuple[int, int]]]]:
    lookup: dict[tuple[int, int], tuple[tuple[int, int], set[tuple[int, int]]]] = {}
    for comp in _components(grid == GREY, connectivity=8):
        direction = _arrow_direction(comp)
        cells = {tuple(map(int, p)) for p in comp}
        for cell in cells:
            lookup[cell] = (direction, cells)
    return lookup


def _snake_specs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    h, w = grid.shape
    specs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG, GREY}):
        comps = _components(grid == color, connectivity=8)
        assert len(comps) == 1, "each snake color must form one start marker"
        cells = [tuple(map(int, p)) for p in comps[0]]
        edge_cells = [(r, c) for r, c in cells if r in (0, h - 1) or c in (0, w - 1)]
        assert len(edge_cells) == 1, "each snake needs exactly one edge head"
        head = edge_cells[0]
        body = [p for p in cells if p != head]
        assert 1 <= len(body) <= 4, "start bar length must be 1..4"
        dirs = {(int(np.sign(r - head[0])), int(np.sign(c - head[1]))) for r, c in body}
        dirs.discard((0, 0))
        assert len(dirs) == 1, "start bar must lie on one ray"
        direction = next(iter(dirs))
        expected = {(head[0] + direction[0] * i, head[1] + direction[1] * i) for i in range(1, len(body) + 1)}
        assert set(body) == expected, "start bar must be contiguous"
        specs.append((head, direction, len(body), color))
    specs.sort(key=lambda item: _edge_order(item[0], h, w))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate colored snakes whose start bars encode initial direction and body length.

    A grey 2x2-minus-one-corner L marker changes direction to the missing corner's diagonal.
    Each snake keeps only its most recent L visited cells; older own-color cells are erased.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows = _arrow_lookup(g)
    out = g.copy()

    for head, direction, length, color in _snake_specs(g):
        pos = head
        body: deque[tuple[int, int]] = deque()
        steps = 0
        while steps < h * w * 4:
            steps += 1
            nr, nc = pos[0] + direction[0], pos[1] + direction[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break

            v = int(out[nr, nc])
            if v not in (BG, GREY, color):
                break

            pos = (nr, nc)
            if (nr, nc) in arrows:
                new_direction, arrow_cells = arrows[(nr, nc)]
                for cell in sorted(arrow_cells):
                    out[cell] = color
                    body.append(cell)
                    while len(body) > length:
                        old = body.popleft()
                        if int(out[old]) == color:
                            out[old] = BG
                direction = new_direction
            else:
                out[nr, nc] = color
                body.append((nr, nc))
                while len(body) > length:
                    old = body.popleft()
                    if int(out[old]) == color:
                        out[old] = BG

    return out


def _arrow_cells(top: int, left: int, missing: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)}
    mr = top if missing[0] < 0 else top + 1
    mc = left if missing[1] < 0 else left + 1
    cells.remove((mr, mc))
    return cells


def _arrow_area_clear(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if grid[r, c] != BG:
            return False
        for dr, dc in DIRS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == GREY:
                return False
    return True


def _try_place_arrow(grid: np.ndarray, r: int, c: int, direction: tuple[int, int]) -> bool:
    h, w = grid.shape
    candidates = []
    for top in (r - 1, r):
        for left in (c - 1, c):
            cells = _arrow_cells(top, left, direction)
            if (r, c) not in cells:
                continue
            if top < 1 or left < 1 or top + 1 >= h - 1 or left + 1 >= w - 1:
                continue
            if _arrow_area_clear(grid, cells):
                candidates.append(cells)
    if not candidates:
        return False
    for rr, cc in random.choice(candidates):
        grid[rr, cc] = GREY
    return True


def _edge_start(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return (0, random.randint(3, w - 4)), (1, 0)
    if side == "bottom":
        return (h - 1, random.randint(3, w - 4)), (-1, 0)
    if side == "left":
        return (random.randint(3, h - 4), 0), (0, 1)
    return (random.randint(3, h - 4), w - 1), (0, -1)


def _draw_start(grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int], length: int, color: int) -> bool:
    h, w = grid.shape
    cells = [head] + [(head[0] + direction[0] * i, head[1] + direction[1] * i) for i in range(1, length + 1)]
    if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != BG for r, c in cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 26)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([c for c in range(1, 10) if c != GREY], random.randint(2, 4))

    for color in colors:
        placed = False
        for _ in range(80):
            head, direction = _edge_start(h, w)
            length = random.randint(1, 4)
            if not _draw_start(grid, head, direction, length, color):
                continue

            # Put one reachable turn marker further along the initial ray when possible.
            rr = head[0] + direction[0] * (length + random.randint(2, 5))
            cc = head[1] + direction[1] * (length + random.randint(2, 5))
            if 2 <= rr < h - 2 and 2 <= cc < w - 2:
                new_dirs = [d for d in ((-1, -1), (-1, 1), (1, -1), (1, 1)) if d != direction]
                _try_place_arrow(grid, rr, cc, random.choice(new_dirs))

            placed = True
            break
        if not placed:
            return None

    # Add a few distractor turn markers that may or may not be encountered.
    for _ in range(random.randint(1, 4)):
        top = random.randint(2, h - 4)
        left = random.randint(2, w - 4)
        direction = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        cells = _arrow_cells(top, left, direction)
        if _arrow_area_clear(grid, cells):
            for r, c in cells:
                grid[r, c] = GREY

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid snake input")
