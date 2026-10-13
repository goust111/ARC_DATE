from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
CYAN = 8

GEN_ATTEMPTS = 800
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _greened_crosses(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cells = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if all(grid[rr, cc] == CYAN for rr, cc in cells):
                for rr, cc in cells:
                    out[rr, cc] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn cyan crosses green, reject blue rectangles whose bounding boxes contain green, and crop the smallest survivor.
    """
    g = np.asarray(grid, dtype=int)
    marked = _greened_crosses(g)

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(g == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        crop = marked[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(crop == GREEN):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, (r0, c0, r1, c1)))
    assert candidates, "expected at least one uncontaminated blue rectangle"
    candidates.sort(key=lambda item: (item[0], item[1]))
    area, box = candidates[0]
    assert len(candidates) == 1 or candidates[1][0] != area, "expected unique smallest survivor"
    r0, c0, r1, c1 = box
    return np.full((r1 - r0 + 1, c1 - c0 + 1), BLUE, dtype=int)


def _fits(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 2 < pr0 or pr1 + 2 < r0 or c1 + 2 < pc0 or pc1 + 2 < c0):
            return False
    return True


def _place_cross(grid: np.ndarray, r: int, c: int) -> None:
    for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        grid[rr, cc] = CYAN


def _draw_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int], contaminated: bool) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    if contaminated:
        cr = random.randint(r0 + 2, r1 - 2)
        cc = random.randint(c0 + 2, c1 - 2)
        _place_cross(grid, cr, cc)


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(16, 23)
        width = random.randint(16, 23)
        grid = np.full((height, width), BLACK, dtype=int)
        rect_count = random.randint(3, 5)
        clean_count = random.randint(1, rect_count - 1)
        contaminated_count = rect_count - clean_count

        dims: list[tuple[int, int]] = []
        areas: set[int] = set()
        while len(dims) < rect_count:
            rh = random.randint(5, 7)
            rw = random.randint(5, 8)
            area = rh * rw
            if area not in areas:
                areas.add(area)
                dims.append((rh, rw))

        placed: list[tuple[int, int, int, int]] = []
        for rh, rw in dims:
            for _attempt in range(100):
                r0 = random.randint(1, height - rh - 2)
                c0 = random.randint(1, width - rw - 2)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if _fits(rect, placed, height, width):
                    placed.append(rect)
                    break
            else:
                break
        if len(placed) != rect_count:
            continue

        order = list(range(rect_count))
        random.shuffle(order)
        contaminated = set(order[:contaminated_count])
        clean = [i for i in range(rect_count) if i not in contaminated]
        if not clean:
            continue
        for idx, rect in enumerate(placed):
            _draw_rectangle(grid, rect, contaminated=idx in contaminated)

        # A few harmless extra crosses keep the template detection from depending on blue objects only.
        for _ in range(random.randint(0, 2)):
            for _try in range(50):
                r = random.randint(2, height - 3)
                c = random.randint(2, width - 3)
                cells = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if all(grid[rr, cc] == BLACK for rr, cc in cells):
                    _place_cross(grid, r, c)
                    break

        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate puzzle 030/9 input")

