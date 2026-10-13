import random
from collections import deque

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _components_by_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _is_complete_square(component: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != c1 - c0:
        return False, None
    pts = set(component)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False, None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False, None
    return True, (r0, c0, r1, c1)


def _corner_marks(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0), (r0, c0 - 1),
        (r0 - 1, c1), (r0, c1 + 1),
        (r1 + 1, c0), (r1, c0 - 1),
        (r1 + 1, c1), (r1, c1 + 1),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find complete square objects and place red markers just outside each corner.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components_by_color(g, color):
            ok, bbox = _is_complete_square(comp)
            if not ok:
                continue
            for r, c in _corner_marks(bbox):
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = RED
    return out


def _region_is_free(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 0) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not bool(occ[rr0:rr1 + 1, cc0:cc1 + 1].any())


def _paint_frame(grid: np.ndarray, color: int, r0: int, c0: int, size: int) -> None:
    r1, c1 = r0 + size - 1, c0 + size - 1
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _paint_valid_square(grid: np.ndarray, rng: random.Random, color: int, r0: int, c0: int, size: int) -> None:
    kind = rng.choice(["frame", "solid", "filled_frame"])
    if kind == "frame":
        _paint_frame(grid, color, r0, c0, size)
        return
    if kind == "solid":
        grid[r0:r0 + size, c0:c0 + size] = color
        return
    _paint_frame(grid, color, r0, c0, size)
    if size >= 4:
        interior = [(r, c) for r in range(r0 + 1, r0 + size - 1) for c in range(c0 + 1, c0 + size - 1)]
        rng.shuffle(interior)
        fill_n = rng.randint(1, max(1, len(interior) // 2))
        for r, c in interior[:fill_n]:
            grid[r, c] = color


def _paint_invalid_object(grid: np.ndarray, rng: random.Random, color: int, r0: int, c0: int, h: int, w: int, kind: str) -> None:
    if kind == "rect_frame":
        rr1, cc1 = r0 + h - 1, c0 + w - 1
        grid[r0, c0:cc1 + 1] = color
        grid[rr1, c0:cc1 + 1] = color
        grid[r0:rr1 + 1, c0] = color
        grid[r0:rr1 + 1, cc1] = color
    elif kind == "bar_h":
        grid[r0, c0:c0 + w] = color
    elif kind == "bar_v":
        grid[r0:r0 + h, c0] = color
    elif kind == "broken_square":
        size = h
        _paint_frame(grid, color, r0, c0, size)
        boundary = []
        for c in range(c0, c0 + size):
            boundary.append((r0, c))
            boundary.append((r0 + size - 1, c))
        for r in range(r0, r0 + size):
            boundary.append((r, c0))
            boundary.append((r, c0 + size - 1))
        rr, cc = rng.choice(boundary)
        grid[rr, cc] = BG
    elif kind == "l_shape":
        grid[r0:r0 + h, c0] = color
        grid[r0 + h - 1, c0:c0 + w] = color
    else:
        raise ValueError(kind)


def _sample_placeable_rect(
    rng: random.Random,
    occ: np.ndarray,
    height: int,
    width: int,
    margin: int,
) -> tuple[int, int] | None:
    h, w = occ.shape
    rows = list(range(0, h - height + 1))
    cols = list(range(0, w - width + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            if _region_is_free(occ, r0, c0, r0 + height - 1, c0 + width - 1, margin=margin):
                return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_valid = rng.randint(1, 4)
    n_invalid = rng.randint(2, 5)
    object_colors = PALETTE.copy()
    rng.shuffle(object_colors)
    while len(object_colors) < n_valid + n_invalid:
        object_colors.extend(PALETTE)

    valid_drawn = 0
    for i in range(n_valid):
        size = rng.randint(2, 6)
        pos = _sample_placeable_rect(rng, occ, size, size, margin=1)
        if pos is None:
            return None
        r0, c0 = pos
        _paint_valid_square(grid, rng, object_colors[i], r0, c0, size)
        occ[r0:r0 + size, c0:c0 + size] = True
        valid_drawn += 1

    invalid_kinds = ["rect_frame", "bar_h", "bar_v", "broken_square", "l_shape"]
    invalid_drawn = 0
    for i in range(n_invalid):
        kind = rng.choice(invalid_kinds)
        if kind == "rect_frame":
            hh = rng.randint(2, 5)
            ww = rng.randint(2, 5)
            if hh == ww:
                ww += 1
        elif kind == "bar_h":
            hh, ww = 1, rng.randint(3, 7)
        elif kind == "bar_v":
            hh, ww = rng.randint(3, 7), 1
        elif kind == "broken_square":
            hh = ww = rng.randint(3, 6)
        else:
            hh = rng.randint(2, 5)
            ww = rng.randint(2, 5)
            if hh == 1 and ww == 1:
                ww = 2
        pos = _sample_placeable_rect(rng, occ, hh, ww, margin=1)
        if pos is None:
            continue
        r0, c0 = pos
        _paint_invalid_object(grid, rng, object_colors[n_valid + i], r0, c0, hh, ww, kind)
        occ[r0:r0 + hh, c0:c0 + ww] = True
        invalid_drawn += 1

    if valid_drawn == 0 or invalid_drawn == 0:
        return None

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None

    red_count = int(np.sum(transformed == RED) - np.sum(grid == RED))
    if red_count < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
