from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
PALETTE = [c for c in range(10) if c != BG]
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.ravel(grid)))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "top": int(arr[:, 0].min()),
                    "left": int(arr[:, 1].min()),
                }
            )
    return comps


def _perimeter(coords: np.ndarray, coord_set: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in coords:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (int(r + dr), int(c + dc)) not in coord_set:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object by a horizontal segment whose length equals
    that object's perimeter, ordered from top to bottom, with one background row
    between consecutive segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one foreground object"

    items: list[tuple[int, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        coord_set = {tuple(map(int, xy)) for xy in coords}
        p = _perimeter(coords, coord_set)
        assert p <= 20
        items.append((comp["top"], comp["left"], p, int(comp["color"])))

    items.sort(key=lambda x: (x[0], x[1]))
    out_h = 2 * len(items) - 1
    out_w = max(p for _, _, p, _ in items)
    out = np.full((out_h, out_w), bg, dtype=int)
    for i, (_top, _left, p, color) in enumerate(items):
        out[2 * i, :p] = color
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_variants() -> list[set[tuple[int, int]]]:
    shapes = [
        {(0, 0)},
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (1, 0), (2, 0)},
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 1), (2, 1), (2, 0)},
        {(0, 0), (0, 1), (1, 1), (2, 1)},
        {(0, 1), (0, 0), (1, 0), (2, 0)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (1, 0), (1, 1), (2, 0)},
        {(0, 1), (1, 0), (1, 1), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2)},
        {(0, 1), (0, 2), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
        {(0, 2), (0, 1), (1, 1), (1, 0), (2, 0)},
        {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
        {(0, 2), (1, 2), (1, 1), (2, 1), (2, 0)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (3, 0)},
        {(0, 0), (0, 1), (0, 2), (0, 3)},
        {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
        {(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)},
    ]
    uniq = []
    seen = set()
    for shape in shapes:
        norm = frozenset(_normalize(shape))
        if norm not in seen:
            seen.add(norm)
            uniq.append(set(norm))
    return uniq


SHAPES = [s for s in _shape_variants() if 4 <= _perimeter(np.array(list(s), dtype=int), set(s)) <= 20]


def _shape_bbox(shape: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _dilate8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _placeable(shape: set[tuple[int, int]], top: int, left: int, h: int, w: int, forbidden: set[tuple[int, int]]) -> bool:
    abs_cells = {(top + r, left + c) for r, c in shape}
    return all(0 <= r < h and 0 <= c < w for r, c in abs_cells) and not (abs_cells & forbidden)


def _sample_object_count() -> int:
    return random.randint(2, 5) if random.random() < 0.75 else 1


def _generate_case() -> np.ndarray | None:
    h = random.randint(7, 18)
    w = random.randint(8, 20)
    grid = np.full((h, w), BG, dtype=int)

    n = _sample_object_count()
    shapes = random.sample(SHAPES, k=n) if n <= len(SHAPES) else [random.choice(SHAPES) for _ in range(n)]
    colors = random.sample(PALETTE, k=n)

    order_rows = sorted(random.sample(range(0, h - 1), k=n))
    used_rows = set()
    forbidden: set[tuple[int, int]] = set()

    placed: list[tuple[int, int, set[tuple[int, int]], int]] = []
    for idx, (shape, color) in enumerate(zip(shapes, colors)):
        sh, sw = _shape_bbox(shape)
        candidate_tops = [r for r in range(max(0, order_rows[idx] - 1), min(h - sh, order_rows[idx] + 1) + 1) if r not in used_rows]
        random.shuffle(candidate_tops)
        if not candidate_tops:
            return None

        done = False
        for top in candidate_tops:
            cols = list(range(0, w - sw + 1))
            random.shuffle(cols)
            for left in cols:
                if _placeable(shape, top, left, h, w, forbidden):
                    abs_cells = {(top + r, left + c) for r, c in shape}
                    forbidden |= _dilate8(abs_cells)
                    used_rows.add(top)
                    placed.append((top, left, shape, color))
                    done = True
                    break
            if done:
                break
        if not done:
            return None

    for top, left, shape, color in placed:
        for r, c in shape:
            grid[top + r, left + c] = color

    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
        return None

    comps = _find_components(grid, BG)
    if len(comps) != n:
        return None
    tops = sorted(comp["top"] for comp in comps)
    if len(set(tops)) != n:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape[0] != 2 * n - 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
