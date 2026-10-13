from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 8
YELLOW = 4
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count yellow markers inside each foreground object, convert each object to a
    solid odd square of its main color, then nest the squares from largest to
    smallest by centering each one on the largest canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one foreground object"

    squares: list[tuple[int, int]] = []
    for comp in comps:
        colors = [int(g[r, c]) for r, c in comp]
        marker_count = sum(color == YELLOW for color in colors)
        assert 1 <= marker_count <= 4, "expected 1..4 yellow markers per object"

        main_colors = [color for color in colors if color != YELLOW]
        assert main_colors, "object must contain a non-yellow main color"
        main_color = Counter(main_colors).most_common(1)[0][0]
        side = 2 * marker_count + 1
        squares.append((side, int(main_color)))

    squares.sort(key=lambda item: item[0], reverse=True)
    canvas_side = squares[0][0]
    out = np.full((canvas_side, canvas_side), bg, dtype=int)

    for side, color in squares:
        start = (canvas_side - side) // 2
        out[start : start + side, start : start + side] = color

    return out


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                cells.append((r, c))
    return cells


def _grow_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randint(1, h - 2), random.randint(1, w - 2))
    mask[start] = True

    while int(mask.sum()) < target_area:
        frontier = []
        for r, c in zip(*np.where(mask)):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        mask[random.choice(frontier)] = True
    return mask


def _make_object(marker_count: int, color: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(max(5, marker_count + 3), max(6, marker_count + 6))
        w = random.randint(max(5, marker_count + 3), max(6, marker_count + 6))
        area_lo = max(marker_count + 8, (h * w) // 2)
        area_hi = h * w - random.randint(0, max(1, (h * w) // 5))
        if area_lo > area_hi:
            continue

        mask = _grow_mask(h, w, random.randint(area_lo, area_hi))
        if int(mask.sum()) < marker_count + 5:
            continue

        interior = _interior_cells(mask)
        if len(interior) < marker_count:
            continue

        if random.random() < 0.8:
            removable = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    neigh = sum(
                        0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
                    if neigh >= 2 and (r, c) not in interior:
                        removable.append((r, c))
            random.shuffle(removable)
            for r, c in removable[: random.randint(0, min(3, len(removable)))]:
                trial = mask.copy()
                trial[r, c] = False
                comps = _find_components(np.where(trial, 1, 0), 0)
                if len(comps) == 1 and len(_interior_cells(trial)) >= marker_count:
                    mask = trial

        interior = _interior_cells(mask)
        if len(interior) < marker_count:
            continue

        obj = np.full((h, w), BG, dtype=int)
        obj[mask] = color
        for r, c in random.sample(interior, marker_count):
            obj[r, c] = YELLOW
        return obj
    return None


def _can_place(canvas: np.ndarray, obj: np.ndarray, top: int, left: int) -> bool:
    h, w = obj.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return bool(np.all(canvas[r0:r1, c0:c1] == BG))


def _place(canvas: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    h, w = obj.shape
    patch = canvas[top : top + h, left : left + w]
    patch[obj != BG] = obj[obj != BG]


def _expected_from_specs(specs: list[tuple[int, int]]) -> np.ndarray:
    squares = sorted(((2 * m + 1, color) for m, color in specs), reverse=True)
    side = squares[0][0]
    out = np.full((side, side), BG, dtype=int)
    for s, color in squares:
        start = (side - s) // 2
        out[start : start + s, start : start + s] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 30)
    W = random.randint(18, 30)
    grid = np.full((H, W), BG, dtype=int)

    object_count = random.randint(1, 4)
    marker_counts = random.sample([1, 2, 3, 4], object_count)
    palette = [c for c in range(10) if c not in (BG, YELLOW)]
    colors = random.sample(palette, object_count)
    specs = list(zip(marker_counts, colors))
    random.shuffle(specs)

    for marker_count, color in specs:
        obj = _make_object(marker_count, color)
        if obj is None:
            return None
        h, w = obj.shape
        placements = [
            (r, c)
            for r in range(H - h + 1)
            for c in range(W - w + 1)
            if _can_place(grid, obj, r, c)
        ]
        if not placements:
            return None
        top, left = random.choice(placements)
        _place(grid, obj, top, left)

    expected = _expected_from_specs(specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
