from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
BLUE = 1
ORANGE = 7

SHAPE_COLORS = (RED, GREEN, YELLOW, BLUE, ORANGE, 6, 8, 9)
GEN_ATTEMPTS = 800


def _guide_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GRAY)]
    assert len(rows) >= 2, "expected two full gray guide rows"
    return [rows[0], rows[-1]]


def _shape_boxes(grid: np.ndarray, guide_rows: list[int]) -> list[tuple[int, int, tuple[int, int, int, int], int]]:
    top, bottom = guide_rows
    boxes: list[tuple[int, int, tuple[int, int, int, int], int]] = []
    for color in sorted(set(int(v) for v in np.unique(grid)) - {BLACK, GRAY}):
        coords = np.argwhere(grid == color)
        coords = coords[(coords[:, 0] > top) & (coords[:, 0] < bottom)]
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        footprint = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(footprint == BLACK) or np.any(footprint == GRAY):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        boxes.append((area, color, (r0, c0, r1, c1), int(np.sum(footprint == color))))
    assert boxes, "expected at least one recoverable foreground rectangle"
    areas = [area for area, _color, _box, _visible in boxes]
    assert len(areas) == len(set(areas)), "shape areas must be unique"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover occluded rectangular footprints, count their true areas, and draw vertical bars between guide rows.
    """
    g = np.asarray(grid, dtype=int)
    guides = _guide_rows(g)
    top, bottom = guides
    boxes = _shape_boxes(g, guides)

    out = g.copy()
    out[top + 1 : bottom, :] = BLACK
    for index, (area, color, _box, _visible) in enumerate(sorted(boxes, key=lambda item: (item[0], item[1]))):
        col = 1 + index * 2
        assert col < out.shape[1], "not enough room for bar columns"
        assert top + area < bottom, "bar length must fit between guide rows"
        out[top + 1 : top + 1 + area, col] = color
    return out


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    rr0, cc0 = max(ar0, br0), max(ac0, bc0)
    rr1, cc1 = min(ar1, br1), min(ac1, bc1)
    if rr0 > rr1 or cc0 > cc1:
        return 0
    return (rr1 - rr0 + 1) * (cc1 - cc0 + 1)


def _sample_rectangles(top: int, bottom: int, width: int, count: int) -> list[tuple[int, int, int, int]]:
    span = bottom - top - 1
    for _ in range(300):
        rects: list[tuple[int, int, int, int]] = []
        areas: set[int] = set()
        for _idx in range(count):
            for _try in range(100):
                rh = random.randint(2, min(4, span - 2))
                rw = random.randint(2, min(5, width - 3))
                area = rh * rw
                if area in areas:
                    continue
                r0 = random.randint(top + 1, bottom - rh - 1)
                c0 = random.randint(2, width - rw - 2)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if rects and not any(_overlap_area(rect, other) > 0 for other in rects):
                    if random.random() < 0.65:
                        continue
                if all(_overlap_area(rect, other) < area for other in rects):
                    rects.append(rect)
                    areas.add(area)
                    break
            else:
                break
        if len(rects) != count:
            continue
        if any(_overlap_area(a, b) > 0 for i, a in enumerate(rects) for b in rects[i + 1 :]):
            return rects
    raise RuntimeError("failed to sample overlapping rectangles")


def _draw_scene(height: int, width: int, top: int, bottom: int, rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[top, :] = GRAY
    grid[bottom, :] = GRAY
    order = list(range(len(rects)))
    random.shuffle(order)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]
    return grid


def _visible_counts_ok(grid: np.ndarray, colors: list[int]) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    return all(counts[color] > 0 for color in colors)


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape_count = random.randint(3, 4)
        width = random.randint(14, 20)
        max_area = 20
        top = random.randint(1, 3)
        bottom = top + max_area + random.randint(3, 5)
        height = bottom + random.randint(2, 4)
        if height > 30:
            continue
        rects = _sample_rectangles(top, bottom, width, shape_count)
        areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects]
        if max(areas) + 1 >= bottom - top:
            continue
        colors = random.sample(SHAPE_COLORS, shape_count)
        grid = _draw_scene(height, width, top, bottom, rects, colors)
        if not _visible_counts_ok(grid, colors):
            continue
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate puzzle 030/10 input")

