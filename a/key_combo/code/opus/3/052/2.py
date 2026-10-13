from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _enclosed_area(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> int:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    black = sub == 0
    area = 0
    for bg in _components(black):
        touches_edge = any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in bg)
        if not touches_edge:
            area += len(bg)
    return area


def main(grid: np.ndarray) -> np.ndarray:
    """Extract hollow rectangles, sort them by frame height, and output enclosed-area bars."""
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(x) for x in np.unique(grid) if x != 0):
        for comp in _components(grid == color):
            area = _enclosed_area(grid, comp, color)
            if area <= 0:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            frames.append((r1 - r0 + 1, c0, area, color, r0))
    if not frames:
        return np.zeros((0, 0), dtype=grid.dtype)
    frames.sort(key=lambda item: (-item[0], item[1], item[4]))
    height = max(item[2] for item in frames)
    out = np.zeros((height, len(frames)), dtype=grid.dtype)
    for c, (_, _, area, color, _) in enumerate(frames):
        out[:area, c] = color
    return out


def _place_rect(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int, hollow: bool) -> None:
    if hollow:
        grid[r : r + h, c] = color
        grid[r : r + h, c + w - 1] = color
        grid[r, c : c + w] = color
        grid[r + h - 1, c : c + w] = color
    else:
        grid[r : r + h, c : c + w] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7], 5)
    hollows = [
        (1, 1, 6, 4, colors[0]),
        (1, 8, 4, 7, colors[1]),
        (10, 3, 5, 5, colors[2]),
        (8, 11, 6, 4, colors[3]),
    ]
    solids = [(12, 12, 3, 4, colors[4]), (5, 15, 3, 3, random.choice([1, 2, 3, 4, 5, 6, 7]))]
    random.shuffle(hollows)
    for r, c, h, w, color in hollows[: random.randint(2, 3)]:
        _place_rect(grid, r, c, h, w, color, True)
    for r, c, h, w, color in solids[: random.randint(1, 2)]:
        _place_rect(grid, r, c, h, w, color, False)
    return grid
