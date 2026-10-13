from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_count(grid: np.ndarray, comp: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros_like(sub, dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True
    seen = np.zeros_like(sub, dtype=bool)
    holes = 0
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            if mask[r, c] or sub[r, c] != BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, sub.shape[0] - 1) or cc in (0, sub.shape[1] - 1):
                    touches_edge = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and not mask[nr, nc] and sub[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign each hollow outline's color by reversing the ordering induced by its enclosed-hole count.

    Input structure:
    - The background is dominant.
    - Several disconnected monochrome closed outline shapes sit on the grid.
    - Different shapes enclose different numbers of background cavities.

    Rule:
    - Detect each outline component and count how many enclosed background holes it contains.
    - Sort the shapes by hole count from smallest to largest.
    - Take the original component colors in the opposite order, from largest hole count to smallest.
    - Repaint each shape in the sorted shape order with the oppositely ordered color list, leaving all pixels in place.

    Key insight:
    - The visible geometry matters only through one hidden attribute: the number of enclosed holes determines a ranking, and the final color assignment is that ranking reversed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    shapes = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            shapes.append({"cells": comp, "color": color, "holes": _hole_count(g, comp)})

    shapes.sort(key=lambda item: (int(item["holes"]), int(item["color"])))
    reversed_colors = [int(item["color"]) for item in reversed(shapes)]
    for item, new_color in zip(shapes, reversed_colors):
        for r, c in item["cells"]:  # type: ignore[index]
            out[r, c] = new_color
    return out


def _draw_shape(grid: np.ndarray, r0: int, c0: int, h: int, holes: int, color: int) -> None:
    w = 3 * holes + 2
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for k in range(1, holes):
        c = c0 + 3 * k
        grid[r0 : r1 + 1, c] = color


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(16, 28)
        w = random.randint(18, 30)
        grid = np.full((h, w), BG, dtype=int)
        hole_counts = random.sample([1, 2, 3, 4], random.randint(3, 4))
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], len(hole_counts))
        ok = True
        for holes, color in zip(hole_counts, colors):
            placed = False
            sh = random.randint(5, 7)
            sw = 3 * holes + 2
            for _ in range(80):
                r0 = random.randint(1, h - sh - 1)
                c0 = random.randint(1, w - sw - 1)
                if np.any(grid[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] != BG):
                    continue
                _draw_shape(grid, r0, c0, sh, holes, color)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
