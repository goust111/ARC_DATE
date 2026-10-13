from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
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


def _read_legend(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    if h >= 2:
        colors: list[int] = []
        used: set[tuple[int, int]] = set()
        c = 0
        ok = False
        while c < w:
            if c + 1 < w and grid[0, c] == grid[1, c] == grid[0, c + 1] == grid[1, c + 1] and int(grid[0, c]) not in (BG, GRAY):
                color = int(grid[0, c])
                colors.append(color)
                used.update({(0, c), (0, c + 1), (1, c), (1, c + 1)})
                c += 2
                ok = True
                if c < w and grid[0, c] == grid[1, c] == GRAY:
                    used.update({(0, c), (1, c)})
                    c += 1
                else:
                    break
            else:
                break
        if ok and len(colors) >= 2:
            return colors, used

    if w >= 2:
        colors = []
        used = set()
        r = 0
        ok = False
        while r < h:
            if r + 1 < h and grid[r, 0] == grid[r, 1] == grid[r + 1, 0] == grid[r + 1, 1] and int(grid[r, 0]) not in (BG, GRAY):
                color = int(grid[r, 0])
                colors.append(color)
                used.update({(r, 0), (r, 1), (r + 1, 0), (r + 1, 1)})
                r += 2
                ok = True
                if r < h and grid[r, 0] == grid[r, 1] == GRAY:
                    used.update({(r, 0), (r, 1)})
                    r += 1
                else:
                    break
            else:
                break
        if ok and len(colors) >= 2:
            return colors, used

    raise AssertionError("failed to detect the color legend")


def _count_holes(grid: np.ndarray, comp: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros_like(sub, dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True

    bg_seen = np.zeros_like(sub, dtype=bool)
    holes = 0
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            if mask[r, c] or sub[r, c] != BG or bg_seen[r, c]:
                continue
            q = deque([(r, c)])
            bg_seen[r, c] = True
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, sub.shape[0] - 1) or cc in (0, sub.shape[1] - 1):
                    touches_edge = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and not mask[nr, nc] and sub[nr, nc] == BG and not bg_seen[nr, nc]:
                        bg_seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object by looking up its hole count in the edge legend strip.

    Input structure:
    - The background is black.
    - A legend strip on the top edge or left edge contains colored 2x2 swatches separated by gray bars.
    - Elsewhere, gray connected objects occupy the main area and may contain enclosed black holes.

    Rule:
    - Read the legend swatches in order from left to right or top to bottom and index them as 0, 1, 2, ...
    - For each gray connected object, count how many black holes it encloses.
    - Recolor all gray pixels of that object with the legend color at the matching index.
    - Leave the legend, the black holes, and the black background unchanged.

    Key insight:
    - The objects are classified by a topological property, not by size or position: the number of enclosed holes is the index into the separate color strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_colors, legend_cells = _read_legend(g)
    object_mask = (g == GRAY).copy()
    for r, c in legend_cells:
        object_mask[r, c] = False

    out = g.copy()
    for comp in _components(object_mask):
        holes = _count_holes(g, comp)
        assert holes < len(legend_colors), "legend too short for object hole count"
        new_color = legend_colors[holes]
        for r, c in comp:
            out[r, c] = new_color
    return out


def _draw_legend_top(grid: np.ndarray, colors: list[int]) -> set[tuple[int, int]]:
    used: set[tuple[int, int]] = set()
    c = 0
    for i, color in enumerate(colors):
        grid[0:2, c : c + 2] = color
        used.update({(0, c), (0, c + 1), (1, c), (1, c + 1)})
        c += 2
        if i != len(colors) - 1:
            grid[0:2, c] = GRAY
            used.update({(0, c), (1, c)})
            c += 1
    return used


def _draw_legend_left(grid: np.ndarray, colors: list[int]) -> set[tuple[int, int]]:
    used: set[tuple[int, int]] = set()
    r = 0
    for i, color in enumerate(colors):
        grid[r : r + 2, 0:2] = color
        used.update({(r, 0), (r, 1), (r + 1, 0), (r + 1, 1)})
        r += 2
        if i != len(colors) - 1:
            grid[r, 0:2] = GRAY
            used.update({(r, 0), (r, 1)})
            r += 1
    return used


def _place_object(grid: np.ndarray, r0: int, c0: int, h: int, w: int, holes: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = GRAY
    interior = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
    random.shuffle(interior)
    chosen: list[tuple[int, int]] = []
    for cell in interior:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
            chosen.append(cell)
            if len(chosen) == holes:
                break
    if len(chosen) != holes:
        raise AssertionError("not enough separated interior cells for requested holes")
    for r, c in chosen:
        grid[r, c] = BG


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        legend_len = random.randint(3, 5)
        legend_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], legend_len)
        top_legend = bool(random.getrandbits(1))

        if top_legend:
            legend_h = 2
            legend_w = 2 * legend_len + legend_len - 1
            h = random.randint(max(16, legend_h + 8), 30)
            w = random.randint(max(18, legend_w + 8), 30)
        else:
            legend_h = 2 * legend_len + legend_len - 1
            legend_w = 2
            h = random.randint(max(18, legend_h + 8), 30)
            w = random.randint(max(16, legend_w + 8), 30)

        grid = np.full((h, w), BG, dtype=int)
        used = _draw_legend_top(grid, legend_colors) if top_legend else _draw_legend_left(grid, legend_colors)

        obj_count = random.randint(2, min(4, legend_len))
        hole_counts = random.sample(range(legend_len), obj_count)
        ok = True
        for holes in hole_counts:
            placed = False
            for _ in range(80):
                oh_lo = max(5, 2 * holes + 3)
                ow_lo = max(5, 2 * holes + 3)
                oh_hi = min(9, h - 2)
                ow_hi = min(9, w - 2)
                if oh_lo > oh_hi or ow_lo > ow_hi:
                    continue
                oh = random.randint(oh_lo, oh_hi)
                ow = random.randint(ow_lo, ow_hi)
                r0 = random.randint(3 if top_legend else 1, h - oh - 1)
                c0 = random.randint(1 if top_legend else 3, w - ow - 1)
                area = grid[max(0, r0 - 1) : min(h, r0 + oh + 1), max(0, c0 - 1) : min(w, c0 + ow + 1)]
                if np.any(area != BG):
                    continue
                try:
                    _place_object(grid, r0, c0, oh, ow, holes)
                except AssertionError:
                    continue
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
