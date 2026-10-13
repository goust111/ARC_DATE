import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_color(grid, color):
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each object rightward along the rail with color-specific stamp spacing.

    Input structure:
    - A full-width grey rail crosses the grid near the bottom.
    - Small colored objects sit directly above the rail.
    - Below the rail, one colored horizontal sample segment per object color encodes a
      spacing value by its length.

    Rule:
    - Read the sample length for each color.
    - Remove the sample area from the output, but keep the rail and original objects.
    - For each object, stamp copies to the right, preserving shape and rail alignment.
      The gap between consecutive copies equals the sample length.
    - If copies overlap, the object whose original position is farther right wins.

    Key insight:
    - The colored samples are numeric parameters in disguise: their segment length sets
      the empty-column spacing between repeated object stamps.
    """
    rail = next(r for r in range(grid.shape[0]) if np.all(grid[r] == GREY))
    out = grid.copy()
    out[rail + 1 :] = BLACK
    spacing = {}
    for color in [int(v) for v in np.unique(grid[rail + 1 :]) if v != BLACK]:
        comp = max(_components_of_color(grid[rail + 1 :], color), key=len)
        spacing[color] = len(comp)
    objs = []
    for color in [int(v) for v in np.unique(grid[:rail]) if v not in (BLACK, GREY)]:
        for comp in _components_of_color(grid[:rail], color):
            objs.append((min(c for _, c in comp), color, comp))
    objs.sort()
    for _, color, comp in objs:
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        step = patch.shape[1] + spacing[color]
        cur = c0 + step
        while cur + patch.shape[1] <= grid.shape[1]:
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    if patch[i, j] != BLACK:
                        out[r0 + i, cur + j] = patch[i, j]
            cur += step
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    rail = random.randint(h - 7, h - 5)
    grid = np.zeros((h, w), dtype=int)
    grid[rail, :] = GREY
    colors = random.sample(COLORS, random.randint(2, 3))
    x = 1
    for color in colors:
        oh, ow = random.randint(2, 3), random.randint(2, 3)
        if x + ow >= w - 1:
            break
        patch = np.zeros((oh, ow), dtype=int)
        for _ in range(random.randint(2, oh * ow)):
            patch[random.randint(0, oh - 1), random.randint(0, ow - 1)] = color
        if not np.any(patch):
            patch[0, 0] = color
        grid[rail - oh : rail, x : x + ow] = patch
        gap = random.randint(2, 5)
        grid[rail + 2, x : x + gap] = color
        x += ow + gap + 2
    return grid
