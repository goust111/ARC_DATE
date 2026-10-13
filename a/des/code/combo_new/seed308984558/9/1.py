import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
    Fill each grey frame interior with the palette color keyed by its hole pattern.

    Input structure:
    - Grey hollow rectangles sit on a black background in the upper part of the grid.
    - Each frame interior is grey except for a connected black hole pattern.
    - Along the bottom sits a row of separated colored square swatches; each swatch has
      its own connected black hole pattern inside.

    Rule:
    - Extract the black hole mask of every palette swatch relative to its bounding box.
    - Extract each frame's interior black hole mask relative to the frame interior.
    - Find the swatch whose hole mask matches the frame's hole mask exactly.
    - Recolor the frame's non-hole interior cells to the swatch color, keep the frame
      border grey, keep the hole black, and leave the palette unchanged.

    Key insight:
    - The black holes are lookup keys rather than damage: matching hole geometry across
      two object families determines which palette color fills each frame.
    """
    h, w = grid.shape
    palette = {}
    for color in [int(v) for v in np.unique(grid[-8:]) if v not in (BLACK, GREY)]:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r0 < h // 2:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            hole = patch == BLACK
            palette[hole.tobytes() + str(hole.shape).encode()] = color

    out = grid.copy()
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2 or r0 >= h // 2:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        hole = inner == BLACK
        key = hole.tobytes() + str(hole.shape).encode()
        color = palette[key]
        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        fill[(inner == GREY)] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        n = random.randint(2, 4)
        size = random.randint(4, 6)
        colors = random.sample(COLORS, n)
        patterns = []
        for _ in range(n):
            hole = np.zeros((size, size), dtype=bool)
            cells = [(random.randint(1, size - 2), random.randint(1, size - 2))]
            for _ in range(random.randint(2, size)):
                r, c = random.choice(cells)
                dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                nr, nc = min(max(1, r + dr), size - 2), min(max(1, c + dc), size - 2)
                cells.append((nr, nc))
            for r, c in set(cells):
                hole[r, c] = True
            patterns.append(hole)
        top_h = size + 4
        w = n * (size + 4) + 2
        h = top_h + size + 3
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=int)
        c = 1
        for color, hole in zip(colors, patterns):
            grid[top_h, c : c + size + 2] = GREY
            grid[top_h + size + 1, c : c + size + 2] = GREY
            grid[top_h : top_h + size + 2, c] = GREY
            grid[top_h : top_h + size + 2, c + size + 1] = GREY
            grid[top_h + 1 : top_h + 1 + size, c + 1 : c + 1 + size] = GREY
            inner = grid[top_h + 1 : top_h + 1 + size, c + 1 : c + 1 + size]
            inner[hole] = BLACK
            c += size + 4
        c = 1
        for color, hole in zip(colors, patterns):
            grid[h - size - 1 : h - 1, c : c + size] = color
            patch = grid[h - size - 1 : h - 1, c : c + size]
            patch[hole] = BLACK
            c += size + 2
        return grid
